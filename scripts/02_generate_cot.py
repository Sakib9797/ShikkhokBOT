# scripts/02_generate_cot.py
"""Phase 2 pilot — Bengali CoT from a local open-source model.

Runs a small sample through whatever is served on the 5090 box, so you can read
the chains and measure the validator pass rate before committing to 10,803 of
them. Resumable: already-generated qids are skipped.

    python scripts/llm_client.py --check                 # confirm the server first
    python scripts/02_generate_cot.py                    # 30 questions
    python scripts/02_generate_cot.py -n 100 --model Qwen/Qwen3-32B-AWQ \
        --out data/cot/pilot_qwen32b.jsonl               # one bake-off entry

Pilot gate — do not scale until all four hold:
  1. The Bengali reads naturally and the steps actually reason (human read).
  2. `_reject` is null on >= 90% of rows.
  3. Answer-leak rate ~ 0% (vs the 9.4% upstream baseline).
  4. Throughput makes the full run a sane wall-clock wait.

Bengali quality varies a lot more between open models than between frontier
APIs, so treat model choice as an experiment: generate a pilot per candidate
and compare them with `02d_model_bakeoff.py` rather than picking on reputation.
"""
import argparse
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cot_core import (COT_SCHEMA, SYSTEM, append_jsonl, build_user,  # noqa: E402
                      load_env, read_jsonl, to_cot, validate)
from llm_client import DEFAULT_MODEL, LLMClient, LLMError  # noqa: E402

POOL = "data/clean/train_pool.jsonl"
FULL_POOL_SIZE = 10803


def generate_one(client, e, max_tokens):
    data, usage = client.complete(SYSTEM, build_user(e), COT_SCHEMA, max_tokens)
    if not isinstance(data, dict) or "steps" not in data:
        raise LLMError(f"off-contract response: {str(data)[:120]}")
    return data, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=30)
    ap.add_argument("--model", default=None, help=f"default: {DEFAULT_MODEL} or $LLM_MODEL")
    ap.add_argument("--base-url", default=None, help="default: $LLM_BASE_URL")
    ap.add_argument("--out", default="data/cot/pilot_cot.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    load_env()
    client = LLMClient(args.base_url, args.model,
                       temperature=args.temperature, max_tokens=args.max_tokens)

    pool = read_jsonl(POOL)
    if not pool:
        sys.exit(f"{POOL} is missing or empty — run scripts/01_prepare.py first.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["qid"] for r in read_jsonl(out)}
    todo = [e for e in pool[:args.count] if e["qid"] not in done]
    if not todo:
        print("nothing new to generate (all requested qids already done)")
        return

    # negotiate the json mode once, serially, so workers do not race on it
    print(f"model    : {client.model}")
    print(f"base_url : {client.base_url}")
    first = todo[0]
    t0 = time.time()
    try:
        data, usage = generate_one(client, first, args.max_tokens)
    except Exception as exc:
        sys.exit(f"first request failed: {type(exc).__name__}: {exc}\n"
                 f"run `python scripts/llm_client.py --check` to isolate it.")
    print(f"json mode: {client.mode}  (first request {time.time()-t0:.1f}s)")

    results = [(first, data, usage, None)]
    rest = todo[1:]

    def work(e):
        try:
            d, u = generate_one(client, e, args.max_tokens)
            return e, d, u, None
        except (LLMError, Exception) as exc:
            return e, None, None, f"{type(exc).__name__}: {str(exc)[:160]}"

    t0 = time.time()
    if rest:
        with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
            for i, r in enumerate(pool_exec.map(work, rest), 2):
                results.append(r)
                if i % 10 == 0:
                    print(f"  {i}/{len(todo)}")
    elapsed = max(time.time() - t0, 1e-6)

    tin = tout = ok = 0
    rejects, failures = {}, {}
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for e, data, usage, err in results:
            if err:
                failures[err.split(":")[0]] = failures.get(err.split(":")[0], 0) + 1
                print(f"  failed {e['qid']}: {err}")
                continue
            ok += 1
            if usage:
                tin += usage.prompt_tokens or 0
                tout += usage.completion_tokens or 0
            row = dict(e)
            row["CoT"] = to_cot(data)
            row["_cot_model"] = client.model
            row["_cot_steps"] = data["steps"]
            row["_reject"] = validate(data, row)
            if row["_reject"]:
                rejects[row["_reject"]] = rejects.get(row["_reject"], 0) + 1
            append_jsonl(f, row)

    if not ok:
        sys.exit("every request failed — check the server and the model name")

    clean = ok - sum(rejects.values())
    per_req = elapsed / max(len(results) - 1, 1)
    print(f"\npilot done: {ok} generated, {len(failures)} transport failures, "
          f"{len(done)} skipped as already-done")
    if tout:
        print(f"tokens: in={tin} out={tout} -> avg {tin/ok:.0f} in / {tout/ok:.0f} out")
    print(f"validator: {clean}/{ok} clean ({100*clean/ok:.0f}%)  rejects={rejects or 'none'}")
    if failures:
        print(f"transport failures: {failures}")

    remaining = FULL_POOL_SIZE - len(done) - ok
    eta_h = per_req * max(remaining, 0) / 3600
    print(f"\nthroughput: {per_req:.1f}s/request at {args.workers} workers "
          f"-> ~{eta_h:.1f}h for the remaining ~{max(remaining,0)} "
          f"(raise --workers to cut this; the 5090 will take more)")
    print(f"\nREVIEW {out} BY EYE before running scripts/02b_bulk_cot.py.")


if __name__ == "__main__":
    main()
