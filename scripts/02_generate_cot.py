# scripts/02_generate_cot.py
"""Phase 2 pilot — genuine Bengali CoT via the Claude API (streaming path).

Runs a small sample through `claude-opus-5` to establish the quality ceiling
and to replace the plan's estimated token costs with measured ones, before any
money goes into the full batch. Resumable: already-generated qids are skipped.

    python scripts/02_generate_cot.py                    # 30 questions, Opus 5
    python scripts/02_generate_cot.py -n 100 --model claude-sonnet-5 \
        --out data/cot/compare_sonnet.jsonl              # the side-by-side

Pilot gate — do not scale until all four hold:
  1. Bengali reads naturally and the steps actually reason (human read of all).
  2. `_reject` is null on >= 90% of rows.
  3. Answer-leak rate ~ 0% (vs the 9.4% upstream baseline).
  4. Measured token averages do not blow past the plan's estimates.
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cot_core import (Rejected, append_jsonl, build_user, extract,  # noqa: E402
                      load_env, read_jsonl, request_params, to_cot, validate)

POOL = "data/clean/train_pool.jsonl"

# per-million-token list prices, for the measured re-cost printout
PRICES = {
    "claude-opus-5":   (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),   # intro pricing
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


def generate_one(client, e, params):
    with client.messages.stream(
        messages=[{"role": "user", "content": build_user(e)}], **params
    ) as stream:
        msg = stream.get_final_message()
    return extract(msg), msg.usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=30)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--out", default="data/cot/pilot_cot.jsonl")
    args = ap.parse_args()

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is empty — put it in .env before running Phase 2.")
    from anthropic import Anthropic

    client = Anthropic()
    params = request_params(args.model, args.effort)

    pool = read_jsonl(POOL)
    if not pool:
        sys.exit(f"{POOL} is missing or empty — run scripts/01_prepare.py first.")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["qid"] for r in read_jsonl(out)}

    tin = tout = cached = n = 0
    rejects = {}
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for e in pool[:args.count]:
            if e["qid"] in done:
                continue
            try:
                data, usage = generate_one(client, e, params)
            except Rejected as why:
                print(f"  rejected {e['qid']}: {why}")
                rejects["api:" + str(why).split(" ")[0]] = \
                    rejects.get("api:" + str(why).split(" ")[0], 0) + 1
                continue
            n += 1
            tin += usage.input_tokens
            tout += usage.output_tokens
            cached += getattr(usage, "cache_read_input_tokens", 0) or 0

            e["CoT"] = to_cot(data)
            e["_cot_model"] = args.model
            e["_cot_steps"] = data["steps"]
            e["_reject"] = validate(data, e)
            if e["_reject"]:
                rejects[e["_reject"]] = rejects.get(e["_reject"], 0) + 1
            append_jsonl(f, e)

    if not n:
        print("nothing new generated (all requested qids already done)")
        return

    ain, aout = tin / n, tout / n
    print(f"\npilot done: {n} generated, {len(done)} skipped as already-done")
    print(f"tokens in={tin} out={tout} cache_read={cached} "
          f"-> avg {ain:.0f} in / {aout:.0f} out per request")
    print(f"validator rejects: {rejects or 'none'} "
          f"({100*(n-sum(v for k,v in rejects.items() if not k.startswith('api:')))/n:.0f}% clean)")

    if args.model in PRICES:
        pin, pout = PRICES[args.model]
        per = (ain * pin + aout * pout) / 1e6
        pool_n = len(pool)
        print(f"\nmeasured re-cost for {args.model} over the full pool ({pool_n}):")
        print(f"  streaming: ${per*pool_n:,.2f}   batches (-50%): ${per*pool_n/2:,.2f}")
    print(f"\nREVIEW {out} BY EYE before running scripts/02b_batch_cot.py.")


if __name__ == "__main__":
    main()
