# scripts/02b_bulk_cot.py
"""Phase 2 full run — the whole training pool through the local model.

Replaces the hosted Batches API: there is no batch queue on a local server, so
this is a concurrent, resumable, append-as-you-go loop instead.

    python scripts/02b_bulk_cot.py                    # generate everything left
    python scripts/02b_bulk_cot.py --workers 16       # push the 5090 harder
    python scripts/02b_bulk_cot.py --retry            # re-run rejected chains
    python scripts/02b_bulk_cot.py --status           # how far along are we

Safe to kill and restart at any point. Every accepted chain is appended and
flushed immediately, and the resume set is rebuilt from the output file on
start, so a crash or a reboot costs you only the in-flight requests. That
matters more here than it did with a hosted batch: this run occupies your GPU
for hours and you will want to stop it for other work.

Rows are keyed by the stable content-hash `qid` throughout — never by position.
"""
import argparse
import json
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cot_core import (COT_SCHEMA, RETRY_NUDGE, SYSTEM, build_user,  # noqa: E402
                      load_env, read_jsonl, to_cot, validate)
from llm_client import DEFAULT_MODEL, LLMClient, LLMError  # noqa: E402

POOL = "data/clean/train_pool.jsonl"
OUT = pathlib.Path("data/cot/all_cot.jsonl")
QUARANTINE = pathlib.Path("data/cot/quarantine.jsonl")

_write_lock = threading.Lock()


def status():
    pool = read_jsonl(POOL)
    have = read_jsonl(OUT)
    clean = sum(1 for r in have if not r.get("_reject"))
    quarantined = len(read_jsonl(QUARANTINE))
    print(f"pool        : {len(pool)}")
    print(f"generated   : {len(have)} ({100*len(have)/max(len(pool),1):.1f}%)")
    print(f"  clean     : {clean}")
    print(f"  rejected  : {len(have)-clean}  (rerun with --retry)")
    print(f"quarantined : {quarantined}")
    print(f"remaining   : {len(pool)-len(have)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help=f"default: {DEFAULT_MODEL} or $LLM_MODEL")
    ap.add_argument("--base-url", default=None, help="default: $LLM_BASE_URL")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="stop after N (0 = no limit)")
    ap.add_argument("--retry", action="store_true",
                    help="regenerate validator-rejected chains with a stricter prompt")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        status()
        return

    load_env()
    pool = read_jsonl(POOL)
    if not pool:
        sys.exit(f"{POOL} is missing or empty — run scripts/01_prepare.py first.")

    have = {r["qid"]: r for r in read_jsonl(OUT)}
    if args.retry:
        todo = [e for e in pool if have.get(e["qid"], {}).get("_reject")]
        verb = "retrying"
    else:
        todo = [e for e in pool if e["qid"] not in have]
        verb = "generating"
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to do — every pool row already has an accepted chain")
        status()
        return

    client = LLMClient(args.base_url, args.model,
                       temperature=args.temperature, max_tokens=args.max_tokens)
    print(f"model    : {client.model}")
    print(f"base_url : {client.base_url}")
    print(f"{verb} {len(todo)} chains at {args.workers} workers")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # negotiate the json mode serially before the workers start
    try:
        client.complete(SYSTEM, build_user(todo[0]), COT_SCHEMA, args.max_tokens)
    except Exception as exc:
        sys.exit(f"first request failed: {type(exc).__name__}: {exc}\n"
                 f"run `python scripts/llm_client.py --check` to isolate it.")
    print(f"json mode: {client.mode}")

    counts = {"ok": 0, "rejected": 0, "quarantined": 0}
    rejects = {}
    t0 = time.time()

    def work(e):
        user = build_user(e) + (RETRY_NUDGE if args.retry else "")
        try:
            data, _ = client.complete(SYSTEM, user, COT_SCHEMA, args.max_tokens)
            if not isinstance(data, dict) or "steps" not in data:
                raise LLMError(f"off-contract: {str(data)[:120]}")
        except Exception as exc:
            return e, None, f"{type(exc).__name__}: {str(exc)[:200]}"
        return e, data, None

    # `--retry` rewrites rows in place, so results are merged in memory and the
    # file is rewritten at the end; the normal path appends and is crash-safe.
    out_fh = None if args.retry else OUT.open("a", encoding="utf-8", newline="\n")
    quarantine = read_jsonl(QUARANTINE)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(work, e) for e in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                e, data, err = fut.result()
                if err:
                    counts["quarantined"] += 1
                    quarantine.append({"qid": e["qid"], "why": err})
                else:
                    row = dict(e)
                    row["CoT"] = to_cot(data)
                    row["_cot_model"] = client.model
                    row["_cot_steps"] = data["steps"]
                    row["_reject"] = validate(data, row)
                    if row["_reject"]:
                        counts["rejected"] += 1
                        rejects[row["_reject"]] = rejects.get(row["_reject"], 0) + 1
                    else:
                        counts["ok"] += 1
                    have[row["qid"]] = row
                    if out_fh is not None:
                        with _write_lock:
                            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                            out_fh.flush()
                if i % 50 == 0 or i == len(todo):
                    rate = i / max(time.time() - t0, 1e-6)
                    eta = (len(todo) - i) / max(rate, 1e-9) / 3600
                    print(f"  {i}/{len(todo)}  {rate*60:.0f}/min  ETA {eta:.1f}h  "
                          f"clean={counts['ok']} rejected={counts['rejected']} "
                          f"failed={counts['quarantined']}")
    except KeyboardInterrupt:
        print("\ninterrupted — progress is on disk, rerun to resume")
    finally:
        if out_fh is not None:
            out_fh.close()

    if args.retry:
        with OUT.open("w", encoding="utf-8", newline="\n") as f:
            for row in have.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with QUARANTINE.open("w", encoding="utf-8", newline="\n") as f:
        for q in quarantine:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\ndone in {(time.time()-t0)/3600:.2f}h: {counts}")
    print(f"validator rejects this pass: {rejects or 'none'}")
    status()
    print("\nnext: python scripts/02b_bulk_cot.py --retry"
          "\nthen: python scripts/02c_baseline_metrics.py --target ours")


if __name__ == "__main__":
    main()
