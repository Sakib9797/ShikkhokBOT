# scripts/02b_batch_cot.py
"""Phase 2 full run — the whole training pool through the Batches API.

50% cheaper than streaming, and the pool fits in one batch (limit 100,000
requests / 256 MB). Three subcommands so a submitted batch survives the machine
going to sleep:

    python scripts/02b_batch_cot.py submit            # create + record batch id
    python scripts/02b_batch_cot.py status            # poll
    python scripts/02b_batch_cot.py collect           # write data/cot/all_cot.jsonl
    python scripts/02b_batch_cot.py submit --retry    # re-run rejected rows only

Two failure modes this guards against:
  * Results come back in ARBITRARY ORDER — everything keys on `custom_id`
    (the stable content-hash qid), never on position.
  * The `fallbacks` parameter is rejected on the Batches API, so refusal
    handling has to be client-side: that is what `extract()` is.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cot_core import (Rejected, build_user, extract, load_env,  # noqa: E402
                      read_jsonl, request_params, to_cot, validate)

POOL = "data/clean/train_pool.jsonl"
OUT = pathlib.Path("data/cot/all_cot.jsonl")
QUARANTINE = pathlib.Path("data/cot/quarantine.jsonl")
STATE = pathlib.Path("outputs/reports/batch_state.json")

# A stricter nudge appended to the user turn when retrying a rejected row.
RETRY_NUDGE = (
    "\n\nগুরুত্বপূর্ণ: আগের চেষ্টাটি বাতিল হয়েছে। কোনো মধ্যবর্তী ধাপে চূড়ান্ত উত্তর "
    "লিখবে না, hint-এর বাক্য হুবহু অনুলিপি করবে না, এবং প্রতিটি ধাপ যেন নতুন "
    "যুক্তি যোগ করে।"
)


def client():
    load_env()
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is empty — put it in .env before running Phase 2.")
    from anthropic import Anthropic
    return Anthropic()


def read_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def write_state(**kw):
    st = read_state()
    st.update(kw)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def pending(retry=False):
    """Rows still needing a chain: never generated, or generated and rejected."""
    pool = read_jsonl(POOL)
    if not pool:
        sys.exit(f"{POOL} is missing or empty — run scripts/01_prepare.py first.")
    have = {r["qid"]: r for r in read_jsonl(OUT)}
    if retry:
        want = {q for q, r in have.items() if r.get("_reject")}
        return [e for e in pool if e["qid"] in want]
    return [e for e in pool if e["qid"] not in have]


def cmd_submit(args):
    from anthropic.types.messages.batch_create_params import Request
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

    todo = pending(args.retry)
    if not todo:
        print("nothing pending — every pool row already has an accepted chain")
        return
    params = request_params(args.model, args.effort)
    reqs = [
        Request(
            custom_id=e["qid"],
            params=MessageCreateParamsNonStreaming(
                messages=[{"role": "user",
                           "content": build_user(e) + (RETRY_NUDGE if args.retry else "")}],
                **params),
        )
        for e in todo
    ]
    batch = client().messages.batches.create(requests=reqs)
    write_state(batch_id=batch.id, model=args.model, n=len(reqs),
                retry=bool(args.retry))
    print(f"submitted batch {batch.id}: {len(reqs)} requests on {args.model}")
    print("poll with:  python scripts/02b_batch_cot.py status")


def cmd_status(args):
    bid = args.batch_id or read_state().get("batch_id")
    if not bid:
        sys.exit("no batch id — submit first, or pass --batch-id")
    b = client().messages.batches.retrieve(bid)
    print(f"{bid}: {b.processing_status}")
    print(f"  counts: {b.request_counts}")
    if b.processing_status == "ended":
        print("collect with:  python scripts/02b_batch_cot.py collect")


def cmd_collect(args):
    bid = args.batch_id or read_state().get("batch_id")
    if not bid:
        sys.exit("no batch id — submit first, or pass --batch-id")
    model = read_state().get("model", args.model)

    by_qid = {e["qid"]: e for e in read_jsonl(POOL)}
    kept = {r["qid"]: r for r in read_jsonl(OUT)}      # merge, so retries overwrite
    quarantine = read_jsonl(QUARANTINE)

    stats = {"succeeded": 0, "quarantined": 0}
    rejects = {}
    for r in client().messages.batches.results(bid):
        e = by_qid.get(r.custom_id)                    # <- keyed by custom_id, never index
        if e is None:
            quarantine.append({"qid": r.custom_id, "why": "unknown custom_id"})
            stats["quarantined"] += 1
            continue
        if r.result.type != "succeeded":
            quarantine.append({"qid": r.custom_id, "why": r.result.type})
            stats["quarantined"] += 1
            continue
        try:
            data = extract(r.result.message)
        except Rejected as why:
            quarantine.append({"qid": r.custom_id, "why": str(why)})
            stats["quarantined"] += 1
            continue
        e = dict(e)
        e["CoT"] = to_cot(data)
        e["_cot_model"] = model
        e["_cot_steps"] = data["steps"]
        e["_reject"] = validate(data, e)
        if e["_reject"]:
            rejects[e["_reject"]] = rejects.get(e["_reject"], 0) + 1
        kept[e["qid"]] = e
        stats["succeeded"] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for e in kept.values():
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with QUARANTINE.open("w", encoding="utf-8", newline="\n") as f:
        for q in quarantine:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    clean = sum(1 for e in kept.values() if not e.get("_reject"))
    print(f"collected: {stats['succeeded']} succeeded, {stats['quarantined']} quarantined")
    print(f"validator rejects this pass: {rejects or 'none'}")
    print(f"{OUT}: {len(kept)} rows, {clean} clean ({100*clean/max(len(kept),1):.1f}%)")
    if len(kept) - clean:
        print("retry the rejected rows:  python scripts/02b_batch_cot.py submit --retry")
    print("then measure ours vs baseline: "
          "python scripts/02c_baseline_metrics.py --target ours")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--batch-id")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("--retry", action="store_true")
    sub.add_parser("status")
    sub.add_parser("collect")
    args = ap.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
