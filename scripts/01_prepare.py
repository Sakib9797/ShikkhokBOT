# scripts/01_prepare.py
"""Phase 1 — merge, dedup, and split.

Produces one clean deduped corpus plus a stratified 100-question hold-out that
is never trained on and never sent to Claude. Reuses the key-tolerant loader
and repair logic from ``00_validate.py``.
"""
import hashlib
import importlib.util
import json
import pathlib
import random
import unicodedata
from collections import defaultdict

# --- import the sibling 00_validate.py (name starts with a digit, so load by path) ---
_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("validate", _HERE / "00_validate.py")
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)

random.seed(42)


def norm(s):
    return unicodedata.normalize("NFC", " ".join(str(s).split()))


def qid(subject, question):
    """Stable content-hash id — survives re-runs, re-orderings, and re-repairs."""
    return hashlib.sha1(f"{subject}|{norm(question)}".encode("utf-8")).hexdigest()[:16]


def main():
    # load + normalize every subject
    rows = []
    for subj, p in v.RAW.items():
        got, _ = v.load_ssc_raw(p, subj)
        for e in got:
            e["ExactAnswer"] = [norm(a) for a in e["ExactAnswer"] if str(a).strip()]
            e["qid"] = qid(subj, e["Question"])
            rows.append(e)

    # dedup on normalized question, remembering every subject it appeared under
    by_q = {}
    for e in rows:
        k = norm(e["Question"])
        if k in by_q:
            by_q[k].setdefault("_subjects", [by_q[k]["_subject"]])
            by_q[k]["_subjects"].append(e["_subject"])
        else:
            e["_subjects"] = [e["_subject"]]
            by_q[k] = e
    deduped = list(by_q.values())

    # stratified 100-question hold-out, proportional by subject
    by_subj = defaultdict(list)
    for e in deduped:
        by_subj[e["_subject"]].append(e)
    test, pool = [], []
    for subj, items in by_subj.items():
        random.shuffle(items)
        n = round(100 * len(items) / len(deduped))
        test += items[:n]
        pool += items[n:]
    test = test[:100]
    test_ids = {e["qid"] for e in test}
    pool = [e for e in pool if e["qid"] not in test_ids]      # belt and braces

    # hard guarantee: no test question (or its content-twin) leaks into the pool
    pool_ids = {e["qid"] for e in pool}
    assert test_ids.isdisjoint(pool_ids), "LEAK: test qid found in train pool"
    assert len(test) == 100, f"expected 100 test rows, got {len(test)}"

    pathlib.Path("data/clean").mkdir(parents=True, exist_ok=True)

    def dump(path, data):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for e in data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    dump("data/clean/merged.jsonl", deduped)
    dump("data/clean/test100.jsonl", test)
    dump("data/clean/train_pool.jsonl", pool)

    # subject proportions, for the exit-criteria check
    corpus_mix = {s: len(items) for s, items in by_subj.items()}
    test_mix = defaultdict(int)
    for e in test:
        test_mix[e["_subject"]] += 1

    print(f"deduped={len(deduped)} test={len(test)} pool={len(pool)}")
    print(f"corpus subject mix: {dict(corpus_mix)}")
    print(f"test100 subject mix: {dict(test_mix)}")
    print("test_ids disjoint from pool_ids: OK")


if __name__ == "__main__":
    main()
