# scripts/02c_baseline_metrics.py
"""Phase 2 baseline — measure the SHIPPED CoT's hint-copy and answer-leak rates.

The upstream `data/SSC-BanglaTutor-CoT/*_with_cot.jsonl` files carry a `CoT`
field that is the `Hints` list reformatted, not reasoning. This script measures
exactly how much, producing the comparison table the paper is built on:

    prior CoT: X% hint-copy, Y% answer leak;  ours: X'%, Y'%

Local, free, no API. Run the same script over `data/cot/all_cot.jsonl` with
`--target ours` once Phase 2 has generated it, to fill in the second row.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cot_core import nrm, strip_step_prefix  # noqa: E402

UPSTREAM = {
    "Biology":   "data/SSC-BanglaTutor-CoT/SSC_Biology_Datasets_with_cot.jsonl",
    "Chemistry": "data/SSC-BanglaTutor-CoT/SSC_Chemistry_Dataset_with_cot.jsonl",
    "Physics":   "data/SSC-BanglaTutor-CoT/SSC_Physics_Dataset_with_cot.jsonl",
}
OURS = {"ours": "data/cot/all_cot.jsonl"}


def get(entry, *names, default=None):
    for n in names:
        if n in entry:
            return entry[n]
    return default


def split_steps(cot):
    """The rendered chain back into steps, dropping the final answer line."""
    lines = [l for l in str(cot).splitlines() if l.strip()]
    return [l for l in lines if not l.startswith("তাই সঠিক উত্তর")]


def score_file(path, subject):
    """Return (rows, step_lines, verbatim_hint_lines, answer_leak_rows)."""
    rows = step_lines = verbatim = leaks = 0
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        cot = get(e, "CoT", "cot")
        if not cot:
            continue
        rows += 1
        steps = split_steps(cot)
        step_lines += len(steps)

        hints = {nrm(h) for h in (get(e, "Hints", "hints") or []) if str(h).strip()}
        verbatim += sum(strip_step_prefix(s) in hints for s in steps)

        gold = get(e, "ExactAnswer", default=[])
        golds = [a for a in (gold if isinstance(gold, list) else [gold]) if str(a).strip()]
        # a leak is the gold answer appearing BEFORE the final line
        body = nrm(" ".join(steps[:-1])) if len(steps) > 1 else ""
        if body and any(nrm(a) in body for a in golds):
            leaks += 1
    return rows, step_lines, verbatim, leaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["upstream", "ours"], default="upstream")
    args = ap.parse_args()
    files = UPSTREAM if args.target == "upstream" else OURS

    missing = [p for p in files.values() if not pathlib.Path(p).exists()]
    if missing:
        sys.exit(f"missing input(s): {missing}")

    out = [f"# CoT quality baseline — {args.target}\n",
           "| Split | Rows | Step lines | Verbatim hints | Hint-copy % | Answer-leak rows | Leak % |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    tot = [0, 0, 0, 0]
    for subject, path in files.items():
        r, sl, v, lk = score_file(path, subject)
        tot = [a + b for a, b in zip(tot, (r, sl, v, lk))]
        out.append(f"| {subject} | {r} | {sl} | {v} | {100*v/max(sl,1):.1f}% "
                   f"| {lk} | {100*lk/max(r,1):.1f}% |")
    r, sl, v, lk = tot
    out.append(f"| **All** | **{r}** | **{sl}** | **{v}** | **{100*v/max(sl,1):.1f}%** "
               f"| **{lk}** | **{100*lk/max(r,1):.1f}%** |")

    out.append("\n*Hint-copy %* = step lines that are a verbatim `Hints` entry, "
               "over all step lines.  \n*Leak %* = rows where the gold answer "
               "appears in a step before the final line.")

    dest = pathlib.Path(f"outputs/reports/cot_baseline_{args.target}.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
