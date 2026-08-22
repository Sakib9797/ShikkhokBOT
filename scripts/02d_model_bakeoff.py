# scripts/02d_model_bakeoff.py
"""Phase 2 model selection — compare pilot files from different local models.

Open models vary far more in Bengali than frontier APIs do, so which one
generates the corpus is an empirical question, not a reputation call. Generate
one pilot per candidate, then rank them on the validators that define the whole
contribution.

    python scripts/02_generate_cot.py -n 100 --model Qwen/Qwen3-32B-AWQ \
        --out data/cot/pilot_qwen3-32b.jsonl
    python scripts/02_generate_cot.py -n 100 --model CohereLabs/aya-expanse-32b \
        --out data/cot/pilot_aya-32b.jsonl
    python scripts/02d_model_bakeoff.py data/cot/pilot_*.jsonl

Pass rate is necessary, not sufficient: a model can satisfy every rule and
still write shallow Bengali. Read the sampled chains this prints before
committing the GPU-hours.
"""
import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cot_core import BN, read_jsonl  # noqa: E402


def summarize(path):
    rows = read_jsonl(path)
    if not rows:
        return None
    n = len(rows)
    rejects = collections.Counter(r["_reject"] for r in rows if r.get("_reject"))
    clean = n - sum(rejects.values())
    steps = [len(r.get("_cot_steps") or []) for r in rows]
    chars = [len(r.get("CoT") or "") for r in rows]
    bn_ratio = []
    for r in rows:
        t = r.get("CoT") or ""
        bn_ratio.append(len(BN.findall(t)) / max(len(t), 1))
    return {
        "path": pathlib.Path(path).name,
        "model": rows[0].get("_cot_model", "?"),
        "n": n,
        "clean": clean,
        "pass": clean / n,
        "leak": rejects.get("answer_leak", 0) / n,
        "hint": rejects.get("hint_copy", 0) / n,
        "wrong": rejects.get("wrong_answer", 0) / n,
        "notbn": rejects.get("not_bengali", 0) / n,
        "count": rejects.get("step_count", 0) / n,
        "steps": sum(steps) / n,
        "chars": sum(chars) / n,
        "bn": sum(bn_ratio) / n,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="pilot jsonl files, one per model")
    ap.add_argument("--samples", type=int, default=2,
                    help="chains to print per model for the human read")
    args = ap.parse_args()

    stats = [s for s in (summarize(p) for p in args.files) if s]
    if not stats:
        sys.exit("no readable pilot files")
    stats.sort(key=lambda s: -s["pass"])

    out = ["# Local model bake-off\n",
           "| Model | n | Pass % | Leak % | Hint-copy % | Wrong ans % | "
           "Not-Bengali % | Bad step count % | Avg steps | Avg chars | Bengali char % |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in stats:
        out.append(
            f"| `{s['model']}` | {s['n']} | **{s['pass']:.0%}** | {s['leak']:.0%} "
            f"| {s['hint']:.0%} | {s['wrong']:.0%} | {s['notbn']:.0%} | {s['count']:.0%} "
            f"| {s['steps']:.1f} | {s['chars']:.0f} | {s['bn']:.0%} |")
    out.append("\nPass % = chains clearing every validator. The two that carry the "
               "paper are **Leak %** (upstream baseline 9.4%) and **Hint-copy %** "
               "(upstream baseline 70.2%) — but note those baselines are corpus-wide "
               "rates while these are per-chain rejection rates, so they are "
               "directionally comparable, not identical measures.\n")
    dest = pathlib.Path("outputs/reports/model_bakeoff.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"wrote {dest}\n")

    for s in stats[:3]:
        print("=" * 70)
        print(f"{s['model']}  —  {s['pass']:.0%} pass")
        for r in [x for x in s["rows"] if not x.get("_reject")][:args.samples]:
            print(f"\n  প্রশ্ন: {r['Question']}")
            for line in (r.get("CoT") or "").splitlines():
                print(f"    {line}")
    print("\n" + "=" * 70)
    print("Read those chains. Pass rate cannot tell you whether the reasoning is "
          "shallow, and shallow-but-valid is the failure mode to watch for.")


if __name__ == "__main__":
    main()
