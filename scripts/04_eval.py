# scripts/04_eval.py
"""Phase 4 — evaluate the fine-tuned adapter against the zero-shot baseline.

Scores the untouched `data/clean/test100.jsonl` (never trained on, never sent
to Claude) on exact-match and ROUGE-L, and writes `outputs/eval/report.md`.

    python scripts/04_eval.py --generate          # needs a GPU: runs both models
    python scripts/04_eval.py                     # score existing predictions

Generation writes `outputs/eval/preds_{base,ft}.jsonl` so the (GPU, slow) half
and the (local, fast) scoring half can run on different machines: generate on
Colab, download the two prediction files, score here.

The ROUGE tokenizer override is load-bearing. `rouge_score`'s default does
`re.sub(r"[^a-z0-9]+", " ", text.lower())`, which strips every Bengali
codepoint and scores the entire test set 0.0.
"""
import argparse
import json
import pathlib
import re
import sys
import unicodedata

TEST = "data/clean/test100.jsonl"
EVAL_DIR = pathlib.Path("outputs/eval")
SYS = "তুমি একজন বাংলা মাধ্যমিক বিজ্ঞান শিক্ষক। ধাপে ধাপে ব্যাখ্যা করে উত্তর দাও।"


class BanglaTokenizer:
    """Keeps Bengali codepoints, which the rouge_score default throws away."""

    def tokenize(self, text):
        return re.findall(r"[ঀ-৿]+|[A-Za-z0-9]+", unicodedata.normalize("NFC", text))


def nrm(s):
    s = unicodedata.normalize("NFC", " ".join(str(s).split()))
    return re.sub(r"[।,;:!?\"'()\[\]]", "", s).strip()


def exact(pred, golds):
    p = nrm(pred)
    return any(nrm(g) and nrm(g) in p for g in golds)


def read_jsonl(path):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# --- generation (GPU) ------------------------------------------------------

def generate(adapter, base, out_path, rows, max_new_tokens=512):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        adapter or base, max_seq_length=2048, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for i, e in enumerate(rows, 1):
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": e["Question"]}]
            ids = tok.apply_chat_template(
                msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
            pred = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({"qid": e["qid"], "pred": pred}, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                print(f"  {i}/{len(rows)}")
    print(f"wrote {out_path}")


# --- scoring (local) -------------------------------------------------------

def score(rows, preds, scorer):
    by_qid = {p["qid"]: p["pred"] for p in preds}
    em = rl = 0.0
    n = 0
    for e in rows:
        pred = by_qid.get(e["qid"])
        if pred is None:
            continue
        n += 1
        golds = [g for g in e["ExactAnswer"] if str(g).strip()]
        em += exact(pred, golds)
        if golds:
            rl += scorer.score(golds[0], pred)["rougeL"].fmeasure
    return n, (em / n if n else 0.0), (rl / n if n else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="run both models (needs GPU)")
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--adapter", default="outputs/adapter")
    args = ap.parse_args()

    rows = read_jsonl(TEST)
    if not rows:
        sys.exit(f"{TEST} missing — run scripts/01_prepare.py first.")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    p_base, p_ft = EVAL_DIR / "preds_base.jsonl", EVAL_DIR / "preds_ft.jsonl"

    if args.generate:
        print("generating baseline (zero-shot)...")
        generate(None, args.base, p_base, rows)
        print("generating fine-tuned...")
        generate(args.adapter, args.base, p_ft, rows)

    missing = [str(p) for p in (p_base, p_ft) if not p.exists()]
    if missing:
        sys.exit(f"no predictions at {missing} — rerun with --generate on a GPU box, "
                 f"or copy the prediction files here.")

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False,
                                      tokenizer=BanglaTokenizer())

    nb, emb, rlb = score(rows, read_jsonl(p_base), scorer)
    nf, emf, rlf = score(rows, read_jsonl(p_ft), scorer)

    md = ["# ShikkhokBot — evaluation\n",
          f"Held-out test set: `{TEST}` ({len(rows)} questions, never trained on, "
          "never sent to Claude).\n",
          "## Automatic metrics\n",
          "| Model | Scored | Exact-match | ROUGE-L |",
          "|---|---:|---:|---:|",
          f"| Llama-3.2-3B-Instruct (zero-shot) | {nb} | {emb:.1%} | {rlb:.3f} |",
          f"| + Bengali-CoT QLoRA (ours) | {nf} | {emf:.1%} | {rlf:.3f} |",
          f"| **Delta** | | **{emf - emb:+.1%}** | **{rlf - rlb:+.3f}** |",
          "\nExact-match = a gold `ExactAnswer` appears in the generation "
          "(punctuation-stripped, NFC-normalized). ROUGE-L uses a Bengali-aware "
          "tokenizer; the `rouge_score` default strips Bengali entirely.\n",
          "## CoT quality vs. the upstream dataset\n",
          "See `outputs/reports/cot_baseline_upstream.md` and "
          "`outputs/reports/cot_baseline_ours.md` — hint-copy % and answer-leak %. "
          "This is the paper's contribution table.\n",
          "## Human evaluation\n",
          "Blind ratings from 5-10 SSC students/teachers on correctness and "
          "clarity: fill in from `outputs/eval/human_ratings.csv`.\n"]
    dest = EVAL_DIR / "report.md"
    dest.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
