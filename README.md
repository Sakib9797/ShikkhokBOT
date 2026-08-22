# ShikkhokBot

A fine-tuned Bangla SSC science tutor (Biology / Chemistry / Physics).

**Research contribution:** genuine Bengali chain-of-thought reasoning generated
with the Claude API, replacing the source corpus's `CoT` field — which is the
`Hints` list reformatted, measured at **70.2% verbatim hint-copy and 9.4%
answer-leak** across 11,258 rows.

Full design rationale, measured corpus figures and cost model: [PLAN.md](PLAN.md).

## Pipeline status

| Phase | What | Status |
|---|---|---|
| 0 | Validate + repair the 29 malformed lines | **done** — 11,286 / 11,286 published rows, 0 quarantined |
| 1 | Merge, dedup, stratified split | **done** — 10,903 unique · 100 test · 10,803 pool |
| 2 | Baseline CoT-quality measurement | **done** — `outputs/reports/cot_baseline_upstream.md` |
| 2 | Generate Bengali CoT via Claude | code ready — **needs `ANTHROPIC_API_KEY`** |
| 3 | QLoRA fine-tune (Colab T4) | code ready — **needs a GPU** |
| 4 | Evaluate vs. zero-shot baseline | code ready — scoring half verified locally |
| 5 | Ship (Gradio + HF Spaces + paper) | code ready — needs the adapter |

## Setup

```bash
pip install -r requirements.txt
```

Put your credentials in `.env` (git-ignored):

```
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
```

## Running it

Phases 0–2a are local and free, and have already been run — the commands are
here so the results are reproducible from the raw corpus.

```bash
python scripts/00_validate.py                       # repair -> 11,286 rows
python scripts/01_prepare.py                        # dedup + split
python scripts/02c_baseline_metrics.py              # the upstream baseline table
```

Phase 2 costs money. Pilot first, read all 30 chains, then scale:

```bash
python scripts/02_generate_cot.py                   # 30 questions on Opus 5
```

The pilot prints measured token averages and re-costs the full run from them.
Gate on all four before scaling: the Bengali reads naturally, `_reject` is null
on ≥90%, answer-leak ≈ 0%, and the token averages match the estimate. A
100-row side-by-side against `claude-sonnet-5` decides the batch model:

```bash
python scripts/02_generate_cot.py -n 100 --model claude-sonnet-5 --out data/cot/compare_sonnet.jsonl
```

Then the full pool through the Batches API (50% off; ~$47 on Sonnet 5):

```bash
python scripts/02b_batch_cot.py submit
```

```bash
python scripts/02b_batch_cot.py status
```

```bash
python scripts/02b_batch_cot.py collect
```

`collect` merges by `custom_id` — batch results come back in arbitrary order,
so nothing keys on position. Rejected chains retry with a stricter prompt:

```bash
python scripts/02b_batch_cot.py submit --retry
```

Then measure ours against the baseline — this fills in the paper's table:

```bash
python scripts/02c_baseline_metrics.py --target ours
```

Phase 3 runs on Colab (T4, free tier), not on Windows. Upload
`data/cot/all_cot.jsonl` and the script; it asserts the p99 token length fits
`max_seq_length` **before** the 3–4 hour run, because Llama-3.2's tokenizer
bloats Bengali and silent truncation would train the model to stop mid-chain.

```bash
python scripts/03_train.py --data all_cot.jsonl
```

Phase 4 splits across machines: generate on the GPU box, score anywhere.

```bash
python scripts/04_eval.py --generate
```

```bash
python scripts/04_eval.py
```

Phase 5:

```bash
python scripts/05_demo.py
```

```bash
python scripts/06_publish.py dataset --repo <user>/ShikkhokBot-SSC-Bangla-CoT
```

## Layout

```
scripts/
  00_validate.py         repair the 29 malformed raw lines; the key-tolerant loader
  01_prepare.py          merge, dedup, stratified 100-question hold-out
  cot_core.py            prompt contract, extraction, validators (shared)
  02_generate_cot.py     streaming pilot + Opus/Sonnet side-by-side
  02b_batch_cot.py       full run via the Batches API (submit/status/collect)
  02c_baseline_metrics.py  hint-copy % and answer-leak %, upstream vs ours
  03_train.py            QLoRA on Llama-3.2-3B (Colab T4)
  04_eval.py             exact-match + Bengali-aware ROUGE-L
  05_demo.py             Gradio chat
  06_publish.py          push dataset / adapter to the Hub
data/clean/              merged · test100 · train_pool
data/cot/                pilot_cot · all_cot · quarantine
outputs/reports/         integrity.md · repairs.jsonl · cot_baseline_*.md
outputs/eval/            preds_base · preds_ft · report.md
DATASET_CARD.md          the HF dataset README
```

`data/` and `outputs/` are git-ignored — they hold ~1 GB of corpora.

## Notes for anyone re-running this

- **`qid` is a content hash, not a row index.** Re-running Phase 1 with a
  different seed does not invalidate Phase 2's results.
- **The ROUGE tokenizer override in `04_eval.py` is load-bearing.**
  `rouge_score`'s default strips every Bengali codepoint and scores 0.0.
- **Print Bengali to a file, not the Windows console.** cp1252 mangles it on
  output, which looks exactly like data corruption and is not.
