# ShikkhokBot

A fine-tuned Bangla SSC science tutor (Biology / Chemistry / Physics).

**Research contribution:** genuine Bengali chain-of-thought reasoning, generated
with open-source models running locally, replacing the source corpus's `CoT`
field — which is the `Hints` list reformatted, measured at **70.2% verbatim
hint-copy and 9.4% answer-leak** across 11,258 rows.

No paid API is involved anywhere in this pipeline. Generation runs against a
local OpenAI-compatible server (vLLM, Ollama, LM Studio, llama.cpp) on a
5090-class GPU box; training and evaluation run on the same machine.

Full design rationale and measured corpus figures: [PLAN.md](PLAN.md).

## Pipeline status

| Phase | What | Status |
|---|---|---|
| 0 | Validate + repair the 29 malformed lines | **done** — 11,286 / 11,286 published rows, 0 quarantined |
| 1 | Merge, dedup, stratified split | **done** — 10,903 unique · 100 test · 10,803 pool |
| 2 | Baseline CoT-quality measurement | **done** — `outputs/reports/cot_baseline_upstream.md` |
| 2 | Generate Bengali CoT locally | code ready — **needs the GPU box serving a model** |
| 3 | QLoRA fine-tune | code ready — **needs the GPU box** |
| 4 | Evaluate vs. zero-shot baseline | code ready — scoring half verified locally |
| 5 | Ship (Gradio + HF Spaces + paper) | code ready — needs the adapter |

## Setup

On this machine:

```bash
pip install -r requirements.txt
```

On the 5090 box, serve a model. vLLM gives the best throughput and real
constrained JSON decoding:

```bash
vllm serve Qwen/Qwen3-32B-AWQ --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.92
```

Ollama is the simpler option if vLLM is fussy — it serves an OpenAI-compatible
API on port 11434 and needs `/v1` on the base URL.

Then point `.env` at it:

```
LLM_BASE_URL=http://192.168.1.50:8000/v1
LLM_MODEL=Qwen/Qwen3-32B-AWQ
LLM_API_KEY=EMPTY
HF_TOKEN=
```

Use the box's LAN IP if you run the pipeline from this PC, or keep `localhost`
and run everything on the 5090 directly — the latter is faster, since it skips
the network round-trip on 10,803 requests.

Confirm the link before anything else:

```bash
python scripts/llm_client.py --check
```

That lists the served models, negotiates a JSON mode, and round-trips one
request. If it fails, nothing downstream will work.

## Choosing the model

Bengali quality varies far more between open models than between hosted APIs,
so this is an experiment, not a reputation call. Generate a pilot per candidate
and rank them on the validators that define the contribution:

```bash
python scripts/02_generate_cot.py -n 100 --model Qwen/Qwen3-32B-AWQ --out data/cot/pilot_qwen3-32b.jsonl
```

```bash
python scripts/02d_model_bakeoff.py data/cot/pilot_*.jsonl
```

Candidates worth trying on 32 GB, all 4-bit: **Qwen3-32B** (strong reasoning,
weaker Bengali), **Aya-Expanse-32B** (built for multilingual, Bengali in scope),
**Gemma-3-27B** (a 256k vocab that tokenizes Bengali far better than Llama's).
The bake-off prints sample chains — read them. Pass rate cannot tell you whether
the reasoning is shallow, and shallow-but-valid is the failure mode to watch.

## Running it

Phases 0–2a are local, free and already run. Commands are here for
reproducibility from the raw corpus.

```bash
python scripts/00_validate.py
```
```bash
python scripts/01_prepare.py
```
```bash
python scripts/02c_baseline_metrics.py
```

Pilot 30 questions and read every chain by eye before scaling:

```bash
python scripts/02_generate_cot.py
```

Gate on all four before the full run: the Bengali reads naturally, `_reject` is
null on ≥90%, answer-leak ≈ 0%, and the printed throughput makes the wall-clock
wait tolerable.

Then generate the whole pool. It is resumable — kill it whenever you need the
GPU and rerun to pick up where it stopped:

```bash
python scripts/02b_bulk_cot.py --workers 8
```
```bash
python scripts/02b_bulk_cot.py --status
```
```bash
python scripts/02b_bulk_cot.py --retry
```

Then produce your half of the paper's comparison table:

```bash
python scripts/02c_baseline_metrics.py --target ours
```

Training. Stop the inference server first — 32 GB does not hold a 32B server
and a training run at the same time:

```bash
python scripts/03_train.py --data data/cot/all_cot.jsonl
```

Evaluation splits across machines: generate on the GPU, score anywhere.

```bash
python scripts/04_eval.py --generate
```
```bash
python scripts/04_eval.py
```

Ship:

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
  llm_client.py          local OpenAI-compatible client + JSON-mode negotiation
  cot_core.py            prompt contract and validators (model-agnostic)
  02_generate_cot.py     pilot; one run per candidate model
  02b_bulk_cot.py        full pool, concurrent and resumable
  02c_baseline_metrics.py  hint-copy % and answer-leak %, upstream vs ours
  02d_model_bakeoff.py   rank candidate models on validator pass rate
  03_train.py            QLoRA fine-tune
  04_eval.py             exact-match + Bengali-aware ROUGE-L
  05_demo.py             Gradio chat
  06_publish.py          push dataset / adapter to the Hub
data/clean/              merged · test100 · train_pool
data/cot/                pilot_* · all_cot · quarantine
outputs/reports/         integrity.md · repairs.jsonl · cot_baseline_* · model_bakeoff
outputs/eval/            preds_base · preds_ft · report.md
DATASET_CARD.md          the HF dataset README
```

`data/` and `outputs/` are git-ignored — they hold ~1.5 GB of corpora.

## Notes for anyone re-running this

- **`qid` is a content hash, not a row index.** Re-running Phase 1 with a
  different seed does not invalidate Phase 2's results.
- **Local servers differ on structured output.** `llm_client.py` tries
  `json_schema`, then vLLM's `guided_json`, then `json_object`, then plain
  prompting, and remembers which one the server accepted.
- **Reasoning models emit `<think>` blocks inline.** These are stripped before
  JSON parsing, as are markdown fences. An unterminated `<think>` is treated as
  truncation rather than parsed around.
- **The ROUGE tokenizer override in `04_eval.py` is load-bearing.**
  `rouge_score`'s default strips every Bengali codepoint and scores 0.0.
- **Print Bengali to a file, not the Windows console.** cp1252 mangles it on
  output, which looks exactly like data corruption and is not.
- **The 5090 is Blackwell (sm_120).** It needs a CUDA 12.8+ PyTorch build. A
  "no kernel image is available" error from bitsandbytes or Unsloth means the
  wrong wheel, not a bug in the training code.
