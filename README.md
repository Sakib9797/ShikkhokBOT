# ShikkhokBot

A fine-tuned Bangla SSC science tutor (Biology / Chemistry / Physics).

**Research contribution:** genuine Bengali chain-of-thought reasoning, generated
with open-source models running locally, replacing the source corpus's `CoT`
field — which is the `Hints` list reformatted, measured at **70.2% verbatim
hint-copy and 9.4% answer-leak** across 11,258 rows.

Generation runs against any OpenAI-compatible endpoint, so you can pick a
backend per run:

| `--provider` | What it is | Trade-off |
|---|---|---|
| `local` (default) | vLLM / Ollama / LM Studio / llama.cpp on the 5090 box | Free, private, no rate limits — you supply the hardware |
| `groq` | Groq's hosted API | Very fast, no GPU needed — metered, rate-limited, and it sees your data |

Same prompt, same schema, same validators either way, so the two are directly
comparable and `02d_model_bakeoff.py` can rank models across both. Training and
evaluation always run on the GPU box.

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

### Option A — Groq (no GPU needed)

Get a key from https://console.groq.com/keys and put it in `.env` as
`GROQ_API_KEY=gsk_...`. That file is git-ignored; never paste a key into a
script or a commit. Then:

```bash
python scripts/llm_client.py --check --provider groq
```

Groq hosts `qwen/qwen3-32b` (the default here) plus Llama and GPT-OSS models.
Its free tier rate-limits, so the client paces itself at 55 requests/minute by
default — tune with `--rpm`, and keep `--workers` modest. On a reasoning model,
add `--reasoning-format hidden` to stop paying for `<think>` tokens you discard.

### Option B — local on the 5090

vLLM gives the best throughput and real constrained JSON decoding:

```bash
vllm serve Qwen/Qwen3-32B-AWQ --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.92
```

Ollama is the simpler option if vLLM is fussy — it serves an OpenAI-compatible
API on port 11434 and needs `/v1` on the base URL.

Then point `.env` at it:

```
LLM_PROVIDER=local
LLM_BASE_URL=http://192.168.1.50:8000/v1
LLM_MODEL=Qwen/Qwen3-32B-AWQ
LLM_API_KEY=EMPTY
GROQ_API_KEY=
HF_TOKEN=
```

`LLM_PROVIDER` sets the default; `--provider` overrides it per run. The
`LLM_BASE_URL` / `LLM_MODEL` pair applies only to the provider named in
`LLM_PROVIDER` — switching to `--provider groq` uses Groq's own endpoint and
model rather than dragging a localhost URL along.

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
python scripts/02_generate_cot.py -n 100 --provider groq --model qwen/qwen3-32b --out data/cot/pilot_groq-qwen3.jsonl
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

On Groq instead, with pacing that respects the free tier:

```bash
python scripts/02b_bulk_cot.py --provider groq --workers 4 --rpm 55 --reasoning-format hidden
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

Ship. The demo defaults to **agent mode**: each question runs a
retrieve → reason → verify → revise loop, and the answer carries an expandable
trace of the steps taken.

```bash
python scripts/05_demo.py --provider groq
```

```bash
python scripts/05_demo.py --backend chat
```

```bash
python scripts/05_demo.py --backend adapter
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
  agent.py               retrieve/verify/revise loop + corpus retriever
  05_demo.py             Gradio chat (agent | chat | adapter backends)
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
- **Endpoints differ on structured output.** `llm_client.py` tries
  `json_schema`, then vLLM's `guided_json` (local only — Groq rejects it), then
  `json_object`, then plain prompting, and remembers which one was accepted.
- **`json_object` mode requires the literal word "JSON" in the prompt.** The
  Bengali system prompt contains it deliberately — do not remove it.
- **The agent's retriever is lexical, not embeddings.** TF-IDF over 10,903
  short Bengali questions builds in 0.3s and searches in ~1ms, with no GPU, no
  model download and no extra dependency. Swap in embeddings only if recall
  proves too low in practice.
- **A strong retrieval match unlocks the stronger checks.** Above a 0.55 score
  the agent holds a verified gold answer, so it can check whether its own draft
  actually lands on it. Below that, only the structural rules apply — the agent
  never invents a gold answer to grade itself against.
- **Rate-limited rows are not quarantined.** If Groq throttles, those rows stay
  pending and the next run picks them up; only genuine failures quarantine.
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
