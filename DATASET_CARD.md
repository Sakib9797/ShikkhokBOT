---
language:
- bn
task_categories:
- question-answering
- text-generation
tags:
- bengali
- bangla
- chain-of-thought
- education
- ssc
size_categories:
- 10K<n<100K
---

# ShikkhokBot — SSC Bangla Science CoT

Bengali chain-of-thought reasoning for SSC-level Biology, Chemistry and Physics
short-answer questions, generated with the Claude API on top of the
[SSC-BanglaTutor](https://huggingface.co/datasets) corpus.

## Why this exists

SSC-BanglaTutor ships a `CoT` field, but it is the `Hints` list reformatted
rather than reasoning. Measured over all 11,258 rows that carry a chain
(`scripts/02c_baseline_metrics.py`):

| Split | Rows | Step lines | Verbatim hints | Hint-copy % | Answer-leak rows | Leak % |
|---|---:|---:|---:|---:|---:|---:|
| Biology | 4,859 | 30,188 | 20,253 | 67.1% | 433 | 8.9% |
| Chemistry | 3,034 | 18,366 | 14,422 | 78.5% | 342 | 11.3% |
| Physics | 3,365 | 20,224 | 13,637 | 67.4% | 282 | 8.4% |
| **All** | **11,258** | **68,778** | **48,312** | **70.2%** | **1,057** | **9.4%** |

*Hint-copy %* = step lines that are a verbatim `Hints` entry.
*Leak %* = rows stating the gold answer in a step before the final line.

This dataset regenerates every chain from the raw questions, with `Hints`
deliberately withheld from the prompt — feeding them back is exactly how the
upstream chains ended up 70% hint-copy. The corresponding row for this dataset
is in `reports/cot_baseline_ours.md`.

## Construction

1. **Repair.** The three raw files have 29 lines that fail `json.loads` — 27
   Physics `Invalid \escape`, one Physics unescaped `"` inside a string, one
   Biology spurious `}`. All 29 were mechanically repaired, taking the corpus
   from 11,257 parseable to **11,286 rows — exactly the published count**.
   Every edit is logged before/after in `reports/` (`repairs.jsonl`), and
   nothing was silently dropped.
2. **Normalize.** Key-tolerant load (the corpus mixes `Question`/`question`,
   `Hints`/`hints`); `ExactAnswer` coerced to `list[str]`; a stable
   content-hash `qid` = `sha1(subject|NFC-normalized question)[:16]`.
3. **Dedup.** 383 cross-subject duplicate questions removed on the NFC-normalized
   question → **10,903 unique**. Every subject a question appeared under is kept
   in `_subjects`.
4. **Split.** A subject-stratified **100-question hold-out** (`test100.jsonl`)
   that is never trained on and never sent to Claude; the remaining **10,803**
   rows form the generation/training pool.
5. **Generate.** Structured JSON output (`steps[3..6]`, `final_answer`) through
   the Batches API. Pilot on `claude-opus-5`, full run on `claude-sonnet-5`.
   The gold answer is supplied privately so the chain lands correctly; `Hints`
   are not.
6. **Validate.** Every chain is machine-checked and the reason is kept in
   `_reject` (null = clean): `step_count`, `answer_leak` (gold answer appears in
   a step), `hint_copy` (>34% of steps are verbatim hints), `not_bengali`
   (<50% Bengali codepoints), `wrong_answer` (final answer misses the gold).
   Rejected rows are retried once with a stricter prompt, then quarantined.

## Fields

| Field | Type | Notes |
|---|---|---|
| `qid` | `str` | stable content hash — safe to join on across re-runs |
| `Question` | `str` | |
| `ExactAnswer` | `list[str]` | normalized from the corpus's mixed `str`/`list` |
| `Hints` | `list[str]` | carried through, **not** used in generation |
| `CoT` | `str` | `ধাপ ১: …` lines + `তাই সঠিক উত্তর: …` |
| `_cot_steps` | `list[str]` | the same chain unrendered |
| `_reject` | `str \| null` | validator verdict; `null` = clean |
| `_subject` / `_subjects` | `str` / `list[str]` | Biology, Chemistry, Physics |
| `Candidates_Answers`, `Convergence`, `Convergence_Ranked`, `TopicTags` | | inherited from the source corpus |

`TopicTags` are near-unique (4,146 distinct tags over 11k rows, with case
variants) — **stratify on subject, not tags.**

## Intended use and limits

Supervised fine-tuning of Bengali instruction models for SSC science tutoring.
Fields are short — question p50 42 characters, answer p50 12 — so these are
short-answer items, not long-form problems.

The chains are model-generated and machine-validated, not expert-verified. A
clean `_reject` means the chain does not leak the answer, does not copy hints,
is in Bengali, and lands on the gold answer; it does not certify that the
reasoning is pedagogically ideal. Sample before relying on it.

Llama-family tokenizers spend 4–8 tokens per Bengali word, so these chains are
token-expensive relative to their character length — check your sequence-length
budget before training.

## License

Inherited from the source SSC-BanglaTutor corpus. Generated chains are released
under the same terms.

## Citation

Generated with the Claude API (`claude-opus-5` pilot, `claude-sonnet-5` full
run). Pipeline: <https://github.com/> — `scripts/00_validate.py` through
`scripts/06_publish.py`.
