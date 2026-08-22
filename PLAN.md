# ShikkhokBot — Execution Plan (v2, verified)

> Fine-tuned Bangla SSC science tutor (Biology / Chemistry / Physics).
> Research novelty: **genuine Bengali Chain-of-Thought (CoT) reasoning** generated with the **Claude (Anthropic) API**, layered on the SSC-BanglaTutor Q&A dataset.
>
> **Status (2026-08-23): Phases 0, 1 and the Phase-2 baseline measurement are DONE and their outputs are on disk. Phases 2 (generation), 3, 4 and 5 are code-complete in `scripts/` and blocked only on an `ANTHROPIC_API_KEY` and a GPU.** See `README.md` for the status table and run order. Every number in §0 was *measured* on the files on disk (see §0.4 for how).
>
> Supersedes plan v1, which contained four factual errors. See §0.1.

---

## 0. Verified state of the repo

### 0.0 Files

| Path | What it is | Measured condition |
|---|---|---|
| `ShikkhokBot_Project_Guide.md` | Master 7-week guide | Reference. Guide says GPT-4o for CoT — **we use Claude instead**. |
| `data/SSC-BanglaTutor/Bio/SSC_Biology_Datasets.jsonl` | Raw Q&A, 4,859 lines | **4,858 parse** · 1 malformed · strict UTF-8 OK · CRLF |
| `data/SSC-BanglaTutor/chem/SSC_Chemistry_Dataset.jsonl` | Raw Q&A, 3,035 lines | **3,034 parse** · 0 malformed · strict UTF-8 OK · CRLF |
| `data/SSC-BanglaTutor/phy/SSC_Physics_Dataset.jsonl` | Raw Q&A, 3,394 lines | **3,365 parse** · 28 malformed · 1 blank · strict UTF-8 OK · **LF only** |
| `data/SSC-BanglaTutor-CoT/*_with_cot.jsonl` | Q&A + `CoT` field | Parses fine, **but the CoT is fake — see §0.2. Discard.** |
| `data/Bangla-TextBook/` | 87,110 rows (`text`) | HF arrow. Aux corpus — **not used in the core pipeline**. |
| `data/Bangla-Instruct/` | 342,391 rows (`instruction`,`response`) | HF arrow. Aux corpus — **not used in the core pipeline**. |
| `preview_ssc.py`, `ssc_preview.txt`, `cot_sample_preview.txt` | Week-1 exploration helpers | Keep as-is |
| `_diag.py`, `_diag2.py`, `_diag_out.txt`, `_diag2_out.txt` | Throwaway verification scripts | **Fold into `scripts/00_validate.py`, then delete** |

Not present, must be created: `scripts/`, `outputs/`, `data/clean/`, `data/cot/`, `.env`, `requirements.txt`, `.gitignore`. **This is not a git repo yet** — `git init` before Phase 0.

### 0.1 Corrections to plan v1

| v1 claimed | Reality |
|---|---|
| Raw files "✅ Clean UTF-8", counts 4,859 / 3,035 / 3,394 | Those were **line** counts. **29 lines are not valid JSON.** Valid rows: 4,858 / 3,034 / 3,365 = **11,257**. |
| "**Encoding corruption** — lone UTF-16 surrogates in CoT files" | **False — a phantom.** All 6 files decode as **strict UTF-8 with no error handler**, and **0 rows** contain a lone surrogate. The garbled `প\udc8dরাচীন` in v1 was the **Windows cp1252 console** mangling Bengali on print, not data damage. |
| Phase 0 exit: "total ≈ 11,288" | Unachievable as written — parsing alone yields 11,257. Corrected in Phase 0. |
| CoT files share the raw schema | **Schema drift exists** — see §0.3. |

### 0.2 The one real defect (confirmed, and it justifies the whole project)

The shipped `CoT` field is **the `Hints` list reformatted**, not reasoning:

```
ধাপ 1: <hint 1>
ধাপ 2: <hint 2>
...
ধাপ N: তাই সঠিক উত্তর হলো "<ExactAnswer>"
```

Measured on Biology (4,859 rows):

- **20,198 / 30,122 step-lines (67.1%) are verbatim hints.**
- **432 / 4,859 rows (8.9%) leak the gold answer before the final line.**

This is exactly the gap the research novelty claims to fill. **Decision: discard `data/SSC-BanglaTutor-CoT/` entirely and regenerate from raw.** These two numbers become the **baseline table in the paper** — "prior CoT: 67.1% hint-copy, 8.9% answer leak; ours: X%, Y%."

> **Measured across all three subjects (2026-08-23, `scripts/02c_baseline_metrics.py`, reproducing the Biology figures above exactly): 70.2% hint-copy (48,312 / 68,778 step lines) and 9.4% answer-leak (1,057 / 11,258 rows). Chemistry is the worst offender at 78.5% / 11.3%.** Full table: `outputs/reports/cot_baseline_upstream.md`.

### 0.3 Newly discovered issues (undocumented in v1)

1. **29 malformed raw lines — and they are the *entire* gap to the published count.**
   `11,257 parsed + 29 malformed = 11,286` — **exactly** the 11,286 the SSC-BanglaTutor paper claims. Repairing them reconstructs the published corpus precisely.
   Breakdown: Physics 27 × `Invalid \escape`, Physics 1 × `Expecting ',' delimiter` (an unescaped `"` inside a string — `সেকেন্ড (")`), Biology 1 × `Extra data` (line 2823).
   **None recover by object-splitting** — a `raw_decode` stream pass confirms they are not two JSON objects glued together. They are genuine escaping bugs from the dataset authors' writer. The 27 backslash cases are mechanically repairable; the remaining 2 need eyeball fixes.

2. **The CoT files already dropped the same bad lines.** CoT Physics = 3,365 = valid raw Physics; CoT Chemistry = 3,034. Whoever built them hit the identical 28 Physics failures and silently skipped them — independent confirmation the breakage is in the raw files, not our reader.

3. **Schema drift.** In CoT Biology: 11 rows use lowercase `question` / `hints`; 10 rows carry an extra `Sources` key; 1 row lacks `Convergence` / `Convergence_Ranked` / `TopicTags`. Chemistry has `TopicTags` on 3,033 / 3,034. **Every loader must be key-tolerant** (`r.get("Question") or r.get("question")`), never `r["Question"]`.

4. **383 cross-subject duplicate questions** (NFC-normalized): Bio 68, Chem 89, Phy 226 → **10,874 unique**. Dedup *before* the test split, or a test question's twin sits in the training set.

5. **`ExactAnswer` is inconsistently typed** — `list` in 9,351 rows, `str` in 1,906. Normalize to `list[str]` at load.

6. **`TopicTags` are unusable for stratification** — 4,146 distinct tags over 11k rows (near-unique), with case variants (`chemistry`/`Chemistry`, `physics`/`Physics`). **Stratify on subject, not tags.** Case-fold tags for reporting only.

7. **Field sizes are small** — Question p50 = 42 chars (max 125); Answer p50 = 12 chars (max 358). These are short-answer items, which keeps CoT generation cheap and makes exact-match a meaningful metric. 1 row has no hints; 0 rows lack an answer.

8. **Line endings differ** (raw Physics is LF, everything else CRLF). Harmless if you always read with `splitlines()`; a landmine if anything ever splits on `"\n"` and leaves `\r`. Write all outputs LF, `ensure_ascii=False`.

### 0.4 How these numbers were obtained

`_diag.py` and `_diag2.py` — file-shape census, strict-UTF-8 decode of all 6 files with **no** error handler, per-line vs stream parse, post-`json.loads` surrogate scan, error taxonomy on the 29 bad lines, `raw_decode` recovery attempt, NFC dedup, percentile field stats, key-presence census, and token/cost inputs. Both wrote UTF-8 **to a file** rather than to stdout — printing Bengali to the Windows console is what produced v1's phantom corruption finding. **This logic becomes `scripts/00_validate.py`; the temp scripts get deleted.**

### 0.5 Raw entry schema

```
Question            : str            # (11 CoT rows use lowercase "question")
Hints               : list[str]      # 5–6 progressive hints
ExactAnswer         : str | list[str]  # normalize → list[str]
Candidates_Answers  : list[str]
Convergence         : list[dict]     # {"Up to 1": 0.6, ...}
Convergence_Ranked  : list[str]
TopicTags           : list[str]      # sometimes absent
```

---

## Phase overview

| Phase | Goal | Output | Guide week |
|---|---|---|---|
| 0 | Validate + **repair the 29 lines** | `scripts/00_validate.py`, `outputs/reports/integrity.md` | 1 |
| 1 | Merge / dedup / split | `data/clean/{merged,train_pool,test100}.jsonl` | 1 |
| 2 | **Genuine Bengali CoT via Claude API** ⭐ | `data/cot/*.jsonl`, HF dataset | 2 |
| 3 | QLoRA fine-tune (Unsloth, Llama-3.2-3B) | LoRA adapter | 3–4 |
| 4 | Evaluate (ROUGE-L, exact-match, human) | `outputs/eval/report.md` | 5 |
| 5 | Ship (Gradio + Spaces + paper) | demo, adapter, dataset, arXiv draft | 6–7 |

Repo layout to create:

```
scripts/     00_validate.py 01_prepare.py 02_generate_cot.py 02b_batch_cot.py
             03_train.py 04_eval.py 05_demo.py
data/clean/  merged.jsonl  train_pool.jsonl  test100.jsonl
data/cot/    pilot_cot.jsonl  all_cot.jsonl  quarantine.jsonl
outputs/     adapter/  eval/  reports/
.env         ANTHROPIC_API_KEY=...   HF_TOKEN=...
.gitignore   .env  data/  outputs/  *_out.txt
requirements.txt
```

---

## Phase 0 — Validate and repair (local, free, do first)

**Goal:** turn 11,257 parseable rows into the full **11,286** published rows, and produce the one key-tolerant loader every later phase imports.

Tasks:
- [x] `git init`; write `.gitignore` (**`.env` first**).
- [x] Port `_diag*.py` logic into `scripts/00_validate.py`; delete the temp files.
- [x] Auto-repair the 27 `Invalid \escape` lines; hand-repair the 2 remaining; write `outputs/reports/repairs.jsonl` documenting **every** edit (before/after) — this goes in the paper's data-cleaning note. **All 29 repaired mechanically; 0 quarantined.**
- [x] Emit `outputs/reports/integrity.md`: per-file counts, repair log, key census, surrogate scan.
- [x] Export `load_ssc_raw()`.

```python
# scripts/00_validate.py
import json, pathlib, re
from collections import Counter

RAW = {
    "Biology":   "data/SSC-BanglaTutor/Bio/SSC_Biology_Datasets.jsonl",
    "Chemistry": "data/SSC-BanglaTutor/chem/SSC_Chemistry_Dataset.jsonl",
    "Physics":   "data/SSC-BanglaTutor/phy/SSC_Physics_Dataset.jsonl",
}
PUBLISHED_TOTAL = 11286          # SSC-BanglaTutor paper, Feb 2026

# a backslash NOT starting a valid JSON escape -> escape it
_BAD_ESC = re.compile(r'\\(?!["\\/bfnrtu])')

def repair(line: str) -> str:
    return _BAD_ESC.sub(r'\\\\', line)

def get(entry, *names, default=None):
    """Key-tolerant accessor — the corpus mixes Question/question, Hints/hints."""
    for n in names:
        if n in entry:
            return entry[n]
    return default

def as_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def load_ssc_raw(path: str, subject: str, repairs: list | None = None):
    """Yield validated dicts. Repairs recoverable lines; quarantines the rest."""
    rows, quarantine = [], []
    for i, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e0:
            fixed = repair(line)
            try:
                entry = json.loads(fixed)
                if repairs is not None:
                    repairs.append({"subject": subject, "line": i,
                                    "error": str(e0), "before": line, "after": fixed})
            except json.JSONDecodeError as e1:
                quarantine.append({"subject": subject, "line": i,
                                   "error": str(e1), "raw": line})
                continue
        entry["_subject"] = subject
        entry["Question"]    = get(entry, "Question", "question", default="")
        entry["Hints"]       = as_list(get(entry, "Hints", "hints"))
        entry["ExactAnswer"] = as_list(get(entry, "ExactAnswer"))
        entry["TopicTags"]   = as_list(get(entry, "TopicTags"))
        rows.append(entry)
    return rows, quarantine

if __name__ == "__main__":
    pathlib.Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    report, repairs, quarantine, total = ["# Integrity report\n"], [], [], 0
    keys = Counter()
    for subj, p in RAW.items():
        rows, q = load_ssc_raw(p, subj, repairs)
        quarantine += q
        total += len(rows)
        for r in rows:
            keys.update(r.keys())
        report.append(f"- {subj}: {len(rows)} rows kept, {len(q)} quarantined")
    report.append(f"\n**Total: {total} / {PUBLISHED_TOTAL} published**")
    report.append(f"\nAuto-repaired lines: {len(repairs)}")
    report.append(f"\nKey census: {dict(keys)}")
    for name, data in (("repairs", repairs), ("quarantine", quarantine)):
        with open(f"outputs/reports/{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    pathlib.Path("outputs/reports/integrity.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
```

**Result (achieved 2026-08-12):** all 29 lines repaired mechanically — the two "needs eyeball" cases turned out to be rule-expressible (a spurious `}` before `Convergence`, and an unescaped `("")` seconds marker). **11,286 / 11,286 published rows, quarantine empty.**

**Exit criteria (corrected, achievable):**
- Auto-repair recovers **27** lines → total **11,284**.
- `outputs/reports/quarantine.jsonl` holds exactly **2** rows (Bio 2823 `Extra data`, Phy `Expecting ','`). Fix both by hand → **11,286 = published count.**
- 0 lone surrogates (already verified — this is a regression check, not a discovery).
- Key census printed; no `KeyError` anywhere downstream.

> If a quarantined line proves unrepairable, **drop it and say so in the dataset card.** 2 rows out of 11,286 is 0.02% and is not worth blocking on — but it must be *documented*, not silently lost, which is exactly what the upstream CoT files did.

---

## Phase 1 — Prepare, dedup, split

**Goal:** one clean deduped corpus + a 100-question test set that is **never** trained on and **never** sent to Claude.

Tasks:
- [x] Merge 3 subjects, normalize `ExactAnswer` → `list[str]`, attach a **stable content-hash `qid`**.
- [x] Dedup on NFC-normalized `Question` (expect ~383 removals → **10,874**). Record every subject a question appeared under in `_subjects`.
- [x] Stratified **100-question** hold-out by subject → `test100.jsonl`.
- [x] Remainder → `train_pool.jsonl` (**10,803 actual**).

```python
# scripts/01_prepare.py
import json, pathlib, random, hashlib, unicodedata
from collections import defaultdict
from importlib import import_module
v = import_module("scripts.00_validate".replace("00", "00"))  # or: from validate import ...

random.seed(42)
def norm(s): return unicodedata.normalize("NFC", " ".join(str(s).split()))
def qid(subject, question): 
    return hashlib.sha1(f"{subject}|{norm(question)}".encode()).hexdigest()[:16]

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

# stratified 100-question hold-out
by_subj = defaultdict(list)
for e in deduped: by_subj[e["_subject"]].append(e)
test, pool = [], []
for subj, items in by_subj.items():
    random.shuffle(items)
    n = round(100 * len(items) / len(deduped))
    test += items[:n]; pool += items[n:]
test = test[:100]
test_ids = {e["qid"] for e in test}
pool = [e for e in pool if e["qid"] not in test_ids]      # belt and braces

pathlib.Path("data/clean").mkdir(parents=True, exist_ok=True)
def dump(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for e in data: f.write(json.dumps(e, ensure_ascii=False) + "\n")
dump("data/clean/merged.jsonl", deduped)
dump("data/clean/test100.jsonl", test)
dump("data/clean/train_pool.jsonl", pool)
print(f"deduped={len(deduped)} test={len(test)} pool={len(pool)}")
```

**Result (achieved 2026-08-12):** `deduped = 10,903`, `test = 100`, `pool = 10,803`. Higher than the plan's 10,874 estimate because that was computed over 11,257 parsed rows; Phase 0 recovered all 29, so 29 more rows survive dedup.

**Exit criteria:** `deduped ≈ 10,874`, `test = 100`, `pool ≈ 10,774`; `test_ids ∩ pool_ids = ∅` asserted; subject proportions in `test100` match the corpus.

> **Why a content-hash `qid` and not `enumerate(i)`:** v1 keyed resumability on list position. Re-run Phase 1 with a different seed or a repaired line and every id shifts — the Phase 2 resume set silently maps to the wrong questions. A hash of `subject|question` is stable across re-runs, re-orderings, and re-repairs.

---

## Phase 2 — Genuine Bengali CoT via the Claude API ⭐

**Goal:** for every training-pool question, a **real** step-by-step Bengali chain that derives the answer from subject principles — not hint-restatement, not answer-leaking.

**Provider:** Anthropic API via the official `anthropic` Python SDK. Needs `ANTHROPIC_API_KEY`.

### 2.1 Corrections to the v1 stub

| v1 code | Problem | Fix |
|---|---|---|
| `max_tokens=1024` | On Opus 5 / Sonnet 5, `max_tokens` bounds **thinking + response text together**. With adaptive thinking on and token-hungry Bengali, 1024 truncates mid-chain and yields invalid JSON. | `max_tokens=4096` |
| `next(b.text for b in msg.content if b.type == "text")` | Raises `StopIteration` on `stop_reason: "refusal"` — an HTTP 200 with empty/partial content. Non-trivial here: this is a **biology** dataset and a `bio` refusal category exists. | Check `stop_reason` first; handle empty content explicitly |
| No `effort` setting | Default is `high`. Mechanical CoT generation doesn't need it, and effort is the **single biggest cost lever**. | `"effort": "medium"` |
| `output_config` had only `format` | `effort` and `format` **share one `output_config` dict** — passing them as two kwargs is an error | merge into one dict |
| `id = enumerate(i)` | Unstable across re-runs (see Phase 1 note) | `custom_id = qid` |

Confirmed correct in v1 and kept: `claude-opus-5` as a valid current model ID; `output_config={"format": {"type":"json_schema", ...}}` (top-level `output_format` is deprecated); `thinking={"type":"adaptive"}` (`budget_tokens` would 400 on these models — thinking is on by default on Opus 5, so the explicit param is correct-but-redundant; keep it for clarity).

### 2.2 Model and cost

Assumptions: ~400 input tokens/request, ~800 output tokens/request **including thinking tokens, which bill as output**. Bengali tokenizes poorly, so treat these as **±2×** — the pilot's `usage` field replaces them with real numbers before committing to the full run.

| Config | 7,000 questions | Full pool (10,774) |
|---|---|---|
| Opus 5, streaming | $154 | $237 |
| **Opus 5, Batches (−50%)** | **$77** | **$119** |
| Sonnet 5 (intro), streaming | $62 | $95 |
| **Sonnet 5 (intro), Batches** | **$31** | **$47** |
| Haiku 4.5, Batches | $15 | $24 |

`effort: "low"` cuts the output side roughly 40% on top of any row.

> **Sonnet 5 intro pricing ($2/$10 vs $3/$15) ends 2026-08-31 — 22 days from today.** Phases 0–1 are a day's work, so the full batch can comfortably land inside that window. Worth targeting.

### 2.3 Decisions (v1 left these open — now answered)

1. **Scope → the full pool (~10,774), not a 5–7k sample.** The guide's "5,000–7,000" was a scoping estimate, not a constraint. At Sonnet-5-batch rates the difference is ~$16. A *complete* reasoning-augmented SSC-BanglaTutor is a materially stronger dataset-card and paper claim than a sample, and more SFT data helps the 3B.
2. **Model → pilot on `claude-opus-5`, full run on `claude-sonnet-5`.** Opus establishes the quality ceiling on 30 questions for ~$0.40. Then generate 100 rows on each and eyeball them side by side; if Sonnet holds, it runs the other 10,674 at ~⅖ the cost. Escalate back to Opus only if the comparison shows a real gap.
3. **Batches API, yes** — 50% off, and 10,774 requests fits one batch (limit 100,000 / 256 MB). Most batches finish in under an hour, 24h max, results retained 29 days.
4. **Phase 3 trains on Colab T4** (free tier, per the guide). Nothing else in the pipeline needs a GPU.

### 2.4 Prompt contract

Bengali system instruction: reason from subject principles toward the answer (`ধাপ ১…`); **never** state the final answer in an intermediate step; 3–6 steps; SSC-curriculum-appropriate. The gold answer is supplied privately so the chain lands correctly. `Hints` are **deliberately withheld from the prompt** — feeding them back is precisely how the upstream dataset ended up 67.1% hint-copy.

> **Add 2–3 Bengali few-shot exemplars to the system prompt.** Two payoffs: better step quality, and it pushes the shared prefix past Opus 5's **512-token minimum cacheable length**, so `cache_control` starts paying off. v1's bare system prompt sits under 512 tokens, where caching simply never engages.

### 2.5 Pilot (30 questions, streaming)

```python
# scripts/02_generate_cot.py — PILOT PATH
import json, pathlib, unicodedata, re
from anthropic import Anthropic

client = Anthropic()                      # reads ANTHROPIC_API_KEY
MODEL  = "claude-opus-5"                  # pilot; full run switches to claude-sonnet-5

COT_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {"type": "string"},
                  "minItems": 3, "maxItems": 6},
        "final_answer": {"type": "string"},
    },
    "required": ["steps", "final_answer"],
    "additionalProperties": False,
}

SYSTEM = (
    "তুমি একজন বাংলা মাধ্যমিক (SSC) বিজ্ঞান শিক্ষক। "
    "প্রশ্নের উত্তরে পৌঁছাতে বিষয়ভিত্তিক নীতির উপর ভিত্তি করে ধাপে ধাপে যুক্তি সাজাও। "
    "প্রতিটি ধাপ আগের ধাপ থেকে যৌক্তিকভাবে এগোবে — শুধু তথ্য পুনরাবৃত্তি নয়। "
    "মধ্যবর্তী কোনো ধাপে চূড়ান্ত উত্তর লিখবে না; উত্তর শুধু final_answer-এ থাকবে। "
    "৩–৬টি ধাপ ব্যবহার করো। সম্পূর্ণ উত্তর বাংলায় লিখবে।"
    # + 2–3 few-shot exemplars here (also pushes the prefix past the 512-token cache floor)
)

def build_user(e):
    return (
        f"বিষয়: {e['_subject']}\n"
        f"অধ্যায়/ট্যাগ: {', '.join(e.get('TopicTags', []))}\n"
        f"প্রশ্ন: {e['Question']}\n"
        f"সঠিক উত্তর (শুধু তোমার জন্য — যেন যুক্তি সঠিক উত্তরে পৌঁছায়): "
        f"{'; '.join(e['ExactAnswer'])}\n"
        f"এখন উপরের নিয়ম মেনে বাংলা reasoning chain তৈরি করো।"
    )

REQ = dict(
    model=MODEL,
    max_tokens=4096,                       # thinking + text share this budget
    thinking={"type": "adaptive"},
    system=SYSTEM,
    output_config={                        # effort and format share ONE dict
        "format": {"type": "json_schema", "schema": COT_SCHEMA},
        "effort": "medium",
    },
)

class Rejected(Exception): pass

def extract(msg):
    if msg.stop_reason == "refusal":
        raise Rejected("refusal")
    blocks = [b.text for b in msg.content if b.type == "text"]
    if not blocks:
        raise Rejected(f"empty content, stop_reason={msg.stop_reason}")
    if msg.stop_reason == "max_tokens":
        raise Rejected("truncated — raise max_tokens")
    return json.loads("".join(blocks))

def generate_one(e):
    with client.messages.stream(
        messages=[{"role": "user", "content": build_user(e)}], **REQ
    ) as stream:
        msg = stream.get_final_message()
    return extract(msg), msg.usage

# ---------- validators: the whole point of the exercise ----------
def nrm(s): return unicodedata.normalize("NFC", " ".join(str(s).split()))
BN = re.compile(r"[ঀ-৿]")

def validate(data, e):
    steps, final = data["steps"], data["final_answer"]
    if not (3 <= len(steps) <= 6):                    return "step_count"
    body = " ".join(steps)
    if any(a and nrm(a) in nrm(body) for a in e["ExactAnswer"]):
        return "answer_leak"                          # baseline to beat: 8.9%
    hints = {nrm(h) for h in e.get("Hints", [])}
    if hints and sum(nrm(s) in hints for s in steps) / len(steps) > 0.34:
        return "hint_copy"                            # baseline to beat: 67.1%
    txt = body + final
    if len(BN.findall(txt)) / max(len(txt), 1) < 0.5:  return "not_bengali"
    if not any(nrm(a) in nrm(final) for a in e["ExactAnswer"]):
        return "wrong_answer"
    return None

def to_cot(data):
    cot = "\n".join(f"ধাপ {i+1}: {s}" for i, s in enumerate(data["steps"]))
    return cot + f"\nতাই সঠিক উত্তর: {data['final_answer']}"

if __name__ == "__main__":
    pool = [json.loads(l) for l in open("data/clean/train_pool.jsonl", encoding="utf-8")]
    out  = pathlib.Path("data/cot/pilot_cot.jsonl"); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["qid"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    tin = tout = 0
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for e in pool[:30]:                            # PILOT
            if e["qid"] in done: continue
            try:
                data, usage = generate_one(e)
            except Rejected as r:
                print(f"  rejected {e['qid']}: {r}"); continue
            tin += usage.input_tokens; tout += usage.output_tokens
            e["CoT"], e["_cot_model"] = to_cot(data), MODEL
            e["_reject"] = validate(data, e)
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"pilot done. tokens in={tin} out={tout} "
          f"-> avg {tin/30:.0f}/{tout/30:.0f} per request")
    print("REVIEW data/cot/pilot_cot.jsonl BY EYE, and recost §2.2 from these averages.")
```

**Pilot gate — do not scale until all four hold:**
1. Bengali reads naturally and the steps actually *reason* (human read of all 30).
2. `_reject` is null on ≥ 27 / 30.
3. Answer-leak rate ≈ 0% (vs the 8.9% baseline).
4. Measured token averages don't blow past §2.2's estimates.

### 2.6 Full run (Batches API)

```python
# scripts/02b_batch_cot.py
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

MODEL = "claude-sonnet-5"                              # decision §2.3

reqs = [Request(custom_id=e["qid"],
                params=MessageCreateParamsNonStreaming(
                    messages=[{"role": "user", "content": build_user(e)}],
                    **{**REQ, "model": MODEL}))
        for e in todo]                                  # todo = pool minus already-done qids

batch = client.messages.batches.create(requests=reqs)
# poll: client.messages.batches.retrieve(batch.id).processing_status == "ended"
for r in client.messages.batches.results(batch.id):
    e = by_qid[r.custom_id]                             # ← key by custom_id, NEVER by position
    if r.result.type != "succeeded":
        quarantine.append({"qid": r.custom_id, "why": r.result.type}); continue
    try:
        data = extract(r.result.message)
    except Rejected as why:
        quarantine.append({"qid": r.custom_id, "why": str(why)}); continue
    e["CoT"], e["_reject"] = to_cot(data), validate(data, e)
```

Operational notes:
- **Results come back in arbitrary order — key by `custom_id`, never by index.** The most common way to silently corrupt a batch pipeline.
- The `fallbacks` parameter is **rejected on the Batches API**, so refusal handling must be client-side — which `extract()` already is.
- Prompt caching works with Batches; the two discounts stack.
- Retry `_reject != null` rows once with a stricter nudge, then quarantine. Report the final quarantine count in the dataset card.
- Structured outputs pay a one-time schema-compilation latency on first use, then cache 24h — irrelevant at this scale.

Publish (**CV item #2**):
- [ ] Push to HF with a dataset card documenting source, the 29-line repair, dedup, generation model + prompt, validator thresholds, and quarantine count. State the license inherited from SSC-BanglaTutor.

**Exit criteria:** pilot approved by eye; full set validated (0 answer-leaks, hint-copy far below 67.1%, ≥ 95% pass rate); HF dataset live.

---

## Phase 3 — QLoRA fine-tune (Unsloth + Llama-3.2-3B, Colab T4)

The one phase that is **not** local Windows.

Tasks:
- [ ] **First, measure token lengths** (see risk box below).
- [ ] Format: system = tutor persona, user = Question, assistant = `CoT`.
- [ ] `meta-llama/Llama-3.2-3B-Instruct` 4-bit via Unsloth; LoRA r=16, alpha=16, attn+mlp targets; 2–3 epochs (~3–4h on T4).
- [ ] Save to `outputs/adapter/`; push to HF (**CV item #3**).

```python
# scripts/03_train.py  (Colab T4)
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MAXLEN = 2048
model, tok = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", max_seq_length=MAXLEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

SYS = "তুমি একজন বাংলা মাধ্যমিক বিজ্ঞান শিক্ষক। ধাপে ধাপে ব্যাখ্যা করে উত্তর দাও।"

def to_text(ex):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user",   "content": ex["Question"]},
            {"role": "assistant", "content": ex["CoT"]}]
    return {"text": tok.apply_chat_template(msgs, tokenize=False)}

ds = load_dataset("json", data_files="data/cot/all_cot.jsonl")["train"].map(to_text)

# GATE: confirm nothing is being silently truncated
lens = [len(tok(t).input_ids) for t in ds["text"][:2000]]
lens.sort(); print("p50", lens[len(lens)//2], "p99", lens[int(len(lens)*.99)], "max", lens[-1])
assert lens[int(len(lens)*.99)] < MAXLEN, "raise MAXLEN or shorten CoT"

SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
    max_seq_length=MAXLEN,
    args=TrainingArguments(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=3, learning_rate=2e-4, warmup_steps=10,
        fp16=True, logging_steps=10, output_dir="outputs/adapter", optim="adamw_8bit"),
).train()
model.save_pretrained("outputs/adapter"); tok.save_pretrained("outputs/adapter")
```

> **⚠ Risk — Llama-3.2's tokenizer is inefficient on Bengali** (often 4–8 tokens per word, since its BPE vocab is English-dominant). A 400-character Bengali CoT can exceed 1,000 tokens. Hence the p99 assertion **before** the 3–4 hour run — silent truncation would cut the reasoning chain off mid-way and train the model to stop early. If p99 breaches 2048: raise `MAXLEN` (costs VRAM/time on a T4) or cap steps at 5. The same tokenizer inefficiency is the honest limitation to name in the paper.

**Exit criteria:** length gate passes; training loss decreases; adapter saved and reloadable.

---

## Phase 4 — Evaluation

**Goal:** quantify the gain over zero-shot baseline on the untouched `test100.jsonl`.

Tasks:
- [ ] Generate all 100 answers from (a) base Llama-3.2-3B zero-shot, (b) fine-tuned adapter.
- [ ] Metrics: ROUGE-L + exact-match on `ExactAnswer`, **with a Bengali-aware tokenizer**.
- [ ] Blind human eval: 5–10 SSC students/teachers rate correctness + clarity.
- [ ] Also report **CoT-quality deltas vs the upstream dataset** — hint-copy % and answer-leak % — since that is the paper's contribution.
- [ ] Write `outputs/eval/report.md`.

```python
# scripts/04_eval.py
import json, re, unicodedata
from rouge_score import rouge_scorer

# ⚠ rouge_score's DEFAULT tokenizer does re.sub(r"[^a-z0-9]+", " ", text.lower())
#   — it strips every Bengali codepoint and scores everything 0.0. Must override.
class BanglaTokenizer:
    def tokenize(self, text):
        return re.findall(r"[ঀ-৿]+|[A-Za-z0-9]+",
                          unicodedata.normalize("NFC", text))

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False,
                                  tokenizer=BanglaTokenizer())

def nrm(s):
    s = unicodedata.normalize("NFC", " ".join(str(s).split()))
    return re.sub(r"[।,;:!?\"'()\[\]]", "", s).strip()

def exact(pred, golds): return any(nrm(g) in nrm(pred) for g in golds)
def rl(pred, gold):     return scorer.score(gold, pred)["rougeL"].fmeasure

# load test100, run base + finetuned generation, accumulate
# exact-match rate and mean ROUGE-L for each, dump outputs/eval/report.md
```

**Exit criteria:** fine-tuned ≥ baseline on both automatic metrics; human ratings collected; CoT-quality table populated.

---

## Phase 5 — Ship

- [ ] Gradio chat demo (`scripts/05_demo.py`) loading base + adapter (**CV item #3**).
- [ ] Deploy free on HF Spaces.
- [ ] Push adapter + augmented dataset to HF.
- [ ] 4-page paper (**CV item #4**): novelty = Claude-generated Bengali CoT; the 67.1% / 8.9% baseline table is the strongest single figure. arXiv, then optionally the BLP Workshop.

```python
# scripts/05_demo.py
import gradio as gr
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("outputs/adapter", load_in_4bit=True)
FastLanguageModel.for_inference(model)

SYS = "তুমি একজন বাংলা মাধ্যমিক বিজ্ঞান শিক্ষক। ধাপে ধাপে ব্যাখ্যা করে উত্তর দাও।"

def chat(q, history):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, return_tensors="pt",
                                  add_generation_prompt=True).to(model.device)
    out = model.generate(ids, max_new_tokens=512)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

gr.ChatInterface(chat, title="ShikkhokBot — বাংলা বিজ্ঞান শিক্ষক").launch()
```

**Exit criteria:** live Space; HF adapter + dataset public; paper draft written.

---

## Environment

```
# requirements.txt
anthropic                 # Phase 2
datasets huggingface_hub
rouge-score
gradio                    # Phase 5
# Colab-only (Phase 3): unsloth trl transformers peft bitsandbytes torch
```

Secrets in `.env`, **`.gitignore`'d before the first commit**: `ANTHROPIC_API_KEY`, `HF_TOKEN`.

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Llama-3.2 tokenizer bloats Bengali past `max_seq_length` | **High** | p99 assertion before training (Phase 3) |
| Claude's Bengali CoT is fluent but pedagogically shallow | Medium | 30-question human pilot gate before spending anything |
| Refusals on biology content (`bio` classifier) | Low–Medium | `extract()` handles `stop_reason: "refusal"`; quarantine + report |
| Sonnet 5 noticeably worse than Opus 5 on Bengali reasoning | Medium | 100-row side-by-side before committing the batch |
| Sonnet intro pricing lapses 2026-08-31 | Certain if slow | Phases 0–1 are one day; target the batch inside the window |
| 2 quarantined rows unrepairable | Low | Drop and document — 0.02% |

## What's left open

Nothing blocking. Phase 2's §2.3 decisions are reversible up to the moment the batch is submitted; the pilot's measured `usage` numbers replace §2.2's estimates before that point.

---
*Plan v2 — all §0 figures measured on disk. Phases 0–1 and the Phase-2 baseline executed; §2.2 costs are still estimates until the pilot runs.*
