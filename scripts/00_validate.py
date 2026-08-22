# scripts/00_validate.py
"""Phase 0 — validate and repair the SSC-BanglaTutor raw corpus.

Turns the 11,257 parseable rows into the full 11,286 published rows by
repairing the 29 malformed lines, and exports the one key-tolerant loader
(``load_ssc_raw``) that every later phase imports.

Repairs applied, in order, to any line that fails ``json.loads``:
  1. escape        : a backslash not starting a valid JSON escape -> ``\\\\``
                     (the 27 Physics ``Invalid \\escape`` lines)
  2. drop_brace    : a spurious ``}`` closing the object right after
                     ``Candidates_Answers`` (Biology line 2823, ``Extra data``)
  3. escape_quotes : an unescaped ``("")`` seconds marker inside a string
                     (Physics line 194, ``Expecting ',' delimiter``)

Every edit is recorded (before/after) in ``outputs/reports/repairs.jsonl``.
"""
import json
import pathlib
import re
from collections import Counter

RAW = {
    "Biology":   "data/SSC-BanglaTutor/Bio/SSC_Biology_Datasets.jsonl",
    "Chemistry": "data/SSC-BanglaTutor/chem/SSC_Chemistry_Dataset.jsonl",
    "Physics":   "data/SSC-BanglaTutor/phy/SSC_Physics_Dataset.jsonl",
}
PUBLISHED_TOTAL = 11286          # SSC-BanglaTutor paper, Feb 2026

# a backslash NOT starting a valid JSON escape -> escape it
_BAD_ESC = re.compile(r'\\(?!["\\/bfnrtu])')


def repair_candidates(line: str):
    """Yield (label, candidate) repairs to try, in priority order.

    Each candidate targets one documented defect. The caller keeps the first
    candidate that parses; the label names which rule fired for the repair log.
    """
    # 1. the 27 Physics `Invalid \escape` lines
    yield "escape", _BAD_ESC.sub(r'\\\\', line)
    # 2. Biology line 2823: a spurious `}` closes the object right after the
    #    Candidates_Answers list, before Convergence -> drop that one brace.
    if '"]}, "Convergence"' in line:
        yield "drop_brace", line.replace('"]}, "Convergence"', '"], "Convergence"', 1)
    # 3. Physics line 194: seconds marker `("")` has unescaped inner quotes.
    if '("")' in line:
        yield "escape_quotes", line.replace('("")', '(\\"\\")', 1)


def get(entry, *names, default=None):
    """Key-tolerant accessor — the corpus mixes Question/question, Hints/hints."""
    for n in names:
        if n in entry:
            return entry[n]
    return default


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def load_ssc_raw(path: str, subject: str, repairs: list | None = None):
    """Yield validated dicts. Repairs recoverable lines; quarantines the rest.

    Returns ``(rows, quarantine)``. Every row is normalized so downstream code
    can rely on the canonical keys ``Question``, ``Hints``, ``ExactAnswer``,
    ``TopicTags`` and the injected ``_subject`` — never raising ``KeyError``.
    """
    rows, quarantine = [], []
    for i, line in enumerate(pathlib.Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e0:
            entry = None
            for label, cand in repair_candidates(line):
                try:
                    entry = json.loads(cand)
                except json.JSONDecodeError:
                    continue
                if repairs is not None:
                    repairs.append({"subject": subject, "line": i, "rule": label,
                                    "error": str(e0), "before": line, "after": cand})
                break
            if entry is None:
                quarantine.append({"subject": subject, "line": i,
                                   "error": str(e0), "raw": line})
                continue
        entry["_subject"] = subject
        entry["Question"]    = get(entry, "Question", "question", default="")
        entry["Hints"]       = as_list(get(entry, "Hints", "hints"))
        entry["ExactAnswer"] = as_list(get(entry, "ExactAnswer"))
        entry["TopicTags"]   = as_list(get(entry, "TopicTags"))
        rows.append(entry)
    return rows, quarantine


def _has_surrogate(s: str) -> bool:
    return any(0xD800 <= ord(c) <= 0xDFFF for c in s)


def _walk_strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _walk_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_strings(v)


if __name__ == "__main__":
    pathlib.Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    report = ["# Integrity report\n"]
    repairs, quarantine, total = [], [], 0
    keys = Counter()
    rule_counts = Counter()
    surrogate_rows = 0

    report.append("## Per-file counts\n")
    for subj, p in RAW.items():
        rows, q = load_ssc_raw(p, subj, repairs)
        quarantine += q
        total += len(rows)
        for r in rows:
            keys.update(r.keys())
            if any(_has_surrogate(s) for s in _walk_strings(r)):
                surrogate_rows += 1
        report.append(f"- {subj}: {len(rows)} rows kept, {len(q)} quarantined")

    for r in repairs:
        rule_counts[r["rule"]] += 1

    report.append(f"\n**Total: {total} / {PUBLISHED_TOTAL} published**")
    report.append(f"\n## Repairs\n\nAuto-repaired lines: {len(repairs)}")
    for rule, c in sorted(rule_counts.items()):
        report.append(f"- `{rule}`: {c}")
    report.append(f"\nRemaining quarantined (unrepairable): {len(quarantine)}")
    report.append(f"\n## Surrogate scan\n\nRows with a lone UTF-16 surrogate: "
                  f"{surrogate_rows} (regression check — expected 0)")
    report.append(f"\n## Key census\n\n```\n{dict(sorted(keys.items()))}\n```")

    for name, data in (("repairs", repairs), ("quarantine", quarantine)):
        with open(f"outputs/reports/{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    pathlib.Path("outputs/reports/integrity.md").write_text(
        "\n".join(report), encoding="utf-8", newline="\n")

    # Console-safe summary (avoids printing Bengali to a cp1252 Windows console)
    print(f"total={total}/{PUBLISHED_TOTAL}  repaired={len(repairs)}  "
          f"quarantined={len(quarantine)}  surrogate_rows={surrogate_rows}")
    print("repair rules:", dict(rule_counts))
    print("wrote outputs/reports/{integrity.md,repairs.jsonl,quarantine.jsonl}")
