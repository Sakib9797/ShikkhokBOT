# scripts/cot_core.py
"""Phase 2 shared machinery — prompt contract and validators.

Imported by ``02_generate_cot.py`` (pilot), ``02b_bulk_cot.py`` (full run),
``02c_baseline_metrics.py`` (the upstream-CoT baseline) and
``02d_model_bakeoff.py`` (local model selection). Kept in its own module
because the numeric-prefixed script names are not importable as packages.

Model-agnostic by design: the prompt and the validators are the same whether
the chains come from a hosted API or from a local Qwen on the 5090. Only
``llm_client.py`` knows how the tokens are produced.

The prompt deliberately withholds ``Hints``: feeding them back is precisely how
the upstream dataset ended up 70% hint-copy.
"""
import json
import pathlib
import re
import unicodedata

from llm_client import load_env  # noqa: F401  (re-exported for the scripts)

# --- structured output contract -------------------------------------------

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

# Few-shot exemplars matter more for a local 32B than they did for a frontier
# model: they pin the output shape, the step granularity, and the register of
# the Bengali. Do not trim these to save prompt tokens.
_EXEMPLARS = """
উদাহরণ ১ —
প্রশ্ন: উদ্ভিদের সালোকসংশ্লেষণ প্রক্রিয়ায় কোন গ্যাস গ্রহণ করা হয়?
ধাপ ১: সালোকসংশ্লেষণ হলো সেই প্রক্রিয়া যেখানে সবুজ উদ্ভিদ সূর্যালোকের শক্তি ব্যবহার করে খাদ্য তৈরি করে।
ধাপ ২: এই প্রক্রিয়ায় উদ্ভিদকে কার্বন কাঠামো তৈরির জন্য একটি কার্বনযুক্ত কাঁচামাল বায়ু থেকে নিতে হয়।
ধাপ ৩: বায়ুমণ্ডলে উপস্থিত কার্বনযুক্ত গ্যাসটি পত্ররন্ধ্র দিয়ে পাতায় প্রবেশ করে এবং ক্লোরোপ্লাস্টে বিক্রিয়ায় অংশ নেয়।
ধাপ ৪: অতএব গ্রহণ করা গ্যাসটি চিহ্নিত করা যায়।
final_answer: কার্বন ডাই-অক্সাইড

উদাহরণ ২ —
প্রশ্ন: বিশুদ্ধ পানির pH মান কত?
ধাপ ১: pH দ্বারা কোনো দ্রবণে হাইড্রোজেন আয়নের ঘনত্ব প্রকাশ করা হয়।
ধাপ ২: বিশুদ্ধ পানিতে সামান্য বিয়োজনে সমান সংখ্যক H⁺ ও OH⁻ আয়ন উৎপন্ন হয়।
ধাপ ৩: দুই ধরনের আয়নের ঘনত্ব সমান হলে দ্রবণটি নিরপেক্ষ, এবং ২৫°C তাপমাত্রায় নিরপেক্ষ বিন্দুর মান নির্ধারিত।
final_answer: ৭

উদাহরণ ৩ —
প্রশ্ন: বলের একক কী?
ধাপ ১: নিউটনের দ্বিতীয় সূত্র অনুযায়ী বল = ভর × ত্বরণ।
ধাপ ২: SI পদ্ধতিতে ভরের একক কিলোগ্রাম এবং ত্বরণের একক মিটার/সেকেন্ড²।
ধাপ ৩: সুতরাং বলের একক দাঁড়ায় কিলোগ্রাম·মিটার/সেকেন্ড², যাকে বিজ্ঞানী নিউটনের নামে একটি নাম দেওয়া হয়েছে।
final_answer: নিউটন
""".strip()

_JSON_RULE = (
    'শুধুমাত্র JSON আউটপুট দাও, এই আকারে: '
    '{"steps": ["...", "...", "..."], "final_answer": "..."} — '
    'কোনো অতিরিক্ত লেখা, ব্যাখ্যা বা markdown fence যোগ করবে না।'
)

SYSTEM = (
    "তুমি একজন বাংলা মাধ্যমিক (SSC) বিজ্ঞান শিক্ষক। "
    "প্রশ্নের উত্তরে পৌঁছাতে বিষয়ভিত্তিক নীতির উপর ভিত্তি করে ধাপে ধাপে যুক্তি সাজাও। "
    "প্রতিটি ধাপ আগের ধাপ থেকে যৌক্তিকভাবে এগোবে — শুধু তথ্য পুনরাবৃত্তি নয়। "
    "মধ্যবর্তী কোনো ধাপে চূড়ান্ত উত্তর লিখবে না; উত্তর শুধু final_answer-এ থাকবে। "
    "৩–৬টি ধাপ ব্যবহার করো। সম্পূর্ণ উত্তর বাংলায় লিখবে।\n\n"
    + _EXEMPLARS + "\n\n" + _JSON_RULE
)

# Appended to the user turn when retrying a chain the validators rejected.
RETRY_NUDGE = (
    "\n\nগুরুত্বপূর্ণ: আগের চেষ্টাটি বাতিল হয়েছে। কোনো মধ্যবর্তী ধাপে চূড়ান্ত উত্তর "
    "লিখবে না, hint-এর বাক্য হুবহু অনুলিপি করবে না, এবং প্রতিটি ধাপ যেন নতুন "
    "যুক্তি যোগ করে।"
)


def build_user(e):
    """The per-question turn. Note: Hints are never included — by design."""
    tags = ", ".join(e.get("TopicTags") or [])
    return (
        f"বিষয়: {e['_subject']}\n"
        f"অধ্যায়/ট্যাগ: {tags}\n"
        f"প্রশ্ন: {e['Question']}\n"
        f"সঠিক উত্তর (শুধু তোমার জন্য — যেন যুক্তি সঠিক উত্তরে পৌঁছায়): "
        f"{'; '.join(e['ExactAnswer'])}\n"
        f"এখন উপরের নিয়ম মেনে বাংলা reasoning chain তৈরি করো।"
    )


class Rejected(Exception):
    """A response we cannot use: unparseable, truncated, or off-contract."""


# --- validators: the whole point of the exercise ---------------------------

BN = re.compile(r"[ঀ-৿]")
_STEP_PREFIX = re.compile(r"^\s*ধাপ\s*[০-৯0-9]+\s*[:：.]\s*")

HINT_COPY_THRESHOLD = 0.34      # baseline to beat: 70.2%


def nrm(s):
    return unicodedata.normalize("NFC", " ".join(str(s).split()))


def strip_step_prefix(s):
    """Drop a leading `ধাপ ৩:` so step text compares against raw hint text."""
    return _STEP_PREFIX.sub("", nrm(s))


def validate(data, e):
    """Return a rejection reason, or None if the chain is usable."""
    steps = [s for s in data.get("steps", []) if str(s).strip()]
    final = data.get("final_answer", "")
    if not (3 <= len(steps) <= 6):
        return "step_count"

    body = " ".join(steps)
    golds = [a for a in e.get("ExactAnswer", []) if str(a).strip()]
    if any(nrm(a) in nrm(body) for a in golds):
        return "answer_leak"                    # baseline to beat: 9.4%

    hints = {nrm(h) for h in (e.get("Hints") or []) if str(h).strip()}
    if hints:
        copied = sum(strip_step_prefix(s) in hints for s in steps)
        if copied / len(steps) > HINT_COPY_THRESHOLD:
            return "hint_copy"

    txt = body + final
    if len(BN.findall(txt)) / max(len(txt), 1) < 0.5:
        return "not_bengali"
    if golds and not any(nrm(a) in nrm(final) for a in golds):
        return "wrong_answer"
    return None


def to_cot(data):
    """Render the structured chain into the training-time CoT string."""
    cot = "\n".join(f"ধাপ {i + 1}: {s}" for i, s in enumerate(data["steps"]))
    return cot + f"\nতাই সঠিক উত্তর: {data['final_answer']}"


# --- shared io helpers -----------------------------------------------------

def read_jsonl(path):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(fh, obj):
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()
