# scripts/agent.py
"""A tutoring agent over the SSC corpus — retrieve, reason, verify, revise.

The plain chat path is one shot: question in, answer out, no memory, no checks.
This wraps the same model in a loop that behaves like an agent:

    1. RETRIEVE  search the 10,903-question corpus for related items (a tool —
                 the agent consults your verified data instead of trusting the
                 model's recall)
    2. REASON    draft a Bengali chain, given the retrieved context and the
                 conversation so far
    3. VERIFY    run the project's own validators over its own draft
    4. REVISE    if a validator fires, tell the model exactly what went wrong
                 and try again, up to --max-attempts

Every run returns a trace of those steps, so the loop is inspectable rather
than a black box — the demo renders it, and it is what makes the behaviour
legible as an agent rather than a chatbot.

Deliberately NOT here: autonomous multi-step goal pursuit, or letting the model
choose arbitrary tools. This project's claim is about data quality; an agent
architecture that outran that claim would be complexity for its own sake.

Retrieval is lexical (shared rare words), not embeddings: no GPU, no extra
dependency, no model download, and on 10,903 short Bengali questions it is
both fast and good enough. Swap in embeddings only if recall proves too low.
"""
import math
import pathlib
import re
import sys
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cot_core import COT_SCHEMA, SYSTEM, read_jsonl, to_cot  # noqa: E402
from llm_client import LLMError  # noqa: E402

CORPUS = "data/clean/merged.jsonl"
_WORD = re.compile(r"[ঀ-৿]+|[A-Za-z0-9]+")

# Above this lexical score we treat a retrieved row as the same question, which
# means we hold its verified gold answer and can check the draft against it.
STRONG_MATCH = 0.55


def tokens(text):
    return _WORD.findall(unicodedata.normalize("NFC", str(text)))


class CorpusIndex:
    """Tiny TF-IDF retriever over the SSC questions. The agent's one tool."""

    def __init__(self, path=CORPUS):
        self.rows = read_jsonl(path)
        self.postings = defaultdict(list)      # term -> [(row_idx, tf), ...]
        self.norms = []
        df = Counter()
        tokenized = []
        for r in self.rows:
            tf = Counter(tokens(r.get("Question", "")))
            tokenized.append(tf)
            df.update(tf.keys())
        n = max(len(self.rows), 1)
        # rare words carry the signal; "কোন"/"কী" appear everywhere and do not
        self.idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}
        for i, tf in enumerate(tokenized):
            norm = 0.0
            for t, c in tf.items():
                w = c * self.idf.get(t, 0.0)
                self.postings[t].append((i, w))
                norm += w * w
            self.norms.append(math.sqrt(norm) or 1.0)

    def search(self, query, k=4):
        qtf = Counter(tokens(query))
        if not qtf:
            return []
        scores = defaultdict(float)
        qnorm = 0.0
        for t, c in qtf.items():
            w = c * self.idf.get(t, 0.0)
            qnorm += w * w
            for i, dw in self.postings.get(t, ()):
                scores[i] += w * dw
        qnorm = math.sqrt(qnorm) or 1.0
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.rows[i], s / (qnorm * self.norms[i])) for i, s in ranked]

    def __len__(self):
        return len(self.rows)


# --- verification -----------------------------------------------------------

BN = re.compile(r"[ঀ-৿]")


def nrm(s):
    return unicodedata.normalize("NFC", " ".join(str(s).split()))


def verify(data, gold=None):
    """Check the agent's own draft. Returns a reason string, or None if fine.

    At inference there is usually no gold answer, so only the structural rules
    apply. When retrieval matched strongly we *do* hold a verified answer, and
    the stronger check — did the chain land on it — becomes available.
    """
    steps = [s for s in data.get("steps", []) if str(s).strip()]
    final = str(data.get("final_answer", "")).strip()
    if not (3 <= len(steps) <= 6):
        return f"{len(steps)} ধাপ — ৩ থেকে ৬টি ধাপ হতে হবে"
    if not final:
        return "final_answer ফাঁকা"
    txt = " ".join(steps) + final
    if len(BN.findall(txt)) / max(len(txt), 1) < 0.5:
        return "উত্তরটি বাংলায় লেখা হয়নি"
    body = " ".join(steps[:-1]) if len(steps) > 1 else ""
    if gold and body and nrm(gold) in nrm(body):
        return "চূড়ান্ত উত্তর মাঝের ধাপে বলে দেওয়া হয়েছে"
    if gold and nrm(gold) not in nrm(final):
        return f"চূড়ান্ত উত্তর ভুল — পাঠ্যক্রম অনুযায়ী সঠিক উত্তর: {gold}"
    return None


# --- the agent --------------------------------------------------------------

class TutorAgent:
    def __init__(self, client, index=None, max_attempts=3, history_turns=4):
        self.client = client
        self.index = index if index is not None else CorpusIndex()
        self.max_attempts = max_attempts
        self.history_turns = history_turns

    # -- step 1
    def retrieve(self, question):
        hits = self.index.search(question, k=4)
        gold = None
        if hits and hits[0][1] >= STRONG_MATCH:
            answers = hits[0][0].get("ExactAnswer") or []
            if answers:
                gold = str(answers[0])
        return hits, gold

    @staticmethod
    def _context_block(hits):
        if not hits:
            return ""
        lines = ["পাঠ্যক্রম থেকে সম্পর্কিত প্রশ্ন ও উত্তর (রেফারেন্স):"]
        for row, score in hits:
            ans = ", ".join(map(str, row.get("ExactAnswer") or []))
            lines.append(f"- ({score:.2f}) {row.get('Question','')} → {ans}")
        return "\n".join(lines) + "\n\n"

    def _history_block(self, history):
        """Recent turns, so follow-ups like 'ধাপ ৩ আবার বোঝাও' resolve."""
        if not history:
            return ""
        turns = history[-self.history_turns * 2:]
        lines = ["আগের কথোপকথন:"]
        for m in turns:
            if isinstance(m, dict):
                role, content = m.get("role"), m.get("content", "")
            else:
                role, content = None, str(m)
            who = "শিক্ষার্থী" if role == "user" else "শিক্ষক"
            lines.append(f"{who}: {str(content)[:400]}")
        return "\n".join(lines) + "\n\n"

    # -- the loop
    def answer(self, question, history=None):
        """Return (answer_text, trace). Trace is a list of (step, detail)."""
        trace = []
        hits, gold = self.retrieve(question)
        trace.append(("🔍 Retrieve",
                      f"searched {len(self.index):,} corpus questions · "
                      + (f"top match {hits[0][1]:.2f}" if hits else "no match")
                      + (f" · verified answer available: {gold}" if gold
                         else " · no verified answer, structural checks only")))

        context = self._context_block(hits) + self._history_block(history)
        feedback = ""
        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            user = (f"{context}"
                    f"শিক্ষার্থীর প্রশ্ন: {question}\n"
                    f"{feedback}"
                    f"উপরের নিয়ম মেনে বাংলা reasoning chain তৈরি করো।")
            try:
                data, _ = self.client.complete(SYSTEM, user, COT_SCHEMA)
            except LLMError as exc:
                trace.append(("⚠️ Error", str(exc)))
                return f"⚠️ {exc}", trace
            except Exception as exc:
                trace.append(("⚠️ Error", f"{type(exc).__name__}: {exc}"))
                return f"⚠️ {type(exc).__name__}: {exc}", trace

            n_steps = len(data.get("steps") or [])
            trace.append((f"🧠 Reason (attempt {attempt})",
                          f"drafted {n_steps} steps"))

            problem = verify(data, gold)
            if problem is None:
                trace.append(("✅ Verify", "passed every check"))
                return to_cot(data), trace

            last_error = problem
            trace.append(("❌ Verify", problem))
            if attempt < self.max_attempts:
                trace.append(("🔁 Revise", "sending the failure back to the model"))
                feedback = (f"তোমার আগের উত্তরে সমস্যা ছিল: {problem}\n"
                            f"এবার সেটি ঠিক করে আবার লেখো।\n")

        # Out of attempts: return the last draft, flagged. Hiding the failure
        # would be worse than showing it — a student deserves to know.
        trace.append(("⚠️ Gave up", f"{self.max_attempts} attempts, still: {last_error}"))
        return (to_cot(data) +
                f"\n\n⚠️ স্বয়ংক্রিয় যাচাইয়ে সমস্যা রয়ে গেছে: {last_error}"), trace


def format_trace(trace):
    return "\n".join(f"**{step}** — {detail}" for step, detail in trace)
