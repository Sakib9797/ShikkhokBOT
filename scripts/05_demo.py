# scripts/05_demo.py
"""Phase 5 — Gradio app: chat with the tutor, and inspect the corpus.

Two backends, so the demo is usable *before* Phase 3 has trained anything:

    python scripts/05_demo.py                      # api backend (works today)
    python scripts/05_demo.py --backend api --provider groq
    python scripts/05_demo.py --backend adapter    # the fine-tuned model
    python scripts/05_demo.py --share              # public link

`api` routes through `llm_client.py` to a local server or Groq, using the same
system prompt the training data was built with. It is a preview of the target
behaviour, not the fine-tuned model — the header says so on screen, because a
demo that quietly shows a 120B when it claims to show a 7B is a lie in a
screenshot.

`adapter` loads `outputs/adapter` and is the real deliverable, available once
`03_train.py` has run.

The second tab is the project's argument made visible: pick any question and
see the upstream corpus's "reasoning" beside ours, with the hint-copy and
answer-leak marks called out. It needs no model at all — it reads the files on
disk — and it doubles as the review surface for the Phase 4 human evaluation.
"""
import argparse
import html
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from agent import CorpusIndex, TutorAgent, format_trace  # noqa: E402
from cot_core import COT_SCHEMA, SYSTEM, nrm, read_jsonl, strip_step_prefix, to_cot  # noqa: E402
from llm_client import LLMError, add_provider_args, client_from_args, load_env  # noqa: E402

TUTOR_SYS = "তুমি একজন বাংলা মাধ্যমিক বিজ্ঞান শিক্ষক। ধাপে ধাপে ব্যাখ্যা করে উত্তর দাও।"

UPSTREAM = {
    "Biology": "data/SSC-BanglaTutor-CoT/SSC_Biology_Datasets_with_cot.jsonl",
    "Chemistry": "data/SSC-BanglaTutor-CoT/SSC_Chemistry_Dataset_with_cot.jsonl",
    "Physics": "data/SSC-BanglaTutor-CoT/SSC_Physics_Dataset_with_cot.jsonl",
}

EXAMPLES = [
    "সালোকসংশ্লেষণ প্রক্রিয়ায় উদ্ভিদ কোন গ্যাস গ্রহণ করে?",
    "বিশুদ্ধ পানির pH মান কত?",
    "নিউটনের দ্বিতীয় সূত্রটি কী?",
    "কোন খাদ্য উপাদানটি নতুন কোষ তৈরিতে সাহায্য করে?",
]


# --- backends --------------------------------------------------------------

def make_api_responder(args):
    """Answer through llm_client (local server or Groq)."""
    client = client_from_args(args, temperature=0.3, max_tokens=2048)

    def respond(question, history):
        user = (f"প্রশ্ন: {question}\n"
                f"উপরের নিয়ম মেনে বাংলা reasoning chain তৈরি করো।")
        try:
            data, _ = client.complete(SYSTEM, user, COT_SCHEMA)
        except LLMError as exc:
            return f"⚠️ {exc}"
        except Exception as exc:
            return f"⚠️ {type(exc).__name__}: {exc}"
        return to_cot(data)

    return respond, f"{client.provider} · {client.model}"


def make_agent_responder(args):
    """Retrieve -> reason -> verify -> revise, with the loop shown to the user."""
    client = client_from_args(args, temperature=0.3, max_tokens=2048)
    agent = TutorAgent(client, max_attempts=args.max_attempts)

    def respond(question, history):
        answer, trace = agent.answer(question, history)
        # the trace is what makes the loop legible rather than a black box
        parts = [
            answer,
            "<details><summary>🤖 এজেন্টের ধাপ / agent trace</summary>",
            format_trace(trace),
            "</details>",
        ]
        return "\n\n".join(parts)

    return respond, f"agent · {client.provider} · {client.model}"


def make_adapter_responder(args):
    """Answer with the fine-tuned adapter — the real deliverable."""
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(args.adapter, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    def respond(question, history):
        msgs = [{"role": "system", "content": TUTOR_SYS},
                {"role": "user", "content": question}]
        ids = tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    return respond, f"fine-tuned adapter · {args.adapter}"


# --- corpus comparison tab -------------------------------------------------

def load_corpus(limit_per_subject=400):
    """Upstream rows keyed by question, plus any chains we have generated."""
    upstream = {}
    for subject, path in UPSTREAM.items():
        p = pathlib.Path(path)
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if i >= limit_per_subject:
                break
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = e.get("Question") or e.get("question")
            if q:
                e["_subject"] = subject
                upstream[nrm(q)] = e

    ours = {}
    cot_dir = pathlib.Path("data/cot")
    if cot_dir.exists():
        for f in sorted(cot_dir.glob("*.jsonl")):
            if f.name == "quarantine.jsonl":
                continue
            for r in read_jsonl(f):
                if r.get("CoT"):
                    ours[nrm(r["Question"])] = r
    return upstream, ours


def _steps_of(cot):
    return [l for l in str(cot).splitlines()
            if l.strip() and not l.startswith("তাই সঠিক উত্তর")]


def render_chain(cot, hints, golds, label):
    """Chain as HTML, marking copied-hint steps and early answer mentions."""
    steps = _steps_of(cot)
    hintset = {nrm(h) for h in (hints or []) if str(h).strip()}
    goldlist = [nrm(g) for g in golds if str(g).strip()]
    rows, copied, leaked = [], 0, 0
    for i, s in enumerate(steps):
        tags = []
        if strip_step_prefix(s) in hintset:
            tags.append("copied hint")
            copied += 1
        if i < len(steps) - 1 and any(g and g in nrm(s) for g in goldlist):
            tags.append("answer stated early")
            leaked += 1
        badge = ("<span style='color:#b91c1c;font-size:.8em;white-space:nowrap'> ← "
                 + ", ".join(tags) + "</span>") if tags else ""
        # Set background AND text colour together. Inheriting the text colour
        # under a hardcoded light background renders white-on-pink in dark mode.
        style = ("background:#fee2e2;color:#7f1d1d" if tags
                 else "background:rgba(127,127,127,.12);color:inherit")
        rows.append(f"<div style='{style};padding:5px 8px;border-radius:4px;"
                    f"margin:3px 0'>{html.escape(s)}{badge}</div>")
    summary = (f"<b>{label}</b> — {copied}/{len(steps)} steps are copied hints"
               f"{', answer stated early' if leaked else ''}")
    return (f"<div style='font-family:system-ui;line-height:1.7;color:inherit'>"
            f"{summary}<div style='margin-top:8px'>{''.join(rows)}</div></div>")


# --- app -------------------------------------------------------------------

def build_app(respond, backend_label, args):
    import gradio as gr

    upstream, ours = load_corpus()
    # put questions we have generated a chain for first: the whole point of the
    # tab is the side-by-side, so landing on an empty right pane wastes it
    both = sorted(set(upstream) & set(ours))
    questions = both + sorted(set(upstream) - set(ours))
    preview = "fine-tuned adapter" in backend_label

    with gr.Blocks(title="ShikkhokBot — বাংলা বিজ্ঞান শিক্ষক") as app:
        gr.Markdown(f"# ShikkhokBot — বাংলা বিজ্ঞান শিক্ষক\n"
                    f"SSC জীববিজ্ঞান, রসায়ন ও পদার্থবিজ্ঞান — ধাপে ধাপে বাংলা ব্যাখ্যা।")
        if not preview:
            gr.Markdown(
                f"> ⚠️ **Preview backend — not the fine-tuned model.** Answers come "
                f"from `{backend_label}`, the same setup that generates the training "
                f"data. It shows the target behaviour; the shipped 7B tutor is what "
                f"`03_train.py` produces.")
        else:
            gr.Markdown(f"Backend: `{backend_label}`")

        if backend_label.startswith("agent"):
            gr.Markdown(
                "**Agent mode.** Each question runs a loop: it searches your "
                "10,903-question corpus for related items, drafts a chain using "
                "them plus the conversation so far, runs the project's own "
                "validators over its own draft, and revises if one fires. Expand "
                "*agent trace* under any answer to see the steps it took.")

        with gr.Tab("Ask the tutor"):
            # Gradio 6 dropped the `type` kwarg; messages format is the default.
            gr.ChatInterface(respond, examples=EXAMPLES)

        with gr.Tab("Compare with the original dataset"):
            gr.Markdown(
                "The corpus ships a `CoT` field that is the **hint list reformatted** "
                "— 70.2% of its step lines are hints copied verbatim. Pick a question "
                "to see it, with the copied steps highlighted. Ours appears alongside "
                f"— {len(both)} of these questions have a generated chain so far.")
            with gr.Row():
                pick = gr.Dropdown(questions, label="Question",
                                   value=questions[0] if questions else None,
                                   filterable=True, scale=4)
                shuffle = gr.Button("🎲 Random (with our chain)"
                                    if both else "🎲 Random", scale=1)
            gold_box = gr.Markdown()
            with gr.Row():
                left = gr.HTML(label="upstream")
                right = gr.HTML(label="ours")

            def show(q):
                e = upstream.get(nrm(q or ""))
                if not e:
                    return "", "", ""
                golds = e.get("ExactAnswer")
                golds = golds if isinstance(golds, list) else [golds]
                hints = e.get("Hints") or e.get("hints") or []
                gold_md = (f"**বিষয়:** {e.get('_subject','?')}  \n"
                           f"**সঠিক উত্তর:** {', '.join(map(str, golds))}")
                up = render_chain(e.get("CoT", ""), hints, golds,
                                  "Original dataset")
                mine = ours.get(nrm(q or ""))
                if mine:
                    mine_html = render_chain(mine["CoT"], hints, golds,
                                             f"Ours · {mine.get('_cot_model','?')}")
                else:
                    mine_html = ("<div style='font-family:system-ui;color:#666'>"
                                 "No generated chain for this question yet — it is "
                                 "not in the pilot sample. Run "
                                 "<code>02b_bulk_cot.py</code> to fill the pool."
                                 "</div>")
                return gold_md, up, mine_html

            pick.change(show, pick, [gold_box, left, right])
            pool = both or questions
            shuffle.click(lambda: random.choice(pool) if pool else None, None, pick)
            if questions:
                app.load(show, pick, [gold_box, left, right])

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="agent",
                    choices=["agent", "chat", "adapter"],
                    help="agent = retrieve/verify/revise loop (default); "
                         "chat = plain one-shot; adapter = fine-tuned model")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="agent: how many times to revise a failing draft")
    ap.add_argument("--adapter", default="outputs/adapter")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    add_provider_args(ap)
    args = ap.parse_args()

    load_env()
    if args.backend == "adapter":
        if not pathlib.Path(args.adapter).exists():
            sys.exit(f"{args.adapter} does not exist — run scripts/03_train.py first, "
                     f"or use --backend agent to preview against a served model.")
        respond, label = make_adapter_responder(args)
    else:
        try:
            respond, label = (make_agent_responder(args) if args.backend == "agent"
                              else make_api_responder(args))
        except LLMError as exc:
            sys.exit(str(exc))

    build_app(respond, label, args).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
