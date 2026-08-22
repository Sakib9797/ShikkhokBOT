# scripts/05_demo.py
"""Phase 5 — Gradio chat demo, deployable free on HF Spaces.

    python scripts/05_demo.py                       # local
    python scripts/05_demo.py --share               # public link

For Spaces: copy this file as `app.py`, add a `requirements.txt` with
`unsloth`, `transformers`, `peft`, `bitsandbytes`, `gradio`, and point
`--adapter` at the pushed HF adapter repo id.
"""
import argparse

SYS = "তুমি একজন বাংলা মাধ্যমিক বিজ্ঞান শিক্ষক। ধাপে ধাপে ব্যাখ্যা করে উত্তর দাও।"

EXAMPLES = [
    "সালোকসংশ্লেষণ প্রক্রিয়ায় উদ্ভিদ কোন গ্যাস গ্রহণ করে?",
    "বিশুদ্ধ পানির pH মান কত?",
    "নিউটনের দ্বিতীয় সূত্রটি কী?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="outputs/adapter")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    import gradio as gr
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(args.adapter, load_in_4bit=True)
    FastLanguageModel.for_inference(model)

    def chat(q, history):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
        ids = tok.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    gr.ChatInterface(
        chat,
        title="ShikkhokBot — বাংলা বিজ্ঞান শিক্ষক",
        description="SSC জীববিজ্ঞান, রসায়ন ও পদার্থবিজ্ঞানের প্রশ্ন করুন — "
                    "ধাপে ধাপে বাংলা ব্যাখ্যা পাবেন।",
        examples=EXAMPLES,
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
