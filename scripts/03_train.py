# scripts/03_train.py
"""Phase 3 — QLoRA fine-tune of Llama-3.2-3B-Instruct on the Bengali CoT set.

Runs on a Colab T4 (free tier), NOT on the local Windows box. Upload
`data/cot/all_cot.jsonl` to the Colab session (or pull it from the HF dataset)
and run:

    !pip install -q unsloth trl transformers peft bitsandbytes accelerate datasets
    !python 03_train.py --data all_cot.jsonl

The length gate before `.train()` is not optional. Llama-3.2's BPE vocab is
English-dominant and spends 4-8 tokens per Bengali word, so a 400-character
chain can exceed 1,000 tokens. Silent truncation would teach the model to stop
mid-reasoning, and you would not find out until three hours later.
"""
import argparse
import json


SYS = "তুমি একজন বাংলা মাধ্যমিক বিজ্ঞান শিক্ষক। ধাপে ধাপে ব্যাখ্যা করে উত্তর দাও।"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cot/all_cot.jsonl")
    ap.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", default="outputs/adapter")
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--keep-rejected", action="store_true",
                    help="train on validator-rejected chains too (default: skip)")
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    model, tok = FastLanguageModel.from_pretrained(
        args.base, max_seq_length=args.maxlen, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    ds = load_dataset("json", data_files=args.data)["train"]
    before = len(ds)
    if not args.keep_rejected:
        ds = ds.filter(lambda e: not e.get("_reject"))
    print(f"training rows: {len(ds)} (of {before}; "
          f"{before - len(ds)} validator-rejected chains dropped)")

    def to_text(ex):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": ex["Question"]},
                {"role": "assistant", "content": ex["CoT"]}]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])

    # GATE: confirm nothing is being silently truncated
    sample = ds["text"][:2000]
    lens = sorted(len(tok(t).input_ids) for t in sample)
    p50, p99, mx = lens[len(lens) // 2], lens[int(len(lens) * .99)], lens[-1]
    print(f"token lengths — p50 {p50}  p99 {p99}  max {mx}  (maxlen {args.maxlen})")
    assert p99 < args.maxlen, (
        f"p99={p99} >= maxlen={args.maxlen}: raise --maxlen (costs T4 VRAM/time) "
        f"or shorten the chains to 5 steps before training")

    SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
        max_seq_length=args.maxlen,
        args=TrainingArguments(
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            num_train_epochs=args.epochs, learning_rate=2e-4, warmup_steps=10,
            fp16=True, logging_steps=10, output_dir=args.out, optim="adamw_8bit",
            save_strategy="epoch", report_to="none"),
    ).train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(f"{args.out}/train_meta.json", "w", encoding="utf-8") as f:
        json.dump({"base": args.base, "rows": len(ds), "epochs": args.epochs,
                   "maxlen": args.maxlen, "p50": p50, "p99": p99, "max": mx}, f, indent=2)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
