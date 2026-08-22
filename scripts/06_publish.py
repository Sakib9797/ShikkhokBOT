# scripts/06_publish.py
"""Phase 5 — push the augmented dataset and the LoRA adapter to the Hub.

    python scripts/06_publish.py dataset --repo <user>/ShikkhokBot-SSC-Bangla-CoT
    python scripts/06_publish.py adapter --repo <user>/ShikkhokBot-Llama-3.2-3B-LoRA

Needs `HF_TOKEN` in `.env`. The dataset push carries `DATASET_CARD.md` as the
repo README, so the card ships with the data rather than trailing behind it.
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cot_core import load_env  # noqa: E402

COT = pathlib.Path("data/cot/all_cot.jsonl")
CARD = pathlib.Path("DATASET_CARD.md")
ADAPTER = pathlib.Path("outputs/adapter")


def api():
    load_env()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is empty — put it in .env before publishing.")
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def push_dataset(args):
    if not COT.exists():
        sys.exit(f"{COT} missing — run scripts/02b_batch_cot.py collect first.")
    a = api()
    a.create_repo(args.repo, repo_type="dataset", exist_ok=True, private=args.private)
    a.upload_file(path_or_fileobj=str(COT), path_in_repo="all_cot.jsonl",
                  repo_id=args.repo, repo_type="dataset")
    for extra, dest in (("data/clean/test100.jsonl", "test100.jsonl"),
                        ("outputs/reports/cot_baseline_upstream.md",
                         "reports/cot_baseline_upstream.md"),
                        ("outputs/reports/cot_baseline_ours.md",
                         "reports/cot_baseline_ours.md"),
                        ("outputs/reports/integrity.md", "reports/integrity.md")):
        if pathlib.Path(extra).exists():
            a.upload_file(path_or_fileobj=extra, path_in_repo=dest,
                          repo_id=args.repo, repo_type="dataset")
    if CARD.exists():
        a.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                      repo_id=args.repo, repo_type="dataset")
    print(f"pushed dataset -> https://huggingface.co/datasets/{args.repo}")


def push_adapter(args):
    if not ADAPTER.exists():
        sys.exit(f"{ADAPTER} missing — run scripts/03_train.py on Colab first.")
    a = api()
    a.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    a.upload_folder(folder_path=str(ADAPTER), repo_id=args.repo, repo_type="model",
                    ignore_patterns=["checkpoint-*", "*.pt", "runs/*"])
    print(f"pushed adapter -> https://huggingface.co/{args.repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["dataset", "adapter"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    (push_dataset if args.what == "dataset" else push_adapter)(args)


if __name__ == "__main__":
    main()
