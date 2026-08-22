# 🎓 ShikkhokBot
## Fine-Tuned LLM as a Bangla Secondary School Tutor (SSC/HSC)

> A complete project guide for your CV | LLM Fine-Tuning

---

## 🔬 Why This Project is Unique

The **SSC-BanglaTutor dataset was published in February 2026** — and no fine-tuned open-source model on top of it exists yet. You would be the first to build one.

LLMs need curriculum-specific information within the national language to provide relevant support. Lack of language-specific knowledge has been shown to be a key limitation — and many communities remain unsupported as AI reshapes the global landscape.

Bangladesh has **~2 million SSC examinees every year**. A working Bangla tutor chatbot has immediately demonstrable real-world impact, making it compelling on a CV and attractive to reviewers for publication.

> **Your unique technical contribution:** Adding Chain-of-Thought (CoT) reasoning chains in Bengali to the existing Q&A pairs — creating a reasoning-augmented version of the dataset that no one else has built.

---

## 📦 Dataset Sources (100% HuggingFace — No Scraping)

| Dataset | Description | Link |
|---|---|---|
| **SSC-BanglaTutor** | 11,286 Q&A pairs — SSC Biology, Chemistry, Physics (Feb 2026) | [sciencedirect.com/science/article/pii/S2352340926001502](https://www.sciencedirect.com/science/article/pii/S2352340926001502) |
| **Bangla-TextBook** | NCTB-aligned instruction pairs from 50 university volunteers | [huggingface.co/datasets/md-nishat-008/Bangla-TextBook](https://huggingface.co/datasets/md-nishat-008/Bangla-TextBook) |
| **Bangla-Instruct** | 500 diverse seed tasks across multiple domains in Bengali | [huggingface.co/datasets/md-nishat-008/Bangla-Instruct](https://huggingface.co/datasets/md-nishat-008/Bangla-Instruct) |
| **Bangla SFT Collection** | All Bengali datasets useful for LLM fine-tuning | [huggingface.co/collections/Mahadih534/bangla-datasets-for-llms-finetuning](https://huggingface.co/collections/Mahadih534/bangla-datasets-for-llms-finetuning) |
| **Bangla NLP Master List** | Validated link list of all Bangla NLP datasets (2025–26) | [github.com/Foysal87/Bangla-NLP-Dataset](https://github.com/Foysal87/Bangla-NLP-Dataset) |

---

## 🛠️ Step-by-Step Execution Plan (7 Weeks)

### 📅 Week 1 — Load & Explore All Datasets

- Download SSC-BanglaTutor, Bangla-TextBook, and Bangla-Instruct using the HuggingFace `datasets` library — no scraping required.

```python
from datasets import load_dataset
ds = load_dataset("md-nishat-008/Bangla-TextBook")
```

- Filter for Science subjects (Physics, Chemistry, Biology). Target **~5,000–7,000 training examples** total.

---

### 📅 Week 2 — Build Chain-of-Thought Instruction Pairs

- The SSC-BanglaTutor dataset has Q&A pairs but **no step-by-step reasoning**. Use GPT-4o to add Bengali CoT reasoning chains to each answer.

**Example format:**
```
Input:  "পানির রাসায়নিক সংকেত কী?" (What is the chemical formula of water?)

Output: "প্রথমে... হাইড্রোজেন ২টি এবং অক্সিজেন ১টি পরমাণু নিয়ে...
         তাই সংকেত H₂O।"
```

- Publish this augmented dataset on HuggingFace — that is a **second CV item** before you even train anything.

---

### 📅 Weeks 3–4 — Fine-Tune with Unsloth + QLoRA

- Use **Unsloth** on Google Colab free tier → [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- Base model: `meta-llama/Llama-3.2-3B-Instruct` — fits in free Colab with 4-bit QLoRA.
- Open the official notebook and start training:
  [colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp)
- Train for **2–3 epochs** (~3–4 hours on the free T4 GPU).

---

### 📅 Week 5 — Evaluate

- Create a held-out test set of **100 SSC-style questions** not in your training data.
- Ask 5–10 actual SSC/HSC students or teachers to compare your model vs. zero-shot Llama 3.2 on the same questions — rate on **correctness + clarity**.
- Report **ROUGE-L** and **exact-match scores** for short-answer questions.

---

### 📅 Weeks 6–7 — Demo + Publish

- Build a Gradio chatbot in ~15 lines of Python → [gradio.app](https://www.gradio.app)
- Deploy for **free** on HuggingFace Spaces → [huggingface.co/spaces](https://huggingface.co/spaces)
- Push the model adapter and augmented dataset to HuggingFace.
- Submit a 4-page paper to **arXiv** → [arxiv.org/submit](https://arxiv.org/submit)
- Optionally submit to the **BLP Workshop** (Bangla Language Processing, co-located with ACL/EMNLP) → [blp-workshop.github.io](https://blp-workshop.github.io)

---

## 💻 All Tools & Links

| Tool / Resource | Link |
|---|---|
| SSC-BanglaTutor paper (dataset) | [sciencedirect.com/...](https://www.sciencedirect.com/science/article/pii/S2352340926001502) |
| Bangla-TextBook (HuggingFace) | [huggingface.co/datasets/md-nishat-008/Bangla-TextBook](https://huggingface.co/datasets/md-nishat-008/Bangla-TextBook) |
| Bangla-Instruct (HuggingFace) | [huggingface.co/datasets/md-nishat-008/Bangla-Instruct](https://huggingface.co/datasets/md-nishat-008/Bangla-Instruct) |
| Bangla SFT collection | [huggingface.co/collections/Mahadih534/...](https://huggingface.co/collections/Mahadih534/bangla-datasets-for-llms-finetuning) |
| Llama 3.2 3B Instruct (base model) | [huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Unsloth — fast QLoRA training | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) |
| Unsloth Llama 3.2 Colab notebook | [colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp) |
| Gradio — demo UI | [gradio.app](https://www.gradio.app) |
| HuggingFace Spaces — free hosting | [huggingface.co/spaces](https://huggingface.co/spaces) |
| arXiv submission | [arxiv.org/submit](https://arxiv.org/submit) |
| Bangla NLP master dataset list | [github.com/Foysal87/Bangla-NLP-Dataset](https://github.com/Foysal87/Bangla-NLP-Dataset) |
| BLP Workshop (Bangla Language Processing) | [blp-workshop.github.io](https://blp-workshop.github.io) |

---

## 🏆 Why This Crushes On Your CV

| Advantage | Detail |
|---|---|
| **Four CV Items From One Project** | Augmented dataset + fine-tuned model + Gradio demo + arXiv paper — each listed separately |
| **Published Feb 2026 — No Model Yet** | SSC-BanglaTutor is so new that you have a clear first-mover window right now |
| **Zero Scraping Required** | Every dataset is a clean `load_dataset()` call — reproducible, ethical, and easy to document |
| **Massive Local Audience** | ~2 million SSC examinees per year in Bangladesh; a working demo is immediately demonstrable impact |
| **CoT Augmentation is Your Research Novelty** | Even if someone else fine-tunes on the same dataset, adding Bengali CoT reasoning chains is a distinct, publishable contribution |
| **Free to Execute** | Google Colab free tier + HuggingFace free hosting + free arXiv = zero cost, no GPU purchase needed |

---

*Generated by Claude · ShikkhokBot LLM Fine-Tuning Project Guide · August 2026*
