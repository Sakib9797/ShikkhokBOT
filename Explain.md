# ShikkhokBot — Explained Simply

This document explains the whole project from beginning to end, in plain words,
for someone who has never seen the code. No prior AI knowledge is assumed.
There is a glossary at the bottom for any term that looks technical.

---

## 1. What are we building?

**A Bangla science teacher that lives in a chat box.**

An SSC student types a question in Bangla. The bot answers — but it does not
just give the answer. It *explains how to get there*, step by step, in Bangla.

Ask it:

> সালোকসংশ্লেষণ প্রক্রিয়ায় উদ্ভিদ কোন গ্যাস গ্রহণ করে?

A plain answer key would say **কার্বন ডাই-অক্সাইড** and stop. Our bot should say:

> ধাপ ১: সালোকসংশ্লেষণ হলো সেই প্রক্রিয়া যেখানে সবুজ উদ্ভিদ সূর্যালোকের শক্তি ব্যবহার করে খাদ্য তৈরি করে।
> ধাপ ২: এই প্রক্রিয়ায় উদ্ভিদকে কার্বন কাঠামো তৈরির জন্য একটি কার্বনযুক্ত কাঁচামাল বায়ু থেকে নিতে হয়।
> ধাপ ৩: বায়ুমণ্ডলে উপস্থিত সেই গ্যাসটি পত্ররন্ধ্র দিয়ে পাতায় প্রবেশ করে।
> তাই সঠিক উত্তর: কার্বন ডাই-অক্সাইড

That difference — **answer** versus **explanation** — is the entire project.
A student who reads only the answer memorises it. A student who reads the steps
can solve the *next* question too.

**Who is it for:** SSC students in Bangladesh studying Biology, Chemistry and
Physics, who mostly do not have a private tutor available at 11pm.

---

## 2. What do we start with?

Someone else already collected the questions. A public dataset called
**SSC-BanglaTutor** contains **11,286 SSC science questions** in Bangla.

Each question in the file looks roughly like this:

| Field | Meaning | Example |
|---|---|---|
| `Question` | the question | বিজ্ঞানের সবচেয়ে প্রাচীন শাখা কোনটি? |
| `Hints` | 5–6 clues, like a quiz game | "প্রাচীন গ্রিক দার্শনিকরা এর ভিত্তি স্থাপন করেন।" |
| `ExactAnswer` | the correct answer | পদার্থবিজ্ঞান |

Split across three subjects:

| Subject | Questions |
|---|---:|
| Biology | 4,859 |
| Chemistry | 3,034 |
| Physics | 3,393 |
| **Total** | **11,286** |

**We did not create this data.** It is other researchers' work, and we credit
them. What we create is the *reasoning* that this data is missing.

---

## 3. The problem we found (this is the heart of the project)

The dataset already claims to have step-by-step reasoning. There is a field
called `CoT` — "Chain of Thought", meaning the thinking steps.

So at first glance, the job looks done. It is not.

**The "reasoning" is fake.** It is just the hint list with the words
"ধাপ ১:", "ধাপ ২:" glued onto the front.

Here is a real row from the Physics file, unedited:

**Question:** বিজ্ঞানের সবচেয়ে প্রাচীন শাখা কোনটি?

**Its hints:**
1. এটি প্রকৃতি, বস্তুজগৎ ও কর্মক্ষমতার ধারণা নিয়ে অনুসন্ধান করে।
2. প্রাচীন গ্রিক দার্শনিকরা এর ভিত্তি স্থাপন করেন।
3. একে বিজ্ঞানের মৌলিক শাখাও বলা হয়।
4. এর ইংরেজি প্রতিশব্দটির প্রথম অক্ষর 'P'।
5. জ্যোতির্বিদ্যা ছিল এর প্রাথমিক গবেষণার অন্যতম বিষয়।

**Its "reasoning":**
> ধাপ 1: এটি প্রকৃতি, বস্তুজগৎ ও কর্মক্ষমতার ধারণা নিয়ে অনুসন্ধান করে।
> ধাপ 2: প্রাচীন গ্রিক দার্শনিকরা এর ভিত্তি স্থাপন করেন।
> ধাপ 3: একে বিজ্ঞানের মৌলিক শাখাও বলা হয়।
> ধাপ 4: এর ইংরেজি প্রতিশব্দটির প্রথম অক্ষর 'P'।
> ধাপ 5: জ্যোতির্বিদ্যা ছিল এর প্রাথমিক গবেষণার অন্যতম বিষয়।
> ধাপ 6: তাই সঠিক উত্তর হলো "পদার্থবিজ্ঞান"

Look at step 4: *"the first letter of its English word is 'P'."*

**No teacher reasons like that.** That is a hangman clue. It tells you nothing
about physics. Steps 1–5 are the five hints, copied word for word, in order.

### We measured exactly how bad it is

We did not want to say "it looks bad" in a research paper. So we wrote a script
that checks every single row (`scripts/02c_baseline_metrics.py`) and counts two
specific failures:

**Failure 1 — copying hints.** How many "reasoning steps" are a hint copied
word-for-word?

**Failure 2 — leaking the answer.** How many rows say the answer in the middle,
before the reasoning has finished? (Like a maths solution that writes
"the answer is 7" in line 2, then keeps working.)

The result, across all 11,258 rows that have a chain:

| Subject | Steps that are copied hints | Rows that leak the answer early |
|---|---:|---:|
| Biology | 67.1% | 8.9% |
| Chemistry | **78.5%** | **11.3%** |
| Physics | 67.4% | 8.4% |
| **All** | **70.2%** | **9.4%** |

**Seven out of every ten "reasoning steps" are copied hints.**

This is good news for us, oddly. It means there is a real, measurable hole to
fill — and "we filled this hole, here are the before-and-after numbers" is
exactly what makes a research paper worth publishing.

---

## 4. The plan, in one sentence

> Throw away the fake reasoning, keep the questions, and generate **real**
> reasoning using a modern AI model — then train a small model to imitate it.

Why train a small model at all, instead of just using the big AI? Because the
big model is huge and needs an expensive machine. Once the small model has
*learned* from the big model's explanations, it can run cheaply and be given
away free. This is called **distillation** — pouring the knowledge of a large
model into a small one.

---

## 5. The steps, one at a time

### Phase 0 — Fix the broken file ✅ done

Data files are never clean. When we tried to read all 11,286 questions, **29 of
them crashed the reader.** They had typing mistakes in the file format — a
stray bracket, a quotation mark in the wrong place.

Most people would just skip those 29 rows. We repaired them instead, and wrote
down every single repair in a log file.

**Why bother for 29 rows out of 11,286?** Two reasons. First, the original
paper claims 11,286 questions — if we silently dropped 29, our numbers would
never match theirs and nobody could check our work. Second, we noticed the
people who built the fake CoT *had* silently dropped them. Being the ones who
did it properly is part of the point.

**Result: 11,286 out of 11,286 recovered. Nothing lost, nothing hidden.**

### Phase 1 — Clean up and set aside a test ✅ done

Two jobs here.

**Remove duplicates.** The same question sometimes appears in two subjects.
We found **383 duplicates** and removed them, leaving **10,903 unique questions**.

**Set aside a test set.** We locked away **100 questions** in a separate file.
The bot will never be trained on these, and they are never shown to the AI that
generates the reasoning.

**Why does this matter so much?** Imagine a teacher gives students the exam
paper to study the night before. Everyone scores 100%. Did they learn anything?
You cannot tell. Same problem here: if we test the bot on questions it was
trained on, it just recites, and our results are meaningless. The 100 locked
questions are our honest exam.

That leaves **10,803 questions** to work with.

### Phase 2 — Generate real reasoning ⏳ next

This is the main event.

We take each of the 10,803 questions and ask a powerful open-source AI model
(such as **Qwen**) to write genuine step-by-step reasoning in Bangla.

You can run that AI in either of two places, and the project supports both with
a single switch:

**On your own PC with the RTX 5090.** The graphics card acts as a small private
AI server. Nothing is sent anywhere, nothing costs money per question, and the
data never leaves your machines. Slower, but free and completely private.

**On Groq**, a company that runs open-source models on their own hardware and
lets you send questions over the internet. Much faster, and you need no GPU at
all — but it is a metered service with usage limits, and your questions travel
to their servers. Since these are public textbook questions with no personal
data in them, that is a fair trade if you want speed.

Both options run the exact same instructions and the exact same quality checks,
so explanations made one way are directly comparable to the other. Each row
records which model wrote it, so you can always tell them apart later.

**Three design decisions that matter:**

**We hide the hints from the AI.** This is deliberate and it is the single most
important choice in the project. If we showed the AI those hints, it would do
exactly what the original authors' tool did — rearrange them and call it
reasoning. So the AI never sees them. It must reason from actual science.

**We show the AI the correct answer privately.** This sounds like cheating, but
it is not. Without it the AI might reason beautifully toward the *wrong*
conclusion, and we would have 10,803 confident, well-written mistakes. Giving it
the destination ensures the path leads somewhere correct. The answer is never
shown to the student, and the AI is forbidden from mentioning it until the last
line.

**Every chain is automatically checked.** A computer program inspects each
generated explanation and rejects it if:

| Check | What it catches |
|---|---|
| `answer_leak` | The answer appears in the middle instead of at the end |
| `hint_copy` | Too many steps are copied hints — the exact sin we are fixing |
| `not_bengali` | The AI drifted into English |
| `wrong_answer` | The chain ends on the wrong conclusion |
| `step_count` | Fewer than 3 or more than 6 steps |

Rejected explanations are automatically retried with a sterner instruction. If
they fail again, they are set aside and *counted in public*, not quietly binned.

**Which AI model?** We do not guess. Bangla ability varies a lot between open
models, so we generate 100 sample explanations from each candidate (Qwen,
Aya, Gemma), then compare their scores on the checks above
(`scripts/02d_model_bakeoff.py`) and read the samples by hand. The winner
generates the full set.

### Phase 3 — Teach the small model ⏳

Now we have ~10,800 questions each paired with a real Bangla explanation.

We take a small open model and show it these thousands of examples until it
learns the pattern: *"when a Bangla science question arrives, respond with
step-by-step Bangla reasoning, then the answer."*

This is **fine-tuning**. An analogy: the base model already speaks and knows
general facts, like a bright graduate. Fine-tuning is the two-week induction
course that teaches it *this specific job*.

We use a shortcut called **QLoRA**, which adjusts a small percentage of the
model instead of rebuilding it. It is the difference between adjusting a suit
and weaving new cloth — vastly cheaper, and the result fits just as well. This
runs on the 5090 in a few hours.

**One trap we guard against.** Bangla is expensive for these models to read.
They chop text into pieces called tokens, and their vocabularies were built
mostly from English, so a single Bangla word can cost 4–8 tokens where an
English word costs 1. If an explanation is too long, the model quietly chops
the end off — and then it learns to *stop halfway through reasoning*, which is
the worst possible lesson. So the script measures the lengths and refuses to
start if they overflow. Better to fail in 10 seconds than to discover it after
four hours.

### Phase 4 — Test it honestly ⏳

Remember the 100 locked-away questions. Now we unlock them.

We ask the same 100 questions to two models:
- **A:** the original small model, untrained
- **B:** our fine-tuned version

Then we compare, three ways:

**Did it get the right answer?** (exact match — a simple percentage)

**Does its wording resemble a good answer?** (ROUGE-L, a standard text-overlap
score)

**Do humans think it is any good?** 5–10 real SSC students and teachers rate
the answers without being told which model wrote which. This is the one that
actually matters — a bot can score well automatically and still explain badly.

We also re-run the Phase 3 measurement on *our* explanations, so the paper can
show the before-and-after side by side:

| | Copied hints | Answer leaked early |
|---|---:|---:|
| Original dataset | 70.2% | 9.4% |
| **Ours** | *(to be filled in)* | *(to be filled in)* |

That table is the paper's punchline.

### Phase 5 — Give it away ⏳

Four things get published, all free:

1. **The chat demo** — a web page anyone can use
2. **The dataset** — ~10,800 Bangla questions with real reasoning, the first of
   its kind for this curriculum
3. **The trained model** — so others can build on it
4. **A short research paper** — describing the problem we found and how we fixed it

---

## 6. How the pieces fit together

```
  Original dataset (11,286 questions, with fake reasoning)
            |
   [Phase 0]  repair 29 broken rows            -> 11,286 recovered
            |
   [Phase 1]  remove 383 duplicates            -> 10,903 unique
            |  lock away 100 for the exam      -> 10,803 to work with
            |
   [Phase 2]  local AI on the 5090 writes      -> ~10,800 real explanations
            |  real Bangla reasoning
            |  (hints hidden, every chain checked)
            |
   [Phase 3]  small model learns from them     -> the trained tutor
            |
   [Phase 4]  test on the 100 locked questions -> honest score
            |
   [Phase 5]  publish demo + data + model + paper
```

---

## 7. Where things stand today

| Phase | Status |
|---|---|
| 0 — Repair the data | ✅ **Done.** 11,286/11,286 recovered |
| 1 — Clean and split | ✅ **Done.** 10,903 unique · 100 locked · 10,803 ready |
| Measuring the problem | ✅ **Done.** 70.2% copied hints, 9.4% leaks |
| 2 — Generate reasoning | ⏳ Code ready. Needs the 5090 serving a model |
| 3 — Train the model | ⏳ Code ready. Needs the 5090 |
| 4 — Evaluate | ⏳ Code ready. Scoring already tested |
| 5 — Publish | ⏳ Code ready. Needs the trained model |

**All the code is written.** What remains is running it on the GPU machine.

---

## 8. Being honest about the limits

A good project states its weaknesses before a reviewer finds them.

**The explanations are written by an AI, not a teacher.** Our automatic checks
confirm a chain does not leak the answer, does not copy hints, is in Bangla,
and ends correctly. They *cannot* confirm the reasoning is pedagogically good.
An explanation can pass every check and still be shallow. That is precisely why
Phase 4 includes real human raters — to put a number on that gap instead of
hiding it.

**The questions are short.** Typical question: 42 characters. Typical answer:
12 characters. These are short-answer recall questions. This is a tutor for
"explain why this fact is true", not for multi-step numerical problems.

**Bangla is not these models' strongest language.** Almost all open models are
trained mostly on English and Chinese. Bangla output can be stiff or subtly
wrong. This is a real limitation and it gets stated plainly in the paper, not
buried.

**We depend on someone else's dataset.** If their questions contain errors, our
explanations inherit them. We credit the original authors and follow their
licence.

---

## 9. Glossary

| Term | In plain words |
|---|---|
| **SSC** | Secondary School Certificate — the Class 10 public exam in Bangladesh |
| **Chain of Thought (CoT)** | The visible step-by-step thinking, not just the final answer |
| **Fine-tuning** | Taking a general AI model and training it further on your specific task |
| **QLoRA** | A cheap fine-tuning shortcut that adjusts a small part of the model |
| **Distillation** | Training a small model to imitate a large one |
| **Hold-out / test set** | Questions deliberately hidden during training, used as an honest exam |
| **Token** | The chunks a model reads text in — roughly a word or part of a word |
| **Tokenizer** | The tool that chops text into tokens. Handles English efficiently, Bangla poorly |
| **Exact match** | Did the correct answer appear in the response? A simple percentage |
| **ROUGE-L** | A standard score for how much the response overlaps a good answer |
| **JSONL** | A text file with one record per line. All our data files are this format |
| **Deduplication** | Removing repeated entries |
| **vLLM / Ollama** | Programs that run an AI model on your own GPU and let other programs talk to it |
| **Groq** | A company that runs open-source AI models on their own hardware and lets you use them over the internet |
| **Rate limit** | A cap on how many requests a service accepts per minute. Local servers have none; Groq does |
| **Validator** | Our automatic checker that accepts or rejects each generated explanation |
| **Quarantine** | Where rejected items go — kept and counted, never silently deleted |

---

## 10. If you only remember three things

1. **The dataset's existing "reasoning" is fake** — 70% of its steps are copied
   hints, and we proved it with a number instead of an opinion.
2. **We generate real Bangla reasoning by hiding the hints**, then machine-check
   every single chain against the exact failures we are trying to fix.
3. **You choose where the AI runs** — free on your own 5090, or fast on Groq —
   and everything produced is given away free: the data, the model, the demo.
