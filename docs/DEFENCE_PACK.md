# Defence pack — presentation and Q&A

For the BMDS2123 presentation on **Saturday 5 September 2026**.

This document exists because of one line in the rubric:

> **Aspect 11 (10 marks)** — *"Student demonstrates strong understanding of the
> project, NLP concepts, and implementation details. Able to explain code,
> system workflow, and design decisions confidently and accurately during the
> presentation and Q&A session."*

Together with the 10% individual presentation mark, **20 marks depend on you
explaining this system, not on the system itself**. The code is finished. This
is the part that still needs work.

**How to use this:** read §1–3 until you can say them without looking. Skim §4
and §5. Practise §6 twice on the actual machine. §7 is what you say when
something goes wrong.

---

## 1. The 90-second answer

When they say *"tell us what you built"*, this is the whole thing:

> We built a Document Management System for Malaysian receipts. You give it a
> photo of a receipt and it pulls out the named entities — the merchant, the
> address, the date, the total, the tax, the line items — and stores them in a
> searchable database.
>
> The hard part is that Malaysian receipts are bilingual. The same field is
> labelled `TOTAL` on one receipt and `JUMLAH BESAR` on the next, and thermal
> printing means the OCR text is damaged. So we use two extraction methods that
> fail in *different* ways: a rule layer that matches printed keywords in both
> languages, and a local language model that reads the layout the way a person
> would. Where they disagree, receipt arithmetic decides — subtotal plus tax
> must equal the total.
>
> Everything runs locally. No API key, no subscription, no data leaving the
> machine — which matters because receipts are financial records.
>
> On the 97 unseen test receipts it gets **63.6%** of fields exactly right and
> **77.3%** with fuzzy matching.

**Do not say more than this unless asked.** It answers "what", "why it's hard",
"how", and "how well" in four sentences.

---

## 2. Rules, LLM, hybrid — the explanation you keep losing

You have asked about this more than once, so here it is in the form that
actually sticks. **Forget the code. Think about two people reading a receipt.**

### The rule layer is a very fast clerk with a checklist

They speak Malay and English. They have a list: *"the total is whatever number
comes after the words TOTAL, JUMLAH, JUMLAH BESAR, GRAND TOTAL, AMAUN…"*

- **Brilliant at:** anything with a printed label next to it. Finds `TOTAL 22.60`
  instantly and is never wrong about it.
- **Useless at:** anything requiring judgement. *"Which line is the shop name?"*
  The clerk has no idea — there's no label saying "shop name". It guesses "the
  top line", which is usually right and sometimes badly wrong.
- **Never invents anything.** If the label isn't there, it returns nothing.
- **Speed:** under 0.1 seconds.

### The LLM is a thoughtful reader who is bad at arithmetic

They actually *read* the receipt and understand it as a document.

- **Brilliant at:** judgement. Knows the shop name is at the top because that's
  where shop names go. Knows where an address stops. Can repair `22-60` into
  `22.60` because it understands it's a price.
- **Bad at:** digits. Drops leading words (`RIANG` instead of `PERNIAGAAN RIANG`)
  and occasionally invents a plausible-looking number.
- **Speed:** 12–26 seconds, depending on how many line items the receipt
  has — it writes one JSON entry per item, so a long receipt costs more.

### The hybrid is asking both, then checking

For each field, both give an answer, and then:

| Situation | What happens | Confidence |
|---|---|---|
| Only the clerk answered | use it | 0.80 |
| Only the reader answered | use it | 0.70 |
| **They agree** | use it | **0.97** |
| They disagree | **whoever owns that field wins** | 0.55 |
| The money doesn't add up | **arithmetic overrules both** | 0.90 |

**"Who owns the field"** is the routing table in `dms/ner.py:54`:

```python
PREFER_LLM = {"ADDRESS", "ITEM"}     # judgement fields -> the reader
PREFER_RULE = {"MERCHANT", "TOTAL", "TAX", "DATE", ...}   # labelled fields -> the clerk
```

We didn't guess this — we **measured it on the training split** and picked the
best combination. That's `tools/tune_on_train.py`.

### Why agreement means something — the one insight to remember

> The two layers fail in **unrelated** ways. The rules break on layout; the model
> drifts on digits. When two methods that fail differently give the *same*
> answer, that answer is very probably correct.

That's the entire justification for the 0.97. If asked "why 0.97 and not 0.9?",
say: *"it's a ranking signal, not a calibrated probability — what matters is
that agreement outranks every single-layer result."*

### A worked example — say this if they want detail

Run `sample_receipt.jpg` — a timber shop, *Kedai Papan Yew Chuan*. OCR damages
**two** numbers on it, and the system repairs both, for different reasons.

**Repair 1 — the total.** OCR reads the total as `4.80`.

1. Both layers report 4.80. **Neither one notices**, because on its own it is a
   perfectly ordinary number.
2. But the receipt also says cash tendered `84.80` and change `0.00`.
3. `paid − change` = **84.80**, which is an independent statement of the amount
   due, computed by the till itself.
4. Before adopting it, the system checks 84.80 is genuinely printed on the
   receipt. It is.
5. Total is corrected to **84.80**.

**Repair 2 — the tax.** The rule layer reads the tax as `80.00`.

1. **Clerk:** finds a tax label, takes `80.00`.
2. **Reader:** says **4.80**.
3. They disagree; TAX is a rule-owned field, so the clerk wins: 80.00.
4. **Then the plausibility check runs:** 80.00 is **94% of the total**.
   Malaysian GST/SST is 6–10%. That cannot be a tax.
5. Rather than just deleting it, the system looks at what the losing layer
   proposed — 4.80 — finds it plausible, and **promotes it**.

```
! TAX: rule='80.00' vs llm='4.80' -> kept rule
! TOTAL: 4.80 -> 84.80 (paid - change, and printed on the receipt)
! TAX 80.00 is 94% of the total - implausible; using 4.80 from the llm layer instead
```

**This is your best 30 seconds in the whole presentation.** In three printed
lines it shows the two layers, a disagreement, the routing table deciding it,
two independent arithmetic checks overruling the result, and the system
explaining every correction instead of applying it silently.

**Verified through the interface**, where Stage 4 additionally prints:

> *This receipt states its prices already include tax, so the subtotal + tax =
> total check is deliberately suppressed.*

Point at that line. It shows the system distinguishing between a check that
becomes invalid on a tax-inclusive receipt (the arithmetic identity, correctly
switched off) and one that stays valid (the magnitude bound, still firing on the
line below). Telling those apart was the bug fixed during preparation — see
limitation 1.

**If asked "what if both layers are wrong?"** — that is repair 1. Both said
4.80. The receipt's own arithmetic still caught it, because `paid − change` is
evidence that comes from neither layer.

---

## 3. Numbers to know cold

Say these without hesitating. Getting a number wrong is worse than not knowing it.

| | |
|---|---|
| **Headline accuracy** | **63.6% exact, 77.3% fuzzy** |
| Measured on | **97 receipts**, the unseen `test` split, 4 annotated fields |
| Rules only | 63.8 / 73.9 |
| LLM only | 52.2 / 64.9 |
| Train score (for comparison) | 63.3 / 77.7 — gap of **−0.2** |
| Best field | `total`, 78.4% · `date`, 80.4% |
| Weakest field | `address`, 36.5% exact (72.9% fuzzy) |
| OCR | **Tesseract**, `msa`+`eng`, ~7.7 s/receipt |
| Language model | **Qwen2.5-1.5B-Instruct**, 3.1 GB, Apache-2.0, runs local |
| Embeddings | **BGE-M3**, 1024 dimensions |
| Database | SQLite + FTS5 — **40 documents, 600 entities, 206 vectors** |
| Tests | **112 checks**, run offline with no model |
| Per receipt | ~30 s full pipeline on GPU |

### The train/test protocol — have this ready, it is your strongest card

> Every choice — the OCR engine, the merge routing, the rule fixes — was made on
> the **training** split, and the test split was scored **once**, at the end.
> Train came out at 63.3% and test at 63.6%, a gap of −0.2 points, so the
> configuration transfers to unseen receipts rather than being fitted to them.

An earlier version of this project tuned on the test set and reported 63.3%.
That number was inflated by the tuning. We rebuilt the protocol, and the honest
figure now happens to land in the same place for a completely different reason —
a better OCR engine, chosen on training data.

**If a marker points at that coincidence, do not fumble it.** The old 63.3% was
EasyOCR measured on 30 receipts with the routing fitted to those same receipts.
The new 63.6% is Tesseract on all 97 test receipts, scored once. Same
neighbourhood, entirely different provenance.

---

## 4. Questions you will be asked

### On NLP concepts

**Q: What is Named Entity Recognition?**
Finding and classifying the spans of text that refer to real-world things. Generic
NER uses PERSON / ORG / LOCATION. Those don't describe a receipt — there is no
"grand total" tag in CoNLL — so we defined a receipt-specific tag set of 15 types
(`dms/schema.py:16`) and recorded which coarse class each one specialises, so the
system stays interoperable with standard NER.

**Q: Why not use a pre-trained NER model like BERT or LayoutLM?**
Three reasons. They tag PERSON/ORG/LOC, not TOTAL/TAX/SUBTOTAL, so we'd need to
fine-tune — and the dataset only annotates 4 fields, which isn't enough. LayoutLM
needs bounding-box supervision we don't have. And they're weak on Bahasa Melayu.
We compared these options in the report, §2.

**Q: Why not Malaya, the Malaysian NLP library?**
We tried it first. It calls `inspect.getargspec()`, which was **removed in Python
3.11** — every Malaya model raises `AttributeError` on our Python 3.14. Beyond
the crash it was also the wrong tool: it does general Malay NER, so it has no
concept of a receipt total. Analysis in report §2.5.

**Q: How do you handle Bahasa Melayu?**
Four places. OCR runs with `['ms','en']` so the decoder favours Malay word
shapes. The lexicon (`dms/lexicon.py`) holds every Malay label — JUMLAH BESAR,
CUKAI, TUNAI, BAKI, TARIKH — and Malay months (Mac, Mei, Ogos). The LLM prompt
is few-shot with **one Malay and one English example**. And the embeddings are
BGE-M3, which is multilingual, so a Malay query can match English text.

**Q: What are embeddings / what is semantic search?**
An embedding turns a piece of text into a list of 1024 numbers positioned so that
things meaning similar things sit close together. Searching by meaning = embed the
query, compare by cosine similarity. It's how `nasi ayam` finds `Chicken Rice`
when they share no letters.

**Q: How is your evaluation metric calculated?**

Four annotated fields — company, date, address, total. For each, three things:

> **Exact** is not string equality. `total` is compared as a *number*, so 84.80
> matches 84.8. `date` is normalised, so 2018-03-10 matches 10/03/2018.
> `company` and `address` are lowercased with punctuation stripped before
> comparing. **Fuzzy** additionally accepts a similarity of 0.85 or better.
> **Coverage** is how often the field was extracted at all.
>
> The headline pools every field judgement — 246 correct out of 387 labelled,
> so 63.6%. That is a *micro* average, not a macro one; here they agree to 0.1
> because the four fields are labelled almost equally often.

**Q: Why not precision, recall and F1?** *(we do report them)*

> We do. Accuracy conflates answering wrongly with not answering, and those cost
> differently — a missing field is visibly missing, a wrong one is silently
> believed. Micro-averaged: rules 66.9 / 63.8 / **65.3**, hybrid 64.2 / 63.6 /
> **63.9**, model alone 52.2 across the board.

**Q (sharp): Several of your rows show precision, recall and F1 all identical.
Is that a bug?**

No, and this is a good one to get right:

> Each field is a single-valued slot. Where a configuration answers on *every*
> receipt, every error is simultaneously a false positive — it asserted
> something untrue — and a false negative, since it failed to produce the truth.
> So the three measures collapse onto accuracy. They separate only where the
> system declines to answer.
>
> `date` is where that shows. The rule layer only commits when it recognises a
> date, on 87.6% of receipts, so it is *more precise* — 88.2% against 80.4% —
> but recalls less: 77.3% against 80.4%. That is a genuine trade, and accuracy
> reports one number for it without saying which way it went.

**Q: What is FTS5?**
SQLite's built-in full-text search extension — an inverted index for fast keyword
lookup. We use it for the text search and our own vector table for meaning.

### On design decisions

**Q: Why two layers instead of just the LLM?**
Measured on the test split: LLM alone 52.2%, rules alone 63.8%, hybrid 63.6%.
The LLM alone is clearly the weakest of the three. The rules also guarantee the
system never invents a value, and they let it run with **no model at all** on a
machine that cannot host one — that is the `--no-llm` flag.

**Q (the hard one): Then why the language model at all? Your own table shows
rules alone score 63.8% and the hybrid 63.6%. The LLM makes it *worse*.**

Do not dodge this. It is the sharpest question in the project and the honest
answer is a good one:

> On exact match they are level — 0.2 points apart, well inside the noise of 97
> receipts. That is a genuine finding, and it was not true earlier: with the
> weaker OCR engine the hybrid led by three points. Improving the reader took
> away most of what the language model had to repair.
>
> It still earns its place on two axes the exact-match column does not show.
> It is **3.4 points ahead on fuzzy match** — 77.3% against 73.9% — so when it
> is wrong it is much closer to right, and on addresses that gap is nine points.
> And it is more **complete**: it finds a date on 100% of receipts against the
> rules' 87.6%, and a total on 95.9% against 93.8%. A field the rules never
> found scores zero on every metric.
>
> The honest summary is that the language model has stopped being an accuracy
> win and become a *robustness* win. On a harder corpus — worse printing, more
> Malay, fewer printed labels — I would expect the gap to reopen.

**Q: So why not ship the rules-only configuration?**
Because that decision would be made by looking at the test score, which is the
one thing the protocol forbids. On the **training** split the hybrid won
(63.3% against 62.3%), so the hybrid was frozen and shipped. Changing it now
because the test result came out differently is exactly the leakage we removed
earlier in the project.

**Q: Why Qwen2.5-1.5B? Isn't there a newer Qwen?**

There is — Qwen3.5. We tested it, and the answer is more interesting than "we
used the newest":

> Qwen3.5 comes in 0.8B, 2B and 4B — there is no 1.5B, and the 4B needs 9.3 GB
> so it will not fit an 8 GB card. We benchmarked 0.8B and 2B against the
> current model. Every Qwen3.5 variant was a **better model** — four to six
> points higher on LLM-only accuracy — but the **hybrid barely moved**, at most
> 0.8 points, because the rule layer already supplies most of what the model
> contributes.

**Q: So did you switch?**

Say this exactly, because it is the strongest methodological answer in the pack:

> We did, and then we reverted. On a 60-receipt comparison Qwen3.5-0.8B led,
> 59.6% against 58.8%, so we adopted it. Re-measured on the full 120-receipt
> training split the ranking reversed — 62.9% against 63.3% — and the test split
> agreed, 62.3% against 63.6%. The whole loss was in `address`, which is the one
> field routed to the language model, and a model half the size reads a long
> damaged address block less well.
>
> The lesson is that a sub-one-point gap at 60 receipts is not a result. We
> treated it as one, made the change, and the larger sample caught it.

**Q: Why not a bigger model?**
Qwen2.5-3B is the most accurate we measured — 61.3% hybrid — but 72.6s per
receipt against ~15s, and 6.8 GB of an 8 GB card. That is ~80 seconds of silence
for a live upload and a real out-of-memory risk, to gain about four field
judgements out of 240. Declined deliberately.

**Q: Did you only try Qwen?**
No — we checked Llama-3.2, Gemma-3, Phi-4-mini, IBM Granite, SmolLM2, Falcon3
and LFM2. Llama and Gemma are **gated**: they need a HuggingFace account and
manual approval, which breaks the project's "no account, no token" constraint.
Phi-4-mini needs 7.1 GB and will not fit. The alias system means any of them can
be swapped in with `--model <name>` and no code change.

**Q: Why SQLite rather than MySQL or Postgres?**
It's a single file, needs no server, and the whole project is meant to run from a
clone with no setup. For a corpus this size the query cost is irrelevant. The
storage layer is one module (`dms/database.py`), so swapping it wouldn't touch
the pipeline.

**Q: How do you highlight the result on the original image?**
Every entity stores character offsets into the OCR text *and* the pixel polygons
of the OCR tokens it overlaps. So one entity can be highlighted in the text and
boxed on the image from the same record.

### The traps

**Q: Your benchmark says the lexical baseline beat BGE-M3 (100% vs 86.4%). So why use embeddings at all?**

This is the sharpest question in the pack. Answer:

> Because that benchmark's probes were lexically similar to the stored text, which
> is exactly where string matching wins. Embeddings earn their place on the
> queries where lexical similarity is **zero** — a Malay query matching English
> text. That's why the cascade runs both and takes the union rather than choosing
> one. Lexical is cheap and precise; semantic covers what it structurally cannot.

**Q: 63% doesn't sound very high.**

> It's an honest number — every choice selected on train, test scored once. It's
> also micro-averaged over four fields including `address`, the hardest, which
> drags the mean down: `date` is 80.4% and `total` 78.4%. On fuzzy match the
> figure is 77.3%. And the OCR ceiling caps us — on some receipts the field
> simply is not legible after thermal printing, so no extractor could recover
> it. We raised the headline from 56.1% to 63.6% purely by changing the reader,
> which is evidence that OCR, not the extraction logic, is the binding
> constraint.

**Q: Did you tune anything on the test set?**
No — and say it firmly. `tools/tune_on_train.py` selects the routing on the
training split; the test split is scored once. We previously had this wrong,
found it, and fixed it.

**Q: What did the AI tools do and what did you do?**
Report §10 is the declaration. Answer honestly and specifically.

---

## 5. Known limitations — answer these before they find them

Volunteering a real limitation reads as mastery. Having none reads as not
understanding your own system.

1. **A bug we found and fixed — tell this as a story, not an apology.**
   `is_tax_inclusive()` was disabling the *entire* validation block on receipts
   that say "inclusive of GST". Disabling the arithmetic identity there is
   correct, because `subtotal + tax = total` is deliberately false. But it also
   disabled the **plausibility bound**, which is a magnitude test and stays valid
   either way. Three stored documents ended up with a "tax" equal to the whole
   subtotal, each time overriding a correct figure from the language model.
   Fixed by moving that one check outside the guard, and improved further: when
   an implausible tax is rejected, the value proposed by the *other* layer is now
   promoted if it is plausible, instead of the field being dropped. Plausible
   taxes went from 7 of 14 to **16 of 22**.

2. **`address` is the weakest field at 36.5% exact** — multi-line, no consistent
   label, worst OCR damage. Rules-only actually scores *higher* on exact (41.7%)
   while the hybrid is far better on fuzzy (72.9% against 63.5%). Our routing
   sends `ADDRESS` to the language model because that won on the training split;
   on test that costs 0.2 points overall. We left it, because changing it after
   seeing the test score is leakage.

3. **OCR is the binding constraint, not the extraction logic.** Changing nothing
   but the reader moved the headline from 56.1% to 63.6%. Roughly a quarter of
   remaining `company` errors are single-character OCR damage — `Matketing` for
   `MARKETING`, `AFON` for `AEON` — which no rule can repair.

4. **The merchant name is still fragile when it spans lines.** We join a
   corporate suffix line backwards onto the brand above it, but only when the
   suffix line cannot stand alone. A broader rule was measured on the training
   split and lost 6.7 points, because it swallowed slogans printed above the
   name.

5. **No layout model.** We read text, not visual structure. A two-column receipt
   confuses the line reconstruction.

---

## 6. Live-edit drills

**All three are verified working.** Practise each twice on the machine you'll
present on. Keep this file open in a second window during the demo.

### Drill 1 — "Add a new entity type" (≈4 min)

The classic *extend the system* request. Add **SERVICE_CHARGE**, common on
Malaysian restaurant receipts.

**Step 1 —** `dms/lexicon.py`, inside `MONEY_LABELS`, add as the **first** entry
(order matters — longest and most specific first):

```python
("SERVICE_CHARGE", [
    "caj perkhidmatan", "caj servis", "service charge", "svc charge",
]),
```

**Step 2 —** `dms/schema.py`, add to `ENTITY_TYPES` and `MONEY_TYPES`:

```python
"SERVICE_CHARGE": "MONEY",
```
```python
MONEY_TYPES = {"SUBTOTAL", "TAX", "TOTAL", "PAID", "CHANGE", "SERVICE_CHARGE"}
```

**Step 3 (only if they ask for the LLM to find it too) —** `dms/ner.py`, add to
`LLM_KEY_TO_TYPE`: `"service_charge": "SERVICE_CHARGE",` and add the name to
`PREFER_RULE`. Then add `"service_charge":num|null` to `HEADER_FIELDS` and
`FULL_FIELDS` in `dms/llm.py`.

**What to say while typing:** *"The rule layer builds its label table from the
lexicon at import time, so adding the vocabulary is all the rules need. The
schema entry tells the merge logic to treat it as money."*

**Verified result** — on a restaurant receipt containing `Caj Perkhidmatan 10%
1.55`, before the change nothing is extracted for that line; after it:

```
SUBTOTAL         15.50
SERVICE_CHARGE    1.55     <- new
TAX               0.93
TOTAL            17.98
```

### Drill 2 — "Change how fuzzy the search is" (≈1 min)

The safest drill. One line, instantly visible in the GUI.

`dms/config.py:81`:

```python
SIMILARITY_THRESHOLD = float(os.getenv("DMS_SIMILARITY", "0.72"))
#                                                         ^^^^ change to 0.95
```

**Verified scores** — this is what the change does:

| query → stored | score | at 0.72 | at 0.95 |
|---|---|---|---|
| `kuala lumpor` → `kuala lumpur` | 0.917 | ✅ found | ❌ lost |
| `petaling jaya` → `petaling java` | 0.923 | ✅ found | ❌ lost |
| `johor bahru` → `kuala lumpur` | 0.125 | ❌ | ❌ |

**What to say:** *"0.72 sits in the measured gap — real typos score 0.87 to 1.00,
unrelated pairs peak at 0.55. Raise it to 0.95 and the system stops tolerating
typos; the misspelling falls through to the place-name fallback instead."*

Restart Streamlit after editing, then search `Kuala Lumpor` to show it.

### Drill 3 — "Add a validation rule" (≈3 min)

Shows you understand the arbitration layer, not just the vocabulary.

`dms/ner.py`, inside `_validate_arithmetic`, after the existing subtotal/tax/total
block:

```python
# Malaysian GST was 6% and SST is 6-10%. A "tax" far outside that band is
# almost always a misread figure rather than a real rate.
if sub is not None and tax is not None and sub > 0 and tax > 0:
    rate = tax / sub
    if not (0.05 <= rate <= 0.11):
        warnings.append(
            f"tax {tax:.2f} is {rate:.1%} of subtotal {sub:.2f}; "
            "Malaysian GST/SST is 6-10%"
        )
```

**Verified result** — run against the 40 stored documents this fires on **6 of
the 22** that have both a subtotal and a tax:

| doc | file | subtotal | tax | rate |
|---|---|---|---|---|
| 12 | test_00011.jpg | 0.38 | 0.38 | 100.0% |
| 25 | test_00024.jpg | 17.05 | 6.30 | 37.0% |
| 15 | test_00014.jpg | 12.00 | 2.12 | 17.7% |
| 29 | test_00028.jpg | 24.11 | 0.52 | 2.2% |

**What to say:** *"Six of twenty-two are outside a plausible Malaysian tax band,
so the rule earns its place immediately — and it also shows its own limits."*

Then point at **doc 29**, which is the interesting one:

> *"2.2% looks wrong, but it is not. That receipt is an AEON basket mixing
> zero-rated groceries with standard-rated items, so the effective rate across
> the whole bill genuinely is below 6%. My check compares tax against the whole
> subtotal, when GST applies only to part of it. A stricter version would need
> the per-item tax codes, which this receipt prints but we do not parse."*

**Why this is the strongest of the three drills.** You add a rule, it finds real
problems, and you can immediately explain which of its own findings are false
positives and why. That is the difference between running code and understanding
it — which is what Aspect 11 actually asks for.

> **This exact drill already paid off once.** Writing it during preparation
> exposed a genuine bug: `is_tax_inclusive()` was suppressing the whole
> validation block on tax-inclusive receipts, including the magnitude check,
> which is still valid there. Three stored documents carried a "tax" equal to
> the entire subtotal. It is fixed — see limitation 1 below — and the plausible
> count went from 7 of 14 to 16 of 22.

---

## 7. The demo — 20 minutes, two people

**Your teammate opens (4 min):** the problem, the background reading, why NER on
receipts, what's been done before. Their report content — nothing about the code.

**You (14 min), driving the GUI:**

| Time | Page | What you show | What you say |
|---|---|---|---|
| 0–2 | How it works | the pipeline | the 90-second answer (§1) |
| 2–6 | Process a receipt | **upload one live** | which filter won and why; the entities appearing with confidence and source |
| 6–8 | Process a receipt | the warnings panel | the tax repair story (§2) — **your best 30 seconds** |
| 8–10 | Process a receipt | boxes on the image | offsets and pixel boxes, why both |
| 10–14 | *hand over* | | |

**Teammate (4 min), Search page:** `Kuala Lumpur` → exact. `Kuala Lumpor` →
still found. `nasi ayam` → Malay query, English result, by meaning.
`Kota Kinabalu` → not in the database, returns other locations — *"this is the
fallback the assignment asks for"*.

**Both (2 min):** results table, limitations, close.

### The live upload

- Database is **already loaded with 40 receipts** — search never depends on the
  live run.
- Have the receipt file **already open in the file picker** before you start.
- It takes **~35 seconds** with Tesseract — measured through the interface on
  the presentation machine: 7.9s reading the image, 26.0s in the language model.
  **Talk through the stages while it runs** — don't stand in silence. That is
  exactly the window for explaining how the preprocessing variant is chosen, and
  it is long enough that silence would be uncomfortable.
- The interface shows each stage as it completes, so there is always something
  on screen to point at while you talk.
- Pick the receipt on Friday and run it three times on the presentation machine.

---

## 8. If something breaks

| What happens | What you do |
|---|---|
| Live upload hangs > 60 s | *"I'll show one already processed"* → Database page. **Don't wait.** |
| Streamlit won't start | `.\.venv\Scripts\streamlit.exe run app.py` — full path, not bare `streamlit` |
| Port in use | add `--server.port 8502` |
| CUDA out of memory | restart the app; or `--model qwen0.5b` |
| Search finds nothing | check the Database page shows 40 documents |
| A question you can't answer | *"I don't know that — what I can tell you is how we handled the related case…"* **Never invent a number.** |

**The last row is the important one.** A marker will respect "I don't know"
followed by something you do know. They will not respect a confident wrong
answer, and Aspect 11 says *"accurately"*.

---

## 9. Checklist for Saturday morning

- [ ] `.\.venv\Scripts\python.exe test_dms.py` → 112 checks pass
- [ ] `.\.venv\Scripts\python.exe tools/prepare_demo.py` → READY TO PRESENT
- [ ] Streamlit opens, Database page shows **40 documents**
- [ ] **Turn "Save to database" OFF while rehearsing.** It defaults to ON, so
      every practice run adds a row and the count drifts away from the 40 quoted
      in the report. If it has drifted, `run_dms.py stats` shows the count and
      `run_dms.py delete --doc-id N --yes` removes the extras
- [ ] Live-upload receipt chosen, and run **3 times** on this machine
- [ ] All four searches tried on this machine
- [ ] Laptop plugged in, sleep disabled, notifications off
- [ ] This file open in a second window
- [ ] Your teammate has read `docs/SEARCH_BRIEFING.md`
