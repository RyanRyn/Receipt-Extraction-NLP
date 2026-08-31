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
> On the 97 unseen test receipts it gets **56.1%** of fields exactly right and
> **68.7%** with fuzzy matching, and the hybrid beats either layer alone.

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
- **Speed:** ~12 seconds.

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

On `sample_receipt.jpg`, the tax line is smudged. OCR reads it as `1.2g`.

1. **Clerk:** sees `CUKAI` but can't parse `1.2g` as a number → reports **0.00**.
2. **Reader:** guesses **1.28**.
3. **They disagree.** TAX is a rule-owned field, so the clerk wins: 0.00.
4. **But then arithmetic runs:** subtotal 21.32 + tax 0.00 = 21.32 ≠ total 22.60. ✗
5. It tries the alternative: 21.32 + **1.28** = 22.60. ✓
6. It also checks 1.28 is genuinely printed on the receipt — it is.
7. Tax is corrected to **1.28**, confidence 0.90, and the system prints *why*:

```
! TAX: 0.00 -> 1.28 (total - subtotal, and printed on the receipt)
```

**This is your best 30 seconds in the whole presentation.** It shows the layers,
the disagreement, the arbitration, and the self-explanation, on real output.

---

## 3. Numbers to know cold

Say these without hesitating. Getting a number wrong is worse than not knowing it.

| | |
|---|---|
| **Headline accuracy** | **56.1% exact, 68.7% fuzzy** |
| Measured on | **97 receipts**, the unseen `test` split, 4 annotated fields |
| Rules only | 53.2 / 66.7 |
| LLM only | 43.2 / 54.5 |
| **Hybrid gain on `total`** | 64.9% → **76.3%** ← the arithmetic validators |
| Weakest field | `address`, 22.9% exact |
| OCR | EasyOCR, Malay + English, ~5 s/receipt |
| Language model | **Qwen2.5-1.5B-Instruct**, 3.1 GB, Apache-2.0, runs local |
| Embeddings | **BGE-M3**, 1024 dimensions |
| Database | SQLite + FTS5 — **40 documents, 600 entities, 206 vectors** |
| Tests | **112 checks**, run offline with no model |
| Per receipt | ~30 s full pipeline on GPU |

### The train/test protocol — have this ready, it is your strongest card

> We tuned every choice on the **training** split and scored the test split
> **once**, at the end. An earlier version tuned on the test set and reported
> 63.3%; that number was inflated by the tuning, so we rebuilt the protocol and
> report the honest 56.1%.

If a marker thinks 56% sounds low, that answer is worth more than the missing
points. Most student projects cannot say it.

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

**Q: What is FTS5?**
SQLite's built-in full-text search extension — an inverted index for fast keyword
lookup. We use it for the text search and our own vector table for meaning.

### On design decisions

**Q: Why two layers instead of just the LLM?**
Measured: LLM alone is 43.2%, rules alone 53.2%, hybrid 56.1%. The rules also
guarantee it never invents a value, and they let the whole system run with **no
model at all** on a machine that can't host one — that's the `--no-llm` flag.

**Q: Then why not just the rules? They're close and much faster.**
The 3 points matter most where it counts — `total` goes from 64.9% to 76.3% —
and the LLM contributes the fields rules can't do at all: address, line items,
and repairing OCR damage.

**Q: Why Qwen2.5-1.5B and not something bigger?**
We benchmarked four (report §5). Qwen2.5-3B scored slightly better on fuzzy but
took 21 s/receipt against 12 s and needed 6.8 GB VRAM against 3.4 GB. 1.5B was
the best accuracy-per-second. The alias system means you can swap with
`--model qwen3b` — no code change.

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

**Q: 56% doesn't sound very high.**

> It's an honest number — selected on train, scored once on test. It's also
> macro-averaged over four fields including `address`, which is the hardest and
> drags the mean down; `total` is 76.3% and `date` 78.4%. And the OCR ceiling
> caps us: on some receipts the field simply isn't legible after thermal
> printing, so no extractor could recover it.

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

1. **The tax-inclusive guard over-suppresses.** When a receipt says "inclusive of
   GST", we disable the arithmetic checks — correctly, because `subtotal + tax =
   total` is deliberately false there. But we *also* disable the plausibility
   bound, which is a magnitude check and still valid. Result: 3 of the 40 stored
   documents carry a "tax" equal to ~100% of the subtotal. Found by writing a
   tax-rate validator (drill 3 below).
2. **`address` is 22.9%.** Multi-line, no consistent label, worst OCR damage.
   Rules-only actually scores marginally *higher* (24.0%) — our routing picked
   the LLM on training-split fuzzy score, and that choice doesn't hold on test
   exact.
3. **The OCR ceiling.** Field recoverability is 87.5% for EasyOCR — on ~1 receipt
   in 8 the field is not legible at all, so extraction cannot succeed.
4. **Tesseract measured better than EasyOCR** (91.2% recoverability) but is 2.3×
   slower, so it's available as an option rather than the default.
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

**Verified result** — run against the 40 stored documents this fires on **7 of
the 14** that have both a subtotal and a tax:

| doc | file | subtotal | tax | rate |
|---|---|---|---|---|
| 36 | test_00004.jpg | 26.60 | 26.60 | 100.0% |
| 67 | test_00035.jpg | 10.40 | 10.40 | 100.0% |
| 38 | test_00006.jpg | 277.90 | 277.00 | 99.7% |
| 60 | test_00028.jpg | 24.11 | 8.58 | 35.6% |

**What to say — and this is the strong move:** *"It immediately flags a real
defect. On those receipts the rule layer read the subtotal again as the tax, and
the language model had it right — but our routing gives TAX to the rules. It gets
through because those receipts say 'inclusive of GST', and we suppress the
plausibility check on tax-inclusive receipts. That suppression is too broad: the
identity check should be disabled, but the magnitude check is still valid."*

Volunteering that is worth more than a clean run.

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
- It takes ~30 seconds. **Talk through the stages while it runs** — don't stand
  in silence. Use the time to explain preprocessing variant selection.
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
- [ ] Live-upload receipt chosen, and run **3 times** on this machine
- [ ] All four searches tried on this machine
- [ ] Laptop plugged in, sleep disabled, notifications off
- [ ] This file open in a second window
- [ ] Your teammate has read `docs/SEARCH_BRIEFING.md`
