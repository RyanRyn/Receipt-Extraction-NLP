# Code walkthrough — what every file does

Companion to `DEFENCE_PACK.md`. That one is what to *say*; this is what the code
*is*, for the on-the-spot coding part of the assessment.

**Read it in pipeline order.** The system is a straight line — image in, searchable
record out — and the files sit along it in that order. If you can name the line,
you can place any file on it.

```
config.py ──── settings for everything below
    │
ocr.py ─────── image  →  text          (+ textutils, for cleaning)
    │
rules.py ───── text   →  entities      (+ lexicon, for Malay/English words)
llm.py ─────── text   →  entities      (the second opinion)
    │
ner.py ─────── merge the two, then check the arithmetic   ← the core contribution
    │
schema.py ──── the shape of the data flowing through
    │
database.py ── store it, search it     (+ embeddings, for meaning)
    │
pipeline.py ── glues the above into one call
    │
app.py / run_dms.py ── the two front ends
```

---

## 1. `dms/config.py` — every setting in one place

**Purpose.** Nothing here computes anything. It holds paths, thresholds, the
model registry, and the hardware check, so that no magic number is buried in the
logic.

| Line | What |
|---|---|
| **58** | `LLM_MODELS` — short name → **public** HuggingFace repo id |
| **105** | `DEFAULT_LLM_KEY` — which one is used by default |
| 162 | `resolve_model()` — alias → repo, **unknown values pass through unchanged** |
| 171 | `detect_device()` — returns `("cuda", float16)` or `("cpu", float32)` |

**The thresholds worth knowing:**

```python
CONF_AGREE          = 0.97   # both layers said the same thing
CONF_RULE_ONLY      = 0.80   # only the rules found it
CONF_LLM_ONLY       = 0.70   # only the model found it
TAX_MAX_FRACTION    = 0.35   # a "tax" above 35% of the total is not a tax
SIMILARITY_THRESHOLD= 0.72   # spelling match for search
```

> **Q: "Why 0.35 for tax?"** Malaysian GST was 6% and SST is 6–10%. Even with a
> 10% service charge, nothing legitimate approaches 35%. Anything above it is a
> misread figure, almost always the subtotal, which sits next to the tax on the
> receipt.

---

## 2. `dms/ocr.py` — picture to text

**Purpose.** Turn pixels into text *that keeps its layout*, because on a receipt
the layout carries the meaning.

**Five preprocessing filters** (lines 52–106): `pre_raw`, `pre_gray_otsu`,
`pre_adaptive`, `pre_clahe_sharp`, `pre_deskew_otsu`. Each is a different guess
at making faded thermal print readable.

**`read(image, variant="auto")`** — runs three filters and keeps whichever scores
best. The score is `_score()`: the **sum of word confidences**, i.e. the expected
number of correctly read words. Not the mean — a filter that reads three words
perfectly should not beat one that reads forty well.

**`group_into_lines()` (line 130) — the most important function in the file.**
OCR returns loose boxes in arbitrary order. This regroups them into visual lines
by vertical centre, within 0.6 × the median glyph height. Without it,
`JUMLAH BESAR` and `60.31` are two unrelated fragments; with it they are one line
and the rule layer can pair them.

It also records `char_start` / `char_end` for every token — which is what later
lets an entity be traced back to a pixel box.

**Two engines, same interface:** `ReceiptOCR` (EasyOCR) and `TesseractOCR`.
`make_ocr()` at line 497 builds whichever is configured, and falls back to
EasyOCR with a warning if Tesseract is not installed.

> **Q: "Why is Tesseract the default?"** Measured on the 120-receipt training
> split: 63.3% exact against EasyOCR's 51.7%, with address nearly doubling, for
> about 2.5s more per receipt. Chosen on training data, never on test.

---

## 3. `dms/textutils.py` — the shared toolbox

**Purpose.** Small text operations used by every layer. No receipt knowledge
lives here, only string handling.

| Function | Job |
|---|---|
| `normalize_key()` | lowercase, strip accents, **punctuation → spaces**. Used for *comparison only* — search and scoring |
| `parse_money()` | `"RM 22-60"` → `22.60`. Handles OCR-damaged separators |
| `repair_digits()` | `O`→`0`, `l`→`1`, but **only inside a token already known to be a number** |
| `normalize_date()` | `"Rabu, 12-07-2017"` → `2017-07-12`, including Malay month names |
| `locate_span()` | Finds a *cleaned* value back in the *messy* original text, so offsets still work after repair |
| `partial_similarity()` | Slides a window over the text — this is what makes `Kuala Lumpor` match inside a full address |
| `strip_graphic_noise()` | Removes logo artefacts (`\| PASAR MINI \| \|`) |

> **Q: "How do you highlight a value the model rewrote?"** `locate_span()`. The
> model returns `22.60`, the receipt says `22-60`. We fuzzy-match the cleaned
> value back onto the source text to recover character offsets, then map those to
> pixel boxes.

---

## 4. `dms/lexicon.py` — the bilingual vocabulary

**Purpose.** Every Malay and English word the rules look for, in one file, so a
new language needs no change to the extraction logic.

```python
MONEY_LABELS = [
    ("TOTAL",    ["jumlah besar", "grand total", "jumlah", "total", ...]),
    ("TAX",      ["cukai gst", "cukai", "gst", "sst", "tax", ...]),
    ("PAID",     ["tunai", "cash", "dibayar", ...]),
    ("CHANGE",   ["baki", "change", ...]),
]
```

**Order matters** and `all_money_labels()` sorts longest-first, so `JUMLAH KECIL`
(subtotal) is matched before the shorter `JUMLAH` (total) can swallow it.

Also holds `NOISE_LINES` (thank-you footers), `SUMMARY_HEADERS` (GST summary
tables), `MALAYSIAN_PLACES`, and `TAX_INCLUSIVE_MARKERS`.

> **Q: "How does it handle Bahasa Melayu?"** Four places: OCR runs `ms`+`en`;
> this lexicon; the LLM prompt has one Malay and one English worked example; and
> the embeddings are multilingual so a Malay query can match English text.

---

## 5. `dms/rules.py` — the deterministic extractor

**Purpose.** Find entities by matching printed keywords and layout patterns.
Fast, exact, and it **cannot invent a value**.

**`extract_rules()` (line 468) is the entry point.** It calls, in order:

- `_money_candidates()` → every labelled amount on every line
- `_pick_money()` → choose one per type, preferring the most specific label
- `_extract_merchant()` → the branded line at the top
- `_extract_address()` → the lines under it carrying a street word or postcode
- `_extract_date` / `_time` / `_phone` / `_payment` / `_items`

**`_extract_merchant()` is the one to read**, because it shows the reasoning:
search the top 7 lines for a company marker (`SDN BHD`, `ENTERPRISE`, `KEDAI`…);
if none, take the first line with ≥4 letters that is not noise and not a
postcode. Then join a suffix line forwards *or* backwards — `POPULAR BOOK` +
`CO. (M) SDN BHD` are one name printed on two lines.

> **Q: "Why did the rules beat the language model?"** 63.8% against 52.2% exact.
> Receipts are semi-structured: most fields have a printed label right next to
> them, which is exactly what a keyword matcher is good at.

---

## 6. `dms/llm.py` — the language model layer

**Purpose.** Ask a small local model to read the receipt and return a JSON
record. Used for what the rules cannot do: judgement.

**The prompt** (lines 42–115): a system prompt, a `RULES` block, the exact JSON
shape, and **two worked examples** — one Malay, one English. That is the
few-shot part.

**`_ensure_loaded()` (272) → `from_pretrained()` (287)** — where the weights
enter VRAM. `_loaded` is a **class-level** dict, so the model loads once per
process, not once per receipt.

**`parse_json_object()` (175) — the defensive part, and worth showing.** A small
model does not always return clean JSON. This strips markdown fences, and if
parsing still fails, `salvage_fields()` recovers whatever individual fields it
can rather than discarding the whole answer. `_evaluate_arithmetic()` handles a
model that writes `"unit_price": 54.40 / 3` — which is not valid JSON.

**Generation is greedy** (`do_sample=False`), so the same receipt always gives
the same answer. That is what makes the results reproducible for a report, and
what makes the response cache safe.

> **Q: "What if the model hallucinates a total?"** Two defences. Every money
> value it returns is checked against the amounts actually printed on the receipt
> (`_value_printed`), and anything not found is dropped. Then the arithmetic
> validators check what survives.

---

## 7. `dms/ner.py` — the hybrid merge ⭐

**This is the contribution. If you only master one file, make it this one.**

**`HybridNER.extract()` is the whole story:**

1. Run the rule layer → entities
2. Run the LLM layer → entities
3. `_merge_single()` per field — reconcile the two
4. `_arbitrate_money()` — let arithmetic settle what routing could not
5. `_repair_total_from_payment()`, `_fix_implausible_tax()`, `_recover_tax_from_text()`
6. Add a null row for every type neither layer found

**`_merge_single()` (line 184):**

| Situation | Result | Confidence |
|---|---|---|
| rules only | rule value | 0.80 |
| LLM only | model value | 0.70 |
| **agree** | either | **0.97** |
| disagree | whoever *owns* the field | 0.55 |

Ownership is `PREFER_LLM = {"ADDRESS", "ITEM"}` at **`ner.py:55`** — everything
else goes to the rules. **Measured on the training split**, not guessed (`tools/tune_on_train.py`).

**`_arbitrate_money()`** is the clever one. When the two disagree on
subtotal/tax/total, it tries every combination of each layer's proposal and keeps
the one satisfying `subtotal + tax = total` — preferring the combination that
**disturbs the least confident readings**, so a value both layers agreed on is
not overwritten to satisfy one that only one proposed.

> **Q: "Why does agreement mean 0.97?"** The two layers fail in *unrelated* ways
> — the rules break on layout, the model drifts on digits. When two methods that
> fail differently agree, the answer is very probably right. It is a ranking
> signal, not a calibrated probability.

---

## 8. `dms/schema.py` — the shape of the data

Three dataclasses, no logic:

- **`Entity`** — type, value, surface text, `start`/`end` offsets, confidence,
  source. `value` may be `None`, meaning *looked for and not on this receipt*.
- **`OcrResult`** — text, lines, tokens, which filter won.
- **`ReceiptDocument`** — everything about one receipt.

**`ENTITY_TYPES` (line 16)** is the tag set: 16 receipt-specific types, each
mapped to the coarse CoNLL class it refines.

> **Q: "Why not standard PERSON/ORG/LOC?"** They do not describe a receipt —
> there is no "grand total" tag in CoNLL. We defined a receipt-specific set and
> recorded which coarse class each one specialises, so it stays interoperable.

---

## 9. `dms/database.py` — storage and search

**Two tables.** `documents` holds the OCR text and the flattened record;
`entities` holds one row per entity with `value`, `value_norm`, `start`, `end`,
`confidence`, `source` and a 1024-float `embedding` blob.

**`search_entities()` (line 512) — the search cascade.** Four stages, first hit
wins, and the mode is returned so a result can explain itself:

1. **exact** — `value_norm = query`
2. **partial** — `value_norm LIKE '%query%'`
3. **similar** — lexical (`difflib`) **and** semantic (cosine), merged
4. **type_fallback** — the query is a Malaysian place we do not hold, so return
   the places we do — *this is the requirement the assignment names*

> **Q: "Why run both lexical and semantic?"** They catch different things.
> Lexical rescues a typo — `Kuala Lumpor`. Semantic rescues a different word for
> a related thing — `nasi ayam` finding `Chicken`, where the lexical score is
> zero because they share no letters.

---

## 10. `dms/embeddings.py` — meaning as numbers

`Embedder` wraps BGE-M3. `encode()` turns text into a 1024-number vector;
`cosine_ranking()` compares one against all stored vectors by dot product.

Only **four** entity types are embedded — `MERCHANT`, `ADDRESS`, `ITEM`,
`CASHIER` — the ones whose values are natural language.

> **Q: "Why not embed everything?"** Because `30.00` and `2024-03-14` have no
> meaning to compare. Embedding them actively hurt: a search for *Kota Kinabalu*
> once returned a payment amount ranked above the addresses.

---

## 11. `dms/pipeline.py` — the glue

`Pipeline.process()` does the whole chain in ~60 lines: OCR → NER → attach pixel
boxes → build a `ReceiptDocument` → store it. It holds the OCR reader, the model
and the database **open across a batch**, because loading them per image would
cost more than the work.

---

## 12. The two front ends

- **`run_dms.py`** — `process`, `search`, `show`, `stats`, `dataset`, `export`,
  `reindex`, `delete`. Flags: `--model`, `--ocr-engine`, `--no-llm`, `--no-store`.
- **`app.py`** — Streamlit, four pages. Models are cached with
  `@st.cache_resource` so they load once for the life of the server.

⚠️ **Streamlit imports modules once at startup.** Editing anything under `dms/`
while it runs changes nothing until you restart it.

---

## 13. `test_dms.py` — 112 checks

Runs offline in seconds with **no model and no GPU**, so a broken install is
caught before 5 GB is downloaded. Several checks are marked `[regression]` —
each is a bug that was found once and is now prevented from returning.

---

## If you are asked to change something live

| Ask | Where | Difficulty |
|---|---|---|
| Change the model | `--model <alias>`, or `config.py:105` | trivial |
| Add a model | one line in `LLM_MODELS`, **or none at all** — any repo id works | trivial |
| Add an entity type | `schema.py` `ENTITY_TYPES` + `lexicon.py` labels | easy |
| Change search fuzziness | `config.py` `SIMILARITY_THRESHOLD` | trivial |
| Add a validation rule | `ner.py` `_validate_arithmetic` | moderate |
| Change which layer owns a field | `ner.py:55` `PREFER_LLM` | trivial, but say it was measured |

**Do the edit at the command line where you can.** Editing a file live means a
typo or a forgotten restart fails silently, and you look stuck when the code is
fine.
