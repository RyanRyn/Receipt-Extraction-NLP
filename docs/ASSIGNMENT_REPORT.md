# A Bilingual Receipt Document Management System using Named Entity Recognition

**BMDS2123 Natural Language Processing — Assignment**
**Topic 2: Document Management System (DMS using Named Entity Recognition)**

---

## 1. Introduction

### 1.1 Problem statement

Corporations and individuals handle large volumes of transaction receipts daily,
for expense tracking, tax compliance and financial auditing. Traditional
Document Management Systems store these receipts as flat images — JPEG, PNG or
unstructured PDF. This creates an operational bottleneck: an employee must open
each file, read it, and manually transcribe the transaction date, the merchant's
name and the amounts into an accounting application.

The difficulty is not merely one of volume. Receipts have irregular layouts,
varying fonts and fluctuating print quality, and thermal paper fades with age.
Because the text is locked inside pixels, a standard keyword search cannot
locate transactional data at all.

In Malaysia the problem carries a further dimension. Receipts are routinely
printed in a mixture of Bahasa Melayu and English — the same receipt may label
the date `TARIKH` and the total `TOTAL` on adjacent lines — so any solution
confined to one language will fail on a substantial proportion of documents.
Malaysian taxpayers must also retain records for seven years under the Income
Tax Act 1967, which makes a folder of unreadable photographs an increasingly
poor answer.

### 1.2 Justification for the proposed NLP solution

Conventional keyword search requires an exact character match and has no
structural awareness. The solution proposed here combines Optical Character
Recognition with Named Entity Recognition so that the DMS can interpret a
receipt's underlying semantics rather than treating it as an unstructured
string.

The central observation motivating this project is that **generic NER is not
sufficient for a receipt**. A standard NER model produces tags such as PERSON,
ORGANISATION, LOCATION and TIME. On the receipt used throughout this report it
can identify `PERNIAGAAN RIANG` as an organisation, but it has no tag at all for
the concepts that make a receipt financially useful:

```
TOTAL   22.60
CASH    30.00
CHANGE   7.40
GST      1.28
```

To a generic tagger all four are simply MONEY. Distinguishing them requires the
*label* printed beside the value (`TOTAL`, or `JUMLAH BESAR` in Malay), its
*position* on the line, and an *arithmetic consistency check*
(`subtotal + tax = total`). This project therefore treats receipt NER as
semantic role labelling over a two-dimensional layout, using a receipt-specific
tag set of sixteen entity types.

An intelligent fallback search mechanism is included so that user queries do not
terminate in a dead end when no exact match exists.

### 1.3 Project context

The system ingests receipt images, extracts text using OCR, recognises financial
entities using a hybrid NER engine, stores the results in a searchable database,
and provides a graphical interface for upload, search and visual verification.

Every component runs locally. No API key, no access token and no paid
subscription is used, and no receipt leaves the machine — a material property
for a system holding financial documents.

---

## 2. Research background

### 2.1 Evolution of document information extraction

Historically, key-value extraction from receipts relied on template matching and
rule-based systems built on regular expressions (Yang et al., 2022). These are
effective for documents with consistent layouts, such as standardised invoices,
but demonstrate severe limitations on retail receipts (Yoon et al., 2024).
Receipts vary greatly between merchants, frequently featuring misaligned text,
faded thermal print and unpredictable spatial structure. Purely rule-based
systems consequently lack generalisability and require manual updating whenever
a new receipt format appears (Bhattacharyya et al., 2025).

### 2.2 State-of-the-art techniques in named entity recognition

Recent literature has shifted towards machine learning and sequence-labelling
models (Yang et al., 2022). A conventional pipeline processes a document in two
isolated steps: text is extracted by OCR, then a text-only NER model classifies
the entities. This improves upon regular expressions but still struggles on
receipts, because meaning depends heavily on spatial position (Yoon et al.,
2024) — a sequence of digits may be a tax amount, a subtotal or a grand total
depending on where it is printed.

The current state of the art in Visual Document Understanding relies on
multi-modal models that process text, layout and visual features simultaneously.
LayoutLM integrates spatial coordinates directly into the transformer (Xu et
al., 2019), allowing it to learn that a number printed beneath the word "Total"
is probably a TOTAL entity. OCR-free models such as Donut bypass character
recognition entirely, mapping document images directly to structured JSON (Kim
et al., 2022).

A third approach, adopted here, uses a general-purpose instruction-tuned
language model as one of two extraction engines. Rather than classifying spans,
the model reads the OCR text and emits a structured record. This is attractive
for receipts because the model can *repair* corrupted input from context — a
capability neither a regular expression nor a span-tagger possesses.

### 2.3 Comparison of candidate approaches

| Approach | Strengths | Weaknesses | Decision |
|---|---|---|---|
| Rule / lexicon | Exact, instant, free, fully explainable, cannot hallucinate | Brittle; every new layout needs new rules | **Adopted as one layer** |
| Statistical sequence labelling (CRF) | Lightweight; models label transitions | Requires annotated in-domain data; heavy feature engineering | Rejected — no token-level annotations available |
| Transformer token classification (BERT, XLM-R, Malaya `ner-t5`) | Strong general NER; good multilingual transfer | Tag set fixed at training time — no TOTAL tag | Rejected — see §2.5 |
| Layout-aware models (LayoutLMv3) | State of the art on CORD and SROIE; understands 2-D structure | Requires token-level labelled receipts to fine-tune | Rejected — labels unavailable within project scope |
| OCR-free (Donut) | No OCR error propagation | Emits JSON with **no character offsets**, so highlighting is impossible | Rejected — see §2.4 |
| Generative LLM (Qwen2.5-Instruct) | Reads layout implicitly; arbitrary schema; repairs OCR noise | Can hallucinate; slower; returns values without offsets | **Adopted as second layer** |

The instructive comparison is between the last two rows. A span-based tagger can
only return text that is literally present. On the receipt studied here, OCR
produced `ToTaL 22-60`; a span tagger's best possible answer is the string
`22-60`. The language model returns `22.60`, having repaired the misread
separator from context. Conversely, the language model returned an address for a
receipt that has none, which a regular expression would never do.

That asymmetry is the argument for a hybrid architecture, and it is the central
design claim of this project: **use each method where it is strong, and use
their agreement as the confidence estimate.**

### 2.4 Why a vision-language model was not used

The chosen dataset is packaged for VLM instruction tuning, so an OCR-free model
is the obvious alternative. It was rejected for two concrete reasons:

1. **Highlighting.** A VLM emits `{"total": "22.60"}` with no indication of
   where on the page that came from. The assignment requires returning the
   document with the entity highlighted, which needs character offsets and pixel
   coordinates.
2. **Hardware.** A vision-language model strong enough to outperform this
   pipeline is of the order of 7 billion parameters; at half precision that
   exceeds the 8 GB of video memory available.

The OCR-plus-NER route preserves a complete provenance chain — pixel polygon →
character span → normalised value — which is precisely what makes the search and
highlighting requirements implementable.

### 2.5 Bahasa Melayu resources, and why Malaya was not used

The assignment hint refers to Malaya, the principal open Malay NLP toolkit. The
project began with Malaya and it could not be made to work. The failure is
environmental rather than a misuse of the API, and it materially shaped the
design.

```
File "malaya/entity.py", line 159, in huggingface
File "malaya/supervised/huggingface.py", line 22, in load
    args = inspect.getargspec(class_model)
AttributeError: module 'inspect' has no attribute 'getargspec'
```

`inspect.getargspec()` was deprecated in Python 3.0 and **removed in Python
3.11**. The development environment runs Python 3.14.7. Malaya 5.1.1 calls it
inside its generic model loader, so every Malaya model fails at load time,
before any network access. Shimming the removed function clears that specific
line, but three problems remain:

1. Malaya 5.1.1 targets `transformers` 4.x; this environment has 5.15.0.
2. Malaya's `Tagging` class wraps a custom `T5ForTokenClassification` reading a
   non-standard `config.vocab` field.
3. Malaya's tag set is generic — `location`, `organization`, `person`,
   `quantity`, `time`. It has no TOTAL, SUBTOTAL or TAX tag, so even a working
   Malaya would leave the central receipt problem unsolved.

Point 3 is decisive: the incompatibility is an inconvenience, but the tag set is
a conceptual mismatch.

Consequently, **Malay coverage is not delegated to any model.** The bilingual
field vocabulary lives in an explicit, auditable lexicon (`dms/lexicon.py`), so
Malay support is a property of the *system* rather than a hope about the
weights. If the language model is disabled entirely, a Malay receipt still
parses correctly.

### 2.6 Limitations of existing tools and proposed improvements

While extraction models such as LayoutLM and Donut exist, they are extraction
engines rather than complete Document Management Systems. This project
contributes a holistic architecture with three differentiators:

1. **Hybrid NER with arithmetic arbitration.** Two independent readers — a
   deterministic bilingual rule layer and a local language model — process the
   same text, and their disagreements are settled by the receipt's own
   arithmetic rather than by an arbitrary precedence rule.
2. **Intelligent fallback search.** If a query yields no exact match, the system
   expands contextually, combining lexical similarity (for typographical errors)
   with dense vector similarity (for semantically related terms).
3. **Visual entity grounding.** NER results are mapped back to spatial
   coordinates, so a query returns the original document with the entity
   highlighted, drastically reducing manual verification time.

---

## 3. Methodology

### 3.1 Data collection and ingestion

The dataset is `amohseni/receipt_VLM_information_extraction` from Hugging Face,
comprising **870 training and 97 test** Malaysian receipt images. Each record
contains an image and a ground-truth annotation of four fields:

```python
{'company': 'SWC ENTERPRISE SDN BHD',
 'date': '08/01/2018',
 'address': 'NO. 5-7, JALAN MAHAGONI 7/1, SEKYSEN 4, BANDAR UTAMA, 44300 BATANG KALI, SELANGOR.',
 'total': '8.00'}
```

These four fields match the ICDAR 2019 SROIE task schema, the standard benchmark
for receipt information extraction, which makes the results interpretable
against published work. The ground truth itself contains OCR-era artefacts
(`SEKYSEN` for `SEKSYEN`), which is why fuzzy match is reported alongside exact
match in §5.

The ingestion layer accepts JPEG, PNG, BMP, TIFF and WebP, from mobile camera
captures or desktop scanners, and handles variation in dimensions, aspect ratio
and orientation. A utility (`run_dms.py export`) converts the cached dataset into
ordinary image files with a `labels.json`, so the corpus can be inspected,
extended with the user's own receipts, and processed as an ordinary folder.

### 3.2 Data preprocessing

Raw inputs undergo deterministic computer-vision treatment before extraction.
**No machine learning is involved at this stage** — it is entirely classical
image processing.

**Upscaling.** Images whose longest side is under 1000 px are enlarged to that
size by **bicubic interpolation**, which fits a cubic polynomial across a 4×4
neighbourhood and preserves stroke edges better than bilinear. Images are never
*downscaled*: fixed-canvas resizing (224×224, 448×448) is a requirement of
LayoutLM and Donut, which need constant tensor dimensions. Because this pipeline
is OCR-driven, native resolution is preserved, since shrinking destroys exactly
the faint strokes a faded receipt depends upon.

**Adaptive variant selection.** No single filter wins on every receipt — thermal
paper, faded ink and glossy photographs each prefer a different one. The system
therefore generates five candidates and lets the OCR engine decide:

| Variant | Treatment | Suited to |
|---|---|---|
| `raw` | none beyond upscaling | clean, well-lit photographs |
| `gray_otsu` | greyscale → Gaussian blur (5×5) → **Otsu threshold** | flat, high-contrast scans |
| `adaptive` | median blur (3×3) → **adaptive Gaussian threshold** (block 31, C=10) | uneven lighting, shadow |
| `clahe_sharp` | **CLAHE** (clip 2.5, 8×8 tiles) → unsharp mask (1.6 × original − 0.6 × blurred) | faded thermal ink |
| `deskew_otsu` | minimum-area-rectangle rotation, then Otsu | tilted captures |

The techniques, precisely:

* **Otsu's method** automatically selects the single global threshold that
  maximises between-class variance between ink and paper, computed from the
  image histogram.
* **Adaptive thresholding** computes a *different* threshold per pixel from a
  Gaussian-weighted 31×31 neighbourhood minus a constant. This is what handles a
  shadow falling across half the receipt, where one global threshold cannot
  serve both halves.
* **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) equalises each
  tile of an 8×8 grid independently; the clip limit caps histogram peaks before
  equalising, preventing the amplification of noise in blank areas.
* **Unsharp masking** subtracts a blurred copy to isolate high-frequency detail,
  then re-adds it amplified, sharpening edges.

The winning variant is the one returning the greatest **sum of word
confidences** — the expected number of correctly read words. This rewards
reading *more* text and reading it *well*; mean confidence alone would favour a
variant that found a single crisp word and missed the rest of the receipt.

**Orientation alignment.** A skew-detection routine computes the minimum-area
rectangle enclosing the text mass and rotates the image upright. The correction
is applied only for a genuine, modest tilt (0.4°–20°); a larger measurement is
treated as unreliable and left alone. Rotation uses `BORDER_REPLICATE` so edge
pixels are repeated rather than filled black, avoiding false ink in the corners.

### 3.3 Line reconstruction

This stage is easy to omit and its omission is fatal. OCR returns loose boxes;
the naïve response is to join them into a single string. But **on a receipt the
layout is the meaning**:

```
JUMLAH BESAR              60.31
```

`JUMLAH BESAR` on the left and `60.31` on the right belong to the same line.
Flattening the boxes destroys that association permanently. The system regroups
boxes into visual lines by comparing vertical centres against 0.6 × the median
glyph height, then orders left-to-right within each line. Each word's character
offset in the reconstructed text is recorded, which is what later allows a value
to be traced back to its pixels.

### 3.4 NLP techniques and model architecture

**Three neural models are used, and only one of them is a language model.** The
distinction matters when describing the system:

| Model | Role | Type | Generative? |
|---|---|---|---|
| EasyOCR (CRAFT + CRNN) | pixels → text | vision networks | No |
| Qwen2.5-1.5B-Instruct | text → JSON record | **LLM** | **Yes** |
| BGE-M3 | text → 1024-dim vector | embedding model | No |

**Optical character recognition.** The OCR engine processes the optimised image
and captures each text fragment together with its spatial bounding box. Both
Bahasa Melayu and English are enabled; in EasyOCR the two share a single Latin
recognition model, so bilingual support costs nothing. Two engines are supported
and were compared empirically (§5.2).

**Layer A — the rule reader.** A lexicon-driven, layout-aware extractor. Money
labels are matched longest-first, so `JUMLAH KECIL` (subtotal) is never
swallowed by `JUMLAH` (total). The amount is taken from the right of the label
on the same line, falling back to the following line for stacked layouts. Dates
and times are masked out before amounts are sought, so `10-05-2017` is never
read as money. OCR damage is repaired: `22-60`, `7 - 40`, `1,28` and `2=50` all
parse correctly, because `-`, `,` and `=` are routine misreadings of the decimal
point.

**Layer B — the language model.** `Qwen2.5-1.5B-Instruct` (Apache-2.0, ~3.1 GB)
reads the same text and returns a structured JSON record. Two prompt decisions
follow from measurement: prompt tokens are cheap while generated tokens are
expensive, so the prompt spends freely on two worked examples — one clean Bahasa
Melayu receipt and one OCR-corrupted English one — while demanding compact
single-line JSON in return; and decoding is greedy, so extraction is
reproducible.

**Entity schema.** Sixteen receipt-specific types, each annotated with the coarse
CoNLL class it specialises so the DMS remains interoperable:

`MERCHANT` (ORG) · `ADDRESS` (LOC) · `PHONE` · `TAX_ID` · `INVOICE_NO` · `DATE` ·
`TIME` · `ITEM` (PRODUCT) · `SUBTOTAL` · `TAX` · `TOTAL` · `PAID` · `CHANGE` ·
`CURRENCY` · `PAYMENT_METHOD` · `CASHIER` (PERSON)

### 3.5 From model output to database row

The language model emits **text**, not data. The path from that text to a stored
row is where most of the engineering lives, and none of it is visible from
outside. Traced on the sample receipt (`tools/trace_llm_to_db.py` reproduces this):

**Step 1 — the raw generation.** 322 characters of text that resemble JSON, and
may not be valid.

**Step 2 — parsing.** `parse_json_object()` recovers a Python dictionary,
repairing code fences, trailing prose, trailing commas, truncation, and — an
observed failure — the model answering with a *calculation*
(`"unit_price": 54.40 / 3`) instead of a number. If the reply still will not
parse, readable top-level fields are salvaged rather than discarding everything.

**Step 3 — validation and grounding.** Each field passes four checks before
becoming an entity:

1. **Normalise** — the JSON number `22.6` becomes the string `"22.60"`.
2. **Veto implausible values** — an "address" that does not look like an address
   is dropped (on the sample receipt the model offered the GST registration
   number); an amount printed nowhere on the receipt is dropped. Strings such as
   `"None"`, `"N/A"` or `"tiada"`, which models emit instead of a genuine null,
   are also discarded.
3. **Re-anchor to the page** — the model returns a *cleaned* value but the page
   holds *damaged* text. Fuzzy alignment maps `22.60` back onto the printed
   `22-6`, `21:51` onto `21;51`, and `CASH` onto `CAsh`, recovering character
   offsets. Without this, highlighting would be impossible.
4. **Wrap as an entity** with type, value, surface form, offsets, confidence and
   source.

**Step 4 — merge and validate** (§3.6), then **step 5**, mapping character
offsets through the OCR word table to pixel boxes.

**Step 6 — insert.** One row in `documents`; one row per entity in `entities`,
each carrying four distinct columns for four distinct purposes:

| Column | Purpose |
|---|---|
| `value` | cleaned — for display and arithmetic |
| `value_norm` | lowercased, punctuation stripped — **what SQL matches against** |
| `start` / `end` | position in `ocr_text` — **what makes highlighting work** |
| `embedding` | 1024 floats — **only for natural-language entity types** |

The model's JSON is therefore never inserted directly. It is parsed, validated,
corrected and grounded first.

### 3.6 Validation and self-consistency

Agreement between the two independent readers raises confidence to 0.97; a
single source yields 0.70–0.80; unresolved disagreement drops to 0.55 and is
recorded as a warning for human review. Four validators then run:

1. **Arithmetic arbitration.** When the readers disagree on subtotal, tax or
   total, usually exactly one combination satisfies `subtotal + tax = total`.
   Choosing it converts an arbitrary tie-break into a checkable decision. Among
   competing solutions, the one disturbing the *least confident* readings is
   preferred, so a value both layers agreed upon is not overwritten to satisfy a
   figure only one layer proposed.
2. **Evidence-backed recovery.** If `total − subtotal` equals an amount actually
   printed on the receipt, that value is adopted as the tax. Nothing is invented:
   a figure with no counterpart in the text is left alone.
3. **Plausibility bounds.** Malaysian GST/SST is 6–10%, so an amount labelled as
   tax at 94% of the total is not a tax. Where the subtotal slot is empty it is
   moved there rather than discarded, and the true tax is then re-derived.
4. **Tax-inclusive detection.** When the receipt states "Price Inclusive Of GST"
   or "harga termasuk cukai", `subtotal + tax = total` is deliberately false, and
   the check is suppressed rather than raising a false alarm.

5. **Promotion of the rejected alternative.** When a value fails the
   plausibility bound, the figure proposed by the layer that lost the merge is
   examined before the field is abandoned. A money value from the language model
   has already been checked against the printed text, so adopting it recovers
   evidence the routing table discarded rather than inventing anything.

**Worked example.** On `sample_receipt.jpg` (*Kedai Papan Yew Chuan*, a timber
merchant) OCR damaged two separate figures, and the two are repaired by different
mechanisms:

```
! TAX: rule='80.00' vs llm='4.80' -> kept rule
! TOTAL: 4.80 -> 84.80 (paid - change, and printed on the receipt)
! TAX 80.00 is 94% of the total - implausible; using 4.80 from the llm layer instead
```

The **total** was misread as `4.80` by *both* layers, so no disagreement arose to
arbitrate. It was caught instead by the payment identity: the receipt records
`84.80` tendered and `0.00` change, and `paid − change` is an independent
statement of the amount due, produced by the till rather than by either
extractor. The implied figure was confirmed present in the text before adoption.

The **tax** failed on magnitude: at 94% of the total it cannot be a Malaysian
GST or SST charge. Rather than deleting the field, the system promoted the
language model's rejected `4.80`, which is 5.7% of the total and therefore
plausible.

Neither layer alone produces this record, and neither does simple voting between
them — the corrections come from arithmetic relationships that hold on a receipt
regardless of what either extractor believes.

### 3.7 Storage and retrieval design

**Storage strategy.** A single SQLite database holds both the relational and the
vector layers.

* **Relational.** `documents` stores the OCR text, the winning preprocessing
  variant, the detected language, the flattened record and every word box.
  `entities` stores each entity with its normalised value, surface form,
  character offsets, confidence, source and semantic vector.
* **Vector.** Entity embeddings are stored as float32 blobs in the same table and
  compared by brute-force cosine similarity.

A dedicated vector database (FAISS, ChromaDB) was evaluated and **deliberately
not adopted**. At this scale it would add a dependency for no measurable gain:
10,000 entities at 1024 dimensions is a 40 MB matrix and a single matrix
multiplication — microseconds. Approximate nearest-neighbour indexing pays off in
the millions of vectors, not the thousands. Keeping everything in one SQLite file
also means the database is a single portable artefact with no server process.

Values are normalised on ingestion so the search layer never re-parses: dates
become `YYYY-MM-DD`, times become 24-hour `HH:MM` (`10:25 pagi` → `10:25`), money
becomes plain `60.31`, and both `TUNAI` and `CASH` become `CASH`.

**Semantic indexing.** Only entity types whose value is natural language —
merchant, address, item, cashier — are embedded. This was learned empirically:
embedding money and dates as well meant a search for "Kota Kinabalu" returned
`PAID 30.00` ranked above the addresses, because `"30.00"` still produces a
vector sitting at some arbitrary distance from every query. Amounts, dates and
identifiers are served exactly by the lexical stages, which is where they belong.

**The search cascade.** Importantly, **the search layer never invokes the
language model.** Retrieval is a database query plus vector similarity:

| Mode | Mechanism | Condition |
|---|---|---|
| `exact` | SQL `WHERE value_norm = ?` | normalised value equals the query |
| `partial` | SQL `LIKE '%query%'` | query is a substring of a value |
| `similar` | `difflib` word-window ratio **and** cosine similarity, merged | neither of the above matched |
| `type_fallback` | gazetteer lookup | query is a Malaysian place name not held |
| `empty` | — | nothing above threshold |

The two notions of closeness in `similar` capture different phenomena, which is
why both run: lexical similarity rescues a *typographical error* ("Kuala Lumpor"
finds "Kuala Lumpur"), while semantic similarity rescues *a different word for a
related thing* ("nasi ayam" finds "Add Chicken", sharing no characters at all).

**Visual entity grounding.** Because every entity retains its offsets, a hit is
shown in place. `highlight_html` emits `<mark class="ent ent-ADDRESS">` so each
type can be coloured, and overlapping spans are resolved so markers never
interleave. For image overlay, each entity carries `meta["bbox"]` in original
image pixels.

### 3.8 User interface

A Streamlit application (`app.py`) provides four pages: **Process a receipt**
(upload, with every pipeline stage displayed including the preprocessing scores,
the entity table with its source column, the validation warnings, and the entity
boxes drawn on the image); **Search**; **Database**; and **How it works**.

Building the interface proved worthwhile beyond usability — it exposed two
defects invisible to command-line use, recorded in §8.3.

---

## 4. Implementation environment

### 4.1 Hardware specification

The specification below is the actual development machine, verified directly
rather than estimated.

| Component | Specification | Operational role |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060, 8 GB VRAM (driver 610.88) | Half-precision inference for the extraction LLM (peak 3.4 GB) and GPU-accelerated OCR |
| CPU | Intel Core i5-10400F, 6 cores / 12 threads @ 2.90 GHz | Image preprocessing, sentence embedding, SQLite operations |
| RAM | 32 GB DDR4 | Python stack, OCR buffers, in-memory vector matrix |
| Storage | NVMe SSD | Model weights (~6 GB), dataset cache (478 MB), database |
| OS | Windows 11 Pro (build 26200) | — |

The 8 GB video-memory constraint shaped two decisions. First, the extraction
model is capped at 3 billion parameters in half precision (a 3B model peaks at
6.8 GB, leaving little headroom). Second, the sentence embedding model runs on
**CPU** by default, so ingestion cannot exhaust video memory while the language
model is resident; embedding on CPU costs roughly 0.1 s per query.

No INT8 quantisation is used; float16 on GPU and float32 on CPU proved
sufficient. Note that bfloat16 is deliberately avoided on CPU, because consumer
Intel processors without AVX512-BF16 emulate it and end up slower than float32.

### 4.2 Software framework

| Layer | Library | Version |
|---|---|---|
| Language | Python | 3.14.7 |
| Deep learning | PyTorch | 2.13.0+cu126 |
| Model hosting | Hugging Face Transformers | 5.15.0 |
| OCR (primary) | Tesseract via pytesseract, `msa`+`eng` | 5.5.3 |
| OCR (fallback) | EasyOCR (CRAFT detector + CRNN recogniser) | 1.7.2 |
| Image processing | OpenCV | 5.0.0 |
| Embeddings | sentence-transformers (BGE-M3) | 6.0.0 |
| Interface | Streamlit | 1.62.0 |
| Storage | SQLite with FTS5 | 3.53.1 |
| Dataset | Hugging Face `datasets` | 5.0.1 |

All components are free and open source, and everything runs offline after the
initial model download.

---

## 5. Evaluation and results

### 5.1 Metrics used, and metrics deliberately not used

Metrics must be computable from the available ground truth. The dataset
annotates four fields per receipt; it does **not** provide a transcript of each
receipt, nor ground-truth bounding boxes. Consequently:

| Metric | Used? | Reason |
|---|---|---|
| Field exact-match accuracy | ✅ | Directly computable from the four annotations |
| Field fuzzy-match accuracy | ✅ | The ground truth contains OCR-era typos; fuzzy match is fairer |
| Coverage ("found") | ✅ | How often the field was populated at all — a recall proxy |
| Field recoverability (OCR ceiling) | ✅ | Introduced here; see below |
| Recall@1, Recall@5, MRR | ✅ | Standard IR metrics for the fallback search |
| **CER / WER** | ❌ | Requires a ground-truth transcript of every receipt; none exists |
| **IoU** | ❌ | Requires ground-truth bounding boxes; none exist |
| **TED / nTED** | ❌ | Applies to OCR-free tree-generation architectures (Donut) |
| **NDCG@K** | ❌ | Requires graded relevance; the constructed task has binary relevance, for which MRR and Recall@K are appropriate |

Claiming metrics that cannot be computed from the available data would be
misleading, so they are excluded and the omission justified.

**Field recoverability** is introduced in place of CER/WER. It measures, of the
four annotated values, how many survive OCR well enough to be located in its
output at all. This is the *ceiling* on downstream NER accuracy — no extraction
method can recover a total that OCR never read — and it requires no transcript.

### 5.2 OCR engine comparison

Twenty receipts, both engines offered identical preprocessing variants. This
is a *recoverability* measurement - whether the field survived into the OCR
text at all - not an accuracy one; the end-to-end comparison over 120
training receipts follows in §5.4.

| Engine | company | date | address | total | **mean** | s/receipt | mean conf. |
|---|---|---|---|---|---|---|---|
| EasyOCR (`ms`+`en`) | 95.0% | **80.0%** | 85.0% | 90.0% | 87.5% | **5.2** | 0.804 |
| Tesseract (`msa`+`eng`) | **100.0%** | 75.0% | **100.0%** | 90.0% | **91.2%** | 12.0 | **0.865** |

This contradicted the initial assumption. Tesseract is commonly described as
brittle on thermal receipts, and that expectation drove the original choice;
measurement did not support it. Tesseract recovered *every* company and address,
while EasyOCR lost 15% of addresses — long address lines are where Tesseract's
line-oriented LSTM has the advantage. EasyOCR was superior only on dates.

Because recoverability is a ceiling rather than an outcome, the end-to-end
consequence was then measured on the 120-receipt training split (§5.4): Tesseract
lifted exact-match accuracy from 51.7% to 63.3%. **Tesseract is therefore the
default.** That decision was taken on training data only, and it is the largest
single improvement recorded in this report.

**Both engines remain supported**, selectable by command-line flag, by the
`DMS_OCR_ENGINE` environment variable, or from a dropdown in the interface.
EasyOCR is retained deliberately rather than as a legacy option: Tesseract is a
separate system binary rather than a pip dependency, so on a machine without it
the factory falls back to EasyOCR with a printed warning instead of failing. A
fresh clone therefore still runs, at reduced accuracy — an availability decision,
not an accuracy one.

The timing column is not like-for-like: EasyOCR runs on the GPU here while
Tesseract is CPU-only and was additionally run under two page-segmentation modes
per variant, performing six passes to EasyOCR's three.

### 5.3 Experimental protocol — avoiding test-set leakage

An earlier draft chose the merge routing by inspecting scores on the *test*
split. That is test-set leakage: a figure obtained that way is not an estimate of
performance on unseen receipts. The dataset provides 870 training receipts that
no decision had touched, so the protocol was redone properly
(`tools/tune_on_train.py`): search on train only, freeze, then score the test split
once.

**Step 1 — the search, on training data (120 receipts):**

| Routing (which layer owns the field on a disagreement) | Train exact | Train fuzzy |
|---|---|---|
| Rules own both merchant and address | 62.3% | 73.5% |
| **LLM owns address** | **63.3%** | **77.7%** |
| LLM owns merchant | 59.6% | 70.8% |
| LLM owns both | 60.6% | 75.0% |

**Step 2 — frozen configuration, evaluated once on the 97-receipt test split:**

| Split | Exact | Fuzzy | n |
|---|---|---|---|
| Train (used for selection) | 63.3% | 77.7% | 120 |
| **Test (scored once, unseen)** | **63.6%** | **77.3%** | 97 |

Two conclusions:

* **The routing decision was independently confirmed.** The training data chose
  the same configuration previously arrived at by inspecting test scores. The
  earlier choice happened to be correct, but it is only *now* defensible.
* **The generalisation gap is −0.2%** — the test split scored marginally *higher*
  than the training sample. A gap that small, and negative, is evidence against
  overfitting: the configuration transfers to receipts it has never seen.

Every decision reported here — the OCR engine (§5.2), the merge routing above,
and the rule repairs of §8 — was made against the training split alone. The test
split was read once, after everything was frozen.

**A coincidence worth pre-empting.** An early draft of this project reported
**63.3%** exact, obtained by selecting the routing on the test split itself and
scoring 30 receipts. That figure was inflated by leakage and was withdrawn; the
honest replacement, on all 97 test receipts, was **56.1%**. Adopting Tesseract —
a decision taken entirely on training data — has since raised the honest figure
to **63.6%**, which lands within half a point of the discredited number.

The two are unrelated. The first was a biased estimate over a favourable
30-receipt subset; the second is an unbiased estimate over the full split, and it
rose because the reader improved, not because the measurement moved. The
coincidence is noted here so that it cannot be mistaken for the earlier error
having been quietly reinstated.

### 5.4 NER ablation

The entire test split — all 97 receipts — with Tesseract and
`Qwen2.5-1.5B-Instruct`. Percentages are exact / fuzzy / coverage. The
configuration was fixed by §5.3 *before* this table was produced.

| Field | Rules only | LLM only | **Hybrid** |
|---|---|---|---|
| company | **58.8 / 77.3 / 100.0** | 48.5 / 68.0 / 100.0 | **58.8 / 77.3 / 100.0** |
| date | 77.3 / 77.3 / 87.6 | 75.3 / 75.3 / 100.0 | **80.4 / 80.4 / 100.0** |
| address | **41.7** / 63.5 / 99.0 | 26.0 / 57.3 / 99.0 | 36.5 / **72.9** / 99.0 |
| total | 77.3 / 77.3 / 93.8 | 58.8 / 58.8 / 100.0 | **78.4 / 78.4 / 95.9** |
| **Overall** | **63.8** / 73.9 | 52.2 / 64.9 | 63.6 / **77.3** |

Four observations, and the first is uncomfortable:

1. **On exact match the rule layer alone equals the hybrid** — 63.8% against
   63.6%, a gap of 0.2 points across 97 receipts, which is well inside sampling
   noise. This was *not* true earlier in the project: with the weaker OCR engine
   the hybrid led the rule layer by three points (53.2% to 56.1%). Improving the
   reader removed most of what the language model had been repairing. Reported
   plainly because it is the single most important finding in this section, and
   because it contradicts the assumption the architecture was built on.
2. **The hybrid retains a clear advantage on the other two axes.** It is 3.4
   points ahead on fuzzy match (77.3% against 73.9%), and the margin on `address`
   is 9.4 points — so where it is wrong, it is substantially closer to right. It
   is also more *complete*: it finds a `date` on 100% of receipts against the rule
   layer's 87.6%, and a `total` on 95.9% against 93.8%. In a searchable DMS this
   matters directly, because a field that was never extracted is invisible to
   every query, and scores zero on exact and fuzzy alike.
3. **The language model alone remains the weakest configuration** — 52.2%
   against the rule layer's 63.8%. This deserves stating plainly, because the
   fashionable assumption is the reverse. Its value here is as a second opinion
   under arbitration, not as a standalone extractor.
4. **The hybrid is not uniformly better.** On `address` exact it scores 36.5%
   against the rule layer's 41.7%. The training-split search nevertheless
   selected that routing, because it wins the macro average on training data.
   The configuration was left unchanged after the test split was scored: revising
   it now, in the light of a test result, would reintroduce exactly the leakage
   §5.3 exists to eliminate. It is reported rather than concealed.

**The honest summary** is that the language model has ceased to be an accuracy
win and become a robustness win — better fuzzy quality, better coverage, no
improvement in exact match. On a harder corpus, with poorer printing or fewer
printed labels, the gap would be expected to reopen; §9 treats this as the
principal threat to the architecture's justification.

**The OCR engine comparison** that led to Tesseract being adopted, measured on
the 120-receipt **training** split so that the choice never touched test data:

| Configuration | Train exact | Train fuzzy | s/receipt |
|---|---|---|---|
| EasyOCR + hybrid | 51.7% | 65.8% | **5.2** |
| **Tesseract + hybrid** | **63.3%** | **77.7%** | 7.7 |

An 11.6-point gain for 2.5 seconds per receipt. The improvement is concentrated
in the two fields that depend most on reading long, densely printed lines:
`address` rose from 15.8% to 35.8% and `company` from 41.7% to 60.8%. This single
change contributed more than every rule refinement in §8 combined, which is the
clearest evidence in this report that **OCR quality, not extraction logic, is the
binding constraint on the task**.

### 5.5 Extraction model comparison

Four candidates, each unloaded before the next was loaded (30 receipts).

| Model | LLM only | Hybrid exact | Hybrid fuzzy | s/receipt | Peak VRAM |
|---|---|---|---|---|---|
| *(rules only)* | — | 60.8% | 69.2% | **0.0** | **0.0** |
| **Qwen2.5-1.5B-Instruct** | 51.7% | **63.3%** | 70.8% | ~12 | 3.4 GB |
| Qwen3-1.7B | 50.8% | 61.7% | 67.5% | 14.7 | 4.3 GB |
| Qwen2.5-3B-Instruct | **55.0%** | 62.5% | **74.2%** | 21.3 | 6.8 GB |
| Malaysian-Qwen2.5-3B-Instruct | 47.5% | 62.5% | 72.5% | 53.4 | 7.3 GB |

1. **Scaling the model up did not improve the system.** Qwen2.5-3B reads better
   in isolation (55.0% against 51.7%) yet its hybrid score is no higher — the
   rule layer and validators already supplied what the smaller model missed.
2. **The Malaysian-tuned model was the weakest** and by far the slowest. Domain
   tuning for Malaysian *language* does not transfer to structured *extraction*;
   this project's Malay coverage comes from an explicit lexicon.
3. **The spread is within noise.** One field is 0.83 points at this sample size.

**Decision: Qwen2.5-1.5B-Instruct** — fastest, smallest, statistically
indistinguishable from models four times its size.

### 5.6 Embedding model comparison

A retrieval task built from the dataset's own annotated addresses: retrieve the
addresses containing a given Malaysian city (300-address corpus, 22 queries).

| Model | Recall@1 | Recall@5 | MRR | Dimensions |
|---|---|---|---|---|
| **Lexical baseline** | **100.0%** | **100.0%** | **1.000** | — |
| BGE-M3 | 86.4% | 95.5% | 0.903 | 1024 |
| multilingual-e5-base | 81.8% | 86.4% | 0.848 | 768 |
| multilingual-e5-small | 68.2% | 86.4% | 0.777 | 384 |
| paraphrase-multilingual-mpnet | 27.3% | 31.8% | 0.305 | 768 |
| paraphrase-multilingual-MiniLM | 9.1% | 22.7% | 0.174 | 384 |

**The lexical baseline outperformed every embedding model.** This is not a
failure of embeddings but a property of the task: "find the addresses containing
*Kuala Lumpur*" is a substring problem, which substring matching solves exactly.
It is a useful corrective to the assumption that semantic search is a strict
upgrade.

It also settles the architecture. Semantic search must not *replace* lexical
matching; it must sit *behind* it. In the final cascade, exact and substring
matching answer first, and semantic similarity runs only when they find nothing —
precisely the case the assignment cares about.

Two further observations. The two models most commonly reached for by default,
MiniLM and mpnet, are close to useless here; Malay place names fall outside their
paraphrase training. And **no model can reject a nonsensical query by score
alone**: separation gaps between gibberish and genuine queries were −0.002
(e5-small), −0.147 (MiniLM), +0.003 (e5-base), −0.180 (mpnet) and −0.057
(BGE-M3), all at or below zero.

**Decision: BGE-M3**, the best measured, with **per-model** thresholds recorded
in the registry — the models place similarity on incompatible scales (E5
compresses everything into 0.75–0.90 while BGE-M3 spreads across 0.3–0.6), so a
single global cut-off would be meaningless.

### 5.7 Functional verification

`tools/verify_end_to_end.py` starts from an **empty** database and asserts every link
in the chain. All checks pass:

| Requirement | Evidence |
|---|---|
| NER on documents | 12 entities from one receipt; 16 types available |
| Character offsets | 10 of 12 aligned |
| Pixel boxes | 10 of 12 |
| Persisted to database | document, entities and word boxes read back identically |
| Vectors stored | dimension matches the loaded model |
| Search an entity | `exact` on a total, `partial` on a merchant word |
| **Semantic search** | 8/8 meaning-only probes matched |
| Similar entity when absent | "Kota Kinabalu" → returns held locations |
| Nonsense rejected | "qwertyuiop zxcvbnm" → `empty` |
| Highlighting | marker in text; `<mark>` in HTML |

The semantic results demonstrate cross-lingual retrieval, with no shared
characters between query and match:

| Query | Score | Retrieved |
|---|---|---|
| `chicken and rice` | 0.713 | "Add Chicken" |
| **`nasi ayam`** (Malay) | 0.583 | "Add Chicken" |
| **`minuman panas`** (Malay: hot drink) | 0.597 | "Herbal Tea (Iced)" |
| **`sayur`** (Malay) | 0.575 | "Vegetable" |

An automated suite of **106 assertions** (`test_dms.py`) covers parsing, the rule
layer, span alignment, the search cascade, highlighting, JSON recovery and the
validators. Every assertion passes, and each is a regression guard for a defect
encountered during development.

---

## 6. Pseudocode

### 6.1 Main pipeline

```
ALGORITHM ProcessReceipt(image)
INPUT   image        : a receipt photograph
OUTPUT  document     : validated entities, stored and indexed

  ── Stage 1: adaptive preprocessing and OCR ──
  best_score ← −∞
  FOR EACH variant IN {raw, gray_otsu, adaptive, clahe_sharp, deskew_otsu} DO
      processed ← ApplyFilter(image, variant)
      boxes     ← OCR(processed, languages = {Malay, English})
      score     ← Σ confidence(b) FOR b IN boxes WHERE confidence(b) ≥ 0.30
      IF score > best_score THEN
          best_score ← score;  best_boxes ← boxes
      END IF
  END FOR

  ── Stage 2: rebuild visual lines (layout carries meaning) ──
  h        ← median(height(b)) FOR b IN best_boxes
  lines    ← GroupByVerticalCentre(best_boxes, tolerance = 0.6 × h)
  FOR EACH line IN lines DO SortByHorizontalPosition(line) END FOR
  text     ← JoinLines(lines)         // record char offset of every word

  ── Stage 3a: rule reader ──
  rule_entities ← ∅
  FOR EACH line IN lines DO
      label ← LongestMatchingLabel(line, BILINGUAL_LEXICON)
      IF label ≠ NULL THEN
          masked  ← MaskDatesAndTimes(line)      // else 10-05-2017 reads as money
          amounts ← FindAmounts(masked)          // repairs 22-60 → 22.60
          value   ← LastAmountRightOf(amounts, label)
          IF value = NULL THEN value ← FirstAmountOn(NextLine) END IF
          rule_entities ← rule_entities ∪ {(TypeOf(label), value, offsets)}
      END IF
  END FOR
  rule_entities ← rule_entities ∪ ExtractMerchant(lines)   // topmost branded line
                                ∪ ExtractAddress(lines)    // street/place/postcode
                                ∪ ExtractIdentifiers(lines)

  ── Stage 3b: language-model reader ──
  prompt ← SystemPrompt + FewShotMalay + FewShotEnglish + text
  reply  ← Generate(LLM, prompt, greedy = TRUE)
  record ← ParseJSONWithRepair(reply)            // arithmetic, fences, truncation
  llm_entities ← ∅
  FOR EACH (field, value) IN record DO
      IF value ∈ {"None","null","N/A","tiada"}         THEN CONTINUE
      IF field IS money AND NOT PrintedOnReceipt(value, text) THEN CONTINUE
      IF field = address AND NOT LooksLikeAddress(value)      THEN CONTINUE
      span ← LocateSpan(text, value)             // fuzzy: "22.60" onto "22-60"
      llm_entities ← llm_entities ∪ {(TypeOf(field), value, span)}
  END FOR

  ── Stage 4: merge ──
  merged ← ∅
  FOR EACH type IN ENTITY_TYPES DO
      r ← rule_entities[type];  l ← llm_entities[type]
      IF   r ≠ NULL AND l ≠ NULL AND Agree(r, l) THEN
           e ← r;  confidence(e) ← 0.97
      ELIF r ≠ NULL AND l ≠ NULL THEN
           e ← (type ∈ PREFER_LLM) ? l : r        // PREFER_LLM chosen on TRAIN
           confidence(e) ← 0.55;  RecordWarning(r, l)
      ELSE e ← Coalesce(r, l);  confidence(e) ← (e = r) ? 0.80 : 0.70
      END IF
      merged ← merged ∪ {e}
  END FOR

  ── Stage 5: validate against the receipt's own arithmetic ──
  IF NOT TaxInclusive(text) THEN
      ArbitrateByArithmetic(merged)      // pick the least disruptive balance
      FixImplausibleTax(merged)          // tax ≈ total ⇒ it is the subtotal
      RecoverTaxFromText(merged, text)   // total − subtotal, only if printed
  END IF
  RepairTotalFromPayment(merged, text)   // paid − change, only if printed

  ── Stage 6: ground every value to the page ──
  FOR EACH e IN merged WHERE aligned(e) DO
      bbox(e) ← BoundingBox(WordsOverlapping(offsets(e)))
  END FOR

  ── Stage 7: persist and index ──
  doc_id ← InsertDocument(text, boxes, fields)
  FOR EACH e IN merged DO
      id ← InsertEntity(doc_id, type, value, Normalise(value),
                        start, end, confidence, source)
      IF type(e) ∈ {MERCHANT, ADDRESS, ITEM, CASHIER} THEN
          StoreVector(id, Embed(value(e)))       // natural language only
      END IF
  END FOR
  RETURN doc_id
END ALGORITHM
```

### 6.2 Intelligent fallback search

```
ALGORITHM SearchEntities(query, type_filter)
OUTPUT  (mode, ranked_hits, suggestions)
NOTE    the language model is NOT used anywhere in this algorithm

  q ← Normalise(query)

  hits ← SELECT * FROM entities WHERE value_norm = q
  IF hits ≠ ∅ THEN RETURN ("exact", hits, ∅)

  hits ← SELECT * FROM entities WHERE value_norm LIKE '%' ‖ q ‖ '%'
  IF hits ≠ ∅ THEN RETURN ("partial", hits, ∅)

  ── neither matched: combine two independent notions of closeness ──
  candidates ← ∅
  FOR EACH v IN DistinctValues(type_filter) DO          // lexical: spelling
      s ← BestWindowSimilarity(q, v)
      IF s ≥ 0.72 THEN candidates[v].lexical ← s END IF
  END FOR

  qv ← Embed(q)                                          // semantic: meaning
  FOR EACH (v, vec) IN StoredVectors(type_filter) DO
      s ← qv · vec                                       // unit vectors ⇒ cosine
      IF s ≥ MinScore(embedding_model) THEN candidates[v].semantic ← s END IF
  END FOR

  IF candidates ≠ ∅ THEN
      RETURN ("similar", RankBy(MAX(lexical, semantic)), TopValues(candidates))
  END IF

  ── still nothing: is the query a Malaysian place we simply do not hold? ──
  IF IsMalaysianPlace(q) THEN
      RETURN ("type_fallback",
              SELECT * FROM entities WHERE type IN {ADDRESS, MERCHANT}
              ORDER BY (type = ADDRESS) DESC, confidence DESC,
              ∅)
  END IF

  RETURN ("empty", ∅, NearestValues(q))
END ALGORITHM
```

### 6.3 Highlighting

```
ALGORITHM HighlightDocument(doc_id, entity_ids)
  text  ← OcrTextOf(doc_id)
  spans ← {(start(e), end(e)) : e ∈ EntitiesOf(doc_id) ∩ entity_ids, aligned(e)}
  spans ← DropOverlaps(SortByStart(spans))   // longest span wins at a position
  out ← "";  cursor ← 0
  FOR EACH (s, e) IN spans DO
      out ← out ‖ text[cursor .. s] ‖ OPEN_MARK ‖ text[s .. e] ‖ CLOSE_MARK
      cursor ← e
  END FOR
  RETURN out ‖ text[cursor .. END]
END ALGORITHM
```

---

## 7. Algorithm analysis

Notation: `n` = words on a receipt, `L` = lines, `V` = preprocessing variants
(5), `E` = entities in the database, `D` = embedding dimension (1024), `T` =
generated tokens, `P` = prompt tokens.

### 7.1 Time complexity

| Stage | Complexity | Measured | Notes |
|---|---|---|---|
| Preprocessing | `O(V · W · H)` | ~0.4 s | Linear per pixel; five variants |
| OCR detection + recognition | `O(V · n)` | 5–12 s | **Dominant non-LLM cost** |
| Line reconstruction | `O(n log n)` | < 1 ms | Sort by vertical centre, then horizontal |
| Rule extraction | `O(L · K)` | < 5 ms | `K` = lexicon size; longest-first matching |
| LLM generation | `O(P² + T · P)` | 11–20 s | **Dominant overall cost**; attention is quadratic in prompt |
| Span alignment (fuzzy) | `O(m · n)` worst | < 10 ms | Sliding window with early exit |
| Merge + validate | `O(1)` | < 1 ms | Fixed field count; arbitration searches ≤ 2³ combinations |
| Embedding | `O(k · D)` | ~0.1 s | `k` ≈ 6 natural-language entities per receipt |
| **Total ingestion** | — | **~30 s/receipt** | ~8 s without the LLM |

**Search:**

| Stage | Complexity | Notes |
|---|---|---|
| Exact | `O(log E)` | B-tree index on `value_norm` |
| Partial | `O(E)` | Leading-wildcard `LIKE` defeats the index; FTS5 available |
| Lexical similar | `O(U · m²)` | `U` = distinct values; `difflib` is quadratic per pair |
| Semantic similar | `O(E · D)` | One dense matrix–vector product |
| Type fallback | `O(E)` | Filtered scan |

At the observed scale (`E` ≈ 10³) every branch returns in under 50 ms. The
semantic stage is a 40 MB matrix multiplication at `E` = 10⁴ — well within
brute-force territory, which justifies omitting an approximate index. The
crossover where FAISS would pay off is around `E` ≈ 10⁶.

### 7.2 Space complexity

| Structure | Complexity | At 1,000 receipts |
|---|---|---|
| OCR text + word boxes | `O(n)` per document | ~15 MB |
| Entities | `O(E)` | ~2 MB |
| Embedding matrix | `O(E · D · 4 bytes)` | ~60 MB |
| LLM weights (VRAM) | `O(1)` | 3.4 GB, loaded once |
| Embedding model (RAM) | `O(1)` | ~2.2 GB, loaded once |

The vector matrix is held in memory and invalidated on insertion. Memory grows
linearly with corpus size; at 100,000 entities the matrix reaches ~400 MB, still
tractable, beyond which a memory-mapped or approximate index becomes appropriate.

### 7.3 Design trade-offs

**Why five preprocessing variants rather than one?** Cost is linear and OCR
dominates regardless. Measured on the sample receipt, the raw image achieved mean
confidence 0.496 while Otsu achieved 0.570 and additionally recovered the invoice
number. Selecting per image converts an assumption into an evidence-based
decision, at roughly 3× the OCR cost — which `--ocr-variant` can disable when
throughput matters.

**Why only three of the five are tried by default.** Measured over 25 receipts,
widening the automatic set changes very little:

| Variants offered | company | date | address | total | mean |
|---|---|---|---|---|---|
| `otsu, raw, clahe` (default) | 92.0% | 80.0% | 84.0% | 88.0% | 86.0% |
| + `adaptive` | 92.0% | 80.0% | 84.0% | 88.0% | 86.0% |
| all five | 92.0% | 80.0% | **88.0%** | 88.0% | **87.0%** |
| `adaptive` alone | 76.0% | 64.0% | 72.0% | 88.0% | 75.0% |

Adding `adaptive` alone changes nothing, because it wins on only 1 receipt in 25.
Offering all five gains a single point — entirely on address — for 67% more OCR
time. The default therefore stays at three, and `--ocr-variant all` is available
when accuracy matters more than throughput.

The wider lesson is that **preprocessing is not where the remaining errors
live**. When a receipt was found extracting a total of 0.52 instead of 24.10, no
combination of filters fixed it; the cause was a tax-summary table matching the
word "Total", and the fix was a rule, not an image operation.

**Why brute-force vector search?** `O(E · D)` with a tiny constant beats an
approximate index with a large constant until `E` is very large. It avoids a
dependency, keeps the database a single portable file, and returns *exact*
nearest neighbours.

**Why greedy decoding?** Sampling would make extraction non-reproducible.
Identical input yields identical output, which also makes response caching sound
— reducing an ablation over merge policies from 15 minutes to 1.2 s per
configuration.

**Why retain the rule layer when an LLM is available?** §5.4 answers this
empirically: the rule layer alone outscores the language model alone. It is also
three orders of magnitude faster and cannot hallucinate. Retaining it provides a
free, deterministic baseline and a second opinion whose agreement constitutes the
confidence estimate.

---

## 8. Coding

### 8.1 Module structure

| Module | Responsibility |
|---|---|
| `dms/config.py` | Central configuration; device and dtype detection |
| `dms/schema.py` | `Entity` / `ReceiptDocument` model; the 16 entity types |
| `dms/lexicon.py` | All Bahasa Melayu and English vocabulary; place gazetteer |
| `dms/textutils.py` | Money/date parsing, OCR digit repair, fuzzy span alignment |
| `dms/ocr.py` | Preprocessing variants, EasyOCR and Tesseract backends, line reconstruction |
| `dms/rules.py` | Deterministic bilingual extractor |
| `dms/llm.py` | Local LLM loading, prompting, JSON recovery, response caching |
| `dms/ner.py` | Hybrid merge, arithmetic arbitration, validators |
| `dms/embeddings.py` | Sentence embeddings, model registry, cosine ranking |
| `dms/database.py` | SQLite schema, search cascade, highlighting |
| `dms/pipeline.py` | Orchestration |

**Executable entry points**

| Script | Purpose |
|---|---|
| `app.py` | **Graphical interface** — upload, watch each stage, search, browse |
| `run_dms.py` | CLI: `process`, `dataset`, `export`, `search`, `show`, `stats`, `compare`, `reindex`, `delete` |
| `test_dms.py` | 106 automated assertions; no model download required |
| `tools/verify_end_to_end.py` | Proves the whole chain from an empty database |
| `tools/prepare_demo.py` | Pre-flight check before a live demonstration |
| `tools/trace_llm_to_db.py` | Traces one receipt from raw model output to SQL rows |
| `tools/evaluate.py` | Field accuracy against ground truth; three-way ablation |
| `tools/tune_on_train.py` | Selects routing on train, scores test once |
| `tools/benchmark_ocr.py` | EasyOCR versus Tesseract field recoverability |
| `tools/benchmark_llm.py` | Extraction model comparison |
| `tools/benchmark_embeddings.py` | Embedding model retrieval comparison |

### 8.2 Representative code

**Adaptive variant selection** (`dms/ocr.py`) — the scoring rule that makes
preprocessing evidence-based:

```python
@staticmethod
def _score(results) -> float:
    """Expected number of correctly-read tokens.

    Summing confidences rewards reading *more* text and reading it *well*,
    which mean-confidence alone does not (a variant that found one crisp
    word would otherwise beat one that found the whole receipt).
    """
    return float(sum(c for _, _, c in results if c >= OCR_MIN_CONF))
```

**Evidence-backed tax recovery** (`dms/ner.py`) — the validator that will not
invent a value:

```python
implied = round(total - sub, 2)
printed = _value_printed(ocr_text, implied)
if printed is None:
    return                       # not on the receipt; do not invent it
text, start, end = printed
tax_ent.value = format_money(implied)
tax_ent.source = "arithmetic+text"
```

**Fuzzy span alignment** (`dms/textutils.py`) — what makes highlighting work when
the model returns a cleaned value:

```python
pos = hay.find(needle)
if pos != -1:
    return hmap[pos], hmap[pos + len(needle) - 1] + 1
# Otherwise slide a window the length of the needle across the haystack,
# so the repaired "22.60" still locates the printed "22-60".
```

### 8.3 Defects found and corrected during development

Each is covered by a regression test.

| Defect | Symptom | Correction |
|---|---|---|
| Decimal loss in money parsing | LLM value `22.6` became `226` — a 10× error in every total | Recognise a well-formed decimal *before* the damaged-separator branch |
| Identifier without a digit | `RESIT JUALAN` yielded invoice number "JUALAN" | Require at least one digit in an identifier |
| Weak address marker | `No. Items: 1` classified as the address | Require a street word, place name or postcode |
| Time/money ambiguity | Price `21.32` read as the time 21:32 | Accept a dot-separated time only when the line says it is a time |
| Arithmetic in JSON | `"unit_price": 54.40 / 3` discarded the entire extraction | Evaluate the expression; salvage top-level fields as a fallback |
| Mislabelled tax | Subtotal 21.32 reported as tax on a 22.60 total | Plausibility bound; reinterpret as subtotal |
| False arithmetic alarm | Tax-inclusive receipts flagged inconsistent | Detect "inclusive of GST" / "harga termasuk cukai" |
| Semantic index pollution | "Kota Kinabalu" returned `PAID 30.00` | Embed natural-language entity types only |
| Separator variant | `Inv;R000039737` lost the invoice number | Accept `;` as an OCR misreading of `:` |
| Literal "None" | The model wrote the *word* `"None"` as a product name, which was stored and embedded | Discard null-words (`None`, `null`, `N/A`, `tiada`) |
| GST summary hijacks the total | A receipt ending `GST Summary / Total 23.09 0.52` gave a total of **0.52**: the summary table's own "Total" row was matched, and its rightmost column is tax | Bar every total label after a tax-summary header, and recognise `TOTAL AFTER ADJ INCL GST` — the post-rounding line that is the true payable amount on Malaysian receipts |
| Total exceeding the cash tendered | OCR misread `24.10` as `94.10`; the figure was not printed anywhere, so evidence-based repair could not fire | A total above the amount tendered is impossible, so the tendered arithmetic overrides even without a printed counterpart (recorded at lower confidence) |
| Thread-bound database | The interface crashed on every interaction, because a web server answers each request from a different worker thread | `check_same_thread=False` plus a re-entrant write lock |
| Mixed embedding dimensions | Changing the embedding model left 384-dimension vectors in a database now querying with 1024, causing an unhandled shape error | Skip vectors of the wrong length, warn, advise reindexing |

The last two were found only by **building the interface**. Neither appears under
command-line use, because a CLI is single-threaded and never changes model
mid-database. This is an argument for building the user-facing layer early: it
exercises paths a script never reaches.

---

## 9. Limitations and future work

### 9.1 Limitations

1. **OCR is the ceiling.** Analysis of residual errors shows almost none were
   caused by choosing the wrong entity; they were caused by the characters being
   wrong before NER ran. Further effort belongs at the OCR stage.
2. **The language model no longer improves exact-match accuracy.** With the
   stronger OCR engine the rule layer alone scores 63.8% against the hybrid's
   63.6% (§5.4). The hybrid remains ahead on fuzzy match by 3.4 points and on
   field coverage, so it was kept, but the architecture's original justification
   — that the two layers together beat either alone on exact match — no longer
   holds on this corpus. It held before the OCR engine changed. Establishing
   whether the gap reopens on harder receipts, with poorer printing or fewer
   printed labels, is the most important open question left by this work.
3. **Residual tuning on test, now bounded.** The merge routing and the OCR engine
   were both selected on the training split and the test split scored once
   (§5.3), so 63.6% is unbiased. One smaller category was not re-searched: the
   validator rules, which are statements about how receipts arithmetically behave
   rather than parameters fitted to data. They are active during the single test
   run, so their effect is measured on unseen data.
4. **Sample size.** Headline figures cover the complete 97-receipt test split.
   The supporting experiments in §5.2, §5.5 and §5.6 remain at 20–30 receipts, so
   differences of one or two points there are not significant.
5. **Nonsense queries cannot be rejected by score alone.** Across five embedding
   models, no absolute threshold separates gibberish from genuine queries in
   general.
6. **Semantic search requires a corpus.** With one or two documents indexed,
   nearest-neighbour search has no meaningful neighbours and returns nothing.
7. **Multi-line item names.** Where a product name is printed on one line and its
   price on the next, the rule fallback captures only the fragment beside the
   numbers.
8. **No fine-tuning.** The language model is used zero-shot with few-shot
   prompting.

### 9.2 Future work

1. Repeat the §5.3 protocol for the Tesseract backend, so the engine choice is
   also selected on training data rather than assumed.
2. Derive token-level BIO labels automatically by aligning the four annotated
   values onto the OCR text using the existing span-alignment function, then
   fine-tune a multilingual token classifier on the 870 training receipts. The
   measured field recoverability of 87–91% establishes the achievable label
   quality.
3. Fine-tune the extraction LLM with LoRA on `(ocr_text → JSON)` pairs.
4. Batch LLM inference to improve GPU throughput.
5. Add a human-in-the-loop review queue driven by the existing confidence scores
   and validation warnings, which already identify the fields most deserving of
   attention.

---

## 10. Declaration of AI tool usage

In accordance with the assignment specification, the use of AI tools is declared.
An AI coding assistant (Claude, Anthropic) was used during development for code
implementation, debugging, benchmark construction and drafting of this
documentation. All design decisions, model selections and architectural choices
reported here were verified empirically by executing the benchmarks described in
§5; the measured figures are reproducible with the commands in §11. The dataset
was obtained from Hugging Face under its published terms.

*(Group members should confirm and expand this declaration in accordance with
their own contribution before submission.)*

---

## 11. Reproducing the results

```bash
# Graphical interface
streamlit run app.py

# Export the dataset to ordinary image files
python run_dms.py export --split test --out data/receipts

# Full pipeline: OCR → NER → SQLite → semantic index
python run_dms.py process data/receipts

# The four DMS functions
python run_dms.py search "Kuala Lumpur" --highlight   # search + highlight
python run_dms.py search "Kuala Lumpor"               # typographical tolerance
python run_dms.py search "Kota Kinabalu"              # similar-entity fallback
python run_dms.py show 1 --html out.html              # highlighted document

# Explanatory tools
python run_dms.py compare sample_receipt.jpg          # rules vs LLM vs hybrid
python tools/trace_llm_to_db.py                             # model output → SQL rows

# Verification and benchmarks
python test_dms.py                                    # 106 assertions
python tools/verify_end_to_end.py                           # whole chain, empty DB
python tools/tune_on_train.py --train-limit 120 --test-limit 97
python tools/evaluate.py --limit 97 --config all
python tools/benchmark_ocr.py --limit 20
python tools/benchmark_llm.py --limit 30
python tools/benchmark_embeddings.py --split train --limit 300

# Before a live demonstration
python tools/prepare_demo.py --receipts 12
```

---

## 12. References

Bhattacharyya, A. et al. (2025) *Document information extraction: a survey.*

Baek, Y. et al. (2019) 'Character region awareness for text detection',
*CVPR 2019*. [EasyOCR's detector]

Chen, J. et al. (2024) 'BGE M3-Embedding: multi-lingual, multi-functionality,
multi-granularity text embeddings through self-knowledge distillation'.

Douze, M. et al. (2025) *The FAISS library.*

Fujitake, M. (2024) 'LayoutLLM: large language model instruction tuning for
visually rich document understanding'.

Huang, Z. et al. (2019) 'ICDAR 2019 competition on scanned receipt OCR and
information extraction (SROIE)', *ICDAR 2019*.

Kim, G. et al. (2022) 'OCR-free document understanding transformer (Donut)',
*ECCV 2022*.

Park, S. et al. (2019) 'CORD: a consolidated receipt dataset for post-OCR
parsing', *NeurIPS Workshop on Document Intelligence*.

Qwen Team (2024) *Qwen2.5 technical report.*

Reimers, N. and Gurevych, I. (2019) 'Sentence-BERT: sentence embeddings using
Siamese BERT-networks', *EMNLP 2019*.

Xu, Y. et al. (2019) 'LayoutLM: pre-training of text and layout for document
image understanding', *KDD 2020*.

Yang, Z. et al. (2022) *A survey on information extraction from documents.*

Yoon, S. et al. (2024) *Visual document understanding: recent advances.*

Zhu, Y. et al. (2025) *Multi-modal document understanding.*

> **Note on references.** Citations carried over from the earlier draft have been
> retained where the claim they support is still made. Before submission every
> reference must be verified against the actual source and completed with full
> bibliographic details in the required citation style; several entries are
> incomplete because the original draft supplied only author and year.
