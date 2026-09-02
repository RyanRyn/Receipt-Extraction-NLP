# Receipt DMS — OCR + NER (local LLM pipeline)

Extraction half of a Document Management System for Malaysian receipts.
Reads a receipt image, recognises the named entities inside it, and stores them
in a searchable SQLite database with the offsets and pixel boxes needed to
highlight any hit back on the original document.

Bahasa Melayu and English, entirely offline — **no API key, no token, no
subscription**.

> ### New to this project?
> **Read [`SETUP.md`](SETUP.md) first.** It walks through installing Python,
> creating a virtual environment and getting the app running, assuming no prior
> Python experience. The commands below assume that is already done.

---

## Quick start — the graphical interface

```bash
.\.venv\Scripts\streamlit.exe run app.py
```

(The full path is used because `streamlit` lives inside the virtual environment
and is not on the system PATH. If you activate the venv first with
`.\.venv\Scripts\Activate.ps1`, plain `streamlit run app.py` works too.)

Opens in your browser with four pages:

| Page | What it does |
|---|---|
| **Process a receipt** | Upload an image and watch every pipeline stage: which filter won and why, the rebuilt text, the entities with confidence and source, the validation checks, and the entity boxes drawn on the image |
| **Search** | Keyword and semantic search, showing which of the four strategies answered, with the hit highlighted in the document |
| **Database** | What is stored, entity counts, and how the storage is organised |
| **How it works** | The workflow explained, with the measured results |

## Or from the command line

```bash
python run_dms.py process sample_receipt.jpg
```

```
merchant        : KEDAI PAPAN YEW CHUAN
address         : LOT 276 JALAN BANTING, 43800 DENGKIL, SELANGOR
phone           : 03-87686092
date            : 2018-03-10
time            : 13:49
subtotal        : 84.80
tax             : 4.80
total           : 84.80
paid            : 84.80
payment_method  : CASH
! TAX: rule='80.00' vs llm='4.80' -> kept rule
! TOTAL: 4.80 -> 84.80 (paid - change, and printed on the receipt)
! TAX 80.00 is 94% of the total - implausible; using 4.80 from the llm layer instead
```

The three `!` lines are the point of the whole system. OCR misread the total as
`4.80` and the tax as `80.00`; neither layer noticed on its own. The receipt's
own arithmetic caught both, and each correction says *why* it was made rather
than being applied silently.

No GPU, or want it instantly? The rule layer alone needs no model download:

```bash
python run_dms.py process sample_receipt.jpg --no-llm
```

---

## Commands

```bash
# process
python run_dms.py process sample_receipt.jpg          # one image
python run_dms.py process receipts/ --limit 10        # a folder
python run_dms.py dataset --limit 20                  # from HuggingFace
python run_dms.py process img.jpg --header-only       # skip line items, ~2x faster

# search  (reference implementation of the teammate's half)
python run_dms.py search "Kuala Lumpur" --highlight
python run_dms.py search "Johor Bahru"                # similar-entity fallback
python run_dms.py search "60.31" --type TOTAL

# inspect
python run_dms.py show 1 --html out.html
python run_dms.py stats

# evaluate against dataset ground truth
python tools/evaluate.py --limit 30 --config all

# tests (no model download, no GPU, a few seconds)
python test_dms.py

# end-to-end proof: image -> OCR -> NER -> database -> semantic search
python tools/verify_end_to_end.py
```

`tools/verify_end_to_end.py` starts from an **empty temporary database** (your real
one is untouched), ingests a receipt plus a small corpus, and asserts every
link in the chain — including a semantic query that shares no words with
anything stored, so only a genuine meaning-based match can succeed.

### Using your own receipts

The HuggingFace copy lives in a cache as parquet blobs. Turn it into a real
folder of images you can open, inspect and screenshot for a report:

```bash
python run_dms.py export --split test --out data/receipts
```

That writes `test_00000.jpg …` plus a `labels.json`. Any folder of images then
works directly:

```bash
python run_dms.py process my_receipts/
python tools/evaluate.py --images my_receipts --labels my_receipts/labels.json
```

Without `--labels` the run still reports coverage (how often each field was
found), which is useful on unlabelled receipts. Labels may be JSON
(`{"file.jpg": {"company":…, "date":…, "address":…, "total":…}}`) or a CSV with
a `filename` column.

### Measured accuracy

**All 97 receipts** of the dataset's `test` split, micro-averaged over its four
annotated fields, read with the default Tesseract engine. *Micro* means every
labelled field judgement is pooled rather than averaging the four per-field
rates; here the two agree to within 0.1 points because the fields are labelled
almost equally often. The configuration was
selected on the *training* split and the test split scored once, so these
figures are not flattered by tuning. Train scored 63.3% against test's 63.6% —
a generalisation gap of −0.2 points, i.e. it transfers to unseen receipts.

| Configuration | Exact | Fuzzy |
|---|---|---|
| Rules only (`--no-llm`) | **63.8%** | 73.9% |
| LLM only (`--no-rules`) | 52.2% | 64.9% |
| **Hybrid** (shipped) | 63.6% | **77.3%** |

Two things in that table deserve saying out loud rather than hiding.

**The rule layer alone matches the hybrid on exact match** — 63.8% against
63.6%, a gap far inside the noise of 97 receipts. That was not true with the
weaker OCR, where the hybrid led by three points. A better reader leaves less
for the language model to repair.

**The hybrid still earns its place on the other two axes.** It is 3.4 points
ahead on fuzzy match, and it is markedly more *complete*: it finds a date on
100% of receipts against the rules' 87.6%, and a total on 95.9% against 93.8%.
On addresses it is 72.9% fuzzy against 63.5% — much closer to right even when
not exactly right.

The shipped configuration routes `ADDRESS` to the language model because that
won on the **training** split (63.3% against 62.3%). On test it turns out to
cost 0.2 points. That is left as it stands: changing it now, having seen the
test score, is precisely the leakage the protocol exists to prevent.

Full per-field breakdown, the train/test protocol and error analysis in
`docs/ASSIGNMENT_REPORT.md` S5.

Useful flags: `--no-llm`, `--no-rules`, `--model <alias|repo>`,
`--ocr-variant auto|all|raw|gray_otsu|adaptive|clahe_sharp|deskew_otsu`,
`--no-store`.

---

## How it works

```
image → OCR (Tesseract msa+eng, adaptive preprocessing, line reconstruction)
      → NER  ├── rule layer  (bilingual lexicon + layout regex)
             └── local LLM   (Qwen2.5-1.5B-Instruct, few-shot BM+EN)
      → merge + arithmetic validation
      → SQLite (+FTS5) with char offsets and pixel boxes
      → semantic index (BGE-M3 embeddings)
      → search: exact → substring → lexical+semantic → place fallback
```

Diagrams of all of this — architecture, pipeline, the merge, the search
cascade — are in [`docs/diagrams/`](docs/diagrams/DIAGRAMS.md). Open
`docs/diagrams/diagrams.html` to export any of them as PNG or SVG.

Full stage-by-stage walkthrough: **`docs/ASSIGNMENT_REPORT.md`** S3.

Each layer handles what it is good at. The rules are exact on printed patterns
and never hallucinate; the LLM reads layout and repairs OCR noise (`22-60` →
`22.60`). Where they disagree, receipt arithmetic (`subtotal + tax = total`)
arbitrates. Agreement between layers is the confidence score.

Full rationale, technology comparison and results: **`docs/ASSIGNMENT_REPORT.md`**.

---

## Files

| Path | Purpose |
|---|---|
| `dms/ocr.py` | preprocessing variants, Tesseract + EasyOCR backends, line reconstruction, polygons |
| `dms/lexicon.py` | all Malay/English vocabulary and the place gazetteer |
| `dms/rules.py` | deterministic bilingual extractor (baseline) |
| `dms/llm.py` | local LLM loading, prompting, JSON recovery |
| `dms/ner.py` | hybrid merge, arithmetic arbitration, validation |
| `dms/database.py` | SQLite schema, search cascade, highlighting |
| `dms/embeddings.py` | sentence embeddings (BGE-M3), model registry, cosine ranking |
| `dms/pipeline.py` | orchestration |
| `app.py` | **graphical interface** (Streamlit), 4 pages |
| `run_dms.py` | CLI: `process` `dataset` `export` `search` `show` `stats` `compare` `reindex` `delete` |
| `test_dms.py` | 106 checks; runs offline, no model needed |
| `tools/verify_end_to_end.py` | proves the whole chain from an empty database |
| `tools/prepare_demo.py` | pre-flight check before a live demonstration |
| `tools/trace_llm_to_db.py` | traces one receipt from raw model output to SQL rows |
| `tools/evaluate.py` | scoring against dataset ground truth + ablation |
| `tools/tune_on_train.py` | selects routing on train, scores test once |
| `tools/benchmark_ocr.py` · `tools/benchmark_llm.py` · `tools/benchmark_embeddings.py` | the model comparisons behind S5 of the report |
| `tools/render_diagrams.py` | builds the diagram page from `docs/diagrams/DIAGRAMS.md` |
| **`docs/ASSIGNMENT_REPORT.md`** | **the submission document** — problem, background, methodology, results, pseudocode, algorithm analysis |
| `docs/diagrams/` | architecture, pipeline, merge and search figures (Mermaid + PNG/SVG export) |
| `docs/DEFENCE_PACK.md` | presentation and Q&A preparation, with live-edit drills |
| `docs/SEARCH_BRIEFING.md` | briefing for whoever demonstrates the search half |

---

## Choosing a model

Default is `Qwen2.5-1.5B-Instruct` (~3.1 GB, Apache-2.0). Swap freely:

```bash
python run_dms.py process img.jpg --model malaysian3b   # Malaysian-tuned 3B
python run_dms.py process img.jpg --model qwen0.5b      # weaker machine
python run_dms.py process img.jpg --model qwen3-1.7b    # newer generation
```

Aliases are defined in `dms/config.py :: LLM_MODELS`; any HuggingFace repo id
also works.

---

## Notes

* **OCR engine.** **Tesseract is the default.** On the 120-receipt training
  split it read the four annotated fields far more accurately than EasyOCR —
  63.3% exact against 51.7%, with `address` more than doubling — for about 2.5s
  more per receipt. It is a separate program rather than a pip package, so
  `SETUP.md` step 7 installs it; without it the system falls back to EasyOCR,
  prints a warning, and scores below the figures above. Switch at any time with
  `--ocr-engine easyocr`, the GUI dropdown, or `DMS_OCR_ENGINE=easyocr`.
* **GPU.** `torch` was reinstalled as `2.13.0+cu126` for the RTX 4060, taking
  the LLM from ~2.7 to ~10–13 tok/s. On a CPU-only machine everything still
  runs; use `--no-llm` for demos. See `requirements.txt`.
* **Malaya is not used.** It calls `inspect.getargspec()`, removed in Python
  3.11, so every Malaya model fails on this Python 3.14 environment. Analysis
  in `docs/ASSIGNMENT_REPORT.md` S2.5.
* **Database.** `data/dms.sqlite3` by default; override with the `DMS_DB`
  environment variable. Delete the file to start clean.
