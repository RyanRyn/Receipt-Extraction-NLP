# Receipt DMS — OCR + NER (local LLM pipeline)

Extraction half of a Document Management System for Malaysian receipts.
Reads a receipt image, recognises the named entities inside it, and stores them
in a searchable SQLite database with the offsets and pixel boxes needed to
highlight any hit back on the original document.

Bahasa Melayu and English, entirely offline — **no API key, no token, no
subscription**.

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
merchant        : PERNIAGAAN RIANG
invoice_no      : R000039737
date            : 2017-05-10
time            : 21:51
subtotal        : 21.32
tax             : 1.28
total           : 22.60
paid            : 30.00
change          : 7.40
payment_method  : CASH
! TAX: 0.00 -> 1.28 (total - subtotal, and printed on the receipt)
```

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

**All 97 receipts** of the dataset's `test` split, macro-averaged over its four
annotated fields. The configuration was selected on the *training* split and the
test split scored once, so these figures are not flattered by tuning.

| Configuration | Exact | Fuzzy |
|---|---|---|
| Rules only (`--no-llm`) | 53.2% | 66.7% |
| LLM only (`--no-rules`) | 43.2% | 54.5% |
| **Hybrid** | **56.1%** | **68.7%** |

The hybrid's advantage is concentrated on `total`, the field the arithmetic
validators protect: 64.9% → **76.3%**. Full per-field breakdown, the
train/test protocol and error analysis in `docs/ASSIGNMENT_REPORT.md` S5.

Useful flags: `--no-llm`, `--no-rules`, `--model <alias|repo>`,
`--ocr-variant auto|all|raw|gray_otsu|adaptive|clahe_sharp|deskew_otsu`,
`--no-store`.

---

## How it works

```
image → OCR (EasyOCR ms+en, adaptive preprocessing, line reconstruction)
      → NER  ├── rule layer  (bilingual lexicon + layout regex)
             └── local LLM   (Qwen2.5-1.5B-Instruct, few-shot BM+EN)
      → merge + arithmetic validation
      → SQLite (+FTS5) with char offsets and pixel boxes
      → semantic index (BGE-M3 embeddings)
      → search: exact → substring → lexical+semantic → place fallback
```

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
| `dms/ocr.py` | preprocessing variants, EasyOCR, line reconstruction, polygons |
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
| **`docs/ASSIGNMENT_REPORT.md`** | **the submission document** — problem, background, methodology, results, pseudocode, algorithm analysis |

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

* **GPU.** `torch` was reinstalled as `2.13.0+cu126` for the RTX 4060, taking
  the LLM from ~2.7 to ~10–13 tok/s. On a CPU-only machine everything still
  runs; use `--no-llm` for demos. See `requirements.txt`.
* **Malaya is not used.** It calls `inspect.getargspec()`, removed in Python
  3.11, so every Malaya model fails on this Python 3.14 environment. Analysis
  in `docs/ASSIGNMENT_REPORT.md` S2.5.
* **Database.** `data/dms.sqlite3` by default; override with the `DMS_DB`
  environment variable. Delete the file to start clean.
