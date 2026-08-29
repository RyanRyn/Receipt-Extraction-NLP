# Setting up the Receipt DMS

Instructions for running this project on a machine that has never seen it.

---

## What you need

| | |
|---|---|
| Python | **3.11 or newer** (developed on 3.14.7) |
| Disk | ~7 GB — about 1.5 GB of libraries and 5.4 GB of models |
| Internet | **required once**, to download the models. Everything runs offline afterwards |
| GPU | optional. An NVIDIA card makes it ~4× faster; without one it still works |

There is **no API key, no token and no subscription** anywhere in this project.

---

## Step 1 — Create a virtual environment

From inside the project folder:

```bash
python -m venv .venv
```

The archive deliberately does **not** contain a virtual environment: it is about
5 GB, and one built on another machine will not work on yours.

## Step 2 — Install PyTorch

Do this **before** the other packages, because the right build depends on your
hardware.

**With an NVIDIA GPU:**

```bash
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple torch torchvision
```

**Without a GPU (or if unsure):**

```bash
.\.venv\Scripts\python.exe -m pip install torch torchvision
```

Check which you got:

```bash
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True` means the GPU is in use. `False` is fine — extraction is slower
(~90 s per receipt instead of ~30 s), and you can pass `--no-llm` for an
instant rule-only run.

## Step 3 — Install everything else

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Step 4 — Check it works, without downloading any model

```bash
.\.venv\Scripts\python.exe test_dms.py
```

Expect `ALL 112 CHECKS PASSED`. This suite deliberately touches no model, so it
proves the installation before anything large is fetched.

## Step 5 — First real run (this downloads the models)

```bash
.\.venv\Scripts\python.exe run_dms.py process sample_receipt.jpg
```

The first run fetches roughly:

| Model | Size | Used for |
|---|---|---|
| EasyOCR (CRAFT + Latin) | ~110 MB | reading the image |
| Qwen2.5-1.5B-Instruct | ~3.1 GB | the language-model reader |
| BGE-M3 | ~2.2 GB | semantic search |

They are cached in your home folder, so this happens **once**. Later runs are
entirely offline.

Expected output:

```
merchant        : PERNIAGAAN RIANG
date            : 2017-05-10
subtotal        : 21.32
tax             : 1.28
total           : 22.60
! TAX: 0.00 -> 1.28 (total - subtotal, and printed on the receipt)
```

## Step 6 — The graphical interface

```bash
.\.venv\Scripts\streamlit.exe run app.py
```

Opens at <http://localhost:8501>.

> `streamlit` lives inside `.venv` and is not on the system PATH, which is why
> the full path is used. If you activate the environment first
> (`.\.venv\Scripts\Activate.ps1`), plain `streamlit run app.py` works too.

---

## If the archive included the database

`data/dms.sqlite3` arrives already populated, so search works immediately:

```bash
.\.venv\Scripts\python.exe run_dms.py stats
.\.venv\Scripts\python.exe run_dms.py search "Kota Kinabalu"
```

If it is empty, or you want to start over, build a corpus:

```bash
.\.venv\Scripts\python.exe run_dms.py process data/receipts --limit 12
```

**Semantic search needs a corpus.** With one or two documents indexed it has no
meaningful neighbours and will look broken. Twelve or more is a sensible
minimum.

---

## Before demonstrating anything

```bash
.\.venv\Scripts\python.exe tools\prepare_demo.py --receipts 12
```

This fills the database, loads every model into memory, and checks all four
required DMS functions. Wait for `READY TO PRESENT`. Run it *before* an
audience is watching — the first receipt takes ~30 s while the models load.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `'streamlit' is not recognized` | not on PATH | use `.\.venv\Scripts\streamlit.exe` |
| `ModuleNotFoundError: dms` | wrong folder | `cd` into the project root first |
| `File does not exist: app.py` | wrong folder | same |
| First run hangs for a minute | downloading models | expected once; watch the progress bar |
| `Port 8501 is already in use` | already running | open <http://localhost:8501>, or add `--server.port 8502` |
| Semantic search finds nothing | corpus too small | process at least 12 receipts |
| `CUDA out of memory` | GPU too small for the model | `--model qwen0.5b`, or `--no-llm` |

---

## What is in the archive

| Path | |
|---|---|
| `dms/` | the pipeline — OCR, rules, LLM, merge, database, embeddings |
| `app.py` | graphical interface |
| `run_dms.py` | command line |
| `test_dms.py` | 112 checks, no model needed |
| `tools/` | evaluation, benchmarks, demo helpers |
| `docs/ASSIGNMENT_REPORT.md` | the report |
| `data/receipts/` | receipt images with ground-truth labels |
| `data/*.json` | the measurements behind the report's results |

Deliberately **excluded**: `.venv` (5 GB, machine-specific), the model cache
(downloaded on first run), `__pycache__`, and the 352 MB training split, which
is not needed to run or demonstrate the system.
