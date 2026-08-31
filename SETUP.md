# Setup guide

How to run this project on a computer that has never seen it. No prior Python
experience assumed — every step says what to type, and what you should see.

Total time: about 20 minutes, most of it waiting for downloads.

---

## What gets installed, and how big it is

| What | Size | When |
|---|---|---|
| Python | ~100 MB | step 1, from python.org |
| PyTorch | ~2.5 GB (GPU) or ~200 MB (CPU) | step 4 |
| Other libraries | ~1 GB | step 5 |
| Tesseract OCR | ~100 MB | step 7 |
| AI models | ~5.4 GB | step 8, automatic on first run |

**About 9 GB of free disk space**, and an internet connection for the first
run. After that everything works offline.

There is **no account, no API key and no subscription** anywhere in this
project. Every model is free and downloads automatically.

---

## Step 1 — Install Python

Skip this if you already have Python 3.11 or newer. To check, open a terminal
(see step 2) and type:

```
python --version
```

If it prints `Python 3.11.x` or higher, go to step 2.

Otherwise download it from **<https://www.python.org/downloads/>**.

> ### ⚠️ The one thing people get wrong
>
> On the first screen of the installer there is a checkbox at the bottom:
>
> **☑ Add python.exe to PATH**
>
> **You must tick it.** If you miss it, every command below fails with
> `'python' is not recognized`. If that happens, run the installer again,
> choose *Modify*, and tick it.

Then click **Install Now** and wait.

---

## Step 2 — Open a terminal inside the project folder

Unzip or clone the project somewhere you can find, for example
`C:\Users\YourName\Documents\receipt-dms`.

**Windows:**
1. Open the project folder in File Explorer
2. Click the address bar at the top (where the folder path is shown)
3. Type `powershell` and press **Enter**

A blue window opens, already pointing at the right folder. That last part
matters — most "file not found" errors are just a terminal sitting in the
wrong place.

**Mac / Linux:** open Terminal and `cd` to the folder.

Check you are in the right place:

```
dir
```

You should see `app.py`, `run_dms.py`, `dms`, `requirements.txt`. If you do
not, you are in the wrong folder — go back to step 2.

*(On Mac/Linux use `ls` instead of `dir`.)*

---

## Step 3 — Create a virtual environment

**What is this?** A private folder holding this project's libraries, so they
cannot clash with anything else on your computer. It is standard practice and
takes one command.

```
python -m venv .venv
```

Nothing is printed; it takes about 20 seconds and creates a hidden `.venv`
folder.

> **Why isn't it already in the project?** It is ~5 GB, and an environment
> built on one machine does not work on another. Everyone makes their own.

### About the commands below

Every command starts with `.\.venv\Scripts\python.exe` instead of just
`python`. That is not a mistake — it points at the Python **inside** the
virtual environment rather than the system one. Using the full path means you
never have to remember whether the environment is "activated".

*(Mac/Linux: use `./.venv/bin/python` instead.)*

---

## Step 4 — Install PyTorch

Install this **before** the other libraries, because the correct version
depends on your graphics card.

**Do you have an NVIDIA graphics card?** If unsure, press `Ctrl+Shift+Esc`
→ Performance tab → look for "GPU". If it says NVIDIA, use the first command.

**With an NVIDIA card** (faster — about 30 seconds per receipt):

```
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple torch torchvision
```

**Without one, or unsure** (works fine, about 90 seconds per receipt):

```
.\.venv\Scripts\python.exe -m pip install torch torchvision
```

This is the big download — 2.5 GB for the GPU version. Expect several minutes.

Check which you ended up with:

```
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`True` = the graphics card will be used. `False` = it will use the processor,
which is slower but perfectly fine.

---

## Step 5 — Install everything else

```
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

A few minutes. Ends with `Successfully installed ...`.

---

## Step 6 — Check it works (no downloads)

```
.\.venv\Scripts\python.exe test_dms.py
```

You should see a long list of `PASS` lines ending with:

```
ALL 112 CHECKS PASSED
```

This deliberately uses **no AI models at all**, so it proves the installation
is correct before you download 5 GB. If it fails here, something in steps 3–5
went wrong — fix that first.

---

## Step 7 — Install Tesseract (the text reader)

This is a **separate program**, not a Python library, so `pip` cannot fetch it.

**Why it matters:** Tesseract is the default reader because it is substantially
more accurate on these receipts — 62.9% of fields exactly right against
EasyOCR's 51.7% on the training split, with addresses nearly twice as good. The
project still runs without it (it falls back to EasyOCR and prints a warning),
but the numbers in the report assume Tesseract.

**Windows** — download the installer from
**<https://github.com/UB-Mannheim/tesseract/wiki>** and run it.

> ### ⚠️ Two things to get right in the installer
>
> 1. On the **"Choose Components"** screen, expand *Additional language data*
>    and tick **Malay (msa)**. Without it the Malay half of every receipt is
>    read by an English-only model.
> 2. Keep the default install location (`C:\Program Files\Tesseract-OCR`).
>    The project looks there automatically.

**Mac:** `brew install tesseract tesseract-lang`
**Linux:** `sudo apt install tesseract-ocr tesseract-ocr-msa`

Check it worked:

```
.\.venv\Scripts\python.exe -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

You should see a version number such as `5.5.3`. If instead you get an error
about the executable not being found, either reinstall to the default location
or set the path once:

```
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## Step 8 — First real run (downloads the models)

```
.\.venv\Scripts\python.exe run_dms.py process sample_receipt.jpg
```

**The first run takes 2–5 minutes** and looks like it has frozen. It has not —
it is downloading:

| Model | Size | For |
|---|---|---|
| EasyOCR | ~110 MB | reading text from the image |
| Qwen2.5-1.5B | ~3.1 GB | understanding the receipt |
| BGE-M3 | ~2.2 GB | search by meaning |

These are saved in your home folder, so it only happens **once**.

Expected result:

```
merchant        : KEDAI PAPAN YEW CHUAN
address         : LOT 276 JALAN BANTING, 43800 DENGKIL, SELANGOR
date            : 2018-03-10
subtotal        : 84.80
tax             : 4.80
total           : 84.80
! TOTAL: 4.80 -> 84.80 (paid - change, and printed on the receipt)
! TAX 80.00 is 94% of the total - implausible; using 4.80 from the llm layer instead
```

Those last two lines are not errors — they are the system correcting values that
OCR damaged, and explaining why. The printed total was misread as `4.80`, and
the amount tendered proved what it should have been.

---

## Step 9 — Get some receipts and fill the database

The receipt images are not included in the repository (they belong to the
public dataset, not to this project). Download them with:

```
.\.venv\Scripts\python.exe run_dms.py export --split test --out data/receipts
```

Then process a batch. **Search needs a reasonable number of documents** — with
only one or two, search by meaning has nothing to compare against and will look
broken:

```
.\.venv\Scripts\python.exe run_dms.py process data/receipts --limit 20
```

About 30 seconds per receipt, so ~10 minutes for 20. Leave it running.

---

## Step 10 — Open the app

```
.\.venv\Scripts\streamlit.exe run app.py
```

Your browser opens at **http://localhost:8501** with four pages: upload a
receipt, search, browse the database, and an explanation of how it works.

To stop it, click the terminal and press **Ctrl + C**.

> Note this one uses `streamlit.exe`, not `python.exe` — Streamlit is its own
> program.

---

## Try these searches

On the **Search** page:

| Type this | What it shows |
|---|---|
| `Kuala Lumpur` | ordinary matching |
| `Kuala Lumpor` | spelled wrong — still found |
| `nasi ayam` | a **Malay** query finding **English** text, by meaning |
| `Kota Kinabalu` | a city that is not stored — returns the other locations instead |

---

## When something goes wrong

| Message | What it means | Fix |
|---|---|---|
| `'python' is not recognized` | PATH box unticked in step 1 | re-run the Python installer, choose *Modify*, tick **Add python.exe to PATH**, reopen the terminal |
| `'streamlit' is not recognized` | using `streamlit` on its own | use the full `.\.venv\Scripts\streamlit.exe` |
| `can't open file 'app.py'` | terminal in the wrong folder | redo step 2; check with `dir` |
| `ModuleNotFoundError: No module named 'dms'` | same — wrong folder | redo step 2 |
| `ModuleNotFoundError: No module named 'torch'` | step 4 skipped or failed | redo step 4 |
| `running scripts is disabled on this system` | PowerShell blocks scripts | run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then retry. Affects only that window |
| Seems frozen on the first receipt | downloading 5 GB of models | wait — it only happens once |
| `Port 8501 is already in use` | app already running | open <http://localhost:8501>, or add `--server.port 8502` |
| Search finds nothing | too few documents | process at least 12 receipts (step 9) |
| `CUDA out of memory` | graphics card too small | add `--model qwen0.5b`, or `--no-llm` |
| `tesseract is unavailable; falling back to EasyOCR` | step 7 skipped | install Tesseract, or ignore it — the system still works, just less accurately |
| Accuracy lower than the report | probably running on the EasyOCR fallback | check step 7; the run prints which engine it used |

---

## Quick reference

```
.\.venv\Scripts\python.exe run_dms.py process sample_receipt.jpg   # one receipt
.\.venv\Scripts\python.exe run_dms.py process data/receipts --limit 20
.\.venv\Scripts\python.exe run_dms.py search "Kuala Lumpur"        # search
.\.venv\Scripts\python.exe run_dms.py stats                        # what is stored
.\.venv\Scripts\python.exe test_dms.py                             # run the checks
.\.venv\Scripts\streamlit.exe run app.py                           # the app
```

Add `--no-llm` to any `process` command for an instant run that uses no AI
model at all — useful for checking things quickly.

---

## What is in the project

| Folder | |
|---|---|
| `dms/` | the system itself — OCR, rules, language model, database, search |
| `tools/` | evaluation scripts and demonstration helpers |
| `docs/` | the project report |
| `data/` | the database and results (receipt images arrive in step 9) |

`README.md` explains what the system does and how it performs.
`docs/ASSIGNMENT_REPORT.md` is the full write-up.
