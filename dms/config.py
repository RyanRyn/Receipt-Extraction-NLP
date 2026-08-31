"""Central configuration + hardware detection for the Receipt DMS."""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DMS_DATA_DIR", PROJECT_ROOT / "data"))
DB_PATH = Path(os.getenv("DMS_DB", DATA_DIR / "dms.sqlite3"))
CACHE_DIR = DATA_DIR / "cache"

for _d in (DATA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------
# EasyOCR language codes. 'ms' (Malay) and 'en' share the same Latin
# recognition model, so requesting both costs nothing extra but lets the
# decoder's language model favour Malay word shapes (JUMLAH, TUNAI, CUKAI...).
# Which OCR backend to use unless a caller says otherwise.
#
# Tesseract, measured over the 120-receipt training split, recovered the four
# annotated fields far more often than EasyOCR - 62.9% exact against 51.7%,
# with address almost doubling - at a cost of roughly 2.5s per receipt. The
# selection was made on the training split, never on test.
#
# It is a separate program rather than a pip dependency, so make_ocr() falls
# back to EasyOCR when it is absent (see dms/ocr.py) and a fresh clone still
# runs without it.
OCR_ENGINE = os.getenv("DMS_OCR_ENGINE", "tesseract")

OCR_LANGS = ["ms", "en"]

# Minimum per-token OCR confidence kept in the reconstructed text.
OCR_MIN_CONF = float(os.getenv("DMS_OCR_MIN_CONF", "0.30"))

# Preprocessing variants attempted in "auto" mode, best-scoring one wins.
# Ordered cheapest/most-likely first. Use "all" to try every variant.
OCR_AUTO_VARIANTS = ["gray_otsu", "raw", "clahe_sharp"]

# Two text boxes belong to the same visual line when their vertical centres
# are closer than this fraction of the median glyph height.
OCR_LINE_TOLERANCE = 0.6

# --------------------------------------------------------------------------
# Local LLM  (all free / no API key / no subscription)
# --------------------------------------------------------------------------
LLM_MODELS = {
    # key                 HuggingFace repo id                         approx size
    "qwen0.5b":     "Qwen/Qwen2.5-0.5B-Instruct",                    # ~1.0 GB
    "qwen1.5b":     "Qwen/Qwen2.5-1.5B-Instruct",                    # ~3.1 GB  <- default
    "qwen3-1.7b":   "Qwen/Qwen3-1.7B",                               # ~3.4 GB
    "qwen3b":       "Qwen/Qwen2.5-3B-Instruct",                      # ~6.2 GB
    # Malaysian-tuned variants (better Bahasa Melayu / Manglish coverage)
    "malaysian1.5b": "mesolitica/Malaysian-Qwen2.5-1.5B-Instruct-v0.1",
    "malaysian3b":   "mesolitica/Malaysian-Qwen2.5-3B-Instruct",
}

DEFAULT_LLM_KEY = os.getenv("DMS_LLM", "qwen1.5b")

# Deterministic decoding: extraction must be reproducible for a report.
LLM_MAX_NEW_TOKENS_HEADER = int(os.getenv("DMS_LLM_MAX_HEADER", "320"))
LLM_MAX_NEW_TOKENS_FULL = int(os.getenv("DMS_LLM_MAX_FULL", "700"))

# --------------------------------------------------------------------------
# Entity merging
# --------------------------------------------------------------------------
# Confidence assigned when the rule layer and the LLM layer agree.
CONF_AGREE = 0.97
CONF_RULE_ONLY = 0.80
CONF_LLM_ONLY = 0.70
# Tolerance (in MYR) for the subtotal + tax == total arithmetic check.
ARITHMETIC_TOLERANCE = 0.05

# The largest share of a total that can plausibly be tax. Malaysian GST was 6%
# and SST is 6-10%; even with a 10% service charge on top, nothing legitimate
# approaches this bound. A "tax" above it is a misread or mislabelled figure -
# typically the subtotal.
TAX_MAX_FRACTION = 0.35
# Above this share, an amount labelled as tax looks exactly like a subtotal.
SUBTOTAL_LIKE_FRACTION = 0.50

# Minimum ratio for the "similar entity" fallback search, applied to the best
# word-window match (see textutils.partial_similarity). Measured on the sample
# corpus, genuine matches score 0.92-1.00 while unrelated pairs peak at 0.55,
# so 0.72 sits in the gap: "Kuala Lumpor" resolves to "Kuala Lumpur", while
# "Johor Bahru" correctly finds nothing and falls through to the type fallback.
SIMILARITY_THRESHOLD = float(os.getenv("DMS_SIMILARITY", "0.72"))

# Minimum cosine similarity for a semantic hit. Embedding models differ in how
# they scale similarity, so this belongs with the model rather than being a
# universal constant; the value here was chosen from the measured separation
# between real queries and nonsense ones (see benchmark_embeddings.py).
SEMANTIC_MIN_SCORE = float(os.getenv("DMS_SEMANTIC_MIN", "0.60"))


def default_receipt() -> Path | None:
    """Any receipt image to use when the caller did not name one.

    Prefers ``sample_receipt.jpg`` in the project root, but falls back to the
    first image in the exported corpus. This keeps the helper scripts working
    if that sample file is ever deleted - none of the pipeline depends on it,
    and neither should the tools that demonstrate the pipeline.
    """
    sample = PROJECT_ROOT / "sample_receipt.jpg"
    if sample.exists():
        return sample
    corpus = DATA_DIR / "receipts"
    if corpus.is_dir():
        for path in sorted(corpus.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                return path
    return None


def resolve_model(key_or_repo: str | None = None) -> str:
    """Map a short alias ('qwen1.5b') to a HuggingFace repo id.

    An unrecognised value is passed through unchanged, so any repo id works.
    """
    key = key_or_repo or DEFAULT_LLM_KEY
    return LLM_MODELS.get(key, key)


def detect_device() -> tuple[str, object]:
    """Return ``(device, dtype)`` best suited to the machine we are on.

    float16 on CUDA; float32 on CPU. bfloat16 is deliberately avoided on CPU:
    consumer Intel chips without AVX512-BF16 emulate it and end up *slower*
    than plain float32.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def describe_runtime() -> str:
    """One-line human readable summary of the compute environment."""
    import torch

    device, dtype = detect_device()
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"CUDA ({name}, {vram:.1f} GB VRAM), dtype={dtype}"
    threads = torch.get_num_threads()
    return (
        f"CPU ({threads} threads, torch {torch.__version__}), dtype={dtype} "
        "- install a CUDA build of torch for a large speed-up"
    )
