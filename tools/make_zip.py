"""
Package the project into a zip a teammate can actually run.

    python tools/make_zip.py                    # code + receipts + database
    python tools/make_zip.py --no-database      # they build their own corpus
    python tools/make_zip.py --code-only        # smallest: source and docs only
    python tools/make_zip.py --out handoff.zip

What is excluded, and why
-------------------------
``.venv``            ~5 GB, and an environment built on one machine does not
                     work on another. SETUP.md explains how to create one.
model cache          ~5.4 GB, lives in the home folder rather than the project,
                     and is downloaded automatically on first run.
``data/receipts_train``  352 MB. The training split is only needed to reproduce
                     the tuning experiment, not to run or demonstrate anything.
``data/cache``       OCR and LLM response caches; regenerated on demand.
``__pycache__``      compiled bytecode, rebuilt automatically.

The result is a few tens of megabytes rather than several gigabytes.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories never included, wherever they appear.
EXCLUDE_DIRS = {".venv", "__pycache__", ".git", ".ipynb_checkpoints",
                ".pytest_cache", ".mypy_cache", "cache", "receipts_train"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".sqlite3-journal", ".zip"}

# Always included: the code, the docs, the setup guide.
CODE = ["dms", "tools", "docs", "app.py", "run_dms.py", "test_dms.py",
        "README.md", "SETUP.md", "requirements.txt", "sample_receipt.jpg"]
# The measurements the report cites.
EVIDENCE = ["data/tuning_report.json", "data/final_test97.json",
            "data/final_tesseract.json", "data/ocr_benchmark.json",
            "data/llm_benchmark.json", "data/embedding_benchmark.json"]


def wanted(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return False
    return path.suffix.lower() not in EXCLUDE_SUFFIXES


def collect(entry: str) -> list[Path]:
    target = ROOT / entry
    if not target.exists():
        return []
    if target.is_file():
        return [target] if wanted(target) else []
    return [p for p in sorted(target.rglob("*")) if p.is_file() and wanted(p)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="receipt_dms.zip")
    ap.add_argument("--no-receipts", action="store_true",
                    help="omit data/receipts (about 35 MB of images)")
    ap.add_argument("--no-database", action="store_true",
                    help="omit the populated database; they build their own")
    ap.add_argument("--code-only", action="store_true",
                    help="source, docs and one sample receipt only")
    args = ap.parse_args(argv)

    entries = list(CODE)
    if not args.code_only:
        entries += EVIDENCE
        if not args.no_receipts:
            entries.append("data/receipts")
        if not args.no_database:
            entries.append("data/dms.sqlite3")

    files: list[Path] = []
    for entry in entries:
        found = collect(entry)
        if not found and not entry.startswith("data/"):
            print(f"  WARNING: {entry} not found — is it missing?")
        files.extend(found)

    missing = [f for f in ("app.py", "run_dms.py", "SETUP.md", "requirements.txt")
               if not (ROOT / f).exists()]
    if missing:
        print(f"error: cannot package, these are missing: {missing}")
        return 1

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            # Everything sits under one folder, so unzipping never scatters
            # files across the recipient's Downloads directory.
            arc = Path("receipt_dms") / path.relative_to(ROOT)
            zf.write(path, arc)
            total += path.stat().st_size

    size = out.stat().st_size
    print(f"\n  wrote {out}")
    print(f"  {len(files)} files, {total/1024/1024:.1f} MB uncompressed "
          f"-> {size/1024/1024:.1f} MB zipped")

    print("\n  included:")
    groups: dict[str, list[int]] = {}
    for path in files:
        top = path.relative_to(ROOT).parts[0]
        groups.setdefault(top, []).append(path.stat().st_size)
    for name, sizes in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        print(f"    {name:<24}{len(sizes):>4} files{sum(sizes)/1024/1024:>9.1f} MB")

    print("\n  excluded on purpose: .venv (~5 GB), the model cache (~5.4 GB, "
          "downloaded on first run),")
    print("                       data/receipts_train (352 MB), caches, "
          "__pycache__")
    print(f"\n  Tell your teammate to unzip it and read SETUP.md first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
