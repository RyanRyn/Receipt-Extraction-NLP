"""
Head-to-head OCR comparison: EasyOCR vs Tesseract, on Malaysian receipts.

    python benchmark_ocr.py --limit 20
    python benchmark_ocr.py --images data/receipts --labels data/receipts/labels.json

Why this exists
---------------
"Which OCR engine is better" is not answerable in the abstract - it depends on
the documents and on what you need out of them. The metric used here is the one
that actually matters for a DMS: **field recoverability**, i.e. of the four
annotated values on each receipt, how many survive OCR well enough to be found
in its output at all.

That number is the *ceiling* on downstream NER accuracy. No NER method, however
clever, can extract a total that OCR never read. Mean character confidence, by
contrast, tells you how sure the engine feels - not whether the information you
need came through.

Both engines are given the same preprocessing variants and each is allowed to
keep whichever one it reads best, so neither is handicapped by a filter tuned
for the other.
"""
from __future__ import annotations

# Running from tools/, so the project root must be on the import path before
# `dms` (or a sibling tool such as evaluate.py) can be imported.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dms.ocr import VARIANTS, ReceiptOCR, group_into_lines
from dms.textutils import (
    find_money_values,
    normalize_date,
    parse_money,
    partial_similarity,
)

RECOVERY_THRESHOLD = 0.80
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# psm 6 = one uniform block, psm 4 = single column of variable-size text.
# Both are the sane choices for a receipt; the better of the two is kept.
TESS_CONFIGS = ["--oem 3 --psm 6", "--oem 3 --psm 4"]
TESS_LANGS = "msa+eng"


# --------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------

class EasyOcrEngine:
    name = "EasyOCR (ms+en)"

    def __init__(self, variants):
        self.reader = ReceiptOCR()
        self.variants = variants

    def read(self, image) -> tuple[str, float]:
        best_text, best_score, best_conf = "", -1.0, 0.0
        img = ReceiptOCR._load(image)
        for variant in self.variants:
            results = self.reader.reader.readtext(VARIANTS[variant](img))
            score = float(sum(c for _, _, c in results if c >= 0.30))
            if score > best_score:
                lines, tokens = group_into_lines(results)
                confs = [t["conf"] for t in tokens]
                best_text = "\n".join(lines)
                best_conf = float(np.mean(confs)) if confs else 0.0
                best_score = score
        return best_text, best_conf


class TesseractEngine:
    name = "Tesseract (msa+eng)"

    def __init__(self, variants):
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
        self.pt = pytesseract
        self.variants = variants

    def read(self, image) -> tuple[str, float]:
        from PIL import Image

        img = ReceiptOCR._load(image)
        best_text, best_score, best_conf = "", -1.0, 0.0
        for variant in self.variants:
            processed = VARIANTS[variant](img)
            pil = Image.fromarray(
                processed if processed.ndim == 2 else processed[:, :, ::-1]
            )
            for config in TESS_CONFIGS:
                data = self.pt.image_to_data(
                    pil, lang=TESS_LANGS, config=config,
                    output_type=self.pt.Output.DICT,
                )
                confs, words = [], []
                for word, conf in zip(data["text"], data["conf"]):
                    try:
                        c = float(conf) / 100.0
                    except (TypeError, ValueError):
                        continue
                    if word.strip() and c >= 0.30:
                        words.append(word.strip())
                        confs.append(c)
                score = float(sum(confs))
                if score > best_score:
                    best_score = score
                    best_conf = float(np.mean(confs)) if confs else 0.0
                    best_text = self.pt.image_to_string(
                        pil, lang=TESS_LANGS, config=config)
        return best_text, best_conf


# --------------------------------------------------------------------------
# field recoverability
# --------------------------------------------------------------------------

def recoverable(field: str, truth, text: str) -> bool:
    """Can ``truth`` be found in this OCR output at all?"""
    if truth is None or str(truth).strip() == "" or not text:
        return False
    truth = str(truth).strip()

    if field == "total":
        want = parse_money(truth)
        if want is None:
            return False
        for line in text.split("\n"):
            for value, _, _ in find_money_values(line):
                if abs(value - want) < 0.01:
                    return True
        return False

    if field == "date":
        want = normalize_date(truth)
        if not want:
            return False
        for line in text.split("\n"):
            got = normalize_date(line)
            if got and got[0] == want[0]:
                return True
        return False

    return partial_similarity(truth, text) >= RECOVERY_THRESHOLD


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="amohseni/receipt_VLM_information_extraction")
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", default=None, help="a local folder instead")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--variants", default="gray_otsu,raw,clahe_sharp",
                    help="preprocessing variants offered to BOTH engines")
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)

    from evaluate import FIELD_MAP, load_hf_items, load_local_items

    if args.images:
        items, source, _ = load_local_items(args.images, args.labels, args.limit)
    else:
        items, source, _ = load_hf_items(args.dataset, args.split, args.limit)
    if not items:
        print(f"no receipts in {source}")
        return 1

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            print(f"unknown variant {v!r}; choose from {sorted(VARIANTS)}")
            return 1

    print(f"source : {source}")
    print(f"receipts: {len(items)}   preprocessing offered: {variants}\n")

    engines = [EasyOcrEngine(variants)]
    try:
        engines.append(TesseractEngine(variants))
    except Exception as exc:
        print(f"(skipping Tesseract: {exc})")

    stats = {}
    for engine in engines:
        found = {f: 0 for f in FIELD_MAP}
        confs, seconds, chars = [], 0.0, 0
        print(f"running {engine.name} ...")
        for i, (key, image, truth) in enumerate(items):
            t0 = time.time()
            text, conf = engine.read(image)
            seconds += time.time() - t0
            confs.append(conf)
            chars += len(text)
            for field in FIELD_MAP:
                if recoverable(field, truth.get(field), text):
                    found[field] += 1
            print(f"  {i + 1}/{len(items)}", end="\r")
        print(f"  {len(items)}/{len(items)} done      ")
        stats[engine.name] = {
            "found": found,
            "mean_conf": float(np.mean(confs)) if confs else 0.0,
            "seconds_per_receipt": seconds / len(items),
            "chars_per_receipt": chars / len(items),
        }

    n = len(items)
    print(f"\n{'=' * 74}")
    print("FIELD RECOVERABILITY - can the annotated value be found in the OCR text?")
    print("(this is the ceiling on what any NER method can then extract)")
    print("=" * 74)
    header = f"  {'ENGINE':<22}" + "".join(f"{f.upper():>10}" for f in FIELD_MAP) + f"{'MEAN':>9}"
    print(header)
    for name, s in stats.items():
        row = f"  {name:<22}"
        for field in FIELD_MAP:
            row += f"{s['found'][field] / n:>10.1%}"
        row += f"{sum(s['found'].values()) / (n * len(FIELD_MAP)):>9.1%}"
        print(row)

    print(f"\n{'=' * 74}\nCOST\n{'=' * 74}")
    print(f"  {'ENGINE':<22}{'SEC/RECEIPT':>13}{'MEAN CONF':>12}{'CHARS':>9}")
    for name, s in stats.items():
        print(f"  {name:<22}{s['seconds_per_receipt']:>13.2f}"
              f"{s['mean_conf']:>12.3f}{s['chars_per_receipt']:>9.0f}")

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"source": source, "n": n, "engines": stats},
                                  indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
