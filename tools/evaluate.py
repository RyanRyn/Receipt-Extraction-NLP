"""
Evaluate the DMS against the ground truth of the HuggingFace receipt dataset.

    python evaluate.py --limit 20                    # hybrid (rules + LLM)
    python evaluate.py --limit 20 --config rules     # rule baseline only
    python evaluate.py --limit 20 --config llm       # LLM only
    python evaluate.py --limit 20 --config all       # all three, side by side

Your own receipts, from a folder on disk instead of HuggingFace:

    python evaluate.py --images data/receipts --labels data/receipts/labels.json
    python evaluate.py --images my_receipts                  # coverage only

Accuracy is measured only over fields that carry ground truth, so an
unlabelled folder honestly reports "-" rather than crediting a blank answer.

The dataset (``amohseni/receipt_VLM_information_extraction``) annotates four
fields per receipt - ``company``, ``date``, ``address`` and ``total`` - which
map onto the DMS entity types MERCHANT, DATE, ADDRESS and TOTAL.

OCR output is cached on disk and shared between configurations, so an ablation
compares the NER layers on *identical* OCR text rather than re-reading every
image three times.
"""
from __future__ import annotations

# Running from tools/, so the project root must be on the import path before
# `dms` (or a sibling tool such as evaluate.py) can be imported.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import argparse
import ast
import json
import time
from pathlib import Path

from dms.config import OCR_ENGINE, CACHE_DIR, describe_runtime
from dms.ner import HybridNER
from dms.ocr import make_ocr
from dms.textutils import (
    normalize_date,
    normalize_key,
    parse_money,
    similarity,
)

FIELD_MAP = {          # dataset field -> DMS field
    "company": "merchant",
    "date": "date",
    "address": "address",
    "total": "total",
}
FUZZY_THRESHOLD = 0.85


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------

def parse_ground_truth(suffix: str) -> dict:
    """The dataset stores its answer as a Python dict literal, not JSON."""
    if not suffix:
        return {}
    try:
        value = ast.literal_eval(suffix.strip())
        return value if isinstance(value, dict) else {}
    except (ValueError, SyntaxError):
        try:
            return json.loads(suffix)
        except json.JSONDecodeError:
            return {}


def load_hf_items(dataset: str, split: str, limit: int):
    """``[(key, image, truth), ...]`` from a HuggingFace dataset."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    n = min(limit, len(ds)) if limit else len(ds)
    items = []
    for i in range(n):
        row = ds[i]
        items.append((f"{split}_{i:05d}", row["image"],
                      parse_ground_truth(row.get("suffix", ""))))
    return items, f"{dataset} [{split}]", len(ds)


def load_local_items(images_dir: str, labels_path: str | None, limit: int):
    """``[(key, path, truth), ...]`` from a folder of your own receipt images.

    ``labels_path`` is optional. Without it the run still reports how often
    each field was *found* (coverage), which is useful on unlabelled receipts;
    accuracy needs labels. Accepted label formats:

    * JSON  ``{"receipt1.jpg": {"company": ..., "date": ..., "address": ..., "total": ...}}``
    * CSV   with a ``filename`` column plus ``company,date,address,total``
    """
    root = Path(images_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"not a folder: {root}")

    labels: dict[str, dict] = {}
    if labels_path:
        p = Path(labels_path)
        if not p.exists():
            raise FileNotFoundError(f"labels file not found: {p}")
        if p.suffix.lower() in (".csv", ".tsv"):
            import csv
            delim = "\t" if p.suffix.lower() == ".tsv" else ","
            with p.open(encoding="utf-8-sig", newline="") as fh:
                for row in csv.DictReader(fh, delimiter=delim):
                    key = (row.get("filename") or row.get("file")
                           or row.get("image") or "").strip()
                    if key:
                        labels[key] = {f: (row.get(f) or None) for f in FIELD_MAP}
        else:
            raw = json.loads(p.read_text(encoding="utf-8"))
            labels = raw if isinstance(raw, dict) else {}

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(f for f in root.iterdir() if f.suffix.lower() in suffixes)
    if limit:
        images = images[:limit]

    items = []
    for img in images:
        truth = labels.get(img.name) or labels.get(img.stem) or {}
        items.append((img.name, img, truth))
    return items, str(root), len(images)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def compare(field: str, predicted, truth) -> tuple[bool, bool]:
    """Return ``(exact, fuzzy)`` correctness for one field."""
    if truth is None or str(truth).strip() == "":
        return (predicted is None, predicted is None)
    if predicted is None:
        return (False, False)

    p, t = str(predicted).strip(), str(truth).strip()

    if field == "total":
        pv, tv = parse_money(p), parse_money(t)
        ok = pv is not None and tv is not None and abs(pv - tv) < 0.01
        return (ok, ok)

    if field == "date":
        pd = normalize_date(p)
        td = normalize_date(t)
        pi = pd[0] if pd else normalize_key(p)
        ti = td[0] if td else normalize_key(t)
        ok = pi == ti
        return (ok, ok)

    exact = normalize_key(p) == normalize_key(t)
    return (exact, exact or similarity(p, t) >= FUZZY_THRESHOLD)


class Scoreboard:
    """Accuracy is measured only over fields that actually carry ground truth.

    Scoring an unlabelled field would otherwise credit "predicted nothing,
    annotated nothing" as a correct answer, which inflates the result on a
    partly-labelled or unlabelled folder. ``found`` is always over every
    receipt, since coverage is meaningful with or without labels.
    """

    def __init__(self, name: str):
        self.name = name
        self.exact = {f: 0 for f in FIELD_MAP}
        self.fuzzy = {f: 0 for f in FIELD_MAP}
        self.found = {f: 0 for f in FIELD_MAP}
        self.labelled = {f: 0 for f in FIELD_MAP}
        self.total = 0
        self.seconds = 0.0

    def update(self, predicted: dict, truth: dict) -> dict:
        self.total += 1
        row = {}
        for ds_field, dms_field in FIELD_MAP.items():
            p = predicted.get(dms_field)
            t = truth.get(ds_field)
            if p is not None:
                self.found[ds_field] += 1
            has_truth = t is not None and str(t).strip() != ""
            ex = fz = False
            if has_truth:
                self.labelled[ds_field] += 1
                ex, fz = compare(ds_field, p, t)
                self.exact[ds_field] += ex
                self.fuzzy[ds_field] += fz
            row[ds_field] = {"pred": p, "true": t, "exact": ex, "fuzzy": fz,
                             "labelled": has_truth}
        return row

    def report(self) -> None:
        n = max(self.total, 1)
        print(f"\n--- {self.name}  ({self.total} receipts, "
              f"{self.seconds / n:.1f}s each) ---")
        print(f"  {'FIELD':<12}{'EXACT':>9}{'FUZZY':>9}{'FOUND':>9}{'LABELLED':>10}")
        for f in FIELD_MAP:
            m = self.labelled[f]
            exact = f"{self.exact[f] / m:>8.1%}" if m else f"{'-':>8}"
            fuzzy = f"{self.fuzzy[f] / m:>9.1%}" if m else f"{'-':>9}"
            print(f"  {f:<12}{exact}{fuzzy}{self.found[f] / n:>9.1%}{m:>10}")
        scored = sum(self.labelled.values())
        if scored:
            # A *micro* average: every labelled field judgement is pooled, so a
            # field with more ground truth carries more weight. It is not the
            # mean of the four per-field rates (a macro average). With this
            # dataset the two nearly coincide, because the fields are labelled
            # 97/97/96/97 times - but they are different measures and the
            # distinction is worth keeping straight.
            micro_e = sum(self.exact.values()) / scored
            micro_f = sum(self.fuzzy.values()) / scored
            print(f"  {'OVERALL':<12}{micro_e:>8.1%}{micro_f:>9.1%}")
        else:
            print(f"  {'OVERALL':<12}{'-':>8}{'-':>9}   (no ground truth supplied)")


# --------------------------------------------------------------------------
# OCR cache
# --------------------------------------------------------------------------

class OcrCache:
    """Disk cache so an ablation re-uses OCR instead of repeating it."""

    def __init__(self, dataset: str, split: str, variant: str):
        key = normalize_key(f"{dataset}-{split}-{variant}").replace(" ", "_")
        self.path = Path(CACHE_DIR) / f"ocr_{key}.json"
        self.data: dict[str, str] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.data = {}

    def get(self, index) -> str | None:
        return self.data.get(str(index))

    def put(self, index, text: str) -> None:
        self.data[str(index)] = text

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False),
                             encoding="utf-8")


# --------------------------------------------------------------------------

CONFIGS = {
    "rules":  dict(use_rules=True,  use_llm=False),
    "llm":    dict(use_rules=False, use_llm=True),
    "hybrid": dict(use_rules=True,  use_llm=True),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="amohseni/receipt_VLM_information_extraction")
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", default=None,
                    help="evaluate a LOCAL folder of receipt images instead of the "
                         "HuggingFace dataset")
    ap.add_argument("--labels", default=None,
                    help="ground truth for --images: a .json or .csv file "
                         "(omit to report coverage only)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--config", default="hybrid",
                    choices=[*CONFIGS, "all"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--ocr-variant", default="auto")
    ap.add_argument("--ocr-engine", default=OCR_ENGINE,
                    choices=["easyocr", "tesseract"])
    ap.add_argument("--report", default=None, help="write per-receipt results as JSON")
    args = ap.parse_args(argv)

    print(f"runtime: {describe_runtime()}")
    if args.images:
        items, source, available = load_local_items(args.images, args.labels, args.limit)
        if not args.labels:
            print("note: no --labels given; EXACT/FUZZY will be 0 and only "
                  "FOUND (coverage) is meaningful")
    else:
        try:
            items, source, available = load_hf_items(args.dataset, args.split, args.limit)
        except ImportError:
            print("error: pip install datasets")
            return 1
    if not items:
        print(f"no receipts found in {source}")
        return 1

    n = len(items)
    print(f"source: {source}")
    print(f"evaluating {n} of {available} receipts\n")

    # ---- stage 1: OCR once, cached and shared across configurations -------
    reader = make_ocr(args.ocr_engine)
    # The engine is part of the cache key, so switching backends does not
    # silently reuse the other one's text.
    cache = OcrCache(source, args.split if not args.images else "local",
                     f"{args.ocr_engine}-{args.ocr_variant}")
    texts, truths = [], []
    for i, (key, image, truth) in enumerate(items):
        truths.append(truth)
        cached = cache.get(key)
        if cached is None:
            t0 = time.time()
            cached = reader.read(image, variant=args.ocr_variant).text
            cache.put(key, cached)
            print(f"  ocr {i + 1}/{n} ({time.time() - t0:.1f}s)", end="\r")
        texts.append(cached)
    cache.save()
    print(f"  ocr complete ({len(cache.data)} cached)      ")

    # ---- stage 2: run each NER configuration over the same text ----------
    configs = list(CONFIGS) if args.config == "all" else [args.config]
    boards, rows = [], {}
    for name in configs:
        board = Scoreboard(name)
        ner = HybridNER(**CONFIGS[name], model=args.model, mode="header",
                        verbose=False, cache=True)
        print(f"\nrunning config: {name}")
        detail = []
        for i, (text, truth) in enumerate(zip(texts, truths)):
            t0 = time.time()
            _, fields, _, _ = ner.extract(text)
            board.seconds += time.time() - t0
            scored = board.update(fields, truth)
            scored["receipt"] = items[i][0]
            detail.append(scored)
            print(f"  {i + 1}/{n}", end="\r")
        print(f"  {n}/{n} done       ")
        boards.append(board)
        rows[name] = detail

    print(f"\n{'=' * 60}\nRESULTS  ({n} receipts, OCR variant={args.ocr_variant})\n{'=' * 60}")
    for board in boards:
        board.report()

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"source": source, "dataset": args.dataset, "split": args.split,
             "images": args.images, "n": n, "configs": rows},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
