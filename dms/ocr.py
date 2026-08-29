"""OCR stage: image preprocessing, EasyOCR (Malay + English), line rebuilding.

Two things here matter more than the choice of OCR engine:

1. **Adaptive preprocessing.** No single binarisation wins on every receipt -
   thermal paper, faded ink and glossy photos each prefer a different one. We
   run a handful of variants and keep whichever reads back the most confident
   text, rather than hard-coding one pipeline and hoping.

2. **Line reconstruction.** EasyOCR returns loose boxes in reading-ish order.
   Flattening them into one string destroys the two-dimensional layout that a
   receipt encodes - and on a receipt the layout *is* the meaning ("JUMLAH
   BESAR" on the left, "60.31" on the right of the *same* line). We regroup the
   boxes into visual lines by their vertical centres before any NER runs.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from dms.config import (
    OCR_AUTO_VARIANTS,
    OCR_LANGS,
    OCR_LINE_TOLERANCE,
    OCR_MIN_CONF,
)
from dms.schema import OcrResult

# --------------------------------------------------------------------------
# Preprocessing variants
# --------------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _upscale_if_small(img: np.ndarray, min_side: int = 1000) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) >= min_side:
        return img
    scale = min_side / max(h, w)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def pre_raw(img: np.ndarray) -> np.ndarray:
    """No preprocessing - EasyOCR's own normalisation often wins on clean photos."""
    return _upscale_if_small(img)


def pre_gray_otsu(img: np.ndarray) -> np.ndarray:
    """Grayscale -> Gaussian blur -> Otsu. Strong on flat, high-contrast scans."""
    gray = _to_gray(_upscale_if_small(img))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, out = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def pre_adaptive(img: np.ndarray) -> np.ndarray:
    """Adaptive threshold - handles uneven lighting / shadow across the paper."""
    gray = _to_gray(_upscale_if_small(img))
    gray = cv2.medianBlur(gray, 3)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )


def pre_clahe_sharp(img: np.ndarray) -> np.ndarray:
    """Contrast-limited equalisation + unsharp mask. Rescues faded thermal ink."""
    gray = _to_gray(_upscale_if_small(img))
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (0, 0), 3)
    return cv2.addWeighted(eq, 1.6, blur, -0.6, 0)


def deskew(img: np.ndarray) -> np.ndarray:
    """Rotate a tilted receipt upright using the text mass's minimum-area box."""
    gray = _to_gray(img)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thr)
    if coords is None:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    # Only correct a genuine, modest tilt; anything larger is probably a
    # mis-measurement and rotating would make the read worse.
    if not (0.4 <= abs(angle) <= 20.0):
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def pre_deskew_otsu(img: np.ndarray) -> np.ndarray:
    return pre_gray_otsu(deskew(img))


VARIANTS = {
    "raw": pre_raw,
    "gray_otsu": pre_gray_otsu,
    "adaptive": pre_adaptive,
    "clahe_sharp": pre_clahe_sharp,
    "deskew_otsu": pre_deskew_otsu,
}


# --------------------------------------------------------------------------
# Line reconstruction
# --------------------------------------------------------------------------

def _bbox_geometry(bbox) -> tuple[float, float, float]:
    """Return ``(y_center, x_left, height)`` for an EasyOCR quadrilateral."""
    pts = np.asarray(bbox, dtype=float)
    ys, xs = pts[:, 1], pts[:, 0]
    return float(ys.mean()), float(xs.min()), float(ys.max() - ys.min())


def group_into_lines(results, min_conf: float = OCR_MIN_CONF,
                     tolerance: float = OCR_LINE_TOLERANCE):
    """Regroup EasyOCR boxes into visual lines.

    Returns ``(lines, tokens)`` where ``lines`` is a list of strings in
    top-to-bottom order and ``tokens`` carries per-box metadata.
    """
    boxes = []
    for bbox, text, conf in results:
        text = (text or "").strip()
        if not text or conf < min_conf:
            continue
        yc, xl, h = _bbox_geometry(bbox)
        boxes.append({
            "text": text,
            "conf": float(conf),
            # EasyOCR hands back numpy int32 corners; cast to plain ints so the
            # whole document stays JSON-serialisable on its way to SQLite.
            "bbox": [[int(round(float(px))), int(round(float(py)))] for px, py in bbox],
            "y": yc, "x": xl, "h": h,
        })
    if not boxes:
        return [], []

    median_h = float(np.median([b["h"] for b in boxes])) or 10.0
    threshold = max(median_h * tolerance, 4.0)

    boxes.sort(key=lambda b: b["y"])
    lines: list[list[dict]] = [[boxes[0]]]
    for box in boxes[1:]:
        anchor = np.mean([b["y"] for b in lines[-1]])
        if abs(box["y"] - anchor) <= threshold:
            lines[-1].append(box)
        else:
            lines.append([box])

    # Build the text and record where every box landed in it. Those offsets are
    # what later lets an entity's character span be traced back to the polygons
    # it came from, so a hit can be boxed on the original image.
    out_lines, out_tokens = [], []
    cursor = 0
    for i, line in enumerate(lines):
        line.sort(key=lambda b: b["x"])
        parts, local = [], 0
        for j, b in enumerate(line):
            if j:
                local += 1                       # the joining space
            b["line"] = i
            b["char_start"] = cursor + local
            b["char_end"] = cursor + local + len(b["text"])
            local += len(b["text"])
            parts.append(b["text"])
            out_tokens.append(b)
        line_text = " ".join(parts)
        out_lines.append(line_text)
        cursor += len(line_text) + 1             # the joining newline
    return out_lines, out_tokens


def polygons_for_span(tokens: list[dict], start: int, end: int) -> list[list]:
    """Every OCR polygon overlapping the character span ``[start, end)``."""
    if start < 0 or end <= start:
        return []
    return [
        t["bbox"] for t in tokens
        if t.get("char_start") is not None
        and start < t["char_end"] and end > t["char_start"]
    ]


def draw_entity_boxes(image, entities, colour_map: dict | None = None,
                      thickness: int = 3):
    """Return a copy of ``image`` with each entity's box drawn and labelled.

    ``entities`` may be Entity objects or the dicts the database returns; either
    way the box is read from ``meta["bbox"]``, which is in the coordinate frame
    of the image as OCR saw it (see the caveat in docs/ASSIGNMENT_REPORT.md (S3.7)).
    """
    img = ReceiptOCR._load(image).copy()
    palette = colour_map or {}
    default = (0, 170, 255)

    for ent in entities:
        meta = ent.get("meta") if isinstance(ent, dict) else getattr(ent, "meta", {})
        etype = ent.get("type") if isinstance(ent, dict) else getattr(ent, "type", "")
        box = (meta or {}).get("bbox")
        if not box:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        colour = palette.get(etype, default)
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)
        # Label above the box, or inside it when the box is at the very top.
        ty = y1 - 6 if y1 > 18 else y2 + 16
        cv2.putText(img, etype, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colour, 1, cv2.LINE_AA)
    return img


def bounding_box(polygons: list[list]) -> list[int] | None:
    """Collapse polygons into one axis-aligned ``[x1, y1, x2, y2]`` rectangle."""
    if not polygons:
        return None
    pts = np.concatenate([np.asarray(p, dtype=float) for p in polygons], axis=0)
    return [int(pts[:, 0].min()), int(pts[:, 1].min()),
            int(pts[:, 0].max()), int(pts[:, 1].max())]


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------

class ReceiptOCR:
    """Lazily-initialised EasyOCR reader configured for Malay + English."""

    _reader = None       # class-level: the model is expensive, load it once

    def __init__(self, langs: list[str] | None = None, gpu: bool | None = None):
        self.langs = langs or OCR_LANGS
        if gpu is None:
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False
        self.gpu = bool(gpu)

    @property
    def reader(self):
        if ReceiptOCR._reader is None:
            import easyocr
            ReceiptOCR._reader = easyocr.Reader(self.langs, gpu=self.gpu, verbose=False)
        return ReceiptOCR._reader

    # -- internals ------------------------------------------------------
    @staticmethod
    def _score(results) -> float:
        """Expected number of correctly-read tokens.

        Summing confidences rewards reading *more* text and reading it *well*,
        which mean-confidence alone does not (a variant that finds one crisp
        word would otherwise beat one that finds the whole receipt).
        """
        return float(sum(c for _, _, c in results if c >= OCR_MIN_CONF))

    def _run_variant(self, img: np.ndarray, name: str):
        processed = VARIANTS[name](img)
        return self.reader.readtext(processed)

    # -- public API -----------------------------------------------------
    def read(self, image, variant: str = "auto") -> OcrResult:
        """OCR ``image`` (path, numpy array or PIL image) into an :class:`OcrResult`."""
        img = self._load(image)
        t0 = time.time()

        if variant == "auto":
            candidates = OCR_AUTO_VARIANTS
        elif variant == "all":
            candidates = list(VARIANTS)
        else:
            candidates = [variant]

        best_name, best_results, best_score = "", [], -1.0
        attempts: dict[str, float] = {}
        for name in candidates:
            if name not in VARIANTS:
                raise ValueError(
                    f"unknown OCR variant {name!r}; choose from {sorted(VARIANTS)}"
                )
            try:
                results = self._run_variant(img, name)
            except Exception as exc:                      # a bad variant must not
                attempts[name] = -1.0                     # sink the whole read
                print(f"  [ocr] variant {name} failed: {exc}")
                continue
            score = self._score(results)
            attempts[name] = round(score, 2)
            if score > best_score:
                best_name, best_results, best_score = name, results, score

        lines, tokens = group_into_lines(best_results)
        confs = [t["conf"] for t in tokens]
        return OcrResult(
            text="\n".join(lines),
            lines=lines,
            tokens=tokens,
            mean_conf=float(np.mean(confs)) if confs else 0.0,
            score=max(best_score, 0.0),
            variant=best_name,
            attempts=attempts | {"_seconds": round(time.time() - t0, 2)},
        )

    @staticmethod
    def _load_image(image) -> np.ndarray:
        return ReceiptOCR._load(image)

    @staticmethod
    def _load(image) -> np.ndarray:
        """Accept a path, a numpy array or a PIL image; return BGR numpy."""
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"receipt image not found: {path}")
            # imdecode (not imread) so non-ASCII Windows paths work.
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"could not decode image: {path}")
            return img
        # PIL / datasets.Image
        try:
            arr = np.array(image.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise TypeError(f"unsupported image input: {type(image)}") from exc


# --------------------------------------------------------------------------
# Tesseract backend
# --------------------------------------------------------------------------

def _find_tesseract() -> str | None:
    """Locate the Tesseract executable (it is a binary, not a Python package)."""
    import os
    import shutil

    env = os.getenv("TESSERACT_CMD")
    if env and Path(env).exists():
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        "/usr/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ):
        if Path(candidate).exists():
            return candidate
    return None


class TesseractOCR:
    """Tesseract backend exposing the same interface as :class:`ReceiptOCR`.

    Measured on 20 receipts from this dataset, Tesseract recovered the four
    annotated fields slightly more often than EasyOCR (91.2% vs 87.5%), and was
    markedly better on long address lines. It is offered as a first-class
    alternative for that reason - use ``--ocr-engine tesseract``.

    Word boxes come from ``image_to_data``, so entities extracted from
    Tesseract output can still be highlighted on the image.
    """

    # psm 6 = one uniform block of text, psm 4 = a single column of varying
    # sizes. Both suit a receipt; whichever reads better is kept.
    CONFIGS = ["--oem 3 --psm 6", "--oem 3 --psm 4"]

    def __init__(self, langs: list[str] | None = None, **_ignored):
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract is not installed - run `pip install pytesseract` "
                "(the Tesseract program itself must also be installed)"
            ) from exc

        exe = _find_tesseract()
        if exe is None:
            raise RuntimeError(
                "the Tesseract executable was not found. Install it (Windows: "
                "the UB Mannheim build) or set the TESSERACT_CMD environment "
                "variable to its full path."
            )
        pytesseract.pytesseract.tesseract_cmd = exe
        self.pt = pytesseract
        self.exe = exe
        # Tesseract language codes differ from EasyOCR's: 'msa', not 'ms'.
        mapping = {"ms": "msa", "en": "eng"}
        self.lang = "+".join(mapping.get(code, code)
                             for code in (langs or OCR_LANGS))

    def _read_once(self, processed: np.ndarray, config: str):
        from PIL import Image

        pil = Image.fromarray(
            processed if processed.ndim == 2 else processed[:, :, ::-1]
        )
        data = self.pt.image_to_data(pil, lang=self.lang, config=config,
                                     output_type=self.pt.Output.DICT)
        results = []
        for i, word in enumerate(data["text"]):
            word = (word or "").strip()
            if not word:
                continue
            try:
                conf = float(data["conf"][i]) / 100.0
            except (TypeError, ValueError):
                continue
            if conf < 0:
                continue
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            # Same quadrilateral shape EasyOCR returns, so everything
            # downstream (line grouping, polygons) works unchanged.
            bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            results.append((bbox, word, conf))
        return results

    def read(self, image, variant: str = "auto") -> OcrResult:
        img = ReceiptOCR._load(image)
        t0 = time.time()

        if variant == "auto":
            candidates = OCR_AUTO_VARIANTS
        elif variant == "all":
            candidates = list(VARIANTS)
        else:
            candidates = [variant]

        best_name, best_results, best_score = "", [], -1.0
        attempts: dict[str, float] = {}
        for name in candidates:
            if name not in VARIANTS:
                raise ValueError(
                    f"unknown OCR variant {name!r}; choose from {sorted(VARIANTS)}"
                )
            processed = VARIANTS[name](img)
            for config in self.CONFIGS:
                try:
                    results = self._read_once(processed, config)
                except Exception as exc:
                    print(f"  [ocr] tesseract {name}/{config} failed: {exc}")
                    continue
                score = float(sum(c for _, _, c in results if c >= OCR_MIN_CONF))
                key = f"{name}{config[-6:]}"
                attempts[key] = round(score, 2)
                if score > best_score:
                    best_name, best_results, best_score = key, results, score

        lines, tokens = group_into_lines(best_results)
        confs = [t["conf"] for t in tokens]
        return OcrResult(
            text="\n".join(lines),
            lines=lines,
            tokens=tokens,
            mean_conf=float(np.mean(confs)) if confs else 0.0,
            score=max(best_score, 0.0),
            variant=best_name,
            attempts=attempts | {"_seconds": round(time.time() - t0, 2)},
        )


OCR_ENGINES = {"easyocr": ReceiptOCR, "tesseract": TesseractOCR}


def make_ocr(engine: str = "easyocr", **kwargs):
    """Build an OCR backend by name (``easyocr`` or ``tesseract``)."""
    if engine not in OCR_ENGINES:
        raise ValueError(
            f"unknown OCR engine {engine!r}; choose from {sorted(OCR_ENGINES)}"
        )
    return OCR_ENGINES[engine](**kwargs)
