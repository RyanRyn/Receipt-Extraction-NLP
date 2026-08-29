"""End-to-end orchestration: image -> OCR -> hybrid NER -> SQLite."""

from __future__ import annotations

import time
from pathlib import Path

from dms.database import ReceiptDB
from dms.ner import HybridNER
from dms.ocr import bounding_box, make_ocr, polygons_for_span
from dms.schema import ReceiptDocument
from dms.textutils import detect_language

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class Pipeline:
    """Holds the OCR reader, the NER stack and the database open across a batch.

    Reusing one instance matters: the OCR model and the LLM weights each take
    tens of seconds to load, so a per-image pipeline would spend all its time
    loading rather than reading.
    """

    def __init__(self, use_rules: bool = True, use_llm: bool = True,
                 model: str | None = None, ocr_variant: str = "auto",
                 ner_mode: str = "full", db_path: str | None = None,
                 store: bool = True, verbose: bool = True,
                 ocr_engine: str = "easyocr", embed: bool = True,
                 embed_model: str | None = None):
        self.ocr = make_ocr(ocr_engine)
        self.ocr_engine = ocr_engine
        self.ner = HybridNER(use_rules=use_rules, use_llm=use_llm,
                             model=model, mode=ner_mode, verbose=verbose)
        self.ocr_variant = ocr_variant
        self.store = store
        self.verbose = verbose
        self.embed = embed
        self.db = (ReceiptDB(db_path, embed_model=embed_model, verbose=verbose)
                   if store else None)

    # ------------------------------------------------------------------
    def process(self, image, filename: str | None = None,
                path: str | None = None) -> ReceiptDocument:
        """Run the full pipeline on one receipt image.

        ``path`` overrides the stored source location, which lets in-memory
        images (e.g. straight from a HuggingFace dataset) carry a stable
        identity instead of colliding on an empty string.
        """
        if path is None:
            path = str(image) if isinstance(image, (str, Path)) else ""
        name = filename or (Path(path).name if path else "in-memory")

        if self.verbose:
            print(f"\n[1/3] OCR  : {name}  [{self.ocr_engine}]")
        t0 = time.time()
        ocr = self.ocr.read(image, variant=self.ocr_variant)
        t_ocr = time.time() - t0
        if self.verbose:
            print(f"      variant={ocr.variant} lines={len(ocr.lines)} "
                  f"mean_conf={ocr.mean_conf:.3f} ({t_ocr:.1f}s)")

        if self.verbose:
            layers = " + ".join(
                n for n, on in (("rules", self.ner.use_rules),
                                ("local LLM", self.ner.use_llm)) if on
            ) or "none"
            print(f"[2/3] NER  : {layers}")
        t0 = time.time()
        entities, fields, items, warnings = self.ner.extract(ocr.text)
        t_ner = time.time() - t0

        # Trace each entity back to the OCR polygons it came from.
        for ent in entities:
            if not ent.is_aligned:
                continue
            polys = polygons_for_span(ocr.tokens, ent.start, ent.end)
            if polys:
                ent.meta = dict(ent.meta or {},
                                bbox=bounding_box(polys),
                                lines=sorted({t["line"] for t in ocr.tokens
                                              if ent.start < t["char_end"]
                                              and ent.end > t["char_start"]}))

        doc = ReceiptDocument(
            filename=name,
            path=path,
            ocr_text=ocr.text,
            ocr_conf=ocr.mean_conf,
            ocr_variant=ocr.variant,
            language=detect_language(ocr.text),
            entities=entities,
            fields=fields,
            items=items,
            tokens=[{k: t[k] for k in ("text", "conf", "bbox", "line",
                                       "char_start", "char_end")}
                    for t in ocr.tokens],
            warnings=warnings,
            timings={"ocr": round(t_ocr, 2), "ner": round(t_ner, 2)},
        )

        if self.store and self.db is not None:
            doc.doc_id = self.db.add_document(doc, embed=self.embed)
            if self.verbose:
                vectors = "+vectors" if self.embed and self.db.embedder else ""
                print(f"[3/3] DB   : stored as doc_id={doc.doc_id} "
                      f"({len(entities)} entities{vectors})")
        elif self.verbose:
            print("[3/3] DB   : skipped (store=False)")
        return doc

    # ------------------------------------------------------------------
    def process_many(self, images, limit: int | None = None) -> list[ReceiptDocument]:
        docs = []
        for i, img in enumerate(images):
            if limit is not None and i >= limit:
                break
            try:
                docs.append(self.process(img))
            except Exception as exc:
                print(f"  !! failed on {img}: {type(exc).__name__}: {exc}")
        return docs

    def close(self) -> None:
        if self.db is not None:
            self.db.close()


# --------------------------------------------------------------------------
# convenience wrappers
# --------------------------------------------------------------------------

def process_receipt(image, **kwargs) -> ReceiptDocument:
    """Process a single receipt image. See :class:`Pipeline` for options."""
    pipe = Pipeline(**kwargs)
    try:
        return pipe.process(image)
    finally:
        pipe.close()


def process_folder(folder: str | Path, limit: int | None = None,
                   **kwargs) -> list[ReceiptDocument]:
    """Process every image in ``folder``."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(root)
    images = sorted(p for p in root.iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"no images found in {root}")
        return []
    pipe = Pipeline(**kwargs)
    try:
        return pipe.process_many(images, limit=limit)
    finally:
        pipe.close()
