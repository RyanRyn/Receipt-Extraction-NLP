"""Entity / document data model shared by every stage of the pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

# --------------------------------------------------------------------------
# Entity inventory
# --------------------------------------------------------------------------
# Generic NER schemas (PERSON / ORG / LOC) do not describe a receipt: they have
# no notion of "the grand total" or "the SST registration number". The tag set
# below is a receipt-specific refinement, with the coarse CoNLL-style class each
# tag specialises noted alongside so the DMS stays interoperable.
ENTITY_TYPES: dict[str, str] = {
    "MERCHANT":       "ORG",   # trading name of the business
    "ADDRESS":        "LOC",   # premise address
    "PHONE":          "MISC",  # contact number
    "TAX_ID":         "MISC",  # GST / SST registration number
    "INVOICE_NO":     "MISC",  # receipt / invoice identifier
    "DATE":           "TIME",
    "TIME":           "TIME",
    "ITEM":           "PRODUCT",
    "SUBTOTAL":       "MONEY",
    "TAX":            "MONEY",
    "TOTAL":          "MONEY",
    "PAID":           "MONEY",
    "CHANGE":         "MONEY",
    "CURRENCY":       "MISC",
    "PAYMENT_METHOD": "MISC",
    "CASHIER":        "PERSON",
}

# Types whose value is numeric money.
MONEY_TYPES = {"SUBTOTAL", "TAX", "TOTAL", "PAID", "CHANGE"}

# Types the "similar entity" search treats as place-like (see database.py).
LOCATION_TYPES = {"ADDRESS", "MERCHANT"}

# Types worth embedding for semantic search: those whose value is natural
# language and therefore *has* a meaning to compare.
#
# Embedding the rest actively harms retrieval. "30.00", "CASH" and "2024-03-14"
# carry no semantics, but they still produce vectors, and those vectors sit at
# some arbitrary distance from every query - so a search for "Kota Kinabalu"
# came back with a payment amount ranked above the addresses. Money, dates and
# identifiers are already served perfectly by exact and substring matching,
# which is where they belong.
SEMANTIC_TYPES = {"MERCHANT", "ADDRESS", "ITEM", "CASHIER"}


@dataclass
class Entity:
    """A single named entity grounded in the OCR text.

    ``start``/``end`` are character offsets into :attr:`ReceiptDocument.ocr_text`.
    They are what lets the search UI highlight the entity inside the document,
    so they are populated even for values the LLM rewrote (via fuzzy
    re-alignment back onto the source text).
    """

    type: str
    # Normalised value ("60.31", "2024-03-14"), or None when this type was
    # looked for and is not on the receipt. None is deliberately distinct from
    # "": the latter would be an empty value that was nevertheless extracted.
    value: str | None
    text: str = ""                # surface form exactly as it appears in the OCR
    start: int = -1               # char offset, -1 when it could not be aligned
    end: int = -1
    confidence: float = 0.0
    source: str = ""              # "rule" | "llm" | "rule+llm"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def coarse_type(self) -> str:
        return ENTITY_TYPES.get(self.type, "MISC")

    @property
    def is_aligned(self) -> bool:
        return self.start >= 0 and self.end > self.start

    @property
    def is_absent(self) -> bool:
        """Was this type looked for and found to be missing?"""
        return self.value is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OcrResult:
    """Output of the OCR stage."""

    text: str = ""                # line-reconstructed text (what NER consumes)
    lines: list[str] = field(default_factory=list)
    tokens: list[dict] = field(default_factory=list)   # {text, conf, bbox, line}
    mean_conf: float = 0.0
    score: float = 0.0            # expected number of correctly read tokens
    variant: str = ""             # winning preprocessing variant
    attempts: dict[str, float] = field(default_factory=dict)  # variant -> score


@dataclass
class ReceiptDocument:
    """A processed receipt: the OCR text plus everything recognised in it."""

    filename: str = ""
    path: str = ""
    ocr_text: str = ""
    ocr_conf: float = 0.0
    ocr_variant: str = ""
    language: str = "unknown"     # "ms" | "en" | "mixed"
    entities: list[Entity] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)   # flattened key fields
    items: list[dict] = field(default_factory=list)
    # OCR boxes with their char offsets, kept so the search UI can draw a
    # highlight on the original image and not only on the text.
    tokens: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    doc_id: int | None = None

    def by_type(self, etype: str) -> list[Entity]:
        return [e for e in self.entities if e.type == etype]

    def first(self, etype: str) -> Entity | None:
        found = self.by_type(etype)
        return found[0] if found else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
