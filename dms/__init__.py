"""
Receipt Document Management System (DMS) with Named Entity Recognition.

A bilingual (Bahasa Melayu / English) pipeline:

    image -> OCR -> hybrid NER (rules + local LLM) -> SQLite (searchable)

Public entry point:
    >>> from dms import process_receipt
    >>> doc = process_receipt("sample_receipt.jpg")
    >>> doc.fields["total"]
"""

from dms.schema import Entity, ReceiptDocument, ENTITY_TYPES
from dms.pipeline import process_receipt, process_folder

__all__ = [
    "Entity",
    "ReceiptDocument",
    "ENTITY_TYPES",
    "process_receipt",
    "process_folder",
]

__version__ = "1.0.0"
