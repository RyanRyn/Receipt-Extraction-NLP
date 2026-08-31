"""
Trace one receipt from the LLM's raw JSON through to the database rows.

    python trace_llm_to_db.py
    python trace_llm_to_db.py --image data/receipts/test_00000.jpg

Shows every transformation between "the model emitted some text" and "a row
exists in SQLite", because that path is where most of the engineering lives and
none of it is visible from the outside.

Writes to a throwaway database, so nothing you already have is affected.
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
import os
import tempfile
from pathlib import Path


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="receipt to trace (default: sample_receipt.jpg, or "
                         "the first image in the exported corpus)")
    args = ap.parse_args(argv)

    tmp = Path(tempfile.mkdtemp()) / "trace.sqlite3"
    os.environ["DMS_DB"] = str(tmp)

    from dms.config import default_receipt
    from dms.database import ReceiptDB
    from dms.llm import LocalLLMExtractor, parse_json_object
    from dms.ner import HybridNER, _llm_to_entities
    from dms.ocr import bounding_box, make_ocr, polygons_for_span
    from dms.schema import ReceiptDocument
    from dms.textutils import detect_language, normalize_key

    image = Path(args.image) if args.image else default_receipt()
    if image is None or not image.exists():
        print("no receipt image found. Pass --image PATH, or run "
              "`python run_dms.py export` to create data/receipts/.")
        return 1

    # ---------------------------------------------------------------- A
    rule("A. OCR — the text the model will be shown")
    ocr = make_ocr().read(image)
    print(ocr.text)

    # ---------------------------------------------------------------- B
    rule("B. What the LLM literally emits (raw generated text)")
    llm = LocalLLMExtractor(verbose=False)
    tok, model = llm._ensure_loaded()

    import torch
    prompt = llm._apply_template(tok, llm._build_messages(ocr.text, "full"))
    enc = tok(prompt, return_tensors="pt").to(llm.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=700, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    reply = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(reply)
    print(f"\n  -> {len(reply)} characters of TEXT. Not a dict, not a row — text.")

    # ---------------------------------------------------------------- C
    rule("C. parse_json_object() — text becomes a Python dict")
    data = parse_json_object(reply)
    if data is None:
        print("  the reply could not be parsed at all")
        return 1
    for key, value in data.items():
        if key.startswith("_"):
            continue
        print(f"  {key:<16} {type(value).__name__:<6} {str(value)[:52]!r}")

    # ---------------------------------------------------------------- D
    rule("D. _llm_to_entities() — dict becomes Entity objects")
    print("  Per field: normalise the value, apply two vetoes, then locate it")
    print("  in the OCR text to recover character offsets.\n")
    llm_entities = _llm_to_entities(data, ocr.text, verify_with_rules=True)

    kept = {e.type for e in llm_entities}
    from dms.ner import LLM_KEY_TO_TYPE
    for key, etype in LLM_KEY_TO_TYPE.items():
        raw = data.get(key)
        if raw is None:
            continue
        ent = next((e for e in llm_entities if e.type == etype), None)
        if ent is None:
            print(f"  {etype:<15} {str(raw)[:26]!r:<28} -> VETOED (not printed "
                  f"on the receipt, or implausible)")
        else:
            span = f"chars {ent.start}-{ent.end}" if ent.is_aligned else "NOT ALIGNED"
            print(f"  {etype:<15} {str(raw)[:26]!r:<28} -> value={ent.value!r} "
                  f"| {span}")
            if ent.is_aligned and ent.text != ent.value:
                print(f"  {'':<15} {'':<28}    printed as {ent.text!r} "
                      "<- cleaned value re-anchored onto the messy original")

    # ---------------------------------------------------------------- E
    rule("E. Merge with the rule layer, then validate")
    entities, fields, items, warnings = HybridNER(verbose=False).extract(ocr.text)
    print(f"  {'TYPE':<15}{'VALUE':<24}{'CONF':>6}  SOURCE")
    for e in entities:
        print(f"  {e.type:<15}{str(e.value)[:22]:<24}{e.confidence:>6.2f}  {e.source}")
    if warnings:
        print("\n  validators fired:")
        for w in warnings:
            print(f"    ! {w}")

    # ---------------------------------------------------------------- F
    rule("F. Attach pixel boxes (character span -> image coordinates)")
    for e in entities:
        if e.is_aligned:
            polys = polygons_for_span(ocr.tokens, e.start, e.end)
            if polys:
                e.meta = dict(e.meta or {}, bbox=bounding_box(polys))
    boxed = [e for e in entities if (e.meta or {}).get("bbox")]
    for e in boxed[:5]:
        print(f"  {e.type:<15} chars {e.start:>4}-{e.end:<4} -> pixels "
              f"{e.meta['bbox']}")
    print(f"  ({len(boxed)} of {len(entities)} entities have a box)")

    # ---------------------------------------------------------------- G
    rule("G. INSERT — what actually lands in SQLite")
    doc = ReceiptDocument(
        filename=image.name, path=str(image), ocr_text=ocr.text,
        ocr_conf=ocr.mean_conf, ocr_variant=ocr.variant,
        language=detect_language(ocr.text), entities=entities, fields=fields,
        items=items,
        tokens=[{k: t[k] for k in ("text", "conf", "bbox", "line",
                                   "char_start", "char_end")} for t in ocr.tokens],
        warnings=warnings)
    db = ReceiptDB(verbose=False)
    doc_id = db.add_document(doc)

    print("  documents  (1 row)")
    row = db.conn.execute(
        "SELECT id, filename, ocr_variant, language, length(ocr_text) t, "
        "length(tokens_json) k FROM documents WHERE id=?", (doc_id,)).fetchone()
    print(f"    id={row['id']}  file={row['filename']}  variant={row['ocr_variant']}"
          f"  lang={row['language']}  ocr_text={row['t']}ch  tokens={row['k']}B")

    print("\n  entities  (one row per entity)")
    print(f"    {'type':<15}{'value':<20}{'value_norm':<20}{'start':>6}{'end':>5}"
          f"{'conf':>6}  {'source':<16}{'vector':>7}")
    for r in db.conn.execute(
            "SELECT type, value, value_norm, start, end, confidence, source, "
            "length(embedding) emb FROM entities WHERE doc_id=? ORDER BY start",
            (doc_id,)):
        emb = f"{r['emb'] // 4}d" if r["emb"] else "-"
        print(f"    {r['type']:<15}{str(r['value'])[:18]:<20}"
              f"{str(r['value_norm'])[:18]:<20}{r['start']:>6}{r['end']:>5}"
              f"{r['confidence']:>6.2f}  {str(r['source']):<16}{emb:>7}")

    print("\n  Note the three separate columns per entity:")
    print("    value       cleaned, for display and arithmetic")
    print("    value_norm  lowercased and stripped, for matching in SQL")
    print("    start/end   where it sits in ocr_text, for highlighting")
    print("    embedding   1024 floats, natural-language entity types only")

    rule("H. Immediately searchable")
    for q in [fields.get("total"), (fields.get("merchant") or " ").split()[-1]]:
        if not q:
            continue
        r = db.search_entities(str(q), limit=3)
        top = r["hits"][0]["value"][:34] if r["hits"] else "-"
        print(f"  search {str(q)[:22]!r:<26} mode={r['mode']:<10} -> {top!r}")

    db.close()
    print(f"\n(throwaway database: {tmp})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
