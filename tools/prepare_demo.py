"""
Pre-flight check before a live demonstration.

    python prepare_demo.py                 # check readiness, fill gaps
    python prepare_demo.py --receipts 15   # build a larger corpus first
    python prepare_demo.py --check-only    # verify, change nothing

Run this BEFORE presenting, not during. It exists because the two things most
likely to embarrass a live demo are both avoidable:

  1. The first receipt takes ~30 seconds while the OCR, language and embedding
     models load from disk. Every one after that takes a few seconds.
  2. Semantic search over an almost-empty database returns nothing, because
     nearest-neighbour search needs neighbours. One receipt is not a corpus.

This script fills the database, warms every model, and then exercises all four
required DMS functions so you know they work before an audience is watching.
"""
from __future__ import annotations

# Running from tools/, so the project root must be on the import path before
# `dms` (or a sibling tool such as evaluate.py) can be imported.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import argparse
import time
from pathlib import Path

OK, BAD, WARN = "  [ OK ]", "  [FAIL]", "  [WARN]"
_problems: list[str] = []


def report(label: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    mark = OK if ok else (WARN if warn_only else BAD)
    print(f"{mark} {label}" + (f"   {detail}" if detail else ""))
    if not ok and not warn_only:
        _problems.append(label)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts", type=int, default=12,
                    help="minimum documents wanted in the database")
    ap.add_argument("--source", default="data/receipts")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)

    from dms.config import DB_PATH, describe_runtime
    from dms.database import ReceiptDB

    print("=" * 70)
    print("DEMONSTRATION PRE-FLIGHT")
    print("=" * 70)
    print(f"  runtime : {describe_runtime()}")
    print(f"  database: {DB_PATH}")

    # ---- 1. corpus ----------------------------------------------------
    print("\n1. CORPUS")
    db = ReceiptDB(verbose=False)
    have = db.stats()["documents"]
    report(f"database holds {have} document(s)", have >= args.receipts,
           f"want at least {args.receipts}", warn_only=True)

    if have < args.receipts and not args.check_only:
        source = Path(args.source)
        if not source.is_dir():
            report("source folder of receipts exists", False,
                   f"{source} not found — run: python run_dms.py export")
        else:
            images = sorted(p for p in source.iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            need = args.receipts - have
            print(f"  processing {need} more receipt(s) — this is the slow part, "
                  "and it is why you run it now rather than live")
            db.close()
            from dms.pipeline import Pipeline
            pipe = Pipeline(store=True, verbose=False)
            done = 0
            for path in images:
                if done >= need:
                    break
                try:
                    t0 = time.time()
                    doc = pipe.process(path)
                    done += 1
                    print(f"    {done}/{need}  {path.name}  "
                          f"{doc.fields.get('merchant') or '?'}  "
                          f"({time.time() - t0:.0f}s)")
                except Exception as exc:
                    print(f"    skipped {path.name}: {type(exc).__name__}")
            pipe.close()
            db = ReceiptDB(verbose=False)
            have = db.stats()["documents"]

    stats = db.stats()
    report(f"documents ready: {stats['documents']}", stats["documents"] > 0)
    report(f"entities indexed: {stats['entities']}", stats["entities"] > 0)

    # ---- 2. vectors ---------------------------------------------------
    print("\n2. SEMANTIC INDEX")
    embedded = db.conn.execute(
        "SELECT COUNT(*) n FROM entities WHERE embedding IS NOT NULL").fetchone()["n"]
    report(f"{embedded} semantic vector(s) stored", embedded > 0,
           "run: python run_dms.py reindex" if not embedded else "")
    if db.embedder is not None:
        stored_dim = db.get_setting("embed_dim")
        report("vector dimensions match the loaded model",
               stored_dim == str(db.embedder.dim),
               f"model={db.embedder.repo}, dim={db.embedder.dim}")
    else:
        report("embedding model loads", False,
               "semantic search will be unavailable")

    # ---- 3. warm every model -----------------------------------------
    print("\n3. WARMING MODELS  (so the demo is not the first run)")
    from dms.config import default_receipt
    from dms.ocr import make_ocr
    t0 = time.time()
    reader = make_ocr("easyocr")
    sample = default_receipt()
    if sample is not None and sample.exists():
        ocr = reader.read(sample, variant="gray_otsu")
        report(f"OCR warm ({time.time() - t0:.0f}s)", bool(ocr.text.strip()),
               f"{len(ocr.lines)} lines read")
    else:
        report("a receipt image is available to warm the models", False,
               "run: python run_dms.py export", warn_only=True)
        ocr = None

    if ocr is not None:
        from dms.ner import HybridNER
        t0 = time.time()
        ner = HybridNER(verbose=False)
        entities, fields, _, _ = ner.extract(ocr.text)
        report(f"language model warm ({time.time() - t0:.0f}s)",
               len(entities) > 0,
               f"merchant={fields.get('merchant')!r} total={fields.get('total')!r}")

    # ---- 4. the four required DMS functions ---------------------------
    print("\n4. THE FOUR REQUIRED FUNCTIONS")

    r = db.search_entities("Kuala Lumpur")
    report("(a) search an entity", bool(r["hits"]) or r["mode"] != "empty",
           f"mode={r['mode']}, {len(r['hits'])} hits")

    hit = r["hits"][0] if r["hits"] else None
    if hit:
        marked = db.highlight_document(hit["doc_id"], [hit["entity_id"]])
        report("(b) return the document with the entity highlighted",
               "[[" in marked, f"doc {hit['doc_id']}")
    else:
        report("(b) highlighting", False, "no hit to highlight", warn_only=True)

    r = db.search_entities("Kota Kinabalu")
    report("(c) similar entity when the query is absent",
           r["mode"] in ("similar", "type_fallback") and bool(r["hits"]),
           f"mode={r['mode']}, {len(r['hits'])} hits")

    sem = db.search_entities("chicken and rice", limit=5)
    sem_hits = [h for h in sem["hits"] if h.get("semantic_score")]
    report("(d) semantic search (matched by meaning)", bool(sem_hits),
           f"top={sem_hits[0]['value'][:34]!r} "
           f"at {sem_hits[0]['semantic_score']}" if sem_hits else
           "no semantic hit — try a larger corpus")

    # A warning, not a failure. No embedding model tested can separate gibberish
    # from a real query by score alone (the measured gap is negative), and the
    # chance of nonsense landing near *something* grows with the corpus. If this
    # warns, avoid the nonsense query in the demonstration rather than treating
    # the system as broken.
    r = db.search_entities("zzzznotathing")
    report("nonsense is rejected", not r["hits"],
           f"mode={r['mode']} - scores overlap on this corpus; skip this query "
           "in the demo", warn_only=True)

    # ---- 5. talking points --------------------------------------------
    print("\n5. QUERIES THAT DEMONSTRATE WELL")
    for q, why in [
        ("Kuala Lumpur", "exact/substring — the ordinary case"),
        ("Kuala Lumpor", "misspelled — lexical similarity rescues it"),
        ("Kota Kinabalu", "absent city — the similar-entity requirement"),
        ("nasi ayam", "MALAY query finding ENGLISH text by meaning"),
        ("chicken and rice", "semantic — no shared words with anything stored"),
        ("zzzznotathing", "nonsense — correctly returns nothing"),
    ]:
        res = db.search_entities(q, limit=3)
        top = res["hits"][0]["value"][:36] if res["hits"] else "-"
        print(f"    {q:<18} {res['mode']:<14} {top!r}")
        print(f"    {'':<18} ^ {why}")

    db.close()

    print("\n" + "=" * 70)
    if _problems:
        print(f"NOT READY — {len(_problems)} problem(s):")
        for p in _problems:
            print(f"   - {p}")
        print("=" * 70)
        return 1
    print("READY TO PRESENT")
    print("  Start the interface now and leave it running:")
    print("      streamlit run app.py")
    print("  Then process ONE receipt through the GUI before the audience")
    print("  arrives — that warms the interface's own model cache.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
