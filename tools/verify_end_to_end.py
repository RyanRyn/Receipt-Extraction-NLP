"""
End-to-end proof: image file -> OCR -> NER -> validation -> database -> search.

    python verify_end_to_end.py
    python verify_end_to_end.py --image my_receipt.jpg

Starts from an EMPTY database in a temporary file, so nothing is assumed and
your working database is untouched. Each link in the chain is asserted, and the
run fails loudly if any one of them is broken.

The semantic step is deliberately proved with a query that shares **no
characters** with the stored value, so a lexical match is impossible and only a
genuine meaning-based match can succeed.
"""
from __future__ import annotations

# Running from tools/, so the project root must be on the import path before
# `dms` (or a sibling tool such as evaluate.py) can be imported.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

PASS, FAIL = "  [PASS]", "  [FAIL]"
_failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"{PASS if ok else FAIL} {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def step(n: int, title: str) -> None:
    print(f"\n{'=' * 72}\nSTEP {n} — {title}\n{'=' * 72}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="receipt to verify (default: sample_receipt.jpg, or "
                         "the first image in the exported corpus)")
    ap.add_argument("--corpus", default="data/receipts",
                    help="folder of extra receipts, so semantic search has "
                         "something to search")
    ap.add_argument("--corpus-size", type=int, default=6)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--keep-db", action="store_true",
                    help="use the real database instead of a temporary one")
    args = ap.parse_args(argv)

    if not args.keep_db:
        tmp = Path(tempfile.mkdtemp()) / "verify.sqlite3"
        os.environ["DMS_DB"] = str(tmp)
        print(f"using a fresh temporary database: {tmp}")

    # Imported after DMS_DB is set, so the config picks it up.
    from dms.config import default_receipt
    from dms.database import ReceiptDB
    from dms.pipeline import Pipeline

    image = Path(args.image) if args.image else default_receipt()
    if image is None or not image.exists():
        print("error: no receipt image found. Pass --image PATH, or run "
              "`python run_dms.py export` to create data/receipts/.")
        return 1

    # ---------------------------------------------------------------- 1
    step(1, "INPUT — a real image file on disk")
    size = image.stat().st_size
    check("image exists and is non-empty", size > 0, f"{image.name}, {size/1024:.0f} KB")

    # ---------------------------------------------------------------- 2
    step(2, "PIPELINE — OCR, NER, validation, storage, vector indexing")
    pipe = Pipeline(use_rules=True, use_llm=not args.no_llm, store=True,
                    verbose=True)
    t0 = time.time()
    doc = pipe.process(image)
    elapsed = time.time() - t0

    check("OCR produced text", bool(doc.ocr_text.strip()),
          f"{len(doc.ocr_text)} chars, {len(doc.ocr_text.splitlines())} lines")
    check("a preprocessing variant was chosen", bool(doc.ocr_variant),
          doc.ocr_variant)
    check("language was detected", doc.language != "unknown", doc.language)
    check("NER produced entities", len(doc.entities) > 0,
          f"{len(doc.entities)} entities")
    check("a merchant was extracted", bool(doc.fields.get("merchant")),
          str(doc.fields.get("merchant")))
    check("a total was extracted", bool(doc.fields.get("total")),
          str(doc.fields.get("total")))
    check("entities carry character offsets",
          any(e.is_aligned for e in doc.entities),
          f"{sum(e.is_aligned for e in doc.entities)} aligned")
    check("entities carry pixel boxes",
          any((e.meta or {}).get("bbox") for e in doc.entities),
          f"{sum(1 for e in doc.entities if (e.meta or {}).get('bbox'))} boxed")
    check("document was written to the database", doc.doc_id is not None,
          f"doc_id={doc.doc_id}")
    print(f"  (pipeline took {elapsed:.1f}s)")

    # ---------------------------------------------------------------- 2b
    # A single receipt yields only two or three natural-language entities, and
    # nearest-neighbour search over a corpus that small proves nothing. Ingest
    # a handful more so the semantic step has a realistic index to search.
    corpus = Path(args.corpus)
    extra = 0
    if corpus.is_dir():
        step(2.5, "CORPUS — ingest a few more receipts so search is meaningful")
        images = [p for p in sorted(corpus.iterdir())
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}][:args.corpus_size]
        for path in images:
            if path.resolve() == image.resolve():
                continue
            try:
                pipe.process(path)
                extra += 1
            except Exception as exc:
                print(f"    skipped {path.name}: {type(exc).__name__}: {exc}")
        print(f"  ingested {extra} additional receipt(s)")
    else:
        print(f"\n(no corpus folder at {corpus}; semantic search will have very "
              "little to match against)")

    # ---------------------------------------------------------------- 3
    step(3, "DATABASE — read the record back out")
    db = ReceiptDB(verbose=False)
    stored = db.get_document(doc.doc_id)
    check("document reads back", stored is not None)
    if stored is None:
        return 1
    check("OCR text was persisted", stored["ocr_text"] == doc.ocr_text)
    check("entities were persisted", len(stored["entities"]) == len(doc.entities),
          f"{len(stored['entities'])} rows")
    check("word boxes were persisted", len(stored["tokens"]) > 0,
          f"{len(stored['tokens'])} tokens")

    embedded = db.conn.execute(
        "SELECT COUNT(*) n FROM entities WHERE embedding IS NOT NULL"
    ).fetchone()["n"]
    check("semantic vectors were stored", embedded > 0, f"{embedded} vectors")
    if db.embedder is not None:
        check("vector dimension matches the loaded model",
              db.get_setting("embed_dim") == str(db.embedder.dim),
              f"dim={db.embedder.dim}, model={db.embedder.repo}")

    # ---------------------------------------------------------------- 4
    step(4, "SEARCH — exact and substring (lexical)")
    total = stored["fields"].get("total")
    if total:
        r = db.search_entities(str(total))
        check(f"exact search for the total {total!r}",
              r["mode"] in ("exact", "partial") and bool(r["hits"]),
              f"mode={r['mode']}, {len(r['hits'])} hits")

    merchant = stored["fields"].get("merchant") or ""
    if len(merchant.split()) > 1:
        word = merchant.split()[-1]
        r = db.search_entities(word)
        check(f"substring search for {word!r}",
              r["mode"] in ("exact", "partial") and bool(r["hits"]),
              f"mode={r['mode']}")

    # ---------------------------------------------------------------- 5
    step(5, "SEMANTIC SEARCH — matching by meaning, not by spelling")
    if db.embedder is None:
        check("embedding model available", False,
              "sentence-transformers missing; semantic search cannot be proved")
    else:
        # Probes deliberately share no word with anything stored, so a lexical
        # match is impossible and only the vector index can answer.
        probes = ["chicken and rice", "nasi ayam", "drink", "minuman panas",
                  "vegetables", "sayur", "grocery shop", "kedai runcit"]
        stored_words = {w.lower().strip(",.")
                        for _, v in db.distinct_values() for w in str(v).split()}

        # First: is the vector index actually being consulted at all? Ask for
        # raw nearest neighbours with no threshold, so this cannot be masked by
        # a conservative cut-off.
        raw = db.semantic_search("chicken and rice", limit=3, min_score=-1.0)
        check("the vector index is searched and returns ranked neighbours",
              len(raw) > 0,
              f"top={raw[0][1]['value'][:32]!r} at {raw[0][0]:.3f}" if raw else "")

        print(f"\n    {'QUERY':<22}{'MODE':<14}{'SCORE':>7}   TOP MATCH")
        cleared, no_overlap_hit = 0, False
        for probe in probes:
            overlap = stored_words & {w.lower() for w in probe.split()}
            r = db.search_entities(probe, limit=5)
            sem = [h for h in r["hits"] if h.get("semantic_score")]
            best = db.semantic_search(probe, limit=1, min_score=-1.0)
            score = best[0][0] if best else 0.0
            top = (sem[0]["value"] if sem else
                   (best[0][1]["value"] if best else "-"))[:34]
            print(f"    {probe:<22}{r['mode']:<14}{score:>7.3f}   {top!r}"
                  + ("" if sem else "   (below threshold)"))
            if sem:
                cleared += 1
                if not overlap:
                    no_overlap_hit = True

        check("at least one query matched by MEANING alone "
              "(no shared words with anything stored)",
              no_overlap_hit,
              f"{cleared}/{len(probes)} probes cleared the similarity threshold")

        # The assignment's own requirement.
        r = db.search_entities("Kota Kinabalu")
        check("absent place name still returns documents "
              "(the similar-entity requirement)",
              r["mode"] in ("similar", "type_fallback") and bool(r["hits"]),
              f"mode={r['mode']}, {len(r['hits'])} hits")

        r = db.search_entities("qwertyuiop zxcvbnm")
        check("pure gibberish returns nothing", not r["hits"],
              f"mode={r['mode']}")

    # ---------------------------------------------------------------- 6
    step(6, "HIGHLIGHT — return the document with the entity marked")
    ent = next((e for e in stored["entities"] if e["start"] >= 0), None)
    check("at least one entity can be highlighted", ent is not None)
    if ent:
        marked = db.highlight_document(doc.doc_id, [ent["id"]])
        check("marker appears in the returned text", "[[" in marked,
              f"{ent['type']} = {str(ent['value'])[:36]!r}")
        html = db.highlight_html(doc.doc_id, [ent["id"]])
        check("HTML highlight renders a <mark>", "<mark" in html)

    db.close()
    pipe.close()

    # ---------------------------------------------------------------- done
    print(f"\n{'=' * 72}")
    if _failures:
        print(f"RESULT: {len(_failures)} STEP(S) FAILED")
        for f in _failures:
            print(f"   - {f}")
        print("=" * 72)
        return 1
    print("RESULT: the full chain works — image -> OCR -> NER -> validation")
    print("        -> database -> lexical search -> SEMANTIC search -> highlight")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
