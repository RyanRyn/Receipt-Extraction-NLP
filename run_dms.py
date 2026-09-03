"""
Receipt DMS - command line interface.

    python run_dms.py process sample_receipt.jpg      # one receipt
    python run_dms.py process receipts/ --limit 10    # a folder
    python run_dms.py process sample_receipt.jpg --no-llm    # rules only, instant
    python run_dms.py dataset --limit 20              # pull the HuggingFace dataset
    python run_dms.py search "Kuala Lumpur"           # entity search
    python run_dms.py search "Johor Bahru"            # similar-entity fallback
    python run_dms.py show 1                          # highlighted document
    python run_dms.py stats                           # what is in the database

The OCR/NER half of the system writes into SQLite; ``search`` and ``show`` are
reference implementations of the search half, provided so the two halves can be
tested together (see docs/ASSIGNMENT_REPORT.md (S3.7)).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dms.config import DB_PATH, describe_runtime
from dms.database import ReceiptDB
from dms.pipeline import IMAGE_SUFFIXES, Pipeline


def _rule(title: str = "") -> None:
    print(f"\n{'=' * 74}")
    if title:
        print(title)
        print("=" * 74)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_process(args) -> int:
    target = Path(args.target)
    if not target.exists():
        print(f"error: not found: {target}", file=sys.stderr)
        return 1

    print(f"runtime: {describe_runtime()}")
    pipeline = Pipeline(
        use_rules=not args.no_rules,
        use_llm=not args.no_llm,
        model=args.model,
        ocr_variant=args.ocr_variant,
        ocr_engine=args.ocr_engine,
        ner_mode="header" if args.header_only else "full",
        store=not args.no_store,
    )
    try:
        if target.is_dir():
            images = sorted(p for p in target.iterdir()
                            if p.suffix.lower() in IMAGE_SUFFIXES)
            if not images:
                print(f"no images in {target}")
                return 1
            # Report what will actually be processed, not what was found:
            # --limit is applied inside process_many, so printing the folder
            # count alone is misleading.
            count = min(args.limit, len(images)) if args.limit else len(images)
            of = f" (of {len(images)} found)" if count < len(images) else ""
            print(f"processing {count} image(s) from {target}{of}")
            docs = pipeline.process_many(images, limit=args.limit)
        else:
            docs = [pipeline.process(target)]

        for doc in docs:
            _rule(f"{doc.filename}   [{doc.language}]  doc_id={doc.doc_id}")
            for key, value in doc.fields.items():
                if value is not None:
                    print(f"  {key:<16}: {value}")
            # Absent fields are null in the record, not dropped from it, so say
            # so here as well: "not printed on this receipt" and "not looked
            # for" are different claims, and a bare list of what was found
            # cannot distinguish them.
            absent = [k for k, v in doc.fields.items() if v is None]
            found = len(doc.fields) - len(absent)
            print(f"  {'-' * 16}   {found}/{len(doc.fields)} fields found")
            if absent:
                print(f"  {'null':<16}: {', '.join(absent)}")
            if doc.items:
                print(f"  {'items':<16}: {len(doc.items)}")
                for it in doc.items:
                    print(f"      - {it['name'][:40]:<42} "
                          f"qty={it.get('qty')} amount={it.get('amount')}")
            for w in doc.warnings:
                print(f"  ! {w}")

        if args.json:
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps([d.to_dict() for d in docs], indent=2, ensure_ascii=False),
                encoding="utf-8")
            print(f"\nwrote {out}")

        print(f"\nprocessed {len(docs)} document(s); database: {DB_PATH}")
    finally:
        pipeline.close()
    return 0


def cmd_dataset(args) -> int:
    """Process receipts straight from the HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("error: `pip install datasets` first", file=sys.stderr)
        return 1

    print(f"loading {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset, split=args.split)
    n = min(args.limit, len(ds))
    print(f"processing {n} of {len(ds)} receipts")
    print(f"runtime: {describe_runtime()}")

    pipeline = Pipeline(
        use_rules=not args.no_rules,
        use_llm=not args.no_llm,
        model=args.model,
        ocr_variant=args.ocr_variant,
        ocr_engine=args.ocr_engine,
        ner_mode="header" if args.header_only else "full",
        store=not args.no_store,
    )
    try:
        for i in range(n):
            row = ds[i]
            doc = pipeline.process(
                row["image"],
                filename=f"{args.split}_{i:05d}.jpg",
                path=f"hf://{args.dataset}/{args.split}/{i}",
            )
            print(f"  -> merchant={doc.fields.get('merchant')!r} "
                  f"total={doc.fields.get('total')!r} date={doc.fields.get('date')!r}")
    finally:
        pipeline.close()
    return 0


def cmd_export(args) -> int:
    """Write the dataset out as ordinary .jpg files plus a labels.json.

    The HuggingFace copy lives inside a cache as parquet blobs, which you
    cannot open, show in a report, or hand to a teammate. This turns it into a
    normal folder of receipts you own.
    """
    import ast

    try:
        from datasets import load_dataset
    except ImportError:
        print("error: `pip install datasets` first", file=sys.stderr)
        return 1

    print(f"loading {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset, split=args.split)
    n = len(ds) if args.limit in (None, 0) else min(args.limit, len(ds))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels: dict[str, dict] = {}

    for i in range(n):
        row = ds[i]
        name = f"{args.split}_{i:05d}.jpg"
        row["image"].convert("RGB").save(out / name, quality=95)
        try:
            truth = ast.literal_eval((row.get("suffix") or "").strip())
        except (ValueError, SyntaxError):
            truth = {}
        labels[name] = truth if isinstance(truth, dict) else {}
        print(f"  {i + 1}/{n}", end="\r")

    (out / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {n} images + labels.json to {out.resolve()}")
    print(f"\nNow usable as a plain folder:\n"
          f"  python run_dms.py process {args.out}\n"
          f"  python evaluate.py --images {args.out} --labels {args.out}/labels.json")
    return 0


def cmd_compare(args) -> int:
    """Run one receipt through all three NER configurations, side by side.

    The three are not different systems - they are the same pipeline with
    layers switched off, which is what makes the ablation in the report a fair
    measurement of each layer's contribution.
    """
    from dms.ner import HybridNER
    from dms.ocr import make_ocr

    image = Path(args.image)
    if not image.exists():
        print(f"error: not found: {image}", file=sys.stderr)
        return 1

    print(f"runtime: {describe_runtime()}")
    print(f"\n[OCR] reading {image.name} once; all three layers see identical text")
    ocr = make_ocr(args.ocr_engine).read(image, variant=args.ocr_variant)
    print(f"      variant={ocr.variant} lines={len(ocr.lines)} "
          f"mean_conf={ocr.mean_conf:.3f}")

    configs = [
        ("rules", dict(use_rules=True, use_llm=False),
         "lexicon + regex only, no model"),
        ("llm", dict(use_rules=False, use_llm=True),
         "Qwen2.5-1.5B only, no regex"),
        ("hybrid", dict(use_rules=True, use_llm=True),
         "both, merged and validated (what ships)"),
    ]

    results, warnings = {}, {}
    for name, flags, blurb in configs:
        print(f"\n[{name}] {blurb}")
        t0 = time.time()
        ner = HybridNER(**flags, model=args.model, verbose=False)
        _, fields, items, warns = ner.extract(ocr.text)
        results[name] = dict(fields, items=len(items))
        warnings[name] = warns
        print(f"      {time.time() - t0:.1f}s")

    keys = [k for k in results["hybrid"] if any(results[c].get(k) is not None
                                                for c in results)]
    _rule("FIELD-BY-FIELD COMPARISON")
    print(f"  {'FIELD':<16}{'RULES':<22}{'LLM':<22}{'HYBRID':<22}")
    print("  " + "-" * 80)
    for key in keys:
        row = [results[c].get(key) for c in ("rules", "llm", "hybrid")]
        cells = "".join(f"{str(v if v is not None else '-')[:20]:<22}" for v in row)
        differs = "  <-- differs" if len({str(v) for v in row}) > 1 else ""
        print(f"  {key:<16}{cells}{differs}")

    _rule("WHAT THE VALIDATORS DID (hybrid only)")
    if warnings["hybrid"]:
        for w in warnings["hybrid"]:
            print(f"  ! {w}")
    else:
        print("  (nothing to correct - both layers agreed and the arithmetic held)")
    return 0


def cmd_delete(args) -> int:
    """Remove documents from the database.

    Deletion cannot be undone, so it asks for confirmation unless ``--yes`` is
    given. Removing a document also removes its entities and their vectors,
    because ``entities`` is declared with ``ON DELETE CASCADE``.
    """
    db = ReceiptDB(verbose=False)
    try:
        stats = db.stats()
        if stats["documents"] == 0:
            print("the database is already empty")
            return 0

        if args.all:
            targets = [d["id"] for d in db.list_documents(limit=100000)]
            what = f"ALL {len(targets)} document(s) and {stats['entities']} entities"
        elif args.doc_id:
            targets = []
            for doc_id in args.doc_id:
                if db.get_document(doc_id) is None:
                    print(f"no document with id {doc_id}", file=sys.stderr)
                else:
                    targets.append(doc_id)
            if not targets:
                return 1
            what = f"document(s) {', '.join(str(t) for t in targets)}"
        else:
            print("specify --all or --doc-id N [N ...]", file=sys.stderr)
            return 1

        print(f"about to permanently delete {what}")
        print(f"database: {DB_PATH}")
        if not args.yes:
            reply = input("type 'delete' to confirm: ").strip().lower()
            if reply != "delete":
                print("cancelled; nothing was removed")
                return 1

        for doc_id in targets:
            db.delete_document(doc_id)
        after = db.stats()
        print(f"removed {len(targets)} document(s). "
              f"{after['documents']} document(s) and {after['entities']} "
              "entities remain.")
    finally:
        db.close()
    return 0


def cmd_reindex(args) -> int:
    """Recompute every semantic vector (after changing the embedding model)."""
    db = ReceiptDB(embed_model=args.embedder, verbose=True)
    try:
        if db.embedder is None:
            print("semantic search unavailable - is sentence-transformers installed?",
                  file=sys.stderr)
            return 1
        print(f"embedding model: {db.embedder.repo} (dim {db.embedder.dim})")
        written = db.reindex_embeddings()
        print(f"re-embedded {written} entities")
        print(f"database: {DB_PATH}")
    finally:
        db.close()
    return 0


def cmd_search(args) -> int:
    db = ReceiptDB(embed_model=getattr(args, "embedder", None), verbose=True)
    try:
        result = db.search_entities(args.query, etype=args.type, limit=args.limit,
                                    semantic=not args.no_semantic)
        _rule(f"search {args.query!r}  ->  mode = {result['mode']}")

        explanation = {
            "exact": "exact entity match",
            "partial": "substring match",
            "similar": ("no exact match; showing the closest entities we hold, "
                        "by spelling and by meaning"),
            "type_fallback": (f"'{args.query}' is not in the database; showing the "
                              "other locations we do hold"),
            "empty": "nothing similar found",
        }[result["mode"]]
        print(explanation)

        if not result["hits"]:
            if result["suggestions"]:
                print(f"\ndid you mean: {', '.join(result['suggestions'][:5])}")
            return 0

        print(f"\n{len(result['hits'])} hit(s):")
        by_doc: dict[int, list[dict]] = {}
        for h in result["hits"]:
            by_doc.setdefault(h["doc_id"], []).append(h)

        for doc_id, hits in by_doc.items():
            print(f"\n  --- doc {doc_id}: {hits[0]['filename']} ---")
            for h in hits:
                extra = ""
                if h.get("similarity"):
                    parts = []
                    if h.get("lexical_score") is not None:
                        parts.append(f"lex={h['lexical_score']:.2f}")
                    if h.get("semantic_score") is not None:
                        parts.append(f"sem={h['semantic_score']:.2f}")
                    extra = f"  [{h.get('matched_by', 'similar')}: {' '.join(parts)}]"
                print(f"    {h['type']:<15} {h['value'][:46]:<48}"
                      f"conf={h['confidence']:.2f}{extra}")
            if args.highlight:
                print()
                snippet = db.highlight_document(doc_id, [h["entity_id"] for h in hits])
                for line in snippet.splitlines():
                    print(f"    | {line}")
    finally:
        db.close()
    return 0


def cmd_show(args) -> int:
    db = ReceiptDB()
    try:
        doc = db.get_document(args.doc_id)
        if not doc:
            print(f"no document with id {args.doc_id}", file=sys.stderr)
            return 1
        _rule(f"doc {doc['id']}: {doc['filename']}   [{doc['language']}]")
        print(f"ocr variant={doc['ocr_variant']}  mean_conf={doc['ocr_conf']:.3f}  "
              f"created={doc['created_at']}")

        _rule("FIELDS")
        for key, value in doc["fields"].items():
            print(f"  {key:<16}: {value}")

        _rule("ENTITIES")
        print(f"  {'TYPE':<16}{'VALUE':<38}{'CONF':>6}  {'SOURCE':<10}{'BBOX'}")
        for e in doc["entities"]:
            bbox = e["meta"].get("bbox")
            print(f"  {e['type']:<16}{str(e['value'])[:36]:<38}{e['confidence']:>6.2f}  "
                  f"{str(e['source']):<10}{bbox if bbox else '-'}")

        _rule("HIGHLIGHTED TEXT")
        print(db.highlight_document(args.doc_id))

        if args.html:
            out = Path(args.html)
            out.write_text(
                "<meta charset='utf-8'><style>body{font-family:monospace;"
                "line-height:1.6}mark{background:#ffe08a;padding:1px 3px;"
                "border-radius:3px}</style>" + db.highlight_html(args.doc_id),
                encoding="utf-8")
            print(f"\nwrote {out}")
    finally:
        db.close()
    return 0


def cmd_stats(args) -> int:
    db = ReceiptDB()
    try:
        stats = db.stats()
        _rule("DATABASE")
        print(f"  path      : {stats['path']}")
        print(f"  documents : {stats['documents']}")
        print(f"  entities  : {stats['entities']}")
        print(f"  fts5      : {'available' if stats['fts'] else 'unavailable (LIKE fallback)'}")
        if stats["by_type"]:
            print("\n  entities by type:")
            for etype, count in stats["by_type"].items():
                print(f"    {etype:<18}{count}")
        docs = db.list_documents(limit=args.limit)
        if docs:
            print("\n  recent documents:")
            for d in docs:
                f = d["fields"]
                print(f"    [{d['id']:>3}] {d['filename'][:26]:<28}"
                      f"{str(f.get('merchant'))[:24]:<26}total={f.get('total')}")
    finally:
        db.close()
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receipt DMS - OCR + NER + searchable storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_pipeline_flags(p):
        p.add_argument("--no-llm", action="store_true", help="rule layer only")
        p.add_argument("--no-rules", action="store_true", help="LLM layer only")
        p.add_argument("--model", default=None, help="LLM alias or HF repo id")
        p.add_argument("--ocr-variant", default="auto",
                       help="auto | all | raw | gray_otsu | adaptive | clahe_sharp | deskew_otsu")
        p.add_argument("--ocr-engine", default=None,
                       choices=["easyocr", "tesseract"],
                       help="OCR backend (see benchmark_ocr.py for a comparison)")
        p.add_argument("--header-only", action="store_true",
                       help="skip line items (about 2x faster)")
        p.add_argument("--no-store", action="store_true", help="do not write to the database")

    p = sub.add_parser("process", help="process an image or a folder")
    p.add_argument("target")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json", default=None, help="also write results to this JSON file")
    add_pipeline_flags(p)
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("dataset", help="process receipts from the HuggingFace dataset")
    p.add_argument("--dataset", default="amohseni/receipt_VLM_information_extraction")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=10)
    add_pipeline_flags(p)
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("export", help="save the dataset as real image files + labels.json")
    p.add_argument("--dataset", default="amohseni/receipt_VLM_information_extraction")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0, help="0 = the whole split")
    p.add_argument("--out", default="data/receipts")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("search", help="search entities (lexical + semantic)")
    p.add_argument("query")
    p.add_argument("--type", default=None, help="restrict to one entity type")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--highlight", action="store_true",
                   help="print the matching documents with the hit highlighted")
    p.add_argument("--no-semantic", action="store_true",
                   help="lexical matching only (disable embeddings)")
    p.add_argument("--embedder", default=None,
                   help="embedding model alias, see dms/embeddings.py")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("compare",
                       help="one receipt through rules / LLM / hybrid, side by side")
    p.add_argument("image")
    p.add_argument("--model", default=None)
    p.add_argument("--ocr-engine", default=None, choices=["easyocr", "tesseract"])
    p.add_argument("--ocr-variant", default="auto")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("delete", help="remove documents from the database")
    p.add_argument("--doc-id", type=int, nargs="+", help="document id(s) to remove")
    p.add_argument("--all", action="store_true", help="remove every document")
    p.add_argument("--yes", action="store_true",
                   help="skip the confirmation prompt (for scripts)")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("reindex", help="rebuild semantic vectors for every entity")
    p.add_argument("--embedder", default=None)
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("show", help="show one document with its entities highlighted")
    p.add_argument("doc_id", type=int)
    p.add_argument("--html", default=None, help="also write an HTML file")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("stats", help="database summary")
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_stats)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
