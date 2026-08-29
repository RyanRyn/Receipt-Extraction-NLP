"""
Compare local LLMs for receipt extraction, on the same OCR text.

    python benchmark_llm.py --limit 30
    python benchmark_llm.py --limit 30 --models qwen1.5b,qwen3b,malaysian3b

Every candidate sees identical OCR output (cached on disk), is scored against
the dataset's ground truth, and is unloaded before the next one is loaded - an
8 GB card holds exactly one 3B model in float16.

Both the LLM-only and the hybrid score are reported. They answer different
questions: LLM-only says how good the model is, hybrid says how much it adds on
top of the rule layer, which is what actually ships.
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
import traceback
from pathlib import Path

from dms.config import LLM_MODELS, describe_runtime, resolve_model
from dms.llm import LocalLLMExtractor
from dms.ner import HybridNER
from dms.ocr import make_ocr

DEFAULT_MODELS = "qwen1.5b,qwen3-1.7b,qwen3b,malaysian3b"


def vram_gb() -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="amohseni/receipt_VLM_information_extraction")
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--ocr-engine", default="easyocr")
    ap.add_argument("--ocr-variant", default="auto")
    ap.add_argument("--report", default="data/llm_benchmark.json")
    args = ap.parse_args(argv)

    from evaluate import (FIELD_MAP, OcrCache, Scoreboard, load_hf_items,
                          load_local_items)

    print(f"runtime: {describe_runtime()}")
    if args.images:
        items, source, _ = load_local_items(args.images, args.labels, args.limit)
    else:
        items, source, _ = load_hf_items(args.dataset, args.split, args.limit)
    if not items:
        print("no receipts found")
        return 1
    n = len(items)
    print(f"source: {source}   receipts: {n}\n")

    # ---- OCR once, shared by every model -------------------------------
    reader = make_ocr(args.ocr_engine)
    cache = OcrCache(source, args.split if not args.images else "local",
                     f"{args.ocr_engine}-{args.ocr_variant}")
    texts, truths = [], []
    for i, (key, image, truth) in enumerate(items):
        truths.append(truth)
        cached = cache.get(key)
        if cached is None:
            cached = reader.read(image, variant=args.ocr_variant).text
            cache.put(key, cached)
            print(f"  ocr {i + 1}/{n}", end="\r")
        texts.append(cached)
    cache.save()
    print(f"  ocr ready ({n} receipts)          ")

    # ---- rule-only reference (model independent) -----------------------
    results: dict[str, dict] = {}
    board = Scoreboard("rules only")
    ner = HybridNER(use_rules=True, use_llm=False, verbose=False)
    for text, truth in zip(texts, truths):
        _, fields, _, _ = ner.extract(text)
        board.update(fields, truth)
    scored = sum(board.labelled.values()) or 1
    results["(rules only)"] = {
        "repo": "-", "llm_exact": None, "llm_fuzzy": None,
        "hybrid_exact": sum(board.exact.values()) / scored,
        "hybrid_fuzzy": sum(board.fuzzy.values()) / scored,
        "seconds": 0.0, "vram_gb": 0.0,
        "per_field": {f: board.exact[f] / max(board.labelled[f], 1) for f in FIELD_MAP},
    }

    # ---- each candidate model ------------------------------------------
    for alias in [m.strip() for m in args.models.split(",") if m.strip()]:
        repo = resolve_model(alias)
        print(f"\n{'=' * 66}\n{alias}  ->  {repo}\n{'=' * 66}")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            boards = {}
            elapsed = 0.0
            for mode_name, use_rules in (("llm", False), ("hybrid", True)):
                b = Scoreboard(mode_name)
                stack = HybridNER(use_rules=use_rules, use_llm=True, model=alias,
                                  mode="header", verbose=False, cache=True)
                t0 = time.time()
                for i, (text, truth) in enumerate(zip(texts, truths)):
                    _, fields, _, _ = stack.extract(text)
                    b.update(fields, truth)
                    print(f"  {mode_name} {i + 1}/{n}", end="\r")
                elapsed = max(elapsed, time.time() - t0)
                print(f"  {mode_name} {n}/{n} done        ")
                boards[mode_name] = b

            peak = vram_gb()
            LocalLLMExtractor.unload()

            sl = sum(boards["llm"].labelled.values()) or 1
            sh = sum(boards["hybrid"].labelled.values()) or 1
            results[alias] = {
                "repo": repo,
                "llm_exact": sum(boards["llm"].exact.values()) / sl,
                "llm_fuzzy": sum(boards["llm"].fuzzy.values()) / sl,
                "hybrid_exact": sum(boards["hybrid"].exact.values()) / sh,
                "hybrid_fuzzy": sum(boards["hybrid"].fuzzy.values()) / sh,
                "seconds": elapsed / n,
                "vram_gb": peak,
                "per_field": {f: boards["hybrid"].exact[f]
                              / max(boards["hybrid"].labelled[f], 1) for f in FIELD_MAP},
            }
            r = results[alias]
            print(f"  LLM-only {r['llm_exact']:.1%} | hybrid {r['hybrid_exact']:.1%} "
                  f"| {r['seconds']:.1f}s/receipt | peak {peak:.1f} GB")
        except Exception as exc:
            print(f"  !! {alias} failed: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
            results[alias] = {"repo": repo, "error": f"{type(exc).__name__}: {exc}"}
            LocalLLMExtractor.unload()

    # ---- report ---------------------------------------------------------
    print(f"\n{'=' * 78}\nEXTRACTION MODEL COMPARISON  ({n} receipts)\n{'=' * 78}")
    print(f"  {'MODEL':<16}{'LLM ONLY':>10}{'HYBRID':>10}{'H-FUZZY':>10}"
          f"{'SEC':>8}{'VRAM':>8}")
    ok = {k: v for k, v in results.items() if "error" not in v}
    for name, r in ok.items():
        lo = f"{r['llm_exact']:>10.1%}" if r["llm_exact"] is not None else f"{'-':>10}"
        print(f"  {name:<16}{lo}{r['hybrid_exact']:>10.1%}{r['hybrid_fuzzy']:>10.1%}"
              f"{r['seconds']:>8.1f}{r['vram_gb']:>8.1f}")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<16}FAILED: {r['error'][:52]}")

    if ok:
        best = max(ok.items(), key=lambda kv: kv[1]["hybrid_exact"])
        print(f"\n  best hybrid exact: {best[0]} ({best[1]['hybrid_exact']:.1%})")
        print(f"\n  {'MODEL':<16}" + "".join(f"{f.upper():>10}" for f in FIELD_MAP))
        for name, r in ok.items():
            print(f"  {name:<16}" + "".join(f"{r['per_field'][f]:>10.1%}"
                                            for f in FIELD_MAP))

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": source, "n": n, "results": results},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
