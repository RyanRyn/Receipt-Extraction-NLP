"""
Select the merge routing on the TRAINING split, freeze it, then report TEST once.

    python tune_on_train.py --train-limit 120 --test-limit 97

Why this exists
---------------
The results previously reported were reached by inspecting performance on the
*test* split: the routing of MERCHANT and ADDRESS to one layer or the other was
chosen because it scored better there. That is test-set leakage, and it makes
the reported figure optimistically biased - it is no longer an estimate of
performance on unseen receipts.

The dataset provides 870 training receipts that no decision has yet touched.
The correct protocol, which this script performs end to end, is:

  1. search the routing options on TRAIN only;
  2. freeze the winner;
  3. evaluate TEST exactly once and report that number.

The gap between the train score and the test score is itself informative: a
large drop indicates the configuration was fitted to noise.

Note on what is and is not being tuned. Only the *routing* is searched here -
which layer owns a field when the two disagree - because that is the choice
that was genuinely fitted to observed scores. The validators (a receipt total
is never 0.00; Malaysian GST is 6-10%; never adopt an amount that is printed
nowhere) are statements about receipts rather than parameters fitted to a
sample, so they are not searched. They are, however, included in both the train
and the test runs, so any benefit they provide is measured on unseen data.
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
from itertools import product
from pathlib import Path

from dms.config import describe_runtime
from dms.ner import HybridNER
from dms.ocr import make_ocr

# Fields whose ownership was chosen by looking at scores, and is therefore
# what has to be re-decided on training data.
TUNABLE = ["MERCHANT", "ADDRESS"]
# Always owned by the LLM: the rule layer has no line-item reader worth the name.
ALWAYS_LLM = {"ITEM"}


def ocr_texts(items, engine, variant, cache, reader, label):
    """OCR every item once, reusing the on-disk cache."""
    texts = []
    for i, (key, image, _) in enumerate(items):
        cached = cache.get(key)
        if cached is None:
            t0 = time.time()
            cached = reader.read(image, variant=variant).text
            cache.put(key, cached)
            print(f"  {label} ocr {i + 1}/{len(items)} ({time.time() - t0:.1f}s)",
                  end="\r")
        texts.append(cached)
    cache.save()
    print(f"  {label} ocr ready ({len(items)} receipts)            ")
    return texts


def score_config(texts, truths, prefer_llm, model, Scoreboard, FIELD_MAP):
    """Macro exact/fuzzy for one routing configuration."""
    board = Scoreboard("cfg")
    ner = HybridNER(use_rules=True, use_llm=True, model=model, mode="header",
                    verbose=False, cache=True, prefer_llm=prefer_llm)
    for text, truth in zip(texts, truths):
        _, fields, _, _ = ner.extract(text)
        board.update(fields, truth)
    scored = sum(board.labelled.values()) or 1
    return {
        "exact": sum(board.exact.values()) / scored,
        "fuzzy": sum(board.fuzzy.values()) / scored,
        "per_field": {f: board.exact[f] / max(board.labelled[f], 1)
                      for f in FIELD_MAP},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="amohseni/receipt_VLM_information_extraction")
    ap.add_argument("--train-limit", type=int, default=120)
    ap.add_argument("--test-limit", type=int, default=97)
    ap.add_argument("--model", default=None)
    ap.add_argument("--ocr-engine", default="easyocr")
    ap.add_argument("--ocr-variant", default="auto")
    ap.add_argument("--report", default="data/tuning_report.json")
    args = ap.parse_args(argv)

    from evaluate import FIELD_MAP, OcrCache, Scoreboard, load_hf_items

    print(f"runtime: {describe_runtime()}")
    train_items, src, _ = load_hf_items(args.dataset, "train", args.train_limit)
    test_items, _, _ = load_hf_items(args.dataset, "test", args.test_limit)
    print(f"train: {len(train_items)} receipts   test: {len(test_items)} receipts\n")

    reader = make_ocr(args.ocr_engine)
    key = f"{args.ocr_engine}-{args.ocr_variant}"
    train_cache = OcrCache(src, "train", key)
    test_cache = OcrCache(src, "test", key)

    train_texts = ocr_texts(train_items, args.ocr_engine, args.ocr_variant,
                            train_cache, reader, "train")
    train_truths = [t for _, _, t in train_items]
    test_texts = ocr_texts(test_items, args.ocr_engine, args.ocr_variant,
                           test_cache, reader, "test ")
    test_truths = [t for _, _, t in test_items]

    # ---- STEP 1: search the routing on TRAIN only ----------------------
    print(f"\n{'=' * 74}\nSTEP 1 - searching routing on the TRAINING split\n{'=' * 74}")
    options = list(product([False, True], repeat=len(TUNABLE)))
    train_results = {}
    for combo in options:
        prefer = set(ALWAYS_LLM) | {f for f, use_llm in zip(TUNABLE, combo) if use_llm}
        name = "+".join(sorted(prefer - ALWAYS_LLM)) or "(rules own both)"
        t0 = time.time()
        train_results[name] = score_config(train_texts, train_truths, prefer,
                                           args.model, Scoreboard, FIELD_MAP)
        train_results[name]["prefer_llm"] = sorted(prefer)
        r = train_results[name]
        print(f"  LLM owns {name:<22} exact {r['exact']:>6.1%}  fuzzy {r['fuzzy']:>6.1%}"
              f"   ({time.time() - t0:.0f}s)")

    best_name = max(train_results, key=lambda k: train_results[k]["exact"])
    best_prefer = set(train_results[best_name]["prefer_llm"])
    print(f"\n  WINNER ON TRAIN: LLM owns {best_name} "
          f"({train_results[best_name]['exact']:.1%} exact)")

    # ---- STEP 2: freeze, then evaluate TEST exactly once ----------------
    print(f"\n{'=' * 74}\nSTEP 2 - configuration frozen; evaluating TEST once\n{'=' * 74}")
    test_result = score_config(test_texts, test_truths, best_prefer, args.model,
                               Scoreboard, FIELD_MAP)

    train_best = train_results[best_name]
    print(f"\n  {'SPLIT':<10}{'EXACT':>10}{'FUZZY':>10}   {'n':>5}")
    print(f"  {'train':<10}{train_best['exact']:>10.1%}{train_best['fuzzy']:>10.1%}"
          f"   {len(train_texts):>5}")
    print(f"  {'TEST':<10}{test_result['exact']:>10.1%}{test_result['fuzzy']:>10.1%}"
          f"   {len(test_texts):>5}")
    drop = train_best["exact"] - test_result["exact"]
    print(f"\n  generalisation gap (train - test): {drop:+.1%}")
    if abs(drop) <= 0.05:
        print("  -> small gap: the configuration transfers to unseen receipts")
    else:
        print("  -> large gap: the configuration was partly fitted to the train sample")

    print(f"\n  {'FIELD':<12}{'TRAIN':>10}{'TEST':>10}")
    for f in FIELD_MAP:
        print(f"  {f:<12}{train_best['per_field'][f]:>10.1%}"
              f"{test_result['per_field'][f]:>10.1%}")

    print(f"\n{'=' * 74}")
    print("This TEST figure is an unbiased estimate: the routing was selected")
    print("on training data and the test split was scored once, afterwards.")
    print("=" * 74)

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": args.dataset, "ocr_engine": args.ocr_engine,
        "n_train": len(train_texts), "n_test": len(test_texts),
        "train_search": train_results, "selected": sorted(best_prefer),
        "test_result": test_result,
        "generalisation_gap": drop,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
