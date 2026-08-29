"""
Compare embedding models for semantic entity search, and against the lexical
baseline already in the system.

    python benchmark_embeddings.py --limit 200
    python benchmark_embeddings.py --models e5-small,bge-m3 --device cpu

The task
--------
Built from the dataset's own ground-truth addresses, so nothing is invented:

* **corpus**  - every distinct annotated address
* **query**   - a Malaysian city name that appears in at least one of them
* **relevant**- the addresses that actually contain that city

Metrics are standard information retrieval: Recall@1, Recall@5 and MRR. The
lexical scorer currently used by ``search_entities`` is measured on exactly the
same task, which is the only way to tell whether embeddings are worth their
weight rather than assuming they are.

A second, harder check follows: query a city that is **absent** from the corpus
and inspect what comes back. That is the assignment's actual requirement -
"return the documents with Kuala Lumpur if Johor Bahru is not found" - and it
cannot be scored by recall, because by construction there is no correct answer.
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
from pathlib import Path

import numpy as np

from dms.embeddings import EMBEDDING_MODELS, Embedder, cosine_ranking
from dms.lexicon import MALAYSIAN_PLACES
from dms.textutils import normalize_key, partial_similarity

DEFAULT_MODELS = "e5-small,minilm,e5-base,mpnet"


def build_task(addresses: list[str], min_hits: int = 1):
    """``(corpus, [(query, {relevant indices})])`` from annotated addresses."""
    corpus = sorted({a.strip() for a in addresses if a and a.strip()})
    normalised = [normalize_key(a) for a in corpus]

    queries = []
    for place in MALAYSIAN_PLACES:
        key = normalize_key(place)
        if len(key) < 4:
            continue
        relevant = {i for i, text in enumerate(normalised) if key in text}
        if len(relevant) >= min_hits:
            queries.append((place, relevant))

    # Drop a query whose city name is contained in another matched query
    # ("johor" inside "johor bahru"), which would otherwise double-count.
    queries.sort(key=lambda q: len(q[0]), reverse=True)
    kept, seen = [], set()
    for place, relevant in queries:
        signature = frozenset(relevant)
        if signature in seen:
            continue
        seen.add(signature)
        kept.append((place, relevant))
    return corpus, kept


def score(ranking: list[int], relevant: set[int]) -> tuple[float, float, float]:
    """``(recall@1, recall@5, reciprocal rank)`` for one query."""
    r1 = 1.0 if ranking and ranking[0] in relevant else 0.0
    r5 = 1.0 if any(i in relevant for i in ranking[:5]) else 0.0
    rr = 0.0
    for position, index in enumerate(ranking, start=1):
        if index in relevant:
            rr = 1.0 / position
            break
    return r1, r5, rr


def evaluate_lexical(corpus, queries):
    t0 = time.time()
    r1 = r5 = mrr = 0.0
    for text, relevant in queries:
        scores = np.array([partial_similarity(text, doc) for doc in corpus])
        ranking = list(np.argsort(-scores))
        a, b, c = score(ranking, relevant)
        r1, r5, mrr = r1 + a, r5 + b, mrr + c
    n = max(len(queries), 1)
    return {"recall@1": r1 / n, "recall@5": r5 / n, "mrr": mrr / n,
            "seconds": time.time() - t0, "dim": None}


# Queries with no sensible answer in a receipt corpus. A usable similarity
# threshold must separate these from real queries; if it cannot, absolute
# scores are not a safe basis for rejection and the system must say so rather
# than pretend otherwise.
NONSENSE_QUERIES = [
    "zzzznotathing", "qwertyuiop asdfgh", "photosynthesis in plants",
    "xkcd flarn blorptastic", "quantum chromodynamics lecture notes",
]


def measure_separation(alias, corpus, queries, device):
    """Top-1 score distribution for real vs nonsense queries."""
    embedder = Embedder(alias, device=device, verbose=False)
    matrix = embedder.encode(corpus)

    def top_scores(texts):
        out = []
        for text in texts:
            scores = cosine_ranking(embedder.encode_one(text, is_query=True), matrix)
            out.append(float(scores.max()) if scores.size else 0.0)
        return out

    real = top_scores([q for q, _ in queries])
    junk = top_scores(NONSENSE_QUERIES)
    Embedder.unload()

    real_min, junk_max = (min(real) if real else 0.0), (max(junk) if junk else 0.0)
    return {
        "real_min": real_min, "real_median": float(np.median(real)) if real else 0.0,
        "junk_max": junk_max, "junk_median": float(np.median(junk)) if junk else 0.0,
        "gap": real_min - junk_max,
        "suggested_threshold": round((real_min + junk_max) / 2, 3)
        if real_min > junk_max else None,
    }


def evaluate_embedder(alias, corpus, queries, device):
    embedder = Embedder(alias, device=device)
    t0 = time.time()
    matrix = embedder.encode(corpus)
    encode_seconds = time.time() - t0

    t0 = time.time()
    r1 = r5 = mrr = 0.0
    for text, relevant in queries:
        vector = embedder.encode_one(text, is_query=True)
        ranking = list(np.argsort(-cosine_ranking(vector, matrix)))
        a, b, c = score(ranking, relevant)
        r1, r5, mrr = r1 + a, r5 + b, mrr + c
    query_seconds = time.time() - t0
    n = max(len(queries), 1)
    result = {"recall@1": r1 / n, "recall@5": r5 / n, "mrr": mrr / n,
              "dim": embedder.dim, "index_seconds": encode_seconds,
              "seconds": query_seconds, "repo": embedder.repo}
    Embedder.unload()
    return result


def absent_city_demo(alias, corpus, device, absent="Johor Bahru", top=3):
    """What comes back for a city the corpus does not contain?"""
    keep = [a for a in corpus if normalize_key(absent) not in normalize_key(a)]
    if not keep:
        return None
    embedder = Embedder(alias, device=device, verbose=False)
    matrix = embedder.encode(keep)
    vector = embedder.encode_one(absent, is_query=True)
    scores = cosine_ranking(vector, matrix)
    order = list(np.argsort(-scores))[:top]
    out = [(float(scores[i]), keep[i]) for i in order]
    Embedder.unload()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="amohseni/receipt_VLM_information_extraction")
    ap.add_argument("--split", default="train")
    ap.add_argument("--labels", default=None,
                    help="use a local labels.json instead of the dataset")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--report", default="data/embedding_benchmark.json")
    args = ap.parse_args(argv)

    # ---- gather annotated addresses -------------------------------------
    addresses: list[str] = []
    if args.labels:
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        addresses = [v.get("address") for v in labels.values() if isinstance(v, dict)]
        source = args.labels
    else:
        import ast
        from datasets import load_dataset
        ds = load_dataset(args.dataset, split=args.split)
        n = min(args.limit, len(ds))
        for i in range(n):
            try:
                row = ast.literal_eval((ds[i].get("suffix") or "").strip())
                if isinstance(row, dict):
                    addresses.append(row.get("address"))
            except (ValueError, SyntaxError):
                continue
        source = f"{args.dataset} [{args.split}] x{n}"

    corpus, queries = build_task([a for a in addresses if a])
    print(f"source : {source}")
    print(f"corpus : {len(corpus)} distinct addresses")
    print(f"queries: {len(queries)} city names present in the corpus")
    if not corpus or not queries:
        print("not enough data to build the task")
        return 1
    print(f"example queries: {[q for q, _ in queries[:6]]}\n")

    results = {"lexical (current)": evaluate_lexical(corpus, queries)}
    print(f"  lexical baseline done "
          f"(recall@1 {results['lexical (current)']['recall@1']:.1%})")

    for alias in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\nrunning {alias} ({EMBEDDING_MODELS.get(alias, {}).get('note', '?')})")
        try:
            results[alias] = evaluate_embedder(alias, corpus, queries, args.device)
            results[alias]["separation"] = measure_separation(
                alias, corpus, queries, args.device)
            r = results[alias]
            print(f"  recall@1 {r['recall@1']:.1%}  recall@5 {r['recall@5']:.1%}  "
                  f"MRR {r['mrr']:.3f}  dim {r['dim']}")
        except Exception as exc:
            print(f"  !! failed: {type(exc).__name__}: {exc}")
            results[alias] = {"error": f"{type(exc).__name__}: {exc}"}
            Embedder.unload()

    print(f"\n{'=' * 74}\nSEMANTIC RETRIEVAL OF ADDRESSES BY CITY NAME\n{'=' * 74}")
    print(f"  {'MODEL':<20}{'RECALL@1':>10}{'RECALL@5':>10}{'MRR':>8}{'DIM':>6}{'QUERY s':>9}")
    ok = {k: v for k, v in results.items() if "error" not in v}
    for name, r in ok.items():
        dim = r["dim"] if r["dim"] else "-"
        print(f"  {name:<20}{r['recall@1']:>10.1%}{r['recall@5']:>10.1%}"
              f"{r['mrr']:>8.3f}{str(dim):>6}{r['seconds']:>9.2f}")
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<20}FAILED: {r['error'][:46]}")

    sep = {k: v["separation"] for k, v in ok.items() if v.get("separation")}
    if sep:
        print(f"\n{'=' * 74}")
        print("CAN AN ABSOLUTE THRESHOLD REJECT A NONSENSE QUERY?")
        print("=" * 74)
        print(f"  {'MODEL':<20}{'REAL min':>10}{'JUNK max':>10}{'GAP':>9}{'THRESHOLD':>12}")
        for name, s in sep.items():
            thr = s["suggested_threshold"]
            print(f"  {name:<20}{s['real_min']:>10.3f}{s['junk_max']:>10.3f}"
                  f"{s['gap']:>+9.3f}{(f'{thr:.3f}' if thr else 'none') :>12}")
        print("\n  A positive gap means a threshold exists that admits every real")
        print("  query and rejects every nonsense one. A negative gap means it does")
        print("  not, and the system should rank rather than reject.")

    if ok:
        best = max(ok.items(), key=lambda kv: kv[1]["mrr"])
        print(f"\n  best by MRR: {best[0]} ({best[1]['mrr']:.3f})")

        demo_alias = best[0] if best[0] in EMBEDDING_MODELS else None
        if demo_alias:
            print(f"\n{'=' * 74}")
            print("ABSENT-CITY BEHAVIOUR - the assignment's requirement")
            print("query 'Johor Bahru' against a corpus with every JB address removed")
            print("=" * 74)
            for sim, address in absent_city_demo(demo_alias, corpus, args.device) or []:
                print(f"  {sim:.3f}  {address[:66]}")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": source, "corpus": len(corpus),
                               "queries": len(queries), "results": results},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
