"""Sentence embeddings for semantic entity search.

Why this exists
---------------
The lexical search in :mod:`dms.database` compares *spellings*. It links
"Kuala Lumpur" to "Kuala Lumpor", because the letters nearly match. It cannot
link "Johor Bahru" to "Kuala Lumpur", because as strings they share almost
nothing - yet a user searching for one Malaysian city plainly wants the others
when theirs is absent.

An embedding model maps text to a vector whose geometry reflects *meaning*, so
Malaysian city names land near each other regardless of spelling. That is what
turns the assignment's similar-entity requirement from a hard-coded gazetteer
lookup into genuine semantic retrieval.

Storage
-------
Vectors are kept as float32 blobs in SQLite and compared by brute-force cosine
similarity. For a corpus of this size that is the right call: 10,000 entities at
384 dimensions is a 15 MB matrix and a single matrix multiply - microseconds.
An approximate index (FAISS, sqlite-vec) only pays off in the millions, and
would add a dependency for no measurable gain here.
"""

from __future__ import annotations

import os

import numpy as np

# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
# All multilingual and all free. The E5 family is trained with asymmetric
# prefixes - queries and stored passages must be marked differently - and
# omitting them measurably degrades retrieval, so the prefixes are part of the
# registry rather than left to the caller.
# ``min_score`` is per model on purpose. These models place similarity on
# completely different scales - E5 packs everything into 0.75-0.90 while BGE-M3
# spreads across 0.3-0.6 - so one global cut-off would be meaningless. Each
# value here is the measured top-1 score of a deliberately nonsensical query
# against a 300-address corpus (benchmark_embeddings.py).
EMBEDDING_MODELS: dict[str, dict] = {
    "e5-small": {
        "repo": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "dim": 384,
        "min_score": 0.82,
        "note": "118M params, ~470 MB - recall@1 68.2%, MRR 0.777",
    },
    "e5-base": {
        "repo": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "dim": 768,
        "min_score": 0.80,
        "note": "278M params, ~1.1 GB - recall@1 81.8%, MRR 0.848",
    },
    "minilm": {
        "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "dim": 384,
        "min_score": 0.45,
        "note": "118M params - recall@1 9.1%, MRR 0.174 - poor on Malay places",
    },
    "mpnet": {
        "repo": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "query_prefix": "",
        "passage_prefix": "",
        "dim": 768,
        "min_score": 0.57,
        "note": "278M params - recall@1 27.3%, MRR 0.305 - poor on Malay places",
    },
    "bge-m3": {
        "repo": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        "dim": 1024,
        # Re-measured on a 113-entity corpus: eight deliberately nonsensical
        # queries topped out at 0.526, while ten genuine ones ranged 0.510-0.713
        # (median 0.597). 0.53 therefore rejects every nonsense query and keeps
        # nine of the ten real ones. The single real query below the line
        # ("Kuala Lumpur", 0.510) never reaches this stage in practice, because
        # substring matching answers it first.
        #
        # The margin is thin and corpus-dependent - the measured gap is -0.016,
        # i.e. the distributions genuinely overlap. This value removes obvious
        # noise; it is not a guarantee, and §5.6 of the report says so.
        "min_score": 0.53,
        "note": "568M params, ~2.2 GB - recall@1 86.4%, MRR 0.903 (best measured)",
    },
}

# Measured on 300 dataset addresses (benchmark_embeddings.py), retrieving
# addresses by the city they contain:
#
#   model      recall@1  MRR      note
#   bge-m3        86.4%  0.903    best measured  <- default
#   e5-base       81.8%  0.848    good balance, 1.1 GB
#   e5-small      68.2%  0.777    lightest usable, 470 MB
#   mpnet         27.3%  0.305    poor on Malay place names
#   minilm         9.1%  0.174    unusable here
#
# The last two are the models people reach for by default; on this data they
# fail badly, which is why the choice was measured rather than assumed.
DEFAULT_EMBEDDER = os.getenv("DMS_EMBEDDER", "bge-m3")

# Embedding runs on CPU by default. It is fast enough there (~0.1 s per query)
# and it keeps the GPU free for the extraction LLM - a 3B model already uses
# 6.8 GB of the 8 GB card, so loading an embedder beside it would risk an
# out-of-memory failure during ingestion.
DEFAULT_EMBED_DEVICE = os.getenv("DMS_EMBED_DEVICE", "cpu")


def resolve_embedder(key: str | None = None) -> tuple[str, dict]:
    """Map an alias to ``(repo_id, config)``; unknown values pass through."""
    key = key or DEFAULT_EMBEDDER
    if key in EMBEDDING_MODELS:
        cfg = EMBEDDING_MODELS[key]
        return cfg["repo"], cfg
    return key, {"repo": key, "query_prefix": "", "passage_prefix": "", "dim": None}


class Embedder:
    """Loads a sentence-transformer once and encodes text to unit vectors.

    Vectors are L2-normalised, so a dot product *is* the cosine similarity and
    ranking needs no further division.
    """

    _loaded: dict[str, object] = {}

    def __init__(self, model: str | None = None, device: str | None = None,
                 verbose: bool = True):
        self.key = model or DEFAULT_EMBEDDER
        self.repo, self.cfg = resolve_embedder(self.key)
        self.verbose = verbose
        if device is None:
            device = DEFAULT_EMBED_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            cache_key = f"{self.repo}@{self.device}"
            if cache_key not in Embedder._loaded:
                from sentence_transformers import SentenceTransformer
                if self.verbose:
                    print(f"  [emb] loading {self.repo} on {self.device} ...")
                Embedder._loaded[cache_key] = SentenceTransformer(
                    self.repo, device=self.device)
            self._model = Embedder._loaded[cache_key]
        return self._model

    @property
    def dim(self) -> int:
        get = getattr(self.model, "get_embedding_dimension", None)
        if get is None:                       # older sentence-transformers
            get = self.model.get_sentence_embedding_dimension
        return int(get())

    @property
    def min_score(self) -> float:
        """Similarity floor for this specific model (see the registry note)."""
        from dms.config import SEMANTIC_MIN_SCORE
        return float(self.cfg.get("min_score") or SEMANTIC_MIN_SCORE)

    def encode(self, texts: list[str], is_query: bool = False,
               batch_size: int = 64) -> np.ndarray:
        """Encode texts to an ``(n, dim)`` float32 array of unit vectors."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.cfg.get("query_prefix" if is_query else "passage_prefix", "")
        prepared = [f"{prefix}{t}" for t in texts]
        vectors = self.model.encode(
            prepared, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    @classmethod
    def unload(cls) -> None:
        """Release loaded embedders (used when benchmarking several)."""
        import gc
        cls._loaded.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# --------------------------------------------------------------------------
# vector <-> blob helpers for SQLite
# --------------------------------------------------------------------------

def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes, dim: int | None = None) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    if dim is not None and vector.size != dim:
        raise ValueError(f"expected {dim} dimensions, stored vector has {vector.size}")
    return vector


def cosine_ranking(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Similarity of ``query`` against every row of ``matrix``.

    Both sides are already unit vectors, so this is a plain dot product.
    """
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ np.asarray(query, dtype=np.float32)
