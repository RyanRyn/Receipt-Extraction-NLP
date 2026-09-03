"""SQLite storage + the search/highlight API the search half of the DMS builds on.

This module is the **integration contract** between the two halves of the
project. The extraction half (OCR + NER) writes documents and entities here;
the search half reads them. Everything the search side needs is provided:

* ``search_entities`` - exact -> partial -> fuzzy -> type-fallback cascade,
  which is what satisfies "if 'Johor Bahru' is not found, still return the
  documents containing 'Kuala Lumpur'";
* character offsets on every entity, so a hit can be highlighted *in place*;
* ``highlight_document`` / ``highlight_html`` to render those offsets.

See ``docs/ASSIGNMENT_REPORT.md (S3.7)`` for the full contract.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable

from dms.config import DB_PATH, SEMANTIC_MIN_SCORE, SIMILARITY_THRESHOLD
from dms.lexicon import MALAYSIAN_PLACES
from dms.schema import LOCATION_TYPES, SEMANTIC_TYPES, Entity, ReceiptDocument
from dms.textutils import normalize_key, partial_similarity

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT NOT NULL,
    path         TEXT,
    ocr_text     TEXT NOT NULL DEFAULT '',
    ocr_conf     REAL DEFAULT 0,
    ocr_variant  TEXT,
    language     TEXT,
    fields_json  TEXT DEFAULT '{}',
    items_json   TEXT DEFAULT '[]',
    warnings_json TEXT DEFAULT '[]',
    tokens_json  TEXT DEFAULT '[]',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    type       TEXT NOT NULL,
    -- NULL means the type was looked for and is not present on this receipt.
    -- It is deliberately distinct from '', which would be an empty value that
    -- *was* extracted. Retrieval treats NULL as inert: it never equals or
    -- matches anything, and the paths that select by type instead of by value
    -- exclude it explicitly.
    value      TEXT,
    value_norm TEXT,
    text       TEXT,
    start      INTEGER DEFAULT -1,
    end        INTEGER DEFAULT -1,
    confidence REAL DEFAULT 0,
    source     TEXT,
    meta_json  TEXT DEFAULT '{}',
    embedding  BLOB
);

-- Which embedding model produced the vectors above. Mixing vectors from two
-- models in one index silently returns nonsense, so the model is recorded and
-- checked rather than assumed.
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_entities_doc   ON entities(doc_id);
CREATE INDEX IF NOT EXISTS idx_entities_type  ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_norm  ON entities(value_norm);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts
    USING fts5(value, text, content='entities', content_rowid='id');
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
    USING fts5(ocr_text, content='documents', content_rowid='id');
"""


def _fts_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


class ReceiptDB:
    """Thin, dependency-free persistence layer over SQLite."""

    def __init__(self, path: str | Path | None = None,
                 embed_model: str | None = None, verbose: bool = False):
        self.path = Path(path or DB_PATH)
        self.embed_model = embed_model
        self.verbose = verbose
        self._embedder = None          # None = not tried, False = unavailable
        self._vector_cache = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because a web front end (Streamlit, Flask)
        # serves each interaction from a different worker thread, and SQLite
        # would otherwise refuse a connection created on another one. Writes
        # are guarded by `self._lock` below, so only one thread mutates the
        # database at a time.
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.has_fts = _fts_available(self.conn)
        if self.has_fts:
            self.conn.executescript(FTS_SCHEMA)
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created."""
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(documents)")}
        for column, ddl in [("tokens_json", "TEXT DEFAULT '[]'")]:
            if column not in have:
                self.conn.execute(f"ALTER TABLE documents ADD COLUMN {column} {ddl}")
        info = list(self.conn.execute("PRAGMA table_info(entities)"))
        have = {r["name"] for r in info}
        if "embedding" not in have:
            self.conn.execute("ALTER TABLE entities ADD COLUMN embedding BLOB")

        # Absent entities are stored with a NULL value, which an older schema
        # forbids. SQLite has no "DROP NOT NULL", so the table is rebuilt in
        # place - the standard twelve-step dance, minus the steps that only
        # matter when the schema shape changes.
        if any(r["name"] == "value" and r["notnull"] for r in info):
            self.conn.executescript("""
                PRAGMA foreign_keys=off;
                BEGIN;
                CREATE TABLE entities_new (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    type       TEXT NOT NULL,
                    value      TEXT,
                    value_norm TEXT,
                    text       TEXT,
                    start      INTEGER DEFAULT -1,
                    end        INTEGER DEFAULT -1,
                    confidence REAL DEFAULT 0,
                    source     TEXT,
                    meta_json  TEXT DEFAULT '{}',
                    embedding  BLOB
                );
                INSERT INTO entities_new
                    SELECT id, doc_id, type, value, value_norm, text, start, end,
                           confidence, source, meta_json, embedding FROM entities;
                DROP TABLE entities;
                ALTER TABLE entities_new RENAME TO entities;
                COMMIT;
                PRAGMA foreign_keys=on;
            """)
            self.conn.executescript(
                "CREATE INDEX IF NOT EXISTS idx_entities_doc ON entities(doc_id);"
                "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);"
                "CREATE INDEX IF NOT EXISTS idx_entities_norm ON entities(value_norm);")
        self.conn.commit()

    # ------------------------------------------------------------------
    # settings
    # ------------------------------------------------------------------
    def get_setting(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?",
                                (key,)).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO settings(key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        self.conn.commit()

    # ------------------------------------------------------------------
    # embeddings
    # ------------------------------------------------------------------
    @property
    def embedder(self):
        """The embedding model, loaded on first use; ``None`` if unavailable.

        Semantic search is an enhancement, not a prerequisite: if
        sentence-transformers is missing or the weights cannot be fetched, the
        lexical cascade still answers every query.
        """
        if self._embedder is False:
            return None
        if self._embedder is None:
            try:
                from dms.embeddings import Embedder
                self._embedder = Embedder(self.embed_model, verbose=self.verbose)
                stored = self.get_setting("embed_model")
                if stored and stored != self._embedder.repo:
                    print(f"  [db] warning: stored vectors came from {stored!r} but "
                          f"{self._embedder.repo!r} is loaded. Run "
                          f"`python run_dms.py reindex` to rebuild them.")
            except Exception as exc:
                if self.verbose:
                    print(f"  [db] semantic search unavailable: {exc}")
                self._embedder = False
                return None
        return self._embedder

    def embed_entities(self, entity_ids: list[int] | None = None,
                       batch_size: int = 256) -> int:
        """Compute and store vectors for entities that lack one."""
        embedder = self.embedder
        if embedder is None:
            return 0
        with self._lock:
            return self._embed_entities(embedder, entity_ids, batch_size)

    def _embed_entities(self, embedder, entity_ids, batch_size) -> int:
        types = ",".join("?" * len(SEMANTIC_TYPES))
        sql = (f"SELECT id, value FROM entities WHERE embedding IS NULL AND value IS NOT NULL "
               f"AND type IN ({types})")
        params: tuple = tuple(sorted(SEMANTIC_TYPES))
        if entity_ids:
            placeholders = ",".join("?" * len(entity_ids))
            sql += f" AND id IN ({placeholders})"
            params = params + tuple(entity_ids)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return 0

        from dms.embeddings import to_blob

        written = 0
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            vectors = embedder.encode([r["value"] for r in chunk])
            self.conn.executemany(
                "UPDATE entities SET embedding = ? WHERE id = ?",
                [(to_blob(v), r["id"]) for v, r in zip(vectors, chunk)])
            written += len(chunk)
        self.conn.commit()
        self.set_setting("embed_model", embedder.repo)
        self.set_setting("embed_dim", str(embedder.dim))
        self._vector_cache = None
        return written

    def reindex_embeddings(self) -> int:
        """Recompute every vector - use after changing the embedding model."""
        with self._lock:
            self.conn.execute("UPDATE entities SET embedding = NULL")
            self.conn.commit()
            self._vector_cache = None
        return self.embed_entities()

    def _vectors(self, etype: str | None = None):
        """``(rows, matrix)`` of every embedded entity, cached in memory.

        Rows whose vector has the wrong number of dimensions are skipped rather
        than crashing the search. This happens whenever the embedding model is
        changed after some entities were already indexed: the old vectors are
        simply a different length, and mixing them would be meaningless even if
        the shapes happened to align. The caller is told to reindex.
        """
        import numpy as np

        if self._vector_cache is None:
            from dms.embeddings import from_blob
            rows = self.conn.execute(
                "SELECT e.id, e.doc_id, e.type, e.value, e.text, e.start, e.end, "
                "e.confidence, e.source, e.embedding, d.filename "
                "FROM entities e JOIN documents d ON d.id = e.doc_id "
                "WHERE e.embedding IS NOT NULL").fetchall()

            want = self.embedder.dim if self.embedder is not None else None
            keep, vectors, stale = [], [], 0
            for row in rows:
                vector = from_blob(row["embedding"])
                if want is not None and vector.size != want:
                    stale += 1
                    continue
                keep.append(row)
                vectors.append(vector)
            if stale:
                print(f"  [db] {stale} entity vector(s) were built by a different "
                      f"embedding model and are being ignored. Run "
                      f"`python run_dms.py reindex` to rebuild them.")
            matrix = (np.vstack(vectors) if vectors
                      else np.zeros((0, want or 1), dtype="float32"))
            self._vector_cache = (keep, matrix)

        rows, matrix = self._vector_cache
        if etype:
            keep = [i for i, r in enumerate(rows) if r["type"] == etype]
            if not keep:
                return [], np.zeros((0, 1), dtype="float32")
            return [rows[i] for i in keep], matrix[keep]
        return rows, matrix

    def semantic_search(self, query: str, etype: str | None = None,
                        limit: int = 25, min_score: float | None = None):
        """Nearest entities by meaning. Returns ``[(score, row), ...]``.

        ``min_score`` defaults to the floor measured for this particular model.
        Note the honest limitation: no tested model separates a nonsensical
        query from a real one by score alone, so this floor removes obvious
        noise but cannot guarantee an empty result for gibberish. Scores are
        returned with every hit so the interface can show a weak match as weak.
        """
        import numpy as np

        embedder = self.embedder
        if embedder is None or not (query or "").strip():
            return []
        if min_score is None:
            min_score = embedder.min_score
        rows, matrix = self._vectors(etype)
        if not rows:
            return []
        from dms.embeddings import cosine_ranking
        scores = cosine_ranking(embedder.encode_one(query, is_query=True), matrix)
        order = np.argsort(-scores)[:limit]
        return [(float(scores[i]), rows[i]) for i in order
                if float(scores[i]) >= min_score]

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------
    def add_document(self, doc: ReceiptDocument, replace: bool = True,
                     embed: bool = True) -> int:
        """Insert (or replace) a processed receipt. Returns its ``doc_id``.

        ``embed`` also computes semantic vectors for the new entities, so the
        document is immediately searchable by meaning as well as by spelling.
        """
        with self._lock:
            return self._add_document(doc, replace, embed)

    def _add_document(self, doc: ReceiptDocument, replace: bool, embed: bool) -> int:
        cur = self.conn.cursor()
        if replace and doc.path:
            row = cur.execute("SELECT id FROM documents WHERE path = ?",
                              (doc.path,)).fetchone()
            if row:
                self.delete_document(row["id"], commit=False)

        cur.execute(
            """INSERT INTO documents
               (filename, path, ocr_text, ocr_conf, ocr_variant, language,
                fields_json, items_json, warnings_json, tokens_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            # NULL rather than '' - SQLite treats NULLs as distinct in a UNIQUE
            # index, so several in-memory documents can coexist without a clash.
            (doc.filename, doc.path or None, doc.ocr_text, doc.ocr_conf, doc.ocr_variant,
             doc.language, json.dumps(doc.fields, ensure_ascii=False),
             json.dumps(doc.items, ensure_ascii=False),
             json.dumps(doc.warnings, ensure_ascii=False),
             json.dumps(doc.tokens, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        doc_id = int(cur.lastrowid)

        new_ids: list[int] = []
        for ent in doc.entities:
            cur.execute(
                """INSERT INTO entities
                   (doc_id, type, value, value_norm, text, start, end,
                    confidence, source, meta_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                # An absent entity stores NULL in both value columns, so it can
                # never satisfy an equality or LIKE test and needs no guarding in
                # the exact and substring stages.
                (doc_id, ent.type, ent.value,
                 None if ent.value is None else normalize_key(ent.value),
                 ent.text, ent.start, ent.end, ent.confidence, ent.source,
                 json.dumps(ent.meta or {}, ensure_ascii=False)),
            )
            new_ids.append(int(cur.lastrowid))
            if self.has_fts and ent.value is not None:
                cur.execute(
                    "INSERT INTO entities_fts(rowid, value, text) VALUES (?,?,?)",
                    (cur.lastrowid, ent.value, ent.text or ""),
                )
        if self.has_fts:
            cur.execute("INSERT INTO documents_fts(rowid, ocr_text) VALUES (?,?)",
                        (doc_id, doc.ocr_text))
        self.conn.commit()
        self._vector_cache = None
        if embed and new_ids:
            self.embed_entities(new_ids)
        doc.doc_id = doc_id
        return doc_id

    def delete_document(self, doc_id: int, commit: bool = True) -> None:
        with self._lock:
            self._delete_document(doc_id, commit)

    def _delete_document(self, doc_id: int, commit: bool = True) -> None:
        cur = self.conn.cursor()
        if self.has_fts:
            for row in cur.execute("SELECT id FROM entities WHERE doc_id = ?",
                                   (doc_id,)).fetchall():
                cur.execute("DELETE FROM entities_fts WHERE rowid = ?", (row["id"],))
            cur.execute("DELETE FROM documents_fts WHERE rowid = ?", (doc_id,))
        cur.execute("DELETE FROM entities WHERE doc_id = ?", (doc_id,))
        cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._vector_cache = None
        if commit:
            self.conn.commit()

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    def get_document(self, doc_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM documents WHERE id = ?",
                                (doc_id,)).fetchone()
        if not row:
            return None
        doc = dict(row)
        doc["fields"] = json.loads(doc.pop("fields_json") or "{}")
        doc["items"] = json.loads(doc.pop("items_json") or "[]")
        doc["warnings"] = json.loads(doc.pop("warnings_json") or "[]")
        doc["tokens"] = json.loads(doc.pop("tokens_json", None) or "[]")
        doc["entities"] = self.get_entities(doc_id)
        return doc

    def get_entities(self, doc_id: int) -> list[dict]:
        rows = self.conn.execute(
            # Absent rows carry start=-1, so ordering by position alone would put
            # them first. What was found comes first; what is missing follows.
            "SELECT * FROM entities WHERE doc_id = ? "
            "ORDER BY (value IS NULL), start", (doc_id,)
        ).fetchall()
        out = []
        for r in rows:
            e = dict(r)
            e["meta"] = json.loads(e.pop("meta_json") or "{}")
            out.append(e)
        return out

    def list_documents(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, filename, path, language, ocr_conf, created_at, fields_json "
            "FROM documents ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["fields"] = json.loads(d.pop("fields_json") or "{}")
            out.append(d)
        return out

    def distinct_values(self, etype: str | None = None) -> list[tuple[str, str]]:
        # Absent-entity rows carry a NULL value and must never be offered as
        # something a query could resemble.
        sql = "SELECT DISTINCT type, value FROM entities WHERE value IS NOT NULL"
        params: tuple = ()
        if etype:
            sql += " AND type = ?"
            params = (etype,)
        return [(r["type"], r["value"]) for r in self.conn.execute(sql, params)]

    def stats(self) -> dict:
        c = self.conn.execute
        return {
            "documents": c("SELECT COUNT(*) n FROM documents").fetchone()["n"],
            # "entities" counts what was actually extracted. The rows recording
            # a type as absent are reported separately, so this number keeps
            # meaning the same thing it did before they existed.
            "entities": c("SELECT COUNT(*) n FROM entities "
                          "WHERE value IS NOT NULL").fetchone()["n"],
            "absent": c("SELECT COUNT(*) n FROM entities "
                        "WHERE value IS NULL").fetchone()["n"],
            "by_type": {r["type"]: r["n"] for r in c(
                "SELECT type, COUNT(*) n FROM entities WHERE value IS NOT NULL "
                "GROUP BY type ORDER BY n DESC")},
            # Found and absent side by side, so a chart can show that a type
            # with few values was looked for every time rather than ignored.
            "by_type_absent": {r["type"]: r["n"] for r in c(
                "SELECT type, COUNT(*) n FROM entities WHERE value IS NULL "
                "GROUP BY type")},
            "fts": self.has_fts,
            "path": str(self.path),
        }

    # ------------------------------------------------------------------
    # search  (the contract used by the search half of the project)
    # ------------------------------------------------------------------
    def _rows_to_hits(self, rows: Iterable[sqlite3.Row], match: str) -> list[dict]:
        hits = []
        for r in rows:
            hits.append({
                "doc_id": r["doc_id"],
                "filename": r["filename"],
                "entity_id": r["id"],
                "type": r["type"],
                "value": r["value"],
                "text": r["text"],
                "start": r["start"],
                "end": r["end"],
                "confidence": r["confidence"],
                "source": r["source"],
                "match": match,
            })
        return hits

    def search_entities(self, query: str, etype: str | None = None,
                        limit: int = 50, semantic: bool = True) -> dict[str, Any]:
        """Search entities with a graceful degradation cascade.

        Returns ``{"query", "mode", "hits", "suggestions"}`` where ``mode`` is
        one of ``exact`` / ``partial`` / ``similar`` / ``type_fallback`` /
        ``empty``. The caller can show ``mode`` to explain *why* a result set
        was returned - e.g. "no match for 'Johor Bahru'; showing other
        locations".

        The ``similar`` stage combines two independent notions of closeness:

        * **lexical** - edit-distance over spellings, which rescues a typo
          ("Kuala Lumpor" finds "Kuala Lumpur");
        * **semantic** - cosine distance between sentence embeddings, which
          rescues a different word for a related thing ("Johor Bahru" finds
          other Malaysian cities even though the strings share nothing).

        Each hit carries ``lexical_score``, ``semantic_score`` and
        ``matched_by`` so the interface can explain which mechanism found it.
        Semantic matching is skipped automatically when no embedding model is
        available, leaving the lexical cascade intact.
        """
        q = (query or "").strip()
        if not q:
            return {"query": query, "mode": "empty", "hits": [], "suggestions": []}
        qn = normalize_key(q)
        base = ("SELECT e.*, d.filename FROM entities e "
                "JOIN documents d ON d.id = e.doc_id ")
        type_clause = " AND e.type = ?" if etype else ""
        extra: tuple = (etype,) if etype else ()

        # 1. exact
        rows = self.conn.execute(
            base + "WHERE e.value_norm = ?" + type_clause + " LIMIT ?",
            (qn, *extra, limit)).fetchall()
        if rows:
            return {"query": q, "mode": "exact",
                    "hits": self._rows_to_hits(rows, "exact"), "suggestions": []}

        # 2. partial / substring
        rows = self.conn.execute(
            base + "WHERE e.value_norm LIKE ?" + type_clause + " LIMIT ?",
            (f"%{qn}%", *extra, limit)).fetchall()
        if rows:
            return {"query": q, "mode": "partial",
                    "hits": self._rows_to_hits(rows, "partial"), "suggestions": []}

        # 3. similar - lexical spelling AND semantic meaning, merged.
        #    They catch different things: lexical rescues a typo
        #    ("Kuala Lumpor"), semantic rescues a different word for a related
        #    thing ("Johor Bahru" -> other Malaysian cities). Running both and
        #    ranking the union covers more than either alone.
        candidates = self.distinct_values(etype)
        scored = sorted(
            ((partial_similarity(q, val), t, val) for t, val in candidates),
            key=lambda x: x[0], reverse=True,
        )
        merged: dict[tuple[str, str], dict] = {}
        for s, t, v in scored:
            if s >= SIMILARITY_THRESHOLD:
                merged[(t, v)] = {"lexical": round(s, 3), "semantic": None}

        if semantic:
            for s, row in self.semantic_search(q, etype, limit=10):
                key = (row["type"], row["value"])
                entry = merged.setdefault(key, {"lexical": None, "semantic": None})
                entry["semantic"] = round(s, 3)

        if merged:
            def rank(item):
                scores = item[1]
                return max(scores["lexical"] or 0.0, scores["semantic"] or 0.0)

            ordered = sorted(merged.items(), key=rank, reverse=True)[:10]
            hits: list[dict] = []
            for (t, val), scores in ordered:
                rows = self.conn.execute(
                    base + "WHERE e.value = ? AND e.type = ? LIMIT ?",
                    (val, t, limit)).fetchall()
                matched = ("both" if scores["lexical"] and scores["semantic"]
                           else "semantic" if scores["semantic"] else "lexical")
                for h in self._rows_to_hits(rows, "similar"):
                    h["similarity"] = rank(((t, val), scores))
                    h["lexical_score"] = scores["lexical"]
                    h["semantic_score"] = scores["semantic"]
                    h["matched_by"] = matched
                    hits.append(h)
            return {"query": q, "mode": "similar", "hits": hits[:limit],
                    "suggestions": [v for (_, v), _ in ordered]}

        # 4. type fallback - the "Johor Bahru -> Kuala Lumpur" behaviour.
        #    The query looks like a place but we hold none of it, so return the
        #    places we *do* hold rather than nothing at all.
        if self._looks_like_place(q) or (etype in LOCATION_TYPES if etype else False):
            types = [etype] if etype else sorted(LOCATION_TYPES)
            placeholders = ",".join("?" * len(types))
            rows = self.conn.execute(
                base + f"WHERE e.type IN ({placeholders}) AND e.value IS NOT NULL "
                "ORDER BY CASE e.type WHEN 'ADDRESS' THEN 0 ELSE 1 END, "
                "e.confidence DESC LIMIT ?", (*types, limit)).fetchall()
            if rows:
                return {"query": q, "mode": "type_fallback",
                        "hits": self._rows_to_hits(rows, "type_fallback"),
                        "suggestions": sorted({r["value"] for r in rows})[:10]}

        return {"query": q, "mode": "empty", "hits": [],
                "suggestions": [v for _, _, v in scored[:5]]}

    @staticmethod
    def _looks_like_place(query: str) -> bool:
        """Is this query a Malaysian place name (even one we have never seen)?"""
        qn = normalize_key(query)
        return any(qn == p or p in qn or qn in p for p in
                   (normalize_key(x) for x in MALAYSIAN_PLACES))

    def search_documents(self, query: str, limit: int = 20) -> list[dict]:
        """Free-text search across the raw OCR text of every document."""
        q = (query or "").strip()
        if not q:
            return []
        if self.has_fts:
            try:
                rows = self.conn.execute(
                    "SELECT d.id, d.filename FROM documents_fts f "
                    "JOIN documents d ON d.id = f.rowid "
                    "WHERE documents_fts MATCH ? LIMIT ?", (q, limit)).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                pass                                    # fall through to LIKE
        rows = self.conn.execute(
            "SELECT id, filename FROM documents WHERE ocr_text LIKE ? LIMIT ?",
            (f"%{q}%", limit)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # highlighting
    # ------------------------------------------------------------------
    def highlight_document(self, doc_id: int, entity_ids: list[int] | None = None,
                           marker: tuple[str, str] = ("[[", "]]")) -> str:
        """Return the document text with the given entities wrapped in markers."""
        doc = self.get_document(doc_id)
        if not doc:
            return ""
        ents = [e for e in doc["entities"]
                if e["start"] >= 0 and e["end"] > e["start"]
                and (entity_ids is None or e["id"] in entity_ids)]
        return apply_spans(doc["ocr_text"],
                           [(e["start"], e["end"]) for e in ents], marker)

    def highlight_html(self, doc_id: int, entity_ids: list[int] | None = None) -> str:
        """Same as :meth:`highlight_document` but as HTML ``<mark>`` spans."""
        doc = self.get_document(doc_id)
        if not doc:
            return ""
        ents = [e for e in doc["entities"]
                if e["start"] >= 0 and e["end"] > e["start"]
                and (entity_ids is None or e["id"] in entity_ids)]
        ents.sort(key=lambda e: e["start"])
        text, out, cursor = doc["ocr_text"], [], 0
        for e in _drop_overlaps(ents):
            out.append(escape(text[cursor:e["start"]]))
            out.append(
                f'<mark class="ent ent-{escape(e["type"])}" '
                f'title="{escape(e["type"])} ({e["confidence"]:.2f})">'
                f'{escape(text[e["start"]:e["end"]])}</mark>'
            )
            cursor = e["end"]
        out.append(escape(text[cursor:]))
        return "".join(out).replace("\n", "<br>\n")

    def close(self) -> None:
        self.conn.close()


# --------------------------------------------------------------------------
# helpers usable without a DB handle
# --------------------------------------------------------------------------

def _drop_overlaps(ents: list[dict]) -> list[dict]:
    """Keep the first of any overlapping spans so markers never interleave."""
    out, last_end = [], -1
    for e in sorted(ents, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        if e["start"] >= last_end:
            out.append(e)
            last_end = e["end"]
    return out


def apply_spans(text: str, spans: list[tuple[int, int]],
                marker: tuple[str, str] = ("[[", "]]")) -> str:
    """Wrap ``spans`` of ``text`` in ``marker``, skipping overlaps."""
    ents = [{"start": s, "end": e} for s, e in spans if 0 <= s < e <= len(text)]
    out, cursor = [], 0
    for e in _drop_overlaps(ents):
        out.append(text[cursor:e["start"]])
        out.append(marker[0] + text[e["start"]:e["end"]] + marker[1])
        cursor = e["end"]
    out.append(text[cursor:])
    return "".join(out)
