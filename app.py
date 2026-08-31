"""
Receipt DMS — graphical interface.

    streamlit run app.py

Four pages:
    1. Process a receipt   — upload an image and watch every stage of the pipeline
    2. Search              — lexical and semantic entity search over the database
    3. Database            — what has been stored
    4. How it works        — the workflow, explained

Models are loaded once and cached for the life of the server, so the first
receipt is slow (weights load) and every one after it is quick.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from dms.config import OCR_ENGINE, DB_PATH, describe_runtime
from dms.database import ReceiptDB
from dms.ner import HybridNER, is_tax_inclusive
from dms.ocr import bounding_box, draw_entity_boxes, make_ocr, polygons_for_span
from dms.schema import ReceiptDocument
from dms.textutils import detect_language

st.set_page_config(page_title="Receipt DMS", page_icon="🧾", layout="wide",
                   initial_sidebar_state="expanded")

# Distinct colours per entity type, in BGR because OpenCV draws in BGR.
COLOURS = {
    "MERCHANT": (255, 120, 0), "ADDRESS": (0, 200, 0), "TOTAL": (0, 0, 255),
    "SUBTOTAL": (0, 140, 255), "TAX": (200, 0, 200), "DATE": (255, 200, 0),
    "TIME": (255, 200, 0), "PAID": (100, 100, 255), "CHANGE": (100, 100, 255),
    "INVOICE_NO": (0, 180, 180), "TAX_ID": (0, 180, 180), "PHONE": (150, 75, 0),
    "ITEM": (128, 128, 128), "PAYMENT_METHOD": (180, 0, 90),
}
CSS_COLOURS = {
    "MERCHANT": "#ff7800", "ADDRESS": "#00c800", "TOTAL": "#e00000",
    "SUBTOTAL": "#ff8c00", "TAX": "#c800c8", "DATE": "#c8a000",
    "TIME": "#c8a000", "PAID": "#6464ff", "CHANGE": "#6464ff",
    "INVOICE_NO": "#00b4b4", "TAX_ID": "#00b4b4", "PHONE": "#964b00",
    "ITEM": "#808080", "PAYMENT_METHOD": "#b4005a",
}


# --------------------------------------------------------------------------
# cached resources
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading OCR engine…")
def get_ocr(engine: str):
    return make_ocr(engine)


@st.cache_resource(show_spinner="Loading the language model…")
def get_ner(use_rules: bool, use_llm: bool, model: str | None):
    return HybridNER(use_rules=use_rules, use_llm=use_llm, model=model,
                     verbose=False)


@st.cache_resource(show_spinner="Opening the database…")
def get_db():
    return ReceiptDB(verbose=False)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def highlight_html(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Wrap the given (start, end, type) spans in coloured <mark> tags."""
    spans = sorted((s for s in spans if s[0] >= 0 and s[1] > s[0]),
                   key=lambda s: s[0])
    out, cursor, last_end = [], 0, -1
    from html import escape
    for start, end, etype in spans:
        if start < last_end:          # skip overlaps so markers cannot interleave
            continue
        colour = CSS_COLOURS.get(etype, "#888")
        out.append(escape(text[cursor:start]))
        out.append(
            f'<mark title="{escape(etype)}" style="background:{colour}33;'
            f'border-bottom:2px solid {colour};padding:1px 2px;border-radius:3px">'
            f'{escape(text[start:end])}</mark>'
        )
        cursor, last_end = end, end
    out.append(escape(text[cursor:]))
    body = "".join(out).replace("\n", "<br>")
    return (f'<div style="font-family:ui-monospace,Consolas,monospace;'
            f'font-size:13px;line-height:1.8">{body}</div>')


# --------------------------------------------------------------------------
# Page 1 — process a receipt
# --------------------------------------------------------------------------

def page_process() -> None:
    st.header("Process a receipt")
    st.caption("Upload a receipt and watch it move through every stage of the "
               "pipeline, from pixels to a searchable, validated record.")

    col_a, col_b = st.columns([2, 1])
    with col_b:
        st.subheader("Settings")
        engines = ["tesseract", "easyocr"]
        engine = st.selectbox(
            "OCR engine", engines,
            index=engines.index(OCR_ENGINE) if OCR_ENGINE in engines else 0,
            help="Tesseract is the default: on the 120-receipt training split "
                 "it read the four annotated fields far more accurately than "
                 "EasyOCR (63.3% against 51.7% exact, with address almost "
                 "doubling), for about 2.5s more per receipt. EasyOCR needs no "
                 "separate installation, so it is kept as the fallback.")
        variant = st.selectbox(
            "Preprocessing", ["auto", "all", "raw", "gray_otsu", "adaptive",
                              "clahe_sharp", "deskew_otsu"], index=0,
            help="'auto' tries three filters and keeps whichever reads best.")
        # Presented as an explicit choice rather than a toggle, because a
        # toggle labelled "use the language model" does not say what happens
        # when it is off. Both options are the ablation reported in section 5.
        method = st.radio(
            "Extraction method",
            ["Rules + language model", "Rules only"],
            index=0,
            help="The rule layer matches printed keywords such as TOTAL and "
                 "JUMLAH BESAR. The language model additionally reads the "
                 "layout, so it recovers fields whose printed label OCR "
                 "destroyed. Turning it off shows exactly what it contributes.")
        use_llm = method.startswith("Rules +")
        store = st.toggle("Save to database", value=True)
        st.caption(f"Runtime: {describe_runtime()}")

    with col_a:
        uploaded = st.file_uploader("Receipt image",
                                    type=["jpg", "jpeg", "png", "bmp", "tif",
                                          "tiff", "webp"])
        samples = sorted(Path("data/receipts").glob("*.jpg")) if \
            Path("data/receipts").exists() else []
        sample_names = ["— none —"] + [p.name for p in samples[:40]]
        chosen = st.selectbox("…or pick a sample already on disk", sample_names)

    image_bytes, source_name = None, None
    if uploaded is not None:
        image_bytes, source_name = uploaded.getvalue(), uploaded.name
    elif chosen != "— none —":
        path = Path("data/receipts") / chosen
        image_bytes, source_name = path.read_bytes(), chosen

    if image_bytes is None:
        st.info("Upload a receipt, or choose a sample, to begin.")
        return

    array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        st.error("That file could not be decoded as an image.")
        return

    if not st.button("Run the pipeline", type="primary", use_container_width=True):
        st.image(bgr_to_pil(array), caption=source_name, width=380)
        return

    # ---- Stage 1 & 2: preprocessing and OCR --------------------------------
    st.divider()
    st.subheader("Stage 1 & 2 · Preprocessing and OCR")
    with st.spinner("Reading the image…"):
        t0 = time.time()
        ocr = get_ocr(engine).read(array, variant=variant)
        t_ocr = time.time() - t0

    if not ocr.text.strip():
        st.error("OCR produced no text. Try a different preprocessing variant.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filter chosen", ocr.variant)
    m2.metric("Mean confidence", f"{ocr.mean_conf:.3f}")
    m3.metric("Lines rebuilt", len(ocr.lines))
    m4.metric("Time", f"{t_ocr:.1f}s")

    with st.expander("Why this filter won"):
        scores = {k: v for k, v in ocr.attempts.items() if not k.startswith("_")}
        if scores:
            st.bar_chart(scores)
        st.caption(
            "Each candidate filter is scored by the **sum of its word "
            "confidences** — the expected number of correctly read words. That "
            "rewards reading *more* text and reading it *well*; average "
            "confidence alone would prefer a filter that found one crisp word "
            "and missed the rest of the receipt.")

    left, right = st.columns(2)
    with left:
        st.image(bgr_to_pil(array), caption="Original", use_container_width=True)
    with right:
        st.text_area("Text after line reconstruction", ocr.text, height=340)
        st.caption(
            "OCR returns loose boxes. They are regrouped into **visual lines** "
            "before anything else runs — on a receipt the layout *is* the "
            "meaning, and flattening it would sever `JUMLAH BESAR` from `60.31`.")

    # ---- Stage 3: NER ------------------------------------------------------
    st.divider()
    st.subheader("Stage 3 · Named entity recognition")
    with st.spinner("Extracting entities…"):
        t0 = time.time()
        ner = get_ner(True, use_llm, None)
        entities, fields, items, warnings = ner.extract(ocr.text)
        t_ner = time.time() - t0

    for ent in entities:
        if ent.is_aligned:
            polys = polygons_for_span(ocr.tokens, ent.start, ent.end)
            if polys:
                ent.meta = dict(ent.meta or {}, bbox=bounding_box(polys))

    c1, c2, c3 = st.columns(3)
    c1.metric("Entities found", len(entities))
    c2.metric("Language detected", detect_language(ocr.text))
    c3.metric("Time", f"{t_ner:.1f}s")

    st.markdown("**Extracted record**")
    shown = {k: v for k, v in fields.items() if v is not None}
    if shown:
        st.dataframe(
            [{"Field": k, "Value": str(v)} for k, v in shown.items()],
            use_container_width=True, hide_index=True)
    if items:
        st.markdown("**Line items**")
        st.dataframe(items, use_container_width=True, hide_index=True)

    with st.expander("Every entity, with its confidence and which layer found it",
                     expanded=False):
        st.dataframe([{
            "Type": e.type, "Value": e.value, "As printed": e.text,
            "Confidence": round(e.confidence, 2), "Found by": e.source,
            "Characters": f"{e.start}–{e.end}" if e.is_aligned else "—",
        } for e in entities], use_container_width=True, hide_index=True)
        st.caption(
            "**Found by** — `rule` = keyword/pattern match · `llm` = the "
            "language model · `rule+llm` = both agreed, so confidence is 0.97 · "
            "`arithmetic+text` = neither read it, but the receipt's own maths "
            "implied a value that is genuinely printed on the paper.")

    # ---- Stage 4: validation ----------------------------------------------
    st.divider()
    st.subheader("Stage 4 · Validation")
    if is_tax_inclusive(ocr.text):
        st.info("This receipt states its prices already include tax, so the "
                "`subtotal + tax = total` check is deliberately suppressed.")
    if warnings:
        for w in warnings:
            st.warning(w)
        st.caption("These are the checks doing their job — a corrected value or "
                   "a disagreement recorded for review, not errors.")
    else:
        st.success("Both layers agreed and the arithmetic balanced. Nothing to "
                   "correct.")

    # ---- Stage 5: grounding -----------------------------------------------
    st.divider()
    st.subheader("Stage 5 · Grounding — where each value came from")
    st.caption("Every value is traced back to the page: "
               "**pixel box → character span → cleaned value**. This is what "
               "makes highlighting possible.")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**On the image**")
        st.image(bgr_to_pil(draw_entity_boxes(array, entities, COLOURS)),
                 use_container_width=True)
    with g2:
        st.markdown("**In the text**")
        st.markdown(
            highlight_html(ocr.text,
                           [(e.start, e.end, e.type) for e in entities
                            if e.is_aligned]),
            unsafe_allow_html=True)

    # ---- Stage 6: storage --------------------------------------------------
    st.divider()
    st.subheader("Stage 6 · Storage and semantic indexing")
    if not store:
        st.info("Saving is switched off, so this receipt was not stored.")
        return

    doc = ReceiptDocument(
        filename=source_name or "upload.jpg", path=f"upload://{source_name}",
        ocr_text=ocr.text, ocr_conf=ocr.mean_conf, ocr_variant=ocr.variant,
        language=detect_language(ocr.text), entities=entities, fields=fields,
        items=items,
        tokens=[{k: t[k] for k in ("text", "conf", "bbox", "line",
                                   "char_start", "char_end")} for t in ocr.tokens],
        warnings=warnings)
    with st.spinner("Storing and embedding…"):
        db = get_db()
        doc_id = db.add_document(doc)
    st.success(f"Stored as document #{doc_id} in `{DB_PATH.name}` — "
               "searchable immediately, by keyword and by meaning.")
    st.caption("Merchant, address and item values are also encoded as vectors "
               "so they can be found semantically. Amounts and dates are not "
               "embedded: a number has no meaning to compare, and embedding it "
               "only pollutes the results.")


# --------------------------------------------------------------------------
# Page 2 — search
# --------------------------------------------------------------------------

def page_search() -> None:
    st.header("Search")
    st.caption("Four strategies, tried in order. The one that answered is "
               "always shown, so a result set can explain itself.")

    db = get_db()
    if db.stats()["documents"] == 0:
        st.warning("The database is empty. Process a receipt first.")
        return

    col1, col2, col3 = st.columns([3, 1, 1])
    query = col1.text_input("Search for an entity",
                            placeholder="Kuala Lumpur · Johor Bahru · 60.31 · nasi")
    etype = col2.selectbox("Type", ["any"] + sorted(
        {t for t, _ in db.distinct_values()}))
    semantic = col3.toggle("Semantic", value=True,
                           help="Off = spelling only, no embeddings")

    st.caption("Try **Kota Kinabalu** — a city not in the database — to see the "
               "similar-entity fallback.")
    if not query.strip():
        return

    with st.spinner("Searching…"):
        result = db.search_entities(query, etype=None if etype == "any" else etype,
                                    limit=50, semantic=semantic)

    explain = {
        "exact": ("✅ Exact match", "The stored value is identical to the query."),
        "partial": ("✅ Contains the query",
                    "The query appears inside a stored value."),
        "similar": ("🔍 Nothing matched exactly — closest entities",
                    "Ranked by spelling similarity and by meaning."),
        "type_fallback": (f"📍 '{query}' is not in the database",
                          "It is recognised as a Malaysian place name, so the "
                          "other locations on file are returned instead — this "
                          "is the fallback the assignment asks for."),
        "empty": ("❌ Nothing found",
                  "No stored entity is close enough, by spelling or by meaning."),
    }[result["mode"]]
    st.subheader(explain[0])
    st.caption(explain[1] + f"  ·  mode = `{result['mode']}`")

    if not result["hits"]:
        if result["suggestions"]:
            st.info("Did you mean: " + " · ".join(result["suggestions"][:5]))
        return

    by_doc: dict[int, list[dict]] = {}
    for hit in result["hits"]:
        by_doc.setdefault(hit["doc_id"], []).append(hit)
    st.write(f"**{len(result['hits'])} entities across "
             f"{len(by_doc)} document(s)**")

    for doc_id, hits in by_doc.items():
        doc = db.get_document(doc_id)
        if not doc:
            continue
        head = doc["fields"].get("merchant") or doc["filename"]
        with st.expander(f"📄 #{doc_id} · {head} · total "
                         f"{doc['fields'].get('total') or '—'}", expanded=True):
            rows = []
            for h in hits:
                rows.append({
                    "Type": h["type"], "Value": h["value"],
                    "Confidence": round(h["confidence"], 2),
                    "Spelling": h.get("lexical_score") or "—",
                    "Meaning": h.get("semantic_score") or "—",
                    "Matched by": h.get("matched_by", h["match"]),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.markdown("**The document, with the hit highlighted**")
            spans = [(h["start"], h["end"], h["type"]) for h in hits
                     if h["start"] >= 0]
            st.markdown(highlight_html(doc["ocr_text"], spans),
                        unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Page 3 — database
# --------------------------------------------------------------------------

def page_database() -> None:
    st.header("Database")
    db = get_db()
    stats = db.stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", stats["documents"])
    c2.metric("Entities", stats["entities"])
    c3.metric("Full-text search", "on" if stats["fts"] else "off")
    embedded = db.conn.execute(
        "SELECT COUNT(*) n FROM entities WHERE embedding IS NOT NULL").fetchone()["n"]
    c4.metric("Vectors", embedded)

    if stats["documents"] == 0:
        st.info("Nothing stored yet.")
        return

    st.subheader("Stored documents")
    st.dataframe([{
        "#": d["id"], "File": d["filename"], "Language": d["language"],
        "Merchant": d["fields"].get("merchant"), "Date": d["fields"].get("date"),
        "Total": d["fields"].get("total"), "OCR confidence": round(d["ocr_conf"], 3),
    } for d in db.list_documents(limit=500)],
        use_container_width=True, hide_index=True)

    st.subheader("Entities by type")
    st.bar_chart(stats["by_type"])

    with st.expander("How the database is organised"):
        st.markdown(f"""
The whole system lives in one SQLite file — `{DB_PATH.name}` — with no server
process, which keeps it portable.

**`documents`** — the OCR text, the filter that won, the detected language, the
flattened record, and every word box.

**`entities`** — one row per entity: the cleaned value, the surface form as
printed, **character offsets**, a confidence, which layer found it, and a
**1024-dimension vector** for semantic search.

Similarity is a plain dot product across all stored vectors. At this scale that
is the right choice: 10,000 entities is a 40 MB matrix and one multiplication —
microseconds. A dedicated vector database such as FAISS only starts to pay off
in the millions, and would add a dependency for no measurable gain.
        """)

    st.caption(
        "Deleting is deliberately not available here — a single stray click "
        "during a demonstration would destroy the corpus, and there is no undo. "
        "Use the command line, which asks for confirmation:  "
        "`python run_dms.py delete --all`")


# --------------------------------------------------------------------------
# Page 4 — how it works
# --------------------------------------------------------------------------

def page_workflow() -> None:
    st.header("How it works")
    st.caption("The full path from a photograph to a searchable, validated "
               "record.")

    st.code("""
   receipt image
        │
   [1] PREPROCESS   five filters generated; the OCR picks the winner
        │
   [2] OCR          Malay + English; every word keeps its pixel box
        │
   [3] REBUILD      boxes regrouped into visual LINES
        │           (layout is meaning: "JUMLAH BESAR" ── "60.31")
        │
        ├──────────────────────┬──────────────────────┐
   [4a] RULE LAYER        [4b] LANGUAGE MODEL
        keyword + pattern      reads and understands
        exact, never invents   handles missing labels
        └──────────────────────┴──────────────────────┘
        │
   [5] MERGE        agree → 0.97 confidence
        │           only one found it → use it
        │           disagree → the field's owner wins
        │
   [6] VALIDATE     subtotal + tax = total?
        │           paid − change = total?
        │           is a 94% "tax" plausible?  (no)
        │
   [7] GROUND       value → character span → pixel box
        │
   [8] STORE        SQLite + 1024-d vectors
        │
   [9] SEARCH       exact → contains → similar → place fallback
        │
   [10] HIGHLIGHT   in the text and on the image
    """, language=None)

    st.subheader("The two readers, and why there are two")
    a, b = st.columns(2)
    a.markdown("""
#### 🔤 The rule layer
Knows a bilingual keyword list — *TOTAL, JUMLAH BESAR, CUKAI, TUNAI, TARIKH* —
finds the label and takes the value beside it.

**Good:** instant, free, completely explainable, and it *cannot* invent a value.

**Bad:** if OCR destroys the printed label, it finds nothing at all.
    """)
    b.markdown("""
#### 🧠 The language model
Qwen2.5-1.5B reads the whole receipt and returns a structured record.

**Good:** understands layout, copes with missing labels, repairs OCR damage
(`22-60` → `22.60`).

**Bad:** slower, and it can quietly guess wrong.
    """)

    st.info("**Hybrid** is not a third model. Both readers run on the same text, "
            "independently, and a referee reconciles their answers — then checks "
            "the result against the receipt's own arithmetic.")

    with st.expander("**A worked example** — how the tax on the sample receipt "
                     "was repaired", expanded=False):
        st.markdown("""
On `sample_receipt.jpg` the tax line is smudged, and OCR reads it as `1.2g`.

1. **Rule layer** — finds the label `CUKAI`, but cannot parse `1.2g` as a
   number, so it reports **0.00**
2. **Language model** — reads the same line as **1.28**
3. **They disagree.** `TAX` is a rule-owned field, so the rule value wins: 0.00
4. **Arithmetic check** — subtotal 21.32 + tax 0.00 = 21.32, but the receipt's
   total says 22.60 ✗
5. **Try the alternative** — 21.32 + **1.28** = 22.60 ✓
6. **Verify before adopting** — is 1.28 genuinely printed on the receipt? It is
7. Tax becomes **1.28** at confidence 0.90, and the system explains itself:
""")
        st.code("! TAX: 0.00 -> 1.28 (total - subtotal, and printed on the "
                "receipt)", language=None)
        st.caption("Neither layer was trusted on its own. The receipt's own "
                   "arithmetic settled it, and the correction is reported "
                   "rather than applied silently.")

    st.subheader("Measured results")
    st.caption("All 97 test receipts, read with Tesseract. The configuration was "
               "chosen on separate training data and the test split scored once, "
               "so these are not flattered by tuning — train scored 63.3% "
               "against test's 63.6%, a gap of −0.2 points.")
    st.dataframe([
        {"Configuration": "Rule layer only", "Exact": "63.8%", "Close enough": "73.9%",
         "Date found": "87.6%"},
        {"Configuration": "Language model only", "Exact": "52.2%", "Close enough": "64.9%",
         "Date found": "100%"},
        {"Configuration": "Hybrid (shipped)", "Exact": "63.6%", "Close enough": "77.3%",
         "Date found": "100%"},
    ], use_container_width=True, hide_index=True)
    st.caption(
        "Read the first two columns together. On **exact** match the rule layer "
        "alone is level with the hybrid — 63.8% against 63.6%, well inside the "
        "noise of 97 receipts. That was not true with the weaker OCR engine, "
        "where the hybrid led by three points: a better reader leaves less for "
        "the language model to repair.")
    st.caption(
        "The hybrid earns its place elsewhere — 3.4 points ahead on close "
        "matches, and more *complete*: it finds a date on every receipt against "
        "the rules' 87.6%, and reaches 72.9% close-match on addresses against "
        "63.5%.")

    st.subheader("Why search has four strategies")
    st.markdown("""
| Strategy | Handles | Example |
|---|---|---|
| **Exact** | the value as stored | `60.31` |
| **Contains** | a word inside a longer value | `Kuala Lumpur` inside a full address |
| **Similar** | typing errors *and* related meanings | `Kuala Lumpor` · `nasi ayam` |
| **Place fallback** | a Malaysian place we simply do not hold | `Johor Bahru` → returns the Kuala Lumpur receipts |

Spelling similarity and meaning similarity catch different things, so both run
and the results are merged. Spelling rescues a typo; meaning rescues a different
word for a related thing.

One honest limitation: no embedding model tested could reliably reject pure
gibberish by similarity score alone, so a per-model threshold is used and weak
matches are shown with their score rather than hidden.
    """)


# --------------------------------------------------------------------------

# Ordered as the system is explained, not as it was built: someone opening the
# app cold - a marker, a new teammate - needs the overview before the controls.
PAGES = {
    "How it works": page_workflow,
    "Process a receipt": page_process,
    "Search": page_search,
    "Database": page_database,
}


def main() -> None:
    st.sidebar.title("🧾 Receipt DMS")
    st.sidebar.caption("Bilingual receipt extraction and search — Bahasa "
                       "Melayu and English, running entirely on this machine.")
    choice = st.sidebar.radio("Page", list(PAGES))
    st.sidebar.divider()
    try:
        stats = get_db().stats()
        st.sidebar.metric("Documents stored", stats["documents"])
        st.sidebar.metric("Entities indexed", stats["entities"])
    except Exception as exc:
        st.sidebar.error(f"Database unavailable: {exc}")
    st.sidebar.divider()
    st.sidebar.caption("No API key, no subscription, no data leaves this "
                       "computer.")
    PAGES[choice]()


if __name__ == "__main__":
    main()
