# System diagrams

Source for the figures in the Methodology section of the report.

Each diagram exists in two forms:

* **Mermaid** — the source below. GitHub renders it automatically.
* **PNG** — `docs/diagrams/*.png`, for pasting into Word.

The Mermaid text is the master copy. If a diagram needs changing, edit it here
and re-export, so the two never drift apart.

---

## Figure 1 — System architecture

The system in layers. Each layer depends only on the one above it, which is why
the language model can be switched off entirely and the rest still runs.

```mermaid
flowchart TB
    subgraph TOP [" &nbsp; EXTRACTION &nbsp; "]
        direction LR
        subgraph IN ["INPUT"]
            direction TB
            I1["Receipt image<br/><i>upload · folder<br/>dataset</i>"]
        end
        subgraph OCRL ["OCR &nbsp;·&nbsp; ocr.py"]
            direction TB
            O1["Preprocessing<br/><i>5 variants</i>"]
            O2["Recognition<br/><b>EasyOCR</b> ms+en<br/><b>Tesseract</b> msa+eng"]
            O3["Line rebuild<br/><i>offsets + boxes</i>"]
            O1 --> O2 --> O3
        end
        subgraph NERL ["NER &nbsp;·&nbsp; ner.py"]
            direction TB
            N1["Rule layer<br/><i>rules · lexicon</i>"]
            N2["LLM layer<br/><i>Qwen2.5-1.5B</i>"]
            N3["Merge<br/><i>per-field routing</i>"]
            N4["Arithmetic<br/>validation"]
            N1 --> N3
            N2 --> N3
            N3 --> N4
        end
        IN --> OCRL --> NERL
    end

    subgraph BOTTOM [" &nbsp; RETRIEVAL &nbsp; "]
        direction LR
        subgraph STORE ["STORAGE &nbsp;·&nbsp; database.py"]
            direction TB
            S1[("SQLite<br/>documents · entities<br/>tokens · FTS5")]
            S2[("Vectors<br/>BGE-M3 · 1024-d")]
        end
        subgraph SEARCH ["SEARCH"]
            direction TB
            R1["Cascade<br/><i>exact → partial<br/>→ similar → fallback</i>"]
            R2["Highlighting<br/><i>text + image</i>"]
            R1 --> R2
        end
        subgraph UI ["INTERFACE"]
            direction TB
            U1["Streamlit GUI<br/><i>app.py</i>"]
            U2["CLI<br/><i>run_dms.py</i>"]
        end
        STORE --> SEARCH --> UI
    end

    TOP --> BOTTOM

    classDef inp fill:#e8f0fe,stroke:#4285f4,stroke-width:2px
    classDef ocr fill:#fef7e0,stroke:#f9ab00,stroke-width:2px
    classDef ner fill:#e6f4ea,stroke:#34a853,stroke-width:2px
    classDef db  fill:#fce8e6,stroke:#ea4335,stroke-width:2px
    classDef ret fill:#f3e8fd,stroke:#a142f4,stroke-width:2px
    classDef ui  fill:#e8eaed,stroke:#5f6368,stroke-width:2px
    class I1 inp
    class O1,O2,O3 ocr
    class N1,N2,N3,N4 ner
    class S1,S2 db
    class R1,R2 ret
    class U1,U2 ui
```

---

## Figure 2 — Processing pipeline

What happens to one receipt, from image file to searchable record.
Timings are for the default configuration on an RTX 4060.

```mermaid
flowchart TB
    subgraph P1 ["STAGE 1 &nbsp;·&nbsp; READ THE IMAGE"]
        direction LR
        A(["Receipt<br/>image"]) --> B["<b>Preprocess</b><br/>upscale 1000 px · deskew<br/>Otsu · adaptive · CLAHE"]
        B --> C{"<b>Best<br/>variant?</b>"}
        C -->|"Σ word conf."| D["<b>OCR</b><br/>EasyOCR ms+en<br/><i>~5 s</i>"]
        D --> E["<b>Rebuild lines</b><br/>group by vertical centre<br/>record char_start/end"]
    end

    subgraph P2 ["STAGE 2 &nbsp;·&nbsp; FIND THE ENTITIES"]
        direction LR
        F["<b>Rule layer</b><br/>bilingual keyword+regex<br/><i>&lt;0.1 s</i>"] --> H
        G["<b>LLM layer</b><br/>few-shot BM+EN → JSON<br/><i>~12 s</i>"] --> H
        H{{"<b>Merge</b><br/><i>Figure 3</i>"}} --> I["<b>Validate</b><br/>subtotal+tax = total<br/>paid−change = total<br/>tax ≤ 35% of total"]
    end

    subgraph P3 ["STAGE 3 &nbsp;·&nbsp; MAKE IT SEARCHABLE"]
        direction LR
        J[("<b>Store</b><br/>entities · offsets<br/>pixel boxes")] --> K[("<b>Embed</b><br/>BGE-M3 vectors<br/>4 entity types")]
        K --> L(["Searchable<br/>record"])
    end

    P1 --> P2 --> P3

    classDef start fill:#e8f0fe,stroke:#4285f4,stroke-width:2px
    classDef proc  fill:#fef7e0,stroke:#f9ab00,stroke-width:1.5px
    classDef ner   fill:#e6f4ea,stroke:#34a853,stroke-width:1.5px
    classDef dec   fill:#fff,stroke:#5f6368,stroke-width:2px
    classDef db    fill:#fce8e6,stroke:#ea4335,stroke-width:1.5px
    class A,L start
    class B,D,E proc
    class F,G ner
    class C,H dec
    class I ner
    class J,K db
```

---

## Figure 3 — Hybrid merge and arbitration

**This is the core contribution.** It runs once per field. The two layers are
never averaged; one of them wins, and the reason it won is recorded.

```mermaid
flowchart TD
    A["One field<br/><i>e.g. TOTAL</i>"] --> B{"Which layers<br/>produced a value?"}

    B -->|"rule only"| C["Keep the rule value<br/><b>confidence 0.80</b>"]
    B -->|"LLM only"| D["Keep the LLM value<br/><b>confidence 0.70</b>"]
    B -->|"neither"| E["Field left empty"]
    B -->|"both"| F{"Do they<br/>agree?"}

    F -->|"yes"| G["Either value<br/><b>confidence 0.97</b><br/><i>agreement is the signal</i>"]
    F -->|"no"| H{"Who owns<br/>this field?"}

    H -->|"ADDRESS · ITEM"| I["LLM wins<br/><b>confidence 0.55</b>"]
    H -->|"everything else"| J["Rules win<br/><b>confidence 0.55</b>"]

    I --> K
    J --> K["Loser kept as<br/><i>alternative</i> in meta"]

    K --> L{"Is the money<br/>arithmetic<br/>consistent?"}
    L -->|"yes"| M["Done"]
    L -->|"no"| N["<b>Arithmetic arbitration</b><br/>try every winner/alternative<br/>combination; keep the one that<br/>balances and disturbs the<br/>least-confident readings"]
    N --> O["<b>confidence 0.90</b><br/><i>source = ...+arithmetic</i>"]

    C --> L
    D --> L
    G --> L

    classDef q    fill:#fff,stroke:#5f6368,stroke-width:2px
    classDef rule fill:#e6f4ea,stroke:#34a853,stroke-width:1.5px
    classDef llm  fill:#e8f0fe,stroke:#4285f4,stroke-width:1.5px
    classDef both fill:#fef7e0,stroke:#f9ab00,stroke-width:2px
    classDef arit fill:#fce8e6,stroke:#ea4335,stroke-width:2px
    class B,F,H,L q
    class C,J rule
    class D,I llm
    class G both
    class N,O arit
```

**Why agreement means something.** The two layers fail in unrelated ways — the
rules break on layout, the model drifts on digits. When two independent methods
that fail differently produce the same answer, that answer is very likely right.
That is the whole justification for the 0.97.

---

## Figure 4 — Search cascade

Each strategy is tried in turn; the first that returns anything wins, and the
mode is reported so a result set can explain itself. The requirement that
*"related documents are returned when the query is absent"* is satisfied by the
last two stages.

```mermaid
flowchart TB
    A(["Query &nbsp;<i>'Kuala Lumpor'</i>"]) --> B{"<b>1. Exact</b><br/>value_norm = query"}
    B -->|"hit"| Z1["mode = <b>exact</b>"]
    B -->|"miss"| C{"<b>2. Partial</b><br/>query inside value"}
    C -->|"hit"| Z2["mode = <b>partial</b>"]
    C -->|"miss"| D["<b>3. Similar</b> &nbsp;— run both, then union"]

    D --> E["Lexical &nbsp;<i>difflib window</i> &nbsp;≥ 0.72"]
    D --> F["Semantic &nbsp;<i>BGE-M3 cosine</i> &nbsp;≥ 0.53"]
    E --> G{"Anything<br/>found?"}
    F --> G
    G -->|"yes"| Z3["mode = <b>similar</b> &nbsp;<i>both scores shown</i>"]
    G -->|"no"| H{"<b>4. A Malaysian place name?</b>"}

    H -->|"yes"| Z4["mode = <b>type_fallback</b> &nbsp;<i>other locations returned</i>"]
    H -->|"no"| Z5["mode = <b>empty</b> &nbsp;<i>+ did-you-mean</i>"]

    classDef q  fill:#fff,stroke:#5f6368,stroke-width:2px
    classDef ok fill:#e6f4ea,stroke:#34a853,stroke-width:1.5px
    classDef mid fill:#fef7e0,stroke:#f9ab00,stroke-width:1.5px
    classDef no fill:#fce8e6,stroke:#ea4335,stroke-width:1.5px
    class B,C,G,H q
    class Z1,Z2 ok
    class Z3,Z4 mid
    class Z5 no
    class D,E,F mid
```

---

## Regenerating the PNGs

```bash
python tools/render_diagrams.py
```

Opens each Mermaid block in a headless browser, renders it and writes
`docs/diagrams/figure1.png` … `figure4.png`.
