"""Turn the Mermaid blocks in docs/diagrams/DIAGRAMS.md into a viewable page.

DIAGRAMS.md is the single source of truth. This script extracts every fenced
``mermaid`` block from it, together with the ``## Figure N — title`` heading
above it, and writes a self-contained HTML page that renders them all.

The page has a "Download PNG" button under each figure, so a report can be
illustrated without installing any tooling: open the page, click, paste into
Word. It also offers SVG, which Word 2016 and later import directly and which
stays sharp when printed.

    python tools/render_diagrams.py
    python tools/render_diagrams.py --open      # and open it in a browser
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import re
import webbrowser

from dms.config import PROJECT_ROOT

DIAGRAM_DIR = PROJECT_ROOT / "docs" / "diagrams"
SOURCE = DIAGRAM_DIR / "DIAGRAMS.md"
OUTPUT = DIAGRAM_DIR / "diagrams.html"

# Matches "## Figure 1 — System architecture" ... ```mermaid ... ```
BLOCK = re.compile(
    r"^##\s+(Figure\s+\d+[^\n]*)\n(.*?)```mermaid\n(.*?)```",
    re.MULTILINE | re.DOTALL,
)


def extract(markdown: str) -> list[dict]:
    """Pull (title, blurb, mermaid source) out of the documentation file."""
    figures = []
    for match in BLOCK.finditer(markdown):
        title, blurb, code = match.groups()
        # The prose between the heading and the fence, minus markdown emphasis.
        text = re.sub(r"[*_`]", "", blurb).strip()
        figures.append({
            "title": title.strip(),
            "blurb": " ".join(text.split()),
            "code": code.strip(),
        })
    return figures


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Receipt DMS — system diagrams</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 0;
         background: #f5f6f8; color: #202124; }}
  header {{ background: #fff; border-bottom: 1px solid #dadce0;
            padding: 24px 32px; }}
  h1 {{ margin: 0 0 6px; font-size: 22px; }}
  header p {{ margin: 0; color: #5f6368; font-size: 14px; max-width: 70ch; }}
  main {{ padding: 24px 32px 64px; }}
  figure {{ background: #fff; border: 1px solid #dadce0; border-radius: 10px;
            margin: 0 0 28px; padding: 22px 24px; }}
  figcaption {{ font-size: 17px; font-weight: 600; margin-bottom: 4px; }}
  .blurb {{ color: #5f6368; font-size: 13.5px; margin-bottom: 16px;
            max-width: 80ch; line-height: 1.5; }}
  .render {{ overflow-x: auto; text-align: center; padding: 8px 0; }}
  .actions {{ margin-top: 14px; display: flex; gap: 8px; align-items: center; }}
  button {{ font: inherit; font-size: 13px; padding: 6px 14px; cursor: pointer;
            border: 1px solid #dadce0; border-radius: 6px; background: #fff; }}
  button:hover {{ background: #f1f3f4; }}
  .hint {{ font-size: 12px; color: #80868b; }}
  #offline {{ display: none; margin: 16px 0 0; padding: 14px 16px;
              background: #fce8e6; border: 1px solid #ea4335; border-radius: 8px;
              font-size: 14px; line-height: 1.55; }}
  #offline code {{ background: #fff; padding: 1px 5px; border-radius: 4px; }}
</style>
<!-- The classic (UMD) build is used deliberately rather than the ES-module one:
     a module script is fetched with CORS, which the browser refuses on a
     file:// page, so double-clicking this file would render nothing. -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body>
<header>
  <h1>Receipt DMS — system diagrams</h1>
  <p>Figures for the Methodology section. Click <b>PNG</b> to save a bitmap for
     Word, or <b>SVG</b> for a version that stays sharp at any size (Word 2016
     and later can insert SVG directly). Generated from
     <code>docs/diagrams/DIAGRAMS.md</code>.</p>
  <div id="offline">
    <b>The diagram library could not be loaded.</b> This page fetches Mermaid
    from the internet the first time it renders. Connect to the internet and
    reload, or view the diagram source in
    <code>docs/diagrams/DIAGRAMS.md</code> — GitHub renders it without any
    tooling.
  </div>
</header>
<main>
{figures}
</main>

<script>
if (typeof mermaid === "undefined") {{
  document.getElementById("offline").style.display = "block";
}} else {{
  mermaid.initialize({{ startOnLoad: true, theme: "base",
                       flowchart: {{ curve: "basis" }} }});
}}

function svgOf(id) {{ return document.getElementById(id).querySelector("svg"); }}

function saveBlob(blob, name) {{
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}}

function downloadSVG(id, name) {{
  const svg = svgOf(id);
  if (!svg) return;
  const text = new XMLSerializer().serializeToString(svg);
  saveBlob(new Blob([text], {{ type: "image/svg+xml" }}), name + ".svg");
}}

function downloadPNG(id, name) {{
  const svg = svgOf(id);
  if (!svg) return;
  // Render at 3x so the image survives being scaled up in a document.
  const scale = 3;
  const box = svg.getBoundingClientRect();
  const clone = svg.cloneNode(true);
  clone.setAttribute("width", box.width);
  clone.setAttribute("height", box.height);
  const text = new XMLSerializer().serializeToString(clone);
  const img = new Image();
  img.onload = () => {{
    const canvas = document.createElement("canvas");
    canvas.width = box.width * scale;
    canvas.height = box.height * scale;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(b => saveBlob(b, name + ".png"), "image/png");
  }};
  img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(text)));
}}

window.downloadSVG = downloadSVG;
window.downloadPNG = downloadPNG;
</script>
</body>
</html>
"""

FIGURE = """  <figure>
    <figcaption>{title}</figcaption>
    <div class="blurb">{blurb}</div>
    <div class="render" id="fig{n}"><pre class="mermaid">{code}</pre></div>
    <div class="actions">
      <button onclick="downloadPNG('fig{n}', 'figure{n}')">Download PNG</button>
      <button onclick="downloadSVG('fig{n}', 'figure{n}')">Download SVG</button>
      <span class="hint">saves as figure{n}.png / figure{n}.svg</span>
    </div>
  </figure>
"""


def build() -> Path:
    if not SOURCE.exists():
        raise SystemExit(f"missing {SOURCE}")
    figures = extract(SOURCE.read_text(encoding="utf-8"))
    if not figures:
        raise SystemExit(f"no ```mermaid blocks found in {SOURCE}")

    body = "".join(
        FIGURE.format(n=i, title=f["title"], blurb=f["blurb"],
                      code=f["code"].replace("&", "&amp;").replace("<", "&lt;"))
        for i, f in enumerate(figures, start=1)
    )
    OUTPUT.write_text(PAGE.format(figures=body), encoding="utf-8")
    print(f"{len(figures)} figure(s) -> {OUTPUT}")
    for i, f in enumerate(figures, start=1):
        print(f"  {i}. {f['title']}")
    return OUTPUT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--open", action="store_true",
                    help="open the page in the default browser afterwards")
    args = ap.parse_args()
    path = build()
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
