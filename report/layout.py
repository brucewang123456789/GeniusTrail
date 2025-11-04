# C:\CiteVizor\report\layout.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Report layout generator (HTML)
- Inputs:
    storage/evidence/_rank/<run_id>.citations.json  (from align/citation_map.py)
    storage/evidence/_rank/<run_id>.summary.json    (from llm/summarize_engineer.py) [optional but recommended]
    storage/evidence/<doc_id>/pack.json             (to resolve figure image paths)
    storage/renders/<doc_id>/charts/                (generated charts from llm/chart_codegen.py)
    storage/evidence/<doc_id>                       (may contain direct web images for image-sourced docs)
- Output:
    storage/reports/<run_id>.html
    storage/reports/<run_id>_assets/*               (copied images)
- Design:
    2-column responsive grid: left (visuals ~70–75%), right (highlights ≤30%).
    Visual priority: charts → web_image → figures. References at end.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, BaseLoader, select_autoescape

from schemas import (
    EVIDENCE_DIR,
    RENDERS_DIR,
    REPORTS_DIR,
    EvidenceType,
)

# ----------------------------- IO helpers -------------------------------------

def _rank_dir() -> Path:
    d = EVIDENCE_DIR / "_rank"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _load_citations(run_id: str) -> Dict[str, Any]:
    return _load_json(_rank_dir() / f"{run_id}.citations.json")

def _load_summary(run_id: str) -> Dict[str, Any]:
    return _load_json(_rank_dir() / f"{run_id}.summary.json")

def _load_pack(doc_id: str) -> Dict[str, Any]:
    p = EVIDENCE_DIR / doc_id / "pack.json"
    return _load_json(p)

def _charts_for_doc(doc_id: str) -> List[Path]:
    d = RENDERS_DIR / doc_id / "charts"
    if not d.exists():
        return []
    return sorted([p for p in d.glob("*.png") if p.is_file()])

def _assets_dir(run_id: str) -> Path:
    d = REPORTS_DIR / f"{run_id}_assets"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _copy_into_assets(src: Path, assets_dir: Path, prefix: str) -> str:
    """
    Copy image into assets dir; return relative path used in HTML.
    """
    ext = (src.suffix.lower() or ".png")
    dst = assets_dir / f"{prefix}{ext}"
    # ensure unique name if collision
    i = 1
    while dst.exists():
        dst = assets_dir / f"{prefix}_{i}{ext}"
        i += 1
    try:
        shutil.copyfile(src, dst)
    except Exception:
        # best-effort: skip if copy fails
        return ""
    return f"{assets_dir.name}/{dst.name}"  # relative to storage/reports

# ----------------------------- Data helpers -----------------------------------

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff"}

def _host_of(u: Optional[str]) -> str:
    if not u:
        return ""
    try:
        from urllib.parse import urlparse
        h = urlparse(u).netloc.lower()
        return h.replace("www.", "")
    except Exception:
        return ""

def _list_web_images_for_doc(doc_id: str) -> List[Path]:
    """
    Discover direct web images saved under evidence/<doc_id>.
    Exclude typical figure/chart folders to avoid duplicates.
    """
    root = EVIDENCE_DIR / doc_id
    if not root.exists():
        return []
    out: List[Path] = []
    exclude_dirs = {"figures", "charts"}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        # exclude extracted figure images and generated charts (handled elsewhere)
        parts = {x.name.lower() for x in p.parents if x != root}
        if "figures" in parts or "charts" in parts:
            continue
        out.append(p)
    # prefer top-level images first, stable order
    out.sort(key=lambda x: (len(x.relative_to(root).parts), x.name.lower()))
    return out

# ----------------------------- Data models ------------------------------------

@dataclass
class VisualItem:
    kind: str            # "chart" | "web_image" | "figure"
    rel_src: str         # "<run_id>_assets/xxxx.png"
    caption: str         # human-readable caption
    label: str           # compact label like [3 p.7, Fig.2] or [image]
    doc_id: str
    evidence_id: Optional[str] = None

# ----------------------------- Template ---------------------------------------

_TEMPLATE = r"""
<!doctype html>
<html lang="{{lang}}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <style>
    :root {
      --accent: #0f766e;
      --muted: #556;
      --bg: #fff;
      --text: #111;
    }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, "Noto Sans", Arial, "Helvetica Neue", Helvetica, "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--text);
      background: var(--bg);
      line-height: 1.5;
    }
    header {
      padding: 28px 28px 14px 28px;
      border-bottom: 2px solid #eee;
    }
    header h1 {
      margin: 0 0 6px 0;
      font-size: 24px;
    }
    header .sub {
      color: var(--muted);
      font-size: 13px;
    }
    .container {
      display: grid;
      grid-template-columns: 3fr 1.2fr; /* ~71% visuals, ~29% text */
      gap: 18px;
      padding: 18px 24px 28px 24px;
    }
    @media (max-width: 1100px) {
      .container { grid-template-columns: 1fr; }
      .sidebar { order: -1; }
    }
    /* Visual grid */
    .visuals {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 14px;
      align-items: start;
    }
    figure {
      margin: 0;
      background: #fafafa;
      border: 1px solid #eee;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 1px rgba(0,0,0,0.04);
    }
    figure img {
      width: 100%;
      height: auto;
      display: block;
    }
    figure figcaption {
      font-size: 12px;
      color: #333;
      padding: 8px 10px;
      border-top: 1px solid #eee;
    }
    .label {
      font-weight: 600; color: var(--accent); margin-right: 6px;
    }
    /* Sidebar highlights */
    .sidebar {
      position: sticky;
      top: 12px;
      align-self: start;
      max-height: calc(100vh - 40px);
      overflow: auto;
      border: 1px solid #eee;
      border-radius: 10px;
      padding: 12px 14px;
      background: #fff;
    }
    .sidebar h2 {
      margin: 2px 0 6px 0;
      font-size: 16px;
    }
    .hl {
      font-size: 14px;
      margin: 0 0 8px 0;
      padding-left: 18px;
      text-indent: -12px;
    }
    .hl::before {
      content: "•";
      color: var(--accent);
      font-weight: 700;
      margin-right: 8px;
    }
    /* References */
    .refs {
      margin: 8px 24px 28px 24px;
      border-top: 2px solid #eee;
      padding-top: 12px;
    }
    .ref-item { font-size: 12px; color: #333; margin: 4px 0; }
    .muted { color: var(--muted); }
    .small { font-size: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>{{ title }}</h1>
    <div class="sub small">
      Audience: {{ audience }} • Image-first report (≥70%) • Run: {{ run_id }} • Query: <span class="muted">{{ query }}</span>
    </div>
  </header>

  <main class="container">
    <section class="visuals">
      {% for v in visuals %}
      <figure>
        <img src="{{ v.rel_src }}" alt="{{ v.kind }} {{ loop.index }}">
        <figcaption>
          <span class="label">{{ v.label }}</span>{{ v.caption }}
          {% if v.evidence_id %}<span class="muted small"> • {{ v.kind }} • {{ v.doc_id }} • {{ v.evidence_id }}</span>{% else %}<span class="muted small"> • {{ v.kind }} • {{ v.doc_id }}</span>{% endif %}
        </figcaption>
      </figure>
      {% endfor %}
    </section>

    <aside class="sidebar">
      <h2>Key highlights</h2>
      {% for h in highlights %}
        <p class="hl">{{ h }}</p>
      {% endfor %}
    </aside>
  </main>

  <section class="refs">
    <div class="small muted">References</div>
    {% for r in refs %}
      <div class="ref-item">[{{ r.ref_id }}] {{ r.title or "&lt;untitled&gt;" }}{% if r.venue %}, {{ r.venue }}{% endif %}{% if r.year %}, {{ r.year }}{% endif %}{% if r.doi %}, doi:{{ r.doi }}{% elif r.url %}, {{ r.url }}{% endif %}</div>
    {% endfor %}
  </section>
</body>
</html>
"""

# ----------------------------- Core generator ---------------------------------

@dataclass
class LayoutSpec:
    title: str
    lang: str = "en"
    audience: str = "engineer"
    max_visuals: int = 18  # cap visuals for perf/size

class ReportLayout:
    def __init__(self, spec: LayoutSpec):
        self.spec = spec
        self.env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))

    def build(self, run_id: str) -> Path:
        ci = _load_citations(run_id)
        if not ci:
            raise FileNotFoundError(f"citations json not found for run_id={run_id}")
        su = _load_summary(run_id)

        title = self.spec.title or f"CiteVizor Report – {run_id}"
        lang = self.spec.lang
        audience = self.spec.audience
        query = ci.get("query", "")

        # Highlights (fallback to top-5 snippets if summary is missing)
        highlights: List[str] = []
        if su and isinstance(su.get("highlights"), list) and su["highlights"]:
            highlights = su["highlights"][:8]
        else:
            for it in ci.get("items", [])[:5]:
                label = it.get("label") or f"[{it.get('ref_id')}]"
                snip = (it.get("snippet") or "").replace("\n", " ").strip()
                txt = f"{snip} {label}".strip()
                if txt:
                    highlights.append(txt)

        visuals: List[VisualItem] = []
        assets_dir = _assets_dir(run_id)

        # doc_ids referenced in citations
        doc_ids = list({it.get("doc_id") for it in ci.get("items", []) if it.get("doc_id")})

        # 1) Charts (from renders)
        for doc_id in doc_ids:
            for p in _charts_for_doc(doc_id):
                rel = _copy_into_assets(p, assets_dir, prefix=f"{doc_id}_chart_{p.stem}")
                if not rel:
                    continue
                cap = f"Auto-generated chart from table in {doc_id}."
                # attach nearest table label if available
                label = ""
                for it in ci.get("items", []):
                    if it.get("doc_id") == doc_id and it.get("type") == "table":
                        label = it.get("label") or ""
                        break
                visuals.append(VisualItem(kind="chart", rel_src=rel, caption=cap, label=label or "[table]", doc_id=doc_id))
                if len(visuals) >= self.spec.max_visuals:
                    break
            if len(visuals) >= self.spec.max_visuals:
                break

        # 2) Web images (direct images downloaded for image-sourced candidates)
        #    Priority after charts, before figures, to help when PDF parsing fails.
        for doc_id in doc_ids:
            if len(visuals) >= self.spec.max_visuals:
                break
            # candidate-type hint for caption
            items_this_doc = [it for it in ci.get("items", []) if it.get("doc_id") == doc_id]
            has_image_source = any((it.get("source") == "image") or (it.get("type") in ("image", "web_image")) for it in items_this_doc)
            if not has_image_source:
                # also allow fallback if an image exists in the doc folder (e.g., from image vertical)
                imgs = _list_web_images_for_doc(doc_id)
                if not imgs:
                    continue
                # no typed hint, but files exist
                candidates = imgs
                caption_hint = ""
                label_hint = "[image]"
            else:
                candidates = _list_web_images_for_doc(doc_id)
                caption_hint = ""
                label_hint = ""
                # derive caption/label from the first image-typed item
                for it in items_this_doc:
                    if (it.get("source") == "image") or (it.get("type") in ("image", "web_image")):
                        caption_hint = (it.get("snippet") or it.get("title") or "").strip()
                        label_hint = (it.get("label") or "[image]").strip() or "[image]"
                        break

            for p in candidates:
                if len(visuals) >= self.spec.max_visuals:
                    break
                rel = _copy_into_assets(p, assets_dir, prefix=f"{doc_id}_webimg_{p.stem}")
                if not rel:
                    continue
                # fallback caption from host if snippet missing
                if not caption_hint:
                    # try to find a url for this doc_id to extract host
                    host = ""
                    for it in items_this_doc:
                        if it.get("url"):
                            host = _host_of(it.get("url"))
                            if host:
                                break
                    caption = f"Image from {host}" if host else "Image from web source"
                else:
                    caption = caption_hint
                label = label_hint or "[image]"
                visuals.append(VisualItem(kind="web_image", rel_src=rel, caption=caption, label=label, doc_id=doc_id))

        # 3) Figures (from pack.json, by evidence_id)
        fig_caps_map = {}
        if su and isinstance(su.get("figure_captions"), dict):
            fig_caps_map = su["figure_captions"]

        for it in ci.get("items", []):
            if len(visuals) >= self.spec.max_visuals:
                break
            if it.get("type") != "figure":
                continue
            doc_id = it.get("doc_id")
            ev_id = it.get("evidence_id")
            pack = _load_pack(doc_id)
            ev = None
            for e in pack.get("figures", []):
                if e.get("id") == ev_id:
                    ev = e
                    break
            if not ev or not ev.get("image_path"):
                continue
            img_path = Path(ev["image_path"])
            if not img_path.exists():
                # handle relative paths saved earlier
                img_path = (EVIDENCE_DIR / doc_id / "figures" / Path(img_path).name)
                if not img_path.exists():
                    continue

            rel = _copy_into_assets(img_path, assets_dir, prefix=f"{doc_id}_fig_{Path(img_path).stem}")
            if not rel:
                continue
            label = it.get("label") or ""
            cap = fig_caps_map.get(ev_id) or ev.get("caption") or (it.get("snippet") or "")
            visuals.append(VisualItem(kind="figure", rel_src=rel, caption=cap, label=label or "[figure]", doc_id=doc_id, evidence_id=ev_id))

        # Render HTML
        tmpl = self.env.from_string(_TEMPLATE)
        html = tmpl.render(
            title=title,
            lang=lang,
            audience=audience,
            run_id=run_id,
            query=query,
            visuals=visuals,
            highlights=highlights,
            refs=ci.get("refs", []),
        )

        out_html = REPORTS_DIR / f"{run_id}.html"
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(html, encoding="utf-8")
        return out_html


# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor report layout (HTML)")
    ap.add_argument("--run-id", required=True, help="Rank run id (base name of citations/summary json)")
    ap.add_argument("--title", default="", help="Report title")
    ap.add_argument("--lang", default="en", help="en|zh|ja")
    ap.add_argument("--audience", default="engineer", help="engineer|scientist")
    ap.add_argument("--max-visuals", type=int, default=18, help="Cap visuals to keep image ratio and page size reasonable")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    spec = LayoutSpec(
        title=args.title or f"CiteVizor Report",
        lang=args.lang,
        audience=args.audience,
        max_visuals=args.max_visuals,
    )
    rl = ReportLayout(spec)
    out = rl.build(args.run_id)
    print(f"[OK] HTML report -> {out}")
    print(f"Assets -> {out.parent / (args.run_id + '_assets')}")
