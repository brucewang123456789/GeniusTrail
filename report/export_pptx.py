# C:\CiteVizor\report\export_pptx.py
# -*- coding: utf-8 -*-
"""
CiteVizor - PPTX exporter

Builds a PowerPoint deck from a ranking run:
  1) Title slide (query + run_id)
  2) Key highlights (1–3 slides, capped)
  3) Visual slides (charts first, then web images, then figures), each with a bottom caption bar:
       [ref label] refined/pack caption  •  doc_id  •  evidence_id
  4) References slides (auto-paginated)

Design goals:
  - Image-first; single visual per slide with consistent bottom caption bar.
  - Deterministic and offline; no LLM calls, no new env keys.
  - Robust to missing files (skips gracefully, prints counts).

Dependencies: python-pptx, Pillow (PIL)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

from PIL import Image

from schemas import EVIDENCE_DIR, RENDERS_DIR, REPORTS_DIR

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
    return _load_json(EVIDENCE_DIR / doc_id / "pack.json")

def _charts_for_doc(doc_id: str) -> List[Path]:
    d = RENDERS_DIR / doc_id / "charts"
    if not d.exists():
        return []
    return sorted([p for p in d.glob("*.png") if p.is_file()])

# ----------------------------- Web image helpers ------------------------------

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
    Exclude extracted figures and generated charts to avoid duplication.
    """
    root = EVIDENCE_DIR / doc_id
    if not root.exists():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            parts = {x.name.lower() for x in p.parents if x != root}
            if "figures" in parts or "charts" in parts:
                continue
            out.append(p)
    out.sort(key=lambda x: (len(x.relative_to(root).parts), x.name.lower()))
    return out

# ----------------------------- Collect visuals --------------------------------

@dataclass
class Visual:
    kind: str          # "chart" | "web_image" | "figure"
    path: Path
    label: str
    caption: str
    doc_id: str
    evidence_id: Optional[str] = None
    ref_url: Optional[str] = None

_LABEL_RE = re.compile(r"\[(\d+)[^\]]*\]")

def _ref_url_from_label(label: str, refs: List[Dict[str, Any]]) -> Optional[str]:
    m = _LABEL_RE.search(label or "")
    if not m:
        return None
    try:
        rid = int(m.group(1))
    except Exception:
        return None
    for r in refs:
        if int(r.get("ref_id", -1)) == rid:
            if r.get("doi"):
                return f"https://doi.org/{r['doi']}"
            return r.get("url")
    return None

def _collect_visuals(ci: Dict[str, Any], su: Dict[str, Any], max_visuals: int = 40) -> List[Visual]:
    """
    Collect visuals in preferred order: charts → web images → figures.
    Web images act as a bypass when PDF parsing fails; files are discovered
    under evidence/<doc_id> excluding figures/ and charts/.
    """
    out: List[Visual] = []
    refs = ci.get("refs", [])
    figcaps = su.get("figure_captions") if isinstance(su.get("figure_captions"), dict) else {}

    doc_ids = list({it.get("doc_id") for it in ci.get("items", []) if it.get("doc_id")})

    # 1) Charts (from renders)
    for doc_id in doc_ids:
        for png in _charts_for_doc(doc_id):
            # try to find a label from a table evidence of this doc
            label = ""
            for it in ci.get("items", []):
                if it.get("doc_id") == doc_id and it.get("type") == "table":
                    label = it.get("label") or ""
                    break
            cap = f"Auto-generated chart from table. Source document: {doc_id}."
            out.append(Visual(
                kind="chart",
                path=png,
                label=label or "[table]",
                caption=cap,
                doc_id=doc_id,
                evidence_id=None,
                ref_url=_ref_url_from_label(label, refs),
            ))
            if len(out) >= max_visuals:
                return out

    # 2) Web images (bypass for image-sourced or discovered images)
    for doc_id in doc_ids:
        if len(out) >= max_visuals:
            break

        # Hints from citations for this doc
        items_this_doc = [it for it in ci.get("items", []) if it.get("doc_id") == doc_id]
        has_image_source = any((it.get("source") == "image") or (it.get("type") in ("image", "web_image")) for it in items_this_doc)
        imgs = _list_web_images_for_doc(doc_id)
        if not imgs:
            continue

        # Derive caption hint and label from citations if present
        caption_hint = ""
        label_hint = ""
        if has_image_source:
            for it in items_this_doc:
                if (it.get("source") == "image") or (it.get("type") in ("image", "web_image")):
                    caption_hint = (it.get("snippet") or it.get("title") or "").strip()
                    label_hint = (it.get("label") or "[image]").strip() or "[image]"
                    break

        # Fallback caption using host if no snippet/title
        if not caption_hint:
            host = ""
            for it in items_this_doc:
                if it.get("url"):
                    host = _host_of(it.get("url"))
                    if host:
                        break
            caption_hint = f"Image from {host}" if host else "Image from web source"
        label_hint = label_hint or "[image]"

        for img in imgs:
            out.append(Visual(
                kind="web_image",
                path=img,
                label=label_hint,
                caption=caption_hint,
                doc_id=doc_id,
                evidence_id=None,
                ref_url=None,  # link remains on references page; optional to add host link
            ))
            if len(out) >= max_visuals:
                return out

    # 3) Figures (from packs)
    for it in ci.get("items", []):
        if len(out) >= max_visuals:
            break
        if it.get("type") != "figure":
            continue
        doc_id = it.get("doc_id")
        ev_id = it.get("evidence_id")
        pack = _load_pack(doc_id)
        fig = None
        for e in pack.get("figures", []):
            if e.get("id") == ev_id:
                fig = e
                break
        if not fig or not fig.get("image_path"):
            continue
        img = Path(fig["image_path"])
        if not img.exists():
            alt = EVIDENCE_DIR / doc_id / "figures" / img.name
            if alt.exists():
                img = alt
            else:
                continue
        label = it.get("label") or ""
        cap = figcaps.get(ev_id) or fig.get("caption") or (it.get("snippet") or "")
        out.append(Visual(
            kind="figure",
            path=img,
            label=label or "[figure]",
            caption=cap,
            doc_id=doc_id,
            evidence_id=ev_id,
            ref_url=_ref_url_from_label(label, refs),
        ))
    return out

# ----------------------------- PPTX building ----------------------------------

ACCENT = RGBColor(15, 118, 110)  # teal-ish accent for labels

def _set_widescreen(prs: Presentation, widescreen: bool) -> None:
    """Set slide size: 16:9 (13.333 x 7.5 in) or classic 4:3 (10 x 7.5 in)."""
    if widescreen:
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
    else:
        prs.slide_width = Inches(10)
        prs.slide_height = Inches(7.5)

def _title_slide(prs: Presentation, title: str, subtitle: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[0])  # Title Slide
    slide.shapes.title.text = title
    sub = slide.placeholders[1]
    sub.text = subtitle
    slide.shapes.title.text_frame.paragraphs[0].font.size = Pt(36)
    sub.text_frame.paragraphs[0].font.size = Pt(16)

def _bullets_slides(prs: Presentation, title: str, bullets: List[str], per_slide: int = 10) -> None:
    for i in range(0, len(bullets), per_slide):
        chunk = bullets[i:i+per_slide]
        s = prs.slides.add_slide(prs.slide_layouts[1])  # Title and Content
        s.shapes.title.text = title if i == 0 else f"{title} (cont.)"
        tf = s.placeholders[1].text_frame
        tf.clear()
        for j, b in enumerate(chunk):
            p = tf.add_paragraph() if j > 0 else tf.paragraphs[0]
            p.level = 0
            p.text = b
            p.font.size = Pt(18)

def _fit_center_image(prs: Presentation, slide, img_path: Path) -> None:
    """Insert an image centered, scaled to fit within margins."""
    with Image.open(img_path) as im:
        w_px, h_px = im.size
    aspect = (w_px / h_px) if h_px else 1.0

    # Margins
    max_w = Inches(11.8)  # leave room for caption bar
    max_h = Inches(6.0)

    if max_w / max_h > aspect:
        height = max_h
        width = height * aspect
    else:
        width = max_w
        height = width / aspect

    # Center using the Presentation's slide size (avoid slide.part.presentation)
    slide_width = prs.slide_width
    left = (slide_width - width) / 2
    top = Inches(0.6)
    slide.shapes.add_picture(str(img_path), left, top, width=width, height=height)

def _caption_bar(slide, label: str, caption: str, link: Optional[str] = None, meta_tail: str = "") -> None:
    """Draw a bottom caption bar with bold label and smaller caption; add hyperlink if provided."""
    left, top, width, height = Inches(0.3), Inches(6.6), Inches(12.7), Inches(0.9)
    rect = slide.shapes.add_shape(
        autoshape_type_id=1,  # Rectangle
        left=left, top=top, width=width, height=height
    )
    fill = rect.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(250, 250, 250)
    line = rect.line
    line.color.rgb = RGBColor(220, 220, 220)

    tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), height - Inches(0.2))
    tf = tb.text_frame
    tf.clear()

    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = (label or "").strip() + " "
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = ACCENT

    run2 = p.add_run()
    cap_text = (caption or "").strip()
    if meta_tail:
        cap_text = f"{cap_text}  \u2022  {meta_tail}"
    run2.text = cap_text[:400]
    run2.font.size = Pt(14)

    if link:
        try:
            run.hyperlink.address = link
        except Exception:
            pass

def _visual_slide(prs: Presentation, v: Visual) -> None:
    s = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only
    _fit_center_image(prs, s, v.path)
    tail = f"{v.doc_id}" + (f" • {v.evidence_id}" if v.evidence_id else "")
    _caption_bar(s, v.label, v.caption, link=v.ref_url, meta_tail=tail)

def _refs_slides(prs: Presentation, refs: List[Dict[str, Any]], per_slide: int = 12) -> None:
    lines: List[str] = []
    for r in refs:
        s = f"[{r.get('ref_id')}] {r.get('title') or '<untitled>'}"
        if r.get("venue"):
            s += f", {r['venue']}"
        if r.get("year"):
            s += f", {r['year']}"
        if r.get("doi"):
            s += f", doi:{r['doi']}"
        elif r.get("url"):
            s += f", {r['url']}"
        lines.append(s)

    for i in range(0, len(lines), per_slide):
        chunk = lines[i:i+per_slide]
        sld = prs.slides.add_slide(prs.slide_layouts[1])
        sld.shapes.title.text = "References" if i == 0 else "References (cont.)"
        tf = sld.placeholders[1].text_frame
        tf.clear()
        for j, line in enumerate(chunk):
            p = tf.add_paragraph() if j > 0 else tf.paragraphs[0]
            p.level = 0
            p.text = line
            p.font.size = Pt(12)

# ----------------------------- Public API -------------------------------------

def build_pptx(run_id: str, title: Optional[str] = None, lang: str = "en", widescreen: bool = True) -> Path:
    """Build a PPTX deck for the given run_id and return the output path."""
    ci = _load_citations(run_id)
    if not ci:
        raise FileNotFoundError(f"citations json not found for run_id={run_id}")
    su = _load_summary(run_id)

    prs = Presentation()
    _set_widescreen(prs, widescreen)

    report_title = title or f"CiteVizor Report – {ci.get('query','')[:64]}"
    subtitle = f"Image-first engineer brief • run_id={run_id}"
    _title_slide(prs, report_title, subtitle)

    highlights = (su.get("highlights") if su else None) or []
    if highlights:
        _bullets_slides(prs, "Key highlights", highlights[:24], per_slide=10)

    visuals = _collect_visuals(ci, su, max_visuals=40)
    for v in visuals:
        _visual_slide(prs, v)

    _refs_slides(prs, ci.get("refs", []), per_slide=12)

    out = REPORTS_DIR / f"{run_id}.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out

# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor PPTX exporter")
    ap.add_argument("--run-id", required=True, help="Ranking run_id (base name of citations/summary json)")
    ap.add_argument("--title", default="", help="Report title")
    ap.add_argument("--lang", default="en", help="Language tag for highlights (en|zh|ja)")
    ap.add_argument("--classic43", action="store_true", help="Use classic 4:3 aspect ratio (default 16:9)")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    out = build_pptx(args.run_id, title=(args.title or None), lang=args.lang, widescreen=(not args.classic43))
    print(f"[OK] PPTX -> {out}")
