# C:\CiteVizor\parse\ocr_fallback.py
# -*- coding: utf-8 -*-
"""
CiteVizor - OCR fallback for scanned/low-text PDF pages

What this module does
- Detect pages that likely need OCR (low text characters, rich in images/vector).
- Render such pages to bitmaps with PyMuPDF (DPI from env/config; default 200).
- Run Tesseract OCR (pytesseract) to obtain word-level boxes and confidences.
- Aggregate words -> paragraphs using Tesseract block/paragraph indices.
- Persist per-page OCR payload to storage/evidence/<doc_id>/ocr.json.
- Optionally merge OCR paragraphs back into pack.json (source="ocr").

No LLM calls. No required new env keys.
Optional env keys if you already use them:
  - FITZ_DPI         : int DPI for rendering (e.g., 200)
  - TESSERACT_EXE    : full path to tesseract executable (Windows convenience)
  - OCR_LANGS        : comma-separated languages (e.g., "eng,chi_sim")

Dependencies: PyMuPDF (fitz), pillow, pytesseract.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from schemas import DOCS_DIR, EVIDENCE_DIR  # canonical project dirs

# Optional central config (graceful fallback)
try:
    from config import load_config  # type: ignore
except Exception:
    load_config = None  # type: ignore

ENV_FILE = "citevizor.env"


# ----------------------------- env/config helpers ------------------------------

def _project_root() -> Path:
    canonical = Path(r"C:\CiteVizor")
    return canonical if canonical.exists() else Path(__file__).resolve().parents[1]

def _read_env_file(root: Path) -> Dict[str, str]:
    p = root / ENV_FILE
    out: Dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out

def _resolve_settings() -> Tuple[int, Optional[str], List[str]]:
    # FITZ_DPI from config or env; Tesseract exe path optional; OCR langs list
    dpi = 200
    exe = os.environ.get("TESSERACT_EXE")
    langs: List[str] = []
    # prefer config
    if load_config:
        try:
            cfg = load_config()
            dpi = int(getattr(cfg, "parse").fitz_dpi or 200)  # type: ignore
        except Exception:
            pass
    # env file / OS env overrides
    env = _read_env_file(_project_root())
    if "FITZ_DPI" in env:
        try:
            dpi = int(env["FITZ_DPI"])
        except Exception:
            pass
    if not exe:
        exe = env.get("TESSERACT_EXE")
    ocr_langs = os.environ.get("OCR_LANGS") or env.get("OCR_LANGS") or ""
    langs = [x.strip() for x in ocr_langs.split(",") if x.strip()]
    return dpi, exe, langs


# ----------------------------- datamodel ---------------------------------------

@dataclass
class OCRWord:
    text: str
    conf: float
    bbox: Tuple[float, float, float, float]  # normalized x0,y0,x1,y1 (0..1)

@dataclass
class OCRParagraph:
    text: str
    conf_avg: float
    bbox: Tuple[float, float, float, float]  # normalized box covering words
    words: List[OCRWord]

@dataclass
class OCRPage:
    page_index: int           # 0-based
    width: int                # rendered bitmap width
    height: int               # rendered bitmap height
    paragraphs: List[OCRParagraph]

@dataclass
class OCRPayload:
    doc_id: str
    pdf_path: str
    time_s: float
    pages_total: int
    pages_ocr: int
    pages: List[OCRPage]
    params: Dict[str, Any]


# ----------------------------- detection heuristics ----------------------------

def _needs_ocr(page: fitz.Page, min_text_chars: int = 40) -> bool:
    """
    Heuristic: if page has fewer than N text chars OR explicit no text blocks,
    and contains images or vector content, deem it an OCR candidate.
    """
    try:
        txt = page.get_text("text") or ""
    except Exception:
        txt = ""
    if len(txt.strip()) >= min_text_chars:
        return False
    # If no text blocks, likely scanned
    try:
        blocks = page.get_text("blocks") or []
    except Exception:
        blocks = []
    if not blocks:
        return True
    # If few blocks and there are images, also OCR
    try:
        imgs = page.get_images(full=True) or []
    except Exception:
        imgs = []
    return len(imgs) > 0


# ----------------------------- render + OCR ------------------------------------

def _render_page(page: fitz.Page, dpi: int) -> Image.Image:
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def _image_to_paragraphs(img: Image.Image, langs: List[str]) -> Tuple[List[OCRParagraph], int, int]:
    """
    Use pytesseract.image_to_data to get a TSV with word boxes.
    Group by (block_num, par_num) to build paragraphs.
    """
    if langs:
        lang = "+".join(langs)
    else:
        lang = "eng"
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    width = img.width
    height = img.height

    # Build paragraphs map: (block, par) -> list of words
    paras: Dict[Tuple[int, int], List[OCRWord]] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        conf = float(data.get("conf", ["-1"])[i])
        # Tesseract may output -1 for artifacts; skip very low conf words
        if conf < 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        # normalize bbox to 0..1
        x0 = max(0.0, min(1.0, x / float(width)))
        y0 = max(0.0, min(1.0, y / float(height)))
        x1 = max(0.0, min(1.0, (x + w) / float(width)))
        y1 = max(0.0, min(1.0, (y + h) / float(height)))
        key = (int(data.get("block_num", [0])[i]), int(data.get("par_num", [0])[i]))
        paras.setdefault(key, []).append(OCRWord(text=txt, conf=conf, bbox=(x0, y0, x1, y1)))

    out_paras: List[OCRParagraph] = []
    for _, ws in paras.items():
        if not ws:
            continue
        # sort by (y,x) to keep reading order
        ws.sort(key=lambda w: (round(w.bbox[1], 3), round(w.bbox[0], 3)))
        text = " ".join(w.text for w in ws)
        conf_avg = sum(w.conf for w in ws) / max(1, len(ws))
        xs0 = min(w.bbox[0] for w in ws); ys0 = min(w.bbox[1] for w in ws)
        xs1 = max(w.bbox[2] for w in ws); ys1 = max(w.bbox[3] for w in ws)
        out_paras.append(OCRParagraph(text=text, conf_avg=float(conf_avg), bbox=(xs0, ys0, xs1, ys1), words=ws))
    # keep stable order (top-to-bottom)
    out_paras.sort(key=lambda p: (round(p.bbox[1], 3), round(p.bbox[0], 3)))
    return out_paras, width, height


# ----------------------------- persistence -------------------------------------

def _ocr_json_path(doc_id: str) -> Path:
    d = EVIDENCE_DIR / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "ocr.json"

def _pack_path(doc_id: str) -> Path:
    return EVIDENCE_DIR / doc_id / "pack.json"

def _load_json(p: Path) -> Any:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------- public API --------------------------------------

def ocr_document(
    doc_id: str,
    pdf_path: Optional[Path] = None,
    lang_codes: Optional[List[str]] = None,
    dpi: Optional[int] = None,
    pages: Optional[List[int]] = None,
    min_text_chars: int = 40,
    merge_pack: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """
    Run OCR fallback for a document.
    Returns: (ocr_json_path, updated_pack_path_if_merged)
    """
    # Resolve settings
    dpi0, tesseract_exe, default_langs = _resolve_settings()
    if dpi is None:
        dpi = dpi0
    if lang_codes is None or not lang_codes:
        lang_codes = default_langs or ["eng"]
    if tesseract_exe:
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe

    # Resolve PDF path
    if pdf_path is None:
        pdf_path = DOCS_DIR / doc_id / f"{doc_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    t0 = time.time()
    doc = fitz.open(str(pdf_path))
    target_pages: List[int] = list(range(doc.page_count)) if not pages else [p for p in pages if 0 <= p < doc.page_count]

    ocr_pages: List[OCRPage] = []
    pages_ocr = 0
    for i in target_pages:
        pg = doc.load_page(i)
        # Decide if OCR is needed (always true if pages explicitly specified)
        need = _needs_ocr(pg, min_text_chars=min_text_chars) or (pages is not None)
        if not need:
            continue
        img = _render_page(pg, dpi=dpi)
        paras, w, h = _image_to_paragraphs(img, langs=lang_codes)
        ocr_pages.append(OCRPage(page_index=i, width=w, height=h, paragraphs=paras))
        pages_ocr += 1

    payload = OCRPayload(
        doc_id=doc_id,
        pdf_path=str(pdf_path),
        time_s=round(time.time() - t0, 3),
        pages_total=int(doc.page_count),
        pages_ocr=int(pages_ocr),
        pages=ocr_pages,
        params={"dpi": dpi, "langs": lang_codes, "min_text_chars": min_text_chars},
    )
    doc.close()

    # Save ocr.json
    ocr_path = _ocr_json_path(doc_id)
    _save_json(ocr_path, {
        "doc_id": payload.doc_id,
        "pdf_path": payload.pdf_path,
        "time_s": payload.time_s,
        "pages_total": payload.pages_total,
        "pages_ocr": payload.pages_ocr,
        "params": payload.params,
        "pages": [
            {
                "page_index": p.page_index,
                "width": p.width,
                "height": p.height,
                "paragraphs": [
                    {
                        "text": par.text,
                        "conf_avg": par.conf_avg,
                        "bbox": par.bbox,
                        "words": [asdict(w) for w in par.words],
                    } for par in p.paragraphs
                ],
            } for p in payload.pages
        ],
    })

    updated_pack: Optional[Path] = None
    if merge_pack and payload.pages:
        updated_pack = _merge_into_pack(doc_id, payload)

    return ocr_path, updated_pack


def _merge_into_pack(doc_id: str, payload: OCRPayload) -> Path:
    """
    Merge OCR paragraphs into evidence/<doc_id>/pack.json as text evidence.
    """
    pack_path = _pack_path(doc_id)
    pack = _load_json(pack_path)
    if not isinstance(pack, dict):
        pack = {}
    items = pack.get("paragraphs")
    if not isinstance(items, list):
        items = []
    count_before = len(items)

    for p in payload.pages:
        for par in p.paragraphs:
            text = (par.text or "").strip()
            if not text:
                continue
            # Avoid flooding: ignore extremely short "paragraphs"
            if len(text) < 20:
                continue
            items.append({
                "id": f"ocr_p{p.page_index}_{int(par.bbox[0]*1000)}_{int(par.bbox[1]*1000)}",
                "page": p.page_index,
                "snippet": text[:500],
                "text": text,
                "bbox": par.bbox,
                "source": "ocr",
                "confidence": round(par.conf_avg, 2),
            })

    pack["paragraphs"] = items
    _save_json(pack_path, pack)
    return pack_path


# ----------------------------- CLI ---------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor OCR fallback for scanned PDFs")
    ap.add_argument("--doc-id", required=True, help="Document id (folder under storage/docs/<doc_id>/<doc_id>.pdf)")
    ap.add_argument("--pdf", default="", help="Optional explicit PDF path (overrides doc-id resolution)")
    ap.add_argument("--lang", default="", help="Comma-separated Tesseract language codes, e.g., eng,chi_sim,jpn")
    ap.add_argument("--dpi", type=int, default=0, help="Rendering DPI (default from config/env or 200)")
    ap.add_argument("--pages", default="", help='Page selection like "0-3,5,9" (0-based); empty = auto-detect')
    ap.add_argument("--min-text-chars", type=int, default=40, help="Threshold to deem a page needs OCR")
    ap.add_argument("--merge", action="store_true", help="Merge OCR paragraphs into evidence/<doc_id>/pack.json")
    return ap.parse_args()

def _parse_pages(expr: str) -> List[int]:
    expr = (expr or "").strip()
    if not expr:
        return []
    out: List[int] = []
    for part in expr.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                ai, bi = int(a), int(b)
                if ai <= bi:
                    out.extend(list(range(ai, bi + 1)))
            except Exception:
                continue
        else:
            try:
                out.append(int(part))
            except Exception:
                continue
    return sorted(set(out))

if __name__ == "__main__":
    args = _parse_args()
    langs = [x.strip() for x in args.lang.split(",") if x.strip()]
    pp = Path(args.pdf) if args.pdf else None
    dpi = args.dpi if args.dpi > 0 else None
    pages = _parse_pages(args.pages)

    # Resolve optional Tesseract exe from env if present (no hard requirement)
    _, t_exe, _ = _resolve_settings()
    if t_exe:
        pytesseract.pytesseract.tesseract_cmd = t_exe

    ocr_path, merged = ocr_document(
        doc_id=args.doc_id,
        pdf_path=pp,
        lang_codes=langs or None,
        dpi=dpi,
        pages=pages or None,
        min_text_chars=max(0, args.min_text_chars),
        merge_pack=args.merge,
    )
    print(f"[OK] OCR -> {ocr_path}")
    if merged:
        print(f"[OK] pack.json updated -> {merged}")
