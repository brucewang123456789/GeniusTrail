# parse/figure_extract.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import os
import re
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional backends
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# Single source of truth for paths and schemas
from schemas import (
    Evidence,
    EvidencePack,
    EvidenceType,
    FetchedDoc,
    EVIDENCE_DIR,
    DOCS_DIR,
    ensure_dirs,
    evidence_dir_for,   # NEW: path-contract helper
)

ENV_FILE = "citevizor.env"
PACK_FILENAME = "pack.json"
_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?)\s+([0-9IVXLC\-.():]+)", re.IGNORECASE)


# ------------------------- env helpers -------------------------

def _load_env(project_root: Path) -> Dict[str, str]:
    """Load key=value pairs from citevizor.env with process env override."""
    env_path = project_root / ENV_FILE
    out: Dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("PDFFIGURES2_JAR", "PDFFIGURES2_TIMEOUT_S", "FITZ_DPI", "MAX_FIGURES_PER_PDF"):
        if os.getenv(k):
            out[k] = os.getenv(k)  # type: ignore
    return out


# ------------------------- path helpers (contract) -------------------------

def _figures_dir(doc_id: str) -> Path:
    """
    Contract: figure images live directly under storage/evidence/<doc_id>/.
    We DO NOT create a nested 'figures' subdir to keep paths flat and predictable.
    """
    d = evidence_dir_for(doc_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pack_path(doc_id: str) -> Path:
    """Return storage/evidence/<doc_id>/pack.json."""
    d = evidence_dir_for(doc_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / PACK_FILENAME


def _stable_fig_id(doc_id: str, page: int, idx: int, bbox: Optional[Tuple[float, float, float, float]], caption: str) -> str:
    """Stable ID that is robust across re-runs with minor caption differences."""
    sig = f"{doc_id}|{page}|{idx}|{(bbox or ())}|{caption[:96]}"
    h = sha256(sig.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}_fg_{page:03d}_{idx:03d}_{h}"


def _final_img_name(doc_id: str, page: int, idx: int, suffix: str = ".png") -> str:
    """Standardized filename for figures: <doc_id>_fg_<page>_<idx>.png"""
    return f"{doc_id}_fg_{page:03d}_{idx:03d}{suffix.lower()}"


def _normalize_caption(txt: str) -> str:
    t = (txt or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _parse_figure_no(caption: str) -> Optional[str]:
    m = _CAPTION_RE.match(caption or "")
    if not m:
        return None
    return m.group(2).strip()


def _prune_short(caption: str) -> bool:
    t = (caption or "").strip()
    return len(t) < 8


def _rel_str(p: Optional[Path], base: Path) -> Optional[str]:
    if p is None:
        return None
    try:
        return str(p.relative_to(base))
    except Exception:
        return str(p)


def _jsonify(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_jsonify(v) for v in o]
    return o


def _sanitize_evidence_list(items, kind: str, doc_id: str, base_dir: Path) -> List[Dict[str, Any]]:
    """
    Sanitize evidence dicts before pydantic validation:
    - Ensure doc_id and type
    - Coerce figure_no/table_no to str
    - Coerce page to int (invalid -> None)
    - Coerce image_path to relative string
    - Support legacy 'img_path'
    """
    out: List[Dict[str, Any]] = []
    if not items:
        return out
    for e in items:
        if not isinstance(e, dict):
            continue
        d = dict(e)

        # Path normalization
        if "img_path" in d and "image_path" not in d:
            d["image_path"] = d.pop("img_path")
        if isinstance(d.get("image_path"), Path):
            d["image_path"] = _rel_str(d["image_path"], base_dir)
        if d.get("image_path") is not None:
            d["image_path"] = str(d["image_path"])

        # Page -> int
        if "page" in d and d["page"] not in (None, ""):
            try:
                d["page"] = int(d["page"])
            except Exception:
                d["page"] = None

        # No./IDs as str
        if "figure_no" in d and d["figure_no"] is not None:
            d["figure_no"] = str(d["figure_no"])
        if "table_no" in d and d["table_no"] is not None:
            d["table_no"] = str(d["table_no"])

        # Type & doc_id
        if "type" not in d or not d["type"]:
            d["type"] = kind
        if "doc_id" not in d or not d["doc_id"]:
            d["doc_id"] = doc_id

        out.append(d)
    return out


# ------------------------- config -------------------------

@dataclass
class FigureExtractConfig:
    project_root: Path
    pdffigures2_jar: Optional[str] = None
    pdffigures2_timeout_s: int = 90
    fitz_dpi: int = 200
    max_figures_per_pdf: int = 128

    @classmethod
    def from_env(cls, project_root: Path) -> "FigureExtractConfig":
        env = _load_env(project_root)
        return cls(
            project_root=project_root,
            pdffigures2_jar=env.get("PDFFIGURES2_JAR"),
            pdffigures2_timeout_s=int(env.get("PDFFIGURES2_TIMEOUT_S", "90")),
            fitz_dpi=int(env.get("FITZ_DPI", "200")),
            max_figures_per_pdf=int(env.get("MAX_FIGURES_PER_PDF", "128")),
        )


# ------------------------- extractor -------------------------

class FigureExtractor:
    def __init__(self, cfg: FigureExtractConfig):
        self.cfg = cfg
        ensure_dirs()

    def extract(self, fetched: FetchedDoc, persist: bool = True) -> EvidencePack:
        """
        Extract figures/tables for a single fetched document.
        - Priority: use fetched.paths.pdf_path (programmatic contract)
        - Output directory: storage/evidence/<doc_id>/
        - Image filenames: <doc_id>_fg_<page>_<idx>.png
        - Pack file: storage/evidence/<doc_id>/pack.json
        """
        pdf_path = fetched.paths.pdf_path
        if not pdf_path or not pdf_path.exists():
            raise FileNotFoundError("PDF not found; run downloader first.")
        doc_id = fetched.doc_id
        fig_dir = _figures_dir(doc_id)
        base_dir = fig_dir  # relative base for image_path

        figures: List[Evidence] = []
        tables: List[Evidence] = []

        used_pdffig = False
        if self.cfg.pdffigures2_jar and Path(self.cfg.pdffigures2_jar).exists():
            try:
                pf_figs, pf_tabs = self._extract_with_pdffigures2(pdf_path, doc_id, fig_dir, base_dir)
                figures.extend(pf_figs)
                tables.extend(pf_tabs)
                used_pdffig = True
            except Exception:
                figures, tables = [], []

        if not figures and not used_pdffig and fitz is not None:
            try:
                figures = self._extract_with_fitz(pdf_path, doc_id, fig_dir, base_dir)
            except Exception:
                figures = []

        if not figures and pdfplumber is not None:
            try:
                figures = self._fallback_page_snapshots(pdf_path, doc_id, fig_dir, base_dir)
            except Exception:
                figures = []

        # Merge with existing pack.json (idempotent)
        pack_dict = self._read_or_init_pack(doc_id)
        exist_fig_ids = {e["id"] if isinstance(e, dict) else e.id for e in pack_dict.get("figures", [])}
        exist_tab_ids = {e["id"] if isinstance(e, dict) else e.id for e in pack_dict.get("tables", [])}

        for ev in figures:
            if ev.id not in exist_fig_ids:
                d = ev.__dict__.copy()
                if isinstance(d.get("image_path"), Path):
                    d["image_path"] = _rel_str(d["image_path"], base_dir)
                pack_dict.setdefault("figures", []).append(d)

        for ev in tables:
            if ev.id not in exist_tab_ids:
                d = ev.__dict__.copy()
                if isinstance(d.get("image_path"), Path):
                    d["image_path"] = _rel_str(d["image_path"], base_dir)
                pack_dict.setdefault("tables", []).append(d)

        # Sanitize three evidence buckets before Pydantic validation
        pack_dict["figures"]    = _sanitize_evidence_list(pack_dict.get("figures",    []), "figure",    doc_id, base_dir)
        pack_dict["tables"]     = _sanitize_evidence_list(pack_dict.get("tables",     []), "table",     doc_id, base_dir)
        pack_dict["paragraphs"] = _sanitize_evidence_list(pack_dict.get("paragraphs", []), "paragraph", doc_id, base_dir)

        if persist:
            self._write_pack(doc_id, pack_dict)

        figs = [Evidence(**e) for e in pack_dict.get("figures", [])]
        tabs = [Evidence(**e) for e in pack_dict.get("tables",  [])]
        paras = [Evidence(**e) for e in pack_dict.get("paragraphs", [])]

        return EvidencePack(doc_id=doc_id, paragraphs=paras, tables=tabs, figures=figs)

    # ---------- pdffigures2 ----------
    def _extract_with_pdffigures2(self, pdf_path: Path, doc_id: str, fig_dir: Path, base_dir: Path) -> Tuple[List[Evidence], List[Evidence]]:
        """
        Use pdffigures2 if available. We write intermediate files to a temp subdir,
        then move/rename final images to the contract path with standardized names.
        """
        out_dir = fig_dir / "_pdffigures2"
        out_dir.mkdir(parents=True, exist_ok=True)
        jar = self.cfg.pdffigures2_jar
        assert jar and Path(jar).exists(), "PDFFIGURES2_JAR not found"

        json_prefix = str(out_dir / "figdata-")
        img_prefix = str(out_dir / "fig-")
        cmd = [
            "java", "-Xms512m", "-Xmx2048m", "-jar", jar,
            "-q", "-t", "4", "-c",
            "-d", json_prefix,
            "-m", img_prefix,
            "-f", "png",
            str(pdf_path),
        ]
        subprocess.run(cmd, cwd=str(out_dir), check=True, timeout=self.cfg.pdffigures2_timeout_s)

        stem = pdf_path.stem
        json_candidates = [
            out_dir / f"figdata-{stem}.pdf.json",
            out_dir / f"figdata-{stem}.json",
        ]
        data: Optional[Dict[str, Any]] = None
        for p in json_candidates:
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    break
                except Exception:
                    continue
        if not data:
            return [], []

        # Collect produced images (loose matching for robustness)
        img_files = sorted(out_dir.glob(f"fig-{stem}.pdf-*.png")) + sorted(out_dir.glob(f"fig-{stem}-*.png")) + sorted(out_dir.glob("fig-*.png"))

        def _move_image(src: Optional[Path], page: int, idx: int) -> Optional[Path]:
            """Move temp image to contract path with standardized name."""
            if not src or not src.exists():
                return None
            final = fig_dir / _final_img_name(doc_id, page or 0, idx, src.suffix or ".png")
            try:
                src.replace(final)
            except Exception:
                final.write_bytes(src.read_bytes())
                try:
                    src.unlink(missing_ok=True)
                except Exception:
                    pass
            return final

        figures: List[Evidence] = []
        tables: List[Evidence] = []

        figs = data.get("figures", [])
        for i, f in enumerate(figs, 1):
            try:
                page = int(f.get("page", 0)) + 1
            except Exception:
                page = 0
            cap = _normalize_caption(f.get("caption") or "")
            img = img_files[i - 1] if i - 1 < len(img_files) else None
            final_img = _move_image(img, page, i)
            evid = Evidence(
                id=_stable_fig_id(doc_id, page or 0, i, None, cap),
                type=EvidenceType.figure,
                doc_id=doc_id,
                page=page or None,
                figure_no=_parse_figure_no(cap) if cap else None,
                caption=cap or None,
                image_path=_rel_str(final_img, base_dir),
            )
            figures.append(evid)

        tabs = data.get("tables", data.get("table", []))
        table_imgs = [p for p in img_files if "table" in p.name.lower()]
        for i, t in enumerate(tabs, 1):
            try:
                page = int(t.get("page", 0)) + 1
            except Exception:
                page = 0
            cap = _normalize_caption(t.get("caption") or "")
            img = table_imgs[i - 1] if i - 1 < len(table_imgs) else None
            final_img = _move_image(img, page, i)
            evid = Evidence(
                id=_stable_fig_id(doc_id, page or 0, i, None, cap),
                type=EvidenceType.table,
                doc_id=doc_id,
                page=page or None,
                table_no=str(i),
                caption=cap or None,
                image_path=_rel_str(final_img, base_dir),
            )
            tables.append(evid)

        # Cleanup intermediate JSON files quietly
        for f in out_dir.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass

        return figures, tables

    # ---------- PyMuPDF fallback ----------
    def _extract_with_fitz(self, pdf_path: Path, doc_id: str, fig_dir: Path, base_dir: Path) -> List[Evidence]:
        """Heuristic image extraction using PyMuPDF when pdffigures2 is unavailable."""
        if fitz is None:
            return []
        doc = fitz.open(str(pdf_path))
        evidences: List[Evidence] = []
        per_pdf = 0

        for pidx in range(len(doc)):
            if per_pdf >= self.cfg.max_figures_per_pdf:
                break
            page = doc[pidx]
            try:
                blocks = page.get_text("blocks")
            except Exception:
                blocks = []
            try:
                raw = page.get_text("rawdict")
                iblocks = [b for b in raw.get("blocks", []) if b.get("type") == 1]
            except Exception:
                iblocks = []

            if not iblocks:
                # Fallback: extract embedded images
                xrefs = page.get_images(full=True)
                for jdx, (xref, *_rest) in enumerate(xrefs, 1):
                    if per_pdf >= self.cfg.max_figures_per_pdf:
                        break
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n - pix.alpha >= 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_bytes = pix.tobytes("png")
                        final_img = fig_dir / _final_img_name(doc_id, pidx + 1, jdx, ".png")
                        final_img.write_bytes(img_bytes)
                        caption = self._nearest_caption_from_blocks(blocks, None)
                        fig_no = _parse_figure_no(caption) if caption else None
                        evidences.append(Evidence(
                            id=_stable_fig_id(doc_id, pidx + 1, jdx, None, caption or ""),
                            type=EvidenceType.figure,
                            doc_id=doc_id,
                            page=pidx + 1,
                            figure_no=fig_no,
                            caption=caption or None,
                            image_path=_rel_str(final_img, base_dir),
                        ))
                        per_pdf += 1
                    except Exception:
                        continue
            else:
                # Clip-render image blocks as figures
                for jdx, ib in enumerate(iblocks, 1):
                    if per_pdf >= self.cfg.max_figures_per_pdf:
                        break
                    bbox = tuple(ib.get("bbox", [])) if ib.get("bbox") else None
                    try:
                        mat = fitz.Matrix(self.cfg.fitz_dpi / 72.0, self.cfg.fitz_dpi / 72.0)
                        rect = fitz.Rect(*bbox) if bbox else page.rect
                        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
                        img_bytes = pix.tobytes("png")
                        final_img = fig_dir / _final_img_name(doc_id, pidx + 1, jdx, ".png")
                        final_img.write_bytes(img_bytes)
                    except Exception:
                        continue
                    caption = self._nearest_caption_from_blocks(blocks, bbox)
                    if caption and _prune_short(caption):
                        caption = None
                    fig_no = _parse_figure_no(caption) if caption else None
                    evidences.append(Evidence(
                        id=_stable_fig_id(doc_id, pidx + 1, jdx, bbox if bbox else None, caption or ""),
                        type=EvidenceType.figure,
                        doc_id=doc_id,
                        page=pidx + 1,
                        figure_no=fig_no,
                        caption=caption or None,
                        image_path=_rel_str(final_img, base_dir),
                        bbox=tuple(bbox) if bbox else None,  # type: ignore
                    ))
                    per_pdf += 1

        doc.close()
        return evidences

    def _nearest_caption_from_blocks(self, blocks: List[Tuple], bbox: Optional[Tuple[float, float, float, float]]) -> Optional[str]:
        """Pick the nearest block that looks like a caption."""
        if not blocks:
            return None
        candidates: List[Tuple[float, str]] = []
        bx0, by0, bx1, by1 = (bbox if bbox else (0, 0, 0, -1))
        for blk in blocks:
            if not isinstance(blk, (list, tuple)) or len(blk) < 5:
                continue
            x0, y0, x1, y1, text = blk[:5]
            if not isinstance(text, str) or not text.strip():
                continue
            t = text.strip().replace("\n", " ")
            if not re.search(r"\b(fig\.?|figure)\s+\w+", t, re.IGNORECASE):
                if bbox is None:
                    continue
            if bbox is not None:
                if y0 < by1:
                    dist = abs(by0 - y1) + 1000  # avoid captions above the box unless very close
                else:
                    dist = y0 - by1
            else:
                dist = 9999
            candidates.append((float(dist), t))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return _normalize_caption(candidates[0][1])

    def _fallback_page_snapshots(self, pdf_path: Path, doc_id: str, fig_dir: Path, base_dir: Path) -> List[Evidence]:
        """
        Last-resort page snapshots to ensure at least some visual content exists.
        Useful for scanned PDFs or when other extractors fail.
        """
        if pdfplumber is None:
            return []
        evidences: List[Evidence] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pidx, page in enumerate(pdf.pages, 1):
                try:
                    img = page.to_image(resolution=200)
                    final_img = fig_dir / _final_img_name(doc_id, pidx, 0, ".png")
                    img.save(str(final_img), format="PNG")
                    evidences.append(Evidence(
                        id=_stable_fig_id(doc_id, pidx, 0, None, f"Page {pidx} snapshot"),
                        type=EvidenceType.figure,
                        doc_id=doc_id,
                        page=pidx,
                        caption=f"Page {pidx} snapshot",
                        image_path=_rel_str(final_img, base_dir),
                    ))
                except Exception:
                    continue
        return evidences

    # ------------------------- pack I/O -------------------------

    def _read_or_init_pack(self, doc_id: str) -> Dict[str, Any]:
        p = _pack_path(doc_id)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"doc_id": doc_id, "paragraphs": [], "tables": [], "figures": []}

    def _write_pack(self, doc_id: str, pack: Dict[str, Any]) -> Path:
        p = _pack_path(doc_id)
        p.write_text(json.dumps(_jsonify(pack), ensure_ascii=False, indent=2), encoding="utf-8")
        return p


# ------------------------- CLI entry -------------------------

if __name__ == "__main__":
    ensure_dirs()
    project_root = Path(__file__).resolve().parents[1]
    cfg = FigureExtractConfig.from_env(project_root)
    extractor = FigureExtractor(cfg)

    if len(sys.argv) < 2:
        print("Usage: python parse/figure_extract.py <doc_id>")
        sys.exit(1)

    doc_id = sys.argv[1].strip()
    pdf_path = DOCS_DIR / doc_id / f"{doc_id}.pdf"
    if not pdf_path.exists():
        alt = DOCS_DIR / f"{doc_id}.pdf"  # legacy fallback
        if alt.exists():
            pdf_path = alt
        else:
            print(f"[ERROR] PDF not found: {DOCS_DIR / doc_id / f'{doc_id}.pdf'} or {alt}")
            sys.exit(2)

    meta_path = pdf_path.parent / f"{doc_id}.meta.json"
    title = doc_id
    url = "https://example.org"
    doi = None
    year = None
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            title = m.get("title") or title
            url = m.get("url") or url
            doi = m.get("doi")
            year = m.get("year")
        except Exception:
            pass

    from schemas import Candidate, DocumentPaths
    cand = Candidate(title=title, url=url, year=year, source="scholar", score=1.0, doi=doi)  # type: ignore
    paths = DocumentPaths(base_dir=pdf_path.parent, pdf_path=pdf_path, meta_json=meta_path)
    fetched = FetchedDoc(doc_id=doc_id, candidate=cand, paths=paths)

    pack = extractor.extract(fetched, persist=True)
    out_path = _pack_path(doc_id)
    print(f"[OK] Figures extracted -> {out_path}")
    print(f"figures={len(pack.figures)} tables={len(pack.tables)} paragraphs={len(pack.paragraphs)}")
