# C:\CiteVizor\parse\pdf_structure.py
# -*- coding: utf-8 -*-
"""
CiteVizor - PDF structure parsing
- Primary: optional GROBID full-text structuring when GROBID_URL is configured.
- Fallback order: pdfplumber → PyMuPDF(fitz) → empty-pack (no exception).
- Produces EvidencePack aligned with schemas.py and persists to storage/evidence/<doc_id>/pack.json.
- Idempotent and defensive: safe on malformed PDFs; partial results still useful downstream.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

# External libs (optional/runtime-checked)
try:
    import pdfplumber  # Fallback extractor #1
except Exception:
    pdfplumber = None  # type: ignore

try:
    import fitz  # PyMuPDF – Fallback extractor #2 (text)
except Exception:
    fitz = None  # type: ignore

try:
    import requests  # For GROBID HTTP client
except Exception:
    requests = None  # type: ignore

# Project schemas / paths
from schemas import (
    Evidence,
    EvidencePack,
    EvidenceType,
    FetchedDoc,
    EVIDENCE_DIR,
    DOCS_DIR,
    ensure_dirs,
)

ENV_FILE_NAME = "citevizor.env"
PACK_FILENAME = "pack.json"

# -------------------------- ENV loader (minimal) ------------------------------

def _load_env(project_root: Path) -> dict:
    env_path = project_root / ENV_FILE_NAME
    out = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    # OS env has priority to override file
    for k in ("GROBID_URL", "GROBID_TIMEOUT_S"):
        if os.getenv(k):
            out[k] = os.getenv(k)
    return out


# ------------------------------ Utilities -------------------------------------

_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?|table)\s+([\w\-.:()]+)?", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*((\d+(\.\d+){0,3})|[IVXLC]+\.)\s+[A-Z].{3,}")
_REFERENCES_RE = re.compile(r"^\s*(references|bibliography)\b", re.IGNORECASE)

def _normalize_text(txt: str) -> str:
    """Basic normalization: de-hyphenation across line breaks, collapse spaces, normalize newlines."""
    if not txt:
        return ""
    txt = re.sub(r"-\s*\n\s*", "", txt)  # remove hyphenation at wraps
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def _split_paragraphs(page_text: str) -> List[str]:
    """Heuristic paragraph split: prefer blank-line boundaries; fallback to sentence blocks."""
    if not page_text:
        return []
    text = _normalize_text(page_text)
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(parts) >= 2:
        return parts
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    buf: List[str] = []
    cur: List[str] = []
    for line in lines:
        cur.append(line)
        if _SECTION_RE.match(line) or len(" ".join(cur)) > 500:
            buf.append(" ".join(cur).strip())
            cur = []
    if cur:
        buf.append(" ".join(cur).strip())
    return [p for p in buf if len(p) > 20]

def _is_caption(line: str) -> Optional[Tuple[str, Optional[str]]]:
    """Return ('figure'|'table', number?) if the line looks like a caption; else None."""
    m = _CAPTION_RE.match(line or "")
    if not m:
        return None
    kind = m.group(1).lower()
    num = m.group(2)
    if "fig" in kind:
        return ("figure", num)
    if "table" in kind:
        return ("table", num)
    return None

def _ev_id(doc_id: str, kind: str, page: int, idx: int, payload: str = "") -> str:
    """Stable evidence id using doc_id + kind + page + idx + short hash."""
    h = sha256(f"{doc_id}|{kind}|{page}|{idx}|{payload[:64]}".encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}_{kind[:2]}_{page:03d}_{idx:03d}_{h}"

def _pack_path(doc_id: str) -> Path:
    d = EVIDENCE_DIR / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d / PACK_FILENAME


# ------------------------------ Parser core -----------------------------------

@dataclass
class PDFStructureParser:
    project_root: Path
    grobid_url: Optional[str] = None
    grobid_timeout_s: int = 60

    @classmethod
    def from_env(cls, project_root: Path) -> "PDFStructureParser":
        env = _load_env(project_root)
        url = env.get("GROBID_URL")  # e.g., http://localhost:8070
        timeout = int(env.get("GROBID_TIMEOUT_S", "60"))
        return cls(project_root=project_root, grobid_url=url, grobid_timeout_s=timeout)

    # -------------------- Public API --------------------

    def parse_pdf(self, fetched: FetchedDoc, persist: bool = True) -> EvidencePack:
        """
        Parse a fetched PDF into an EvidencePack.
        Priority: GROBID if configured and available; else pdfplumber; else fitz; else empty-pack.
        """
        ensure_dirs()
        pdf_path = fetched.paths.pdf_path
        if not pdf_path or not pdf_path.exists():
            raise FileNotFoundError("PDF file not found; ensure downloader fetched a PDF.")

        # 1) GROBID (if configured)
        if self.grobid_url and requests is not None:
            try:
                pack = self._parse_with_grobid(pdf_path, fetched)
                if persist:
                    self._persist_pack(fetched.doc_id, pack)
                return pack
            except Exception:
                # Soft-fallback to local extractors
                pass

        # 2) Local fallbacks: pdfplumber → fitz → empty
        if pdfplumber is not None:
            pack = self._parse_with_pdfplumber(pdf_path, fetched)
            if persist:
                self._persist_pack(fetched.doc_id, pack)
            return pack

        if fitz is not None:
            pack = self._parse_with_fitz_text(pdf_path, fetched)
            if persist:
                self._persist_pack(fetched.doc_id, pack)
            return pack

        # 3) Last resort: return empty pack (do not raise) to avoid parse.error
        empty = EvidencePack(doc_id=fetched.doc_id, paragraphs=[], tables=[], figures=[])
        if persist:
            self._persist_pack(fetched.doc_id, empty)
        return empty

    # -------------------- GROBID path --------------------

    def _parse_with_grobid(self, pdf_path: Path, fetched: FetchedDoc) -> EvidencePack:
        """Use GROBID's /api/processFulltextDocument to obtain TEI XML; extract paragraphs and captions."""
        assert self.grobid_url, "GROBID_URL required"
        assert requests is not None, "requests lib required"

        url = self.grobid_url.rstrip("/") + "/api/processFulltextDocument"
        files = {"input": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")}
        data = {
            "consolidateHeader": "1",
            "consolidateCitations": "0",
            "includeRawCitations": "0",
            "teiCoordinates": "p,figure,figDesc,table",
        }
        resp = requests.post(url, files=files, data=data, timeout=self.grobid_timeout_s)
        resp.raise_for_status()
        tei_xml = resp.text

        root = ET.fromstring(tei_xml)
        ns = {"tei": "http://www.tei-c.org/ns/1.0"}

        paragraphs: List[Evidence] = []
        captions: List[Evidence] = []

        # Extract paragraphs with coordinates if available
        for i, p in enumerate(root.findall(".//tei:body//tei:p", ns), 1):
            text = _normalize_text("".join(p.itertext()))
            if not text or len(text) < 30:
                continue
            page = None
            facs = p.attrib.get("{http://www.tei-c.org/ns/1.0}facs") or p.attrib.get("facs")
            if facs and "#page" in facs:
                try:
                    page = int(re.sub(r"[^\d]", "", facs))
                except Exception:
                    page = None
            ev = Evidence(
                id=_ev_id(fetched.doc_id, "paragraph", page or 0, i, text),
                type=EvidenceType.paragraph,
                doc_id=fetched.doc_id,
                page=page,
                text=text,
                doi=fetched.candidate.doi,
                source_url=fetched.candidate.url,  # type: ignore
            )
            paragraphs.append(ev)

        # Extract figure and table captions
        for j, fig in enumerate(root.findall(".//tei:figure", ns), 1):
            desc = fig.find(".//tei:figDesc", ns)
            cap = _normalize_text("".join(desc.itertext())) if desc is not None else ""
            if not cap:
                continue
            num = (fig.attrib.get("n") or "").strip() or None
            ev = Evidence(
                id=_ev_id(fetched.doc_id, "caption", 0, j, cap),
                type=EvidenceType.caption,
                doc_id=fetched.doc_id,
                page=None,
                figure_no=num,
                caption=cap,
                doi=fetched.candidate.doi,
                source_url=fetched.candidate.url,  # type: ignore
            )
            captions.append(ev)

        for k, tb in enumerate(root.findall(".//tei:table", ns), 1):
            head = tb.find(".//tei:head", ns)
            cap = _normalize_text("".join(head.itertext())) if head is not None else ""
            if not cap:
                continue
            num = (tb.attrib.get("n") or "").strip() or None
            ev = Evidence(
                id=_ev_id(fetched.doc_id, "caption", 0, 1000 + k, cap),
                type=EvidenceType.caption,
                doc_id=fetched.doc_id,
                page=None,
                table_no=num,
                caption=cap,
                doi=fetched.candidate.doi,
                source_url=fetched.candidate.url,  # type: ignore
            )
            captions.append(ev)

        # Keep existing behavior (do not alter semantics here)
        return EvidencePack(
            doc_id=fetched.doc_id,
            paragraphs=paragraphs,
            tables=[],
            figures=captions
        )

    # -------------------- pdfplumber fallback --------------------

    def _parse_with_pdfplumber(self, pdf_path: Path, fetched: FetchedDoc) -> EvidencePack:
        """Page-by-page text extraction; paragraph segmentation; caption spotting."""
        assert pdfplumber is not None, "pdfplumber not available"

        paragraphs: List[Evidence] = []
        captions: List[Evidence] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for pidx, page in enumerate(pdf.pages, 1):
                try:
                    raw = page.extract_text(x_tolerance=2, y_tolerance=2, layout=True) or ""
                except Exception:
                    raw = page.extract_text() or ""
                if not raw:
                    continue

                is_refs_page = bool(_REFERENCES_RE.match(raw.splitlines()[0] if raw.splitlines() else ""))

                for ln in raw.splitlines():
                    cap = _is_caption(ln)
                    if cap:
                        kind, num = cap
                        ev = Evidence(
                            id=_ev_id(fetched.doc_id, "caption", pidx, len(captions) + 1, ln),
                            type=EvidenceType.caption,
                            doc_id=fetched.doc_id,
                            page=pidx,
                            figure_no=num if kind == "figure" else None,
                            table_no=num if kind == "table" else None,
                            caption=_normalize_text(ln),
                            doi=fetched.candidate.doi,
                            source_url=fetched.candidate.url,  # type: ignore
                        )
                        captions.append(ev)

                paras = _split_paragraphs(raw)
                for i, para in enumerate(paras, 1):
                    ev = Evidence(
                        id=_ev_id(fetched.doc_id, "paragraph", pidx, i, para),
                        type=EvidenceType.paragraph,
                        doc_id=fetched.doc_id,
                        page=pidx,
                        text=para,
                        doi=fetched.candidate.doi,
                        source_url=fetched.candidate.url,  # type: ignore
                    )
                    paragraphs.append(ev)

        return EvidencePack(doc_id=fetched.doc_id, paragraphs=paragraphs, tables=[], figures=[])

    # -------------------- fitz fallback --------------------

    def _parse_with_fitz_text(self, pdf_path: Path, fetched: FetchedDoc) -> EvidencePack:
        """
        Simplest text fallback using PyMuPDF (fitz).
        - get_text("text") per page
        - paragraph split via _split_paragraphs
        - naive caption spotting (line-based)
        """
        assert fitz is not None, "fitz not available"
        paragraphs: List[Evidence] = []
        captions: List[Evidence] = []

        doc = fitz.open(str(pdf_path))
        try:
            for pidx in range(len(doc)):
                page = doc[pidx]
                try:
                    raw = page.get_text("text") or ""
                except Exception:
                    raw = page.get_text() or ""
                if not raw:
                    continue

                for ln in raw.splitlines():
                    cap = _is_caption(ln)
                    if cap:
                        kind, num = cap
                        ev = Evidence(
                            id=_ev_id(fetched.doc_id, "caption", pidx + 1, len(captions) + 1, ln),
                            type=EvidenceType.caption,
                            doc_id=fetched.doc_id,
                            page=pidx + 1,
                            figure_no=num if kind == "figure" else None,
                            table_no=num if kind == "table" else None,
                            caption=_normalize_text(ln),
                            doi=fetched.candidate.doi,
                            source_url=fetched.candidate.url,  # type: ignore
                        )
                        captions.append(ev)

                paras = _split_paragraphs(raw)
                for i, para in enumerate(paras, 1):
                    ev = Evidence(
                        id=_ev_id(fetched.doc_id, "paragraph", pidx + 1, i, para),
                        type=EvidenceType.paragraph,
                        doc_id=fetched.doc_id,
                        page=pidx + 1,
                        text=para,
                        doi=fetched.candidate.doi,
                        source_url=fetched.candidate.url,  # type: ignore
                    )
                    paragraphs.append(ev)
        finally:
            doc.close()

        return EvidencePack(doc_id=fetched.doc_id, paragraphs=paragraphs, tables=[], figures=[])

    # -------------------- persistence --------------------

    def _persist_pack(self, doc_id: str, pack: EvidencePack) -> Path:
        path = _pack_path(doc_id)
        payload = {
            "doc_id": pack.doc_id,
            "paragraphs": [e.__dict__ for e in pack.paragraphs],
            "tables": [e.__dict__ for e in pack.tables],
            "figures": [e.__dict__ for e in pack.figures],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


# ------------------------------ CLI self-test ---------------------------------

if __name__ == "__main__":
    """
    Smoke test:
      1) Ensure a PDF exists under storage/docs/<doc_id>/<doc_id>.pdf (use downloader first).
      2) Optionally set GROBID_URL in citevizor.env or OS env for better structuring.
      3) Run: python parse/pdf_structure.py <doc_id>
    """
    ensure_dirs()
    project_root = Path(__file__).resolve().parents[1]
    parser = PDFStructureParser.from_env(project_root)

    if len(sys.argv) < 2:
        print("Usage: python parse/pdf_structure.py <doc_id>")
        sys.exit(1)

    doc_id = sys.argv[1].strip()
    pdf_path = DOCS_DIR / doc_id / f"{doc_id}.pdf"
    meta_path = DOCS_DIR / doc_id / f"{doc_id}.meta.json"
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found at: {pdf_path}")
        sys.exit(2)
    if not meta_path.exists():
        print(f"[WARN] meta.json not found: {meta_path} (continuing)")

    # Minimal FetchedDoc reconstruction for standalone test
    candidate_stub = {
        "title": doc_id,
        "url": "https://example.org",
        "year": None,
        "source": "scholar",
        "score": 1.0,
        "authors": [],
        "venue": None,
        "is_review": None,
        "open_access": None,
        "doi": None,
    }
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            candidate_stub.update({
                "title": meta.get("title") or doc_id,
                "url": meta.get("url") or candidate_stub["url"],
                "year": meta.get("year"),
                "doi": meta.get("doi"),
            })
        except Exception:
            pass

    from schemas import Candidate, DocumentPaths

    cand = Candidate(
        title=candidate_stub["title"],
        url=candidate_stub["url"],  # type: ignore
        year=candidate_stub["year"],
        source="scholar",
        score=1.0,
        authors=candidate_stub["authors"],
        venue=candidate_stub["venue"],
        is_review=candidate_stub["is_review"],
        open_access=candidate_stub["open_access"],
        doi=candidate_stub["doi"],
    )
    paths = DocumentPaths(base_dir=pdf_path.parent, pdf_path=pdf_path, meta_json=meta_path)
    fetched = FetchedDoc(doc_id=doc_id, candidate=cand, paths=paths)

    pack = parser.parse_pdf(fetched, persist=True)
    out_path = _pack_path(doc_id)
    print(f"[OK] Parsed PDF -> {out_path}")
    print(f"  paragraphs={len(pack.paragraphs)} tables={len(pack.tables)} figures={len(pack.figures)}")
