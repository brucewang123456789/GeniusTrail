# parse/parse_dispatch.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any
import os

try:
    from infra.logging import get_logger, span
except Exception:
    def get_logger(name=None):
        class _L:
            def info(self,*a,**k): pass
            def debug(self,*a,**k): pass
            def warning(self,*a,**k): pass
            def error(self,*a,**k): pass
        return _L()
    class span:
        def __init__(self,*a,**k): pass
        def __enter__(self): return self
        def __exit__(self, et, e, tb): return False

LOG = get_logger("parse.dispatch")

# ---- downloader interop (normalization hook) ----
try:
    from fetch.downloader import FetchedDoc as DLFetchedDoc, finalize_pdf_path
except Exception:
    @dataclass
    class DLFetchedDoc:
        ok: bool = True
        reason: Optional[str] = None
        path: Optional[str] = None
        url: Optional[str] = None
        final_url: Optional[str] = None
        status: Optional[int] = None
        content_type: Optional[str] = None
        host: Optional[str] = None
        doc_id: Optional[str] = None
    def finalize_pdf_path(obj: Any, **kwargs) -> Any:
        # no-op fallback
        return obj

# ---- schemas path contract helpers ----
try:
    from schemas import pdf_path_for, evidence_dir_for
except Exception:
    def pdf_path_for(doc_id: str) -> Path:
        return Path("/workspace/storage/docs") / doc_id / f"{doc_id}.pdf"
    def evidence_dir_for(doc_id: str) -> Path:
        return Path("/workspace/storage/evidence") / doc_id

# local compatibility alias so rest of file can import FetchedDoc
try:
    from fetch.downloader import FetchedDoc
except Exception:
    @dataclass
    class FetchedDoc:
        doc_id: str
        paths: Any

# ---- schema objects used to build a programmatic FetchedDoc for parsers ----
# (Minimal, safe imports – no behavior change elsewhere)
from schemas import (
    Candidate as SCandidate,
    DocumentPaths as SDocPaths,
    FetchedDoc as SFetchedDoc,
    SourceType as SSourceType,
)

from parse.pdf_structure import PDFStructureParser
from parse.table_extract import TableExtractConfig, TableExtractor
from parse.figure_extract import FigureExtractConfig, FigureExtractor

try:
    from parse.ocr_fallback import ocr_document
except Exception:
    ocr_document = None

try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None

PDF_MAGIC = b"%PDF-"
PDF_MIN_BYTES = int(os.getenv("CITEVIZOR_PDF_MIN_BYTES", "1024"))

@dataclass
class ParseResult:
    doc_id: str
    status: str
    paragraphs: int
    tables: int
    figures: int
    meta: Dict[str, Any]

# ---------------------- small helpers ----------------------

def _file_health_check(pdf_path: Optional[Path]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "exists": False, "size": 0, "is_pdf_header": False,
        "pages": None, "encrypted": None, "scan_hint": None
    }
    if not pdf_path or not isinstance(pdf_path, Path):
        return info
    try:
        if pdf_path.exists():
            info["exists"] = True
            info["size"] = pdf_path.stat().st_size
            with open(pdf_path, "rb") as f:
                head = f.read(5)
            info["is_pdf_header"] = (head == PDF_MAGIC)
            if PyPDF2 is not None:
                try:
                    with open(pdf_path, "rb") as fh:
                        reader = PyPDF2.PdfReader(fh)
                        info["pages"] = len(reader.pages) if getattr(reader, "pages", None) is not None else None
                        info["encrypted"] = bool(getattr(reader, "is_encrypted", False))
                except Exception:
                    pass
        return info
    except Exception:
        return info

def _ensure_pdf_path_contract(fd: FetchedDoc) -> Optional[Path]:
    """
    Ensure fd.paths.pdf_path exists and obeys the path contract:
      storage/docs/<doc_id>/<doc_id>.pdf
    Steps:
      1) If paths.pdf_path is None or missing, try to infer via pdf_path_for(doc_id).
      2) Call finalize_pdf_path(fd) to normalize and write back into fd if needed.
      3) Return the resolved Path (or None if not found).
    """
    try:
        # 1) try existing path
        pdf_path: Optional[Path] = getattr(getattr(fd, "paths", None), "pdf_path", None)
        if pdf_path and isinstance(pdf_path, Path) and pdf_path.exists():
            try:
                finalize_pdf_path(fd)  # make sure it's normalized to contract path
            except Exception:
                pass
            return getattr(getattr(fd, "paths", None), "pdf_path", None)

        # 2) infer by contract
        did = getattr(fd, "doc_id", None)
        if not did:
            return None
        inferred = pdf_path_for(did)
        if inferred.exists():
            # write back to fd.paths.pdf_path if possible
            try:
                setattr(getattr(fd, "paths"), "pdf_path", inferred)
            except Exception:
                pass
            try:
                finalize_pdf_path(fd)
            except Exception:
                pass
            return inferred

        # 3) as a last resort, try to normalize if fd has a temp local path field
        tmp = getattr(fd, "path", None)
        if tmp:
            try:
                setattr(getattr(fd, "paths"), "pdf_path", Path(tmp))
                finalize_pdf_path(fd)
                p2 = getattr(getattr(fd, "paths", None), "pdf_path", None)
                return p2 if (p2 and p2.exists()) else None
            except Exception:
                return None
        return None
    except Exception:
        return None

def _run_parsers_on_valid_pdf(
    fd: SFetchedDoc,
    enable_ocr: bool,
    ps: PDFStructureParser,
    tex: TableExtractor,
    fex: FigureExtractor
) -> Tuple[str, int, int, int, str]:
    """
    Run the full parser chain on a valid PDF:
      - structure -> tables -> figures
      - if paragraphs are empty and enable_ocr, trigger OCR fallback
      - if all empty, mark as empty_parse
    """
    status = "parsed"

    # structure
    pack = ps.parse_pdf(fd, persist=True)

    # tables
    pack = tex.extract(fd, persist=True)

    # figures (ensure figures are generated even if called earlier from backend)
    pack = fex.extract(fd, persist=True)

    para_count = len(getattr(pack, "paragraphs", []) or [])
    table_count = len(getattr(pack, "tables", []) or [])
    fig_count   = len(getattr(pack, "figures", []) or [])

    need_ocr = enable_ocr and (para_count == 0 or para_count is None)
    if need_ocr and ocr_document and getattr(fd.paths, "pdf_path", None):
        try:
            ocr_document(doc_id=fd.doc_id, pdf_path=fd.paths.pdf_path, merge_pack=True)
            status = "ocr_ok"
        except Exception as e:
            LOG.warning("ocr.fail", extra={"doc_id": fd.doc_id, "err": str(e)[:200]})

    if para_count == 0 and table_count == 0 and fig_count == 0:
        status = "empty_parse"

    return fd.doc_id, para_count, table_count, fig_count, status

# ---------------------- main entrypoints ----------------------

def parse_all(
    fetched: Iterable[DLFetchedDoc],
    enable_ocr: bool = False,
    project_root: Optional[Path] = None
) -> List[ParseResult]:
    """
    Contract-aware parser dispatcher:
      - Before parsing, normalize each fd's pdf_path to the canonical location.
      - If pdf_path missing but canonical file exists, adopt it.
      - If invalid, log skip_invalid with reason.
      - Otherwise run structure->table->figure and OCR fallback if requested.
    """
    results: List[ParseResult] = []
    root = project_root or Path.cwd()

    ps = PDFStructureParser.from_env(root)
    tcfg = TableExtractConfig.from_env(root); tex = TableExtractor(tcfg)
    fcfg = FigureExtractConfig.from_env(root); fex = FigureExtractor(fcfg)

    with span(LOG, "parse.dispatch"):
        for fd in (fetched or []):
            try:
                # Step 0: ensure pdf_path obeys the path contract and exists
                pdf_p = _ensure_pdf_path_contract(fd)

                # Health check
                hc = _file_health_check(pdf_p)
                hc_log = {k: (int(v) if isinstance(v, bool) else v) for k, v in hc.items()}
                hc_log.update({"doc_id": getattr(fd, "doc_id", "")})

                reason = None
                if not hc.get("exists"):
                    reason = "not_exists"
                elif int(hc.get("size") or 0) < PDF_MIN_BYTES:
                    reason = "too_small"
                elif not hc.get("is_pdf_header"):
                    reason = "no_pdf_header"

                if reason is not None:
                    hc_log["reason"] = reason
                    LOG.info("parse.skip_invalid", extra=hc_log)
                    results.append(ParseResult(
                        doc_id=getattr(fd, "doc_id", ""),
                        status="skip_invalid",
                        paragraphs=0, tables=0, figures=0,
                        meta=hc_log
                    ))
                    continue

                # Build a schema.FetchedDoc for the parser chain (minimal fields only)
                did = getattr(fd, "doc_id", "")
                paths = SDocPaths(base_dir=pdf_p.parent, pdf_path=pdf_p, meta_json=pdf_p.parent / f"{did}.meta.json")
                cand  = SCandidate(title=did or "doc", url=None, year=None, source=SSourceType.web, score=1.0)
                schema_fd = SFetchedDoc(doc_id=did, candidate=cand, paths=paths)

                # Parse chain on valid PDF (schema_fd has .paths.* as required by parsers)
                doc_id, p, t, f, st = _run_parsers_on_valid_pdf(schema_fd, enable_ocr, ps, tex, fex)
                LOG.info("parse.doc", extra={"doc_id": doc_id, "paragraphs": p, "tables": t, "figures": f, "status": st})
                results.append(ParseResult(
                    doc_id=doc_id, status=st, paragraphs=p, tables=t, figures=f, meta=hc_log
                ))

            except Exception as e:
                dd = getattr(fd, "doc_id", "")
                LOG.warning("parse.error", extra={"doc_id": dd, "err": str(e)[:200]})
                results.append(ParseResult(
                    doc_id=dd, status="error", paragraphs=0, tables=0, figures=0, meta={"err": str(e)[:200]}
                ))

    summary = {
        "total": len(results),
        "status_counts": {k: sum(1 for r in results if r.status == k)
                          for k in ("parsed", "ocr_ok", "empty_parse", "skip_invalid", "error")}
    }
    LOG.info("parse.summary", extra=summary)
    return results

def parse_one(
    fd: DLFetchedDoc,
    enable_ocr: bool = False,
    project_root: Optional[Path] = None
) -> ParseResult:
    res = parse_all([fd], enable_ocr=enable_ocr, project_root=project_root)
    return res[0] if res else ParseResult(doc_id=getattr(fd, "doc_id", ""), status="error",
                                          paragraphs=0, tables=0, figures=0, meta={})
