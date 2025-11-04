# C:\CiteVizor\citevizor_backend.py
# -*- coding: utf-8 -*-
"""
CiteVizor - one command to run the entire backend pipeline.

Flow
1) Plan queries -> retrieval (Scholar + Web) -> aggregate & dedupe
2) Predownload resolve (DOI/OA/landing → direct PDF when possible)
3) Download (streaming, PDF-first, rich reasons, per-host throttle)
3.1) Figure extract (IMMEDIATE, right after download; path-contract enforced)
4) Parse dispatch (structure/tables/figures + OCR fallback)
5) Evidence ranking -> Citation map -> Engineer-oriented summary (LLM)
6) Chart building -> HTML layout -> optional PPTX export
7) QA checks -> Trace manifest (incl. trace.downloads)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- paths & schema contracts ----
from schemas import (
    STORAGE_DIR, EVIDENCE_DIR, RENDERS_DIR, REPORTS_DIR,
    Candidate, SearchPlan, YearRange,
    # NEW: schema-side objects to build a programmatic extract call
    FetchedDoc as SchemaFetchedDoc,
    DocumentPaths as SchemaDocPaths,
)
from schemas import pdf_path_for  # path-contract helper

# ---- logging (unified, zero-behavior-change to other modules) ----
try:
    from infra.logging import setup_logging, get_logger, set_run_id, span
except Exception:
    def setup_logging(*a, **k): return None
    def get_logger(name=None):
        class _L:
            def info(self, *a, **k): pass
            def debug(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass
        return _L()
    def set_run_id(*a, **k): return None
    class span:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, et, e, tb): return False

LOG = get_logger("backend")

# ---- retrieval ----
from retrieval.serper_scholar import SerperScholarClient
try:
    from retrieval.serper_web import SerperWebClient
except Exception:
    SerperWebClient = None
from retrieval.aggregate_rank import aggregate_candidates

# ---- predownload (NEW) ----
from retrieval.predownload import resolve_list as predownload_resolve

# ---- fetch ----
from fetch.downloader import Downloader, DownloadConfig, FetchedDoc  # downloader's FetchedDoc (download result shape)
from fetch.downloader import finalize_pdf_path  # NEW: enforce storage/docs/<doc_id>/<doc_id>.pdf

# ---- parsing ----
from parse.parse_dispatch import parse_all, ParseResult

# ---- figure extract (IMMEDIATE) ----
from parse.figure_extract import FigureExtractConfig, FigureExtractor  # programmatic extractor

# ---- align & mapping ----
from align.metadata_enricher import MetadataEnricher
from align.dedupe import Deduper
from align.evidence_rank import EvidenceRanker, EvidenceRankConfig
from align.citation_map import CitationMapper, CitationMapperConfig
from align.citation_map import _save_citations

# ---- llm / viz ----
from llm.client_vllm import VLLMClient, VLLMConfig
from llm.client_vllm import LLMUnavailableError, LLMServerError
from llm.summarize_engineer import EngineerSummarizer, SummarizerConfig
from viz.chart_builder import ChartBuilder

# ---- report / export ----
from report.layout import ReportLayout, LayoutSpec
try:
    from report.exporters import export_all  # if available
except Exception:
    export_all = None
# Compatibility: support both build_pptx() and export_pptx() without requiring either one.
try:
    from report.export_pptx import build_pptx as _build_pptx
except Exception:
    _build_pptx = None
try:
    from report.export_pptx import export_pptx as _export_pptx
except Exception:
    _export_pptx = None

# ---- QA / trace ----
try:
    from qa.consistency_checks import run_checks
except Exception:
    run_checks = None
try:
    from qa.trace_manifest import write_manifest
except Exception:
    write_manifest = None


# =====================================================================================

@dataclass
class BackendArgs:
    query: str
    lang: str
    years: Optional[str]
    gl: Optional[str]
    k_scholar: int
    k_web: int
    topk_rank: int
    limit_tables: int
    enable_ocr: bool
    serve_preview: bool
    export_pptx_flag: bool


# ---------- planning ----------

def _make_years(years: Optional[str]) -> Optional[YearRange]:
    if not years:
        return None
    years = years.strip()
    try:
        if "-" in years:
            a, b = years.split("-", 1)
            return YearRange(start=int(a), end=int(b))
        if years.startswith(">="):
            return YearRange(start=int(years[2:]), end=None)
        if years.startswith("<="):
            return YearRange(start=None, end=int(years[2:]))
    except Exception:
        return None
    return None


def _plan_queries(user_query: str, lang: str, k_top: int, years: Optional[str]) -> SearchPlan:
    q = user_query.strip()
    exps = [q, f"{q} review", f"{q} meta-analysis", f"{q} survey"]
    if lang.lower().startswith("zh"):
        exps += [f"{q} 综述", f"{q} 系统综述", f"{q} 元分析"]
    if lang.lower().startswith("ja"):
        exps += [f"{q} レビュー", f"{q} サーベイ"]
    yr = _make_years(years)

    yr_filters = {}
    if yr:
        for attr in ("model_dump", "dict"):
            m = getattr(yr, attr, None)
            if callable(m):
                try:
                    yr_filters = m()
                    break
                except Exception:
                    pass

    return SearchPlan(
        queries=list(dict.fromkeys([e for e in exps if e.strip()])),
        k_top=max(5, k_top),
        filters={"years": yr_filters},
    )


# ---------- util ----------

def _ensure_dirs() -> None:
    for d in (STORAGE_DIR, EVIDENCE_DIR, RENDERS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _gen_run_id() -> str:
    t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"run-{t}-{uuid.uuid4().hex[:6]}"


def _summary_json_path(run_id: str) -> Path:
    out_dir = EVIDENCE_DIR / "_rank"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{run_id}.summary.json"


def _write_minimal_summary(run_id: str, reason: str) -> Path:
    p = _summary_json_path(run_id)
    payload = {
        "run_id": run_id,
        "highlights": [],
        "figure_captions": [],
        "notes": [],
        "error": reason,
        "time": int(time.time()),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _to_dict_safe(x):
    for attr in ("model_dump", "dict"):
        m = getattr(x, attr, None)
        if callable(m):
            try:
                return m()
            except Exception:
                pass
    return {
        "title": getattr(x, "title", ""),
        "url": getattr(x, "url", None),
        "doi": getattr(x, "doi", None),
        "year": getattr(x, "year", None),
        "source": str(getattr(x, "source", "web")),
        "score": float(getattr(x, "score", 0.0)),
    }


def _persist_retrieval_agg(run_id: str, items):
    try:
        out_dir = STORAGE_DIR / "retrieval"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "plan": {"plan_id": run_id},
            "items": [_to_dict_safe(it) for it in (items or [])],
            "stats": {"count": len(items or [])},
        }
        (out_dir / f"{run_id}.agg.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOG.info("retrieval.persist", extra={"path": str(out_dir / f"{run_id}.agg.json"), "count": len(items or [])})
    except Exception as e:
        LOG.warning("retrieval.persist_fail", extra={"err": str(e)[:200]})


def _sanitize_pack_json(doc_id: str) -> None:
    try:
        pack_path = EVIDENCE_DIR / doc_id / "pack.json"
        if not pack_path.exists():
            return
        data = json.loads(pack_path.read_text(encoding="utf-8"))

        def sani(items, kind):
            out = []
            for e in items or []:
                if not isinstance(e, dict):
                    continue
                x = dict(e)
                if "img_path" in x and "image_path" not in x:
                    x["image_path"] = x.pop("img_path")
                if x.get("image_path") is not None:
                    x["image_path"] = str(x["image_path"])
                if "page" in x and x["page"] not in (None, ""):
                    try:
                        x["page"] = int(x["page"])
                    except Exception:
                        x["page"] = None
                if "figure_no" in x and x["figure_no"] is not None:
                    x["figure_no"] = str(x["figure_no"])
                if "table_no" in x and x["table_no"] is not None:
                    x["table_no"] = str(x["table_no"])
                if not x.get("type"):
                    x["type"] = kind
                if not x.get("doc_id"):
                    x["doc_id"] = doc_id
                out.append(x)
            return out

        data["figures"] = sani(data.get("figures", []), "figure")
        data["tables"] = sani(data.get("tables", []), "table")
        if "paragraphs" in data and isinstance(data["paragraphs"], list):
            for p in data["paragraphs"]:
                if isinstance(p, dict) and not p.get("doc_id"):
                    p["doc_id"] = doc_id

        pack_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        LOG.warning("pack.sanitize_fail", extra={"doc_id": doc_id, "err": str(e)[:200]})


# =====================================================================================

def run_backend(cfg: BackendArgs) -> Dict[str, Any]:
    _ensure_dirs()
    run_id = _gen_run_id()
    set_run_id(run_id)
    LOG.info("run.start", extra={"run_id": run_id, "query": cfg.query, "lang": cfg.lang})

    # -------- 1) retrieval --------
    with span(LOG, "retrieval"):
        plan = _plan_queries(cfg.query, cfg.lang, k_top=max(cfg.k_scholar, cfg.k_web, 5), years=cfg.years)
        LOG.info("retrieval.plan", extra={"queries": len(plan.queries), "years": plan.filters.get("years", {})})

        scholar = SerperScholarClient(api_key=None, project_root=Path.cwd())
        cand_s = scholar.search_with_plan(plan, limit_per_query=8, lang="en", gl=cfg.gl)

        cand_w: List[Candidate] = []
        if SerperWebClient and cfg.k_web > 0:
            try:
                webc = SerperWebClient(api_key=None, project_root=Path.cwd())
                cand_w = webc.search_multi(plan, limit_per_query=6, lang="en", gl=cfg.gl)
            except Exception as e:
                LOG.warning("retrieval.web_error", extra={"err": str(e)[:200]})

        cands = aggregate_candidates(
            scholar=cand_s[: cfg.k_scholar],
            web=cand_w[: cfg.k_web],
            max_total=max(cfg.k_scholar + cfg.k_web, 5),
        )
        LOG.info("retrieval.done", extra={"candidates": len(cands)})
        if not cands:
            _persist_retrieval_agg(run_id, [])
            raise RuntimeError("No candidates found.")
        _persist_retrieval_agg(run_id, cands)

    # -------- 1.5) predownload resolve --------
    with span(LOG, "predownload.resolve"):
        try:
            cands = predownload_resolve(cands)
        except Exception as e:
            LOG.warning("predownload.skip_error", extra={"err": str(e)[:200]})
        def _is_pdf(u: Optional[str]) -> bool:
            u = (u or "").lower()
            return u.endswith(".pdf") or (".pdf?" in u)
        pdf_like = sum(1 for c in cands if _is_pdf(getattr(c, "url", None)))
        LOG.info("predownload.stats", extra={"total": len(cands), "pdf_like": pdf_like})

    # -------- 2) download --------
    fetched: List[FetchedDoc] = []
    downloads_trace: List[Dict[str, Any]] = []
    with span(LOG, "download"):
        dl = Downloader(DownloadConfig.from_env(Path.cwd()))
        # gather trace inside this dict; downloader will append to downloads_trace
        trace_holder: Dict[str, Any] = {"downloads": []}
        docs_out_dir = STORAGE_DIR / "docs"
        docs_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            # convert Candidate -> minimal dict expected by downloader
            cand_dicts = []
            for c in cands:
                # NEW: pass a stable doc_id and source to the downloader
                _doc_id = getattr(c, "doc_id", None)
                if not _doc_id and hasattr(c, "stable_key"):
                    _doc_id = c.stable_key()
                cand_dicts.append({
                    "doc_id": _doc_id,                           # ensure non-empty doc_id
                    "source": str(getattr(c, "source", "")),     # propagate source (web/image/scholar)
                    "url": getattr(c, "url", None),
                    "pdf_url": getattr(c, "pdf_url", None) if hasattr(c, "pdf_url") else None,
                    "link": getattr(c, "link", None) if hasattr(c, "link") else None,
                    "referer": getattr(c, "referer", None) if hasattr(c, "referer") else None,
                    "host": None,
                })
            fetched = dl.download_many(cand_dicts, str(docs_out_dir), trace=trace_holder)
        except Exception as e:
            LOG.warning("download.error", extra={"err": str(e)[:200]})

        downloads_trace = list(trace_holder.get("downloads", []))
        LOG.info("download.done", extra={"docs": len([f for f in fetched if f.ok])})
        if not fetched:
            LOG.warning("download.none", extra={"hint": "ranker will fallback to retrieval agg"})

    # -------- 3.1) figure extract (IMMEDIATE) --------
    # Enforce file path contract and trigger extraction right after download.
    with span(LOG, "figure.extract_immediate"):
        fcfg = FigureExtractConfig.from_env(Path.cwd())
        extractor = FigureExtractor(fcfg)
        total_figs = 0
        total_tabs = 0
        triggered = 0

        for f in (fetched or []):
            try:
                did = getattr(f, "doc_id", None)
                if not did:
                    continue

                # Normalize to storage/docs/<doc_id>/<doc_id>.pdf if it is a PDF
                try:
                    finalize_pdf_path(f)  # safe no-op for non-PDFs
                except Exception:
                    pass

                # Locate the canonical PDF path and skip if not a real PDF
                pdf_p = pdf_path_for(did)
                if not pdf_p.exists() or pdf_p.suffix.lower() != ".pdf":
                    # Non-PDF downloads (images) are already registered by downloader as web_images.
                    continue

                # Build a schema.FetchedDoc for programmatic extract()
                paths = SchemaDocPaths(base_dir=pdf_p.parent, pdf_path=pdf_p, meta_json=pdf_p.parent / f"{did}.meta.json")
                cand = Candidate(title=did, url=None, year=None, source="web", score=1.0)  # minimal safe candidate
                sfd = SchemaFetchedDoc(doc_id=did, candidate=cand, paths=paths)

                pack = extractor.extract(sfd, persist=True)
                total_figs += len(getattr(pack, "figures", []) or [])
                total_tabs += len(getattr(pack, "tables", []) or [])
                triggered += 1
            except Exception as e:
                LOG.warning("figure.extract_immediate.fail", extra={"doc_id": getattr(f, "doc_id", ""), "err": str(e)[:200]})

        LOG.info("figure.extract_immediate.done", extra={"docs": triggered, "figures": total_figs, "tables": total_tabs})

    # -------- 4) parse --------
    with span(LOG, "parse"):
        results: List[ParseResult] = parse_all(fetched, enable_ocr=cfg.enable_ocr, project_root=Path.cwd())
        for r in results:
            LOG.info("parse.doc", extra={
                "doc_id": r.doc_id, "status": r.status,
                "paragraphs": r.paragraphs, "tables": r.tables, "figures": r.figures
            })

    # -------- 4.5) sanitize any legacy pack.json --------
    with span(LOG, "pack.sanitize"):
        for fd in fetched:
            try:
                if hasattr(fd, "doc_id") and fd.doc_id:
                    _sanitize_pack_json(fd.doc_id)
            except Exception as e:
                LOG.warning("pack.sanitize_error", extra={"doc_id": getattr(fd, "doc_id", ""), "err": str(e)[:200]})

    # -------- 5) metadata enrich + dedupe --------
    with span(LOG, "align.prep"):
        try:
            enricher = MetadataEnricher()
            for fd in fetched:
                if hasattr(fd, "doc_id") and fd.doc_id:
                    enricher.enrich_doc(fd.doc_id)
            LOG.info("meta.enrich_ok")
        except Exception as e:
            LOG.warning("meta.enrich_error", extra={"err": str(e)[:200]})
        try:
            d = Deduper()
            doc_ids = [fd.doc_id for fd in fetched if getattr(fd, "doc_id", None)]
            if doc_ids:
                d.run_on_docs(doc_ids)
        except Exception as e:
            LOG.warning("dedupe.error", extra={"err": str(e)[:200]})

    # -------- 6) evidence ranking --------
    with span(LOG, "rank"):
        rcfg = EvidenceRankConfig.from_env(Path.cwd())
        ranker = EvidenceRanker(rcfg)
        subqs = _plan_queries(cfg.query, cfg.lang, 8, cfg.years).queries
        doc_ids = [fd.doc_id for fd in fetched if getattr(fd, "doc_id", None)]
        rank_json = ranker.rank(query=cfg.query, doc_ids=(doc_ids or None), sub_queries=subqs, top_k=cfg.topk_rank)
        run_id_final = rank_json.get("run_id") or run_id
        set_run_id(run_id_final)
        LOG.info("rank.done", extra={"run_id": run_id_final, "items": len(rank_json.get("items", []))})

    # -------- 7) citation map --------
    with span(LOG, "citation_map"):
        mapper = CitationMapper(CitationMapperConfig.from_env(Path.cwd()))
        payload = mapper.build(rank_json)
        map_path = _save_citations(run_id_final, payload)
        LOG.info("citations.saved", extra={"path": str(map_path)})

    # -------- 8) summarization (LLM) --------
    summarization_ok = True
    with span(LOG, "summarize"):
        vcfg = VLLMConfig.from_env(Path.cwd())
        vcli = VLLMClient(vcfg)
        ready = vcli.wait_until_ready(max_wait_s=30)
        if not ready:
            summarization_ok = False
            _write_minimal_summary(run_id_final, "LLM unavailable: health check failed")
            LOG.error("summarize.abort_no_llm", extra={"run_id": run_id_final})
        else:
            try:
                scfg = SummarizerConfig.from_env(Path.cwd())
                summarizer = EngineerSummarizer(scfg, vcli)
                tb = summarizer.summarize(payload)
                LOG.info("summarize.done", extra={"highlights": len(tb.highlights), "figcaps": len(tb.figure_captions)})
            except (LLMUnavailableError, LLMServerError) as e:
                summarization_ok = False
                _write_minimal_summary(run_id_final, f"LLM error: {str(e)[:200]}")
                LOG.error("summarize.error_llm", extra={"err": str(e)[:200]})
            except Exception as e:
                summarization_ok = False
                _write_minimal_summary(run_id_final, f"Summarizer exception: {str(e)[:200]}")
                LOG.error("summarize.error", extra={"err": str(e)[:200]})

    # -------- 9) charts --------
    with span(LOG, "charts"):
        total_visuals = 0
        if fetched:
            builder = ChartBuilder()
            for fd in fetched:
                try:
                    if not getattr(fd, "doc_id", None):
                        continue
                    manifest_path = builder.build_for_doc(fd.doc_id, limit_per_doc=cfg.limit_tables)
                    try:
                        m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                        if isinstance(m, list):
                            total_visuals += len(m)
                        elif isinstance(m, dict):
                            total_visuals += len(m.get("records", []))
                    except Exception:
                        pass
                except Exception as e:
                    LOG.warning("charts.build_for_doc_fail", extra={"doc_id": getattr(fd, "doc_id", ""), "err": str(e)[:200]})
            LOG.info("charts.done", extra={"visuals": total_visuals})
        else:
            LOG.warning("charts.skip", extra={"reason": "no_fetched_docs"})

    # -------- 10) layout (HTML) --------
    with span(LOG, "layout"):
        spec = LayoutSpec(title=f"CiteVizor Report – {cfg.query[:64]}", lang=cfg.lang, audience="engineer", max_visuals=20)
        layout = ReportLayout(spec)
        html_path = layout.build(run_id_final)
        LOG.info("layout.html", extra={"path": str(html_path)})

    # -------- 11) export (PPTX) --------
    pptx_path: Optional[Path] = None
    if cfg.export_pptx_flag:
        with span(LOG, "export.pptx"):
            try:
                if export_all:
                    out = export_all(run_id_final, html_path)
                    pptx_path = Path(out.get("pptx")) if isinstance(out, dict) and out.get("pptx") else None
                elif _build_pptx:
                    pptx_path = Path(_build_pptx(run_id_final))
                elif _export_pptx:
                    pptx_path = Path(_export_pptx(run_id_final))
                LOG.info("export.pptx_done", extra={"path": str(pptx_path) if pptx_path else ""})
            except Exception as e:
                LOG.warning("export.pptx_fail", extra={"err": str(e)[:200]})

    # -------- 12) QA checks --------
    qa_report: Dict[str, Any] = {}
    if run_checks:
        with span(LOG, "qa.checks"):
            try:
                qa_report = run_checks(run_id_final) or {}
                LOG.info("qa.ok", extra={"issues": len(qa_report.get("issues", []))})
            except Exception as e:
                LOG.warning("qa.error", extra={"err": str(e)[:200]})

    # -------- 13) trace manifest --------
    manifest_path: Optional[Path] = None
    with span(LOG, "trace.manifest"):
        retrieval_candidates = []
        try:
            retrieval_candidates = [_to_dict_safe(c) for c in (cands or [])]
        except Exception:
            retrieval_candidates = []

        manifest = {
            "run_id": run_id_final,
            "query": cfg.query,
            "lang": cfg.lang,
            "years": cfg.years,
            "retrieval": {"candidates": retrieval_candidates},
            "artifacts": {
                "citations_json": str(EVIDENCE_DIR / "_rank" / f"{run_id_final}.citations.json"),
                "summary_json": str(_summary_json_path(run_id_final)),
                "html_report": str(html_path),
                "pptx": (str(pptx_path) if pptx_path else None),
            },
            "qa": qa_report,
            "time": int(time.time()),
            "trace": {"downloads": downloads_trace},
        }
        try:
            alt_path = None
            if write_manifest:
                try:
                    alt_path = write_manifest(run_id_final, manifest)
                except Exception as e:
                    LOG.warning("trace.external_writer_fail", extra={"err": str(e)[:200]})
                    alt_path = None

            runs_dir = REPORTS_DIR / "_runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = runs_dir / f"{run_id_final}.manifest.json"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

            LOG.info("trace.saved", extra={"path": str(manifest_path), "external": (str(alt_path) if alt_path else None)})
        except Exception as e:
            LOG.warning("trace.error", extra={"err": str(e)[:200]})

    # -------- 14) optional: preview server --------
    if cfg.serve_preview:
        LOG.info("preview.start", extra={"run_id": run_id_final})
        try:
            import uvicorn
            uvicorn.run("review.preview_server:app", host="0.0.0.0", port=8787, reload=False, workers=1)
        except Exception as e:
            LOG.warning("preview.fail", extra={"err": str(e)[:200]})

    return {
        "run_id": run_id_final,
        "html": str(html_path),
        "pptx": (str(pptx_path) if pptx_path else None),
        "manifest": (str(manifest_path) if manifest_path else None),
    }


# =====================================================================================

def _parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor backend one-shot runner")
    ap.add_argument("--query", required=True, help="Research question")
    ap.add_argument("--lang", default="en", help="en|zh|ja")
    ap.add_argument("--years", default="", help='Year filter like "2021-2025", ">=2020"')
    ap.add_argument("--gl", default="", help="Serper geolocation code (e.g., jp, hk)")
    ap.add_argument("--k-scholar", type=int, default=6, help="Top N from Scholar")
    ap.add_argument("--k-web", type=int, default=4, help="Top N from general web/patents/news/universities")
    ap.add_argument("--topk-rank", type=int, default=60, help="Top-K evidence retained after ranking")
    ap.add_argument("--limit-tables", type=int, default=6, help="Max visuals (charts+figures) per document")
    ap.add_argument("--enable-ocr", action="store_true", help="Enable OCR fallback for scanned PDFs")
    ap.add_argument("--serve-preview", action="store_true", help="Start preview server after pipeline finishes")
    ap.add_argument("--export-pptx", action="store_true", help="Export PPTX in addition to HTML")
    return ap.parse_args()


def main() -> int:
    args = _parse_cli()
    cfg = BackendArgs(
        query=args.query,
        lang=args.lang,
        years=(args.years or None),
        gl=(args.gl or None),
        k_scholar=max(1, args.k_scholar),
        k_web=max(0, args.k_web),
        topk_rank=max(10, args.topk_rank),
        limit_tables=max(1, args.limit_tables),
        enable_ocr=bool(args.enable_ocr),
        serve_preview=bool(args.serve_preview),
        export_pptx_flag=bool(args.export_pptx),
    )

    try:
        setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    except Exception:
        pass

    try:
        out = run_backend(cfg)
        LOG.info("run.done", extra=out)
        print("\n=== DONE ===")
        print(f"run_id : {out['run_id']}")
        print(f"HTML   : {out['html']}")
        if out.get("pptx"): print(f"PPTX   : {out['pptx']}")
        if out.get("manifest"): print(f"Manifest: {out['manifest']}")
        print()
        return 0
    except Exception as e:
        LOG.error("run.fail", extra={"err": str(e)[:500]})
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
