# C:\CiteVizor\pipeline\run_once.py
# -*- coding: utf-8 -*-
"""
CiteVizor - End-to-end one-shot pipeline runner

Stages:
  1) Retrieval (Serper Scholar) -> Candidates
  2) Download (PDF-first) -> storage/docs/<doc_id>/*
  3) Parse (pdf_structure + table_extract + figure_extract) -> storage/evidence/<doc_id>/pack.json
  4) Align & rank (BM25 + boosts) -> storage/evidence/_rank/<run_id>.json
  5) Citation map -> storage/evidence/_rank/<run_id>.citations.json
  6) Engineer summary (LLM via vLLM) -> storage/evidence/_rank/<run_id>.summary.json
  7) Chart codegen (LLM) -> storage/renders/<doc_id>/charts/*.png
  8) HTML layout -> storage/reports/<run_id>.html

Model & serving:
  - Uses your local vLLM OpenAI-compatible server with model:
      Qwen/Qwen2.5-14B-Instruct-AWQ  (HF: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ)
  - No new env keys required; reuse existing citevizor.env settings.

Run example:
  python pipeline/run_once.py --query "latest design trends of PV encapsulant for seaside corrosion" --lang en --k 5
  python pipeline/run_once.py --query "海边光伏组件封装材料最新趋势" --lang zh --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Core schemas/paths
from schemas import (
    SearchPlan, YearRange, STORAGE_DIR, EVIDENCE_DIR, RENDERS_DIR, REPORTS_DIR
)

# Retrieval
from retrieval.serper_scholar import SerperScholarClient

# Fetch
from fetch.downloader import DownloadConfig, Downloader

# Parse
from parse.pdf_structure import PDFStructureParser
from parse.table_extract import TableExtractConfig, TableExtractor
from parse.figure_extract import FigureExtractConfig, FigureExtractor

# Align
from align.evidence_rank import EvidenceRankConfig, EvidenceRanker
from align.citation_map import CitationMapperConfig, CitationMapper

# LLM (vLLM client + summarizer + chart codegen)
from llm.client_vllm import VLLMConfig, VLLMClient
from llm.summarize_engineer import SummarizerConfig, EngineerSummarizer
from llm.chart_codegen import CodegenConfig, ChartCodegen

# Report
from report.layout import ReportLayout, LayoutSpec

# --- unified logging (no behavior change) ---
from infra.logging import setup_logging, get_logger, set_run_id
log = get_logger("pipeline")

# ------------------------------ helpers ---------------------------------------

def _project_root() -> Path:
    # .../CiteVizor/pipeline/run_once.py -> parents[1] == project root
    return Path(__file__).resolve().parents[1]

def _log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}")

def _warn(stage: str, msg: str) -> None:
    print(f"[{stage}][WARN] {msg}")

def _err(stage: str, msg: str) -> None:
    print(f"[{stage}][ERROR] {msg}")


def _make_plan(user_query: str, lang: str, years: Optional[str], k_top: int) -> SearchPlan:
    """
    Lightweight query planner: expands the user query with review/meta-analysis variations and
    keeps k_top for retrieval aggregation. Year range can be "2020-2025" or ">=2021".
    """
    q = user_query.strip()
    expansions = [
        q,
        f"{q} review",
        f"{q} meta-analysis",
        f"{q} survey",
    ]
    if lang.lower().startswith("zh"):
        expansions += [f"{q} 综述", f"{q} 系统综述", f"{q} 元分析"]
    if lang.lower().startswith("ja"):
        expansions += [f"{q} レビュー", f"{q} サーベイ"]

    yr = None
    if years:
        years = years.strip()
        try:
            if "-" in years:
                a, b = years.split("-", 1)
                yr = YearRange(start=int(a), end=int(b))
            elif years.startswith(">="):
                yr = YearRange(start=int(years[2:]), end=None)
            elif years.startswith("<="):
                yr = YearRange(start=None, end=int(years[2:]))
        except Exception:
            yr = None

    plan = SearchPlan(
        queries=[e for e in expansions if e.strip()],
        k_top=max(5, k_top),
        filters={"years": yr.dict() if yr else {}}
    )
    return plan


# ------------------------------ pipeline --------------------------------------

@dataclass
class PipelineArgs:
    query: str
    lang: str
    years: Optional[str]
    k: int
    gl: Optional[str]
    limit_tables: int
    topk_rank: int

def run_once(cfg: PipelineArgs) -> Path:
    root = _project_root()
    # Ensure base dirs exist
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Retrieval ----------
    _log("Retrieval", "building search plan…")
    plan = _make_plan(cfg.query, cfg.lang, cfg.years, k_top=max(10, cfg.k * 3))

    _log("Retrieval", f"queries={len(plan.queries)}  k_top={plan.k_top}")
    retr = SerperScholarClient(api_key=None, project_root=root)
    cands = retr.search_with_plan(
        plan,
        limit_per_query=8,
        lang=("en" if cfg.lang == "en" else "en"),
        gl=cfg.gl
    )
    if not cands:
        raise RuntimeError("No candidates returned from Serper.")
    cands = cands[: cfg.k]
    _log("Retrieval", f"picked {len(cands)} candidates")

    # ---------- 2) Download ----------
    dl = Downloader(DownloadConfig.from_env(root))
    fetched_ids: List[str] = []
    for i, cand in enumerate(cands, 1):
        try:
            fd = dl.fetch_candidate(cand)
            fetched_ids.append(fd.doc_id)
            _log("Download", f"{i}/{len(cands)} OK -> {fd.doc_id}")
        except Exception as e:
            _warn("Download", f"skip '{cand.title}': {e}")

    if not fetched_ids:
        raise RuntimeError("No PDFs downloaded; aborting.")

    # ---------- 3) Parse (PDF -> evidence pack) ----------
    parser = PDFStructureParser.from_env(root)
    tcfg = TableExtractConfig.from_env(root)
    tex = TableExtractor(tcfg)
    fcfg = FigureExtractConfig.from_env(root)
    fex = FigureExtractor(fcfg)

    for j, doc_id in enumerate(fetched_ids, 1):
        try:
            from schemas import Candidate, DocumentPaths, FetchedDoc
            # Reconstruct minimal FetchedDoc from paths/meta for parser APIs
            pdf_path = STORAGE_DIR / "docs" / doc_id / f"{doc_id}.pdf"
            meta_path = STORAGE_DIR / "docs" / doc_id / f"{doc_id}.meta.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            cand = Candidate(
                title=meta.get("title") or doc_id,
                url=meta.get("url") or "https://example.org",
                year=meta.get("year"),
                source="scholar",
                score=1.0,
                doi=meta.get("doi"),
            )
            paths = DocumentPaths(base_dir=pdf_path.parent, pdf_path=pdf_path, meta_json=meta_path)
            fetched = FetchedDoc(doc_id=doc_id, candidate=cand, paths=paths)

            pack = parser.parse_pdf(fetched, persist=True)
            _log("Parse", f"{j}/{len(fetched_ids)} paragraphs={len(pack.paragraphs)}")

            pack = tex.extract(fetched, persist=True)
            _log("Parse", f"tables so far={len(pack.tables)}")

            pack = fex.extract(fetched, persist=True)
            _log("Parse", f"figures so far={len(pack.figures)}")
        except Exception as e:
            _warn("Parse", f"{doc_id}: {e}")

    # ---------- 4) Align & rank ----------
    rcfg = EvidenceRankConfig.from_env(root)
    ranker = EvidenceRanker(rcfg)
    try:
        rank_json = ranker.rank(
            query=cfg.query,
            doc_ids=fetched_ids,
            sub_queries=plan.queries,
            top_k=cfg.topk_rank
        )
    except Exception as e:
        raise RuntimeError(f"Ranking failed: {e}")
    run_id = rank_json.get("run_id")
    _log("Rank", f"run_id={run_id} items={len(rank_json.get('items', []))}")

    # ---- bind run_id to logging context (for structured logs) ----
    try:
        if run_id:
            set_run_id(str(run_id))
            log.info("rank complete", extra={"items": len(rank_json.get("items", []))})
    except Exception:
        # logging must never break the pipeline
        pass

    # ---------- 5) Citation map ----------
    mapper = CitationMapper(CitationMapperConfig.from_env(root))
    payload = mapper.build(rank_json)
    from align.citation_map import _save_citations  # reuse helper
    map_path = _save_citations(run_id, payload)
    _log("Citations", f"map -> {map_path.name} refs={len(payload.get('refs', []))}")

    # ---------- 6) Summarize (LLM via vLLM) ----------
    vcfg = VLLMConfig.from_env(root)
    vclient = VLLMClient(vcfg)
    scfg = SummarizerConfig.from_env(root)
    summarizer = EngineerSummarizer(scfg, vclient)
    tb = summarizer.summarize(payload)
    _log("Summarize", f"highlights={len(tb.highlights)} figure_captions={len(tb.figure_captions)}")

    # ---------- 7) Chart codegen (per doc) ----------
    ccfg = CodegenConfig.default(root)
    codegen = ChartCodegen(ccfg, vclient)
    for doc_id in fetched_ids:
        try:
            codes = codegen.generate_for_doc(doc_id, limit=cfg.limit_tables)
            _log("Charts", f"{doc_id}: generated {len(codes)} charts")
        except Exception as e:
            _warn("Charts", f"{doc_id}: {e}")

    # ---------- 8) Layout (HTML) ----------
    spec = LayoutSpec(
        title=f"CiteVizor Report – {cfg.query[:64]}",
        lang=cfg.lang,
        audience="engineer",
        max_visuals=18
    )
    rl = ReportLayout(spec)
    out_html = rl.build(run_id)
    _log("Report", f"HTML -> {out_html}")
    return out_html


# ------------------------------ CLI -------------------------------------------

def _parse_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CiteVizor one-shot pipeline runner")
    ap.add_argument("--query", required=True, help="User query (research question)")
    ap.add_argument("--lang", default="en", help="en|zh|ja (affects query expansions and summary language)")
    ap.add_argument("--years", default="", help='Optional year filter like "2021-2025" or ">=2020"')
    ap.add_argument("--k", type=int, default=5, help="Number of top candidates to download")
    ap.add_argument("--gl", default="", help="Optional country code for Serper (e.g., jp, hk)")
    ap.add_argument("--limit-tables", type=int, default=6, help="Max tables to chart per document")
    ap.add_argument("--topk-rank", type=int, default=50, help="Top-K evidence items to keep in ranking")
    return ap

if __name__ == "__main__":
    args = _parse_args().parse_args()
    cfg = PipelineArgs(
        query=args.query,
        lang=args.lang,
        years=(args.years or None),
        k=max(1, args.k),
        gl=(args.gl or None),
        limit_tables=max(1, args.limit_tables),
        topk_rank=max(10, args.topk_rank),
    )

    # initialize logging once for this process (pretty console + JSON file)
    try:
        setup_logging()
        log.info("pipeline start", extra={"query": cfg.query})
    except Exception:
        pass

    try:
        out = run_once(cfg)
        print(f"\n[OK] Pipeline finished. Open report:\n  {out}\n")
    except Exception as e:
        print("\n[FAIL] Pipeline failed.")
        print(e)
        traceback.print_exc()
        sys.exit(2)
# C:\CiteVizor\pipeline\run_once.py
# -*- coding: utf-8 -*-
"""
CiteVizor - End-to-end one-shot pipeline runner

Stages:
  1) Retrieval (Serper Scholar) -> Candidates
  2) Download (PDF-first) -> storage/docs/<doc_id>/*
  3) Parse (pdf_structure + table_extract + figure_extract) -> storage/evidence/<doc_id>/pack.json
  4) Align & rank (BM25 + boosts) -> storage/evidence/_rank/<run_id>.json
  5) Citation map -> storage/evidence/_rank/<run_id>.citations.json
  6) Engineer summary (LLM via vLLM) -> storage/evidence/_rank/<run_id>.summary.json
  7) Chart codegen (LLM) -> storage/renders/<doc_id>/charts/*.png
  8) HTML layout -> storage/reports/<run_id>.html

Model & serving:
  - Uses your local vLLM OpenAI-compatible server with model:
      Qwen/Qwen2.5-14B-Instruct-AWQ  (HF: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ)
  - No new env keys required; reuse existing citevizor.env settings.

Run example:
  python pipeline/run_once.py --query "latest design trends of PV encapsulant for seaside corrosion" --lang en --k 5
  python pipeline/run_once.py --query "海边光伏组件封装材料最新趋势" --lang zh --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Core schemas/paths
from schemas import (
    SearchPlan, YearRange, STORAGE_DIR, EVIDENCE_DIR, RENDERS_DIR, REPORTS_DIR
)

# Retrieval
from retrieval.serper_scholar import SerperScholarClient

# Fetch
from fetch.downloader import DownloadConfig, Downloader

# Parse
from parse.pdf_structure import PDFStructureParser
from parse.table_extract import TableExtractConfig, TableExtractor
from parse.figure_extract import FigureExtractConfig, FigureExtractor

# Align
from align.evidence_rank import EvidenceRankConfig, EvidenceRanker
from align.citation_map import CitationMapperConfig, CitationMapper

# LLM (vLLM client + summarizer + chart codegen)
from llm.client_vllm import VLLMConfig, VLLMClient
from llm.summarize_engineer import SummarizerConfig, EngineerSummarizer
from llm.chart_codegen import CodegenConfig, ChartCodegen

# Report
from report.layout import ReportLayout, LayoutSpec

# --- unified logging (no behavior change) ---
from infra.logging import setup_logging, get_logger, set_run_id
log = get_logger("pipeline")

# ------------------------------ helpers ---------------------------------------

def _project_root() -> Path:
    # .../CiteVizor/pipeline/run_once.py -> parents[1] == project root
    return Path(__file__).resolve().parents[1]

def _log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}")

def _warn(stage: str, msg: str) -> None:
    print(f"[{stage}][WARN] {msg}")

def _err(stage: str, msg: str) -> None:
    print(f"[{stage}][ERROR] {msg}")


def _make_plan(user_query: str, lang: str, years: Optional[str], k_top: int) -> SearchPlan:
    """
    Lightweight query planner: expands the user query with review/meta-analysis variations and
    keeps k_top for retrieval aggregation. Year range can be "2020-2025" or ">=2021".
    """
    q = user_query.strip()
    expansions = [
        q,
        f"{q} review",
        f"{q} meta-analysis",
        f"{q} survey",
    ]
    if lang.lower().startswith("zh"):
        expansions += [f"{q} 综述", f"{q} 系统综述", f"{q} 元分析"]
    if lang.lower().startswith("ja"):
        expansions += [f"{q} レビュー", f"{q} サーベイ"]

    yr = None
    if years:
        years = years.strip()
        try:
            if "-" in years:
                a, b = years.split("-", 1)
                yr = YearRange(start=int(a), end=int(b))
            elif years.startswith(">="):
                yr = YearRange(start=int(years[2:]), end=None)
            elif years.startswith("<="):
                yr = YearRange(start=None, end=int(years[2:]))
        except Exception:
            yr = None

    plan = SearchPlan(
        queries=[e for e in expansions if e.strip()],
        k_top=max(5, k_top),
        filters={"years": yr.dict() if yr else {}}
    )
    return plan


# ------------------------------ pipeline --------------------------------------

@dataclass
class PipelineArgs:
    query: str
    lang: str
    years: Optional[str]
    k: int
    gl: Optional[str]
    limit_tables: int
    topk_rank: int

def run_once(cfg: PipelineArgs) -> Path:
    root = _project_root()
    # Ensure base dirs exist
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Retrieval ----------
    _log("Retrieval", "building search plan…")
    plan = _make_plan(cfg.query, cfg.lang, cfg.years, k_top=max(10, cfg.k * 3))

    _log("Retrieval", f"queries={len(plan.queries)}  k_top={plan.k_top}")
    retr = SerperScholarClient(api_key=None, project_root=root)
    cands = retr.search_with_plan(
        plan,
        limit_per_query=8,
        lang=("en" if cfg.lang == "en" else "en"),
        gl=cfg.gl
    )
    if not cands:
        raise RuntimeError("No candidates returned from Serper.")
    cands = cands[: cfg.k]
    _log("Retrieval", f"picked {len(cands)} candidates")

    # ---------- 2) Download ----------
    dl = Downloader(DownloadConfig.from_env(root))
    fetched_ids: List[str] = []
    for i, cand in enumerate(cands, 1):
        try:
            fd = dl.fetch_candidate(cand)
            fetched_ids.append(fd.doc_id)
            _log("Download", f"{i}/{len(cands)} OK -> {fd.doc_id}")
        except Exception as e:
            _warn("Download", f"skip '{cand.title}': {e}")

    if not fetched_ids:
        raise RuntimeError("No PDFs downloaded; aborting.")

    # ---------- 3) Parse (PDF -> evidence pack) ----------
    parser = PDFStructureParser.from_env(root)
    tcfg = TableExtractConfig.from_env(root)
    tex = TableExtractor(tcfg)
    fcfg = FigureExtractConfig.from_env(root)
    fex = FigureExtractor(fcfg)

    for j, doc_id in enumerate(fetched_ids, 1):
        try:
            from schemas import Candidate, DocumentPaths, FetchedDoc
            # Reconstruct minimal FetchedDoc from paths/meta for parser APIs
            pdf_path = STORAGE_DIR / "docs" / doc_id / f"{doc_id}.pdf"
            meta_path = STORAGE_DIR / "docs" / doc_id / f"{doc_id}.meta.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            cand = Candidate(
                title=meta.get("title") or doc_id,
                url=meta.get("url") or "https://example.org",
                year=meta.get("year"),
                source="scholar",
                score=1.0,
                doi=meta.get("doi"),
            )
            paths = DocumentPaths(base_dir=pdf_path.parent, pdf_path=pdf_path, meta_json=meta_path)
            fetched = FetchedDoc(doc_id=doc_id, candidate=cand, paths=paths)

            pack = parser.parse_pdf(fetched, persist=True)
            _log("Parse", f"{j}/{len(fetched_ids)} paragraphs={len(pack.paragraphs)}")

            pack = tex.extract(fetched, persist=True)
            _log("Parse", f"tables so far={len(pack.tables)}")

            pack = fex.extract(fetched, persist=True)
            _log("Parse", f"figures so far={len(pack.figures)}")
        except Exception as e:
            _warn("Parse", f"{doc_id}: {e}")

    # ---------- 4) Align & rank ----------
    rcfg = EvidenceRankConfig.from_env(root)
    ranker = EvidenceRanker(rcfg)
    try:
        rank_json = ranker.rank(
            query=cfg.query,
            doc_ids=fetched_ids,
            sub_queries=plan.queries,
            top_k=cfg.topk_rank
        )
    except Exception as e:
        raise RuntimeError(f"Ranking failed: {e}")
    run_id = rank_json.get("run_id")
    _log("Rank", f"run_id={run_id} items={len(rank_json.get('items', []))}")

    # ---- bind run_id to logging context (for structured logs) ----
    try:
        if run_id:
            set_run_id(str(run_id))
            log.info("rank complete", extra={"items": len(rank_json.get("items", []))})
    except Exception:
        # logging must never break the pipeline
        pass

    # ---------- 5) Citation map ----------
    mapper = CitationMapper(CitationMapperConfig.from_env(root))
    payload = mapper.build(rank_json)
    from align.citation_map import _save_citations  # reuse helper
    map_path = _save_citations(run_id, payload)
    _log("Citations", f"map -> {map_path.name} refs={len(payload.get('refs', []))}")

    # ---------- 6) Summarize (LLM via vLLM) ----------
    vcfg = VLLMConfig.from_env(root)
    vclient = VLLMClient(vcfg)
    scfg = SummarizerConfig.from_env(root)
    summarizer = EngineerSummarizer(scfg, vclient)
    tb = summarizer.summarize(payload)
    _log("Summarize", f"highlights={len(tb.highlights)} figure_captions={len(tb.figure_captions)}")

    # ---------- 7) Chart codegen (per doc) ----------
    ccfg = CodegenConfig.default(root)
    codegen = ChartCodegen(ccfg, vclient)
    for doc_id in fetched_ids:
        try:
            codes = codegen.generate_for_doc(doc_id, limit=cfg.limit_tables)
            _log("Charts", f"{doc_id}: generated {len(codes)} charts")
        except Exception as e:
            _warn("Charts", f"{doc_id}: {e}")

    # ---------- 8) Layout (HTML) ----------
    spec = LayoutSpec(
        title=f"CiteVizor Report – {cfg.query[:64]}",
        lang=cfg.lang,
        audience="engineer",
        max_visuals=18
    )
    rl = ReportLayout(spec)
    out_html = rl.build(run_id)
    _log("Report", f"HTML -> {out_html}")
    return out_html


# ------------------------------ CLI -------------------------------------------

def _parse_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CiteVizor one-shot pipeline runner")
    ap.add_argument("--query", required=True, help="User query (research question)")
    ap.add_argument("--lang", default="en", help="en|zh|ja (affects query expansions and summary language)")
    ap.add_argument("--years", default="", help='Optional year filter like "2021-2025" or ">=2020"')
    ap.add_argument("--k", type=int, default=5, help="Number of top candidates to download")
    ap.add_argument("--gl", default="", help="Optional country code for Serper (e.g., jp, hk)")
    ap.add_argument("--limit-tables", type=int, default=6, help="Max tables to chart per document")
    ap.add_argument("--topk-rank", type=int, default=50, help="Top-K evidence items to keep in ranking")
    return ap

if __name__ == "__main__":
    args = _parse_args().parse_args()
    cfg = PipelineArgs(
        query=args.query,
        lang=args.lang,
        years=(args.years or None),
        k=max(1, args.k),
        gl=(args.gl or None),
        limit_tables=max(1, args.limit_tables),
        topk_rank=max(10, args.topk_rank),
    )

    # initialize logging once for this process (pretty console + JSON file)
    try:
        setup_logging()
        log.info("pipeline start", extra={"query": cfg.query})
    except Exception:
        pass

    try:
        out = run_once(cfg)
        print(f"\n[OK] Pipeline finished. Open report:\n  {out}\n")
    except Exception as e:
        print("\n[FAIL] Pipeline failed.")
        print(e)
        traceback.print_exc()
        sys.exit(2)
