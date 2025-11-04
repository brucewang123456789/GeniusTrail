# C:\CiteVizor\main.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Unified CLI entrypoint

Subcommands
  plan     : Build and persist a SearchPlan from a user query.
  run      : Execute the end-to-end pipeline once (retrieval->parse->rank->summarize->charts->layout).
  export   : Export a finished run to HTML or PPTX.
  qa       : Offline consistency checks over a run's artifacts.
  health   : Quick health check (vLLM server; env hints).

Notes
  - No new env keys are required here; modules read citevizor.env as needed.
  - LLM model used elsewhere remains Qwen/Qwen2.5-14B-Instruct-AWQ on vLLM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Core paths/contracts
from schemas import EVIDENCE_DIR, STORAGE_DIR, REPORTS_DIR

# Planner
from planner.plan_query import QueryPlanner, PlannerConfig

# Pipeline
from pipeline.run_once import run_once as pipeline_run_once, PipelineArgs

# Exporters
from report.exporters import export_html, export_pptx

# QA
try:
    from qa.consistency_checks import run_checks as qa_run_checks, _save_report as qa_save_report  # type: ignore
except Exception:
    qa_run_checks = None  # type: ignore
    qa_save_report = None  # type: ignore

# vLLM health (optional)
try:
    from llm.client_vllm import VLLMClient, VLLMConfig  # type: ignore
except Exception:
    VLLMClient = None  # type: ignore
    VLLMConfig = None  # type: ignore

ENV_FILE = "citevizor.env"


# ----------------------------- util -------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parent

def _rank_dir() -> Path:
    d = EVIDENCE_DIR / "_rank"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_env(project_root: Path) -> Dict[str, str]:
    env_path = project_root / ENV_FILE
    out: Dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out

def _print_json(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


# ----------------------------- commands ---------------------------------------

def cmd_plan(args: argparse.Namespace) -> None:
    root = _project_root()
    cfg = PlannerConfig.from_env(root)
    if args.use_llm:
        cfg.use_llm = True
    planner = QueryPlanner(cfg)
    plan, path = planner.build_plan(
        user_query=args.query,
        lang=args.lang,
        years=(args.years or None),
        k_top=args.k_top,
    )
    print(f"[OK] SearchPlan -> {path}")
    print(f"  plan_id={plan.plan_id}  queries={len(plan.queries)}  k_top={plan.k_top}")
    for i, q in enumerate(plan.queries, 1):
        print(f"   {i:02d}. {q}")


def cmd_run(args: argparse.Namespace) -> None:
    cfg = PipelineArgs(
        query=args.query,
        lang=args.lang,
        years=(args.years or None),
        k=max(1, args.k),
        gl=(args.gl or None),
        limit_tables=max(1, args.limit_tables),
        topk_rank=max(10, args.topk_rank),
    )
    out_html = pipeline_run_once(cfg)
    print(f"[OK] Pipeline finished -> {out_html}")

    # Optional export to PPTX in the same shot
    if args.export in ("pptx", "both"):
        rid = Path(out_html).stem  # run_id matches file name without extension
        pptx = export_pptx(run_id=rid, title=args.title or f"CiteVizor Report – {args.query[:64]}", lang=args.lang, widescreen=(not args.classic43))
        print(f"[OK] PPTX -> {pptx}")
    if args.export in ("html", "both") and args.title:
        # Re-render HTML with a custom title (layout already made one; this overwrites)
        rid = Path(out_html).stem
        html = export_html(run_id=rid, title=args.title, lang=args.lang, max_visuals=args.max_visuals)
        print(f"[OK] HTML (retitled) -> {html}")


def cmd_export(args: argparse.Namespace) -> None:
    if args.fmt == "html":
        out = export_html(run_id=args.run_id, title=args.title, lang=args.lang, max_visuals=args.max_visuals)
        print(f"[OK] HTML -> {out}")
    elif args.fmt == "pptx":
        out = export_pptx(run_id=args.run_id, title=args.title, lang=args.lang, widescreen=(not args.classic43))
        print(f"[OK] PPTX -> {out}")
    else:
        # both
        out1 = export_html(run_id=args.run_id, title=args.title, lang=args.lang, max_visuals=args.max_visuals)
        out2 = export_pptx(run_id=args.run_id, title=args.title, lang=args.lang, widescreen=(not args.classic43))
        print(f"[OK] HTML -> {out1}\n[OK] PPTX -> {out2}")


def cmd_qa(args: argparse.Namespace) -> None:
    if qa_run_checks is None:
        raise RuntimeError("qa.consistency_checks not available.")
    rep = qa_run_checks(args.run_id, lang=args.lang)
    if qa_save_report:
        out = qa_save_report(rep)
        print(f"[OK] QA report -> {out}")
    else:
        # fallback to simple save
        out = _rank_dir() / f"{args.run_id}.qa.json"
        payload = {
            "run_id": rep.run_id,
            "query": rep.query,
            "stats": rep.stats,
            "artifacts": rep.artifacts,
            "findings": [f.__dict__ for f in rep.findings],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] QA report -> {out}")
    print(f"Stats: errors={rep.stats['errors']} warnings={rep.stats['warnings']} infos={rep.stats['infos']}")
    if rep.stats["errors"] > 0:
        sys.exit(2)


def cmd_health(args: argparse.Namespace) -> None:
    root = _project_root()
    env = _load_env(root)

    print("[Health] citevizor.env presence:", "OK" if (root / ENV_FILE).exists() else "missing")
    # vLLM check
    if VLLMClient and VLLMConfig:
        try:
            vc = VLLMClient(VLLMConfig.from_env(root))
            models = vc.health_check()
            ids = [m.get("id") for m in models.get("data", [])]
            print("[Health] vLLM /models OK:", ids[:5])
        except Exception as e:
            print("[Health] vLLM FAILED:", e)
    else:
        print("[Health] vLLM client not importable; skip.")

    # Minimal hints for retrieval keys
    if env.get("SERPER_API_KEY"):
        print("[Health] Serper key: OK")
    else:
        print("[Health] Serper key: missing (retrieval.serper_scholar will fail without it)")

    # Storage dirs
    for d in (STORAGE_DIR, EVIDENCE_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print("[Health] storage dirs:", STORAGE_DIR, EVIDENCE_DIR, REPORTS_DIR)


# ----------------------------- CLI parser -------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="CiteVizor", description="CiteVizor unified CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # plan
    sp = sub.add_parser("plan", help="Build a SearchPlan from a query")
    sp.add_argument("--query", required=True, help="User research question")
    sp.add_argument("--lang", default="auto", help="auto|en|zh|ja")
    sp.add_argument("--years", default="", help='e.g., "2019-2024", ">=2021", "last 5 years", "近三年"')
    sp.add_argument("--k-top", type=int, default=12, help="k_top for retrieval")
    sp.add_argument("--use-llm", action="store_true", help="Enable LLM-based query expansions")
    sp.set_defaults(func=cmd_plan)

    # run
    sr = sub.add_parser("run", help="Run the end-to-end pipeline once")
    sr.add_argument("--query", required=True, help="User research question")
    sr.add_argument("--lang", default="en", help="en|zh|ja")
    sr.add_argument("--years", default="", help='Optional year filter: "2021-2025" or ">=2020"')
    sr.add_argument("--k", type=int, default=5, help="Top candidates to download")
    sr.add_argument("--gl", default="", help="Serper country code (e.g., jp, hk)")
    sr.add_argument("--limit-tables", type=int, default=6, help="Max tables to chart per document")
    sr.add_argument("--topk-rank", type=int, default=50, help="Top-K evidence items for ranking")
    # export knobs
    sr.add_argument("--export", choices=["none", "html", "pptx", "both"], default="html", help="Auto-export after pipeline")
    sr.add_argument("--title", default="", help="Optional report title override")
    sr.add_argument("--max-visuals", type=int, default=18, help="HTML visual cap when re-exporting")
    sr.add_argument("--classic43", action="store_true", help="PPTX: use classic 4:3")
    sr.set_defaults(func=cmd_run)

    # export
    se = sub.add_parser("export", help="Export an existing run")
    se.add_argument("--run-id", required=True, help="Ranking run_id")
    se.add_argument("--fmt", choices=["html", "pptx", "both"], required=True, help="Export format")
    se.add_argument("--title", default="", help="Report title")
    se.add_argument("--lang", default="en", help="en|zh|ja")
    se.add_argument("--max-visuals", type=int, default=18, help="HTML: cap visuals")
    se.add_argument("--classic43", action="store_true", help="PPTX: use classic 4:3")
    se.set_defaults(func=cmd_export)

    # qa
    sq = sub.add_parser("qa", help="Run offline QA checks")
    sq.add_argument("--run-id", required=True, help="Ranking run_id")
    sq.add_argument("--lang", default="en", help="Expected language for highlights")
    sq.set_defaults(func=cmd_qa)

    # health
    sh = sub.add_parser("health", help="Environment and vLLM server health")
    sh.set_defaults(func=cmd_health)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
