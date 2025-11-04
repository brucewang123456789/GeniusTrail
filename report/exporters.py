# C:\CiteVizor\report\exporters.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Exporters dispatcher (single stable entrypoint)

Responsibilities
- Provide a consistent CLI and Python API to export a run into different formats.
- Delegate actual rendering to specialized modules:
    * HTML -> report.layout.ReportLayout
    * PPTX -> report.export_pptx.build_pptx
- Do minimal preflight checks (citations.json presence), uniform logging, and
  keep this module thin so business logic stays in submodules.

This file does NOT use any LLM and does NOT require env variables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from schemas import EVIDENCE_DIR  # for _rank dir resolution
from report.layout import ReportLayout, LayoutSpec
from report.export_pptx import build_pptx


# ----------------------------- helpers ----------------------------------------

def _rank_dir() -> Path:
    d = EVIDENCE_DIR / "_rank"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _citations_path(run_id: str) -> Path:
    return _rank_dir() / f"{run_id}.citations.json"

def _summary_path(run_id: str) -> Path:
    return _rank_dir() / f"{run_id}.summary.json"

def _preflight_or_raise(run_id: str) -> None:
    """Ensure core artifacts exist before delegating to exporters."""
    cit = _citations_path(run_id)
    if not cit.exists():
        raise FileNotFoundError(
            f"citations json not found for run_id={run_id}: {cit}\n"
            "Run the pipeline or citation_map step first."
        )
    # summary is optional; exporters handle fallback if missing


# ----------------------------- public API -------------------------------------

def export_html(
    run_id: str,
    title: str = "",
    lang: str = "en",
    audience: str = "engineer",
    max_visuals: int = 18,
) -> Path:
    """
    Export an HTML report for the given run_id.

    Returns:
        Path to the generated HTML file under storage/reports/.
    """
    _preflight_or_raise(run_id)
    spec = LayoutSpec(
        title=title or "CiteVizor Report",
        lang=lang,
        audience=audience,
        max_visuals=max_visuals,
    )
    rl = ReportLayout(spec)
    return rl.build(run_id)


def export_pptx(
    run_id: str,
    title: str = "",
    lang: str = "en",
    widescreen: bool = True,
) -> Path:
    """
    Export a PPTX deck for the given run_id.

    Returns:
        Path to the generated PPTX file under storage/reports/.
    """
    _preflight_or_raise(run_id)
    return build_pptx(run_id, title=(title or None), lang=lang, widescreen=widescreen)


# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor exporters dispatcher")
    ap.add_argument("--run-id", required=True, help="Ranking run_id (base name of citations/summary json)")
    ap.add_argument("--fmt", required=True, choices=["html", "pptx"], help="Output format")
    ap.add_argument("--title", default="", help="Report title")
    ap.add_argument("--lang", default="en", help="Report language: en|zh|ja")
    # HTML-only options
    ap.add_argument("--audience", default="engineer", help="HTML layout audience hint")
    ap.add_argument("--max-visuals", type=int, default=18, help="HTML: cap number of visuals")
    # PPTX-only options
    ap.add_argument("--classic43", action="store_true", help="PPTX: use classic 4:3 instead of 16:9")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.fmt == "html":
        out = export_html(
            run_id=args.run_id,
            title=args.title,
            lang=args.lang,
            audience=args.audience,
            max_visuals=args.max_visuals,
        )
        print(f"[OK] HTML -> {out}")
    else:
        out = export_pptx(
            run_id=args.run_id,
            title=args.title,
            lang=args.lang,
            widescreen=(not args.classic43),
        )
        print(f"[OK] PPTX -> {out}")
