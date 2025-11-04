# C:\CiteVizor\simulate_cli.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Interactive CLI simulator for end-to-end backend

Purpose
- Let a user (or PM) type a research query and quickly run the whole backend pipeline.
- Keeps citevizor_backend.py untouched; this script only gathers params and loops.
- Default demo uses: "Attention Is All You Need" (2017, Transformers).

No extra dependencies. Safe to run on Windows/Linux/macOS.
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Optional

# unified logging (graceful fallback)
try:
    from infra.logging import setup_logging, get_logger
except Exception:
    def setup_logging(*a, **k): return None
    def get_logger(name=None):
        class _L:
            def info(self,*a,**k): pass
            def warning(self,*a,**k): pass
            def error(self,*a,**k): pass
        return _L()

LOG = get_logger("cli")

# import backend contracts
try:
    from citevizor_backend import BackendArgs, run_backend  # type: ignore
except Exception as e:
    print("[FATAL] Cannot import citevizor_backend:", e)
    sys.exit(2)


def _ask(prompt: str, default: Optional[str] = None) -> str:
    s = input(f"{prompt} " + (f"[{default}] " if default else ""))
    s = s.strip()
    return s if s else (default or "")

def _ask_int(prompt: str, default: int) -> int:
    s = _ask(prompt, str(default))
    try:
        return max(0, int(s))
    except Exception:
        return default

def _ask_bool(prompt: str, default: bool) -> bool:
    s = _ask(prompt + " (y/n)", "y" if default else "n").lower()
    return s in ("y", "yes", "1", "true", "t")

def _banner() -> None:
    print("\n==============================================")
    print("  CiteVizor – Interactive Backend Simulator")
    print("==============================================\n")

def _demo_query() -> str:
    # default demo focusing on the 2017 Transformer paper
    return ("Key contributions and follow-up work of 'Attention Is All You Need' (2017) "
            "and the evolution of Transformer architectures in NLP/CV")

def run_once_interactive() -> None:
    _banner()
    # defaults oriented for a quick but meaningful run
    q_default = _demo_query()
    query = _ask("Enter your research question:", q_default)
    lang = _ask("Language (en|zh|ja):", "en")
    years = _ask("Year filter (e.g., 2018-2024, >=2019) [empty=no filter]:", "")
    gl = _ask("Serper geolocation (e.g., us, jp, hk) [empty=auto]:", "")
    k_scholar = _ask_int("Top N from Google Scholar:", 6)
    k_web = _ask_int("Top N from general web/patents/news/universities:", 4)
    topk_rank = _ask_int("Evidence top-K after ranking:", 60)
    limit_tables = _ask_int("Max tables to chart per document:", 6)
    enable_ocr = _ask_bool("Enable OCR fallback for scanned PDFs?", False)
    export_pptx = _ask_bool("Export PPTX in addition to HTML?", True)
    serve_preview = _ask_bool("Start preview server after the run?", False)

    # echo config
    print("\n----- Config Summary -----")
    print(f"query         : {query}")
    print(f"lang/years    : {lang} / {years or '(none)'}")
    print(f"k_scholar/web : {k_scholar} / {k_web}")
    print(f"rank K        : {topk_rank}")
    print(f"tables limit  : {limit_tables}")
    print(f"OCR/PPTX/UI   : {enable_ocr} / {export_pptx} / {serve_preview}")
    print("--------------------------\n")

    cfg = BackendArgs(
        query=query,
        lang=lang,
        years=(years or None),
        gl=(gl or None),
        k_scholar=max(1, k_scholar),
        k_web=max(0, k_web),
        topk_rank=max(10, topk_rank),
        limit_tables=max(1, limit_tables),
        enable_ocr=enable_ocr,
        serve_preview=serve_preview,
        export_pptx_flag=export_pptx,
    )

    try:
        out = run_backend(cfg)
        print("\n=== PIPELINE FINISHED ===")
        print(f"run_id : {out.get('run_id')}")
        print(f"HTML   : {out.get('html')}")
        if out.get("pptx"):     print(f"PPTX   : {out.get('pptx')}")
        if out.get("manifest"): print(f"Manifest: {out.get('manifest')}")
        print()
        if not serve_preview:
            print("Tip: run with 'Start preview server = y' next time to review results at http://127.0.0.1:8787")
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except Exception as e:
        LOG.error("cli.run_error", extra={"err": str(e)[:500]})
        print("\n[FAIL] See logs for details.")
        raise

def main() -> int:
    # one-time logging setup for CLI
    try:
        setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
        LOG.info("cli.start")
    except Exception:
        pass

    while True:
        run_once_interactive()
        again = _ask_bool("Run another experiment?", False)
        if not again:
            break
    print("Bye.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
