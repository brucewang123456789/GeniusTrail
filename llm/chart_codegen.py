# C:\CiteVizor\llm\chart_codegen.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Chart code generator (tables -> matplotlib PNG)

Workflow:
  1) Load tables from storage/evidence/<doc_id>/pack.json.
  2) For each table, build a compact "table profile" (columns, dtypes, sample, ranges).
  3) Ask local vLLM (OpenAI-compatible) to generate a SINGLE function:
       def render_chart(csv_path, out_path):  # matplotlib-only, one figure, no seaborn/styles
  4) Static checks on the code (forbidden imports/calls), then sandbox-exec to produce a PNG.
  5) Persist artifacts:
       - PNG under storage/renders/<doc_id>/charts/chart_<table_idx>.png
       - script under storage/renders/<doc_id>/charts/scripts/<evidence_id>.py
       - summary JSON under storage/renders/<doc_id>/charts/_charts.json

Notes:
  * Uses Qwen/Qwen2.5-14B-Instruct-AWQ via vLLM by default (see client_vllm.py).
  * English-only comments by request.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Headless matplotlib backend for any subprocess-free rendering
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from schemas import (
    EVIDENCE_DIR,
    RENDERS_DIR,
    EvidenceType,
    ChartCode,
)
from llm.client_vllm import VLLMClient, VLLMConfig


# ----------------------------- IO helpers -------------------------------------

def _pack_path(doc_id: str) -> Path:
    return EVIDENCE_DIR / doc_id / "pack.json"

def _charts_dir(doc_id: str) -> Path:
    d = RENDERS_DIR / doc_id / "charts"
    (d / "scripts").mkdir(parents=True, exist_ok=True)
    return d

def _charts_manifest(doc_id: str) -> Path:
    return _charts_dir(doc_id) / "_charts.json"

def _read_pack(doc_id: str) -> Dict[str, Any]:
    p = _pack_path(doc_id)
    if not p.exists():
        raise FileNotFoundError(f"pack.json not found for doc_id={doc_id}: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def _append_manifest(doc_id: str, record: Dict[str, Any]) -> None:
    mf = _charts_manifest(doc_id)
    data = []
    if mf.exists():
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            data = []
    data.append(record)
    _write_text(mf, json.dumps(data, ensure_ascii=False, indent=2))


# ----------------------------- Table profiling --------------------------------

def _profile_table(csv_path: Path, sample_rows: int = 12) -> Dict[str, Any]:
    """
    Build a compact profile to guide the LLM:
      - column names and inferred dtypes
      - numeric columns and basic stats
      - a small head sample (rows x cols)
    """
    df = pd.read_csv(csv_path, nrows=2000)  # cap read for speed
    head = df.head(sample_rows)
    # Basic dtype mapping
    dtypes = {c: str(head[c].dtype) for c in head.columns}
    # Numeric stats
    numeric_cols = [c for c in head.columns if pd.api.types.is_numeric_dtype(head[c])]
    stats = {}
    for c in numeric_cols:
        s = head[c].dropna()
        if s.empty:
            continue
        stats[c] = {
            "min": float(np.nanmin(s)),
            "max": float(np.nanmax(s)),
            "mean": float(np.nanmean(s)),
            "std": float(np.nanstd(s)),
            "n": int(s.shape[0]),
        }
    # Sample view (string)
    with csv_path.open("r", encoding="utf-8-sig") as f:
        rdr = csv.reader(f)
        rows = []
        for i, row in enumerate(rdr):
            if i >= sample_rows:
                break
            rows.append(row[:10])
    sample_str = "\n".join([" | ".join(r) for r in rows])

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "dtypes": dtypes,
        "numeric_stats": stats,
        "sample": sample_str[:2000],
        "filename": csv_path.name,
    }


# ----------------------------- LLM prompt -------------------------------------

def _system_prompt() -> str:
    return (
        "You are a code generator that writes a SINGLE matplotlib chart function for engineers.\n"
        "Hard requirements:\n"
        "1) Output ONLY Python code; define exactly one function:\n"
        "   def render_chart(csv_path, out_path):\n"
        "2) Use matplotlib.pyplot as plt (NO seaborn, NO styles, NO multiple subplots).\n"
        "3) Read the CSV from 'csv_path' using pandas; do not assume column order—detect headers.\n"
        "4) Create ONE figure with clear axes labels/units from headers; rotate x labels if crowded.\n"
        "5) Prefer sensible chart type (line/bar/scatter) from the provided profile; if unsure, fall back to bar.\n"
        "6) Tight layout, legend if multiple series, no plt.show().\n"
        "7) Save with: plt.savefig(out_path, dpi=200, bbox_inches='tight'); then plt.close().\n"
        "8) No filesystem/network/OS access beyond reading 'csv_path' and writing 'out_path'.\n"
        "9) Do not set colors/styles; rely on matplotlib defaults.\n"
    )

def _user_prompt(table_profile: Dict[str, Any]) -> str:
    prof = json.dumps(table_profile, ensure_ascii=False, indent=2)
    return (
        "Goal: produce a single informative chart for an engineering report from the table profile below.\n"
        "Pick the best chart type (line, bar, or scatter) based on dtypes/ranges; if time-series, parse the x-axis smartly.\n"
        "Table profile:\n"
        f"{prof}\n"
        "Return ONLY valid Python code for:\n"
        "def render_chart(csv_path, out_path):\n"
    )


# ----------------------------- Code guards ------------------------------------

_FORBIDDEN_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+shutil\b",
    r"\bfrom\s+os\b",
    r"\bopen\s*\(",
    r"\brequests\b",
    r"\burllib\b",
    r"\bhttpx\b",
    r"\bsocket\b",
    r"\bplt\.show\s*\(",
]

def _static_checks(code: str) -> None:
    # quick hygiene
    if "def render_chart" not in code:
        raise ValueError("The model did not define 'render_chart'.")
    for pat in _FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            raise ValueError(f"Forbidden usage matched: {pat}")

def _extract_code_block(text: str) -> str:
    """
    Extract python code from ```python ... ``` or ``` ... ```; fallback to whole text.
    """
    m = re.search(r"```python\s+([\s\S]*?)```", text, re.IGNORECASE)
    if not m:
        m = re.search(r"```\s*([\s\S]*?)```", text)
    return m.group(1).strip() if m else text.strip()


# ----------------------------- Sandbox exec -----------------------------------

def _exec_render(code: str, csv_path: Path, out_path: Path) -> None:
    """
    Execute the generated code with a controlled namespace.
    We pre-import pandas/numpy/matplotlib and provide csv_path/out_path variables.
    """
    # Build safe globals: allow imports already in code for pandas/matplotlib/numpy; forbid OS ops by static check.
    globals_ns: Dict[str, Any] = {
        "__name__": "__chart_codegen__",
        "pd": pd,
        "np": np,
        "matplotlib": matplotlib,
        "plt": plt,
    }
    locals_ns: Dict[str, Any] = {}

    exec(code, globals_ns, locals_ns)  # may define render_chart
    fn = locals_ns.get("render_chart") or globals_ns.get("render_chart")
    if not callable(fn):
        raise ValueError("No callable 'render_chart' found after execution.")
    # Call the function
    fn(str(csv_path), str(out_path))
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Chart was not saved; missing or empty output file.")


# ----------------------------- Generator core ---------------------------------

@dataclass
class CodegenConfig:
    project_root: Path
    limit_per_doc: int = 6   # max number of tables to chart per doc

    @classmethod
    def default(cls, project_root: Path) -> "CodegenConfig":
        return cls(project_root=project_root, limit_per_doc=6)

class ChartCodegen:
    def __init__(self, cfg: CodegenConfig, llm: VLLMClient):
        self.cfg = cfg
        self.llm = llm

    def generate_for_doc(self, doc_id: str, limit: Optional[int] = None, overwrite: bool = False) -> List[ChartCode]:
        pack = _read_pack(doc_id)
        tables = [t for t in pack.get("tables", []) if t.get("csv_path")]
        if not tables:
            return []

        charts_dir = _charts_dir(doc_id)
        max_n = min(len(tables), limit or self.cfg.limit_per_doc)

        results: List[ChartCode] = []
        for idx, t in enumerate(tables[:max_n], 1):
            ev_id = t.get("id")
            csv_path = Path(t.get("csv_path"))
            if not csv_path.exists():
                continue

            png_out = charts_dir / f"chart_{idx:02d}.png"
            script_out = charts_dir / "scripts" / f"{ev_id or 'table'}_{idx:02d}.py"
            if png_out.exists() and script_out.exists() and not overwrite:
                # already generated
                results.append(ChartCode(evidence_id=ev_id, code=script_out.read_text(encoding="utf-8")))
                continue

            # Build table profile & prompt
            profile = _profile_table(csv_path)
            messages = [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(profile)},
            ]
            # LLM call
            res = self.llm.chat(messages, max_tokens=900)
            code = _extract_code_block(res.text)

            # Guardrails and exec
            try:
                _static_checks(code)
            except Exception as e:
                # Attach reason and continue (skip this table)
                err_msg = f"# Rejected code for evidence={ev_id}: {e}\n"
                _write_text(script_out, err_msg + code)
                continue

            try:
                _exec_render(code, csv_path, png_out)
            except Exception as e:
                # Persist failure for debugging
                tb = traceback.format_exc()
                _write_text(script_out, f"# Execution error: {e}\n# Traceback:\n{tb}\n\n" + code)
                continue

            # Persist script
            _write_text(script_out, code)

            # Record
            cc = ChartCode(evidence_id=ev_id, code=code, note=f"Saved to {png_out.name}")
            results.append(cc)

            # Manifest append
            _append_manifest(doc_id, {
                "evidence_id": ev_id,
                "csv": str(csv_path),
                "png": str(png_out),
                "script": str(script_out),
            })

        return results


# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor chart code generator")
    ap.add_argument("--doc-id", required=True, help="Document id (folder name under storage/evidence/<doc_id>)")
    ap.add_argument("--limit", type=int, default=0, help="Max number of tables to chart")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate charts even if files exist")
    return ap.parse_args()

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    vcfg = VLLMConfig.from_env(project_root)
    client = VLLMClient(vcfg)
    cfg = CodegenConfig.default(project_root)
    gen = ChartCodegen(cfg, client)

    args = _parse_args()
    lim = args.limit if args.limit and args.limit > 0 else None

    codes = gen.generate_for_doc(args.doc_id, limit=lim, overwrite=bool(args.overwrite))
    out_dir = _charts_dir(args.doc_id)
    print(f"[OK] Generated {len(codes)} charts → {out_dir}")
    for i, c in enumerate(codes, 1):
        print(f"  #{i:02d} evidence={c.evidence_id} script-hash={c.script_hash()}")
