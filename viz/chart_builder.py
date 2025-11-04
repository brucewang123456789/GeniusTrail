# /workspace/viz/chart_builder.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Chart Builder (deterministic, LLM-optional)

Enhancements vs previous:
  - Idempotent: skip duplicates via stable chart_id hash.
  - Failure logging: write JSONL to _fail.jsonl instead of raising.
  - Figure passthrough fallback: when no/insufficient tables, copy figures into charts
    to satisfy "visual-first" requirement.

Public API (stable)
  - class ChartBuilder:
        .build_for_doc(doc_id: str, limit_per_doc:int=6) -> Path
        .build_for_run(run_id: str, limit_per_doc:int=6) -> List[Path]
        .run(run_id: Optional[str]=None, doc_id: Optional[str]=None, limit_per_doc:int=6)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype  # noqa: E402

from schemas import EVIDENCE_DIR, RENDERS_DIR  # project dirs

# Optional LLM-based codegen (metadata-only; rendering stays deterministic)
try:
    from llm.chart_codegen import suggest_chart_code  # (csv_path, evidence_id) -> (code_str, meta)
except Exception:
    suggest_chart_code = None  # type: ignore

__all__ = ["ChartBuilder", "build_for_doc", "build_for_run"]

# ----------------------------- file helpers ------------------------------------

def _rank_dir() -> Path:
    d = EVIDENCE_DIR / "_rank"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _append_jsonl(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _pack_path(doc_id: str) -> Path:
    return EVIDENCE_DIR / doc_id / "pack.json"

def _charts_dir(doc_id: str) -> Path:
    d = RENDERS_DIR / doc_id / "charts"
    d.mkdir(parents=True, exist_ok=True)
    (d / "scripts").mkdir(parents=True, exist_ok=True)
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d

def _manifest_path(doc_id: str) -> Path:
    return _charts_dir(doc_id) / "_charts.json"

def _fail_log_path(doc_id: str) -> Path:
    return _charts_dir(doc_id) / "_fail.jsonl"

# ----------------------------- datamodel --------------------------------------

@dataclass
class ChartSpec:
    chart_type: str               # "line" | "bar" | "scatter"
    x: str
    y: List[str]                  # one or more series
    title: str
    note: str                     # extra info (units, inferred hints)

@dataclass
class ChartRecord:
    evidence_id: str
    doc_id: str
    chart_id: str                 # stable hash to dedupe
    png: str
    script: str
    chart_type: str
    columns: List[str]
    n_rows: int
    n_cols: int
    title: str
    note: str
    # csv path can be absent for figure passthrough
    csv: Optional[str] = None

# ----------------------------- heuristics --------------------------------------

_TIME_NAMES = {"date", "year", "time", "month", "day", "quarter"}

def _infer_spec(df: pd.DataFrame, evid: str) -> Optional[ChartSpec]:
    """
    Deterministic rule-based chart suggestion:
      1) If a datetime-like column + >=1 numeric -> line (up to 3 series)
      2) Else if a low-cardinality categorical + 1 numeric -> bar
      3) Else if >=2 numeric -> scatter (first two)
      4) Else fallback: bar(index vs first numeric)
    """
    if df is None or df.empty:
        return None
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return None

    cols = list(df.columns)
    dt_candidates = []
    for c in cols:
        s = df[c]
        if is_datetime64_any_dtype(s):
            dt_candidates.append(c)
            continue
        if c.strip().lower() in _TIME_NAMES:
            try:
                pd.to_datetime(s, errors="raise")
                dt_candidates.append(c)
            except Exception:
                pass

    num_cols = [c for c in cols if is_numeric_dtype(df[c])]
    # 1) line
    if dt_candidates and num_cols:
        x = dt_candidates[0]
        y = num_cols[: min(3, len(num_cols))]
        return ChartSpec(chart_type="line", x=x, y=y, title=f"{evid} – time series", note="heuristic: datetime + numeric")
    # 2) bar
    cat_cand = None
    for c in cols:
        if c in num_cols:
            continue
        uniq = df[c].dropna().nunique()
        if 2 <= uniq <= 20:
            cat_cand = c
            break
    if cat_cand and num_cols:
        return ChartSpec(chart_type="bar", x=cat_cand, y=[num_cols[0]], title=f"{evid} – distribution", note="heuristic: low-card categorical")
    # 3) scatter
    if len(num_cols) >= 2:
        return ChartSpec(chart_type="scatter", x=num_cols[0], y=[num_cols[1]], title=f"{evid} – scatter", note="heuristic: two numeric")
    # 4) fallback
    if num_cols:
        df.reset_index(inplace=True)
        return ChartSpec(chart_type="bar", x="index", y=[num_cols[0]], title=f"{evid} – bar(index)", note="heuristic: index vs numeric")
    return None

# ----------------------------- rendering ---------------------------------------

def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)[:64]

def _stable_chart_id(doc_id: str, csv_path: Optional[Path], spec: Optional[ChartSpec], extra: str = "") -> str:
    h = hashlib.sha256()
    h.update(doc_id.encode("utf-8"))
    h.update((str(csv_path) if csv_path else "").encode("utf-8"))
    if spec:
        h.update(spec.chart_type.encode("utf-8"))
        h.update(spec.x.encode("utf-8"))
        for y in (spec.y or []):
            h.update(y.encode("utf-8"))
        h.update(spec.title.encode("utf-8"))
    if extra:
        h.update(extra.encode("utf-8"))
    return h.hexdigest()[:16]

def _render_chart(df: pd.DataFrame, spec: ChartSpec, out_png: Path) -> None:
    """Render using matplotlib only, single chart per image."""
    plt.figure(figsize=(12, 7), dpi=120)
    if spec.chart_type == "line":
        try:
            xvals = pd.to_datetime(df[spec.x], errors="coerce")
        except Exception:
            xvals = df[spec.x]
        for y in spec.y:
            plt.plot(xvals, df[y], label=y)
        plt.legend(loc="best", fontsize=9)
        plt.xlabel(spec.x); plt.ylabel(", ".join(spec.y))
    elif spec.chart_type == "bar":
        xvals = df[spec.x].astype(str)
        y = spec.y[0]
        plt.bar(xvals, df[y])
        plt.xticks(rotation=25, ha="right")
        plt.xlabel(spec.x); plt.ylabel(y)
    elif spec.chart_type == "scatter":
        x = spec.x; y = spec.y[0]
        plt.scatter(df[x], df[y])
        plt.xlabel(x); plt.ylabel(y)
    else:
        plt.text(0.5, 0.5, f"Unsupported chart_type: {spec.chart_type}", ha="center", va="center")
    plt.title(spec.title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png))
    plt.close()

def _emit_script(csv_path: Path, spec: ChartSpec, out_script: Path) -> None:
    """Write a standalone Python script to regenerate the PNG (for provenance)."""
    code = f'''\
# Auto-generated by CiteVizor chart_builder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"{csv_path}", encoding="utf-8")

plt.figure(figsize=(12, 7), dpi=120)
chart_type = "{spec.chart_type}"
title = "{spec.title}"
if chart_type == "line":
    try:
        x = pd.to_datetime(df["{spec.x}"], errors="coerce")
    except Exception:
        x = df["{spec.x}"]
    for col in {spec.y}:
        plt.plot(x, df[col], label=col)
    plt.legend(loc="best", fontsize=9)
    plt.xlabel("{spec.x}"); plt.ylabel(", ".join({spec.y}))
elif chart_type == "bar":
    x = df["{spec.x}"].astype(str)
    y = "{spec.y[0]}"
    plt.bar(x, df[y])
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("{spec.x}"); plt.ylabel(y)
elif chart_type == "scatter":
    x = "{spec.x}"; y = "{spec.y[0]}"
    plt.scatter(df[x], df[y])
    plt.xlabel(x); plt.ylabel(y)
plt.title(title)
plt.tight_layout()
import sys
if len(sys.argv) > 1:
    plt.savefig(sys.argv[1])
else:
    plt.savefig("chart.png")
'''
    out_script.parent.mkdir(parents=True, exist_ok=True)
    out_script.write_text(textwrap.dedent(code), encoding="utf-8")

# ----------------------------- pack readers ------------------------------------

def _load_tables_from_pack(doc_id: str) -> List[Dict[str, Any]]:
    p = _pack_path(doc_id)
    pack = _load_json(p)
    out: List[Dict[str, Any]] = []
    for t in (pack.get("tables") or []):
        csvp = t.get("csv_path")
        evid = t.get("id") or t.get("evidence_id") or ""
        if csvp and evid:
            out.append({"evidence_id": str(evid), "csv_path": csvp})
    return out

def _load_figures_from_pack(doc_id: str) -> List[Dict[str, Any]]:
    p = _pack_path(doc_id)
    pack = _load_json(p)
    out: List[Dict[str, Any]] = []
    for fig in (pack.get("figures") or []):
        img = fig.get("image_path") or fig.get("img_path")
        evid = fig.get("id") or fig.get("evidence_id") or ""
        cap = fig.get("caption") or ""
        if img and evid:
            out.append({"evidence_id": str(evid), "image_path": img, "caption": cap})
    return out

def _read_csv(csv_path: Path, max_rows: int = 10000) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path, nrows=max_rows, encoding="utf-8")
        df = df.dropna(how="all")
        for c in df.columns:
            if not is_numeric_dtype(df[c]):
                try:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="ignore")
                except Exception:
                    pass
        return df
    except Exception:
        return None

def _maybe_codegen(csv_path: Path, evid: str) -> Optional[ChartSpec]:
    """
    If llm.chart_codegen.suggest_chart_code is present, call it.
    Expected signature:
        suggest_chart_code(csv_path: str, evidence_id: str) -> Tuple[str code, ChartSpec-like dict]
    """
    if suggest_chart_code is None:
        return None
    try:
        code_str, meta = suggest_chart_code(str(csv_path), evid)  # type: ignore
        if not isinstance(meta, dict):
            return None
        chart_type = meta.get("chart_type")
        x = meta.get("x"); y = meta.get("y") or []
        title = meta.get("title") or f"{evid}"
        note = meta.get("note") or "llm-codegen"
        if chart_type not in ("line", "bar", "scatter"):
            return None
        if not x or not y:
            return None
        return ChartSpec(chart_type=chart_type, x=x, y=list(y), title=title, note=note)
    except Exception:
        return None

def _resolve_run_doc_ids(run_id: str) -> List[str]:
    """
    Be tolerant to different rank artifact shapes:
      - <run_id>.citations.json with {"refs":[{"doc_id":...}, ...]}
      - <run_id>.json with {"refs":[...]}
      - <run_id>.dedup.json with {"refs":[...]}
    """
    rank = _rank_dir()
    for name in (f"{run_id}.citations.json", f"{run_id}.json", f"{run_id}.dedup.json"):
        obj = _load_json(rank / name)
        if obj:
            refs = obj.get("refs") or []
            doc_ids = sorted({r.get("doc_id") for r in refs if r.get("doc_id")})
            if doc_ids:
                return doc_ids
    return []

# ----------------------------- Class API ---------------------------------------

class ChartBuilder:
    """Stable class used by citevizor_backend.py"""
    def __init__(self) -> None:
        pass

    # ---------- main per-doc pipeline ----------

    def build_for_doc(self, doc_id: str, limit_per_doc: int = 6) -> Path:
        charts_dir = _charts_dir(doc_id)
        manifest_path = _manifest_path(doc_id)
        fail_log = _fail_log_path(doc_id)

        # load existing manifest for idempotency
        existing: List[Dict[str, Any]] = []
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing_ids = {r.get("chart_id") for r in (existing or []) if r.get("chart_id")}

        manifest_add: List[Dict[str, Any]] = []
        produced = 0

        # 1) build from tables
        tables = _load_tables_from_pack(doc_id)
        for t in tables:
            if produced >= max(1, limit_per_doc):
                break
            evid = t["evidence_id"]
            csv_path = Path(t["csv_path"])
            if not csv_path.exists():
                alt = EVIDENCE_DIR / doc_id / "tables" / csv_path.name
                if not alt.exists():
                    _append_jsonl(fail_log, {"doc_id": doc_id, "evidence_id": evid, "reason": "csv_missing", "path": str(csv_path)})
                    continue
                csv_path = alt

            df = _read_csv(csv_path)
            if df is None or df.empty or len(df.columns) < 2:
                _append_jsonl(fail_log, {"doc_id": doc_id, "evidence_id": evid, "reason": "csv_empty_or_too_few_cols"})
                continue

            spec = _maybe_codegen(csv_path, evid) or _infer_spec(df, evid)
            if not spec:
                _append_jsonl(fail_log, {"doc_id": doc_id, "evidence_id": evid, "reason": "no_chart_spec"})
                continue

            chart_id = _stable_chart_id(doc_id, csv_path, spec)
            if chart_id in existing_ids:
                # skip duplicate
                continue

            base = f"{_safe_name(evid)}_{produced+1:02d}"
            out_png = charts_dir / f"{base}.png"
            out_script = charts_dir / "scripts" / f"{base}.py"

            _emit_script(csv_path, spec, out_script)
            try:
                _render_chart(df, spec, out_png)
            except Exception as e:
                _append_jsonl(fail_log, {"doc_id": doc_id, "evidence_id": evid, "reason": "render_fail", "err": str(e)[:200]})
                continue

            rec = ChartRecord(
                evidence_id=evid,
                doc_id=doc_id,
                chart_id=chart_id,
                png=str(out_png),
                script=str(out_script),
                chart_type=spec.chart_type,
                columns=[spec.x] + list(spec.y),
                n_rows=int(df.shape[0]),
                n_cols=int(df.shape[1]),
                title=spec.title,
                note=spec.note,
                csv=str(csv_path),
            )
            manifest_add.append(asdict(rec))
            existing_ids.add(chart_id)
            produced += 1

        # 2) passthrough figures if still below limit
        if produced < max(1, limit_per_doc):
            figs = _load_figures_from_pack(doc_id)
            for fg in figs:
                if produced >= max(1, limit_per_doc):
                    break
                src = Path(fg["image_path"])
                if not src.is_absolute():
                    # try resolve relative to evidence dir
                    cand = EVIDENCE_DIR / doc_id / str(src)
                    src = cand if cand.exists() else src
                if not src.exists():
                    _append_jsonl(fail_log, {"doc_id": doc_id, "evidence_id": fg["evidence_id"], "reason": "figure_missing", "path": str(src)})
                    continue

                # produce deterministic id based on image content if possible
                try:
                    img_hash = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
                except Exception:
                    img_hash = src.name[:16]
                chart_id = _stable_chart_id(doc_id, None, None, extra=f"figure:{img_hash}")
                if chart_id in existing_ids:
                    continue

                dst = charts_dir / "figures" / f"{_safe_name(fg['evidence_id'])}_{img_hash}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # copy (not move) to keep provenance
                    shutil.copyfile(src, dst)
                except Exception as e:
                    _append_jsonl(fail_log, {"doc_id": doc_id, "evidence_id": fg["evidence_id"], "reason": "figure_copy_fail", "err": str(e)[:200]})
                    continue

                rec = ChartRecord(
                    evidence_id=fg["evidence_id"],
                    doc_id=doc_id,
                    chart_id=chart_id,
                    png=str(dst),
                    script="",                     # no script for passthrough
                    chart_type="figure",           # explicit label for layout
                    columns=[],
                    n_rows=0,
                    n_cols=0,
                    title=fg.get("caption") or fg["evidence_id"],
                    note="figure-passthrough",
                    csv=None
                )
                manifest_add.append(asdict(rec))
                existing_ids.add(chart_id)
                produced += 1

        # 3) write/merge manifest
        final = (existing or []) + manifest_add
        manifest_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path

    # ---------- batch by run ----------

    def build_for_run(self, run_id: str, limit_per_doc: int = 6) -> List[Path]:
        doc_ids = _resolve_run_doc_ids(run_id)
        if not doc_ids:
            raise FileNotFoundError(f"No rank artifacts (refs) found for run_id={run_id}")
        outs: List[Path] = []
        for doc_id in doc_ids:
            try:
                p = self.build_for_doc(doc_id, limit_per_doc=limit_per_doc)
                outs.append(p)
            except Exception as e:
                _append_jsonl(_fail_log_path(doc_id), {"doc_id": doc_id, "reason": "build_for_doc_fail", "err": str(e)[:200]})
                continue
        return outs

    # ---------- CLI-compatible entry ----------

    def run(self,
            run_id: Optional[str] = None,
            doc_id: Optional[str] = None,
            limit_per_doc: int = 6) -> Union[Path, List[Path]]:
        if bool(run_id) == bool(doc_id):
            raise ValueError("Specify exactly one of run_id or doc_id.")
        if doc_id:
            return self.build_for_doc(doc_id, limit_per_doc=limit_per_doc)
        else:
            return self.build_for_run(run_id or "", limit_per_doc=limit_per_doc)

# ----------------------------- Back-compat fns ---------------------------------

def build_for_doc(doc_id: str, limit_per_doc: int = 6) -> Path:
    return ChartBuilder().build_for_doc(doc_id, limit_per_doc=limit_per_doc)

def build_for_run(run_id: str, limit_per_doc: int = 6) -> List[Path]:
    return ChartBuilder().build_for_run(run_id, limit_per_doc=limit_per_doc)

# ----------------------------- CLI ---------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor chart builder (matplotlib-only)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--doc-id", help="Target a single document id")
    g.add_argument("--run-id", help="Generate charts for all docs in a run")
    ap.add_argument("--limit-per-doc", type=int, default=6, help="Max visuals (charts+figures) per document")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if args.doc_id:
        out = build_for_doc(args.doc_id, limit_per_doc=max(1, args.limit_per_doc))
        print(f"[OK] Charts manifest -> {out}")
    else:
        outs = build_for_run(args.run_id, limit_per_doc=max(1, args.limit_per_doc))
        print("[OK] Charts manifests:")
        for p in outs:
            print(" -", p)
