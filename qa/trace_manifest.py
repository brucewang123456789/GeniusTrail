# qa/trace_manifest.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Trace Manifest (Provenance) Generator

This module is NOT in the critical download/parse path. It only records
artifacts near the end of a run. The warnings you see earlier (download.fail,
parse.fail, summarize.abort_no_llm, etc.) originate elsewhere. Here we make
manifest writing deterministic and resilient so you always get a file.

Key improvements:
- Respect CITEVIZOR_TRACE_DIR if set; otherwise fall back to reports/_runs.
- Create directories robustly; atomic write (tmp file -> rename).
- Never raise on normal errors; still return "" on failure to match caller.
- Keep everything else intact to avoid side effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from schemas import STORAGE_DIR, EVIDENCE_DIR, RENDERS_DIR, REPORTS_DIR

ENV_FILE = "citevizor.env"


# ----------------------------- file utils -------------------------------------

def _file_info(p: Path, do_hash: bool) -> Dict[str, Any]:
    """Return a compact dict for a file; if not exists, mark 'exists': False."""
    info = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return info
    try:
        stat = p.stat()
        info.update({
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
        })
        if do_hash:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            info["sha256"] = h.hexdigest()
    except Exception as e:
        info["error"] = str(e)
    return info


def _rank_dir() -> Path:
    d = EVIDENCE_DIR / "_rank"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_json(p: Path) -> Dict[str, Any] | List[Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_env(project_root: Path, redact: bool = True) -> Dict[str, Any]:
    """Load citevizor.env (key=value) into a dict; redact likely secrets."""
    path = project_root / ENV_FILE
    out: Dict[str, Any] = {"_env_path": str(path), "_exists": path.exists()}
    if not path.exists():
        return out
    raw: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        raw[k.strip()] = v.strip().strip('"').strip("'")
    if not redact:
        out.update(raw)
        return out

    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "API_KEY")
    redacted = {}
    for k, v in raw.items():
        if any(m in k.upper() for m in secret_markers):
            redacted[k] = "<redacted>"
        else:
            redacted[k] = v
    out.update(redacted)
    return out


# ----------------------------- models -----------------------------------------

@dataclass
class Artifact:
    kind: str
    info: Dict[str, Any]

@dataclass
class DocBundle:
    doc_id: str
    pdf: Artifact
    meta_json: Artifact
    pack_json: Artifact
    table_count: int
    figure_count: int
    charts: List[Artifact]
    figures: List[Artifact]

@dataclass
class TraceManifest:
    run_id: str
    created_at: int
    query: str
    platform: Dict[str, Any]
    env_snapshot: Dict[str, Any]
    llm: Dict[str, Any]
    core: Dict[str, Artifact]
    docs: List[DocBundle]
    edges: List[Dict[str, str]]


# ----------------------------- collectors -------------------------------------

def _collect_core(run_id: str, hash_all: bool) -> Dict[str, Artifact]:
    core: Dict[str, Artifact] = {}
    cit = _rank_dir() / f"{run_id}.citations.json"
    summ = _rank_dir() / f"{run_id}.summary.json"
    html = REPORTS_DIR / f"{run_id}.html"
    pptx = REPORTS_DIR / f"{run_id}.pptx"

    core["citations_json"] = Artifact("citations_json", _file_info(cit, True))
    core["summary_json"]   = Artifact("summary_json", _file_info(summ, True))
    core["report_html"]    = Artifact("report_html", _file_info(html, hash_all))
    core["report_pptx"]    = Artifact("report_pptx", _file_info(pptx, hash_all))
    return core


def _collect_docs(run_id: str, hash_all: bool) -> List[DocBundle]:
    """
    Build per-document bundles based on refs/doc_ids referenced in citations.json.
    Figures list is best-effort: it tries to find image files recorded in pack.json.
    """
    ci = _load_json(_rank_dir() / f"{run_id}.citations.json")
    refs = ci.get("refs", []) if isinstance(ci, dict) else []
    doc_ids = list({r.get("doc_id") for r in refs if isinstance(r, dict) and r.get("doc_id")})

    bundles: List[DocBundle] = []
    for doc_id in doc_ids:
        docs_dir = STORAGE_DIR / "docs" / doc_id
        pdf = docs_dir / f"{doc_id}.pdf"
        meta = docs_dir / f"{doc_id}.meta.json"
        pack = EVIDENCE_DIR / doc_id / "pack.json"

        pack_json = _load_json(pack)
        tables = pack_json.get("tables", []) if isinstance(pack_json, dict) else []
        figures = pack_json.get("figures", []) if isinstance(pack_json, dict) else []

        charts_dir = RENDERS_DIR / doc_id / "charts"
        charts_manifest = charts_dir / "_charts.json"
        chart_list: List[Artifact] = []
        if charts_manifest.exists():
            try:
                data = _load_json(charts_manifest)
                if isinstance(data, dict):
                    data = data.get("records") or []
                if not isinstance(data, list):
                    data = []
                for r in data:
                    if not isinstance(r, dict):
                        continue
                    png = Path(r.get("png") or "")
                    if not png.is_absolute():
                        png = charts_dir / Path(png).name
                    chart_list.append(Artifact("chart_png", _file_info(png, hash_all)))
            except Exception:
                for p in charts_dir.glob("*.png"):
                    chart_list.append(Artifact("chart_png", _file_info(p, hash_all)))

        figure_imgs: List[Artifact] = []
        for f in figures:
            if not isinstance(f, dict):
                continue
            img = f.get("image_path")
            if not img:
                continue
            p = Path(img)
            if not p.exists():
                alt = EVIDENCE_DIR / doc_id / "figures" / Path(img).name
                p = alt if alt.exists() else Path(img)
            figure_imgs.append(Artifact("figure_img", _file_info(p, hash_all)))

        bundle = DocBundle(
            doc_id=doc_id,
            pdf=Artifact("pdf", _file_info(pdf, True)),
            meta_json=Artifact("meta_json", _file_info(meta, True)),
            pack_json=Artifact("pack_json", _file_info(pack, True)),
            table_count=len(tables),
            figure_count=len(figures),
            charts=chart_list,
            figures=figure_imgs,
        )
        bundles.append(bundle)
    return bundles


def _collect_edges(manifest: "TraceManifest") -> List[Dict[str, str]]:
    """
    Minimal provenance:
      pdf -> pack_json -> citations_json -> summary_json -> report_{html,pptx}
      charts/figures derive from packs.
    """
    edges: List[Dict[str, str]] = []
    rid = manifest.run_id
    core_ids = {
        "cit": f"rank:{rid}:citations_json",
        "sum": f"rank:{rid}:summary_json",
        "html": f"rank:{rid}:report_html",
        "pptx": f"rank:{rid}:report_pptx",
    }
    for d in manifest.docs:
        n_pdf = f"doc:{d.doc_id}:pdf"
        n_pack = f"doc:{d.doc_id}:pack_json"
        edges.append({"from": n_pdf, "to": n_pack, "type": "parsed_from"})
        edges.append({"from": n_pack, "to": core_ids["cit"], "type": "evidence_source"})
        for _ in d.charts:
            edges.append({"from": n_pack, "to": f"doc:{d.doc_id}:chart_png", "type": "chart_from_table"})
        for _ in d.figures:
            edges.append({"from": n_pack, "to": f"doc:{d.doc_id}:figure_img", "type": "figure_from_pdf"})
    edges.append({"from": core_ids["cit"], "to": core_ids["sum"], "type": "summarized_by_llm"})
    edges.append({"from": core_ids["sum"], "to": core_ids["html"], "type": "laid_out"})
    edges.append({"from": core_ids["sum"], "to": core_ids["pptx"], "type": "exported"})
    return edges


def _llm_snapshot(run_id: str, project_root: Path) -> Dict[str, Any]:
    """Pull model/usage from summary.json; non-secret serving context from env."""
    summ = _load_json(_rank_dir() / f"{run_id}.summary.json")
    llm = {
        "model": summ.get("model") if isinstance(summ, dict) else None,
        "usage": summ.get("llm_usage") if isinstance(summ, dict) else None,
    }
    env = _read_env(project_root, redact=True)
    if env.get("VLLM_MODEL"):
        llm["serving_model"] = env.get("VLLM_MODEL")
    if env.get("VLLM_ENDPOINT"):
        llm["endpoint"] = env.get("VLLM_ENDPOINT")
    return llm


# ----------------------------- public API -------------------------------------

def _resolve_trace_dir() -> Path:
    """Prefer CITEVIZOR_TRACE_DIR; fallback to reports/_runs."""
    env_dir = os.getenv("CITEVIZOR_TRACE_DIR", "").strip()
    if env_dir:
        p = Path(env_dir)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            pass  # will fallback below
    p = REPORTS_DIR / "_runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_manifest(run_id: str, manifest: Dict[str, Any]) -> str:
    """
    Backend-facing API expected by citevizor_backend.py:
    Persist under <trace_dir>/<run_id>.manifest.json and return the path.
    Never raise; on failure return "".
    """
    try:
        out_dir = _resolve_trace_dir()
        out_path = out_dir / f"{run_id}.manifest.json"
        payload = json.dumps(manifest, ensure_ascii=False, indent=2)
        _atomic_write_text(out_path, payload)
        return str(out_path)
    except Exception:
        # Keep silent to avoid interrupting pipeline; caller logs its own warnings.
        return ""


def build_trace_manifest(run_id: str, hash_all: bool = False, redact_env: bool = True) -> Path:
    """Build and persist a richer provenance manifest for a ranking run."""
    project_root = Path(__file__).resolve().parents[1]

    ci_path = _rank_dir() / f"{run_id}.citations.json"
    ci = _load_json(ci_path)

    core = _collect_core(run_id, hash_all)
    docs = _collect_docs(run_id, hash_all)
    env_snapshot = _read_env(project_root, redact=redact_env)
    llm = _llm_snapshot(run_id, project_root)

    manifest = TraceManifest(
        run_id=run_id,
        created_at=int(time.time()),
        query=ci.get("query", "") if isinstance(ci, dict) else "",
        platform={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
        },
        env_snapshot=env_snapshot,
        llm=llm,
        core=core,
        docs=docs,
        edges=[],  # fill below
    )
    manifest.edges = _collect_edges(manifest)

    payload = {
        "run_id": manifest.run_id,
        "created_at": manifest.created_at,
        "query": manifest.query,
        "platform": manifest.platform,
        "env_snapshot": manifest.env_snapshot,
        "llm": manifest.llm,
        "core": {k: asdict(v) for k, v in manifest.core.items()},
        "docs": [asdict(d) for d in manifest.docs],
        "edges": manifest.edges,
    }

    out = _rank_dir() / f"{run_id}.trace.json"
    _atomic_write_text(out, json.dumps(payload, ensure_ascii=False, indent=2))
    return out


# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor provenance manifest generator")
    ap.add_argument("--run-id", required=True, help="Ranking run_id")
    ap.add_argument("--full-hash", action="store_true", help="Hash every artifact (slower, but fully reproducible)")
    ap.add_argument("--no-redact-env", action="store_true", help="Do not redact env values in the snapshot")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = build_trace_manifest(args.run_id, hash_all=bool(args.full_hash), redact_env=(not args.no_redact_env))
    print(f"[OK] Trace manifest -> {out}")
