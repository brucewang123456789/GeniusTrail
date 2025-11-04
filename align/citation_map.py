# C:\CiteVizor\align\citation_map.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Citation mapping
- Inputs: ranked evidence JSON from align/evidence_rank.py (run_id or path).
- Looks up pack.json and meta.json to attach page/figure/table + DOI/URL.
- Outputs: storage/evidence/_rank/<run_id>.citations.json with:
    {
      run_id, query, items: [...], refs: [...], style, generated_at
    }
  where each item has a compact label like "[3 p.7, Fig.2]" pointing to refs[ref_id-1].
- Defensive and idempotent; no external dependencies beyond stdlib + schemas.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas import EVIDENCE_DIR, DOCS_DIR  # project paths

ENV_FILE = "citevizor.env"

# ----------------------------- Env loader -------------------------------------

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
    # OS env override
    for k in ("CITATION_STYLE",):
        if os.getenv(k):
            out[k] = os.getenv(k)  # type: ignore
    return out

# ----------------------------- IO helpers -------------------------------------

def _rank_dir() -> Path:
    d = EVIDENCE_DIR / "_rank"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_rank(run_id: Optional[str], path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        if not run_id:
            raise ValueError("Either --run-id or --rank-path must be provided.")
        path = _rank_dir() / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Rank file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def _load_pack(doc_id: str) -> Dict[str, Any]:
    p = EVIDENCE_DIR / doc_id / "pack.json"
    if not p.exists():
        return {"doc_id": doc_id, "paragraphs": [], "tables": [], "figures": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"doc_id": doc_id, "paragraphs": [], "tables": [], "figures": []}

def _load_meta(doc_id: str) -> Dict[str, Any]:
    m = DOCS_DIR / doc_id / f"{doc_id}.meta.json"
    if not m.exists():
        return {}
    try:
        return json.loads(m.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_citations(run_id: str, payload: Dict[str, Any]) -> Path:
    out_path = _rank_dir() / f"{run_id}.citations.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

# ----------------------------- Formatting -------------------------------------

def _compact_label(ref_id: int, page: Optional[int], fig_no: Optional[str], tab_no: Optional[str]) -> str:
    parts: List[str] = [f"[{ref_id}"]
    if page:
        parts.append(f"p.{page}")
    if fig_no:
        parts.append(f"Fig.{fig_no}")
    if tab_no:
        parts.append(f"Table.{tab_no}")
    return ", ".join(parts) + "]"

def _ieee_ref_text(ref: Dict[str, Any]) -> str:
    # Minimal IEEE-like reference line; downstream can format fully.
    title = ref.get("title") or "<untitled>"
    venue = ref.get("venue") or ""
    year = ref.get("year") or ""
    doi = ref.get("doi") or ""
    url = ref.get("url") or ""
    bits = [title]
    if venue:
        bits.append(venue)
    if year:
        bits.append(str(year))
    if doi:
        bits.append(f"doi:{doi}")
    elif url:
        bits.append(url)
    return ", ".join(bits)

# ----------------------------- Core mapper ------------------------------------

@dataclass
class CitationMapperConfig:
    project_root: Path
    style: str = "compact"  # "compact" | "ieee"

    @classmethod
    def from_env(cls, project_root: Path) -> "CitationMapperConfig":
        env = _load_env(project_root)
        style = env.get("CITATION_STYLE", "compact").lower()
        if style not in ("compact", "ieee"):
            style = "compact"
        return cls(project_root=project_root, style=style)

class CitationMapper:
    def __init__(self, cfg: CitationMapperConfig):
        self.cfg = cfg

    def build(self, rank_json: Dict[str, Any], top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Build citation mapping for the ranked evidence.
        """
        items = rank_json.get("items", [])
        if top_k:
            items = items[:top_k]

        # Assign a reference id to each distinct doc_id (order of first appearance)
        doc_to_ref: Dict[str, int] = {}
        refs: List[Dict[str, Any]] = []
        def _ensure_ref(doc_id: str) -> int:
            if doc_id in doc_to_ref:
                return doc_to_ref[doc_id]
            meta = _load_meta(doc_id)
            ref_id = len(refs) + 1
            doc_to_ref[doc_id] = ref_id
            refs.append({
                "ref_id": ref_id,
                "doc_id": doc_id,
                "title": meta.get("title"),
                "venue": meta.get("venue"),
                "year": meta.get("year"),
                "doi": meta.get("doi"),
                "url": meta.get("final_url") or meta.get("url"),
                "authors": meta.get("authors"),
            })
            return ref_id

        mapped_items: List[Dict[str, Any]] = []
        for it in items:
            doc_id = it.get("doc_id")
            ev_id = it.get("evidence_id")
            if not doc_id or not ev_id:
                # Skip malformed entries
                continue
            ref_id = _ensure_ref(doc_id)

            # Try to enrich page/figure/table from pack.json if missing
            page = it.get("page")
            fig_no = it.get("figure_no")
            tab_no = it.get("table_no")
            if page is None or (fig_no is None and tab_no is None):
                pack = _load_pack(doc_id)
                # Search evidence by id (first match)
                ev = _find_evidence(pack, ev_id)
                if ev:
                    page = page if page is not None else ev.get("page")
                    fig_no = fig_no if fig_no is not None else ev.get("figure_no")
                    tab_no = tab_no if tab_no is not None else ev.get("table_no")

            label = _compact_label(ref_id, page, fig_no, tab_no) if self.cfg.style == "compact" else f"[{ref_id}]"
            mapped_items.append({
                "rank": it.get("rank"),
                "type": it.get("type"),
                "doc_id": doc_id,
                "evidence_id": ev_id,
                "ref_id": ref_id,
                "label": label,
                "page": page,
                "figure_no": fig_no,
                "table_no": tab_no,
                "snippet": it.get("snippet"),
                "score": it.get("score"),
                "year": it.get("year"),
            })

        # Optional IEEE-like reference strings
        if self.cfg.style == "ieee":
            for r in refs:
                r["text"] = _ieee_ref_text(r)

        payload = {
            "run_id": rank_json.get("run_id"),
            "query": rank_json.get("query"),
            "sub_queries": rank_json.get("sub_queries"),
            "generated_at": int(time.time()),
            "style": self.cfg.style,
            "items": mapped_items,
            "refs": refs,
            "meta": rank_json.get("meta", {}),
        }
        return payload

# ----------------------------- Search helpers ---------------------------------

def _find_evidence(pack: Dict[str, Any], evidence_id: str) -> Optional[Dict[str, Any]]:
    """
    Linear search within pack.json (small per-doc; acceptable).
    """
    for key in ("paragraphs", "tables", "figures"):
        for e in pack.get(key, []):
            if e.get("id") == evidence_id:
                return e
    return None

# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor citation mapper")
    ap.add_argument("--run-id", help="Rank run id (e.g., from evidence_rank output)")
    ap.add_argument("--rank-path", help="Explicit path to rank JSON")
    ap.add_argument("--style", default="", help="compact|ieee (overrides env)")
    ap.add_argument("--topk", type=int, default=0, help="Optional top-K truncation")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = CitationMapperConfig.from_env(project_root)
    if args.style:
        s = args.style.lower()
        if s in ("compact", "ieee"):
            cfg.style = s

    rank_path = Path(args.rank_path) if args.rank_path else None
    rank_json = _load_rank(args.run_id, rank_path)

    mapper = CitationMapper(cfg)
    payload = mapper.build(rank_json, top_k=(args.topk if args.topk and args.topk > 0 else None))
    out_path = _save_citations(rank_json.get("run_id") or "unknown", payload)

    print(f"[OK] Citation map -> {out_path}")
    print(f"Style: {payload['style']} | refs={len(payload['refs'])} | items={len(payload['items'])}")
    # Show top few for quick inspection
    for it in payload["items"][: min(10, len(payload["items"]))]:
        print(f"  #{it['rank']:02d} {it['label']} {it['type']} {it['doc_id']} {it['snippet'][:80]}...")
