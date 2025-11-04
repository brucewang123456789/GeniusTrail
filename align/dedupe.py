# /workspace/align/dedupe.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Cross-document deduplication for ranked evidence (robust edition)

What it does
  - Reads a ranking artifact (evidence_rank output or citations map) for a given run_id.
  - Clusters near-duplicate items per type:
      * paragraph/text  -> 64-bit SimHash over normalized tokens (Hamming <= 8).
      * figure/image    -> 64-bit dHash on the figure bitmap (Hamming <= 6).
      * table/csv       -> stable CSV fingerprint (header + sample + numeric sketch).
  - Picks a canonical representative per cluster: highest 'score', tie-broken by newer 'year'.
  - Writes a deduplicated JSON: storage/evidence/_rank/<run_id>.dedup.json

Robustness changes
  - Soft IO: missing/ill-formed rank JSON will not raise; falls back to empty shape.
  - Safe clustering: items with empty features (no tokens / no image / no csv) are excluded from clustering
    to avoid false positive merges.
  - Back-compat shim: .run_on_docs(doc_ids, run_id=None) added (no-op dedupe over packs; returns path or None).

Public API
  - class Deduper(...).run(run_id: str) -> Path
  - class Deduper(...).run_in_memory(items: List[dict], refs: List[dict], run_id: str, query: str = "") -> dict
  - class Deduper(...).run_on_docs(doc_ids: List[str], run_id: Optional[str] = None) -> Optional[Path]
  - legacy function dedupe_run(cfg: DedupeConfig) -> Path
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

from schemas import EVIDENCE_DIR, STORAGE_DIR  # project directories

__all__ = ["Deduper", "DedupeConfig", "dedupe_run"]

# ----------------------------- IO helpers -------------------------------------

def _rank_dir() -> Path:
    d = EVIDENCE_DIR / "_rank"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _load_rank_like(run_id: str) -> Dict[str, Any]:
    """
    Load rank-like artifact. Prefer citations; fallback to rank json.
    Soft-fail: if both missing or malformed, return empty shape (won't raise).
    """
    p_cit = _rank_dir() / f"{run_id}.citations.json"
    p_rank = _rank_dir() / f"{run_id}.json"
    obj = _load_json(p_cit) if p_cit.exists() else _load_json(p_rank)
    if not isinstance(obj, dict):
        obj = {}
    # normalize shape (soft)
    items = obj.get("items") if isinstance(obj.get("items"), list) else []
    refs  = obj.get("refs")  if isinstance(obj.get("refs"), list)  else []
    return {
        "run_id": run_id,
        "query": obj.get("query", ""),
        "items": items,
        "refs":  refs,
    }

def _pack_path(doc_id: str) -> Path:
    return EVIDENCE_DIR / doc_id / "pack.json"

def _load_pack(doc_id: str) -> Dict[str, Any]:
    return _load_json(_pack_path(doc_id))

# ----------------------------- small text utils --------------------------------

_WORD = re.compile(r"[A-Za-z0-9]+", re.UNICODE)

def _normalize_text(s: str) -> List[str]:
    s = (s or "").lower()
    # strip brackets like [1], [2 p.7]
    s = re.sub(r"\[[^\]]+\]", " ", s)
    toks = _WORD.findall(s)
    return toks[:256]  # cap tokens

def _shingles(tokens: List[str], k: int = 3) -> List[str]:
    if len(tokens) < k:
        return []
    return [" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)]

# ----------------------------- simhash / dhash ---------------------------------

def _simhash64(features: List[str]) -> Optional[int]:
    """Classic 64-bit SimHash with BLAKE2b hashing; returns None for empty features."""
    if not features:
        return None
    bits = [0] * 64
    for f in features:
        h = int(hashlib.blake2b(f.encode("utf-8"), digest_size=8).hexdigest(), 16)  # 64-bit
        w = 2
        for i in range(64):
            bits[i] += w if (h >> i) & 1 else -w
    out = 0
    for i in range(64):
        if bits[i] > 0:
            out |= (1 << i)
    return out

def _hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def _dhash64(img: Image.Image, size: int = 8) -> int:
    """
    Perceptual difference hash (dHash) -> 64-bit for size=8.
    Convert to grayscale, resize to (size+1, size), compare neighbors horizontally.
    """
    g = ImageOps.grayscale(img)
    g = g.resize((size + 1, size), Image.BILINEAR)
    pix = list(g.getdata())
    rows = [pix[i*(size+1):(i+1)*(size+1)] for i in range(size)]
    out = 0
    bit = 0
    for r in rows:
        for x in range(size):
            out |= (1 if r[x] > r[x+1] else 0) << bit
            bit += 1
    return out

# ----------------------------- table fingerprint -------------------------------

def _csv_fingerprint(csv_path: Path, sample_rows: int = 50) -> Optional[str]:
    """
    Stable CSV fingerprint:
      - normalized header row (lowercase, trim)
      - sample of first N rows (string-trim + simple numeric rounding)
    Returns None if file missing/empty/unreadable.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        with csv_path.open("r", encoding="utf-8-sig") as f:
            rdr = csv.reader(f)
            rows = []
            for i, row in enumerate(rdr):
                if i == 0:
                    header = [c.strip().lower() for c in row]
                    rows.append(("__HEADER__", header))
                else:
                    norm = []
                    for c in row:
                        c = (c or "").strip()
                        # numeric rounding
                        try:
                            v = float(c.replace(",", ""))
                            norm.append(f"{round(v, 6)}")
                        except Exception:
                            norm.append(c.lower())
                    rows.append(("__ROW__", norm))
                if i >= sample_rows:
                    break
        h = hashlib.blake2b(digest_size=16)
        for tag, arr in rows:
            h.update(tag.encode())
            for cell in arr:
                h.update(b"|")
                h.update(cell.encode(errors="ignore"))
        return h.hexdigest()
    except Exception:
        return None

# ----------------------------- models -----------------------------------------

@dataclass
class DedupeConfig:
    run_id: str
    text_hamming_th: int = 8     # <=8 for SimHash duplicates
    image_hamming_th: int = 6    # <=6 for dHash duplicates
    enable_text: bool = True
    enable_figure: bool = True
    enable_table: bool = True
    keep_duplicates: bool = False  # if True, keep dupes but annotate; else drop

@dataclass
class Cluster:
    type: str                    # "paragraph" | "figure" | "table" | other
    group_id: str
    members: List[str]           # item_ids
    rep_item_id: str
    reason: str

# ----------------------------- collectors for paths ----------------------------

def _figure_path(it: Dict[str, Any]) -> Optional[Path]:
    if (it.get("type") or "") != "figure":
        return None
    doc = it.get("doc_id")
    evid = it.get("evidence_id")
    if not doc or not evid:
        return None
    pack = _load_json(_pack_path(doc))
    figs = pack.get("figures", []) if isinstance(pack, dict) else []
    for e in figs:
        # tolerate id/evidence_id and image_path/img_path
        if (e.get("id") or e.get("evidence_id")) == evid:
            img = e.get("image_path") or e.get("img_path")
            if not img:
                continue
            p = Path(img)
            if p.exists():
                return p
            # fallback to evidence/figures folder
            alt = EVIDENCE_DIR / doc / "figures" / p.name
            if alt.exists():
                return alt
            # fallback to nested relative
            alt2 = EVIDENCE_DIR / doc / str(p)
            if alt2.exists():
                return alt2
    return None

def _table_csv_path(it: Dict[str, Any]) -> Optional[Path]:
    if (it.get("type") or "") != "table":
        return None
    doc = it.get("doc_id")
    evid = it.get("evidence_id")
    if not doc or not evid:
        return None
    pack = _load_json(_pack_path(doc))
    tabs = pack.get("tables", []) if isinstance(pack, dict) else []
    for t in tabs:
        if (t.get("id") or t.get("evidence_id")) == evid:
            csvp = t.get("csv_path")
            if not csvp:
                continue
            p = Path(csvp)
            if p.exists():
                return p
            # fallback to evidence/tables folder
            alt = EVIDENCE_DIR / doc / "tables" / p.name
            if alt.exists():
                return alt
            alt2 = EVIDENCE_DIR / doc / str(p)
            if alt2.exists():
                return alt2
    return None

# ----------------------------- identity & scoring ------------------------------

def _item_id(it: Dict[str, Any]) -> str:
    eid = it.get("evidence_id")
    if eid:
        return str(eid)
    doc = it.get("doc_id") or "doc"
    typ = it.get("type") or "item"
    page = it.get("page") or ""
    fig = it.get("figure_no") or ""
    tab = it.get("table_no") or ""
    # snippet hash for stability; tolerate missing
    snip = (it.get("snippet") or "")[:160]
    sig = hashlib.blake2b(snip.encode("utf-8", errors="ignore"), digest_size=6).hexdigest()
    return f"{doc}:{typ}:{page}:{fig}:{tab}:{sig}"

def _item_score(it: Dict[str, Any]) -> float:
    try:
        return float(it.get("score", 0.0))
    except Exception:
        return 0.0

def _item_year(it: Dict[str, Any], doc2year: Dict[str, int]) -> int:
    if it.get("doc_id") and it["doc_id"] in doc2year:
        return int(doc2year[it["doc_id"]])
    try:
        return int(it.get("year")) if it.get("year") else -1
    except Exception:
        return -1

# ----------------------------- clustering logic --------------------------------

def _cluster_text(items: List[Dict[str, Any]], th: int) -> Tuple[List[Cluster], Dict[str, str]]:
    """Clusters for text-like items. Skip items with empty token features (no snippet)."""
    fps: Dict[str, int] = {}
    order: List[str] = []
    for it in items:
        iid = _item_id(it)
        toks = _normalize_text(it.get("snippet") or "")
        feats = toks + _shingles(toks, 3)
        h = _simhash64(feats)
        if h is None:
            # empty features -> don't cluster; treat as singleton later
            continue
        order.append(iid)
        fps[iid] = h

    if not order:
        return [], {}

    groups: List[List[str]] = []
    group_ids: Dict[str, str] = {}
    for iid in order:
        placed = False
        for g in groups:
            rep = g[0]
            if _hamming64(fps[iid], fps[rep]) <= th:
                g.append(iid)
                group_ids[iid] = group_ids[rep]
                placed = True
                break
        if not placed:
            gid = f"g_txt_{len(groups)+1:03d}"
            groups.append([iid])
            group_ids[iid] = gid

    clusters: List[Cluster] = []
    for g in groups:
        clusters.append(Cluster(type="paragraph", group_id=group_ids[g[0]], members=g, rep_item_id=g[0],
                                reason=f"simhash_hamming<= {th}"))
    return clusters, group_ids

def _cluster_figures(items: List[Dict[str, Any]], th: int) -> Tuple[List[Cluster], Dict[str, str]]:
    fps: Dict[str, int] = {}
    order: List[str] = []
    for it in items:
        iid = _item_id(it)
        p = _figure_path(it)
        if not p:
            continue
        try:
            with Image.open(p) as im:
                fps[iid] = _dhash64(im)
                order.append(iid)
        except Exception:
            continue
    if not order:
        return [], {}

    groups: List[List[str]] = []
    group_ids: Dict[str, str] = {}
    for iid in order:
        placed = False
        for g in groups:
            rep = g[0]
            if _hamming64(fps[iid], fps[rep]) <= th:
                g.append(iid)
                group_ids[iid] = group_ids[rep]
                placed = True
                break
        if not placed:
            gid = f"g_fig_{len(groups)+1:03d}"
            groups.append([iid])
            group_ids[iid] = gid

    clusters: List[Cluster] = []
    for g in groups:
        clusters.append(Cluster(type="figure", group_id=group_ids[g[0]], members=g, rep_item_id=g[0],
                                reason=f"dhash_hamming<= {th}"))
    return clusters, group_ids

def _cluster_tables(items: List[Dict[str, Any]]) -> Tuple[List[Cluster], Dict[str, str]]:
    fps: Dict[str, str] = {}
    order: List[str] = []
    for it in items:
        iid = _item_id(it)
        p = _table_csv_path(it)
        if not p:
            continue
        fp = _csv_fingerprint(p)
        if not fp:
            continue
        fps[iid] = fp
        order.append(iid)
    if not order:
        return [], {}

    groups: Dict[str, List[str]] = {}
    for iid in order:
        groups.setdefault(fps[iid], []).append(iid)

    clusters: List[Cluster] = []
    group_ids: Dict[str, str] = {}
    for idx, (fp, members) in enumerate(groups.items(), 1):
        gid = f"g_tbl_{idx:03d}"
        for m in members:
            group_ids[m] = gid
        clusters.append(Cluster(type="table", group_id=gid, members=members, rep_item_id=members[0],
                                reason="csv_fingerprint_equal"))
    return clusters, group_ids

# ----------------------------- winner selection --------------------------------

def _pick_representatives(clusters: List[Cluster],
                          item_by_id: Dict[str, Any],
                          doc2year: Dict[str, int]) -> Dict[str, str]:
    """Return mapping group_id -> rep_item_id by max(score), tie -> newer year."""
    rep: Dict[str, str] = {}
    for cl in clusters:
        best = cl.rep_item_id
        best_s = _item_score(item_by_id.get(best, {}))
        best_y = _item_year(item_by_id.get(best, {}), doc2year)
        for iid in cl.members:
            it = item_by_id.get(iid, {})
            s = _item_score(it); y = _item_year(it, doc2year)
            if s > best_s or (math.isclose(s, best_s) and y > best_y):
                best, best_s, best_y = iid, s, y
        rep[cl.group_id] = best
    return rep

# ----------------------------- core engine -------------------------------------

def _dedupe_core(items: List[Dict[str, Any]],
                 refs: List[Dict[str, Any]],
                 run_id: str,
                 query: str,
                 text_hamming_th: int,
                 image_hamming_th: int,
                 enable_text: bool,
                 enable_figure: bool,
                 enable_table: bool,
                 keep_duplicates: bool) -> Dict[str, Any]:
    """Pure function producing the dedupe payload (no file IO)."""

    # Helper maps
    item_by_id: Dict[str, Dict[str, Any]] = {}
    for it in items or []:
        it["_item_id"] = _item_id(it)
        item_by_id[it["_item_id"]] = it
    doc2year: Dict[str, int] = {}
    for r in refs or []:
        if r.get("doc_id") and r.get("year"):
            try:
                doc2year[str(r["doc_id"])] = int(r["year"])
            except Exception:
                pass

    # Split by type
    text_like = [it for it in (items or []) if (it.get("type") or "") in ("paragraph", "text", "section")]
    figures   = [it for it in (items or []) if (it.get("type") or "") == "figure"]
    tables    = [it for it in (items or []) if (it.get("type") or "") == "table"]

    clusters: List[Cluster] = []
    id2group: Dict[str, str] = {}

    # Text clusters
    if enable_text and text_like:
        c, m = _cluster_text(text_like, th=text_hamming_th)
        clusters.extend(c); id2group.update(m)

    # Figure clusters
    if enable_figure and figures:
        c, m = _cluster_figures(figures, th=image_hamming_th)
        clusters.extend(c); id2group.update(m)

    # Table clusters
    if enable_table and tables:
        c, m = _cluster_tables(tables)
        clusters.extend(c); id2group.update(m)

    # Mark items with group ids (singletons included)
    for it in items or []:
        iid = it["_item_id"]
        if iid in id2group:
            it["group_id"] = id2group[iid]

    # Representatives
    reps = _pick_representatives(clusters, item_by_id, doc2year)

    # Annotate & optionally remove dupes
    removed = 0
    for cl in clusters:
        rep_id = reps.get(cl.group_id, cl.rep_item_id)
        for iid in cl.members:
            it = item_by_id.get(iid)
            if not it:
                continue
            it["group_id"] = cl.group_id
            if iid == rep_id:
                it["is_rep"] = True
                it["dedupe_reason"] = cl.reason
            else:
                it["is_rep"] = False
                it["dup_of"] = rep_id
                it["dedupe_reason"] = cl.reason
                removed += 1

    out_items = [it for it in (items or []) if it.get("is_rep") or keep_duplicates]

    payload = {
        "run_id": run_id,
        "query": query,
        "stats": {
            "items_in": len(items or []),
            "clusters": len(clusters),
            "removed": removed if not keep_duplicates else 0,
            "items_out": len(out_items),
        },
        "clusters": [asdict(c) for c in clusters],
        "items": out_items,
        "refs": refs or [],
    }
    return payload

# ----------------------------- class API ---------------------------------------

@dataclass
class DedupeConfig:
    run_id: str
    text_hamming_th: int = 8
    image_hamming_th: int = 6
    enable_text: bool = True
    enable_figure: bool = True
    enable_table: bool = True
    keep_duplicates: bool = False

class Deduper:
    """
    Stable import used by citevizor_backend.py:
        from align.dedupe import Deduper

    Modes:
      - .run(run_id) -> Path                 # file-based (reads & writes storage/evidence/_rank)
      - .run_in_memory(items, refs, ...)     # pure in-memory; returns payload dict
      - .run_on_docs(doc_ids, run_id=None)   # back-compat shim to avoid AttributeError in backend
    """
    def __init__(self,
                 text_hamming_th: int = 8,
                 image_hamming_th: int = 6,
                 enable_text: bool = True,
                 enable_figure: bool = True,
                 enable_table: bool = True,
                 keep_duplicates: bool = False):
        self.text_hamming_th = int(max(0, text_hamming_th))
        self.image_hamming_th = int(max(0, image_hamming_th))
        self.enable_text = bool(enable_text)
        self.enable_figure = bool(enable_figure)
        self.enable_table = bool(enable_table)
        self.keep_duplicates = bool(keep_duplicates)

    def run(self, run_id: str) -> Path:
        obj = _load_rank_like(run_id)
        items: List[Dict[str, Any]] = obj.get("items", [])
        refs: List[Dict[str, Any]] = obj.get("refs", [])
        query: str = obj.get("query", "")
        payload = _dedupe_core(
            items, refs, run_id, query,
            self.text_hamming_th, self.image_hamming_th,
            self.enable_text, self.enable_figure, self.enable_table,
            self.keep_duplicates
        )
        out_path = _rank_dir() / f"{run_id}.dedup.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def run_in_memory(self,
                      items: List[Dict[str, Any]],
                      refs: List[Dict[str, Any]],
                      run_id: str,
                      query: str = "") -> Dict[str, Any]:
        return _dedupe_core(
            items, refs, run_id, query,
            self.text_hamming_th, self.image_hamming_th,
            self.enable_text, self.enable_figure, self.enable_table,
            self.keep_duplicates
        )

    # ---- Back-compat shim -----------------------------------------------------
    def run_on_docs(self, doc_ids: List[str], run_id: Optional[str] = None) -> Optional[Path]:
        """
        Provided to avoid AttributeError in pipelines that call Deduper().run_on_docs([...]).
        This method does NOT perform pack-based dedupe; it simply attempts to run() if run_id is supplied,
        otherwise returns None (no-op). Safe to call.
        """
        if run_id:
            try:
                return self.run(run_id)
            except Exception:
                return None
        # No-op if we don't know the run_id; dedupe relies on rank artifacts by design.
        return None

# ----------------------------- legacy function (CLI) ---------------------------

def dedupe_run(cfg: DedupeConfig) -> Path:
    dd = Deduper(
        text_hamming_th=cfg.text_hamming_th,
        image_hamming_th=cfg.image_hamming_th,
        enable_text=cfg.enable_text,
        enable_figure=cfg.enable_figure,
        enable_table=cfg.enable_table,
        keep_duplicates=cfg.keep_duplicates,
    )
    return dd.run(cfg.run_id)

# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor cross-document deduplication")
    ap.add_argument("--run-id", required=True, help="Ranking run_id")
    ap.add_argument("--no-text", action="store_true", help="Disable text/paragraph dedup")
    ap.add_argument("--no-figure", action="store_true", help="Disable figure/image dedup")
    ap.add_argument("--no-table", action="store_true", help="Disable table/csv dedup")
    ap.add_argument("--text-hamming", type=int, default=8, help="SimHash Hamming threshold for text (default 8)")
    ap.add_argument("--image-hamming", type=int, default=6, help="dHash Hamming threshold for figures (default 6)")
    ap.add_argument("--keep-duplicates", action="store_true", help="Keep duplicates (annotate only)")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    cfg = DedupeConfig(
        run_id=args.run_id,
        text_hamming_th=max(0, args.text_hamming),
        image_hamming_th=max(0, args.image_hamming),
        enable_text=not args.no_text,
        enable_figure=not args.no_figure,
        enable_table=not args.no_table,
        keep_duplicates=args.keep_duplicates,
    )
    out = dedupe_run(cfg)
    print(f"[OK] Dedupe -> {out}")
