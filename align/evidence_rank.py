# C:\CiteVizor\align\evidence_rank.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Evidence ranking (BM25 + recency/type boosts, optional cross-encoder rerank)

- Inputs: user query (+optional sub-queries), doc_ids to search (else auto-scan).
- Corpus: EvidencePack (paragraphs, tables[CSV head sample], figures[captions]).
- Ranking: Pure-Python BM25 with CJK-aware tokenization → type/recency boosts.
- Optional: cross-encoder reranker if locally available (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2'
  or 'BAAI/bge-reranker-large') loaded via sentence_transformers; if missing, safely skipped.
- Outputs: Top-K evidence list with provenance (doc_id/page/figure_no/table_no) to storage/evidence/_rank/<run_id>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Project schemas/paths (no heavy deps)
from schemas import (
    EVIDENCE_DIR,
    DOCS_DIR,
    STORAGE_DIR,
    EvidenceType,
)

ENV_FILE = "citevizor.env"

# ----------------------------- ENV loader -------------------------------------

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
    for k in (
        "RERANKER_ENABLED", "RERANKER_MODEL", "RERANKER_DEVICE",
        "RECENCY_HALFLIFE_YEARS", "TYPE_BOOST_FIGURE", "TYPE_BOOST_TABLE",
        "STOPWORDS_EXTRA",
    ):
        if os.getenv(k):
            out[k] = os.getenv(k)  # type: ignore
    return out


# ----------------------------- Tokenization -----------------------------------

# Basic English stopword list (compact) + optional extra
_EN_STOP = set("""
a an the of to in for on with by and or if is are was were be been being from as at this that these those it its into over under above below between about
""".split())

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7ff]")  # zh/jp/kr

def _is_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s))

_WORD_RE = re.compile(r"[A-Za-z0-9\.\-\+\%μΩ°/]{2,}")  # keep units/symbols commonly seen in papers

def _tokenize(text: str, extra_stop: Optional[Iterable[str]] = None) -> List[str]:
    """
    Mixed tokenizer:
      - If CJK present: generate overlapping bigrams for CJK blocks + Latin tokenization for others.
      - Else: Latin tokenization only.
    """
    if not text:
        return []
    t = text.lower()
    toks: List[str] = []
    if _is_cjk(t):
        # Split into chunks: CJK vs non-CJK
        buf = []
        is_cjk = None
        for ch in t:
            ch_is_cjk = _is_cjk(ch)
            if is_cjk is None or ch_is_cjk == is_cjk:
                buf.append(ch)
                is_cjk = ch_is_cjk
            else:
                toks += _emit_tokens("".join(buf), is_cjk)
                buf = [ch]
                is_cjk = ch_is_cjk
        if buf:
            toks += _emit_tokens("".join(buf), is_cjk)
    else:
        toks += [m.group(0) for m in _WORD_RE.finditer(t)]
    # stopword filtering (only for latin-ish tokens)
    stops = set(_EN_STOP)
    if extra_stop:
        stops |= set([w.strip().lower() for w in extra_stop if w.strip()])
    toks = [w for w in toks if not (w.isalpha() and w in stops)]
    return toks

def _emit_tokens(chunk: str, is_cjk: bool) -> List[str]:
    if not chunk:
        return []
    if is_cjk:
        # character bigrams (overlapping)
        chars = [c for c in chunk if not c.isspace()]
        return [chars[i] + chars[i+1] for i in range(len(chars)-1)] if len(chars) > 1 else chars
    else:
        return [m.group(0) for m in _WORD_RE.finditer(chunk)]


# ----------------------------- BM25 (pure Python) -----------------------------

@dataclass
class _BM25Index:
    doc_tokens: List[List[str]]
    df: Dict[str, int]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: List[str], tokenizer) -> "_BM25Index":
        doc_tokens = [tokenizer(d) for d in docs]
        df: Dict[str, int] = {}
        for toks in doc_tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        avgdl = sum(len(t) for t in doc_tokens) / max(1, len(doc_tokens))
        return cls(doc_tokens, df, avgdl)

    def score(self, query: str, tokenizer) -> List[float]:
        q_tokens = tokenizer(query)
        N = len(self.doc_tokens)
        # Precompute IDF
        idf = {}
        for w in set(q_tokens):
            n = self.df.get(w, 0)
            # Robertson-Sparck Jones idf
            idf[w] = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
        scores = []
        for toks in self.doc_tokens:
            dl = len(toks)
            tf = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in q_tokens:
                if w not in tf:
                    continue
                f = tf[w]
                denom = f + self.k1 * (1 - self.b + self.b * (dl / max(1e-9, self.avgdl)))
                s += idf.get(w, 0.0) * (f * (self.k1 + 1)) / denom
            scores.append(s)
        return scores


# ----------------------------- Utilities --------------------------------------

def _rank_dir() -> Path:
    d = EVIDENCE_DIR / "_rank"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _list_doc_ids() -> List[str]:
    return sorted([p.name for p in EVIDENCE_DIR.iterdir() if p.is_dir() and (p / "pack.json").exists() and p.name != "_rank"])

def _read_pack(doc_id: str) -> Dict[str, Any]:
    p = EVIDENCE_DIR / doc_id / "pack.json"
    if not p.exists():
        return {"doc_id": doc_id, "paragraphs": [], "tables": [], "figures": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"doc_id": doc_id, "paragraphs": [], "tables": [], "figures": []}

def _read_meta_year(doc_id: str) -> Optional[int]:
    m = DOCS_DIR / doc_id / f"{doc_id}.meta.json"
    if not m.exists():
        return None
    try:
        j = json.loads(m.read_text(encoding="utf-8"))
        return int(j["year"]) if j.get("year") else None
    except Exception:
        return None

def _csv_head_preview(csv_path: Path, max_rows: int = 5, max_cols: int = 8) -> Tuple[str, str]:
    """
    Return (flat_text_for_indexing, preview_for_snippet)
    """
    try:
        flat = []
        view = []
        with csv_path.open("r", encoding="utf-8-sig") as f:
            rdr = csv.reader(f)
            rows = []
            for i, row in enumerate(rdr):
                if i >= max_rows:
                    break
                rows.append(row[:max_cols])
        if not rows:
            return "", ""
        # Flat text (for BM25)
        for r in rows:
            flat.append(" ".join([c.strip() for c in r if c and c.strip()]))
        # Pretty preview
        head = rows[0]
        tail = rows[1:]
        view.append(" | ".join(head))
        for r in tail:
            view.append(" | ".join(r))
        return " . ".join(flat), "\n".join(view)
    except Exception:
        return "", ""

def _run_id(query: str) -> str:
    return sha256(f"{query}|{int(time.time())}".encode("utf-8")).hexdigest()[:12]

def _recency_boost(year: Optional[int], halflife: float = 4.0, max_bonus: float = 0.6) -> float:
    if not year:
        return 0.0
    now = time.gmtime().tm_year
    age = max(0, now - year)
    # Exponential decay -> scaled to max_bonus
    return max_bonus * math.exp(-age / max(0.5, halflife))

def _type_boost(ev_type: str, b_fig: float = 0.15, b_tab: float = 0.10) -> float:
    if ev_type == EvidenceType.figure.value or ev_type == "figure":
        return b_fig
    if ev_type == EvidenceType.table.value or ev_type == "table":
        return b_tab
    return 0.0

def _normalize_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    s = re.sub(r"\s+", " ", s)
    return s[:4000]

# --- Fallback: bootstrap minimal paragraph evidence from retrieval candidates ---

def _load_json_safe(p: Path) -> Dict[str, Any]:
    try:
        if p.exists() and p.stat().st_size > 0:
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _fallback_from_retrieval(run_id: str) -> Dict[str, Any]:
    """
    Build a minimal evidence payload using retrieval outputs when parsing yielded nothing.
    Returns a dict with keys: items (list), refs (list). Items are 'paragraph' type.
    """
    retr_dir = STORAGE_DIR / "retrieval"
    rank_dir = EVIDENCE_DIR / "_rank"
    candidates: List[Dict[str, Any]] = []

    # Prefer aggregated candidates if present (latest .agg.json)
    agg_files = list(retr_dir.glob("*.agg.json"))
    if agg_files:
        agg_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        obj = _load_json_safe(agg_files[0])
        for it in obj.get("items", []):
            candidates.append(it)

    # Fallback: try prior rank inputs
    if not candidates:
        for p in (rank_dir / f"{run_id}.json", rank_dir / f"{run_id}.citations.json"):
            obj = _load_json_safe(p)
            # possible shapes: {"results":[{"query":"..","items":[...]}, ...]}
            for blk in obj.get("results", []):
                for it in blk.get("items", []):
                    candidates.append(it)

    # Normalize to paragraph-like items (ensure both 'text' and 'snippet')
    items: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []
    seen_doc = set()

    for i, c in enumerate(candidates[:60], 1):
        title = (c.get("title") or c.get("name") or "").strip()
        if not title:
            continue
        url = (c.get("url") or c.get("link") or c.get("href") or c.get("source_url") or "")
        doi = c.get("doi") or c.get("DOI")
        year = c.get("year")
        if not year:
            date_like = c.get("date") or c.get("published_at") or c.get("pub_date")
            if isinstance(date_like, str):
                m = re.search(r"(\d{4})", date_like)
                if m:
                    year = int(m.group(1))

        base = (str(doi) if doi else title.lower()).encode("utf-8", errors="ignore")
        doc_id = sha256(base).hexdigest()[:16]
        if doc_id not in seen_doc:
            refs.append({"doc_id": doc_id, "title": title, "url": url, "doi": doi, "year": year})
            seen_doc.add(doc_id)

        snippet = (c.get("snippet") or c.get("abstract") or c.get("description") or title).strip()
        text = _normalize_text(snippet)
        if not text:
            continue
        items.append({
            "type": "paragraph",
            "doc_id": doc_id,
            "evidence_id": f"fb_{i:03d}",
            "page": None,
            "figure_no": None,
            "table_no": None,
            "text": text,
            "snippet": text[:320],
            "year": year,
        })

    return {"items": items, "refs": refs}


# ----------------------------- Reranker (optional) ----------------------------

class _OptionalCrossEncoder:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self.model = CrossEncoder(model_name, device=device or "cpu")
        except Exception:
            self.model = None

    def available(self) -> bool:
        return self.model is not None

    def rerank(self, query: str, pairs: List[Tuple[str, str]], top_k: int) -> List[int]:
        """
        Return indices of pairs sorted by cross-encoder score (desc).
        Pairs: [(doc_text, meta_id), ...] -> but we'll pass (query, text) to CE; keep meta separately.
        Here, 'pairs' is actually [(text, meta_id)] and we internally feed (query, text).
        """
        if not self.model or not pairs:
            return list(range(len(pairs)))
        texts = [(query, t[0]) for t in pairs]
        scores = self.model.predict(texts, convert_to_numpy=True)
        idxs = list(range(len(pairs)))
        idxs.sort(key=lambda i: float(scores[i]), reverse=True)
        return idxs[:top_k]


# ----------------------------- Ranker core ------------------------------------

@dataclass
class EvidenceRankConfig:
    project_root: Path
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: Optional[str] = None
    recency_halflife_years: float = 4.0
    type_boost_figure: float = 0.15
    type_boost_table: float = 0.10
    stopwords_extra: List[str] = None  # type: ignore

    @classmethod
    def from_env(cls, project_root: Path) -> "EvidenceRankConfig":
        env = _load_env(project_root)
        return cls(
            project_root=project_root,
            reranker_enabled=(env.get("RERANKER_ENABLED", "1") not in ("0", "false", "False")),
            reranker_model=env.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            reranker_device=env.get("RERANKER_DEVICE"),
            recency_halflife_years=float(env.get("RECENCY_HALFLIFE_YEARS", "4.0")),
            type_boost_figure=float(env.get("TYPE_BOOST_FIGURE", "0.15")),
            type_boost_table=float(env.get("TYPE_BOOST_TABLE", "0.10")),
            stopwords_extra=[w for w in (env.get("STOPWORDS_EXTRA", "") or "").split(",") if w.strip()],
        )


class EvidenceRanker:
    def __init__(self, cfg: EvidenceRankConfig):
        self.cfg = cfg
        # Optional CE
        self.ce = _OptionalCrossEncoder(cfg.reranker_model, cfg.reranker_device) if cfg.reranker_enabled else _OptionalCrossEncoder("none")
        # Tokenizer with extra stops bound
        self.tokenizer = lambda s: _tokenize(s, cfg.stopwords_extra)

    # Public API
    def rank(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        sub_queries: Optional[List[str]] = None,
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """
        Return a dict report with run_id, query, items (ranked), and write to _rank/<run_id>.json
        """
        # Prefer parsed packs; if none, we'll try retrieval-fallback later
        doc_ids = doc_ids or _list_doc_ids()

        # Build corpus from packs
        items: List[Dict[str, Any]] = []
        refs_from_packs: List[Dict[str, Any]] = []  # not used in output but available for future
        if doc_ids:
            for doc_id in doc_ids:
                pack = _read_pack(doc_id)
                year = _read_meta_year(doc_id)
                # paragraphs
                for e in pack.get("paragraphs", []):
                    text = _normalize_text(e.get("text", ""))
                    if not text:
                        continue
                    items.append({
                        "doc_id": doc_id,
                        "evidence_id": e.get("id"),
                        "type": e.get("type") or "paragraph",
                        "page": e.get("page"),
                        "figure_no": e.get("figure_no"),
                        "table_no": e.get("table_no"),
                        "text": text,
                        "snippet": text[:320],
                        "year": year,
                    })
                # figures (captions)
                for e in pack.get("figures", []):
                    caption = _normalize_text(e.get("caption", ""))
                    if not caption:
                        continue
                    items.append({
                        "doc_id": doc_id,
                        "evidence_id": e.get("id"),
                        "type": "figure",
                        "page": e.get("page"),
                        "figure_no": e.get("figure_no"),
                        "table_no": None,
                        "text": caption,
                        "snippet": caption[:320],
                        "year": year,
                    })
                # tables (CSV preview)
                for e in pack.get("tables", []):
                    csv_path = e.get("csv_path")
                    if not csv_path:
                        continue
                    csv_path = Path(csv_path)
                    flat, preview = _csv_head_preview(csv_path)
                    flat = _normalize_text(flat)
                    if not flat:
                        continue
                    items.append({
                        "doc_id": doc_id,
                        "evidence_id": e.get("id"),
                        "type": "table",
                        "page": e.get("page"),
                        "table_no": e.get("table_no"),
                        "figure_no": None,
                        "text": flat,
                        "snippet": preview[:320],
                        "year": year,
                    })

        # Fallback when no items parsed from packs
        used_fallback = False
        if not items:
            # No packs or packs empty: bootstrap from retrieval candidates
            rid_probe = _run_id(query)  # only for probing filenames; actual output rid will be regenerated below
            fb = _fallback_from_retrieval(rid_probe)
            fb_items = fb.get("items", [])
            if fb_items:
                items = fb_items
                used_fallback = True

        if not items:
            # still nothing – stop with explicit guidance
            raise RuntimeError(
                "No evidence content available to rank. "
                "Parsing yielded no paragraphs/tables/figures and retrieval fallback was empty. "
                "Check: (1) fetch/downloader saved HTML/PDF correctly; "
                "(2) parse wrote storage/evidence/<doc_id>/pack.json with non-empty lists."
            )

        # Build BM25 index
        docs_texts = [it["text"] for it in items]
        bm25 = _BM25Index.build(docs_texts, self.tokenizer)

        # Score: support sub-queries (plan.queries) → take max per item
        queries = sub_queries if (sub_queries and len(sub_queries) > 0) else [query]
        bm_scores_accum = [0.0] * len(items)
        for q in queries:
            s = bm25.score(q, self.tokenizer)
            for i in range(len(items)):
                bm_scores_accum[i] = max(bm_scores_accum[i], s[i])  # max-pooling across sub-queries

        # Apply boosts
        scores = []
        for i, it in enumerate(items):
            base = bm_scores_accum[i]
            # type boost
            tb = _type_boost(it["type"], self.cfg.type_boost_figure, self.cfg.type_boost_table)
            # recency boost
            rb = _recency_boost(it.get("year"), self.cfg.recency_halflife_years)
            scores.append(base + tb + rb)

        # Initial ranking
        order = list(range(len(items)))
        order.sort(key=lambda i: scores[i], reverse=True)

        # Optional CE rerank top-N window
        WINDOW = min(200, len(order))
        if self.ce.available() and WINDOW > 0:
            pairs = [(items[i]["text"], items[i]["evidence_id"]) for i in order[:WINDOW]]
            reranked_local = self.ce.rerank(query, pairs, top_k=WINDOW)
            order = [order[idx] for idx in reranked_local] + order[WINDOW:]

        # Dedup by normalized snippet hash (keep first / highest score)
        seen = set()
        final = []
        for i in order:
            it = items[i]
            key = sha256((it["type"] + "|" + it["snippet"]).encode("utf-8")).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            it_out = {
                "rank": len(final) + 1,
                "score": round(float(scores[i]), 4),
                "type": it["type"],
                "doc_id": it["doc_id"],
                "evidence_id": it["evidence_id"],
                "page": it.get("page"),
                "figure_no": it.get("figure_no"),
                "table_no": it.get("table_no"),
                "snippet": it["snippet"],
                "year": it.get("year"),
            }
            final.append(it_out)
            if len(final) >= top_k:
                break

        rid = _run_id("|".join(queries))
        out = {
            "run_id": rid,
            "query": query,
            "sub_queries": queries,
            "generated_at": int(time.time()),
            "items": final,
            "meta": {
                "doc_ids": doc_ids if doc_ids else [],
                "bm25_docs": len(items),
                "reranker_used": self.ce.available(),
                "recency_halflife_years": self.cfg.recency_halflife_years,
                "type_boosts": {"figure": self.cfg.type_boost_figure, "table": self.cfg.type_boost_table},
                "fallback_used": used_fallback,
            },
        }
        out_path = _rank_dir() / f"{rid}.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor evidence ranker")
    ap.add_argument("--query", required=True, help="User query")
    ap.add_argument("--doc-ids", default="", help="Comma-separated doc_ids; if empty, auto-scan all packs")
    ap.add_argument("--subq", default="", help="Optional sub-queries separated by || (from planner)")
    ap.add_argument("--topk", type=int, default=50, help="Top-K to output")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg = EvidenceRankConfig.from_env(project_root)
    ranker = EvidenceRanker(cfg)
    doc_ids = [d.strip() for d in args.doc_ids.split(",") if d.strip()] or None
    subq = [s.strip() for s in args.subq.split("||") if s.strip()] if args.subq else None
    out = ranker.rank(query=args.query, doc_ids=doc_ids, sub_queries=subq, top_k=args.topk)
    out_path = _rank_dir() / f"{out['run_id']}.json"
    print(f"[OK] Ranked evidence -> {out_path}")
    print(f"Top {min(len(out['items']), args.topk)} results:")
    for it in out["items"][: min(10, len(out["items"]))]:
        print(f"  #{it['rank']:02d} [{it['type']}] {it['doc_id']} p.{it.get('page')}: {it['snippet'][:120]} ... (score={it['score']})")
