# /workspace/retrieval/aggregate_rank.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Aggregate ranking (RRF + recency + MMR) with backward-compatible API.

Public API (stable & forgiving)
  - rank(plan, results_by_query, top_k=50, cfg=None) -> Dict
  - aggregate_candidates(*args, **kwargs) -> List[Candidate]
      Accepts ALL of the following forms:
        • aggregate_candidates(plan, results_by_query, top_k=50, ...)
        • aggregate_candidates(results_by_query, top_k=50, ...)              # plan omitted
        • aggregate_candidates(results=[...], top_k=50, ...)                  # flat list
        • aggregate_candidates(per_query={q:[...]}, topk_rank=60, ...)       # alias keys
        • aggregate_candidates(items=[...], ...)                              # alias
      Returns List[Candidate] sorted by fused score (desc).

Notes:
- This module is NOT the root cause of repeated download warnings, but it lacked
  normalized PDF hints. We emit `pdf_url` and `trace.normalizer` so the
  downloader can prefer direct/open links and you can audit the applied rule.
- [image-safe] When Candidate.source == "image", we deliberately skip any PDF
  normalization/bonus; images will flow as-is through the pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas import Candidate, STORAGE_DIR  # project contracts & paths

__all__ = [
    "AggregateConfig",
    "aggregate_candidates",
    "rank",
]

# ----------------------------- text & url utils -------------------------------

_WORD = re.compile(r"[A-Za-z0-9]+", re.UNICODE)
_HOST_RE = re.compile(r"^https?://([^/]+)/", re.I)
_PDF_EXT_RE = re.compile(r"\.pdf(?:$|\?)", re.I)

# OA-leaning domains with typically direct or stable PDF links
_DOWNLOAD_HOST_WHITELIST = {
    "arxiv.org",
    "aclanthology.org",
    "openreview.net",
    "biorxiv.org",
    "medrxiv.org",
    "hal.science",
    "osf.io",
}

def _norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", (s or "").strip())
    s = re.sub(r"\s+", " ", s)
    return s

def _canon_title(s: str) -> str:
    t = _norm_text(s).lower()
    t = re.sub(r"[-–—:;,.!?\"'`~()\[\]{}<>]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _title_tokens(s: str) -> List[str]:
    return _WORD.findall(_canon_title(s))[:64]

def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    if inter == 0:
        return 0.0
    return inter / float(len(A | B))

def _host_of(url: Optional[str]) -> str:
    if not url:
        return ""
    m = _HOST_RE.search(url)
    host = (m.group(1).lower() if m else "").replace("www.", "")
    return host

def _year_or_na(y: Any) -> Optional[int]:
    try:
        yi = int(y)
        now = time.gmtime().tm_year
        if 1900 <= yi <= now:
            return yi
    except Exception:
        pass
    return None

def _is_probably_pdf(url: Optional[str]) -> bool:
    if not url:
        return False
    return bool(_PDF_EXT_RE.search(url))

# ----------------------------- URL normalization ------------------------------

def _normalize_to_pdf(url: Optional[str], *, source: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to infer a stable, direct PDF URL from a landing/abstract page.
    Returns (pdf_url, rule_name) or (None, None) if no rule matched.

    [image-safe] For image candidates (source == "image"), skip normalization.
    Rules are conservative and only fire on well-known domains.
    """
    # [image-safe] never normalize images to PDF
    if (source or "").lower() == "image":
        return None, None

    if not url:
        return None, None
    u = url.strip()
    host = _host_of(u)

    # Already a PDF -> no change, but mark the rule
    if _is_probably_pdf(u):
        return u, "pass_through_pdf"

    # arXiv: abs -> pdf
    # https://arxiv.org/abs/1706.03762  => https://arxiv.org/pdf/1706.03762.pdf
    if host == "arxiv.org":
        m = re.search(r"/abs/([0-9]+\.[0-9]+)(v[0-9]+)?", u)
        if m:
            pid = m.group(1)
            return f"https://arxiv.org/pdf/{pid}.pdf", "arxiv_abs_to_pdf"

    # ACL Anthology pages are often direct PDF already; if not, try suffix
    if host == "aclanthology.org":
        # https://aclanthology.org/P18-1031 -> append .pdf
        if not _is_probably_pdf(u) and re.search(r"/[A-Z0-9\-]+$", u):
            return u.rstrip("/") + ".pdf", "acl_append_pdf"

    # OpenReview: either already /pdf?id=... or /forum?id=...
    if host == "openreview.net":
        if "pdf?id=" in u:
            return u, "openreview_pass_pdf"
        m = re.search(r"/forum\?id=([A-Za-z0-9\-_]+)", u)
        if m:
            return f"https://openreview.net/pdf?id={m.group(1)}", "openreview_forum_to_pdf"

    # bioRxiv / medRxiv: common pattern ends with 'full.pdf'
    if host in {"biorxiv.org", "medrxiv.org"}:
        # if it's a content path without .pdf, try 'full.pdf'
        if not _is_probably_pdf(u) and "/content/" in u:
            if not u.endswith("/"):
                u = u + "/"
            return u + "full.pdf", f"{host}_full_pdf"

    # HAL
    if host == "hal.science":
        # https://hal.science/hal-012345/document -> append .pdf if missing
        if "/document" in u and not _is_probably_pdf(u):
            return u + ".pdf", "hal_document_pdf"

    # OSF: many files are already direct; no strong generic rule
    if host == "osf.io":
        if _is_probably_pdf(u):
            return u, "osf_pass_pdf"

    return None, None

def _dedup_key(c: Candidate) -> str:
    # Strong key priority: DOI -> canonical title -> canonical URL
    if c.doi:
        return f"doi::{str(c.doi).strip().lower()}"
    ct = _canon_title(c.title)
    if ct:
        return "title::" + sha256(ct.encode("utf-8")).hexdigest()[:16]
    u = (c.url or "").strip().lower()
    return "url::" + sha256(u.encode("utf-8")).hexdigest()[:16] if u else "rand::" + sha256(c.title.encode("utf-8")).hexdigest()[:8]

# ----------------------------- datamodel --------------------------------------

@dataclass
class AggregateConfig:
    rrf_k: int = 60                    # RRF constant
    recency_halflife_years: float = 4.0
    mmr_lambda: float = 0.80           # relevance vs diversity
    max_per_domain: int = 4
    top_k: int = 50
    # Downloadability booster (small additive bonus to base RRF before recency)
    dl_pdf_bonus: float = 0.12         # .pdf URL
    dl_oa_host_bonus: float = 0.08     # whitelisted OA-like hosts
    dl_cap: float = 0.18               # maximum additive bonus per item

@dataclass
class TraceScore:
    rrf: float
    recency: float
    base: float
    dl_bonus: float
    final: float
    per_query_ranks: Dict[str, int]
    host: str
    normalizer: Optional[str] = None   # rule name applied for pdf_url (if any)

@dataclass
class AggCandidate:
    cand: Candidate
    score: float
    trace: TraceScore
    query_hits: int
    pdf_url: Optional[str] = None      # normalized direct PDF if detected

# ----------------------------- core fusion ------------------------------------

def _rrf_score(rank: int, k: int) -> float:
    return 1.0 / (k + rank)

def _recency_weight(year: Optional[int], halflife: float) -> float:
    if not year:
        return 0.90  # slightly penalize unknown years
    now = time.gmtime().tm_year
    dt = max(0, now - int(year))
    # Exponential decay: w = 2^(-dt/halflife)
    return pow(2.0, -dt / max(halflife, 0.5))

def _downloadability_bonus(url: Optional[str], cfg: AggregateConfig, *, source: Optional[str] = None) -> float:
    """Small additive bonus to RRF if the link looks directly downloadable.
    [image-safe] Images不参与PDF可下载性加分。
    """
    if (source or "").lower() == "image":
        return 0.0
    if not url:
        return 0.0
    host = _host_of(url)
    bonus = 0.0
    if _is_probably_pdf(url):
        bonus += cfg.dl_pdf_bonus
    if host in _DOWNLOAD_HOST_WHITELIST:
        bonus += cfg.dl_oa_host_bonus
    # cap to avoid over-influence
    return min(cfg.dl_cap, bonus)

def _fuse_rrf(results_by_query: Dict[str, List[Candidate]], cfg: AggregateConfig) -> Dict[str, AggCandidate]:
    # Build per-candidate accumulators keyed by dedup_key
    acc: Dict[str, AggCandidate] = {}
    for q, items in (results_by_query or {}).items():
        for idx, c in enumerate(items or [], 1):  # 1-based rank
            key = _dedup_key(c)
            rrf = _rrf_score(idx, cfg.rrf_k)
            year = _year_or_na(c.year)
            rec = _recency_weight(year, cfg.recency_halflife_years)
            host = _host_of(c.url)

            # [image-safe] image 源不吃 PDF bonus
            dlb = _downloadability_bonus(c.url, cfg, source=c.source)

            # compute normalized pdf url once per view
            norm_pdf, rule = _normalize_to_pdf(c.url, source=c.source)  # [image-safe]

            if key not in acc:
                base = rrf + dlb                # add dl bonus BEFORE recency
                final = base * rec
                acc[key] = AggCandidate(
                    cand=Candidate(
                        title=_norm_text(c.title),
                        url=c.url,
                        year=year,
                        source=c.source,
                        score=final,
                        doi=c.doi,
                    ),
                    score=final,
                    trace=TraceScore(
                        rrf=rrf, recency=rec, base=base - dlb, dl_bonus=dlb, final=final,
                        per_query_ranks={q: idx}, host=host, normalizer=rule
                    ),
                    query_hits=1,
                    pdf_url=norm_pdf
                )
            else:
                a = acc[key]
                # accumulate RRF; keep max recency (favor newer)
                a.trace.rrf += rrf
                a.trace.base += rrf

                # accumulate/refresh normalization: prefer a direct PDF if any view found one
                if not a.pdf_url and norm_pdf:
                    a.pdf_url = norm_pdf
                    a.trace.normalizer = rule or a.trace.normalizer

                # If this view has additional dl bonus (e.g., another query yielded .pdf), apply but cap
                added_dlb = _downloadability_bonus(c.url, cfg, source=c.source)  # [image-safe]
                new_dlb = min(cfg.dl_cap, a.trace.dl_bonus + added_dlb)
                a.trace.dl_bonus = new_dlb

                prev_rank = a.trace.per_query_ranks.get(q)
                a.trace.per_query_ranks[q] = min(idx, prev_rank) if prev_rank else idx
                a.trace.recency = max(a.trace.recency, rec)  # keep stronger (newer)

                base_total = a.trace.base + a.trace.dl_bonus
                a.score = base_total * a.trace.recency
                a.trace.final = a.score
                a.cand.score = a.score
    return acc

# ----------------------------- MMR selection ----------------------------------

def _mmr_select(cands: List[AggCandidate], cfg: AggregateConfig) -> List[AggCandidate]:
    """
    Select top_k with MMR and domain cap to ensure diversity.
    Similarity uses Jaccard over title tokens.
    """
    # Precompute tokens for similarity
    for a in cands:
        setattr(a, "_tok", _title_tokens(a.cand.title))

    selected: List[AggCandidate] = []
    domain_count: Dict[str, int] = {}
    remaining = cands[:]

    while remaining and len(selected) < cfg.top_k:
        best_i = -1
        best_val = -1e9
        for i, a in enumerate(remaining):
            host = _host_of(a.cand.url)
            dom_penalty = 0.0
            if host:
                c = domain_count.get(host, 0)
                if c >= cfg.max_per_domain:
                    dom_penalty = 0.5  # suppress heavily repeated host
            if not selected:
                div = 0.0
            else:
                sim_max = 0.0
                for s in selected:
                    sim = _jaccard(getattr(a, "_tok"), getattr(s, "_tok"))
                    if sim > sim_max:
                        sim_max = sim
                div = sim_max
            val = cfg.mmr_lambda * a.score - (1.0 - cfg.mmr_lambda) * div - dom_penalty
            if val > best_val:
                best_val = val
                best_i = i
        choice = remaining.pop(best_i)
        selected.append(choice)
        host = _host_of(choice.cand.url)
        if host:
            domain_count[host] = domain_count.get(host, 0) + 1

    return selected

# ----------------------------- public API -------------------------------------

def rank(
    plan: Optional[Dict[str, Any]],
    results_by_query: Dict[str, List[Candidate]],
    top_k: int = 50,
    cfg: Optional[AggregateConfig] = None
) -> Dict[str, Any]:
    """
    Fuse multi-query results into a single ranked list.

    Returns a JSON-serializable dict:
      {
        "plan_id": "...",
        "stats": {...},
        "items": [
          {
            "title": "...",
            "url": "...",            # original url (PDF/web/image)
            "pdf_url": "...",        # NEW: normalized direct PDF if available (None for images)
            "year": 2020,
            "source": "scholar" | "web" | "news" | "image",
            "doi": "...",
            "score": 0.123,
            "rank": 1,
            "trace": {
              "rrf": ...,
              "recency": ...,
              "base": ...,
              "dl_bonus": ...,
              "final": ...,
              "per_query_ranks": {...},
              "host": "arxiv.org",
              "hits": 2,
              "normalizer": "arxiv_abs_to_pdf"   # rule name if applied; None for images
            }
          }, ...
        ]
      }
    """
    cfg = cfg or AggregateConfig(top_k=top_k)
    acc = _fuse_rrf(results_by_query, cfg)
    pooled = list(acc.values())
    pooled.sort(key=lambda a: a.score, reverse=True)
    cfg.top_k = top_k
    selected = _mmr_select(pooled, cfg)

    out_items: List[Dict[str, Any]] = []
    for i, a in enumerate(selected, 1):
        o = {
            "title": a.cand.title,
            "url": a.cand.url,
            "pdf_url": a.pdf_url,                     # may be None (esp. for images)
            "year": a.cand.year,
            "source": a.cand.source,
            "doi": a.cand.doi,
            "score": round(a.score, 6),
            "rank": i,
            "trace": {
                "rrf": round(a.trace.rrf, 6),
                "recency": round(a.trace.recency, 6),
                "base": round(a.trace.base, 6),
                "dl_bonus": round(a.trace.dl_bonus, 6),
                "final": round(a.trace.final, 6),
                "per_query_ranks": a.trace.per_query_ranks,
                "host": a.trace.host,
                "hits": a.query_hits,
                "normalizer": a.trace.normalizer,
            }
        }
        out_items.append(o)

    plan_id = (plan or {}).get("plan_id") or ""
    return {
        "plan_id": plan_id,
        "stats": {
            "queries": len(results_by_query or {}),
            "pooled": len(pooled),
            "selected": len(out_items),
            "rrf_k": cfg.rrf_k,
            "recency_halflife_years": cfg.recency_halflife_years,
            "mmr_lambda": cfg.mmr_lambda,
            "max_per_domain": cfg.max_per_domain,
            "dl_pdf_bonus": cfg.dl_pdf_bonus,
            "dl_oa_host_bonus": cfg.dl_oa_host_bonus,
            "dl_cap": cfg.dl_cap,
        },
        "items": out_items,
    }

# ---- tolerant, back-compat wrapper -------------------------------------------

def _as_candidate(obj: Any) -> Optional[Candidate]:
    if isinstance(obj, Candidate):
        return obj
    if isinstance(obj, dict):
        try:
            return Candidate(
                title=obj.get("title") or "",
                url=obj.get("url"),
                year=obj.get("year"),
                source=obj.get("source") or "scholar",
                score=float(obj.get("score", 0.0)),
                doi=obj.get("doi"),
            )
        except Exception:
            return None
    return None

def _normalize_results_by_query(val: Any) -> Dict[str, List[Candidate]]:
    if val is None:
        return {}
    if isinstance(val, list):
        bucket: List[Candidate] = []
        for it in val:
            c = _as_candidate(it)
            if c:
                bucket.append(c)
        return {"q1": bucket}
    if isinstance(val, dict):
        out: Dict[str, List[Candidate]] = {}
        for k, v in val.items():
            raw_items = None
            if isinstance(v, list):
                raw_items = v
            elif isinstance(v, dict):
                raw_items = (
                    v.get("items")
                    or v.get("results")
                    or v.get("candidates")
                    or v.get("docs")
                    or v.get("hits")
                    or []
                )
            else:
                raw_items = [v]
            bucket: List[Candidate] = []
            for it in (raw_items or []):
                c = _as_candidate(it)
                if c:
                    bucket.append(c)
            out[str(k) or "q"] = bucket
        return out
    return {}

def aggregate_candidates(*args: Any, **kwargs: Any) -> List[Candidate]:
    """
    Backward-compatible glue. Accepts many signatures (see module docstring).
    """
    plan: Optional[Dict[str, Any]] = None
    results_any: Any = None

    if len(args) == 2:
        if isinstance(args[0], dict) or args[0] is None:
            plan = args[0]
            results_any = args[1]
        else:
            results_any = args[0]
            plan = args[1] if isinstance(args[1], dict) else None
    elif len(args) == 1:
        results_any = args[0]

    if plan is None:
        for key in ("plan", "search_plan", "planning"):
            if key in kwargs:
                plan = kwargs.pop(key)
                break

    if results_any is None:
        for key in ("results_by_query", "per_query", "query_results", "results", "candidates", "items"):
            if key in kwargs:
                results_any = kwargs.pop(key)
                break
        if results_any is None:
            buckets = {}
            for key in ("scholar", "web", "arxiv", "crossref", "patent", "news", "university", "image"):
                if key in kwargs:
                    buckets[key] = kwargs.pop(key)
            if buckets:
                results_any = buckets

    results_by_query: Dict[str, List[Candidate]] = _normalize_results_by_query(results_any)

    top_k = kwargs.pop("top_k", None)
    if top_k is None:
        top_k = kwargs.pop("topk", None)
    if top_k is None:
        top_k = kwargs.pop("topk_rank", None)
    if top_k is None:
        top_k = kwargs.pop("max_total", None)
    if top_k is None:
        top_k = 50
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 50

    payload = rank(plan if isinstance(plan, dict) else None, results_by_query, top_k=top_k, cfg=None)

    # Back-compat: return Candidate list (without pdf_url). Downstream that reads
    # the JSON payload can still access `pdf_url` from the serialized file.
    items = payload.get("items", [])
    out: List[Candidate] = []
    for it in items:
        c = _as_candidate(it)
        if c:
            out.append(c)
    return out

# ----------------------------- file I/O (CLI) ---------------------------------

def _read_candidates(obj: Any) -> Candidate:
    return Candidate(
        title=obj.get("title") or "",
        url=obj.get("url"),
        year=obj.get("year"),
        source=obj.get("source") or "scholar",
        score=obj.get("score") or 0.0,
        doi=obj.get("doi"),
    )

def _load_input(path: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, List[Candidate]]]:
    root = json.loads(path.read_text(encoding="utf-8"))
    plan = root.get("plan") if isinstance(root, dict) else None
    res: Dict[str, List[Candidate]] = {}
    if isinstance(root, dict) and isinstance(root.get("results"), list):
        for block in root["results"]:
            q = (block.get("query") or "").strip()
            items = [_read_candidates(x) for x in (block.get("items") or [])]
            res[q or f"q{len(res)+1}"] = items
    elif isinstance(root, list):
        res["q1"] = [_read_candidates(x) for x in root]
    else:
        raise ValueError("Invalid input JSON structure.")
    return plan, res

def _save_output(payload: Dict[str, Any], plan: Optional[Dict[str, Any]], in_path: Path, out_path: Optional[Path]) -> Path:
    if out_path:
        p = out_path
    else:
        if plan and plan.get("plan_id"):
            d = STORAGE_DIR / "retrieval"
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{plan['plan_id']}.agg.json"
        else:
            p = in_path.with_suffix(".agg.json")
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor aggregate ranker (RRF + recency + MMR + downloadability)")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON as described in module docstring")
    ap.add_argument("--out", dest="out_path", default="", help="Output JSON path (optional)")
    ap.add_argument("--top-k", type=int, default=50, help="Final number of candidates to select")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF constant (higher -> flatter)")
    ap.add_argument("--recency-halflife", type=float, default=4.0, help="Years half-life for recency boost")
    ap.add_argument("--mmr-lambda", type=float, default=0.80, help="MMR lambda: relevance-vs-diversity tradeoff")
    ap.add_argument("--max-per-domain", type=int, default=4, help="Domain cap to prevent publisher dominance")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else None

    plan, res = _load_input(in_path)
    cfg = AggregateConfig(
        rrf_k=max(1, args.rrf_k),
        recency_halflife_years=max(0.25, args.recency_halflife),
        mmr_lambda=min(0.99, max(0.01, args.mmr_lambda)),
        max_per_domain=max(1, args.max_per_domain),
        top_k=max(1, args.top_k),
    )
    payload = rank(plan, res, top_k=cfg.top_k, cfg=cfg)
    out = _save_output(payload, plan, in_path, out_path)
    print(f"[OK] Aggregate ranking -> {out}")
