# C:\CiteVizor\retrieval\serper_web.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Serper Web/News/Patents/Universities/Images retriever (hardened)

Why this file:
- It is *not* the source of your current WARN/download issues (you ran with --k-web 0),
  but we harden it so that when web is enabled it never pollutes the pipeline.

What changed (vs previous):
- Added Serper Images vertical ("images") to fetch image direct links as Candidates with source="image".
- Endpoints discovery now includes images; robust normalization/dedup for imageUrl.
- All existing verticals' behavior remains unchanged and interface-compatible.

This module does not fetch PDFs or images; it only returns Candidate(url/title/…).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Project schemas
from schemas import Candidate  # dataclass: title, url, year, source, score, doi

# Optional central config
try:
    from config import load_config  # type: ignore
except Exception:
    load_config = None  # type: ignore

# ------------------------------------------------------------------------------
# Config & env helpers
# ------------------------------------------------------------------------------

DEFAULT_SERPER_SEARCH = "https://google.serper.dev/search"
DEFAULT_SERPER_NEWS   = "https://google.serper.dev/news"
DEFAULT_SERPER_IMAGES = "https://google.serper.dev/images"
ENV_FILE = "citevizor.env"

def _read_env_file(project_root: Path) -> Dict[str, str]:
    p = project_root / ENV_FILE
    if not p.exists():
        return {}
    out: Dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out

def _discover_root() -> Path:
    canonical = Path(r"C:\CiteVizor")
    return canonical if canonical.exists() else Path(__file__).resolve().parents[1]

def _get_serper_key() -> str:
    k = os.environ.get("SERPER_API_KEY")
    if k:
        return k
    if load_config:
        try:
            cfg = load_config()
            if getattr(cfg, "retrieval", None) and cfg.retrieval.serper_api_key:
                return cfg.retrieval.serper_api_key  # type: ignore[attr-defined]
        except Exception:
            pass
    env = _read_env_file(_discover_root())
    k = env.get("SERPER_API_KEY", "")
    if k:
        return k
    raise RuntimeError("SERPER_API_KEY missing. Set env or put it in citevizor.env")

def _get_endpoints() -> Tuple[str, str, str]:
    """
    Allow overriding base via SERPER_BASE, e.g. https://google.serper.dev
    """
    base = os.environ.get("SERPER_BASE", "").rstrip("/")
    if base:
        return f"{base}/search", f"{base}/news", f"{base}/images"
    return DEFAULT_SERPER_SEARCH, DEFAULT_SERPER_NEWS, DEFAULT_SERPER_IMAGES

# ------------------------------------------------------------------------------
# Query shaping
# ------------------------------------------------------------------------------

_UNI_SITES = [
    "site:.edu", "site:.edu.cn", "site:.edu.au", "site:.edu.sg",
    "site:.ac.uk", "site:.ac.jp", "site:.ac.kr", "site:.ac.in", "site:.ac.id", "site:.ac.cn",
]

_PATENT_SITES = [
    "site:patents.google.com",
    "site:worldwide.espacenet.com",
]

# academic/publisher whitelist (best-effort) for images use-cases
_ACADEMIC_HINTS = [
    "site:arxiv.org", "site:ieee.org", "site:ieeexplore.ieee.org", "site:dl.acm.org",
    "site:springer.com", "site:nature.com", "site:sciencedirect.com", "site:cell.com",
    "site:mdpi.com", "site:wiley.com", "site:oup.com", "site:plos.org", "site:frontiersin.org",
    "site:ncbi.nlm.nih.gov", "site:researchgate.net"
]

def _join_sites(sites: List[str]) -> str:
    return " OR ".join(sites)

def _augment_query(q: str, vertical: str, sites_csv: str = "") -> str:
    """
    Add site filters for some verticals unless caller already added site:...
    For images, if the caller didn't provide sites, we don't force a filter here,
    because upstream pipeline may prefer broader recall and filter later.
    """
    q = (q or "").strip()
    site_filter = ""
    if sites_csv:
        xs = [s.strip() for s in sites_csv.split(",") if s.strip()]
        if xs:
            site_filter = _join_sites(xs)
    elif vertical == "patents":
        site_filter = _join_sites(_PATENT_SITES)
    elif vertical == "universities":
        site_filter = _join_sites(_UNI_SITES)
    elif vertical == "images":
        # Optional: uncomment to bias toward academic domains by default
        # site_filter = _join_sites(_ACADEMIC_HINTS)
        site_filter = ""

    if site_filter:
        if "site:" in q:
            return q
        return f"{q} ({site_filter})"
    return q

def _tbs_of(time_range: str) -> Optional[str]:
    # Google 'qdr' codes for recency
    tr = (time_range or "").strip().lower()
    return {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}.get(tr)

# ------------------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------------------

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.I)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def _find_doi(*texts: Optional[str]) -> Optional[str]:
    for t in texts:
        if not t:
            continue
        m = _DOI_RE.search(t)
        if m:
            return m.group(0).rstrip(").,;]")
    return None

def _year_from_date(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    m = _YEAR_RE.search(date_str)
    if not m:
        return None
    try:
        y = int(m.group(0))
        now = time.gmtime().tm_year
        if 1900 <= y <= now:
            return y
    except Exception:
        pass
    return None

def _source_label(vertical: str) -> str:
    if vertical in ("patents", "universities"):
        return vertical
    if vertical == "images":
        return "image"
    return "news" if vertical == "news" else "web"

def _is_http_url(u: Optional[str]) -> bool:
    if not u:
        return False
    u = u.strip()
    return u.startswith("http://") or u.startswith("https://")

def _normalize_url(u: str) -> str:
    # Basic canonicalization; do not over-normalize to avoid merging distinct URLs.
    return u.strip()

# ------------------------------------------------------------------------------
# HTTP call with retry/backoff
# ------------------------------------------------------------------------------

def _serper_call(url: str, payload: Dict[str, Any], api_key: str, timeout_s: int = 12) -> Dict[str, Any]:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    backoffs = [0.5, 1.0, 2.0]  # small backoff for 429/5xx
    for i, delay in enumerate([0.0] + backoffs):
        if delay:
            time.sleep(delay)
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            if r.status_code in (429, 500, 502, 503, 504):
                # retryable
                if i < len(backoffs):
                    continue
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            # fail fast for 401/403/400 etc.
            raise RuntimeError(f"Serper HTTP {getattr(e.response,'status_code',-1)}: {getattr(e.response,'text','')[:300]}")
        except requests.RequestException as e:
            if i < len(backoffs):
                continue
            raise RuntimeError(f"Serper request failed: {e}")
    raise RuntimeError("Serper call exhausted retries.")

# ------------------------------------------------------------------------------
# Normalize -> Candidate
# ------------------------------------------------------------------------------

def _from_organic(items: List[Dict[str, Any]], vertical: str) -> List[Candidate]:
    out: List[Candidate] = []
    src = _source_label(vertical)
    seen: set[str] = set()
    for it in items or []:
        title = (it.get("title") or it.get("snippet") or "").strip()
        url = it.get("link") or it.get("url")
        snippet = it.get("snippet") or ""
        date_str = it.get("date") or it.get("formattedUrl") or ""
        if not _is_http_url(url):
            continue
        url = _normalize_url(url)
        if url in seen:
            continue
        seen.add(url)
        year = _year_from_date(date_str) or _year_from_date(snippet)
        doi = _find_doi(url, snippet)
        out.append(Candidate(title=title, url=url, year=year, source=src, score=0.0, doi=doi))
    return out

def _from_news(items: List[Dict[str, Any]]) -> List[Candidate]:
    out: List[Candidate] = []
    seen: set[str] = set()
    for it in items or []:
        title = (it.get("title") or "").strip()
        url = it.get("link") or it.get("url")
        if not _is_http_url(url):
            continue
        url = _normalize_url(url)
        if url in seen:
            continue
        seen.add(url)
        date_str = it.get("date") or ""
        year = _year_from_date(date_str)
        source = (it.get("source") or "news").lower()
        snippet = it.get("snippet") or ""
        doi = _find_doi(url, snippet)
        out.append(Candidate(title=title, url=url, year=year, source=source, score=0.0, doi=doi))
    return out

def _from_images(items: List[Dict[str, Any]]) -> List[Candidate]:
    """
    Serper Images response shape (typical):
      { "images": [ { "title": "...", "imageUrl": "...", "source": "...", "link": "...", "date": "...", "snippet": "..." }, ... ] }
    We return Candidate with url = imageUrl, source="image". Year best-effort from 'date'/'snippet'.
    """
    out: List[Candidate] = []
    seen: set[str] = set()
    for it in items or []:
        image_url = it.get("imageUrl") or it.get("imageUrlLarge") or ""
        if not _is_http_url(image_url):
            continue
        image_url = _normalize_url(image_url)
        if image_url in seen:
            continue
        seen.add(image_url)

        title = (it.get("title") or it.get("source") or "").strip()
        date_str = it.get("date") or ""
        snippet = it.get("snippet") or ""
        year = _year_from_date(date_str) or _year_from_date(snippet)
        # DOIs rarely appear here, but we still scan title/snippet just in case.
        doi = _find_doi(title, snippet, it.get("link"))

        out.append(Candidate(
            title=title,
            url=image_url,       # <-- image direct link
            year=year,
            source="image",
            score=0.0,
            doi=doi
        ))
    return out

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def search(
    query: str,
    vertical: str = "web",             # one of: web, news, patents, universities, images
    gl: Optional[str] = None,
    num: int = 20,
    time_range: str = "",              # "", "day", "week", "month", "year"
    sites: str = "",                   # comma-separated site filters; overrides defaults
    page: int = 1,
    timeout_s: int = 12,
) -> Tuple[str, List[Candidate]]:
    """
    Run a Serper search and return (final_query, candidates).
    """
    api_key = _get_serper_key()
    serper_search, serper_news, serper_images = _get_endpoints()

    q_final = _augment_query(query, vertical, sites_csv=sites)
    tbs = _tbs_of(time_range)

    payload: Dict[str, Any] = {"q": q_final}
    if gl:
        payload["gl"] = gl
    if page and page > 1:
        payload["page"] = int(page)
    payload["num"] = max(1, min(int(num), 100))
    if tbs:
        payload["tbs"] = tbs

    if vertical == "news":
        j = _serper_call(serper_news, payload, api_key, timeout_s=timeout_s)
        items = j.get("news") or []
        cands = _from_news(items)
    elif vertical == "images":
        j = _serper_call(serper_images, payload, api_key, timeout_s=timeout_s)
        items = j.get("images") or []
        cands = _from_images(items)
    else:
        j = _serper_call(serper_search, payload, api_key, timeout_s=timeout_s)
        items = j.get("organic") or []
        cands = _from_organic(items, vertical=vertical)

    return q_final, cands

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor Serper web/news/images retriever")
    ap.add_argument("--query", "-q", required=True, help="Search query")
    ap.add_argument("--vertical", "-v", default="web",
                    choices=["web", "news", "patents", "universities", "images"],
                    help="Search vertical")
    ap.add_argument("--gl", default="", help="Country code (e.g., us, jp, hk)")
    ap.add_argument("--num", type=int, default=20, help="Number of results to request")
    ap.add_argument("--time-range", default="", choices=["", "day", "week", "month", "year"], help="Recency filter")
    ap.add_argument("--sites", default="", help="Comma-separated site filters (overrides defaults for the vertical)")
    ap.add_argument("--page", type=int, default=1, help="Result page (1-based)")
    ap.add_argument("--out", default="", help="Output JSON path; if omitted, print to stdout")
    ap.add_argument("--save-raw", default="", help="Optional path to save raw Serper JSON")
    return ap.parse_args()

def _to_results_block(q_final: str, cands: List[Candidate]) -> Dict[str, Any]:
    return {"results": [{"query": q_final, "items": [asdict(c) for c in cands]}]}

if __name__ == "__main__":
    args = _parse_args()
    q_final, cands = search(
        query=args.query,
        vertical=args.vertical,
        gl=(args.gl or None),
        num=args.num,
        time_range=args.time_range,
        sites=args.sites,
        page=max(1, args.page),
    )
    payload = _to_results_block(q_final, cands)
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] results -> {args.out}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
