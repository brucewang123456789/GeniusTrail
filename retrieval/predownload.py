# /workspace/retrieval/predownload.py
# -*- coding: utf-8 -*-
"""
CiteVizor predownload resolver (scholar-aware, PDF-first, DOI+meta sniffing, image bypass)

Purpose:
- Runs between retrieval and fetch/downloader.py.
- For each Candidate, try to resolve a direct PDF URL and/or DOI before downloading.
- Classify the candidate as: 'pdf' | 'image' | 'landing' | 'unknown'.
- Never throws: failures are swallowed and the original Candidate is passed through.

Compatibility & Safety:
- DO NOT create new attrs on Candidate (e.g., no "pdf_url"). Some projects use strict models.
- Only touch fields that already exist. In practice:
    * If a direct PDF is found -> set cand.url = <pdf_direct_link> (absolute-normalized).
    * If an image candidate -> set cand.url = <image_direct_link> and classify as 'image' (no DOI/PDF follow).
    * If DOI is found -> set cand.doi = <doi> ONLY IF the model actually has "doi".
    * If no URL on Candidate and we have a landing -> set cand.url = landing (absolute-normalized).
- Scholar landing pages are ignored (we never try to "download" scholar.google.*).

Hardening:
- Absolute URL normalization for ANY discovered link (html meta/link/anchor, DOI link headers, heuristics, and hints).
- Batch resolve_list() never interrupts pipeline: per-item try/except and pass-through.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict
from urllib.parse import urlparse, urljoin
from pathlib import Path

import requests

from schemas import Candidate  # project type

# Reuse centralized per-domain heuristics from downloader if available
try:
    from fetch.downloader import guess_pdf_urls, DownloadConfig  # optional
except Exception:
    # Minimal stubs to avoid import-time hard failure (early bootstrap)
    def guess_pdf_urls(original_url: Optional[str], doi: Optional[str]) -> List[str]:
        u = (original_url or "").strip()
        out = []
        if u:
            if not u.lower().endswith(".pdf"):
                out.append(u + ".pdf")
            out.append(u)
        if "abs/" in u:
            out.append(u.replace("/abs/", "/pdf/") + ".pdf")
        if "doi.org/" in u:
            out.append(u.replace("https://doi.org/", "https://dx.doi.org/"))
        return list(dict.fromkeys([x for x in out if x]))

    @dataclass
    class DownloadConfig:
        project_root: object = None
        user_agent: str = "CiteVizor/0.2 (+predownload)"
        https_proxy: Optional[str] = None
        http_proxy: Optional[str] = None
        max_mb: int = 120
        connect_timeout: int = 12
        read_timeout: int = 45
        max_retries: int = 3
        accept_language: Optional[str] = "en-US,en;q=0.9"
        unpaywall_email: Optional[str] = None
        ca_bundle: Optional[str] = None

        @classmethod
        def from_env(cls, project_root):
            return cls()

# ---------- utilities ----------

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b", re.IGNORECASE)
IMG_EXT_RE = re.compile(r"\.(png|jpe?g|webp|gif|tiff?)($|\?)", re.IGNORECASE)

def _is_scholar(u: Optional[str]) -> bool:
    if not u:
        return False
    host = urlparse(u).netloc.lower()
    return host.startswith("scholar.google.") or "scholar.googleusercontent" in host

def _abs(s: Optional[str]) -> str:
    return (s or "").strip()

def _clean_header_val(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return s.replace("\r", "").replace("\n", "").strip()

def _normalize_url(link: Optional[str], base: Optional[str] = None) -> Optional[str]:
    """
    Normalize any link into an absolute http(s) URL.
    """
    if not link:
        return None
    link = link.strip()
    if link.startswith("//"):
        link = "https:" + link
    if base and not urlparse(link).scheme:
        link = urljoin(base, link)
    pr = urlparse(link)
    if pr.scheme not in ("http", "https"):
        return None
    return link

def _is_image_url(u: Optional[str]) -> bool:
    """
    Lightweight image detection by URL pattern.
    This is a conservative pre-check; real content detection happens in downloader.
    """
    if not u:
        return False
    return bool(IMG_EXT_RE.search(u))

def _build_session(cfg: DownloadConfig) -> requests.Session:
    s = requests.Session()
    ua = _clean_header_val(getattr(cfg, "user_agent", None)) or "CiteVizor/0.2 (+predownload)"
    al = _clean_header_val(getattr(cfg, "accept_language", None))
    headers: Dict[str, str] = {"User-Agent": ua}
    if al:
        headers["Accept-Language"] = al
    s.headers.update(headers)

    proxies: Dict[str, str] = {}
    hp = _clean_header_val(getattr(cfg, "http_proxy", None))
    sp = _clean_header_val(getattr(cfg, "https_proxy", None))
    if sp:
        proxies["https"] = sp
    if hp:
        proxies["http"] = hp
    if proxies:
        s.proxies.update(proxies)
    cab = _clean_header_val(getattr(cfg, "ca_bundle", None))
    if cab:
        os.environ["REQUESTS_CA_BUNDLE"] = cab
    return s

def _parse_link_header(value: str) -> List[Tuple[str, Dict[str, str]]]:
    out: List[Tuple[str, Dict[str, str]]] = []
    for part in value.split(","):
        part = part.strip()
        if not part or "<" not in part or ">" not in part:
            continue
        url = part[1 : part.index(">")].strip()
        params_str = part[part.index(">") + 1 :].strip()
        params: Dict[str, str] = {}
        for kv in params_str.split(";"):
            kv = kv.strip()
            if not kv:
                continue
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k.strip().lower()] = v.strip().strip('"')
            else:
                params[kv.strip().lower()] = ""
        out.append((url, params))
    return out

def _resolve_via_doi(session: requests.Session, doi: str, timeout: Tuple[int, int]) -> Tuple[Optional[str], Optional[str]]:
    """
    doi.org content negotiation; try to extract (pdf_url, html_url).
    """
    base = f"https://doi.org/{doi}"
    try:
        r = session.get(base, allow_redirects=True, timeout=timeout,
                        headers={"Accept": "application/pdf, application/*, */*"})
        final = str(r.url)
        pdf_url: Optional[str] = None
        html_url: Optional[str] = None

        chain = list(r.history) + [r]
        for resp in chain:
            l = resp.headers.get("Link") or resp.headers.get("link")
            if not l:
                continue
            for url, params in _parse_link_header(l):
                t = (params.get("type") or "").lower()
                rel = (params.get("rel") or "").lower()
                nu = _normalize_url(url, base=final)
                if not nu:
                    continue
                if "pdf" in t or rel == "related":
                    pdf_url = pdf_url or nu
                if "html" in t or rel in {"alternate", "canonical"}:
                    html_url = html_url or nu

        ct = (r.headers.get("Content-Type") or "").lower()
        if "pdf" in ct:
            pdf_url = pdf_url or _normalize_url(final, base=final)
        html_url = html_url or _normalize_url(final, base=final)
        return pdf_url, html_url
    except requests.RequestException:
        return None, None

def _resolve_unpaywall(session: requests.Session, doi: str, email: Optional[str], timeout: Tuple[int, int]) -> Optional[str]:
    email = _clean_header_val(email)
    if not email:
        return None
    api = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        rr = session.get(api, timeout=timeout)
        if rr.status_code >= 400:
            return None
        data = rr.json()
        best = data.get("best_oa_location") or {}
        for key in ("url_for_pdf", "url"):
            if best.get(key):
                nu = _normalize_url(best[key], base=None)
                if nu and nu.lower().endswith(".pdf"):
                    return nu
        for loc in (data.get("oa_locations") or []):
            for key in ("url_for_pdf", "url"):
                if loc.get(key):
                    nu = _normalize_url(loc[key], base=None)
                    if nu and nu.lower().endswith(".pdf"):
                        return nu
    except Exception:
        return None
    return None

def _fetch_html(session: requests.Session, url: str, timeout: Tuple[int, int]) -> Optional[str]:
    try:
        pr = urlparse(url)
        if pr.scheme not in ("http", "https"):
            return None
        r = session.get(url, allow_redirects=True, timeout=timeout)
        ct = (r.headers.get("Content-Type") or "").lower()
        if "html" not in ct and "<html" not in r.text[:2048].lower():
            return None
        return r.text
    except requests.RequestException:
        return None

def _html_find_pdf_and_doi(html: str, base_url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Lightweight HTML regex parsing: citation_pdf_url, link[rel=alternate][type=application/pdf],
    obvious anchors, plus DOI. Any link is normalized to absolute http(s).
    """
    pdf_url = None
    doi = None

    m = re.search(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', html, flags=re.I)
    if m:
        pdf_url = _normalize_url(m.group(1).strip(), base=base_url)

    if not pdf_url:
        m = re.search(r'<link[^>]+rel=["\']alternate["\'][^>]+type=["\']application/pdf["\'][^>]+href=["\']([^"\']+)["\']',
                      html, flags=re.I)
        if m:
            pdf_url = _normalize_url(m.group(1).strip(), base=base_url)

    if not pdf_url:
        m = re.search(r'<a[^>]+href=["\']([^"\']+\.pdf[^"\']*)["\'][^>]*>', html, flags=re.I)
        if m:
            pdf_url = _normalize_url(m.group(1).strip(), base=base_url)

    m = re.search(r'<meta[^>]+name=["\']citation_doi["\'][^>]+content=["\']([^"\']+)["\']', html, flags=re.I)
    if m:
        doi = m.group(1).strip()
    if not doi:
        m = DOI_RE.search(html)
        if m:
            doi = m.group(1)

    return pdf_url, doi

# ---------- output typing ----------

@dataclass
class PredownloadResult:
    kind: str                 # 'pdf' | 'image' | 'landing' | 'unknown'
    resolved_pdf: Optional[str]
    resolved_html: Optional[str]
    doi: Optional[str]

# ---------- core APIs ----------

def resolve_one(cand: Candidate, cfg: Optional[DownloadConfig] = None) -> Tuple[Candidate, PredownloadResult]:
    """
    Enrich a single Candidate. Never raises.
    Safe field policy:
      - If PDF found -> set cand.url to direct PDF (absolute-normalized).
      - If image candidate -> set cand.url to direct image and classify as 'image' (no DOI/PDF follow).
      - If DOI found -> set cand.doi ONLY IF model has "doi".
      - Never create new attributes (e.g., no pdf_url).
    """
    # Normalize cfg
    if cfg is None or not isinstance(cfg, DownloadConfig):
        pr = None
        try:
            pr = getattr(cfg, "project_root", None)
        except Exception:
            pr = None
        cfg = DownloadConfig.from_env(pr or Path.cwd())

    session = _build_session(cfg)
    timeout = (int(getattr(cfg, "connect_timeout", 12)),
               int(getattr(cfg, "read_timeout", 45)))

    # Normalize existing fields
    url = _abs(getattr(cand, "url", None))
    pdf_hint_raw = _abs(getattr(cand, "pdf", None))
    doi = _abs(getattr(cand, "doi", None))
    source = _abs(getattr(cand, "source", None)).lower()

    # Skip scholar landings
    if _is_scholar(url):
        url = ""

    # Early image bypass (source says image, or URL looks like an image)
    if source == "image" or _is_image_url(url) or _is_image_url(pdf_hint_raw):
        img_link = _normalize_url(url or pdf_hint_raw, base=None)
        if img_link:
            try:
                cand.url = img_link
            except Exception:
                pass
            # Do not touch DOI for image candidates
            return cand, PredownloadResult(kind="image", resolved_pdf=None, resolved_html=None, doi=doi or None)
        # If no usable link, fall through to normal logic

    # 1) Immediate PDF hint
    if pdf_hint_raw:
        pdf_hint = _normalize_url(pdf_hint_raw, base=url or None)
        if pdf_hint and (pdf_hint.lower().endswith(".pdf") or ("pdf" in pdf_hint.lower())):
            try:
                cand.url = pdf_hint
            except Exception:
                pass
            return cand, PredownloadResult(kind="pdf", resolved_pdf=pdf_hint, resolved_html=url or None, doi=doi or None)

    # 2) DOI from url
    if not doi and url:
        m = DOI_RE.search(url)
        if m:
            doi = m.group(1)

    resolved_pdf: Optional[str] = None
    resolved_html: Optional[str] = None

    # 3) Unpaywall
    if doi:
        oa_pdf = _resolve_unpaywall(session, doi, getattr(cfg, "unpaywall_email", None), timeout)
        if oa_pdf:
            resolved_pdf = oa_pdf

    # 4) DOI negotiation
    if doi and not resolved_pdf:
        pdf2, html2 = _resolve_via_doi(session, doi, timeout)
        if pdf2:
            resolved_pdf = pdf2
        if html2:
            resolved_html = html2

    # 5) Landing HTML parse
    landing_to_parse = resolved_html or _normalize_url(url, base=None) or url
    if not resolved_pdf and landing_to_parse:
        html = _fetch_html(session, landing_to_parse, timeout)
        if html:
            pdf_from_html, doi_from_html = _html_find_pdf_and_doi(html, base_url=landing_to_parse)
            if pdf_from_html:
                resolved_pdf = pdf_from_html
            if not doi and doi_from_html:
                doi = doi_from_html

    # 6) Heuristic guesses
    if not resolved_pdf:
        base_for_guess = landing_to_parse or url or None
        for g in guess_pdf_urls(url, doi):
            ng = _normalize_url(g, base=base_for_guess)
            if ng and (ng.lower().endswith(".pdf") or "pdf" in ng.lower()):
                resolved_pdf = ng
                break

    # 7) classify
    kind = "pdf" if resolved_pdf else ("landing" if landing_to_parse else "unknown")

    # 8) Safe mutations
    if resolved_pdf:
        try:
            cand.url = resolved_pdf
        except Exception:
            pass
    if doi and hasattr(cand, "doi"):
        try:
            if not getattr(cand, "doi", None):
                cand.doi = doi
        except Exception:
            pass
    if (not getattr(cand, "url", None)) and landing_to_parse:
        land_abs = _normalize_url(landing_to_parse, base=None) or landing_to_parse
        try:
            cand.url = land_abs
        except Exception:
            pass

    return cand, PredownloadResult(kind=kind, resolved_pdf=resolved_pdf,
                                   resolved_html=landing_to_parse or None, doi=doi or None)

def resolve_list(cands: Iterable[Candidate], cfg: Optional[DownloadConfig] = None) -> List[Candidate]:
    """
    Resolve a list; order preserved.
    NEVER throw: any single failure is logged (if logging available) and the original candidate is passed through.
    """
    if cfg is None or not isinstance(cfg, DownloadConfig):
        cfg = DownloadConfig.from_env(Path.cwd())

    out: List[Candidate] = []
    for c in (cands or []):
        try:
            c2, _ = resolve_one(c, cfg)
            out.append(c2)
        except Exception as e:
            try:
                from infra.logging import get_logger
                get_logger("predownload").warning("predownload.skip_one", extra={
                    "title": getattr(c, "title", "")[:120], "err": str(e)[:200]
                })
            except Exception:
                pass
            out.append(c)
    return out

# --------------------------- CLI smoke test (optional) -------------------------

if __name__ == "__main__":
    cfg = DownloadConfig.from_env(Path.cwd())
    demo = [
        Candidate(title="arXiv PDF", url="https://arxiv.org/abs/1706.03762", source="web", score=1.0),
        Candidate(title="DOI Landing", url="https://doi.org/10.1145/3366423.3380129", source="web", score=0.8),
        Candidate(title="Scholar Page", url="https://scholar.google.com/scholar?cluster=...", source="scholar", score=0.7),
        Candidate(title="Direct Image", url="https://example.com/fig1.png", source="image", score=0.6),
    ]
    for c in resolve_list(demo, cfg):
        print("[Resolved]", getattr(c, "title", ""),
              "url=", getattr(c, "url", None),
              "doi=", getattr(c, "doi", None))
