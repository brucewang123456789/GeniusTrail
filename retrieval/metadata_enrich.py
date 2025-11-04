# C:\CiteVizor\retrieval\metadata_enrich.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Metadata normalization & enrichment for academic candidates

Key fixes (2025-10-24):
- Prefer downloadable PDF URLs when constructing downstream Candidate:
  url := pdf_url > final_url > url (previously final_url/url first).
- Add tiny arXiv abs->pdf mapping as a safety net (only if pdf_url missing).
- Respect USER_AGENT and ACCEPT_LANGUAGE from env for polite enrichment calls.
- Harden URL canonicalization in dedup fallback (use pdf_url if present).
- Keep original behavior and public API stable; failure modes remain silent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from schemas import Candidate, DOCS_DIR  # project contracts

# ----------------------------- logging shim -----------------------------------

try:
    from infra.logging import get_logger  # project logger
    LOG = get_logger("metadata_enrich")
except Exception:  # fallback no-op logger
    class _L:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
    LOG = _L()

ENV_FILE = "citevizor.env"

# ----------------------------- env loader -------------------------------------

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
    # OS overrides
    for k in (
        "ENRICH_CROSSREF_ENABLED", "ENRICH_OPENALEX_ENABLED", "ENRICH_SEMSCHOLAR_ENABLED",
        "ENRICH_TIMEOUT_S", "ENRICH_BACKOFF_BASE", "SEMANTIC_SCHOLAR_API_KEY", "CROSSREF_MAILTO",
        "USER_AGENT", "ACCEPT_LANGUAGE"
    ):
        if os.getenv(k) is not None:
            out[k] = os.getenv(k)  # type: ignore
    return out


# ----------------------------- text/doi/url utils -----------------------------

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.I)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7ff]")
_ARXIV_ABS_RE = re.compile(r"(?:^|/)arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?", re.I)

def _normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", (t or "").strip())
    t = re.sub(r"\s+", " ", t)
    return t

def _canon_title(t: str) -> str:
    """Lowercased, stripped punctuation/quotes, collapse whitespace; keep alnum and a few symbols."""
    t = _normalize_text(t).lower()
    t = re.sub(r"[-–—:;,.!?\"'`~()\[\]{}<>]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _looks_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))

def _extract_doi_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    m = _DOI_RE.search(url)
    return m.group(0) if m else None

def _extract_doi_from_text(text: str) -> Optional[str]:
    m = _DOI_RE.search(text or "")
    return m.group(0) if m else None

def _norm_year(y: Any) -> Optional[int]:
    try:
        yi = int(y)
        return yi if 1900 <= yi <= time.gmtime().tm_year else None
    except Exception:
        return None

def _canonical_url(url: str) -> str:
    """Normalize for dedup comparison; keep host/path; drop scheme, trailing slashes/fragments."""
    u = (url or "").strip()
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = u.rstrip("/#")
    return u

def _doc_id_for(doi: Optional[str], title: str, year: Optional[int]) -> str:
    """Stable doc_id: prefer DOI hash; else title-hash + year token."""
    if doi:
        h = sha256(doi.lower().encode("utf-8")).hexdigest()[:12]
        return f"doi_{h}"
    base = _canon_title(title)
    h = sha256(base.encode("utf-8")).hexdigest()[:10]
    y = str(year) if year else "na"
    return f"t{h}_{y}"

def _arxiv_abs_to_pdf(url: Optional[str]) -> Optional[str]:
    """If URL is an arXiv abstract page, return its PDF URL; else None."""
    if not url:
        return None
    m = _ARXIV_ABS_RE.search(url)
    if not m:
        return None
    arx_id = m.group(1)
    return f"https://arxiv.org/pdf/{arx_id}.pdf"


# ----------------------------- HTTP helpers -----------------------------------

@dataclass
class HTTPConfig:
    timeout_s: int = 8
    backoff_base: float = 0.6
    crossref_enabled: bool = True
    openalex_enabled: bool = True
    semsch_enabled: bool = False
    s2_api_key: Optional[str] = None
    crossref_mailto: Optional[str] = None
    user_agent: str = "CiteVizor/metadata-enrich"
    accept_language: Optional[str] = None

    @classmethod
    def from_env(cls, project_root: Path) -> "HTTPConfig":
        env = _load_env(project_root)
        return cls(
            timeout_s=int(env.get("ENRICH_TIMEOUT_S", "8")),
            backoff_base=float(env.get("ENRICH_BACKOFF_BASE", "0.6")),
            crossref_enabled=(env.get("ENRICH_CROSSREF_ENABLED", "1") not in ("0", "false", "False")),
            openalex_enabled=(env.get("ENRICH_OPENALEX_ENABLED", "1") not in ("0", "false", "False")),
            semsch_enabled=(env.get("ENRICH_SEMSCHOLAR_ENABLED", "0") not in ("0", "false", "False")),
            s2_api_key=env.get("SEMANTIC_SCHOLAR_API_KEY"),
            crossref_mailto=env.get("CROSSREF_MAILTO"),
            user_agent=env.get("USER_AGENT", "CiteVizor/0.2 (+pdf-first)"),
            accept_language=env.get("ACCEPT_LANGUAGE"),
        )

class HTTP:
    def __init__(self, cfg: HTTPConfig):
        self.cfg = cfg
        self.sess = requests.Session()
        headers = {"User-Agent": cfg.user_agent}
        if cfg.accept_language:
            headers["Accept-Language"] = cfg.accept_language
        self.sess.headers.update(headers)
        if cfg.s2_api_key:
            self.sess.headers.update({"x-api-key": cfg.s2_api_key})

    def get_json(self, url: str, params: Dict[str, Any] = None, retries: int = 2) -> Optional[Dict[str, Any]]:
        params = params or {}
        for attempt in range(1, retries + 1):
            try:
                r = self.sess.get(url, params=params, timeout=self.cfg.timeout_s)
                if r.status_code in (429, 500, 502, 503, 504):
                    self._sleep_backoff(attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception:
                if attempt >= retries:
                    return None
                self._sleep_backoff(attempt)
        return None

    def _sleep_backoff(self, attempt: int) -> None:
        t = min(8.0, self.cfg.backoff_base * (2 ** (attempt - 1)))
        time.sleep(t)


# ----------------------------- enrichers --------------------------------------

def _enrich_crossref(http: HTTP, doi: str) -> Dict[str, Any]:
    if not http.cfg.crossref_enabled or not doi:
        return {}
    url = f"https://api.crossref.org/works/{doi}"
    params = {}
    if http.cfg.crossref_mailto:
        params["mailto"] = http.cfg.crossref_mailto
    j = http.get_json(url, params=params)
    if not j or "message" not in j:
        return {}
    m = j["message"]
    title = (m.get("title") or [""])[0] if isinstance(m.get("title"), list) else (m.get("title") or "")
    year = None
    for key in ("published-print", "published-online", "issued"):
        try:
            parts = m.get(key, {}).get("date-parts", [[]])[0]
            if parts and isinstance(parts[0], int):
                year = parts[0]
                break
        except Exception:
            pass
    authors = []
    for a in (m.get("author") or []):
        name = " ".join([x for x in [a.get("given"), a.get("family")] if x])
        if name.strip():
            authors.append(name.strip())
    venue = m.get("container-title")
    if isinstance(venue, list):
        venue = venue[0] if venue else None
    return {
        "title": title,
        "year": year,
        "authors": authors,
        "venue": venue,
        "publisher": m.get("publisher"),
        "doi": (m.get("DOI") or doi or "").lower() or None,
        "final_url": m.get("URL"),
        "lang": m.get("language"),
    }

def _enrich_openalex(http: HTTP, doi: Optional[str], title: str) -> Dict[str, Any]:
    if not http.cfg.openalex_enabled:
        return {}
    if doi:
        j = http.get_json(f"https://api.openalex.org/works/https://doi.org/{doi}")
        if j and isinstance(j, dict) and j.get("id"):
            return _oa_pick(j)
    # fallback search by title
    q = title.strip().strip('"')
    if not q:
        return {}
    j = http.get_json("https://api.openalex.org/works", params={"search": q, "per_page": 1})
    if j and isinstance(j.get("results"), list) and j["results"]:
        return _oa_pick(j["results"][0])
    return {}

def _oa_pick(obj: Dict[str, Any]) -> Dict[str, Any]:
    try:
        year = obj.get("publication_year")
        host = obj.get("host_venue") or {}
        open_access = obj.get("open_access", {})
        best_oa = open_access.get("oa_url")
        authors = []
        for a in obj.get("authorships", []):
            name = a.get("author", {}).get("display_name")
            if name:
                authors.append(name)
        title = obj.get("title")
        doi = (obj.get("doi") or "").replace("https://doi.org/", "").lower() or None
        return {
            "title": title,
            "year": year,
            "venue": host.get("display_name"),
            "publisher": host.get("publisher"),
            "doi": doi,
            "final_url": obj.get("primary_location", {}).get("source", {}).get("homepage_url") or obj.get("primary_location", {}).get("landing_page_url"),
            "pdf_url": best_oa,
            "open_access": bool(best_oa),
            "lang": obj.get("language"),
        }
    except Exception:
        return {}

def _enrich_semantic_scholar(http: HTTP, doi: Optional[str], title: str) -> Dict[str, Any]:
    if not http.cfg.semsch_enabled:
        return {}
    base = "https://api.semanticscholar.org/graph/v1/paper/"
    fields = "title,year,venue,publicationTypes,journal,openAccessPdf,externalIds,authors.name,url"
    if doi:
        j = http.get_json(base + f"DOI:{doi}", params={"fields": fields})
        if j and isinstance(j, dict) and j.get("paperId"):
            return _s2_pick(j)
    # search by title
    j = http.get_json("https://api.semanticscholar.org/graph/v1/paper/search", params={"query": title, "limit": 1, "fields": fields})
    if j and isinstance(j.get("data"), list) and j["data"]:
        return _s2_pick(j["data"][0])
    return {}

def _s2_pick(obj: Dict[str, Any]) -> Dict[str, Any]:
    try:
        doi = (obj.get("externalIds") or {}).get("DOI")
        authors = [a.get("name") for a in obj.get("authors", []) if a.get("name")]
        pdf_url = (obj.get("openAccessPdf") or {}).get("url")
        venue = (obj.get("journal") or {}).get("name") or obj.get("venue")
        return {
            "title": obj.get("title"),
            "year": obj.get("year"),
            "venue": venue,
            "doi": (doi or "").lower() or None,
            "final_url": obj.get("url"),
            "pdf_url": pdf_url,
            "open_access": bool(pdf_url),
        }
    except Exception:
        return {}

# ----------------------------- core logic -------------------------------------

@dataclass
class EnrichedRecord:
    doc_id: str
    candidate: Candidate
    meta: Dict[str, Any]

class MetadataEnricher:
    def __init__(self, project_root: Optional[Path] = None):
        # Make constructor compatible with backend shim that calls with no args
        self.root = project_root or Path(__file__).resolve().parents[1]
        self.http = HTTP(HTTPConfig.from_env(self.root))

    # -------- NEW: per-doc safe enrichment used by backend align.prep ---------

    def enrich_doc(self, doc_id: str) -> None:
        """
        Best-effort, idempotent enrichment for a single fetched document.
        - If meta.json missing or marked ok==False, log info and return.
        - If present, enrich lightweight fields and write <doc_id>.meta.enriched.json.
        - Never raises; logs only info on failures.
        """
        try:
            base = DOCS_DIR / doc_id
            meta_path = base / f"{doc_id}.meta.json"
            if not meta_path.exists():
                LOG.info("meta.skip_no_doc", extra={"doc_id": doc_id, "reason": "meta_json_missing"})
                return

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("ok") is False:
                LOG.info("meta.skip_no_doc", extra={"doc_id": doc_id, "reason": "download_not_ok"})
                return

            # Prepare seeds
            title = _normalize_text(meta.get("title") or "")
            doi = (meta.get("doi") or "").strip() or _extract_doi_from_text(title) or None

            # Optional online enrichment
            enriched: Dict[str, Any] = {}
            if doi:
                enriched.update(_enrich_crossref(self.http, doi))
                if not enriched.get("title"):
                    enriched.update(_enrich_openalex(self.http, doi, title))
            else:
                if title:
                    enriched.update(_enrich_openalex(self.http, None, title))

            out = self._merge_meta(meta, enriched)
            # arXiv abs->pdf safety net if still no pdf_url
            if not out.get("pdf_url"):
                guess = _arxiv_abs_to_pdf(out.get("final_url") or out.get("url"))
                if guess:
                    out["pdf_url"] = guess

            out["enriched"] = True
            out["enriched_at"] = int(time.time())

            enr_path = base / f"{doc_id}.meta.enriched.json"
            enr_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            LOG.info("meta.enrich_ok", extra={"doc_id": doc_id})

        except Exception as e:
            LOG.info("meta.enrich_skip", extra={"doc_id": doc_id, "err": str(e)[:200]})
            return

    # -------- Batch pipeline (used in retrieval stage) ------------------------

    def enrich_and_dedup(self, candidates: List[Candidate]) -> List[Candidate]:
        """Normalize + deduplicate + optionally enrich; persist meta.json; return normalized candidates."""
        # First pass: local normalization and DOI/url extraction
        temp: List[Tuple[Candidate, Dict[str, Any]]] = []
        for c in candidates:
            title = _normalize_text(c.title)
            year = _norm_year(c.year)
            url = _normalize_text(c.url or "")
            doi = (c.doi or _extract_doi_from_url(url) or _extract_doi_from_text(title))
            meta = {
                "title": title,
                "year": year,
                "url": url,
                "source": c.source,
                "score": c.score,
                "doi": doi.lower() if doi else None,
                "lang": ("zh" if _looks_cjk(title) else "en"),
            }
            temp.append((c, meta))

        # Second pass: optional online enrichment (prefer DOI path)
        records: List[EnrichedRecord] = []
        for c, m in temp:
            enriched = {}
            if m.get("doi"):
                enriched.update(_enrich_crossref(self.http, m["doi"]))
                if not enriched.get("title"):
                    enriched.update(_enrich_openalex(self.http, m["doi"], m["title"]))
                if self.http.cfg.semsch_enabled and not enriched.get("title"):
                    enriched.update(_enrich_semantic_scholar(self.http, m["doi"], m["title"]))
            else:
                # no DOI -> try OpenAlex/S2 by title
                enriched.update(_enrich_openalex(self.http, None, m["title"]))
                if self.http.cfg.semsch_enabled and not enriched.get("title"):
                    enriched.update(_enrich_semantic_scholar(self.http, None, m["title"]))

            merged = self._merge_meta(m, enriched)

            # Safety net: arXiv abs -> pdf if no pdf_url yet
            if not merged.get("pdf_url"):
                guess = _arxiv_abs_to_pdf(merged.get("final_url") or merged.get("url"))
                if guess:
                    merged["pdf_url"] = guess

            doc_id = _doc_id_for(merged.get("doi"), merged.get("title") or c.title, merged.get("year"))

            # persist meta (side-effect used by downstream)
            self._write_meta(doc_id, merged)

            # >>> CORE FIX <<< choose a downloadable URL for downstream Candidate
            downstream_url = merged.get("pdf_url") or merged.get("final_url") or merged.get("url") or c.url

            # normalized candidate for downstream
            nc = Candidate(
                title=merged.get("title") or c.title,
                url=downstream_url,
                year=merged.get("year") or c.year,
                source=c.source or "scholar",
                score=c.score or 1.0,
                doi=merged.get("doi") or c.doi,
            )
            records.append(EnrichedRecord(doc_id=doc_id, candidate=nc, meta=merged))

        # Third pass: dedup (DOI > canonical-title(+/-1y) > URL)
        by_key: Dict[str, EnrichedRecord] = {}
        for r in records:
            k = self._dedup_key(r.meta)
            if k not in by_key:
                by_key[k] = r
            else:
                # keep higher score or richer meta
                prev = by_key[k]
                if (r.candidate.score or 0) > (prev.candidate.score or 0):
                    by_key[k] = r
                else:
                    if self._meta_richness(r.meta) > self._meta_richness(prev.meta):
                        by_key[k] = r

        # de-duplicate doc_ids as well (edge: different keys but same DOI hash)
        final_cands: List[Candidate] = []
        seen_doc = set()
        for r in by_key.values():
            if r.doc_id in seen_doc:
                continue
            seen_doc.add(r.doc_id)
            final_cands.append(r.candidate)
        return final_cands

    # ------------------ helpers ------------------

    @staticmethod
    def _merge_meta(base: Dict[str, Any], ext: Dict[str, Any]) -> Dict[str, Any]:
        """Prefer external enrichment if present; else fallback to base."""
        out = dict(base)
        for k, v in ext.items():
            if v in (None, "", [], {}):
                continue
            if k == "year":
                vv = _norm_year(v)
                if vv:
                    out[k] = vv
            else:
                out[k] = v
        out["title"] = _normalize_text(out.get("title") or "")
        out["lang"] = out.get("lang") or ("zh" if _looks_cjk(out["title"]) else "en")
        return out

    @staticmethod
    def _meta_richness(m: Dict[str, Any]) -> int:
        score = 0
        for k in ("doi", "venue", "publisher", "authors", "pdf_url", "open_access", "final_url"):
            if m.get(k):
                score += 1
        return score

    @staticmethod
    def _dedup_key(m: Dict[str, Any]) -> str:
        if m.get("doi"):
            return f"doi::{m['doi'].lower()}"
        ct = _canon_title(m.get("title") or "")
        y = _norm_year(m.get("year"))
        yb = f"{y}" if y else "na"
        if y:
            yb = f"{y-1}-{y+1}"
        # Prefer pdf_url in URL fallback to cluster landing vs PDF together
        url_fallback = m.get("pdf_url") or m.get("url") or ""
        return f"title::{ct}::{yb}" if ct else f"url::{_canonical_url(url_fallback)}"

    def _write_meta(self, doc_id: str, meta: Dict[str, Any]) -> None:
        d = DOCS_DIR / doc_id
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{doc_id}.meta.json"
        payload = {
            "doc_id": doc_id,
            "title": meta.get("title"),
            "year": meta.get("year"),
            "doi": meta.get("doi"),
            "url": meta.get("url"),
            "final_url": meta.get("final_url"),
            "pdf_url": meta.get("pdf_url"),
            "open_access": meta.get("open_access"),
            "venue": meta.get("venue"),
            "publisher": meta.get("publisher"),
            "authors": meta.get("authors"),
            "lang": meta.get("lang"),
            "source": meta.get("source"),
            "score": meta.get("score"),
            "created_at": int(time.time()),
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor metadata normalizer/enricher")
    ap.add_argument("--in", dest="in_path", required=True, help="Input candidates JSON path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output enriched candidates JSON path")
    return ap.parse_args()

def _read_candidates(path: Path) -> List[Candidate]:
    arr = json.loads(path.read_text(encoding="utf-8"))
    out: List[Candidate] = []
    for o in arr:
        out.append(Candidate(
            title=o.get("title") or "",
            url=o.get("url"),
            year=o.get("year"),
            source=o.get("source") or "scholar",
            score=o.get("score") or 1.0,
            doi=o.get("doi"),
        ))
    return out

def _write_candidates(path: Path, cands: List[Candidate]) -> None:
    arr = []
    for c in cands:
        arr.append({
            "title": c.title,
            "url": c.url,
            "year": c.year,
            "source": c.source,
            "score": c.score,
            "doi": c.doi,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    args = _parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    enricher = MetadataEnricher(project_root)
    cands = _read_candidates(in_path)
    enriched = enricher.enrich_and_dedup(cands)
    _write_candidates(out_path, enriched)
    print(f"[OK] Enriched {len(enriched)} candidates -> {out_path}")
