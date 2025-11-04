# -*- coding: utf-8 -*-
"""
Serper Google Scholar retriever for CiteVizor.

Goals:
- Scholar-only search (no general web).
- Return candidates that prefer directly downloadable PDFs.
- Populate fields expected downstream: url, pdf_url, host, year, open_access, source, score...
- Optional strict OA-only filtering via env: CITEVIZOR_SCHOLAR_ONLY_OA=1

Env:
- SERPER_API_KEY            (from ./citevizor.env or environment)
- CITEVIZOR_SCHOLAR_ONLY_OA (1 to keep only open-access domains)

Cache:
- storage/cache/serper_scholar_*.json
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.parse as ul
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from pydantic import ValidationError

# Project schemas
from schemas import (
    Candidate,
    SearchPlan,
    SourceType,
    YearRange,
    STORAGE_DIR,
    CACHE_DIR,
)

SERPER_SCHOLAR_ENDPOINT = "https://google.serper.dev/scholar"
ENV_FILE_NAME = "citevizor.env"
CACHE_PREFIX = "serper_scholar_"

# DOI + common ids
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b", re.I)
ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
ARXIV_PDF_RE = re.compile(r"arxiv\.org/pdf/(\d{4}\.\d{4,5})(?:v\d+)?\.pdf", re.I)

# OA domain whitelist
OA_HOSTS = {
    "arxiv.org",
    "www.arxiv.org",
    "aclanthology.org",
    "openreview.net",
    "proceedings.mlr.press",
    "papers.nips.cc",
    "proceedings.neurips.cc",
    "hal.science",
    "www.hal.science",
    "biorxiv.org",
    "www.biorxiv.org",
}

def load_env_file(project_root: Path) -> Dict[str, str]:
    env_path = project_root / ENV_FILE_NAME
    out: Dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


@dataclass
class SerperScholarClient:
    api_key: Optional[str]
    project_root: Path

    def __post_init__(self):
        if not self.api_key:
            env_vars = load_env_file(self.project_root)
            self.api_key = env_vars.get("SERPER_API_KEY") or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "SERPER_API_KEY not found. Put it in ./citevizor.env or export it in the environment."
            )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.strict_oa = os.getenv("CITEVIZOR_SCHOLAR_ONLY_OA", "0") == "1"

    def search_with_plan(
        self,
        plan: SearchPlan,
        limit_per_query: int = 8,
        lang: str = "en",
        gl: Optional[str] = None,
        timeout_s: int = 20,
        max_retries: int = 3,
    ) -> List[Candidate]:
        if not plan.queries:
            raise ValueError("SearchPlan.queries is empty; prepare the plan before calling search.")

        bag: Dict[str, Candidate] = {}
        for q in plan.queries:
            raw = self._scholar_query_cached(
                query=q,
                num=limit_per_query,
                lang=lang,
                gl=gl,
                timeout_s=timeout_s,
                max_retries=max_retries,
            )
            for c in self._parse_items(raw):
                if c.source != SourceType.scholar:
                    continue
                if self._filtered_out_by_year(plan, c):
                    continue
                if self.strict_oa and not self._is_allowed_host(c.host):
                    continue
                k = c.stable_key()
                prev = bag.get(k)
                if not prev or c.score > prev.score:
                    bag[k] = c

        ranked = sorted(bag.values(), key=lambda c: (c.score, c.year or 0, c.title.lower()), reverse=True)
        return ranked[: plan.k_top]

    # ---------------- internal ----------------

    @staticmethod
    def _is_allowed_host(host: Optional[str]) -> bool:
        if not host:
            return False
        h = host.lower()
        if h in OA_HOSTS:
            return True
        # allow subdomains of OA hosts (e.g., proceedings.neurips.cc pages)
        for base in OA_HOSTS:
            if h.endswith("." + base):
                return True
        return False

    @staticmethod
    def _filtered_out_by_year(plan: SearchPlan, cand: Candidate) -> bool:
        yr: Optional[YearRange] = None
        y = plan.filters.get("years") if isinstance(plan.filters, dict) else None
        if isinstance(y, dict):
            try:
                yr = YearRange(**y)  # type: ignore
            except ValidationError:
                yr = None
        if yr and cand.year:
            if yr.start and cand.year < yr.start:
                return True
            if yr.end and cand.year > yr.end:
                return True
        return False

    def _scholar_query_cached(
        self,
        query: str,
        num: int,
        lang: str,
        gl: Optional[str],
        timeout_s: int,
        max_retries: int,
    ) -> Dict:
        cache_key = f"{CACHE_PREFIX}{sha256(f'{query}|{num}|{lang}|{gl}'.encode()).hexdigest()[:16]}.json"
        cache_path = CACHE_DIR / cache_key
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        payload = {"q": query, "num": max(1, min(20, num)), "hl": lang}
        if gl:
            payload["gl"] = gl
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = requests.post(SERPER_SCHOLAR_ENDPOINT, headers=headers, json=payload, timeout=timeout_s)
                if resp.status_code == 200:
                    data = resp.json()
                    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    return data
                if resp.status_code in (429, 500, 502, 503, 504):
                    self._sleep_backoff(attempt)
                    continue
                # Non-retryable
                body = self._safe_text(resp)
                cache_path.write_text(json.dumps({"error": resp.status_code, "body": body}, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
                raise RuntimeError(f"Serper error {resp.status_code}: {body}")
            except requests.RequestException as e:
                if attempt >= max_retries:
                    raise RuntimeError(f"Serper request failed after retries: {e}") from e
                self._sleep_backoff(attempt)

    @staticmethod
    def _safe_text(resp: requests.Response) -> str:
        try:
            return resp.text[:5000]
        except Exception:
            return "<no-body>"

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        base = min(8.0, 0.6 * (2 ** (attempt - 1)))
        time.sleep(base + random.uniform(0, 0.5))

    # ----------- parsing & normalization ----------------

    @staticmethod
    def _unwrap_link(link: str) -> str:
        if not link:
            return link
        try:
            parsed = ul.urlparse(link)
            qs = ul.parse_qs(parsed.query)
            if "url" in qs and qs["url"]:
                inner = qs["url"][0]
                if inner.startswith("http"):
                    return inner
        except Exception:
            pass
        return link

    @staticmethod
    def _rewrite_to_pdf(link: str) -> Tuple[str, Optional[bool]]:
        """
        Return (pdf_url, open_access_hint) if we can confidently rewrite to PDF.
        If not sure, return (original_link, None).
        """
        url = (link or "").strip()
        if not url:
            return url, None
        low = url.lower()

        # arXiv
        m_abs = ARXIV_ABS_RE.search(low)
        if m_abs:
            arx_id = m_abs.group(1)
            return f"https://arxiv.org/pdf/{arx_id}.pdf", True
        if ARXIV_PDF_RE.search(low):
            return url, True

        # ACL Anthology
        if "aclanthology.org" in low and not low.endswith(".pdf"):
            return url.rstrip("/") + ".pdf", True
        if "aclanthology.org" in low and low.endswith(".pdf"):
            return url, True

        # OpenReview: often /pdf?id=...
        if "openreview.net" in low:
            return url, None

        # PMLR, NeurIPS proceedings often already link to PDFs or html; keep url
        return url, None

    def _parse_items(self, data: Dict) -> List[Candidate]:
        out: List[Candidate] = []
        organic = data.get("organic") or data.get("results") or []
        for item in organic:
            try:
                title = (item.get("title") or "").strip()
                raw_link = (item.get("link") or item.get("url") or "").strip()
                if not title or not raw_link:
                    continue

                link_unwrapped = self._unwrap_link(raw_link)
                pdf_url, oa_hint = self._rewrite_to_pdf(link_unwrapped)

                snippet = (item.get("snippet") or item.get("description") or "")[:1000]
                year = self._extract_year(item, snippet)
                doi = self._extract_doi(item, link_unwrapped, snippet)

                authors = []
                if isinstance(item.get("authors"), list):
                    authors = [a for a in item["authors"] if isinstance(a, str)]
                elif isinstance(item.get("authors"), str):
                    authors = [a.strip() for a in item["authors"].split(",") if a.strip()]

                venue = None
                pubinfo = item.get("publicationInfo") or item.get("publication")
                if isinstance(pubinfo, str):
                    venue = pubinfo.strip()

                # score: recency + small bonuses
                is_review = self._is_review(title, snippet)
                recency_score = self._recency(year)
                score = 1.0 + recency_score + (0.1 if is_review else 0.0) + (0.08 if oa_hint else 0.0)

                host = ul.urlparse(pdf_url or link_unwrapped).netloc

                cand = Candidate(
                    title=title,
                    url=link_unwrapped,
                    pdf_url=pdf_url,
                    host=host,
                    doi=doi,
                    year=year,
                    source=SourceType.scholar,
                    score=score,
                    authors=authors,
                    venue=venue,
                    is_review=is_review,
                    open_access=oa_hint,
                )
                out.append(cand)
            except Exception:
                continue
        return out

    @staticmethod
    def _extract_year(item: Dict, snippet: str) -> Optional[int]:
        for key in ("year", "publicationYear", "date"):
            if key in item:
                try:
                    txt = str(item[key])
                    m = re.search(r"(19|20)\d{2}", txt)
                    if m:
                        return int(m.group(0))
                except Exception:
                    pass
        m2 = re.search(r"(19|20)\d{2}", snippet or "")
        return int(m2.group(0)) if m2 else None

    @staticmethod
    def _extract_doi(item: Dict, link: str, snippet: str) -> Optional[str]:
        for key in ("doi", "DOI"):
            if key in item and isinstance(item[key], str):
                m = DOI_RE.search(item[key])
                if m:
                    return m.group(0)
        for txt in (link, snippet):
            m = DOI_RE.search(txt or "")
            if m:
                return m.group(0)
        meta = item.get("inlineLinks") or {}
        if isinstance(meta, dict):
            try:
                txt = json.dumps(meta, ensure_ascii=False)
                m = DOI_RE.search(txt)
                if m:
                    return m.group(0)
            except Exception:
                pass
        return None

    @staticmethod
    def _is_review(title: str, snippet: str) -> bool:
        t = f"{title} {snippet}".lower()
        return any(k in t for k in ["review", "systematic review", "survey", "meta-analysis"])

    @staticmethod
    def _recency(year: Optional[int]) -> float:
        if not year:
            return 0.0
        now_y = time.gmtime().tm_year
        age = max(0, now_y - year)
        return max(0.0, 0.6 - 0.1 * age)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    client = SerperScholarClient(api_key=None, project_root=PROJECT_ROOT)
    plan = SearchPlan(
        queries=["Attention Is All You Need", "retrieval augmented generation survey"],
        k_top=10,
        filters={"years": {"start": 2016, "end": time.gmtime().tm_year}},
    )
    results = client.search_with_plan(plan=plan, limit_per_query=5, lang="en", gl=None)
    print(f"[Serper] got {len(results)} candidates")
    for i, c in enumerate(results, 1):
        print(f"{i:02d}. ({c.year}) {c.title} | {c.doi or '-'} | host={c.host} | pdf={bool(c.pdf_url)} -> {c.pdf_url or c.url}")
