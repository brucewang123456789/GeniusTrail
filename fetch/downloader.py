# downloader.py
"""
Robust PDF-first (now PDF+Image) downloader with per-host rate limiting,
HEAD+GET probing, DOI content-negotiation, strict binary detection, and rich tracing.

Path contract (2025-10-27):
- PDF MUST be stored at: storage/docs/<doc_id>/<doc_id>.pdf
- HTML (if persisted) SHOULD be at: storage/docs/<doc_id>/<doc_id>.html
- Evidence images from the web are copied to: storage/evidence/<doc_id>/web_images/*
- All callers should prefer schemas.DOCS_DIR as out_dir. This module will normalize paths even if out_dir differs.

Backward-compat layer:
- Exports `Downloader`, `DownloadConfig`, `FetchedDoc` for legacy imports.
- Also exports low-level `PDFDownloader` and `download_many`.
- NEW: exports `finalize_pdf_path(obj, ...)` to enforce contract for downstream objects (schema FetchedDoc or local FetchedDoc/dict).
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse, urlunparse
import logging
from http.client import RemoteDisconnected

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---- unified path contract (schemas) ----
try:
    # Prefer the single source of truth from schemas.py
    from schemas import (
        DOCS_DIR,
        EVIDENCE_DIR,
        pdf_path_for,
    )
except Exception:
    # Fallback if schemas is not importable (should not happen in normal runs)
    DOCS_DIR = Path(os.getenv("CITEVIZOR_DOCS_DIR", "/workspace/storage/docs"))
    EVIDENCE_DIR = Path(os.getenv("CITEVIZOR_EVIDENCE_DIR", "/workspace/storage/evidence"))

__all__ = [
    "Downloader",
    "DownloadConfig",
    "FetchedDoc",
    "PDFDownloader",
    "download_many",
    "finalize_pdf_path",
]

# ---------------- constants / helpers ----------------

_PDF_MAGIC = b"%PDF-"
# Common image magics
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xFF\xD8\xFF"
_GIF_MAGIC = b"GIF8"
_RIFF_MAGIC = b"RIFF"  # WebP uses RIFF header + "WEBP" marker
_TIFF_MAGIC_LE = b"II*\x00"
_TIFF_MAGIC_BE = b"MM\x00*"

_HTML_SNIPPETS = (b"<html", b"<!doctype html", b"<head", b"<body")

_REASON_OK_PDF = "ok_pdf"
_REASON_OK_IMG = "ok_image"

_SNIFF_N = 8 * 1024  # bytes used for magic/HTML sniffing

_IMAGE_MIN_BYTES_DEFAULT = 256  # guardrail for tiny/empty images

_MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/tiff": ".tif",
    "image/x-tiff": ".tif",
}

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff"}


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v else default


def _parse_rate_hosts(s: str) -> Dict[str, float]:
    """
    Parse 'host=rpm,host2=rpm' -> seconds to sleep per request.
    rpm<=0 disables throttling for that host.
    """
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        h, r = part.split("=", 1)
        try:
            rpm = float(r)
            out[h.strip().lower()] = 60.0 / rpm if rpm > 0 else 0.0
        except Exception:
            continue
    return out


# ---------------- public dataclasses ----------------

@dataclass
class DownloadConfig:
    timeout: Optional[int] = None
    max_redirects: Optional[int] = None
    min_pdf_bytes: Optional[int] = None
    user_agent: Optional[str] = None
    rate_hosts: Optional[str] = None  # "host=rpm,host2=rpm"
    min_image_bytes: Optional[int] = None

    @staticmethod
    def from_env(*_args, **_kwargs) -> "DownloadConfig":
        return DownloadConfig(
            timeout=_env_int("CITEVIZOR_TIMEOUT", 20),
            max_redirects=_env_int("CITEVIZOR_MAX_REDIRECTS", 5),
            min_pdf_bytes=_env_int("CITEVIZOR_PDF_MIN_BYTES", 1024),
            user_agent=_env_str("CITEVIZOR_USER_AGENT", "CiteVizorBot/1.0"),
            rate_hosts=os.environ.get("CITEVIZOR_RATE_HOSTS", ""),
            min_image_bytes=_env_int("CITEVIZOR_IMAGE_MIN_BYTES", _IMAGE_MIN_BYTES_DEFAULT),
        )


@dataclass
class FetchedDoc:
    ok: bool
    reason: Optional[str]
    path: Optional[str]          # final saved file path (with correct extension)
    url: str
    final_url: Optional[str]
    status: Optional[int]
    content_type: Optional[str]
    host: Optional[str]
    doc_id: Optional[str] = None  # carried through to parser/layout


# ---------------- rate limiting ----------------

class _RateLimiter:
    """Simple per-host throttle to avoid burst 403/429."""

    def __init__(self, host_sleep: Dict[str, float]):
        self.host_sleep = host_sleep
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def sleep_if_needed(self, host: str) -> None:
        host = (host or "").lower()
        delay = self.host_sleep.get(host, 0.0)
        if delay <= 0:
            return
        with self._lock:
            last = self._last.get(host, 0.0)
            now = time.time()
            wait = (last + delay) - now
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.time()


# ---------------- low-level PDF(+Image) downloader ----------------

class PDFDownloader:

    def __init__(
        self,
        timeout: int | None = None,
        max_redirects: int | None = None,
        min_pdf_bytes: int | None = None,
        user_agent: str | None = None,
        rate_hosts: str | None = None,
        session: Optional[requests.Session] = None,
        min_image_bytes: int | None = None,
    ):
        self.timeout = _env_int("CITEVIZOR_TIMEOUT", 20) if timeout is None else timeout
        self.max_redirects = _env_int("CITEVIZOR_MAX_REDIRECTS", 5) if max_redirects is None else max_redirects
        self.min_pdf_bytes = _env_int("CITEVIZOR_PDF_MIN_BYTES", 1024) if min_pdf_bytes is None else min_pdf_bytes
        self.min_image_bytes = _env_int("CITEVIZOR_IMAGE_MIN_BYTES", _IMAGE_MIN_BYTES_DEFAULT) if min_image_bytes is None else min_image_bytes
        self.ua = _env_str("CITEVIZOR_USER_AGENT", "CiteVizorBot/1.0") if user_agent is None else user_agent

        rate_hosts_str = os.environ.get("CITEVIZOR_RATE_HOSTS", "") if rate_hosts is None else rate_hosts
        self.rate_limiter = _RateLimiter(_parse_rate_hosts(rate_hosts_str))

        self.sess = session or self._make_session()

    def _make_session(self) -> requests.Session:
        s = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["HEAD", "GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=32)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update({"User-Agent": self.ua, "Accept": "*/*"})
        lvl = os.environ.get("CITEVIZOR_URLLIB3_LEVEL")
        if lvl:
            logging.getLogger("urllib3").setLevel(getattr(logging, lvl.upper(), logging.WARNING))
        if os.environ.get("CITEVIZOR_HTTP_CONN_CLOSE") == "1":
            s.headers.update({"Connection": "close"})
        return s

    # ---------- type checks ----------

    @staticmethod
    def _is_http_ok(status: Optional[int]) -> bool:
        return status is not None and 200 <= status < 300

    @staticmethod
    def _looks_like_pdf_ct(ct: Optional[str]) -> bool:
        return bool(ct) and "application/pdf" in ct.lower()

    @staticmethod
    def _looks_like_html_ct(ct: Optional[str]) -> bool:
        if not ct:
            return False
        l = ct.lower()
        return "text/html" in l or "application/xhtml" in l

    @staticmethod
    def _looks_like_image_ct(ct: Optional[str]) -> bool:
        return bool(ct) and ct.lower().startswith("image/")

    @staticmethod
    def _sniff_is_pdf(buf: bytes) -> bool:
        return buf.startswith(_PDF_MAGIC)

    @staticmethod
    def _sniff_image_ext(buf: bytes) -> Optional[str]:
        """Return image extension if magic matches; else None."""
        if buf.startswith(_PNG_MAGIC):
            return ".png"
        if buf.startswith(_JPEG_MAGIC):
            return ".jpg"
        if buf.startswith(_GIF_MAGIC):
            return ".gif"
        if buf.startswith(_TIFF_MAGIC_LE) or buf.startswith(_TIFF_MAGIC_BE):
            return ".tif"
        if buf[:4] == _RIFF_MAGIC and b"WEBP" in buf[:16]:
            return ".webp"
        return None

    @staticmethod
    def _norm_url(s: str) -> str:
        try:
            u = urlparse(s)
            u = u._replace(fragment="")
            return urlunparse(u)
        except Exception:
            return s

    @staticmethod
    def _is_doi_like_url(u: str) -> bool:
        if not u:
            return False
        h = urlparse(u).netloc.lower()
        return any(
            x in h
            for x in (
                "doi.org",
                "dx.doi.org",
                "science.org",
                "acm.org",
                "ieee.org",
                "springer.com",
                "nature.com",
                "wiley.com",
                "tandfonline.com",
                "sciencedirect.com",
                "cell.com",
            )
        )

    # ---------- probing helpers ----------

    def _read_head_once(self, r: requests.Response) -> Tuple[bytes, bytes]:
        """
        Read EXACTLY one chunk from the stream. Return (head_for_sniff, tail_of_first_chunk).
        """
        it = r.iter_content(chunk_size=_SNIFF_N)
        try:
            first = next(it)
        except StopIteration:
            return b"", b""
        head = first[:_SNIFF_N]
        tail = first[_SNIFF_N:]
        return head, tail

    # ---------- core ----------

    def fetch(self, url: str, referer: Optional[str] = None) -> FetchedDoc:
        url = self._norm_url(url)
        host = urlparse(url).netloc.lower()
        self.rate_limiter.sleep_if_needed(host)

        headers = {"User-Agent": self.ua}
        if referer:
            headers["Referer"] = referer

        # 1) HEAD probe
        try:
            head_resp = self.sess.head(
                url,
                allow_redirects=True,
                timeout=self.timeout,
                headers={**headers, "Accept": "application/pdf, image/*;q=0.9, */*;q=0.1"},
            )
            final_url = head_resp.url
            status = head_resp.status_code
            ct = (head_resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()

            if len(head_resp.history) > self.max_redirects:
                return FetchedDoc(False, "max_redirects", None, url, final_url, status, ct, host)

            if self._is_http_ok(status) and self._looks_like_pdf_ct(ct):
                return self._get_binary(final_url, headers=headers)

            if self._is_http_ok(status) and self._looks_like_image_ct(ct):
                return self._get_binary(final_url, headers=headers)

            if self._is_http_ok(status) and self._looks_like_html_ct(ct):
                if self._is_doi_like_url(final_url):
                    return self._get_with_accept_pdf(final_url, headers=headers)
                return FetchedDoc(False, "html_not_binary", None, url, final_url, status, ct, host)

            return self._get_binary(final_url, headers=headers)

        except requests.exceptions.SSLError:
            return FetchedDoc(False, "bad_ssl", None, url, None, None, None, host)
        except requests.exceptions.Timeout:
            return FetchedDoc(False, "timeout", None, url, None, None, None, host)
        except requests.exceptions.TooManyRedirects:
            return FetchedDoc(False, "max_redirects", None, url, None, None, None, host)
        except RemoteDisconnected:
            return FetchedDoc(False, "remote_disconnected", None, url, None, None, None, host)
        except requests.exceptions.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            reason = "http_error" if status and 400 <= status else "network_error"
            return FetchedDoc(False, reason, None, url, None, status, None, host)

    def _get_with_accept_pdf(self, url: str, headers: Dict[str, str]) -> FetchedDoc:
        h = {**headers, "Accept": "application/pdf, image/*;q=0.9, application/octet-stream;q=0.8, */*;q=0.1"}
        host = urlparse(url).netloc.lower()
        try:
            r = self.sess.get(url, allow_redirects=True, timeout=self.timeout, headers=h, stream=True)
            if len(r.history) > self.max_redirects:
                return FetchedDoc(False, "max_redirects", None, url, r.url, r.status_code, None, host)

            return self._classify_and_save(r)

        except requests.exceptions.SSLError:
            return FetchedDoc(False, "bad_ssl", None, url, None, None, None, host)
        except requests.exceptions.Timeout:
            return FetchedDoc(False, "timeout", None, url, None, None, None, host)
        except requests.exceptions.TooManyRedirects:
            return FetchedDoc(False, "max_redirects", None, url, None, None, None, host)
        except RemoteDisconnected:
            return FetchedDoc(False, "remote_disconnected", None, url, None, None, None, host)
        except requests.exceptions.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            reason = "http_error" if status and 400 <= status else "network_error"
            return FetchedDoc(False, reason, None, url, None, status, None, host)

    def _get_binary(self, url: str, headers: Dict[str, str]) -> FetchedDoc:
        host = urlparse(url).netloc.lower()
        try:
            r = self.sess.get(url, allow_redirects=True, timeout=self.timeout, headers=headers, stream=True)
            if len(r.history) > self.max_redirects:
                return FetchedDoc(False, "max_redirects", None, url, r.url, r.status_code, None, host)

            return self._classify_and_save(r)

        except requests.exceptions.SSLError:
            return FetchedDoc(False, "bad_ssl", None, url, None, None, None, host)
        except requests.exceptions.Timeout:
            return FetchedDoc(False, "timeout", None, url, None, None, None, host)
        except requests.exceptions.TooManyRedirects:
            return FetchedDoc(False, "max_redirects", None, url, None, None, None, host)
        except RemoteDisconnected:
            return FetchedDoc(False, "remote_disconnected", None, url, None, None, None, host)
        except requests.exceptions.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            reason = "http_error" if status and 400 <= status else "network_error"
            return FetchedDoc(False, reason, None, url, None, status, None, host)

    # ---- classify + save ----

    def _classify_and_save(self, r: requests.Response) -> FetchedDoc:

        status = r.status_code
        ct = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        host = urlparse(r.url).netloc.lower()

        sniff, tail = self._read_head_once(r)

        if self._looks_like_html_ct(ct) or self._sniff_is_html(sniff):
            reason = "paywall_html" if status in (200, 302, 303) else "html_not_binary"
            return FetchedDoc(False, reason, None, r.request.url, r.url, status, ct, host)

        if self._looks_like_pdf_ct(ct) or self._sniff_is_pdf(sniff):
            return self._save_stream(r, sniff, tail, ext=".pdf", min_bytes=self.min_pdf_bytes, ok_reason=_REASON_OK_PDF)

        img_ext = None
        if self._looks_like_image_ct(ct):
            img_ext = _MIME_TO_EXT.get(ct, None)
        if not img_ext:
            img_ext = self._sniff_image_ext(sniff)

        if img_ext:
            return self._save_stream(r, sniff, tail, ext=img_ext, min_bytes=self.min_image_bytes, ok_reason=_REASON_OK_IMG)

        return FetchedDoc(False, "unknown_binary", None, r.request.url, r.url, status, ct, host)

    @staticmethod
    def _sniff_is_html(buf: bytes) -> bool:
        low = buf[:2048].lower()
        return any(tag in low for tag in _HTML_SNIPPETS)

    def _save_stream(self, r: requests.Response, head: bytes, first_tail: bytes, *, ext: str, min_bytes: int, ok_reason: str) -> FetchedDoc:
        fd, path = tempfile.mkstemp(prefix="citevizor_", suffix=ext)
        total = 0
        try:
            with os.fdopen(fd, "wb") as f:
                if head:
                    f.write(head)
                    total += len(head)
                if first_tail:
                    f.write(first_tail)
                    total += len(first_tail)
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total += len(chunk)

            if total < max(0, int(min_bytes or 0)):
                try:
                    os.remove(path)
                except OSError:
                    pass
                return FetchedDoc(
                    False,
                    "too_small_binary",
                    None,
                    r.request.url,
                    r.url,
                    r.status_code,
                    (r.headers.get("Content-Type") or "").split(";")[0].strip().lower(),
                    urlparse(r.url).netloc.lower(),
                )

            return FetchedDoc(
                True,
                ok_reason,
                path,
                r.request.url,
                r.url,
                r.status_code,
                (r.headers.get("Content-Type") or "").split(";")[0].strip().lower(),
                urlparse(r.url).netloc.lower(),
            )

        except Exception:
            try:
                os.remove(path)
            except OSError:
                pass
            return FetchedDoc(
                False,
                "write_error",
                None,
                r.request.url,
                r.url,
                r.status_code,
                (r.headers.get("Content-Type") or "").split(";")[0].strip().lower(),
                urlparse(r.url).netloc.lower(),
            )


# ---------------- batch helpers (path contract) ----------------

def _ensure_dir(d: str | Path) -> None:
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _candidate_url(c: Dict[str, Any]) -> Optional[str]:
    return c.get("pdf_url") or c.get("url") or c.get("link")


def _ext_from_temp_path(path: Optional[str]) -> str:
    if not path:
        return ".bin"
    base = os.path.basename(path)
    _, ext = os.path.splitext(base)
    return ext or ".bin"


def _stable_doc_id(c: Dict[str, Any], fallback_key: str) -> str:
    """Prefer incoming doc_id; else use sha1 of URL-ish string as a stable id."""
    d = (c.get("doc_id") or "").strip()
    if d:
        return d
    key = (c.get("url") or c.get("pdf_url") or c.get("link") or fallback_key or "").encode("utf-8", "ignore")
    return hashlib.sha1(key).hexdigest()[:16]


def _is_image_ct_or_ext(content_type: Optional[str], path: Optional[str]) -> bool:
    if content_type and content_type.lower().startswith("image/"):
        return True
    if path:
        _, ext = os.path.splitext(path)
        if ext.lower() in _IMG_EXTS:
            return True
    return False


def _register_web_image(doc_id: str, src_path: Path, hint: str = "") -> None:
    """
    Copy the downloaded image into evidence/<doc_id>/web_images and append a web_image entry to pack.json.
    Non-throwing; best-effort.
    """
    try:
        if not doc_id or not src_path.exists():
            return
        ev_dir = Path(EVIDENCE_DIR) / doc_id / "web_images"
        ev_dir.mkdir(parents=True, exist_ok=True)
        dst = ev_dir / f"web_{uuid.uuid4().hex[:8]}{src_path.suffix.lower() or '.png'}"
        shutil.copyfile(str(src_path), str(dst))

        pack_p = ev_dir.parent / "pack.json"
        try:
            data = json.loads(pack_p.read_text(encoding="utf-8")) if pack_p.exists() else {"doc_id": doc_id, "paragraphs": [], "tables": [], "figures": []}
        except Exception:
            data = {"doc_id": doc_id, "paragraphs": [], "tables": [], "figures": []}
        figs = data.get("figures") or []
        figs.append({
            "id": f"webimg-{uuid.uuid4().hex[:8]}",
            "type": "web_image",
            "doc_id": doc_id,
            "image_path": str(dst),
            "caption": (hint or "")[:200],
        })
        data["figures"] = figs
        pack_p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # swallow any errors; downloading must not fail because of registration
        pass


def _normalize_to_contract(doc_id: str, final_path: str) -> str:
    """
    Ensure the file location obeys the contract:
      - If extension is .pdf => move to storage/docs/<doc_id>/<doc_id>.pdf
      - Else (image, etc.) leave in place (images are handled by _register_web_image)
    Returns the possibly-updated final path.
    """
    try:
        if not final_path:
            return final_path
        src = Path(final_path)
        if not src.exists():
            return final_path
        if src.suffix.lower() != ".pdf":
            return final_path  # non-PDF is not normalized here
        target = pdf_path_for(doc_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == target.resolve():
            return str(target)
        # move/replace atomically when possible
        try:
            os.replace(str(src), str(target))
        except OSError:
            with open(src, "rb") as s, open(target, "wb") as t:
                t.write(s.read())
            try:
                os.remove(src)
            except OSError:
                pass
        return str(target)
    except Exception:
        return final_path


# ---------------- batch API ----------------

def download_many(
    candidates: Iterable[Dict[str, Any]],
    out_dir: str | Path,
    trace: Optional[Dict[str, Any]] = None,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """
    For each candidate:
      - choose url (pdf_url > url > link)
      - fetch a real PDF or Image
      - if ok: move to out_dir/<doc_id>/<doc_id>.<ext> (stable), then normalize PDF to contract path
      - image: also register into evidence/<doc_id>/web_images and append to pack.json
      - append a normalized trace.downloads item (with doc_id)
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    if trace is None:
        trace = {}
    downloads_list: List[Dict[str, Any]] = trace.setdefault("downloads", [])

    dl = PDFDownloader(session=session)
    results: List[Dict[str, Any]] = []

    for c in candidates:
        url = _candidate_url(c)
        host = (c.get("host") or (urlparse(url).netloc if url else "")).lower() if url else ""
        if not url:
            res = FetchedDoc(False, "no_url", None, "", None, None, None, host, doc_id=c.get("doc_id"))
        else:
            res = dl.fetch(url, referer=c.get("referer"))

        # decide doc_id EARLY and ensure per-doc directory
        doc_id = _stable_doc_id(c, fallback_key=(res.final_url or res.url or ""))

        final_path = None
        if res.ok and res.path:
            ext = _ext_from_temp_path(res.path)
            doc_dir = out_dir / doc_id
            _ensure_dir(doc_dir)
            final_path = str(doc_dir / f"{doc_id}{ext}")
            try:
                os.replace(res.path, final_path)
            except OSError:
                with open(res.path, "rb") as src, open(final_path, "wb") as dst:
                    dst.write(src.read())
                try:
                    os.remove(res.path)
                except OSError:
                    pass

            # normalize PDF path to contract
            final_path = _normalize_to_contract(doc_id, final_path)

            # register image evidence (best-effort)
            if _is_image_ct_or_ext(res.content_type, final_path):
                _register_web_image(doc_id, Path(final_path), hint=(c.get("url") or ""))

        downloads_list.append(
            {
                "host": res.host,
                "url": res.url,
                "final_url": res.final_url,
                "status": res.status,
                "content_type": res.content_type,
                "ok": bool(res.ok),
                "reason": res.reason,
                "path": final_path,
                "doc_id": doc_id,
            }
        )

        results.append(
            {
                "ok": bool(res.ok),
                "reason": res.reason,
                "path": final_path,
                "url": res.url,
                "final_url": res.final_url,
                "status": res.status,
                "content_type": res.content_type,
                "host": res.host,
                "doc_id": doc_id,
            }
        )

    return results


# ---------------- legacy wrapper ----------------

class Downloader:

    def __init__(self, config: Optional[DownloadConfig] = None, session: Optional[requests.Session] = None):
        self.config = config or DownloadConfig.from_env()
        self._pdf = PDFDownloader(
            timeout=self.config.timeout,
            max_redirects=self.config.max_redirects,
            min_pdf_bytes=self.config.min_pdf_bytes,
            user_agent=self.config.user_agent,
            rate_hosts=self.config.rate_hosts,
            session=session,
            min_image_bytes=self.config.min_image_bytes,
        )

    def download_one(
        self,
        candidate: Dict[str, Any],
        out_dir: str | Path,
        trace: Optional[Dict[str, Any]] = None,
    ) -> FetchedDoc:
        out_dir = Path(out_dir)
        _ensure_dir(out_dir)
        if trace is None:
            trace = {}
        trace.setdefault("downloads", [])

        url = _candidate_url(candidate)
        host = (candidate.get("host") or (urlparse(url).netloc if url else "")).lower() if url else ""
        if not url:
            res = FetchedDoc(False, "no_url", None, "", None, None, None, host)
        else:
            res = self._pdf.fetch(url, referer=candidate.get("referer"))

        doc_id = _stable_doc_id(candidate, fallback_key=(res.final_url or res.url or ""))

        final_path = None
        if res.ok and res.path:
            ext = _ext_from_temp_path(res.path)
            doc_dir = out_dir / doc_id
            _ensure_dir(doc_dir)
            final_path = str(doc_dir / f"{doc_id}{ext}")
            try:
                os.replace(res.path, final_path)
            except OSError:
                with open(res.path, "rb") as src, open(final_path, "wb") as dst:
                    dst.write(src.read())
                try:
                    os.remove(res.path)
                except OSError:
                    pass

            # normalize PDF path to contract
            final_path = _normalize_to_contract(doc_id, final_path)

            if _is_image_ct_or_ext(res.content_type, final_path):
                _register_web_image(doc_id, Path(final_path), hint=(candidate.get("url") or ""))

        trace["downloads"].append(
            {
                "host": res.host,
                "url": res.url,
                "final_url": res.final_url,
                "status": res.status,
                "content_type": res.content_type,
                "ok": bool(res.ok),
                "reason": res.reason,
                "path": final_path,
                "doc_id": doc_id,
            }
        )

        return FetchedDoc(
            ok=res.ok,
            reason=res.reason,
            path=final_path,
            url=res.url,
            final_url=res.final_url,
            status=res.status,
            content_type=res.content_type,
            host=res.host,
            doc_id=doc_id,
        )

    def download_many(
        self,
        candidates: Iterable[Dict[str, Any]],
        out_dir: str | Path,
        trace: Optional[Dict[str, Any]] = None,
    ) -> List[FetchedDoc]:
        if trace is None:
            trace = {}
        dicts = download_many(candidates, out_dir, trace, session=self._pdf.sess)
        out: List[FetchedDoc] = []
        for d in dicts:
            out.append(
                FetchedDoc(
                    ok=bool(d.get("ok")),
                    reason=d.get("reason"),
                    path=d.get("path"),
                    url=d.get("url", ""),
                    final_url=d.get("final_url"),
                    status=d.get("status"),
                    content_type=d.get("content_type"),
                    host=d.get("host"),
                    doc_id=d.get("doc_id"),
                )
            )
        return out


# ---------------- public normalization hook for backend ----------------

def _extract_schema_fd_paths(obj: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to read (doc_id, pdf_path) from a schema.FetchedDoc-like object.
    Returns (doc_id, pdf_path_str)
    """
    try:
        doc_id = getattr(obj, "doc_id", None)
        paths = getattr(obj, "paths", None)
        pdf_path = getattr(paths, "pdf_path", None) if paths else None
        return (doc_id, str(pdf_path) if pdf_path else None)
    except Exception:
        return (None, None)


def _write_schema_fd_pdf_path(obj: Any, new_path: str) -> None:
    """
    Try to write back pdf_path into a schema.FetchedDoc-like object.
    Safe no-op if structure differs.
    """
    try:
        paths = getattr(obj, "paths", None)
        if paths is not None:
            # pydantic models usually accept assignment to attributes backed by validators
            setattr(paths, "pdf_path", Path(new_path))
    except Exception:
        pass


def finalize_pdf_path(obj: Any, *, docs_dir: Optional[Union[str, Path]] = None) -> Any:
    """
    Enforce the normalized PDF path for a given object.
    - Accepts: 
        * schema.FetchedDoc (with .doc_id and .paths.pdf_path)
        * this module's FetchedDoc (with .doc_id and .path)
        * a dict containing {"doc_id": ..., "path": ...}
    - If the pointed file is a PDF and not already at storage/docs/<doc_id>/<doc_id>.pdf,
      it will be moved there and the object will be updated.
    - Returns the original object (possibly mutated).
    """
    try:
        # read doc_id + current path from various shapes
        doc_id = None
        cur_path = None

        if isinstance(obj, dict):
            doc_id = obj.get("doc_id")
            cur_path = obj.get("path")
        else:
            # try schema-like
            doc_id, cur_path = _extract_schema_fd_paths(obj)
            if not (doc_id and cur_path):
                # try local FetchedDoc
                doc_id = getattr(obj, "doc_id", doc_id)
                cur_path = getattr(obj, "path", cur_path)

        if not (doc_id and cur_path):
            return obj

        # only normalize PDFs
        if Path(cur_path).suffix.lower() != ".pdf":
            return obj

        target = pdf_path_for(doc_id)
        if docs_dir:
            # If caller wants to override root, honor it but keep filename contract
            docs_dir = Path(docs_dir)
            target = docs_dir / doc_id / f"{doc_id}.pdf"

        target.parent.mkdir(parents=True, exist_ok=True)
        src = Path(cur_path)
        if not src.exists():
            return obj

        if src.resolve() != target.resolve():
            try:
                os.replace(str(src), str(target))
            except OSError:
                with open(src, "rb") as s, open(target, "wb") as t:
                    t.write(s.read())
                try:
                    os.remove(src)
                except OSError:
                    pass

        # write back to object shape
        if isinstance(obj, dict):
            obj["path"] = str(target)
        else:
            # schema-like first
            _write_schema_fd_pdf_path(obj, str(target))
            # local FetchedDoc fallback
            try:
                setattr(obj, "path", str(target))
            except Exception:
                pass

        return obj
    except Exception:
        return obj
