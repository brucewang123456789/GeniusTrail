# C:\CiteVizor\storage\http_cache.py
# -*- coding: utf-8 -*-
"""
CiteVizor - HTTP disk cache with ETag/Last-Modified, TTL, and LRU pruning

What this module provides
- Disk-backed GET cache (bytes) with atomic writes and per-entry file locks
- Conditional requests using ETag / Last-Modified to minimize bandwidth
- TTL controls (default TTL, min TTL clamp), offline mode (serve-stale)
- Max cache size with simple LRU pruning (by last_access)
- Structured logging (integrates with infra.logging if available; otherwise no-op)

Optional environment keys (all optional; can also be supplied via config.load_config().raw):
  HTTP_CACHE_DIR            : default <STORAGE_DIR>/http_cache
  HTTP_CACHE_MAX_BYTES      : default 2_000_000_000 (about 2GB)
  HTTP_CACHE_DEFAULT_TTL_S  : default 7 days
  HTTP_CACHE_MIN_TTL_S      : default 600 seconds
  HTTP_CACHE_OFFLINE        : "1" to force offline (never hit network)
  HTTP_TIMEOUT_S            : default 20 seconds
  HTTP_CACHE_USER_AGENT     : default "CiteVizor/1.0 (+cache)"

No circular imports: this module only imports STORAGE_DIR from `schemas`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

# Canonical storage root (no circular deps)
try:
    from schemas import STORAGE_DIR  # type: ignore
except Exception:
    STORAGE_DIR = Path.cwd() / "storage"

# Optional config/infra logging
try:
    from config import load_config  # type: ignore
except Exception:
    load_config = None  # type: ignore

try:
    from infra.logging import setup_logging, get_logger, span  # type: ignore
except Exception:
    def setup_logging(*a, **k):  # type: ignore
        return None
    def get_logger(name: str):  # type: ignore
        class _L:
            def info(self, *a, **k): pass
            def debug(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass
        return _L()
    class span:  # type: ignore
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, et, e, tb): return False

LOG = get_logger("http_cache")


# ----------------------------- small utils -------------------------------------

def _env_or_cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    if v is not None:
        return v
    if load_config:
        try:
            cfg = load_config()
            raw = getattr(cfg, "raw", {}) or {}
            if key in raw:
                return str(raw[key])
        except Exception:
            pass
    return default

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    dest = Path(dest)
    _ensure_dir(dest.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".tmp-{dest.name}.", dir=str(dest.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, dest)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def _atomic_write_json(dest: Path, obj: Any) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(dest, payload)

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _now_s() -> int:
    return int(time.time())

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

class FileLock:
    """
    Simple file lock <target>.lock; cooperative locking for single-host use.
    """
    def __init__(self, lock_path: Path, timeout_s: float = 15.0, poll_s: float = 0.05, stale_s: float = 600.0):
        self.lock_path = Path(lock_path)
        self.timeout_s = max(0.0, timeout_s)
        self.poll_s = max(0.01, poll_s)
        self.stale_s = max(60.0, stale_s)
        self._fd = None

    def acquire(self) -> None:
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, f"pid={os.getpid()} ts={_now_s()}\n".encode("utf-8"))
                return
            except FileExistsError:
                try:
                    st = self.lock_path.stat()
                    if (_now_s() - int(st.st_mtime)) > self.stale_s:
                        os.remove(self.lock_path); continue
                except FileNotFoundError:
                    continue
                if time.time() >= deadline:
                    raise TimeoutError(f"Lock timeout: {self.lock_path}")
                time.sleep(self.poll_s)

    def release(self) -> None:
        try:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass
        finally:
            self._fd = None

    def __enter__(self) -> "FileLock":
        self.acquire(); return self

    def __exit__(self, et, e, tb) -> None:
        self.release()


# ----------------------------- datamodel ---------------------------------------

@dataclass
class CacheResponse:
    """
    Lightweight response object that mimics requests.Response for the parts we use.
    """
    url: str
    status_code: int
    headers: Dict[str, str]
    content: bytes
    from_cache: bool
    cache_path: Optional[Path]
    encoding: Optional[str]

    def text(self) -> str:
        enc = self.encoding or "utf-8"
        try:
            return self.content.decode(enc, errors="replace")
        except Exception:
            return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text())


# ----------------------------- HttpCache ---------------------------------------

class HttpCache:
    """
    Disk cache for HTTP GET with conditional requests and TTL.

    File layout (default):
      <STORAGE_DIR>/http_cache/
        2a/2a9f... .bin
        2a/2a9f... .json
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_bytes: Optional[int] = None,
        default_ttl_s: Optional[int] = None,
        min_ttl_s: Optional[int] = None,
        offline: Optional[bool] = None,
        timeout_s: Optional[float] = None,
        user_agent: Optional[str] = None,
    ):
        root = STORAGE_DIR
        # resolve from env/config with safe defaults
        cd = cache_dir or Path(_env_or_cfg("HTTP_CACHE_DIR") or (root / "http_cache"))
        mb = int(_env_or_cfg("HTTP_CACHE_MAX_BYTES", "2000000000") or 2_000_000_000)
        ttl = int(_env_or_cfg("HTTP_CACHE_DEFAULT_TTL_S", "604800") or 604800)  # 7d
        mttl = int(_env_or_cfg("HTTP_CACHE_MIN_TTL_S", "600") or 600)           # 10m
        off = bool(int(_env_or_cfg("HTTP_CACHE_OFFLINE", "0") or "0"))
        to = float(_env_or_cfg("HTTP_TIMEOUT_S", "20") or 20.0)
        ua = _env_or_cfg("HTTP_CACHE_USER_AGENT", "CiteVizor/1.0 (+cache)")

        self.cache_dir = _ensure_dir(cd)
        self.max_bytes = max_bytes if max_bytes is not None else mb
        self.default_ttl_s = default_ttl_s if default_ttl_s is not None else ttl
        self.min_ttl_s = min_ttl_s if min_ttl_s is not None else mttl
        self.offline = off if offline is None else offline
        self.timeout_s = to if timeout_s is None else timeout_s
        self.user_agent = ua if user_agent is None else user_agent

        self._inmem_locks: Dict[str, threading.Lock] = {}
        LOG.info("http_cache.init", extra={
            "dir": str(self.cache_dir), "max_bytes": self.max_bytes,
            "ttl": self.default_ttl_s, "min_ttl": self.min_ttl_s, "offline": self.offline
        })

    # ---------- key & file paths ----------

    def _key_of(self, url: str) -> str:
        # Key only by URL (avoid leaking secrets in headers to cache key)
        return _sha256(url)

    def _paths_of(self, key: str) -> Tuple[Path, Path]:
        sub = key[:2]
        base = _ensure_dir(self.cache_dir / sub) / key
        return base.with_suffix(".bin"), base.with_suffix(".json")

    # ---------- helpers ----------

    def _entry_lock(self, bin_path: Path) -> FileLock:
        return FileLock(Path(str(bin_path) + ".lock"))

    def _now(self) -> int:
        return _now_s()

    def _is_fresh(self, meta: Dict[str, Any]) -> bool:
        now = self._now()
        exp = int(meta.get("expires_at", 0))
        return exp > now

    def _clamp_ttl(self, ttl_s: Optional[int]) -> int:
        if ttl_s is None or ttl_s <= 0:
            ttl_s = self.default_ttl_s
        return max(self.min_ttl_s, int(ttl_s))

    def _headers_pick(self, resp_headers: Dict[str, str]) -> Dict[str, str]:
        # Keep a small set of headers that are useful to restore context
        keep = ("Content-Type", "Content-Encoding", "ETag", "Last-Modified", "Date", "Cache-Control", "Expires")
        out: Dict[str, str] = {}
        for k, v in resp_headers.items():
            if k in keep:
                out[k] = v
        return out

    # ---------- public API ----------

    def get(
        self,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        ttl_s: Optional[int] = None,
        allow_cache: bool = True,
        force_refresh: bool = False,
        timeout_s: Optional[float] = None,
    ) -> CacheResponse:
        """
        GET with cache. Returns CacheResponse.
        - If allow_cache and fresh entry exists, returns it.
        - Else, uses ETag/Last-Modified (if present) for conditional request.
        - If offline==True and no fresh cache, returns stale if present, otherwise raises.
        """
        key = self._key_of(url)
        bin_path, meta_path = self._paths_of(key)
        lock = self._entry_lock(bin_path)
        req_timeout = timeout_s if timeout_s is not None else self.timeout_s
        ttl = self._clamp_ttl(ttl_s)

        with lock:  # serialize writers/readers for this entry
            meta = _read_json(meta_path)
            # serve fresh cache if allowed and not forcing refresh
            if allow_cache and not force_refresh and meta.get("status_code") == 200 and self._is_fresh(meta):
                try:
                    content = bin_path.read_bytes()
                    meta["last_access"] = self._now()
                    _atomic_write_json(meta_path, meta)
                    LOG.debug("cache.hit", extra={"url": url})
                    return CacheResponse(
                        url=url,
                        status_code=int(meta.get("status_code", 200)),
                        headers=meta.get("headers", {}),
                        content=content,
                        from_cache=True,
                        cache_path=bin_path,
                        encoding=meta.get("encoding"),
                    )
                except Exception:
                    # fall through to re-fetch if corrupted
                    LOG.warning("cache.corrupt", extra={"url": url})

            # offline path: serve stale if any
            if self.offline:
                if bin_path.exists() and meta.get("status_code") == 200:
                    content = bin_path.read_bytes()
                    LOG.info("cache.offline_stale", extra={"url": url})
                    return CacheResponse(
                        url=url,
                        status_code=int(meta.get("status_code", 200)),
                        headers=meta.get("headers", {}),
                        content=content,
                        from_cache=True,
                        cache_path=bin_path,
                        encoding=meta.get("encoding"),
                    )
                raise RuntimeError(f"offline mode: no cache for {url}")

            # Build request headers with conditional validators
            req_headers = {"User-Agent": self.user_agent}
            if headers:
                req_headers.update(headers)
            if not force_refresh and meta:
                et = meta.get("etag")
                lm = meta.get("last_modified")
                if et:
                    req_headers["If-None-Match"] = et
                if lm:
                    req_headers["If-Modified-Since"] = lm

            with span(LOG, "http.get", url=url):
                r = requests.get(url, headers=req_headers, timeout=req_timeout)
                status = r.status_code

            if status == 304 and bin_path.exists():
                # Not modified: keep old bytes; extend freshness
                content = bin_path.read_bytes()
                new_meta = {
                    **meta,
                    "status_code": 200,  # effective 200 from cache
                    "headers": meta.get("headers", {}),
                    "etag": meta.get("etag"),
                    "last_modified": meta.get("last_modified"),
                    "fetched_at": meta.get("fetched_at", self._now()),
                    "expires_at": self._now() + ttl,
                    "last_access": self._now(),
                }
                _atomic_write_json(meta_path, new_meta)
                LOG.info("cache.304", extra={"url": url})
                return CacheResponse(
                    url=url, status_code=200, headers=new_meta.get("headers", {}),
                    content=content, from_cache=True, cache_path=bin_path,
                    encoding=new_meta.get("encoding"),
                )

            # New content (200 or others)
            content = r.content or b""
            ct = r.headers.get("Content-Type") or ""
            enc = r.encoding
            etag = r.headers.get("ETag")
            lastm = r.headers.get("Last-Modified")
            picked_headers = self._headers_pick(r.headers)

            # Cache only 200 responses; other statuses are returned but not cached long
            if status == 200:
                meta_new = {
                    "url": url,
                    "status_code": int(status),
                    "headers": picked_headers,
                    "etag": etag,
                    "last_modified": lastm,
                    "content_type": ct,
                    "encoding": enc,
                    "fetched_at": self._now(),
                    "expires_at": self._now() + ttl,
                    "last_access": self._now(),
                    "size": len(content),
                }
                _atomic_write_bytes(bin_path, content)
                _atomic_write_json(meta_path, meta_new)
                LOG.info("cache.store", extra={"url": url, "size": len(content)})
                self._maybe_prune()
            else:
                LOG.warning("http.non200", extra={"url": url, "status": status})

            return CacheResponse(
                url=url, status_code=status, headers=picked_headers,
                content=content, from_cache=False, cache_path=bin_path if status == 200 else None,
                encoding=enc,
            )

    # ---------- maintenance ----------

    def invalidate(self, url: str) -> None:
        key = self._key_of(url)
        bin_path, meta_path = self._paths_of(key)
        with self._entry_lock(bin_path):
            try:
                bin_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                (Path(str(bin_path) + ".lock")).unlink(missing_ok=True)
                LOG.info("cache.invalidate", extra={"url": url})
            except Exception as e:
                LOG.warning("cache.invalidate_error", extra={"url": url, "err": str(e)[:200]})

    def stat(self) -> Dict[str, Any]:
        total = 0
        files = 0
        for p in self.cache_dir.rglob("*.bin"):
            try:
                total += p.stat().st_size
                files += 1
            except Exception:
                pass
        return {"dir": str(self.cache_dir), "bytes": total, "files": files}

    def _maybe_prune(self) -> None:
        st = self.stat()
        if st["bytes"] <= self.max_bytes:
            return
        LOG.info("cache.prune.start", extra={"bytes": st["bytes"], "limit": self.max_bytes})

        # Collect (meta_path, last_access, size, bin_path)
        entries = []
        for meta_path in self.cache_dir.rglob("*.json"):
            try:
                meta = _read_json(meta_path)
                la = int(meta.get("last_access") or meta.get("fetched_at") or 0)
                bin_path = meta_path.with_suffix(".bin")
                size = bin_path.stat().st_size if bin_path.exists() else 0
                entries.append((meta_path, la, size, bin_path))
            except Exception:
                continue

        # Sort by last_access ascending (oldest first)
        entries.sort(key=lambda x: x[1])
        bytes_now = st["bytes"]
        deleted = 0

        for meta_path, _, size, bin_path in entries:
            if bytes_now <= self.max_bytes:
                break
            try:
                with self._entry_lock(bin_path):
                    bin_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                bytes_now -= size
                deleted += 1
            except Exception:
                pass

        LOG.info("cache.prune.done", extra={"deleted": deleted, "bytes": bytes_now})

    # ---------- convenience ----------

    def get_json(self, url: str, **kw) -> Tuple[Dict[str, Any], CacheResponse]:
        r = self.get(url, **kw)
        try:
            return json.loads(r.text()), r
        except Exception as e:
            raise RuntimeError(f"JSON decode failed for {url}: {e}")

    def get_text(self, url: str, **kw) -> Tuple[str, CacheResponse]:
        r = self.get(url, **kw)
        return r.text(), r

    def get_bytes(self, url: str, **kw) -> Tuple[bytes, CacheResponse]:
        r = self.get(url, **kw)
        return r.content, r


# ----------------------------- CLI self-check ----------------------------------

if __name__ == "__main__":
    """
    Quick manual test (requires internet):
      python storage/http_cache.py https://httpbin.org/etag/abc
    """
    import argparse
    ap = argparse.ArgumentParser(description="CiteVizor HTTP cache self-check")
    ap.add_argument("url", help="URL to GET")
    ap.add_argument("--ttl", type=int, default=60, help="TTL seconds")
    args = ap.parse_args()

    setup_logging()
    cache = HttpCache()
    # first hit (store)
    with span(LOG, "first_get", url=args.url):
        r1 = cache.get(args.url, ttl_s=args.ttl)
        print("1st:", r1.status_code, "cache?", r1.from_cache, "len:", len(r1.content))
    # second hit (should be cached or 304)
    with span(LOG, "second_get", url=args.url):
        r2 = cache.get(args.url, ttl_s=args.ttl)
        print("2nd:", r2.status_code, "cache?", r2.from_cache, "len:", len(r2.content))
    print("STAT:", cache.stat())
