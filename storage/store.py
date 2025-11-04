# C:\CiteVizor\storage\store.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Storage helpers (atomic writes, simple file locks, path conventions)

Goals
  - Provide a single source of truth for storage paths and atomic file I/O.
  - Avoid partial writes on crashes; ensure deterministic artifact names.
  - Cross-platform, no third-party deps; work well on Windows (RTX 4090 dev box).

This module does NOT use any LLM and introduces NO new .env keys.

Conventions (aligned with `schemas.py`)
  STORAGE_DIR  = C:\CiteVizor\storage
   ├─ docs/<doc_id>/
   │    ├─ <doc_id>.pdf
   │    └─ <doc_id>.meta.json
   ├─ evidence/<doc_id>/
   │    ├─ pack.json
   │    ├─ figures/*.png
   │    └─ tables/*.csv
   ├─ evidence/_rank/
   │    ├─ <run_id>.citations.json
   │    ├─ <run_id>.summary.json
   │    ├─ <run_id>.biblio.json
   │    ├─ <run_id>.trace.json
   │    └─ <run_id>.dedup.json
   ├─ renders/<doc_id>/charts/
   │    ├─ *.png
   │    └─ scripts/*.py
   └─ reports/
        ├─ <run_id>.html
        └─ <run_id>.pptx
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Import canonical directories from project contracts
from schemas import STORAGE_DIR, DOCS_DIR, EVIDENCE_DIR, RENDERS_DIR, REPORTS_DIR


# ----------------------------- path helpers ------------------------------------

def ensure_dir(p: Path) -> Path:
    """Create directory (parents ok); return the path for chaining."""
    p.mkdir(parents=True, exist_ok=True)
    return p

def rank_dir() -> Path:
    """Return storage/evidence/_rank dir, ensuring it exists."""
    return ensure_dir(EVIDENCE_DIR / "_rank")

def doc_dir(doc_id: str) -> Path:
    return ensure_dir(DOCS_DIR / doc_id)

def doc_pdf_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / f"{doc_id}.pdf"

def doc_meta_path(doc_id: str) -> Path:
    return doc_dir(doc_id) / f"{doc_id}.meta.json"

def evidence_pack_path(doc_id: str) -> Path:
    return ensure_dir(EVIDENCE_DIR / doc_id) / "pack.json"

def evidence_figures_dir(doc_id: str) -> Path:
    return ensure_dir(EVIDENCE_DIR / doc_id / "figures")

def evidence_tables_dir(doc_id: str) -> Path:
    return ensure_dir(EVIDENCE_DIR / doc_id / "tables")

def charts_dir(doc_id: str) -> Path:
    d = ensure_dir(RENTERS_DIR := (RENDERS_DIR / doc_id / "charts"))
    ensure_dir(RENTERS_DIR / "scripts")
    return RENTERs_DIR if (RENTERS_DIR := d) else d  # keep local var for IDEs

def charts_manifest_path(doc_id: str) -> Path:
    return charts_dir(doc_id) / "_charts.json"

def report_html_path(run_id: str) -> Path:
    return ensure_dir(REPORTS_DIR) / f"{run_id}.html"

def report_pptx_path(run_id: str) -> Path:
    return ensure_dir(REPORTS_DIR) / f"{run_id}.pptx"

def run_artifact_path(run_id: str, suffix: str) -> Path:
    """
    Return storage/evidence/_rank/<run_id>.<suffix>.json, e.g. suffix='citations'|'summary'|'biblio'|'trace'|'dedup'.
    """
    return rank_dir() / f"{run_id}.{suffix}.json"


# ----------------------------- simple file lock --------------------------------

class FileLock:
    """
    A simple lock using an adjacent `<target>.lock` file; works without extra deps.
    - Acquire by creating the lock file with O_EXCL; write pid + ts for debugging.
    - If lock exists and is stale (older than `stale_seconds`), attempt to break it.
    - This is cooperative and sufficient for our single-host pipeline.
    """

    def __init__(self, lock_path: Path, timeout_s: float = 30.0, poll_s: float = 0.1, stale_seconds: float = 900.0):
        self.lock_path = Path(lock_path)
        self.timeout_s = max(0.0, timeout_s)
        self.poll_s = max(0.01, poll_s)
        self.stale_seconds = max(60.0, stale_seconds)
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        deadline = time.time() + self.timeout_s
        while True:
            try:
                # O_CREAT|O_EXCL ensures fail if exists; O_RDWR to keep a handle
                self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                payload = f"pid={os.getpid()} ts={int(time.time())}\n"
                os.write(self._fd, payload.encode("utf-8"))
                return
            except FileExistsError:
                # Check staleness
                try:
                    st = self.lock_path.stat()
                    if (time.time() - st.st_mtime) > self.stale_seconds:
                        # stale; try to break
                        try:
                            os.remove(self.lock_path)
                            continue
                        except Exception:
                            pass
                except FileNotFoundError:
                    continue
                if time.time() >= deadline:
                    raise TimeoutError(f"Failed to acquire lock: {self.lock_path}")
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
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def lock_for(target_path: Path) -> FileLock:
    """Create a FileLock for `<target>.lock`."""
    return FileLock(Path(f"{target_path}.lock"))


# ----------------------------- atomic writes -----------------------------------

def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    """
    Write bytes atomically:
      - Create a temp file in the same directory
      - Flush+fsync
      - Replace destination using os.replace (atomic on Windows/Unix)
    """
    dest = Path(dest)
    ensure_dir(dest.parent)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".tmp-{dest.name}.", dir=str(dest.parent))
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dest)  # atomic
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def atomic_write_text(dest: Path, text: str, encoding: str = "utf-8") -> None:
    _atomic_write_bytes(dest, text.encode(encoding))

def atomic_write_json(dest: Path, obj: Any, ensure_ascii: bool = False, indent: int = 2) -> None:
    payload = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent)
    atomic_write_text(dest, payload, encoding="utf-8")

def atomic_copy(src: Path, dest: Path) -> None:
    """
    Copy file to `dest` atomically by copying to temp and replacing.
    """
    src = Path(src); dest = Path(dest)
    if not src.exists():
        raise FileNotFoundError(src)
    ensure_dir(dest.parent)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".cp-{dest.name}.", dir=str(dest.parent))
    try:
        with os.fdopen(tmp_fd, "wb") as w, open(src, "rb") as r:
            shutil.copyfileobj(r, w, length=1024 * 1024)
            w.flush(); os.fsync(w.fileno())
        os.replace(tmp_path, dest)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# ----------------------------- JSON read helpers -------------------------------

def read_json(path: Path) -> Any:
    """Read JSON; return {} or [] on errors."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return {}

def write_json_locked(path: Path, obj: Any, timeout_s: float = 30.0) -> None:
    """Write JSON with an adjacent file lock to serialize writers."""
    l = lock_for(path)
    with l:
        atomic_write_json(path, obj)


# ----------------------------- small utilities ---------------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    if not Path(path).exists():
        return None
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_filename(name: str, limit: int = 128) -> str:
    s = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in (name or ""))
    return s[:limit] if len(s) > limit else s

def rotate_keep_latest(dir_path: Path, pattern: str, keep: int = 10) -> List[Path]:
    """
    Keep only the latest `keep` files matching the pattern (sorted by mtime desc).
    Return the list of deleted paths.
    """
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    to_delete = files[keep:]
    for p in to_delete:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    return to_delete


# ----------------------------- high-level APIs ---------------------------------

# Document bundle APIs

@dataclass
class DocPaths:
    base: Path
    pdf: Path
    meta: Path
    pack: Path
    figures_dir: Path
    tables_dir: Path
    charts_dir: Path
    charts_manifest: Path

def get_doc_paths(doc_id: str) -> DocPaths:
    base = doc_dir(doc_id)
    pd = base / f"{doc_id}.pdf"
    meta = base / f"{doc_id}.meta.json"
    pack = evidence_pack_path(doc_id)
    figs = evidence_figures_dir(doc_id)
    tabs = evidence_tables_dir(doc_id)
    chd = charts_dir(doc_id)
    chm = chd / "_charts.json"
    return DocPaths(base=base, pdf=pd, meta=meta, pack=pack,
                    figures_dir=figs, tables_dir=tabs, charts_dir=chd, charts_manifest=chm)

def save_doc_pdf(doc_id: str, src_pdf: Path) -> Path:
    """Copy a downloaded PDF into canonical location atomically."""
    dest = doc_pdf_path(doc_id)
    atomic_copy(src_pdf, dest)
    return dest

def save_doc_meta(doc_id: str, meta: Dict[str, Any]) -> Path:
    p = doc_meta_path(doc_id)
    write_json_locked(p, meta)
    return p

def save_pack(doc_id: str, pack: Dict[str, Any]) -> Path:
    p = evidence_pack_path(doc_id)
    write_json_locked(p, pack)
    return p

def append_charts_manifest(doc_id: str, records: List[Dict[str, Any]]) -> Path:
    """Append chart records to `_charts.json` (read-modify-write under lock)."""
    p = charts_manifest_path(doc_id)
    l = lock_for(p)
    with l:
        existing = read_json(p)
        if not isinstance(existing, list):
            existing = []
        existing.extend(records or [])
        atomic_write_json(p, existing)
    return p


# Run-level artifact APIs

def read_run_artifact(run_id: str, suffix: str) -> Any:
    """Read storage/evidence/_rank/<run_id>.<suffix>.json safely."""
    return read_json(run_artifact_path(run_id, suffix))

def write_run_artifact(run_id: str, suffix: str, obj: Any) -> Path:
    """Write storage/evidence/_rank/<run_id>.<suffix>.json atomically."""
    p = run_artifact_path(run_id, suffix)
    write_json_locked(p, obj)
    return p

def list_run_artifacts(run_id: str) -> Dict[str, Path]:
    """Return existing known artifacts for a run."""
    d = rank_dir()
    known = {}
    for sfx in ("citations", "summary", "biblio", "trace", "dedup", "qa"):
        p = d / f"{run_id}.{sfx}.json"
        if p.exists():
            known[sfx] = p
    # reports (html/pptx)
    html = report_html_path(run_id)
    pptx = report_pptx_path(run_id)
    if html.exists():
        known["html"] = html
    if pptx.exists():
        known["pptx"] = pptx
    return known


# ----------------------------- example CLI (optional) --------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CiteVizor storage helper CLI")
    ap.add_argument("--stat", metavar="RUN_ID", help="Print known artifacts for a run_id")
    ap.add_argument("--sha", metavar="FILE", help="Compute sha256 of a file")
    args = ap.parse_args()

    if args.stat:
        m = list_run_artifacts(args.stat)
        print("Known artifacts:")
        for k, p in m.items():
            try:
                print(f" - {k:10s} {p}  ({p.stat().st_size} bytes)")
            except Exception:
                print(f" - {k:10s} {p}")
    if args.sha:
        print(sha256_file(Path(args.sha)))
