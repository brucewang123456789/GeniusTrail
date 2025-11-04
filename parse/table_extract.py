# C:\CiteVizor\parse\table_extract.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Table extraction (PDF -> CSV/JSON)
Priority: Tabula(JSON) -> pdfplumber fallback.
- Produces Evidence(type=table, csv_path, page, table_no?) and appends to EvidencePack.tables
- Writes per-table CSV under storage/evidence/<doc_id>/tables/table_{page}_{idx}.csv (utf-8-sig)
- Idempotent: re-runs won't duplicate tables (based on stable ids); safe on partial failures.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional deps
try:
    import tabula  # requires Java
except Exception:
    tabula = None  # type: ignore

try:
    import pdfplumber
except Exception:
    pdfplumber = None  # type: ignore

# Project schemas / paths
from schemas import (
    Evidence,
    EvidencePack,
    EvidenceType,
    FetchedDoc,
    EVIDENCE_DIR,
    DOCS_DIR,
    ensure_dirs,
)

ENV_FILE = "citevizor.env"
PACK_FILENAME = "pack.json"


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
    for k in ("TABULA_ENABLED", "TABULA_PAGES", "TABLE_MIN_ROWS", "TABLE_MIN_COLS", "MAX_TABLES_PER_PDF"):
        if os.getenv(k):
            out[k] = os.getenv(k)  # type: ignore
    return out


# ----------------------------- Utilities --------------------------------------

def _tables_dir(doc_id: str) -> Path:
    d = EVIDENCE_DIR / doc_id / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _pack_path(doc_id: str) -> Path:
    d = EVIDENCE_DIR / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d / PACK_FILENAME

def _stable_table_id(doc_id: str, page: int, idx: int, shape: Tuple[int, int], sample: str) -> str:
    h = sha256(f"{doc_id}|{page}|{idx}|{shape[0]}x{shape[1]}|{sample[:96]}".encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}_tb_{page:03d}_{idx:03d}_{h}"

def _clean_cell(s: Any) -> str:
    if s is None:
        return ""
    # join multi-line to single line, trim spaces
    return re.sub(r"\s+", " ", str(s)).strip()

def _prune_empty_rows_cols(grid: List[List[str]]) -> List[List[str]]:
    if not grid:
        return grid
    # Drop fully-empty rows
    rows = [row for row in grid if any(c.strip() for c in row)]
    if not rows:
        return []
    # Normalize width
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    # Drop fully-empty columns
    keep_idx = [j for j in range(width) if any(r[j].strip() for r in rows)]
    pruned = [[r[j] for j in keep_idx] for r in rows]
    return pruned

def _has_min_shape(grid: List[List[str]], min_rows: int, min_cols: int) -> bool:
    if not grid:
        return False
    r = len(grid)
    c = max(len(row) for row in grid) if grid else 0
    return r >= min_rows and c >= min_cols

def _write_csv(path: Path, grid: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for row in grid:
            w.writerow(row)


# ----------------------------- Extractors -------------------------------------

@dataclass
class TableExtractConfig:
    project_root: Path
    tabula_enabled: bool = True
    tabula_pages: str = "all"   # "all" or "1-5" etc.
    min_rows: int = 2
    min_cols: int = 2
    max_tables_per_pdf: int = 64

    @classmethod
    def from_env(cls, project_root: Path) -> "TableExtractConfig":
        env = _load_env(project_root)
        return cls(
            project_root=project_root,
            tabula_enabled=(env.get("TABULA_ENABLED", "1") not in ("0", "false", "False")),
            tabula_pages=env.get("TABULA_PAGES", "all"),
            min_rows=int(env.get("TABLE_MIN_ROWS", "2")),
            min_cols=int(env.get("TABLE_MIN_COLS", "2")),
            max_tables_per_pdf=int(env.get("MAX_TABLES_PER_PDF", "64")),
        )


class TableExtractor:
    def __init__(self, cfg: TableExtractConfig):
        self.cfg = cfg
        ensure_dirs()

    # Public API
    def extract(self, fetched: FetchedDoc, persist: bool = True) -> EvidencePack:
        """
        Extract tables from fetched PDF; merge into existing EvidencePack (pack.json).
        """
        pdf_path = fetched.paths.pdf_path
        if not pdf_path or not pdf_path.exists():
            raise FileNotFoundError("PDF not found; run downloader first.")

        doc_id = fetched.doc_id
        tables_dir = _tables_dir(doc_id)

        found: List[Evidence] = []
        # Prefer Tabula JSON if enabled and present
        if self.cfg.tabula_enabled and tabula is not None:
            try:
                found = self._extract_with_tabula(pdf_path, doc_id, tables_dir)
            except Exception:
                # Silent fallback
                found = []

        # Fallback to pdfplumber
        if not found and pdfplumber is not None:
            try:
                found = self._extract_with_pdfplumber(pdf_path, doc_id, tables_dir)
            except Exception:
                found = []

        if not found:
            # Return pack untouched or created empty
            pack = self._read_or_init_pack(doc_id)
            if persist:
                self._write_pack(doc_id, pack)
            return pack

        # Merge into pack (idempotent by id)
        pack = self._read_or_init_pack(doc_id)
        existing_ids = {e["id"] if isinstance(e, dict) else e.id for e in pack.get("tables", [])}
        for ev in found:
            if ev.id not in existing_ids:
                pack.setdefault("tables", []).append(ev.__dict__)
        if persist:
            self._write_pack(doc_id, pack)

        # Rehydrate EvidencePack object to return
        return EvidencePack(
            doc_id=doc_id,
            paragraphs=[Evidence(**e) for e in pack.get("paragraphs", [])],
            tables=[Evidence(**e) for e in pack.get("tables", [])],
            figures=[Evidence(**e) for e in pack.get("figures", [])],
        )

    # --------------------- Tabula path (JSON) ---------------------

    def _extract_with_tabula(self, pdf_path: Path, doc_id: str, tables_dir: Path) -> List[Evidence]:
        """
        Use tabula.convert_into(output_format='json') to get page-aware cell grid,
        then normalize & save CSV per table.
        """
        tmp_json = tables_dir / "_tabula.json"
        # Execute Tabula to JSON file
        # Lattice+stream combined approach: run twice and merge best—here we run 'stream' first;
        # if you want lattice, set guess=False + lattice=True; many PDFs work fine with default guess=True.
        tabula.convert_into(
            input_path=str(pdf_path),
            output_path=str(tmp_json),
            output_format="json",
            pages=self.cfg.tabula_pages,
            guess=True,
        )
        if not tmp_json.exists() or tmp_json.stat().st_size == 0:
            return []

        data = json.loads(tmp_json.read_text(encoding="utf-8"))
        evidences: List[Evidence] = []
        per_pdf_count = 0

        # Tabula JSON is a list of tables; each has "page" and "data" (rows -> columns -> {"text":...})
        for idx, obj in enumerate(data, 1):
            if per_pdf_count >= self.cfg.max_tables_per_pdf:
                break
            page = int(obj.get("page") or 0)
            raw_rows = obj.get("data") or []
            grid: List[List[str]] = []
            for row in raw_rows:
                if not isinstance(row, list):
                    continue
                cells = [_clean_cell(cell.get("text") if isinstance(cell, dict) else cell) for cell in row]
                grid.append(cells)

            grid = _prune_empty_rows_cols(grid)
            if not _has_min_shape(grid, self.cfg.min_rows, self.cfg.min_cols):
                continue

            # Build evidence id and CSV path
            sample = " ".join((grid[0] if grid else [])[:5])
            ev_id = _stable_table_id(doc_id, page=page or 0, idx=idx, shape=(len(grid), len(grid[0]) if grid else 0), sample=sample)
            csv_path = tables_dir / f"table_{page:03d}_{idx:03d}.csv"
            _write_csv(csv_path, grid)

            ev = Evidence(
                id=ev_id,
                type=EvidenceType.table,
                doc_id=doc_id,
                page=page or None,
                table_no=None,               # table number mapping handled in citation_map later (optional)
                csv_path=csv_path,
                doi=None,
                source_url=None,
            )
            evidences.append(ev)
            per_pdf_count += 1

        # Clean up temp JSON (optional keep for debug)
        try:
            tmp_json.unlink(missing_ok=True)
        except Exception:
            pass
        return evidences

    # --------------------- pdfplumber fallback ---------------------

    def _extract_with_pdfplumber(self, pdf_path: Path, doc_id: str, tables_dir: Path) -> List[Evidence]:
        """
        Best-effort extraction when Tabula/Java is unavailable.
        """
        if pdfplumber is None:
            return []
        evidences: List[Evidence] = []
        per_pdf_count = 0
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pidx, page in enumerate(pdf.pages, 1):
                # Heuristics for table extraction; tweak if needed
                try:
                    tables = page.extract_tables(table_settings={
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5,
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                        "edge_min_length": 3,
                        "min_words_vertical": 1,
                        "min_words_horizontal": 1,
                    })
                    # If nothing with 'lines', try 'text' strategy
                    if not tables:
                        tables = page.extract_tables(table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "intersection_tolerance": 5,
                        })
                except Exception:
                    tables = []

                if not tables:
                    continue

                for tidx, tbl in enumerate(tables, 1):
                    if per_pdf_count >= self.cfg.max_tables_per_pdf:
                        break
                    # tbl is List[List[str|None]]
                    grid = [[_clean_cell(c) for c in (row or [])] for row in tbl if isinstance(tbl, list)]
                    grid = _prune_empty_rows_cols(grid)
                    if not _has_min_shape(grid, self.cfg.min_rows, self.cfg.min_cols):
                        continue

                    sample = " ".join((grid[0] if grid else [])[:5])
                    ev_id = _stable_table_id(doc_id, page=pidx, idx=tidx, shape=(len(grid), len(grid[0]) if grid else 0), sample=sample)
                    csv_path = tables_dir / f"table_{pidx:03d}_{tidx:03d}.csv"
                    _write_csv(csv_path, grid)

                    ev = Evidence(
                        id=ev_id,
                        type=EvidenceType.table,
                        doc_id=doc_id,
                        page=pidx,
                        table_no=None,
                        csv_path=csv_path,
                        doi=None,
                        source_url=None,
                    )
                    evidences.append(ev)
                    per_pdf_count += 1

        return evidences

    # --------------------- Pack IO ---------------------

    def _read_or_init_pack(self, doc_id: str) -> Dict[str, Any]:
        p = _pack_path(doc_id)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        # Initialize empty pack
        return {"doc_id": doc_id, "paragraphs": [], "tables": [], "figures": []}

    def _write_pack(self, doc_id: str, pack: Dict[str, Any]) -> Path:
        p = _pack_path(doc_id)
        p.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
        return p


# ----------------------------- CLI self-test ----------------------------------

if __name__ == "__main__":
    """
    Smoke test:
      1) Ensure storage/docs/<doc_id>/<doc_id>.pdf exists (use downloader first).
      2) Optionally set TABULA_ENABLED=1 and have Java installed for better results.
      3) Run: python parse/table_extract.py <doc_id>
    """
    ensure_dirs()
    project_root = Path(__file__).resolve().parents[1]
    cfg = TableExtractConfig.from_env(project_root)
    extractor = TableExtractor(cfg)

    if len(sys.argv) < 2:
        print("Usage: python parse/table_extract.py <doc_id>")
        sys.exit(1)

    doc_id = sys.argv[1].strip()
    pdf_path = DOCS_DIR / doc_id / f"{doc_id}.pdf"
    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}")
        sys.exit(2)

    # Minimal FetchedDoc reconstruction (from meta if present)
    meta_path = DOCS_DIR / doc_id / f"{doc_id}.meta.json"
    title = doc_id
    url = "https://example.org"
    doi = None
    year = None
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            title = m.get("title") or title
            url = m.get("url") or url
            doi = m.get("doi")
            year = m.get("year")
        except Exception:
            pass

    from schemas import Candidate, DocumentPaths

    cand = Candidate(
        title=title,
        url=url,  # type: ignore
        year=year,
        source="scholar",
        score=1.0,
        doi=doi,
    )
    paths = DocumentPaths(base_dir=pdf_path.parent, pdf_path=pdf_path, meta_json=meta_path)
    fetched = FetchedDoc(doc_id=doc_id, candidate=cand, paths=paths)

    pack = extractor.extract(fetched, persist=True)
    out_path = _pack_path(doc_id)
    print(f"[OK] Tables extracted -> {out_path}")
    print(f"  tables={len(pack.tables)} paragraphs={len(pack.paragraphs)} figures={len(pack.figures)}")
