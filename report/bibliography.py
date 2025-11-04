# C:\CiteVizor\report\bibliography.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Bibliography formatter

Purpose
  - Read citations for a run (storage/evidence/_rank/<run_id>.citations.json).
  - Produce human-readable reference strings in multiple styles (IEEE/APA/ACM/plain).
  - Optionally export a BibTeX file with best-effort entry typing.
  - Persist a machine-readable biblio json for downstream (HTML/PPTX) renderers.

No LLM calls. No new environment variables required.

Expected input shape (from citation_map/evidence_rank):
  {
    "run_id": "...",
    "query": "...",
    "refs": [
      {
        "ref_id": 1,                # integer label used in [1], [2], ...
        "doc_id": "doi_abcd1234",   # stable doc id
        "title": "Article title",
        "year": 2024,
        "venue": "Journal/Conference",
        "publisher": "Publisher",   # optional
        "authors": ["Alice Smith", "Bob Zhang"],  # optional
        "doi": "10.xxxx/yyy",       # optional
        "url": "https://..."        # optional
      },
      ...
    ]
  }
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas import EVIDENCE_DIR  # for _rank dir


# ----------------------------- FS helpers -------------------------------------

def _rank_dir() -> Path:
    d = EVIDENCE_DIR / "_rank"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------- author utilities --------------------------------

_INITIAL_RE = re.compile(r"(^|[\s\-])([A-Za-z])[^-\s]*")

def _normalize_name(name: str) -> Tuple[str, str]:
    """
    Split "Given ... Family" into (given, family) conservatively; fall back gracefully.
    Heuristics:
      - If a comma exists: "Family, Given" -> (Given, Family)
      - Else: last token as family, the rest as given
    """
    s = (name or "").strip()
    if not s:
        return "", ""
    if "," in s:
        family, given = [t.strip() for t in s.split(",", 1)]
        return given, family
    parts = s.split()
    if len(parts) == 1:
        return "", parts[0]
    return " ".join(parts[:-1]), parts[-1]

def _ieee_author_list(authors: List[str]) -> str:
    """
    IEEE-like short form: "A. B. Family, C. D. Family, and E. F. Family"
    """
    short = []
    for a in (authors or []):
        g, f = _normalize_name(a)
        # initials for given
        inits = " ".join([p[0].upper() + "." for p in g.split() if p])
        if f:
            short.append(f"{inits} {f}".strip())
        else:
            short.append(inits or a)
    if not short:
        return ""
    if len(short) == 1:
        return short[0]
    return ", ".join(short[:-1]) + ", and " + short[-1]

def _apa_author_list(authors: List[str]) -> str:
    """
    APA-like: "Family, A. B., Family, C. D., & Family, E. F."
    """
    parts = []
    for a in (authors or []):
        g, f = _normalize_name(a)
        inits = "".join([p[0].upper() + "." for p in g.split() if p])
        if f:
            parts.append(f"{f}, {inits}".strip())
        else:
            parts.append(a)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", & " + parts[-1]

def _acm_author_list(authors: List[str]) -> str:
    """
    ACM-like: "Family, Given; Family, Given; and Family, Given"
    """
    parts = []
    for a in (authors or []):
        g, f = _normalize_name(a)
        if f:
            parts.append(f"{f}, {g}".strip())
        else:
            parts.append(a)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "; ".join(parts[:-1]) + "; and " + parts[-1]


# ----------------------------- style formatters --------------------------------

def _fmt_ieee(ref: Dict[str, Any], linkify: bool = True) -> str:
    au = _ieee_author_list(ref.get("authors") or [])
    title = ref.get("title") or "<untitled>"
    venue = ref.get("venue") or ""
    year = ref.get("year")
    doi = ref.get("doi")
    url = ref.get("url")

    segs = []
    if au:
        segs.append(au + ",")
    segs.append(f"“{title}”,")
    if venue:
        segs.append(venue + ",")
    if year:
        segs.append(str(year) + ".")
    if doi:
        segs.append(f"doi:{doi}")
    elif url:
        segs.append(url if not linkify else url)
    return " ".join(segs).replace(" ,", ",")

def _fmt_apa(ref: Dict[str, Any], linkify: bool = True) -> str:
    au = _apa_author_list(ref.get("authors") or [])
    title = ref.get("title") or "<untitled>"
    venue = ref.get("venue") or ""
    year = ref.get("year")
    doi = ref.get("doi")
    url = ref.get("url")

    segs = []
    if au:
        segs.append(au)
    if year:
        segs.append(f"({year}).")
    segs.append(f"{title}.")
    if venue:
        segs.append(venue + ".")
    if doi:
        segs.append(f"https://doi.org/{doi}")
    elif url:
        segs.append(url if not linkify else url)
    return " ".join(segs)

def _fmt_acm(ref: Dict[str, Any], linkify: bool = True) -> str:
    au = _acm_author_list(ref.get("authors") or [])
    title = ref.get("title") or "<untitled>"
    venue = ref.get("venue") or ""
    year = ref.get("year")
    doi = ref.get("doi")
    url = ref.get("url")

    segs = []
    if au:
        segs.append(au + ".")
    segs.append(f"{title}.")
    if venue:
        segs.append(venue + ".")
    if year:
        segs.append(str(year) + ".")
    if doi:
        segs.append(f"DOI:{doi}")
    elif url:
        segs.append(url if not linkify else url)
    return " ".join(segs)

def _fmt_plain(ref: Dict[str, Any], linkify: bool = True) -> str:
    title = ref.get("title") or "<untitled>"
    segs = [title]
    if ref.get("year"):
        segs.append(str(ref["year"]))
    if ref.get("venue"):
        segs.append(f"({ref['venue']})")
    if ref.get("doi"):
        segs.append(f"doi:{ref['doi']}")
    elif ref.get("url"):
        segs.append(ref["url"] if not linkify else ref["url"])
    return " ".join(segs)


# ----------------------------- BibTeX export -----------------------------------

def _bibtex_key(ref: Dict[str, Any]) -> str:
    """
    Stable key: family name of first author + year + short hash of title.
    """
    au = ref.get("authors") or []
    given, family = _normalize_name(au[0]) if au else ("", "ref")
    y = str(ref.get("year") or "na")
    import hashlib
    h = hashlib.blake2b((ref.get("title") or "").encode("utf-8"), digest_size=3).hexdigest()
    base = f"{(family or 'ref').lower()}{y}{h}"
    # sanitize
    return re.sub(r"[^A-Za-z0-9]+", "", base)

def _bibtex_entry_type(ref: Dict[str, Any]) -> str:
    """Best-effort guess."""
    v = (ref.get("venue") or "").lower()
    if "proc" in v or "conference" in v or "symposium" in v or "conf." in v:
        return "inproceedings"
    if "journal" in v or "transactions" in v or "nature" in v or "science" in v:
        return "article"
    return "misc"

def _bibtex_escape(s: str) -> str:
    return s.replace("{", "\\{").replace("}", "\\}")

def _to_bibtex(ref: Dict[str, Any]) -> str:
    key = _bibtex_key(ref)
    typ = _bibtex_entry_type(ref)
    fields = []
    # Authors (keep original string; BibTeX expects "and"-joined)
    au = ref.get("authors") or []
    if au:
        fields.append(f"  author = {{{' and '.join(_bibtex_escape(a) for a in au)}}}")
    if ref.get("title"):
        fields.append(f"  title = {{{_bibtex_escape(ref['title'])}}}")
    if ref.get("venue"):
        if typ == "article":
            fields.append(f"  journal = {{{_bibtex_escape(ref['venue'])}}}")
        else:
            fields.append(f"  booktitle = {{{_bibtex_escape(ref['venue'])}}}")
    if ref.get("year"):
        fields.append(f"  year = {{{ref['year']}}}")
    if ref.get("doi"):
        fields.append(f"  doi = {{{ref['doi']}}}")
    if ref.get("url"):
        fields.append(f"  url = {{{_bibtex_escape(ref['url'])}}}")
    if ref.get("publisher"):
        fields.append(f"  publisher = {{{_bibtex_escape(ref['publisher'])}}}")
    inner = ",\n".join(fields)
    return f"@{typ}{{{key},\n{inner}\n}}"

def export_bibtex(run_id: str, refs: List[Dict[str, Any]]) -> Path:
    out = _rank_dir() / f"{run_id}.bib"
    entries = [_to_bibtex(r) for r in refs]
    out.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    return out


# ----------------------------- public API --------------------------------------

@dataclass
class BiblioItem:
    ref_id: int
    text: str          # formatted reference
    style: str
    doc_id: str
    doi: Optional[str]
    url: Optional[str]

def build_bibliography(
    run_id: str,
    style: str = "ieee",
    linkify: bool = True,
    write_files: bool = True,
    also_bibtex: bool = False,
) -> Tuple[List[BiblioItem], Optional[Path], Optional[Path]]:
    """
    Build formatted references for a run; optionally persist JSON and BibTeX.

    Returns:
        (items, json_path, bib_path)
    """
    cit = _load_json(_rank_dir() / f"{run_id}.citations.json")
    if not cit or not isinstance(cit.get("refs"), list):
        raise FileNotFoundError(f"citations json not found or invalid for run_id={run_id}")

    # Sort by ref_id to keep labels stable
    refs = sorted(cit["refs"], key=lambda r: int(r.get("ref_id", 0)))

    fmt = {
        "ieee": _fmt_ieee,
        "apa": _fmt_apa,
        "acm": _fmt_acm,
        "plain": _fmt_plain,
    }.get(style.lower())
    if fmt is None:
        raise ValueError(f"Unknown bibliography style: {style}")

    items: List[BiblioItem] = []
    for r in refs:
        text = fmt(r, linkify=linkify)
        items.append(BiblioItem(
            ref_id=int(r.get("ref_id", 0)),
            text=text,
            style=style.lower(),
            doc_id=r.get("doc_id") or "",
            doi=r.get("doi"),
            url=r.get("url"),
        ))

    json_path: Optional[Path] = None
    bib_path: Optional[Path] = None
    if write_files:
        payload = {
            "run_id": run_id,
            "style": style.lower(),
            "count": len(items),
            "items": [asdict(x) for x in items],
        }
        json_path = _rank_dir() / f"{run_id}.biblio.json"
        _save_json(json_path, payload)
        if also_bibtex:
            bib_path = export_bibtex(run_id, refs)

    return items, json_path, bib_path


# ----------------------------- CLI ---------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor bibliography generator")
    ap.add_argument("--run-id", required=True, help="Ranking run_id")
    ap.add_argument("--style", default="ieee", choices=["ieee", "apa", "acm", "plain"], help="Citation style")
    ap.add_argument("--no-link", action="store_true", help="Do not include URL/DOI hyperlinks")
    ap.add_argument("--no-write", action="store_true", help="Do not write biblio.json")
    ap.add_argument("--bibtex", action="store_true", help="Also export a .bib file")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    items, jpath, bpath = build_bibliography(
        run_id=args.run_id,
        style=args.style,
        linkify=(not args.no_link),
        write_files=(not args.no_write),
        also_bibtex=args.bibtex,
    )
    print(f"[OK] Bibliography items: {len(items)}")
    if jpath:
        print(f"[OK] biblio json -> {jpath}")
    if bpath:
        print(f"[OK] bibtex -> {bpath}")
