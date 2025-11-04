# qa/consistency_checks.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Consistency Checks (soft validation, non-blocking)

Purpose:
- Prevent "warning chains" by validating only ACTIVE evidence.
- ACTIVE doc := (pack.json exists with any paragraphs/tables/figures) OR (charts PNG exist) OR (figure images exist on disk).
- Never raise; always return a compact dict with 'issues' list and 'stats'.

Outputs:
  {
    "run_id": ...,
    "issues": [ {severity, code, message, ...}, ... ],
    "stats": { docs_total, docs_active, issues_warn, issues_info, ... }
  }
"""

from __future__ import annotations
import json, csv, re, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas import EVIDENCE_DIR, RENDERS_DIR, REPORTS_DIR

# ----------------------------- helpers -----------------------------------------

def _rank_dir() -> Path:
    d = EVIDENCE_DIR / "_rank"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_json(p: Path) -> Any:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _sha256_path(p: Path) -> Optional[str]:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None

def _pack_path(doc_id: str) -> Path:
    return EVIDENCE_DIR / doc_id / "pack.json"

def _charts_manifest(doc_id: str) -> Path:
    return RENDERS_DIR / doc_id / "charts" / "_charts.json"

def _list_doc_ids_from_citations(run_id: str) -> List[str]:
    ci = _load_json(_rank_dir() / f"{run_id}.citations.json")
    refs = ci.get("refs", []) if isinstance(ci, dict) else []
    seen, out = set(), []
    for r in refs:
        d = r.get("doc_id") if isinstance(r, dict) else None
        if d and d not in seen:
            seen.add(d); out.append(d)
    return out

# ----------------------------- web image support -------------------------------

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff"}

def _list_web_images_for_doc(doc_id: str) -> List[Path]:
    """
    Discover direct web images saved under evidence/<doc_id>.
    Exclude extracted figures and generated charts to avoid duplicates.
    """
    root = EVIDENCE_DIR / doc_id
    if not root.exists():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            parts = {x.name.lower() for x in p.parents if x != root}
            if "figures" in parts or "charts" in parts:
                continue
            out.append(p)
    # stable ordering: shallowest first, then name
    out.sort(key=lambda x: (len(x.relative_to(root).parts), x.name.lower()))
    return out

# ----------------------------- "active" detection ------------------------------

def _is_active_doc(doc_id: str) -> Tuple[bool, Dict[str, Any]]:
    """
    ACTIVE if:
      - pack.json exists and has any paragraphs/tables/figures
      - OR charts dir has any PNGs
      - OR any figure image pointed by pack.json exists on disk
      - OR direct web images exist under evidence/<doc_id> (excluding figures/ and charts/)
    """
    meta: Dict[str, Any] = {
        "doc_id": doc_id, "has_pack": False, "counts": {"p":0,"t":0,"f":0},
        "has_charts": False, "chart_pngs": 0, "has_fig_imgs": False,
        "has_web_images": False, "web_images": 0
    }

    # pack presence + counts
    pack_p = _pack_path(doc_id)
    if pack_p.exists():
        meta["has_pack"] = True
        data = _load_json(pack_p) if isinstance(pack_p, Path) else {}
        paras = data.get("paragraphs") or []
        tables = data.get("tables") or []
        figs   = data.get("figures") or []
        meta["counts"] = {"p": len(paras or []), "t": len(tables or []), "f": len(figs or [])}
        # any figure image exists?
        for f in (figs or []):
            if not isinstance(f, dict):
                continue
            img = f.get("image_path") or f.get("img_path")
            if not img:
                continue
            p = Path(img)
            if not p.is_absolute():
                p = (EVIDENCE_DIR / doc_id / str(img))
            if p.exists():
                meta["has_fig_imgs"] = True
                break

    # charts PNG presence
    charts_dir = _charts_manifest(doc_id).parent
    if charts_dir.exists():
        pngs = len(list(charts_dir.rglob("*.png")))
        meta["chart_pngs"] = pngs
        meta["has_charts"] = pngs > 0

    # web images presence (image bypass)
    web_imgs = _list_web_images_for_doc(doc_id)
    if web_imgs:
        meta["has_web_images"] = True
        meta["web_images"] = len(web_imgs)

    counts = meta["counts"]
    active = (
        (meta["has_pack"] and (counts["p"] + counts["t"] + counts["f"] > 0))
        or meta["has_charts"]
        or meta["has_fig_imgs"]
        or meta["has_web_images"]
    )
    return active, meta

# ----------------------------- datamodel ---------------------------------------

@dataclass
class Issue:
    severity: str      # "info" | "warn"
    code: str
    message: str
    doc_id: Optional[str] = None
    ref_idx: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None

def _issue(sev: str, code: str, message: str, **kw) -> Dict[str, Any]:
    """
    Build Issue dict safely:
    - keep only doc_id/ref_idx as first-class fields;
    - merge every other kw into 'extra' to avoid constructor errors.
    """
    doc_id = kw.pop("doc_id", None)
    ref_idx = kw.pop("ref_idx", None)

    # merge provided 'extra' with remaining kwargs (index/text/count/label/...).
    extra = kw.pop("extra", None)
    rest = kw if kw else None
    if isinstance(extra, dict) and rest:
        m = dict(extra); m.update(rest); extra = m
    elif extra is None:
        extra = rest
    else:
        # extra exists but is not a dict; wrap it and merge rest if any
        if rest:
            extra = {"extra": extra, **rest}

    return asdict(Issue(severity=sev, code=code, message=str(message),
                        doc_id=doc_id, ref_idx=ref_idx, extra=extra))

# ----------------------------- checks ------------------------------------------

_LABEL_RE = re.compile(r"\[(\d+)([^\]]*)\]")

def _check_citation_items(ci: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    issues: List[Dict[str, Any]] = []
    refs = ci.get("refs", [])
    items = ci.get("items", [])
    ref_ids = {int(r["ref_id"]) for r in refs if "ref_id" in r}
    doc_to_ref = {r.get("doc_id"): int(r["ref_id"]) for r in refs if r.get("doc_id")}
    for it in items:
        label = (it.get("label") or "").strip()
        m = _LABEL_RE.search(label)
        if not m:
            issues.append(_issue("warn", "item.label_missing", "Ranked item has no valid label.", evidence_id=it.get("evidence_id")))
            continue
        rid = int(m.group(1))
        if rid not in ref_ids:
            issues.append(_issue("warn", "item.label_bad_ref", "Label ref_id not found in refs (soft).", label=label, doc_id=it.get("doc_id")))
        if it.get("doc_id") and it["doc_id"] in doc_to_ref and doc_to_ref[it["doc_id"]] != rid:
            issues.append(_issue("warn", "item.label_doc_mismatch", "Label ref_id does not match this item's doc_id (soft).", label=label, doc_id=it.get("doc_id")))
    return issues, doc_to_ref

def _check_highlights(ci: Dict[str, Any], su: Dict[str, Any], lang: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    refs = {int(r["ref_id"]) for r in ci.get("refs", []) if r.get("ref_id")}
    bullets = su.get("highlights") or []
    if not isinstance(bullets, list) or not bullets:
        return [_issue("warn", "highlights.empty", "Summary highlights missing or empty.")]

    def _looks_cjk(s: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7ff]", s or ""))

    def _is_english_heavy(s: str) -> bool:
        if not s:
            return True
        total = len(s)
        latin = sum(1 for ch in s if "A" <= ch <= "Z" or "a" <= ch <= "z" or ch.isdigit() or ch.isspace())
        return latin / max(1, total) >= 0.6

    seen_norm = set()
    for idx, h in enumerate(bullets, 1):
        t = (h or "").strip()
        if not t:
            issues.append(_issue("warn", "highlight.empty", "Empty highlight.", index=idx))
            continue
        key = re.sub(r"\s+", " ", t.lower())
        if key in seen_norm:
            issues.append(_issue("info", "highlight.dup", "Duplicate highlight.", index=idx, text=t[:120]))
        seen_norm.add(key)
        m = _LABEL_RE.findall(t)
        if not m:
            issues.append(_issue("warn", "highlight.no_label", "Highlight does not contain any citation label.", index=idx, text=t[:160]))
        else:
            ok = False
            for rid_str, _ in m:
                try:
                    if int(rid_str) in refs:
                        ok = True; break
                except Exception:
                    pass
            if not ok:
                issues.append(_issue("warn", "highlight.bad_ref", "Highlight cites unknown ref id.", index=idx, text=t[:160]))

        if lang.startswith("en") and not _is_english_heavy(t):
            issues.append(_issue("info", "highlight.lang_mix", "Expected English highlight but looks non-English.", index=idx))
        if lang.startswith(("zh", "ja")) and not _looks_cjk(t):
            issues.append(_issue("info", "highlight.lang_mix", "Expected CJK highlight but looks non-CJK.", index=idx))

    if len(bullets) > 12:
        issues.append(_issue("info", "highlight.too_many", "Too many highlights; consider ≤10 for readability.", count=len(bullets)))
    return issues

def _check_pack_integrity(doc_id: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    p = _pack_path(doc_id)
    if not p.exists():
        issues.append(_issue("warn", "pack.missing", "pack.json not found", doc_id=doc_id))
        return issues
    data = _load_json(p)
    if not isinstance(data, dict) or not data:
        issues.append(_issue("warn", "pack.empty", "pack.json is empty or unreadable", doc_id=doc_id))
        return issues
    n_p = len(data.get("paragraphs") or [])
    n_t = len(data.get("tables") or [])
    n_f = len(data.get("figures") or [])
    if n_p + n_t + n_f == 0:
        issues.append(_issue("info", "pack.no_content", "No paragraphs/tables/figures extracted", doc_id=doc_id))
    if n_t > 0:
        bad = 0
        for t in (data.get("tables") or []):
            if not isinstance(t, dict) or not t.get("csv_path"):
                bad += 1
        if bad:
            issues.append(_issue("warn", "pack.tables_missing_csv", f"{bad} table(s) missing csv_path", doc_id=doc_id))
    return issues

def _check_tables_and_charts(doc_id: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    data = _load_json(_pack_path(doc_id))
    tables = data.get("tables", []) if isinstance(data, dict) else []
    # chart manifest
    mf = _charts_manifest(doc_id)
    charts_ok = False
    if mf.exists():
        m = _load_json(mf)
        if isinstance(m, dict):
            recs = m.get("records") or []
            charts_ok = isinstance(recs, list) and len(recs) > 0
        elif isinstance(m, list):
            charts_ok = len(m) > 0

    # if tables exist but no charts, warn (builder expected to try)
    if (tables and len(tables) > 0) and not charts_ok:
        issues.append(_issue("warn", "charts.missing_for_tables",
                             "Tables were extracted but no charts were generated", doc_id=doc_id))

    # CSV sanity (soft)
    for t in (tables or []):
        if not isinstance(t, dict):
            continue
        csvp = t.get("csv_path")
        if not csvp:
            issues.append(_issue("warn", "table.no_csv", "Table evidence missing csv_path", doc_id=doc_id))
            continue
        p = Path(csvp)
        if not p.exists() or p.stat().st_size == 0:
            issues.append(_issue("warn", "table.csv_missing", "CSV file missing or empty", doc_id=doc_id, extra={"path": csvp}))
            continue
        try:
            with p.open("r", encoding="utf-8-sig") as f:
                rdr = csv.reader(f)
                row = next(rdr, None)
                if not row or len(row) < 2:
                    issues.append(_issue("info", "table.csv_thin", "CSV seems narrow; verify extraction quality", doc_id=doc_id, extra={"path": csvp}))
        except Exception as e:
            issues.append(_issue("warn", "table.csv_read_fail", f"CSV cannot be read: {str(e)[:120]}", doc_id=doc_id, extra={"path": csvp}))
    return issues

def _check_html_visual_ratio(run_id: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    html = REPORTS_DIR / f"{run_id}.html"
    if not html.exists():
        issues.append(_issue("info", "report.html_absent", "HTML report not found; layout may not have run"))
        return issues
    txt = html.read_text(encoding="utf-8", errors="ignore")
    imgs = len(re.findall(r"<img\s", txt, flags=re.IGNORECASE))
    hls  = len(re.findall(r'class="hl"', txt))
    total_blocks = imgs + max(1, hls)
    ratio = imgs / total_blocks
    if ratio < 0.70:
        issues.append(_issue("warn", "visual.ratio_low", "Visual ratio appears below 70%", extra={"img": imgs, "highlights": hls, "ratio": round(ratio,3)}))
    else:
        issues.append(_issue("info", "visual.ratio_ok", "Visual ratio meets target (>=70%)", extra={"img": imgs, "highlights": hls, "ratio": round(ratio,3)}))
    return issues

def _check_web_images(doc_id: str, has_image_source_hint: bool) -> List[Dict[str, Any]]:
    """
    Soft checks for the image bypass:
      - If citations hint this doc is image-sourced but no web images saved, warn.
      - If web images exist, add an info with count.
    """
    issues: List[Dict[str, Any]] = []
    imgs = _list_web_images_for_doc(doc_id)
    if has_image_source_hint and not imgs:
        issues.append(_issue("warn", "image.source_but_no_files",
                             "Citations indicate image source but no web images were found on disk",
                             doc_id=doc_id))
    if imgs:
        issues.append(_issue("info", "image.web_images_present",
                             "Web images discovered for this doc",
                             doc_id=doc_id, count=len(imgs)))
    return issues

# ----------------------------- public API --------------------------------------

def run_checks(run_id: str, lang: str = "en") -> Dict[str, Any]:
    """
    Return dict with issues & stats; never raise.
    """
    cit_path = _rank_dir() / f"{run_id}.citations.json"
    sum_path = _rank_dir() / f"{run_id}.summary.json"
    ci_raw = _load_json(cit_path)
    su_raw = _load_json(sum_path)

    # Friendly short-circuit for degenerate runs
    if not isinstance(ci_raw, dict) or not ci_raw.get("refs"):
        return {
            "run_id": run_id,
            "issues": [_issue("info", "pipeline.empty", "No references to check; skipping QA")],
            "stats": {"docs_total": 0, "docs_active": 0, "issues_warn": 0, "issues_info": 1},
            "artifacts": {
                "citations_path": str(cit_path),
                "citations_sha256": _sha256_path(cit_path) if cit_path.exists() else None,
                "summary_path": str(sum_path) if sum_path.exists() else "",
                "summary_sha256": _sha256_path(sum_path) if sum_path.exists() else None,
                "html_path": str(REPORTS_DIR / f"{run_id}.html"),
            },
        }

    ci: Dict[str, Any] = ci_raw if isinstance(ci_raw, dict) else {}
    su: Dict[str, Any] = su_raw if isinstance(su_raw, dict) else {}

    issues: List[Dict[str, Any]] = []

    # (1) soft check on citation labels
    f1, _map = _check_citation_items(ci)
    issues.extend(f1)

    # (2) highlights (soft)
    issues.extend(_check_highlights(ci, su, lang=lang))

    # Determine active docs
    doc_ids = _list_doc_ids_from_citations(run_id)

    # Build image-source hints per doc from citation items
    items = ci.get("items", []) if isinstance(ci, dict) else []
    image_hint_by_doc: Dict[str, bool] = {}
    for it in items:
        d = it.get("doc_id")
        if not d:
            continue
        if it.get("source") == "image" or (it.get("type") in ("image", "web_image")):
            image_hint_by_doc[d] = True

    active_map: Dict[str, bool] = {}
    inactive_warns: List[Dict[str, Any]] = []
    for d in doc_ids:
        a, meta = _is_active_doc(d)
        active_map[d] = a
        if not a:
            inactive_warns.append(_issue("warn", "ref.inactive_doc",
                                         "Referenced doc is inactive (no usable evidence/visuals); skipped deep checks",
                                         doc_id=d, extra={"active_meta": meta}))
    # add inactive warnings (once)
    issues.extend(inactive_warns)

    # (3) deep checks only on ACTIVE docs
    for d in doc_ids:
        if not active_map.get(d, False):
            continue
        # pack integrity and tables/charts
        issues.extend(_check_pack_integrity(d))
        issues.extend(_check_tables_and_charts(d))
        # web image soft checks
        issues.extend(_check_web_images(d, image_hint_by_doc.get(d, False)))

    # (4) overall HTML visual ratio (informational)
    issues.extend(_check_html_visual_ratio(run_id))

    report = {
        "run_id": run_id,
        "issues": issues,
        "stats": {
            "docs_total": len(doc_ids),
            "docs_active": sum(1 for v in active_map.values() if v),
            "issues_warn": sum(1 for it in issues if it.get("severity") == "warn"),
            "issues_info": sum(1 for it in issues if it.get("severity") == "info"),
        },
        "artifacts": {
            "citations_path": str(cit_path),
            "citations_sha256": _sha256_path(cit_path) if cit_path.exists() else None,
            "summary_path": str(sum_path) if sum_path.exists() else "",
            "summary_sha256": _sha256_path(sum_path) if sum_path.exists() else None,
            "html_path": str(REPORTS_DIR / f"{run_id}.html"),
        },
    }
    return report

# ----------------------------- CLI --------------------------------------------

if __name__ == "__main__":
    import argparse, json as _json, sys
    ap = argparse.ArgumentParser(description="CiteVizor QA consistency checks (soft)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    out = run_checks(args.run_id, lang=args.lang)
    print(_json.dumps(out, ensure_ascii=False, indent=2))
    # never hard-fail; purely diagnostic
