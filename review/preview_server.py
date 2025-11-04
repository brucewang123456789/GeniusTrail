# C:\CiteVizor\review\preview_server.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Minimal review/preview server (Human-in-the-loop)

What this provides
- Static mount to /storage -> C:\CiteVizor\storage (serves charts/figures directly).
- Run list page:          GET /runs
- Preview UI for a run:   GET /runs/{run_id}
- Data API:               GET /api/run/{run_id}/data
- Save review:            POST /api/run/{run_id}/save
- Health check:           GET /health

Artifacts used (read-only)
- storage/evidence/_rank/<run_id>.citations.json
- storage/evidence/_rank/<run_id>.summary.json    (optional; pulls default highlights/title)
- storage/renders/<doc_id>/charts/_charts.json    (optional)
- storage/evidence/<doc_id>/pack.json             (figures -> image_path)

Artifact written
- storage/evidence/_rank/<run_id>.review.json

No LLM calls. No new environment variables required.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from schemas import STORAGE_DIR, EVIDENCE_DIR, RENDERS_DIR  # canonical paths

# ----------------------------- logging (safe, zero-impact) ---------------------
# If infra.logging is present, we use unified structured logging;
# otherwise, we fall back to no-op stubs (never break the server).
try:
    from infra.logging import (
        setup_logging,
        add_fastapi_request_logging,
        get_logger,
        set_run_id,
        set_request_id,
        log_event,
    )
except Exception:  # graceful fallback
    def setup_logging(*args, **kwargs):  # type: ignore
        return None
    def add_fastapi_request_logging(*args, **kwargs):  # type: ignore
        return None
    def get_logger(name: Optional[str] = None):  # type: ignore
        class _L:
            def info(self, *a, **k): pass
            def debug(self, *a, **k): pass
            def warning(self, *a, **k): pass
            def error(self, *a, **k): pass
        return _L()
    def set_run_id(*args, **kwargs):  # type: ignore
        return None
    def set_request_id(*args, **kwargs):  # type: ignore
        return ""
    def log_event(*args, **kwargs):  # type: ignore
        return None

LOG = get_logger("preview")

# ----------------------------- FS helpers --------------------------------------

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
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _rel_storage_url(p: Path) -> str:
    """
    Build a URL under /storage for a path inside STORAGE_DIR.
    """
    try:
        rel = p.resolve().relative_to(STORAGE_DIR.resolve())
    except Exception:
        return ""
    # Use forward slashes for URLs
    return "/storage/" + str(rel).replace("\\", "/")

def _charts_manifest(doc_id: str) -> List[Dict[str, Any]]:
    p = RENDERS_DIR / doc_id / "charts" / "_charts.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []

def _pack_json(doc_id: str) -> Dict[str, Any]:
    return _load_json(EVIDENCE_DIR / doc_id / "pack.json")

def _citations(run_id: str) -> Dict[str, Any]:
    return _load_json(_rank_dir() / f"{run_id}.citations.json")

def _summary(run_id: str) -> Dict[str, Any]:
    return _load_json(_rank_dir() / f"{run_id}.summary.json")

def _review_path(run_id: str) -> Path:
    return _rank_dir() / f"{run_id}.review.json"

def _list_runs() -> List[str]:
    return sorted([p.stem.split(".")[0] for p in _rank_dir().glob("*.citations.json")])

# ----------------------------- preview datamodel --------------------------------

@dataclass
class ItemView:
    item_id: str           # unique: type:doc_id:evidence_id (or hashed snippet)
    type: str              # table|figure|paragraph|...
    doc_id: str
    evidence_id: Optional[str]
    label: str
    snippet: str
    img_url: Optional[str] # chart/figure preview if any
    score: float

def _item_uid(it: Dict[str, Any]) -> str:
    # Prefer evidence_id when present; fall back to a short hash of snippet.
    ev = it.get("evidence_id")
    t = it.get("type") or "item"
    d = it.get("doc_id") or "doc"
    if ev:
        return f"{t}:{d}:{ev}"
    snip = (it.get("snippet") or "")[:160]
    import hashlib
    sig = hashlib.blake2b(snip.encode("utf-8", errors="ignore"), digest_size=6).hexdigest()
    return f"{t}:{d}:snip_{sig}"

def _index_visuals_for_doc(doc_id: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return two maps:
      charts:  evidence_id -> /storage/...png
      figures: evidence_id -> /storage/...png
    """
    # Charts
    charts: Dict[str, str] = {}
    for rec in _charts_manifest(doc_id):
        evid = str(rec.get("evidence_id") or "")
        png = Path(rec.get("png") or "")
        if evid and png.exists():
            charts[evid] = _rel_storage_url(png)
        else:
            # try relative to charts dir
            if evid:
                alt = RENDERS_DIR / doc_id / "charts" / Path(png.name if png else f"{evid}.png")
                if alt.exists():
                    charts[evid] = _rel_storage_url(alt)

    # Figures
    figures: Dict[str, str] = {}
    pack = _pack_json(doc_id)
    for f in pack.get("figures", []):
        evid = str(f.get("id") or "")
        img = Path(f.get("image_path") or "")
        if evid and img.exists():
            figures[evid] = _rel_storage_url(img)
        elif evid:
            alt = EVIDENCE_DIR / doc_id / "figures" / img.name
            if alt.exists():
                figures[evid] = _rel_storage_url(alt)
    return charts, figures

def _build_item_views(run_id: str, limit: int = 120) -> List[ItemView]:
    ci = _citations(run_id)
    items = ci.get("items") or []
    out: List[ItemView] = []
    # Pre-index visuals by doc to avoid repeated file reads
    cache: Dict[str, Tuple[Dict[str, str], Dict[str, str]]] = {}

    for it in items[:limit]:
        doc = it.get("doc_id") or ""
        if doc not in cache:
            cache[doc] = _index_visuals_for_doc(doc)
        charts, figures = cache[doc]
        evi = str(it.get("evidence_id") or "")

        img_url: Optional[str] = None
        if it.get("type") == "table" and evi in charts:
            img_url = charts[evi]
        elif it.get("type") == "figure" and evi in figures:
            img_url = figures[evi]

        out.append(ItemView(
            item_id=_item_uid(it),
            type=it.get("type") or "item",
            doc_id=doc,
            evidence_id=(evi or None),
            label=it.get("label") or "",
            snippet=it.get("snippet") or "",
            img_url=img_url,
            score=float(it.get("score") or 0.0),
        ))
    return out

# ----------------------------- FastAPI app -------------------------------------

app = FastAPI(title="CiteVizor Preview Server", version="0.1.0")

# initialize logging once; attach request logging (safe if called multiple times)
try:
    setup_logging()
    add_fastapi_request_logging(app)
    LOG.info("preview server init", extra={"storage": str(STORAGE_DIR)})
except Exception:
    pass

# Mount storage for serving images/charts directly
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR), html=False), name="storage")

@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        runs = _list_runs()[:10]
        LOG.info("health", extra={"runs_count": len(runs)})
        return {"ok": True, "storage": str(STORAGE_DIR), "runs": runs}
    except Exception as e:
        LOG.error("health_error", extra={"error": repr(e)})
        return {"ok": False, "error": str(e)}

# ---------- HTML pages ----------

def _html_runs() -> str:
    runs = _list_runs()
    lis = "\n".join([f'<li><a href="/runs/{r}">{r}</a></li>' for r in runs])
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>CiteVizor – Runs</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 8px; }}
    .hint {{ color:#777; margin-bottom:16px; }}
  </style>
</head>
<body>
  <h1>Available runs</h1>
  <div class="hint">Click a run id to open the review UI.</div>
  <ul>
    {lis or "<li>No runs found. Execute the pipeline first.</li>"}
  </ul>
</body>
</html>"""

@app.get("/", response_class=Response)
def root() -> Response:
    try:
        html = _html_runs()
        LOG.info("page_runs")
        return Response(content=html, media_type="text/html")
    except Exception as e:
        LOG.error("page_runs_error", extra={"error": repr(e)})
        return Response(content="Internal error", media_type="text/plain", status_code=500)

@app.get("/runs", response_class=Response)
def runs() -> Response:
    return root()

@app.get("/runs/{run_id}", response_class=Response)
def run_preview(run_id: str) -> Response:
    # Bind run_id into logging context so every log line is traceable.
    try:
        if run_id:
            set_run_id(str(run_id))
    except Exception:
        pass

    # A self-contained HTML + JS UI (no external assets).
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>CiteVizor – Review {run_id}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: Arial, sans-serif; margin: 0; }}
    header {{ padding: 12px 16px; background: #0f766e; color: #fff; }}
    header a {{ color:#d1fae5; text-decoration:none; margin-right:12px; }}
    .wrap {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; padding: 16px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; background:#fff; }}
    .card h2 {{ margin:0; padding:10px 12px; border-bottom:1px solid #e5e7eb; font-size:16px; background:#f8fafc; }}
    .list {{ max-height: calc(100vh - 160px); overflow: auto; }}
    .item {{ display:flex; gap:12px; padding:10px 12px; border-bottom:1px solid #f1f5f9; align-items:flex-start; }}
    .thumb {{ width:180px; height:120px; background:#f1f5f9; display:flex; align-items:center; justify-content:center; overflow:hidden; border:1px solid #e2e8f0; border-radius:4px; }}
    .thumb img {{ max-width:100%; max-height:100%; }}
    .meta {{ flex:1; }}
    .meta .label {{ color:#0f766e; font-weight:bold; }}
    .meta textarea {{ width:100%; min-height:60px; resize: vertical; }}
    .row {{ margin-top:6px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
    .pill {{ font-size:12px; background:#ecfeff; color:#0e7490; padding:2px 6px; border-radius:4px; border:1px solid #cffafe; }}
    .panel {{ padding:12px; }}
    .panel input[type="text"] {{ width:100%; padding:8px; }}
    .panel textarea {{ width:100%; min-height:150px; }}
    .actions {{ display:flex; gap:8px; justify-content:flex-end; padding:10px; border-top:1px solid #e5e7eb; background:#f8fafc; }}
    button {{ padding:8px 12px; border:1px solid #0f766e; background:#0f766e; color:#fff; border-radius:6px; cursor:pointer; }}
    button.secondary {{ background:#fff; color:#0f766e; }}
    .muted {{ color:#64748b; font-size:12px; }}
  </style>
</head>
<body>
  <header>
    <a href="/runs">&larr; runs</a>
    <strong>Review – {run_id}</strong>
  </header>
  <div class="wrap">
    <div class="card">
      <h2>Visual evidence (check to include; drag-to-reorder soon)</h2>
      <div class="list" id="evid-list"></div>
      <div class="actions">
        <button class="secondary" id="btnReload">Reload</button>
        <button id="btnSave">Save review</button>
      </div>
    </div>
    <div class="card">
      <h2>Report meta</h2>
      <div class="panel">
        <label class="muted">Title</label>
        <input type="text" id="reportTitle" placeholder="CiteVizor Report"/>
        <div class="row muted" id="queryRow"></div>
        <div style="height:8px"></div>
        <label class="muted">Key highlights (one per line)</label>
        <textarea id="highlights"></textarea>
        <div class="row" style="margin-top:10px">
          <span class="pill" id="statItems">items: 0</span>
          <span class="pill" id="statSelected">selected: 0</span>
        </div>
      </div>
    </div>
  </div>

<script>
const RUN_ID = {json.dumps(run_id)!r};
let STATE = null;   // loaded json
let REVIEW = null;  // existing review

function el(tag, attrs={{}}, children=[]) {{
  const e = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)) {{
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v);
  }}
  for (const c of children) e.appendChild(c);
  return e;
}}

async function loadData() {{
  const r = await fetch(`/api/run/${{RUN_ID}}/data`);
  const j = await r.json();
  STATE = j;
  REVIEW = j.review || {{}};
  document.getElementById("reportTitle").value = REVIEW.title || j.title || "CiteVizor Report";
  document.getElementById("highlights").value = (REVIEW.highlights || j.highlights || []).join("\\n");
  const qrow = document.getElementById("queryRow");
  qrow.innerHTML = `<span class="pill">query</span> <span class="muted">${{j.query || ""}}</span>`;
  renderList(j.items, REVIEW);
}}

function renderList(items, review) {{
  const list = document.getElementById("evid-list");
  list.innerHTML = "";
  const selectedSet = new Set(review.selected_ids || []);
  const edits = review.edits || {{}};
  let selCount = 0;

  for (const it of items) {{
    const isSel = selectedSet.has(it.item_id);
    if (isSel) selCount++;

    const chk = el("input", {{type: "checkbox"}});
    chk.checked = isSel;
    chk.addEventListener("change", () => updateStats());

    const caption = el("textarea");
    caption.value = (edits[it.item_id]?.caption) || it.snippet || "";

    const label = el("input", {{type: "text"}});
    label.value = (edits[it.item_id]?.label) || it.label || "";

    const row = el("div", {{class:"item"}}, [
      el("div", {{class:"thumb"}}, [
        it.img_url ? el("img", {{src: it.img_url}}) : el("div", {{class:"muted", html:"no preview"}})
      ]),
      el("div", {{class:"meta"}}, [
        el("div", {{class:"row"}}, [
          chk,
          el("span", {{class:"pill"}}, [document.createTextNode(it.type)]),
          el("span", {{class:"pill"}}, [document.createTextNode("doc:"+it.doc_id)]),
          el("span", {{class:"pill"}}, [document.createTextNode("score:"+Number(it.score).toFixed(3))]),
        ]),
        el("div", {{class:"label"}}, [document.createTextNode(it.label || "[ref]")]),
        el("div", {{class:"muted"}}, [document.createTextNode(it.item_id)]),
        <div class="row">
          <input type="text" placeholder="label override"/>
        </div>
        <div class="muted">Caption (editable):</div>
        {{}}
      ])
    ]);

    // The textarea element above is created separately to ensure correct DOM nesting
    const textarea = row.querySelector('.meta').appendChild(document.createElement('textarea'));
    textarea.value = caption.value;

    // Bind label input (the one with actual value)
    row.querySelector('.row input[type="text"]').value = label.value;

    // Attach metadata for later extraction
    row.dataset.itemId = it.item_id;
    row.dataset.docId = it.doc_id;

    // on edit, just mark dirty
    textarea.addEventListener("input", () => row.dataset.edited = "1");
    row.querySelector('.row input[type="text"]').addEventListener("input", () => row.dataset.edited = "1");

    list.appendChild(row);
  }}

  document.getElementById("statItems").innerText = "items: " + items.length;
  document.getElementById("statSelected").innerText = "selected: " + selCount;
}}

function collectReview() {{
  const list = document.getElementById("evid-list");
  const rows = Array.from(list.querySelectorAll(".item"));
  const selected_ids = [];
  const order = [];
  const edits = {{}};

  for (const r of rows) {{
    const id = r.dataset.itemId;
    order.push(id);
    const chk = r.querySelector('input[type="checkbox"]');
    if (chk && chk.checked) selected_ids.push(id);

    const capEl = r.querySelector('.meta textarea');
    const labEl = r.querySelector('.row input[type="text"]');
    const cap = capEl ? capEl.value : "";
    const lab = labEl ? labEl.value : "";
    if (cap || lab) {{
      edits[id] = {{ caption: cap, label: lab }};
    }}
  }}

  const title = document.getElementById("reportTitle").value || "";
  const highlights = (document.getElementById("highlights").value || "").split(/\\n/).map(s => s.trim()).filter(Boolean);

  return {{
    run_id: RUN_ID,
    title, highlights,
    selected_ids, order, edits,
    updated_at: Math.floor(Date.now()/1000)
  }};
}}

async function saveReview() {{
  const payload = collectReview();
  const r = await fetch(`/api/run/${{RUN_ID}}/save`, {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: JSON.stringify(payload)
  }});
  if (r.ok) {{
    alert("Saved.");
  }} else {{
    const t = await r.text();
    alert("Save failed: " + t);
  }}
}}

function updateStats() {{
  const list = document.getElementById("evid-list");
  const sel = list.querySelectorAll('input[type="checkbox"]:checked').length;
  document.getElementById("statSelected").innerText = "selected: " + sel;
}}

document.getElementById("btnSave").addEventListener("click", saveReview);
document.getElementById("btnReload").addEventListener("click", loadData);
loadData();
</script>
</body>
</html>"""
    try:
        LOG.info("page_run", extra={"run_id": run_id})
    except Exception:
        pass
    return Response(content=html, media_type="text/html")

# ---------- JSON APIs ----------

@app.get("/api/run/{run_id}/data")
def api_run_data(run_id: str, request: Request) -> JSONResponse:
    # bind run_id for this request scope (best effort)
    try:
        if run_id:
            set_run_id(str(run_id))
    except Exception:
        pass

    ci = _citations(run_id)
    if not ci:
        LOG.warning("api_data_citations_missing", extra={"run_id": run_id})
        return JSONResponse({"error": "citations not found"}, status_code=404)

    su = _summary(run_id)
    rv = _load_json(_review_path(run_id))

    # Default title/highlights
    title = (rv.get("title") if rv else None) or (su.get("title") if su else None) or f"CiteVizor Report – {ci.get('query','')[:48]}"
    highs = (rv.get("highlights") if rv else None) or (su.get("highlights") if su else None) or []

    items = [asdict(iv) for iv in _build_item_views(run_id, limit=160)]

    payload = {
        "run_id": run_id,
        "query": ci.get("query"),
        "title": title,
        "highlights": highs,
        "items": items,
        "review": rv or {},
    }

    try:
        LOG.info("api_data", extra={"run_id": run_id, "items": len(items)})
    except Exception:
        pass
    return JSONResponse(payload)

@app.post("/api/run/{run_id}/save")
async def api_save_review(run_id: str, req: Request) -> JSONResponse:
    # bind run_id for this request scope (best effort)
    try:
        if run_id:
            set_run_id(str(run_id))
    except Exception:
        pass

    try:
        body = await req.json()
    except Exception as e:
        LOG.error("api_save_invalid_json", extra={"run_id": run_id, "error": repr(e)})
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    # Minimal validation
    if body.get("run_id") and str(body["run_id"]) != str(run_id):
        LOG.warning("api_save_runid_mismatch", extra={"run_id": run_id})
        return JSONResponse({"error": "run_id mismatch"}, status_code=400)

    allowed_keys = {"title", "highlights", "selected_ids", "order", "edits", "updated_at", "run_id"}
    clean = {k: v for k, v in body.items() if k in allowed_keys}
    clean["run_id"] = run_id
    clean.setdefault("updated_at", int(time.time()))

    # Persist
    p = _review_path(run_id)
    try:
        _save_json(p, clean)
        LOG.info("api_save_ok", extra={"run_id": run_id, "path": str(p), "selected": len(clean.get("selected_ids", []))})
        return JSONResponse({"ok": True, "path": str(p)})
    except Exception as e:
        LOG.error("api_save_error", extra={"run_id": run_id, "error": repr(e)})
        return JSONResponse({"error": "save failed"}, status_code=500)

# ----------------------------- CLI runner --------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor preview/review server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    return ap.parse_args()

def main():
    args = _parse_args()
    # setup logging once in CLI mode too
    try:
        setup_logging()
        LOG.info("preview_cli_start", extra={"host": args.host, "port": args.port})
    except Exception:
        pass
    import uvicorn
    uvicorn.run("review.preview_server:app", host=args.host, port=args.port, reload=False, workers=1)

if __name__ == "__main__":
    main()
