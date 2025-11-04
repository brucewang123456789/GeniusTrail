# C:\CiteVizor\schemas.py
# -*- coding: utf-8 -*-
"""
CiteVizor – Core data contracts (schemas)
- Single source of truth for all modules.
- Relative paths rooted at project directory (this file's parent).
- Leaves room for future LLM providers/models and retrieval sources.

Enhancements (2025-10-24):
- Evidence pre-sanitization:
  * accept legacy key 'img_path' -> map to 'image_path'
  * coerce figure_no/table_no (int -> str)
  * coerce page (str -> int)
  * coerce type (str -> EvidenceType)
- DocumentPaths/Evidence path validators remain tolerant for str -> Path.

Enhancements (2025-10-27):
- Introduced explicit path contract helpers:
  * pdf_path_for(doc_id), html_path_for(doc_id), meta_json_for(doc_id)
  * evidence_dir_for(doc_id), figures_glob_for(doc_id)
  * export_base_for(doc_id=None, run_id=None), report_paths_for(doc_id, run_id=None)
- Kept backward compatibility (make_doc_paths remains available).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime
from hashlib import sha256
import re

# --------------------------------------------------------------------------------------
# Pydantic v1/v2 compatibility layer
# --------------------------------------------------------------------------------------
_V2 = True
try:
    # pydantic v2
    from pydantic import BaseModel, Field, HttpUrl, validator, model_validator, ConfigDict
except Exception:  # pragma: no cover
    # pydantic v1 fallback
    _V2 = False
    from pydantic.v1 import (  # type: ignore
        BaseModel, Field, HttpUrl, validator, root_validator as model_validator
    )
    class ConfigDict(dict):  # dummy for v1
        pass

# --------------------------------------------------------------------------------------
# Project-relative paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent  # e.g., C:\CiteVizor
STORAGE_DIR: Path = PROJECT_ROOT / "storage"
DOCS_DIR: Path = STORAGE_DIR / "docs"            # raw PDFs/HTML
EVIDENCE_DIR: Path = STORAGE_DIR / "evidence"    # paragraphs/tables/figures
RENDERS_DIR: Path = STORAGE_DIR / "renders"      # generated charts/diagrams
REPORTS_DIR: Path = STORAGE_DIR / "reports"      # html/pdf outputs
MANIFEST_DIR: Path = STORAGE_DIR / "manifests"   # trace manifests
CACHE_DIR: Path = STORAGE_DIR / "cache"          # transient caches


def ensure_dirs() -> None:
    """Create required storage subdirectories if not present."""
    for p in (STORAGE_DIR, DOCS_DIR, EVIDENCE_DIR, RENDERS_DIR, REPORTS_DIR, MANIFEST_DIR, CACHE_DIR):
        p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Path contract helpers (NEW) – the only source of truth for filenames/locations
# --------------------------------------------------------------------------------------

def pdf_path_for(doc_id: str) -> Path:
    """Return normalized PDF path: storage/docs/<doc_id>/<doc_id>.pdf"""
    return DOCS_DIR / doc_id / f"{doc_id}.pdf"

def html_path_for(doc_id: str) -> Path:
    """Return normalized raw-HTML path (if saved): storage/docs/<doc_id>/<doc_id>.html"""
    return DOCS_DIR / doc_id / f"{doc_id}.html"

def meta_json_for(doc_id: str) -> Path:
    """Return normalized metadata JSON path: storage/docs/<doc_id>/<doc_id>.meta.json"""
    return DOCS_DIR / doc_id / f"{doc_id}.meta.json"

def evidence_dir_for(doc_id: str) -> Path:
    """Return evidence directory: storage/evidence/<doc_id>/"""
    return EVIDENCE_DIR / doc_id

def figures_glob_for(doc_id: str) -> str:
    """Return glob pattern for figure images emitted by figure_extract: <doc_id>_fg_*.*"""
    return str(evidence_dir_for(doc_id) / f"{doc_id}_fg_*.*")

def export_base_for(*, doc_id: Optional[str] = None, run_id: Optional[str] = None) -> Path:
    """
    Return base path (without extension) for downstream exports (HTML/PDF/PPTX).
    If run_id is provided, use '<run_id>_<doc_id>' to ensure uniqueness across a run.
    Otherwise default to '<doc_id>'.
    Example: storage/reports/20251027_abcd1234 -> (.html/.pdf/.pptx)
    """
    if doc_id is None and run_id is None:
        raise ValueError("export_base_for requires at least doc_id or run_id")
    base_name = (f"{run_id}_{doc_id}" if (run_id and doc_id) else (run_id or doc_id))
    return REPORTS_DIR / str(base_name)

def report_paths_for(doc_id: str, *, run_id: Optional[str] = None) -> Dict[str, Path]:
    """
    Return a dict with canonical output paths:
      - html: storage/reports/<base>.html
      - pdf:  storage/reports/<base>.pdf
      - pptx: storage/reports/<base>.pptx
      - assets_dir: storage/reports/<base>.assets/
    """
    base = export_base_for(doc_id=doc_id, run_id=run_id)
    return {
        "html": base.with_suffix(".html"),
        "pdf": base.with_suffix(".pdf"),
        "pptx": base.with_suffix(".pptx"),
        "assets_dir": Path(f"{base}.assets"),
    }

# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------

class LanguageCode(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    ja = "ja"

class Audience(str, Enum):
    engineer = "engineer"
    scientist = "scientist"

class ReportStyle(str, Enum):
    IRE = "IRE"
    DesignPV = "DesignPV"
    Default = "Default"

class ChartBackend(str, Enum):
    matplotlib = "matplotlib"
    plotly = "plotly"

class SourceType(str, Enum):
    scholar = "scholar"          # Google Scholar via Serper
    arxiv = "arxiv"
    crossref = "crossref"
    patent = "patent"
    news = "news"
    university = "university"
    web = "web"                  # generic web (fallback)
    image = "image"              # image vertical (Serper Images)  # <- added

class EvidenceType(str, Enum):
    paragraph = "paragraph"
    table = "table"
    figure = "figure"
    equation = "equation"
    caption = "caption"
    web_image = "web_image"      # direct web image evidence       # <- added

class ModelProvider(str, Enum):
    vllm = "vllm"                # OpenAI-compatible/vLLM self-host
    openai = "openai"            # Hosted (placeholder for future)
    hf_transformers = "hf_transformers"  # Local transformers runner
    other = "other"

# --------------------------------------------------------------------------------------
# Model / Tool configurations (future-proof for switching models & sources)
# --------------------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """LLM runtime configuration; designed to be swappable without touching pipelines."""
    provider: ModelProvider = Field(default=ModelProvider.vllm)
    model_name: str = Field(default="Qwen2.5-14B-Instruct-AWQ")
    endpoint_url: Optional[HttpUrl] = Field(default=None, description="HTTP endpoint if provider exposes one")
    api_key_env: Optional[str] = Field(default=None, description="Environment variable name holding API key")
    context_window: int = Field(default=16384)
    max_output_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    use_tools: bool = Field(default=True, description="Allow tool-calling / function-calling")
    quantization: Optional[str] = Field(default="awq-4bit", description="awq-4bit/gptq-4bit/none etc.")

    # v2-native config + v1 fallback
    if _V2:
        # Added protected_namespaces=() to avoid warning when using fields like "model_name"
        model_config = ConfigDict(
            extra="ignore",
            protected_namespaces=()  # <- precise fix for the pydantic protected namespace warning
        )  # type: ignore[attr-defined]
    else:  # pragma: no cover
        class Config:
            extra = "ignore"

class RetrievalSourceConfig(BaseModel):
    """Switches for real-time retrieval sources."""
    enable_scholar: bool = True
    enable_arxiv: bool = True
    enable_crossref: bool = True
    enable_patent: bool = False
    enable_news: bool = False
    enable_university: bool = False

# --------------------------------------------------------------------------------------
# Core data contracts
# --------------------------------------------------------------------------------------

class YearRange(BaseModel):
    start: Optional[int] = Field(default=None, ge=1900, le=2100)
    end: Optional[int] = Field(default=None, ge=1900, le=2100)

    @model_validator(mode="after")
    def _validate_range(self) -> "YearRange":  # type: ignore[override]
        if self.start and self.end and self.start > self.end:
            raise ValueError("YearRange.start must be <= YearRange.end")
        return self

class UserQuery(BaseModel):
    query: str
    lang: LanguageCode = LanguageCode.auto
    region: Optional[str] = Field(default=None, description="e.g., Japan/HongKong/China or ISO country")
    years: Optional[YearRange] = None
    facets: Dict[str, Any] = Field(default_factory=dict)  # material/device/metric, etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)

class SearchPlan(BaseModel):
    queries: List[str] = Field(default_factory=list)
    sources: List[SourceType] = Field(default_factory=lambda: [SourceType.scholar, SourceType.arxiv, SourceType.crossref])
    k_top: int = 20
    filters: Dict[str, Any] = Field(default_factory=dict)  # region/years/review-only etc.
    planner_model: Optional[str] = None  # which LLM generated this plan

class Candidate(BaseModel):
    # Minimal metadata needed before fetching
    doc_id: Optional[str] = None   # will be filled after normalization
    title: str
    url: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    source: SourceType
    score: float = 0.0

    authors: List[str] = Field(default_factory=list)
    venue: Optional[str] = None
    is_review: Optional[bool] = None
    open_access: Optional[bool] = None
    region: Optional[str] = None

    @validator("url", pre=True)
    def _normalize_url(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() in {"n/a", "na", "null", "none", "-"}:
            return None
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", s):
            s = "http://" + s
        return s

    def stable_key(self) -> str:
        """Generate a stable dedupe key (doi preferred; else title+year)."""
        base = (self.doi or f"{self.title}|{self.year or ''}").lower()
        return sha256(base.encode("utf-8")).hexdigest()[:16]

class DocumentPaths(BaseModel):
    """Filesystem locations for raw and normalized files (relative paths)."""
    base_dir: Path = Field(default=DOCS_DIR)
    pdf_path: Optional[Path] = None
    html_path: Optional[Path] = None
    meta_json: Optional[Path] = None

    @validator("base_dir", pre=True)
    def _pathify(cls, v: Any) -> Path:
        return Path(v)

    @validator("pdf_path", "html_path", "meta_json", pre=True)
    def _pathify_opt(cls, v: Any) -> Optional[Path]:
        return None if v is None else Path(v)

    def ensure_parent(self) -> None:
        """Ensure parent directories exist for any populated path fields."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for p in (self.pdf_path, self.html_path, self.meta_json):
            if p is not None:
                Path(p).parent.mkdir(parents=True, exist_ok=True)

class FetchedDoc(BaseModel):
    doc_id: str
    candidate: Candidate
    paths: DocumentPaths
    bytes_size: Optional[int] = None
    md5: Optional[str] = None

# ----------------------------- Evidence & Packs --------------------------------

def _coerce_int(v: Any) -> Optional[int]:
    if v in (None, "", "null", "None"):
        return None
    try:
        return int(v)
    except Exception:
        return None

def _coerce_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    return str(v)

def _evidence_pre_sanitize(data: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort sanitation for legacy fields and soft type coercions."""
    x = dict(data) if isinstance(data, dict) else {}

    # img_path -> image_path
    if "img_path" in x and "image_path" not in x:
        x["image_path"] = x.pop("img_path")

    # figure_no/table_no -> str
    if "figure_no" in x and x["figure_no"] is not None:
        x["figure_no"] = _coerce_str(x["figure_no"])
    if "table_no" in x and x["table_no"] is not None:
        x["table_no"] = _coerce_str(x["table_no"])

    # page -> int
    if "page" in x:
        x["page"] = _coerce_int(x.get("page"))

    # type -> EvidenceType if provided as string
    if "type" in x and isinstance(x["type"], str):
        try:
            x["type"] = EvidenceType(x["type"])
        except Exception:
            # leave as-is; Pydantic will raise if invalid
            pass

    # path-like coercions handled by field validators; ensure strings ok
    for k in ("csv_path", "image_path"):
        if k in x and x[k] is not None:
            x[k] = str(x[k])

    return x

class Evidence(BaseModel):
    """Unified evidence slice extracted from documents."""
    id: str
    type: EvidenceType
    doc_id: str
    page: Optional[int] = None
    figure_no: Optional[str] = None
    table_no: Optional[str] = None
    # Content fields (only one of them will be set depending on type)
    text: Optional[str] = None                 # for paragraphs/captions
    csv_path: Optional[Path] = None            # for tables (normalized to CSV/JSON)
    image_path: Optional[Path] = None          # for figures (PNG/SVG)
    caption: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # x1,y1,x2,y2 (optional)

    source_url: Optional[HttpUrl] = None       # redundancy for quick preview
    doi: Optional[str] = None

    # v2/v1 compatible "pre" normalization
    if _V2:
        @model_validator(mode="before")
        @classmethod
        def _pre(cls, data: Any) -> Any:
            if isinstance(data, dict):
                return _evidence_pre_sanitize(data)
            return data
    else:  # pragma: no cover
        @model_validator(pre=True)
        def _pre_v1(cls, values: Dict[str, Any]) -> Dict[str, Any]:
            return _evidence_pre_sanitize(values)

    @validator("csv_path", "image_path", pre=True)
    def _to_path(cls, v: Any) -> Optional[Path]:
        return None if v is None else Path(v)

class EvidencePack(BaseModel):
    """All extracted evidence for a single document."""
    doc_id: str
    paragraphs: List[Evidence] = Field(default_factory=list)
    tables: List[Evidence] = Field(default_factory=list)
    figures: List[Evidence] = Field(default_factory=list)

class Claim(BaseModel):
    """Atomic claim synthesized from evidence with explicit citations."""
    slug: str
    text: str
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)

class Citation(BaseModel):
    doc_id: str
    evidence_id: str
    page: Optional[int] = None
    figure_no: Optional[str] = None
    table_no: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[HttpUrl] = None

class AlignedEvidence(BaseModel):
    claims: List[Claim] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)

class ChartCode(BaseModel):
    """Reproducible plotting code produced by the LLM (matplotlib/plotly)."""
    evidence_id: Optional[str] = None  # which table/image it was derived from
    backend: ChartBackend = ChartBackend.matplotlib
    code: str  # Python code snippet to render the chart
    note: Optional[str] = None

    def script_hash(self) -> str:
        return sha256(self.code.encode("utf-8")).hexdigest()[:12]

class TextBlocks(BaseModel):
    """Minimal necessary text outputs."""
    highlights: List[str] = Field(default_factory=list)       # key takeaways in engineer voice
    figure_captions: Dict[str, str] = Field(default_factory=dict)  # evidence_id -> caption text
    chart_codes: List[ChartCode] = Field(default_factory=list)

class ReportSpec(BaseModel):
    title: str
    audience: Audience = Audience.engineer
    image_ratio_target: float = Field(default=0.7, ge=0.5, le=0.95)
    style: ReportStyle = ReportStyle.IRE
    page_limit: int = 8
    export_formats: List[str] = Field(default_factory=lambda: ["html", "pdf"])

class ReportBundle(BaseModel):
    html_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    assets_dir: Optional[Path] = None

    @validator("html_path", "pdf_path", "assets_dir", pre=True)
    def _to_path(cls, v: Any) -> Optional[Path]:
        return None if v is None else Path(v)

class ConsistencyIssue(BaseModel):
    code: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)

class QAResult(BaseModel):
    passed: bool = False
    issues: List[ConsistencyIssue] = Field(default_factory=list)

class TraceManifest(BaseModel):
    """Provenance & reproducibility."""
    run_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    user_query: UserQuery
    # NOTE: avoid the reserved name `model_config` to be safe across Pydantic versions
    llm_config: ModelConfig
    retrieval_config: RetrievalSourceConfig
    candidates: List[Candidate] = Field(default_factory=list)
    docs: List[FetchedDoc] = Field(default_factory=list)
    evidences: List[Evidence] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    chart_scripts: List[ChartCode] = Field(default_factory=list)
    report: Optional[ReportBundle] = None
    qa: Optional[QAResult] = None
    artifacts_dir: Path = Field(default=STORAGE_DIR)

    @validator("artifacts_dir", pre=True)
    def _to_path(cls, v: Any) -> Path:
        return Path(v)

# --------------------------------------------------------------------------------------
# Helper factories (kept for backward compatibility)
# --------------------------------------------------------------------------------------

def make_doc_paths(doc_id: str) -> DocumentPaths:
    """Create document paths for a given doc_id under DOCS_DIR."""
    ddir = DOCS_DIR / doc_id
    return DocumentPaths(
        base_dir=ddir,
        pdf_path=ddir / f"{doc_id}.pdf",
        html_path=ddir / f"{doc_id}.html",
        meta_json=ddir / f"{doc_id}.meta.json",
    )

def default_model_config() -> ModelConfig:
    """Factory for current recommended local model on RTX 4090."""
    return ModelConfig(
        provider=ModelProvider.vllm,
        model_name="Qwen2.5-14B-Instruct-AWQ",
        endpoint_url=None,  # fill with http://127.0.0.1:8000/v1 if using OpenAI-compatible vLLM
        api_key_env=None,
        context_window=16384,
        max_output_tokens=1024,
        temperature=0.2,
        top_p=0.9,
        use_tools=True,
        quantization="awq-4bit",
    )

def default_retrieval_config() -> RetrievalSourceConfig:
    return RetrievalSourceConfig(
        enable_scholar=True, enable_arxiv=True, enable_crossref=True,
        enable_patent=False, enable_news=False, enable_university=False
    )

# --------------------------------------------------------------------------------------
# Self-test (safe to run; creates folders only)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    ensure_dirs()
    uq = UserQuery(query="最新的太阳能板设计趋势", lang=LanguageCode.zh, years=YearRange(start=2022, end=2025))
    plan = SearchPlan(queries=["solar panel film", "coastal corrosion PV encapsulant"], planner_model="Qwen2.5-14B")
    cand = Candidate(title="Recent advances in PV encapsulants", url="https://example.org/paper", year=2024, source=SourceType.scholar)
    doc_id = cand.stable_key()
    doc_paths = make_doc_paths(doc_id=doc_id)
    fetched = FetchedDoc(doc_id=doc_id, candidate=cand, paths=doc_paths)
    # legacy-like evidence payload to test coercions
    ev_raw = {
        "id": "e1",
        "type": "figure",
        "doc_id": fetched.doc_id,
        "page": "2",
        "figure_no": 1,
        "img_path": str(RENDERS_DIR / "legacy.png"),
        "caption": "Figure 1. Demo caption"
    }
    ev = Evidence(**ev_raw)
    assert isinstance(ev.image_path, Path) and ev.figure_no == "1" and ev.page == 2

    # smoke test for new path helpers
    assert pdf_path_for(doc_id) == DOCS_DIR / doc_id / f"{doc_id}.pdf"
    rp = report_paths_for(doc_id, run_id="20251027")
    assert rp["html"].suffix == ".html" and rp["assets_dir"].name.endswith(".assets")

    print("CiteVizor schemas self-test OK")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Storage dirs ensured at:", STORAGE_DIR)
