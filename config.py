# C:\CiteVizor\config.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Centralized configuration loader and validator

What this module does
- Load key=value pairs from `citevizor.env` at project root and merge with OS env.
- Provide typed dataclasses for the subsystems (retrieval/LLM/parse/rank/planner/enrich).
- Offer light validation (required vs optional) and a helper to write a template `.env`.
- Keep comments and docs in ENGLISH (per project standard).

Notes
- No new env keys are required by this module; it only reads what's present.
- LLM choice remains Qwen/Qwen2.5-14B-Instruct-AWQ served via local vLLM (OpenAI-compatible).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


ENV_FILE = "citevizor.env"


# ----------------------------- project root ------------------------------------

def discover_project_root() -> Path:
    """
    Resolve project root; by convention we deploy under 'C:\\CiteVizor' on Windows.
    Fallback to the directory of this file if the canonical path is not present.
    """
    canonical = Path(r"C:\CiteVizor")
    if canonical.exists():
        return canonical
    return Path(__file__).resolve().parent


def _read_env_file(root: Path) -> Dict[str, str]:
    """Read simple KEY=VALUE lines (no export, no interpolation)."""
    p = root / ENV_FILE
    out: Dict[str, str] = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _merge_env(file_env: Dict[str, str]) -> Dict[str, str]:
    """
    Merge file env with OS env (OS overrides).
    Only flat key-value, no expansion.
    """
    merged = dict(file_env)
    for k, v in os.environ.items():
        if not k:
            continue
        # If OS has an override, use it
        merged[k] = v
    return merged


# ----------------------------- typed configs -----------------------------------

@dataclass
class RetrievalConfig:
    # Serper (Google Scholar)
    serper_api_key: Optional[str] = None
    serper_endpoint: str = "https://google.serper.dev/scholar"
    serper_timeout_s: int = 12
    default_gl: Optional[str] = None  # country code override if needed

@dataclass
class VLLMServing:
    # OpenAI-compatible vLLM server
    endpoint: str = "http://127.0.0.1:8000/v1"
    api_key: Optional[str] = None     # keep optional for local dev
    model: Optional[str] = None       # e.g., "Qwen/Qwen2.5-14B-Instruct-AWQ"
    request_timeout_s: int = 60
    max_output_tokens: int = 1024

@dataclass
class ParseConfig:
    # PDF/figure parsers
    pdffigures2_jar: Optional[str] = None
    pdffigures2_timeout_s: int = 90
    fitz_dpi: int = 200
    max_figures_per_pdf: int = 128
    # Optional external services
    grobid_url: Optional[str] = None
    tabula_java: Optional[str] = None

@dataclass
class RankConfig:
    # Optional cross-encoder reranker
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: Optional[str] = None
    # Recency/type boosts
    recency_halflife_years: float = 4.0
    type_boost_figure: float = 0.15
    type_boost_table: float = 0.10
    # Stopwords for snippet scoring (comma-separated in env)
    stopwords_extra: List[str] = None  # populated from env

@dataclass
class PlannerConfigEnv:
    use_llm: bool = False
    query_budget: int = 12
    default_lang: str = "en"

@dataclass
class EnrichConfig:
    crossref_enabled: bool = True
    openalex_enabled: bool = True
    semsch_enabled: bool = False
    timeout_s: int = 8
    backoff_base: float = 0.6
    s2_api_key: Optional[str] = None
    crossref_mailto: Optional[str] = None

@dataclass
class AppConfig:
    root: Path
    raw: Dict[str, str]
    retrieval: RetrievalConfig
    vllm: VLLMServing
    parse: ParseConfig
    rank: RankConfig
    planner: PlannerConfigEnv
    enrich: EnrichConfig

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "root": str(self.root),
            "retrieval": asdict(self.retrieval),
            "vllm": asdict(self.vllm),
            "parse": asdict(self.parse),
            "rank": {
                **asdict(self.rank),
                # render list or empty
                "stopwords_extra": self.rank.stopwords_extra or [],
            },
            "planner": asdict(self.planner),
            "enrich": asdict(self.enrich),
        }
        return out


# ----------------------------- map env -> config -------------------------------

def _bool(x: str, default: bool = False) -> bool:
    if x is None:
        return default
    return str(x).strip().lower() not in ("0", "false", "no", "off", "")

def _int(x: str, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _float(x: str, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _list_csv(x: str) -> List[str]:
    if not x:
        return []
    return [t.strip() for t in x.split(",") if t.strip()]

def load_config(project_root: Path | None = None) -> AppConfig:
    """
    Load configuration from citevizor.env + OS env into typed dataclasses.
    No exceptions for missing optional values; required checks via `validate()`.
    """
    root = project_root or discover_project_root()
    file_env = _read_env_file(root)
    env = _merge_env(file_env)

    retrieval = RetrievalConfig(
        serper_api_key=env.get("SERPER_API_KEY"),
        serper_endpoint=env.get("SERPER_ENDPOINT", "https://google.serper.dev/scholar"),
        serper_timeout_s=_int(env.get("SERPER_TIMEOUT_S", ""), 12),
        default_gl=env.get("SERPER_DEFAULT_GL") or None,
    )

    vllm = VLLMServing(
        endpoint=env.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1"),
        api_key=env.get("VLLM_API_KEY") or env.get("OPENAI_API_KEY"),
        model=env.get("VLLM_MODEL") or "Qwen/Qwen2.5-14B-Instruct-AWQ",
        request_timeout_s=_int(env.get("VLLM_TIMEOUT_S", ""), 60),
        max_output_tokens=_int(env.get("VLLM_MAX_TOKENS", ""), 1024),
    )

    parse = ParseConfig(
        pdffigures2_jar=env.get("PDFFIGURES2_JAR") or env.get("PDFFIGURES2_PATH"),
        pdffigures2_timeout_s=_int(env.get("PDFFIGURES2_TIMEOUT_S", ""), 90),
        fitz_dpi=_int(env.get("FITZ_DPI", ""), 200),
        max_figures_per_pdf=_int(env.get("MAX_FIGURES_PER_PDF", ""), 128),
        grobid_url=env.get("GROBID_URL"),
        tabula_java=env.get("TABULA_JAVA"),
    )

    rank = RankConfig(
        reranker_enabled=_bool(env.get("RERANKER_ENABLED", "0")),
        reranker_model=env.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        reranker_device=env.get("RERANKER_DEVICE"),
        recency_halflife_years=_float(env.get("RECENCY_HALFLIFE_YEARS", ""), 4.0),
        type_boost_figure=_float(env.get("TYPE_BOOST_FIGURE", ""), 0.15),
        type_boost_table=_float(env.get("TYPE_BOOST_TABLE", ""), 0.10),
        stopwords_extra=_list_csv(env.get("STOPWORDS_EXTRA", "")),
    )

    planner = PlannerConfigEnv(
        use_llm=_bool(env.get("PLANNER_USE_LLM", "0")),
        query_budget=_int(env.get("PLANNER_QUERY_BUDGET", ""), 12),
        default_lang=env.get("PLANNER_LANG_DEFAULT", "en"),
    )

    enrich = EnrichConfig(
        crossref_enabled=_bool(env.get("ENRICH_CROSSREF_ENABLED", "1")),
        openalex_enabled=_bool(env.get("ENRICH_OPENALEX_ENABLED", "1")),
        semsch_enabled=_bool(env.get("ENRICH_SEMSCHOLAR_ENABLED", "0")),
        timeout_s=_int(env.get("ENRICH_TIMEOUT_S", ""), 8),
        backoff_base=_float(env.get("ENRICH_BACKOFF_BASE", ""), 0.6),
        s2_api_key=env.get("SEMANTIC_SCHOLAR_API_KEY"),
        crossref_mailto=env.get("CROSSREF_MAILTO"),
    )

    return AppConfig(
        root=root,
        raw=env,
        retrieval=retrieval,
        vllm=vllm,
        parse=parse,
        rank=rank,
        planner=planner,
        enrich=enrich,
    )


# ----------------------------- validation & advice -----------------------------

def validate(cfg: AppConfig) -> Tuple[bool, List[str], List[str]]:
    """
    Validate presence of critical keys and provide suggestions.
    Returns: (ok, errors, warnings)
    """
    errors: List[str] = []
    warns: List[str] = []

    # Retrieval key is critical for web search-based flow
    if not cfg.retrieval.serper_api_key:
        errors.append("SERPER_API_KEY is missing (Google Scholar retrieval will fail).")

    # vLLM is local; api_key can be optional. Ensure endpoint looks sane.
    if not cfg.vllm.endpoint.startswith("http"):
        errors.append("VLLM_ENDPOINT must be a valid http(s) URL.")

    # Parse hints
    if not cfg.parse.pdffigures2_jar:
        warns.append("PDFFIGURES2_JAR not set (figure extraction will fallback or be limited).")

    # Reranker optional, but clarify when enabled without model
    if cfg.rank.reranker_enabled and not cfg.rank.reranker_model:
        warns.append("RERANKER_ENABLED=1 but RERANKER_MODEL is empty (will be ignored).")

    return (len(errors) == 0), errors, warns


# ----------------------------- .env template writer ----------------------------

_TEMPLATE = """\
# ================== CiteVizor Environment (english-only comments) ==================
# Minimal keys (retrieval + local vLLM). Comments are intentionally concise.

# --- Retrieval (Google Scholar via Serper) ---
SERPER_API_KEY=
# SERPER_ENDPOINT=https://google.serper.dev/scholar
# SERPER_TIMEOUT_S=12
# SERPER_DEFAULT_GL=us

# --- vLLM (OpenAI-compatible) ---
VLLM_ENDPOINT=http://127.0.0.1:8000/v1
# If your server requires a key, set one; otherwise leave empty.
# VLLM_API_KEY=
# Recommended model for RTX 4090 (4-bit AWQ):
VLLM_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
# VLLM_TIMEOUT_S=60
# VLLM_MAX_TOKENS=1024

# --- Parsing / Figures / Tables ---
# PDFFIGURES2_JAR=C:\\tools\\pdffigures2\\pdffigures2-assembly-0.2.2.jar
# PDFFIGURES2_TIMEOUT_S=90
# FITZ_DPI=200
# MAX_FIGURES_PER_PDF=128
# GROBID_URL=http://127.0.0.1:8070
# TABULA_JAVA=C:\\Program Files\\Java\\bin\\java.exe

# --- Ranking / Reranker (optional) ---
# RERANKER_ENABLED=1
# RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# RERANKER_DEVICE=cuda:0
# RECENCY_HALFLIFE_YEARS=4.0
# TYPE_BOOST_FIGURE=0.15
# TYPE_BOOST_TABLE=0.10
# STOPWORDS_EXTRA=et,al,figure,table,introduction,conclusion

# --- Planner (optional LLM expansion) ---
# PLANNER_USE_LLM=0
# PLANNER_QUERY_BUDGET=12
# PLANNER_LANG_DEFAULT=en

# --- Metadata enrichment (optional internet lookups) ---
# ENRICH_CROSSREF_ENABLED=1
# ENRICH_OPENALEX_ENABLED=1
# ENRICH_SEMSCHOLAR_ENABLED=0
# ENRICH_TIMEOUT_S=8
# ENRICH_BACKOFF_BASE=0.6
# SEMANTIC_SCHOLAR_API_KEY=
# CROSSREF_MAILTO=you@example.com
"""

def write_env_template(path: Path | None = None) -> Path:
    """
    Write an english-only `.env` template; does not overwrite if file exists.
    """
    root = discover_project_root()
    out = (path or (root / ENV_FILE))
    if out.exists():
        return out
    out.write_text(_TEMPLATE, encoding="utf-8")
    return out


# ----------------------------- CLI (optional) ----------------------------------

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="CiteVizor config helper")
    ap.add_argument("--print", action="store_true", help="Print merged configuration as JSON")
    ap.add_argument("--validate", action="store_true", help="Validate required keys")
    ap.add_argument("--write-template", action="store_true", help="Write a starter citevizor.env if missing")
    args = ap.parse_args()

    cfg = load_config()
    if args.write_template:
        p = write_env_template()
        print(f"[OK] Template written (or already present): {p}")

    if args.validate:
        ok, errs, warns = validate(cfg)
        print(f"[VALIDATE] ok={ok}")
        for e in errs:
            print(f"  ERROR: {e}")
        for w in warns:
            print(f"  WARN:  {w}")
        if not ok:
            raise SystemExit(2)

    if args.print:
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
