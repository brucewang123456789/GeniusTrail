# C:\CiteVizor\planner\plan_query.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Query planner for academic retrieval (Google Scholar-first)

Goals:
  - Turn a user's research question into a compact set of high-recall queries.
  - Prefer rule-based expansions (deterministic) with optional LLM augmentation.
  - Normalize years ("2019-2024", ">=2021", "last 5 years", "近三年") into schemas.YearRange.
  - Persist a SearchPlan under storage/plans/<plan_id>.json to be consumed by Serper client.

LLM (optional):
  - When PLANNER_USE_LLM=1, use local vLLM OpenAI-compatible server with model:
    Qwen/Qwen2.5-14B-Instruct-AWQ  (https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ)
  - The LLM adds candidate queries; final selection is still de-duplicated and budgeted.

This script does NOT require any new env variables to run;
optional planner keys are read if present.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from schemas import STORAGE_DIR, SearchPlan, YearRange  # project contracts
# Optional vLLM client (only used if PLANNER_USE_LLM=1 or --use-llm)
try:
    from llm.client_vllm import VLLMClient, VLLMConfig  # type: ignore
except Exception:
    VLLMClient = None  # type: ignore
    VLLMConfig = None  # type: ignore

ENV_FILE = "citevizor.env"

# ----------------------------- env loader -------------------------------------

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
    # OS env overrides
    for k in ("PLANNER_USE_LLM", "PLANNER_QUERY_BUDGET", "PLANNER_LANG_DEFAULT"):
        if os.getenv(k) is not None:
            out[k] = os.getenv(k)  # type: ignore
    return out

# ----------------------------- language utils ---------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7ff]")

def detect_lang(text: str, default: str = "en") -> str:
    if _CJK_RE.search(text or ""):
        # Heuristic: prefer zh if Han chars; JA if KANA present; keep simple.
        if re.search(r"[\u3040-\u30ff]", text):
            return "ja"
        return "zh"
    return default or "en"

# ----------------------------- year parsing -----------------------------------

def parse_year_range(s: str) -> Optional[YearRange]:
    """
    Parse "2019-2024", ">=2021", "<=2020", "last 5 years", "past 3 years", "近三年", "近5年".
    """
    if not s:
        return None
    s = s.strip().lower()
    now = time.gmtime().tm_year

    # 2019-2024
    m = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if 1900 <= a <= now and 1900 <= b <= now and a <= b:
            return YearRange(start=a, end=b)

    # >=2021 or <=2020
    m = re.match(r"^\s*(>=|<=)\s*(\d{4})\s*$", s)
    if m:
        op, y = m.group(1), int(m.group(2))
        if 1900 <= y <= now:
            return YearRange(start=y, end=None) if op == ">=" else YearRange(start=None, end=y)

    # last/past N years
    m = re.search(r"(last|past)\s+(\d+)\s+years", s)
    if m:
        n = int(m.group(2))
        if 1 <= n <= 50:
            return YearRange(start=now - n, end=now)

    # Chinese "近3年"/"近五年"
    m = re.search(r"近\s*([0-9一二三四五六七八九十]+)\s*年", s)
    if m:
        num = m.group(1)
        map_cn = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
        if num.isdigit():
            n = int(num)
        else:
            # very small parser for <= 20
            n = 0
            for ch in num:
                n = n * 10 + map_cn.get(ch, 0)
            if n == 0:
                n = 3
        n = max(1, min(50, n))
        return YearRange(start=now - n, end=now)

    return None

# ----------------------------- query helpers ----------------------------------

_STOPWORDS = {
    "en": set("latest current update updated new novel recent review survey overview trends trend research study studies".split()),
    "zh": set(list("最新 近期 近年 研究 综述 调研 趋势 进展 概述".split())),
    "ja": set(list("最新 最近 近年 レビュー サーベイ 概観 概要 動向".split())),
}

# Domain-agnostic expansion lexicon (kept small and conservative)
_SUFFIXES = {
    "en": ["review", "meta-analysis", "survey", "systematic review", "state of the art"],
    "zh": ["综述", "系统综述", "元分析", "进展", "综述性研究"],
    "ja": ["レビュー", "サーベイ", "メタアナリシス", "総説"],
}
_NEUTRAL_TWEAKS = {
    "en": ["performance", "degradation", "stability", "design", "materials", "modeling"],
    "zh": ["性能", "失效", "稳定性", "设计", "材料", "建模"],
    "ja": ["性能", "劣化", "安定性", "設計", "材料", "モデリング"],
}

def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", (text or "").strip())
    t = re.sub(r"\s+", " ", t)
    return t

def _drop_leading_fluff(q: str, lang: str) -> str:
    toks = q.split()
    if lang == "en":
        toks = [t for t in toks if t.lower() not in _STOPWORDS["en"]]
    return " ".join(toks) if toks else q

def expand_rule_based(user_query: str, lang: str) -> List[str]:
    """
    Deterministic expansions:
      - base, quoted phrase, suffix variants, neutral tweaks.
      - Avoid over-expansion; keep ≤ 10-12 suggestions pre-dedup.
    """
    q = _normalize(user_query)
    base = _drop_leading_fluff(q, lang)
    out: List[str] = []

    # Base and quoted
    out.append(base)
    if len(base.split()) > 1:
        out.append(f"\"{base}\"")

    # Suffixes
    for suf in _SUFFIXES.get(lang, _SUFFIXES["en"]):
        out.append(f"{base} {suf}")

    # Neutral tweaks (only add a few)
    for t in _NEUTRAL_TWEAKS.get(lang, _NEUTRAL_TWEAKS["en"])[:3]:
        out.append(f"{base} {t}")

    # Simple alias: "PV" <-> "photovoltaic"
    if lang == "en":
        if re.search(r"\bphotovoltaic(s)?\b", base, re.I):
            out.append(re.sub(r"(?i)photovoltaic(s)?", "PV", base))
        if re.search(r"\bPV\b", base):
            out.append(re.sub(r"\bPV\b", "photovoltaic", base, flags=re.I))

    # Dedup (case-insensitive)
    uniq = []
    seen = set()
    for s in out:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq[:14]

# ----------------------------- optional LLM -----------------------------------

@dataclass
class PlannerLLM:
    enabled: bool
    client: Optional[VLLMClient]

    @classmethod
    def from_env(cls, project_root: Path) -> "PlannerLLM":
        env = _load_env(project_root)
        use = (env.get("PLANNER_USE_LLM", "0") not in ("0", "false", "False"))
        if not use or VLLMClient is None or VLLMConfig is None:
            return cls(False, None)
        return cls(True, VLLMClient(VLLMConfig.from_env(project_root)))

    def propose(self, user_query: str, lang: str, budget: int = 12) -> List[str]:
        """
        Ask the local model to propose diversified, Scholar-ready queries.
        """
        if not self.enabled or not self.client:
            return []
        sys_prompt = (
            "You are an academic search query planner. "
            "Propose concise Google Scholar queries (no explanations, no bullets). "
            "Prefer exact phrases in quotes, add a few 'review|survey|meta-analysis' variants, "
            "and avoid site: filters. Return a JSON array of strings only."
        )
        user = f"Language: {lang}\nUser question: {user_query}\nReturn up to {budget} queries."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user}
        ]
        res = self.client.chat(messages, max_tokens=400)
        text = res.text.strip()
        # Extract JSON array
        m = re.search(r"\[[\s\S]*\]$", text)
        try:
            arr = json.loads(m.group(0) if m else text)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            return []
        return []

# ----------------------------- plan builder -----------------------------------

def _budgeted_merge(rule_qs: List[str], llm_qs: List[str], budget: int) -> List[str]:
    out: List[str] = []
    seen = set()
    # Interleave for diversity
    i = j = 0
    while len(out) < budget and (i < len(rule_qs) or j < len(llm_qs)):
        if i < len(rule_qs):
            k = rule_qs[i].lower()
            if k not in seen:
                seen.add(k); out.append(rule_qs[i])
            i += 1
        if len(out) >= budget:
            break
        if j < len(llm_qs):
            k = llm_qs[j].lower()
            if k not in seen:
                seen.add(k); out.append(llm_qs[j])
            j += 1
    return out

def _plan_id(queries: List[str], yr: Optional[YearRange]) -> str:
    sig = "\n".join(queries) + "|" + (json.dumps(yr.__dict__) if yr else "")
    return sha256(sig.encode("utf-8")).hexdigest()[:12]

def _save_plan(plan: SearchPlan, project_root: Path) -> Path:
    d = STORAGE_DIR / "plans"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{plan.plan_id}.json"
    payload = {
        "plan_id": plan.plan_id,
        "queries": plan.queries,
        "k_top": plan.k_top,
        "filters": plan.filters,
        "created_at": int(time.time()),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

# ----------------------------- public API -------------------------------------

@dataclass
class PlannerConfig:
    project_root: Path
    default_lang: str = "en"
    query_budget: int = 12
    use_llm: bool = False

    @classmethod
    def from_env(cls, project_root: Path) -> "PlannerConfig":
        env = _load_env(project_root)
        return cls(
            project_root=project_root,
            default_lang=(env.get("PLANNER_LANG_DEFAULT") or "en"),
            query_budget=int(env.get("PLANNER_QUERY_BUDGET", "12")),
            use_llm=(env.get("PLANNER_USE_LLM", "0") not in ("0", "false", "False")),
        )

class QueryPlanner:
    def __init__(self, cfg: PlannerConfig, llm: Optional[PlannerLLM] = None):
        self.cfg = cfg
        self.llm = llm or PlannerLLM.from_env(cfg.project_root)

    def build_plan(
        self,
        user_query: str,
        lang: str = "auto",
        years: Optional[str] = None,
        k_top: int = 12,
    ) -> Tuple[SearchPlan, Path]:
        # Resolve language
        lang = detect_lang(user_query, default=self.cfg.default_lang) if lang == "auto" else lang

        # Parse year range (optional)
        yr = parse_year_range(years) if years else None

        # Rule-based expansions
        rb_qs = expand_rule_based(user_query, lang)

        # Optional LLM expansions
        llm_qs = []
        if self.cfg.use_llm and self.llm.enabled:
            llm_qs = self.llm.propose(user_query, lang, budget=self.cfg.query_budget)

        # Merge under budget
        budget = max(6, min(24, self.cfg.query_budget))
        queries = _budgeted_merge(rb_qs, llm_qs, budget)

        # Build SearchPlan (filters keep years; add lang hint for retrieval if needed)
        filters: Dict[str, Any] = {"years": yr.__dict__ if yr else {}}
        plan_id = _plan_id(queries, yr)
        plan = SearchPlan(queries=queries, k_top=k_top, filters=filters, plan_id=plan_id)  # type: ignore

        # Persist
        path = _save_plan(plan, self.cfg.project_root)
        return plan, path

# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor academic query planner")
    ap.add_argument("--query", required=True, help="User research question")
    ap.add_argument("--lang", default="auto", help="auto|en|zh|ja")
    ap.add_argument("--years", default="", help='Year filter e.g. "2019-2024", ">=2021", "last 5 years", "近三年"')
    ap.add_argument("--k-top", type=int, default=12, help="k_top to request from retrieval stage")
    ap.add_argument("--use-llm", action="store_true", help="Enable LLM-based expansions (overrides env)")
    return ap.parse_args()

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    cfg = PlannerConfig.from_env(project_root)
    args = _parse_args()
    if args.use_llm:
        cfg.use_llm = True

    planner = QueryPlanner(cfg)
    plan, path = planner.build_plan(
        user_query=args.query,
        lang=args.lang,
        years=(args.years or None),
        k_top=args.k_top,
    )
    print(f"[OK] SearchPlan -> {path}")
    print(f"  plan_id={plan.plan_id}  queries={len(plan.queries)}  k_top={plan.k_top}  years={plan.filters.get('years')}")
    for i, q in enumerate(plan.queries, 1):
        print(f"   {i:02d}. {q}")
