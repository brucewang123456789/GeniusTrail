# C:\CiteVizor\llm\summarize_engineer.py
# -*- coding: utf-8 -*-
"""
CiteVizor - Engineer-oriented summarizer (hardened)
- Input : citation map JSON from align/citation_map.py (storage/evidence/_rank/<run_id>.citations.json)
- LLM   : calls local vLLM OpenAI-compatible server via client_vllm.py (no guided decoding knobs)
- Output: schemas.TextBlocks (highlights + refined figure_captions), persisted to .../<run_id>.summary.json
- Design: strictly evidence-grounded; every highlight ends with labels like [3 p.7, Fig.2]
- Robust: LLM errors/500/timeouts auto-fallback to rule-based extractive summary (no model).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from schemas import STORAGE_DIR, EVIDENCE_DIR, TextBlocks  # data contracts
from llm.client_vllm import VLLMClient, VLLMConfig        # LLM client

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
    # OS env override
    for k in (
        "SUMMARIZER_LANG",
        "SUMMARIZER_TOPK",
        "SUMMARIZER_MAX_HIGHLIGHTS",
        "SUMMARIZER_JSON_STRICT",
        "CITEVIZOR_SUMMARY_FALLBACK",
    ):
        if os.getenv(k) is not None:
            out[k] = os.getenv(k)  # type: ignore
    return out

# ----------------------------- IO helpers -------------------------------------

def _rank_dir() -> Path:
    d = EVIDENCE_DIR / "_rank"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _load_citations(run_id: Optional[str], path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        if not run_id:
            raise ValueError("Either --run-id or --citations-path must be provided.")
        path = _rank_dir() / f"{run_id}.citations.json"
    if not path.exists():
        raise FileNotFoundError(f"Citation map not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def _save_summary(run_id: str, payload: Dict[str, Any]) -> Path:
    out_path = _rank_dir() / f"{run_id}.summary.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

# ----------------------------- formatting for LLM ------------------------------

def _fmt_evidence_items(items: List[Dict[str, Any]]) -> str:
    """
    Compact evidence section fed to the LLM.
    """
    lines = []
    for it in items:
        lab = it.get("label") or f"[{it.get('ref_id')}]"
        typ = it.get("type")
        pg = f" p.{it['page']}" if it.get("page") else ""
        fig = f" Fig.{it['figure_no']}" if it.get("figure_no") else ""
        tab = f" Table.{it['table_no']}" if it.get("table_no") else ""
        snippet = (it.get("snippet") or "").replace("\n", " ").strip()
        snippet = re.sub(r"\s+", " ", snippet)[:600]
        lines.append(f"{lab} ({typ}{pg}{fig}{tab}): {snippet}")
    return "\n".join(lines)

def _fmt_refs(refs: List[Dict[str, Any]]) -> str:
    """
    Minimal references section fed to the LLM.
    """
    lines = []
    for r in refs:
        rid = r.get("ref_id")
        title = r.get("title") or "<untitled>"
        year = r.get("year") or ""
        doi = r.get("doi") or ""
        url = r.get("url") or ""
        venue = r.get("venue") or ""
        bits = [f"[{rid}] {title}"]
        if venue:
            bits.append(venue)
        if year:
            bits.append(str(year))
        if doi:
            bits.append(f"doi:{doi}")
        elif url:
            bits.append(url)
        lines.append(", ".join(bits))
    return "\n".join(lines)

# ----------------------------- config -----------------------------------------

@dataclass
class SummarizerConfig:
    project_root: Path
    lang: str = "en"
    topk: int = 20
    max_highlights: int = 8
    json_strict: bool = True
    fallback_on: bool = True  # new: enable rule-based fallback

    @classmethod
    def from_env(cls, project_root: Path) -> "SummarizerConfig":
        env = _load_env(project_root)
        return cls(
            project_root=project_root,
            lang=(env.get("SUMMARIZER_LANG") or "en").lower(),
            topk=int(env.get("SUMMARIZER_TOPK", "20")),
            max_highlights=int(env.get("SUMMARIZER_MAX_HIGHLIGHTS", "8")),
            json_strict=(env.get("SUMMARIZER_JSON_STRICT", "1") not in ("0", "false", "False")),
            fallback_on=(env.get("CITEVIZOR_SUMMARY_FALLBACK", "1") not in ("0", "false", "False")),
        )

# ----------------------------- fallback (no-LLM) ------------------------------

def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\-\u4e00-\u9fff]+", text.lower())

def _idf_scores(docs: List[List[str]]) -> Dict[str, float]:
    df = Counter()
    for toks in docs:
        df.update(set(toks))
    N = max(1, len(docs))
    return {w: math.log(1.0 + N / (1 + c)) for w, c in df.items()}

def _topk_sentences(text: str, k: int = 2) -> List[str]:
    sents = re.split(r'(?<=[。．\.!?])\s+', text)
    toks_per_sent = [_tokenize_words(s) for s in sents]
    idf = _idf_scores(toks_per_sent)
    scored: List[Tuple[float, int]] = []
    for i, toks in enumerate(toks_per_sent):
        tf = Counter(toks)
        score = sum(tf[w] * idf.get(w, 0.0) for w in tf)
        scored.append((score, i))
    keep_idx = [i for _, i in sorted(scored, reverse=True)[:k] if i < len(sents)]
    out = [sents[i].strip() for i in keep_idx if sents[i].strip()]
    return out

def _labels_for_item(it: Dict[str, Any]) -> str:
    lab = it.get("label") or f"[{it.get('ref_id')}]"
    parts = [lab]
    # enrich page/figure/table if available (keeps your style)
    trail = []
    if it.get("page"):
        trail.append(f"p.{it['page']}")
    if it.get("figure_no"):
        trail.append(f"Fig.{it['figure_no']}")
    if it.get("table_no"):
        trail.append(f"Table.{it['table_no']}")
    if trail:
        parts[-1] = parts[-1][:-1] + " " + ", ".join(trail) + "]"
    return parts[0]

def _fallback_engineer_summary(
    query: str,
    items: List[Dict[str, Any]],
    refs: List[Dict[str, Any]],
    lang: str,
    max_highlights: int,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Build engineer-oriented bullets and refined figure captions without any LLM.
    - Bullets: top-K evidence snippets/titles, each ends with label(s).
    - Figure captions: for figure-type evidence, concise caption ending with label.
    """
    bullets: List[str] = []
    figcaps: Dict[str, str] = {}

    # Rank by simple informativeness: prefer figures/tables > text; longer snippet a bit higher.
    def _score(it: Dict[str, Any]) -> float:
        typ = (it.get("type") or "").lower()
        bonus = 2.0 if "figure" in typ else (1.5 if "table" in typ else 1.0)
        sn = (it.get("snippet") or "")
        return bonus * (1.0 + min(400, len(sn)) / 200.0)

    ranked = sorted(items, key=_score, reverse=True)[:max_highlights]

    # Build bullets
    for it in ranked:
        lab = _labels_for_item(it)
        title = it.get("title") or it.get("name") or ""
        note = it.get("snippet") or it.get("caption") or ""
        text = f"{title}. {note}".strip(". ").strip()
        # pick at most two salient sentences
        lines = _topk_sentences(text, k=2) or ([note.strip()] if note else [])
        if not lines:
            continue
        first = lines[0]
        # normalize spaces
        first = re.sub(r"\s+", " ", first)[:300]
        bullets.append(f"{first} {lab}".strip())

    if not bullets:
        # absolute fallback
        bullets = [f"No LLM summary; extracted {len(items)} evidence items for query: {query}"]

    # Figure captions
    for it in items:
        typ = (it.get("type") or "").lower()
        if "figure" in typ:
            evid = it.get("id") or it.get("evidence_id") or it.get("ref_id") or None
            if not evid:
                continue
            lab = _labels_for_item(it)
            base = it.get("caption") or it.get("snippet") or it.get("title") or "Figure"
            base = re.sub(r"\s+", " ", base).strip()
            # concise
            base = (base[:180] + "…") if len(base) > 180 else base
            figcaps[str(evid)] = f"{base} {lab}".strip()

    return bullets[:max_highlights], figcaps

# ----------------------------- core summarizer --------------------------------

class EngineerSummarizer:
    def __init__(self, cfg: SummarizerConfig, llm: VLLMClient):
        self.cfg = cfg
        self.llm = llm

    def summarize(self, citations_json: Dict[str, Any], topk: Optional[int] = None) -> TextBlocks:
        items = citations_json.get("items", []) or []
        refs = citations_json.get("refs", []) or []
        query = citations_json.get("query", "") or ""
        run_id = citations_json.get("run_id") or "unknown"

        # Truncate to configured top-K
        K = min(len(items), topk or self.cfg.topk)
        items = items[:K]

        # -------------- Attempt LLM path (no guided/response_format) --------------
        sys_prompt = self._system_prompt(self.cfg.lang, self.cfg.max_highlights)
        evidence_text = _fmt_evidence_items(items)
        refs_text = _fmt_refs(refs)
        json_schema = (
            "{\n"
            '  "highlights": ["<bullet ending with one or more labels like [3 p.7, Fig.2]>", "..."],\n'
            '  "figure_captions": [{"evidence_id":"<id of figure evidence>","caption":"<short refined caption ending with a label>"}]\n'
            "}\n"
        )

        user_prompt = (
            f"User query:\n{query}\n\n"
            "Evidence:\n"
            "----------------------------------------\n"
            f"{evidence_text}\n"
            "----------------------------------------\n\n"
            "References:\n"
            f"{refs_text}\n\n"
            "Return JSON only matching this schema:\n"
            f"{json_schema}"
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = self.llm.chat(messages)  # client 已统一“净化 payload”与可选软失败
        text = (result.text or "").strip()
        data = self._parse_json_block(text, strict=self.cfg.json_strict)

        ok_json = isinstance(data, dict) and isinstance(data.get("highlights"), list)
        llm_h = (data.get("highlights") if ok_json else None) or []
        llm_fc_list = (data.get("figure_captions") if ok_json else None) or []

        # -------------- Fallback if LLM fails/500/non-JSON --------------
        need_fallback = (
            not ok_json or
            not llm_h or
            (isinstance(result.finish_reason, str) and result.finish_reason == "error") or
            (isinstance(result.raw, dict) and result.raw.get("error"))
        )

        if need_fallback and self.cfg.fallback_on:
            fb_highlights, fb_figcaps = _fallback_engineer_summary(
                query=query,
                items=items,
                refs=refs,
                lang=self.cfg.lang,
                max_highlights=self.cfg.max_highlights,
            )
            highlights = fb_highlights
            figure_captions = fb_figcaps
        else:
            # LLM path accepted → normalize to dict {evidence_id: caption}
            highlights = [str(h) for h in llm_h][: self.cfg.max_highlights]
            figure_captions: Dict[str, str] = {}
            if isinstance(llm_fc_list, list):
                for fc in llm_fc_list:
                    if isinstance(fc, dict) and fc.get("evidence_id") and fc.get("caption"):
                        figure_captions[str(fc["evidence_id"])] = str(fc["caption"])

            # If LLM forgot labels, minimally append top item's label to last bullet
            if highlights and not highlights[-1].strip().endswith("]") and items:
                highlights[-1] = highlights[-1].rstrip(". ") + " " + _labels_for_item(items[0])

        # -------------- Build & persist ----------------
        tb = TextBlocks(
            highlights=highlights,
            figure_captions=figure_captions,
            chart_codes=[],  # chart code will be produced later
        )

        payload = {
            "run_id": run_id,
            "query": query,
            "lang": self.cfg.lang,
            "max_highlights": self.cfg.max_highlights,
            "highlights": tb.highlights,
            "figure_captions": tb.figure_captions,
            "llm_usage": result.usage,
            "model": self.llm.cfg.model,
            "llm_error": (result.raw.get("error") if isinstance(result.raw, dict) else None),
            "fallback_used": bool(need_fallback and self.cfg.fallback_on),
        }
        _save_summary(run_id, payload)
        return tb

    # ------------------ helpers ------------------

    @staticmethod
    def _system_prompt(lang: str, max_highlights: int) -> str:
        return (
            "You are an engineering-focused scientific summarizer named CiteVizor.\n"
            "Rules:\n"
            f"1) Write in '{lang}'. Keep {max_highlights} bullets or fewer, concise and actionable.\n"
            "2) Use only the provided Evidence and References; do NOT invent facts.\n"
            "3) Every bullet MUST end with one or more labels like [3 p.7, Fig.2] that point to evidence.\n"
            "4) Focus on variables → setup/device → metrics → operating ranges → constraints.\n"
            "5) Prefer figures/tables for claims; if evidence is insufficient, write 'Insufficient evidence [ref]'.\n"
            "6) Output JSON ONLY matching the requested schema—no prose outside JSON.\n"
        )

    @staticmethod
    def _parse_json_block(text: str, strict: bool = True) -> Dict[str, Any]:
        """
        Try parsing JSON as-is; then extract the largest {...} block fallback.
        """
        try:
            return json.loads(text)
        except Exception:
            if strict:
                m = re.search(r"\{[\s\S]*\}\s*$", text)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        return {}
                return {}
            matches = re.findall(r"\{[\s\S]*?\}", text)
            for m in reversed(matches):
                try:
                    return json.loads(m)
                except Exception:
                    continue
            return {}

# ----------------------------- CLI --------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CiteVizor engineer summarizer")
    ap.add_argument("--run-id", help="Rank run id used to locate <run_id>.citations.json", default="")
    ap.add_argument("--citations-path", help="Explicit path to citations json", default="")
    ap.add_argument("--lang", help="en|zh|ja; overrides env", default="")
    ap.add_argument("--topk", type=int, default=0, help="Override number of evidence items to feed")
    return ap.parse_args()

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    s_cfg = SummarizerConfig.from_env(project_root)
    v_cfg = VLLMConfig.from_env(project_root)
    client = VLLMClient(v_cfg)
    summarizer = EngineerSummarizer(s_cfg, client)

    args = _parse_args()
    if args.lang:
        summarizer.cfg.lang = args.lang.lower()
    if args.topk and args.topk > 0:
        summarizer.cfg.topk = args.topk

    run_id = args.run_id or None
    citations_path = Path(args.citations_path) if args.citations_path else None
    citations = _load_citations(run_id, citations_path)

    tb = summarizer.summarize(citations)
    out_path = _rank_dir() / f"{citations.get('run_id','unknown')}.summary.json"
    print(f"[OK] Summary -> {out_path}")
    print("Highlights:")
    for i, h in enumerate(tb.highlights, 1):
        print(f"  {i:02d}. {h}")
