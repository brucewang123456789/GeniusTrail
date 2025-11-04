# C:\CiteVizor\llm\client_vllm.py
# -*- coding: utf-8 -*-
"""
CiteVizor - vLLM OpenAI-compatible client (Force-Completions Edition)

Purpose:
- Eliminate all /v1/chat/completions calls to avoid guided-decoding initialization
  (lm-format-enforcer/outlines/transformers import chain), which caused 500s.
- Always use /v1/completions with a deterministic messages→prompt merger.
- Keep soft-fail behavior so upper layers can fallback without crashing.
- Provide clear health checks, retries with backoff, and small, portable payloads.

Notes:
- Tools / function calling are intentionally unsupported in this edition.
- Streaming is disabled to maximize stability; add later if you need it.
"""

from __future__ import annotations

import json
import os
import random
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator

import requests

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("CITEVIZOR_LOGLEVEL", "INFO"))

ENV_FILE = "citevizor.env"

# ----------------------------- Exceptions -------------------------------------


class LLMUnavailableError(RuntimeError):
    """vLLM endpoint not reachable (connection refused/DNS/timeout)."""


class LLMServerError(RuntimeError):
    """vLLM responded but with 4xx/5xx/429 or malformed payload."""


# ----------------------------- ENV loader -------------------------------------


def _load_env(project_root: Path) -> Dict[str, str]:
    """Load env from citevizor.env then allow OS env to override via common aliases."""
    env_path = project_root / ENV_FILE
    out: Dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")

    aliases = {
        "VLLM_ENDPOINT": [
            "VLLM_ENDPOINT",
            "CITEVIZOR_LLM_API_BASE",
            "CITEVIZOR_LLM_BASE_URL",
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
        ],
        "VLLM_API_KEY": ["VLLM_API_KEY", "CITEVIZOR_LLM_API_KEY", "OPENAI_API_KEY"],
        "VLLM_MODEL": ["VLLM_MODEL", "CITEVIZOR_LLM_MODEL", "OPENAI_MODEL"],
        "VLLM_TIMEOUT_S": ["VLLM_TIMEOUT_S"],
        "VLLM_MAX_TOKENS": ["VLLM_MAX_TOKENS"],
        "VLLM_TEMPERATURE": ["VLLM_TEMPERATURE"],
        "VLLM_TOP_P": ["VLLM_TOP_P"],
        "VLLM_SEED": ["VLLM_SEED"],
        "VLLM_TLS_VERIFY": ["VLLM_TLS_VERIFY"],
        "CITEVIZOR_LLM_SOFTFAIL": ["CITEVIZOR_LLM_SOFTFAIL"],
    }
    for canonical, keys in aliases.items():
        for k in keys:
            if os.getenv(k) is not None:
                out[canonical] = os.getenv(k)  # type: ignore
                break
    return out


def _normalize_endpoint(ep: str) -> str:
    """Ensure base ends with '/v1' (no trailing slash)."""
    base = (ep or "").strip().rstrip("/")
    if not base:
        return "http://127.0.0.1:8000/v1"
    lowered = base.lower()
    if lowered.endswith("/v1"):
        return base
    if "/v1/" in lowered:
        idx = lowered.rfind("/v1")
        return base[: idx + 3]
    return base + "/v1"


# ----------------------------- Data contracts ---------------------------------


@dataclass
class VLLMConfig:
    endpoint: str
    api_key: Optional[str]
    model: str
    timeout_s: int
    max_tokens: int
    temperature: float
    top_p: float
    seed: Optional[int]
    tls_verify: bool
    soft_fail: bool

    @classmethod
    def from_env(cls, project_root: Path) -> "VLLMConfig":
        env = _load_env(project_root)
        endpoint_raw = env.get("VLLM_ENDPOINT") or "http://127.0.0.1:8000"
        endpoint = _normalize_endpoint(endpoint_raw)
        api_key = env.get("VLLM_API_KEY")
        model = env.get("VLLM_MODEL") or "Qwen/Qwen2.5-14B-Instruct-AWQ"
        timeout_s = int(env.get("VLLM_TIMEOUT_S", "60"))
        max_tokens = int(env.get("VLLM_MAX_TOKENS", "1024"))
        temperature = float(env.get("VLLM_TEMPERATURE", "0.2"))
        top_p = float(env.get("VLLM_TOP_P", "0.9"))
        seed = int(env["VLLM_SEED"]) if env.get("VLLM_SEED") else None
        tls_verify = not (str(env.get("VLLM_TLS_VERIFY", "1")).strip() == "0")
        soft_fail = str(env.get("CITEVIZOR_LLM_SOFTFAIL", "0")).lower() in ("1", "true", "yes")
        return cls(endpoint, api_key, model, timeout_s, max_tokens, temperature, top_p, seed, tls_verify, soft_fail)


@dataclass
class ChatResult:
    text: str
    usage: Dict[str, Any]
    finish_reason: Optional[str]
    tool_calls: List[Dict[str, Any]]
    raw: Dict[str, Any]


# ----------------------------- Client impl ------------------------------------


class VLLMClient:
    """
    Force-Completions client:
    - Always call /v1/completions to avoid guided-decoding backends on the server.
    - Convert OpenAI-style "messages" into a plain prompt deterministically.
    """

    def __init__(self, cfg: VLLMConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if self.cfg.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.cfg.api_key}"})
        self.session.verify = self.cfg.tls_verify
        self._last_ok_ts: float = 0.0
        self._ok_ttl_s: float = 15.0

    # --------------------- public methods ---------------------

    def health_check(self) -> Dict[str, Any]:
        """GET /v1/models to verify server is up."""
        url = self._url("/models")
        try:
            r = self.session.get(url, timeout=min(5, self.cfg.timeout_s))
            if r.status_code in (401, 403):
                raise LLMServerError(f"LLM auth failed ({r.status_code}). Check API key.")
            r.raise_for_status()
            data = r.json()
            self._last_ok_ts = time.time()
            return data
        except requests.ConnectTimeout as e:
            raise LLMUnavailableError(f"LLM health_check timeout at {url}: {e}") from e
        except requests.ConnectionError as e:
            raise LLMUnavailableError(f"LLM health_check connect error at {url}: {e}") from e
        except requests.RequestException as e:
            raise LLMServerError(f"LLM health_check error at {url}: {e}") from e
        except ValueError as e:
            raise LLMServerError(f"LLM health_check returned non-JSON from {url}: {e}") from e

    def wait_until_ready(self, max_wait_s: int = 8) -> bool:
        """Poll /models until ready or timeout; returns True if healthy."""
        deadline = time.time() + max_wait_s
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            try:
                self.health_check()
                return True
            except (LLMUnavailableError, LLMServerError):
                self._sleep_backoff(attempt, base=0.35, cap=2.0)
        return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 3,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Non-streaming call implemented on top of /v1/completions.
        Ignores tools/function calling by design.
        """
        self._ensure_ready()

        prompt = self._messages_to_prompt(messages)
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "prompt": prompt,
            "temperature": self._pick(temperature, self.cfg.temperature),
            "top_p": self._pick(top_p, self.cfg.top_p),
            "max_tokens": self._pick(max_tokens, self.cfg.max_tokens),
        }
        if self.cfg.seed is not None:
            payload["seed"] = self.cfg.seed

        url = self._url("/completions")
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.post(url, data=json.dumps(payload), timeout=self.cfg.timeout_s)
                if resp.status_code in (401, 403):
                    return self._maybe_softfail_error(f"LLM auth failed ({resp.status_code}). Check API key.", resp)
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < retries:
                        self._sleep_backoff(attempt)
                        continue
                    return self._maybe_softfail_error(f"LLM server error {resp.status_code}: {resp.text[:256]}", resp)
                resp.raise_for_status()
                data = resp.json()
                return self._normalize_completions_response(data)
            except requests.ConnectionError as e:
                if attempt < retries:
                    self._sleep_backoff(attempt)
                    continue
                return self._maybe_softfail_error(f"LLM unreachable at {url}: {e}")
            except requests.Timeout as e:
                if attempt < retries:
                    self._sleep_backoff(attempt)
                    continue
                return self._maybe_softfail_error(f"LLM timeout at {url}: {e}")
            except requests.RequestException as e:
                if attempt < retries:
                    self._sleep_backoff(attempt)
                    continue
                return self._maybe_softfail_error(f"LLM request failed at {url}: {e}")
            except ValueError as e:
                if attempt < retries:
                    self._sleep_backoff(attempt)
                    continue
                return self._maybe_softfail_error(f"LLM returned non-JSON payload: {e}")

        return self._maybe_softfail_error("vLLM completions request failed after retries")

    def chat_stream(self, *_a, **_kw) -> Generator[str, None, ChatResult]:
        """Streaming is intentionally disabled in this edition."""
        raise NotImplementedError("Streaming is disabled in Force-Completions client.")

    # --------------------- internals ---------------------

    def _ensure_ready(self) -> None:
        if (time.time() - self._last_ok_ts) < self._ok_ttl_s:
            return
        self.health_check()

    def _normalize_completions_response(self, data: Dict[str, Any]) -> ChatResult:
        """Map /v1/completions response to ChatResult."""
        choices = data.get("choices") or [{}]
        ch = choices[0] or {}
        text = ch.get("text") or ""
        usage = data.get("usage") or {}
        finish_reason = ch.get("finish_reason")
        return ChatResult(text=text, usage=usage, finish_reason=finish_reason, tool_calls=[], raw=data)

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Deterministic merge:
          [SYSTEM]\n...\n\n[USER]\n...\n\n[ASSISTANT]\n...\n
        This keeps enough structure while staying pure text.
        """
        parts: List[str] = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                parts.append(f"[SYSTEM]\n{content}\n")
            elif role == "user":
                parts.append(f"[USER]\n{content}\n")
            else:
                parts.append(f"[{role.upper()}]\n{content}\n")
        return "\n".join(parts).strip()

    def _url(self, path: str) -> str:
        base = self.cfg.endpoint.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    @staticmethod
    def _pick(v: Optional[Any], fallback: Any) -> Any:
        return fallback if v is None else v

    @staticmethod
    def _sleep_backoff(attempt: int, base: float = 0.6, cap: float = 8.0) -> None:
        delay = min(cap, base * (2 ** (attempt - 1))) + random.uniform(0, 0.4)
        time.sleep(delay)

    def _maybe_softfail_error(self, message: str, resp: Optional[requests.Response] = None) -> ChatResult:
        """Return ChatResult error when soft-fail is enabled; otherwise raise."""
        if not self.cfg.soft_fail:
            if resp is not None and 400 <= resp.status_code < 500:
                raise LLMServerError(message)
            raise LLMUnavailableError(message)
        logger.error("[LLM] soft-fail: %s", message)
        raw = {
            "error": message,
            "status": getattr(resp, "status_code", None),
            "text": (getattr(resp, "text", None)[:256] if resp is not None else None),
        }
        return ChatResult(text="", usage={}, finish_reason="error", tool_calls=[], raw=raw)


# ----------------------------- CLI smoke test ---------------------------------


if __name__ == "__main__":
    """
    Quick test:
      1) Start vLLM OpenAI server, e.g.:
         python -m vllm.entrypoints.openai.api_server \
           --model Qwen/Qwen2.5-14B-Instruct-AWQ --port 8000
      2) Set env (any alias works):
           CITEVIZOR_LLM_API_BASE / OPENAI_API_BASE / OPENAI_BASE_URL / VLLM_ENDPOINT
           and CITEVIZOR_LLM_API_KEY / OPENAI_API_KEY / VLLM_API_KEY if required.
      3) Run:  python llm/client_vllm.py "Say one short sentence."
    """
    project_root = Path(__file__).resolve().parents[1]
    cfg = VLLMConfig.from_env(project_root)
    print("[cfg] endpoint:", cfg.endpoint, "model:", cfg.model)
    client = VLLMClient(cfg)
    ok = client.wait_until_ready(max_wait_s=6)
    print("[Health] ready:", ok)
    msg = [{"role": "user", "content": "Say hi in one short sentence."}]
    try:
        res = client.chat(msg)
        print("[Text]", (res.text or res.raw))
        print("[Usage]", res.usage)
    except (LLMUnavailableError, LLMServerError) as e:
        print("[Error]", e)
