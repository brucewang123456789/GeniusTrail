from __future__ import annotations

from dotenv import load_dotenv

# Load environment variables before anything else
load_dotenv()

"""
Grok-3 client for Veltraxor.
Implements chat() and stream_chat(); no hidden characters.
"""

import json
import uuid
import logging
from typing import Any, Dict, List, AsyncIterator

import httpx
from config import settings

# ───────────────── logging ─────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("llm-client")


class LLMClient:
    """Minimal wrapper around Grok-3 chat completions."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # Use the provided model/base_url if available, otherwise use the ones from settings
        self.model: str = model or settings.VELTRAX_MODEL
        self.base_url: str = base_url or settings.XAI_API_URL  # base_url set to the correct URL

        # Check for key environment variables
        if not self.base_url:
            log.error("LLMClient initialization failed: XAI_API_URL is not set or empty")
        if not getattr(settings, "XAI_API_KEY", None):
            log.error("LLMClient initialization warning: XAI_API_KEY is not set or empty")
        # Log initialization information for debugging
        log.info("LLMClient init: model=%s, base_url=%s", self.model, self.base_url)

        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {settings.XAI_API_KEY}",
            "Content-Type": "application/json",
        }

    # ────────── one-shot ──────────
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kw: Any,
    ) -> Dict[str, Any]:
        """
        Send a one-shot chat request and return the dict (e.g. {"choices":[...], ...}).
        If the request fails, a stub structure will be returned.
        """
        last_msg = messages[-1].get("content") if messages else None
        log.info(">> entering chat(); last user msg=%r", last_msg)
        payload = {"model": self.model, "messages": messages, "stream": False} | kw
        log.debug("LLMClient.chat payload: %s", payload)
        try:
            r = httpx.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            log.info("<< chat() HTTP %s", r.status_code)
            resp_json = r.json()
            log.debug("LLMClient.chat response JSON: %s", resp_json)
            return resp_json
        except Exception as exc:
            log.error("chat() failed: %s", exc, exc_info=True)
            return self._stub(last_msg or "")

    # ────────── streaming ──────────
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        **kw: Any,
    ) -> AsyncIterator[str]:
        """
        Send a streaming request (stream=True), parse SSE data line by line, and yield content fragments.
        If it fails, yield the stub stream prefix.
        """
        last_msg = messages[-1].get("content") if messages else None
        log.info(">> entering stream_chat(); last user msg=%r", last_msg)
        payload = {"model": self.model, "messages": messages, "stream": True} | kw
        log.debug("LLMClient.stream_chat payload: %s", payload)
        try:
            async with httpx.AsyncClient(timeout=None) as cli:
                async with cli.stream(
                    "POST",
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    log.info("<< stream_chat() HTTP %s", resp.status_code)
                    async for line in resp.aiter_lines():
                        # SSE format: each line starts with "data:"
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            chunk = json.loads(raw)
                        except Exception as e:
                            log.error("Failed to parse stream chunk as JSON: %s; raw=%r", e, raw)
                            continue
                        # Expected structure like OpenAI-style: {"choices":[{"delta":{"content": "..."}}, ...]}
                        choices = chunk.get("choices")
                        if choices and isinstance(choices, list):
                            delta = choices[0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
        except Exception as exc:
            log.error("stream_chat() failed: %s", exc, exc_info=True)
            # Stub stream return: single yield stub prefix
            yield "[stub-stream] " + (last_msg or "")
            return

    # ────────── helper ──────────
    @staticmethod
    def _stub(text: str) -> Dict[str, Any]:
        """Return a dummy completion if the real call fails."""
        return {
            "id": str(uuid.uuid4()),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"[stub] {text}"},
                    "finish_reason": "stop",
                }
            ],
        }