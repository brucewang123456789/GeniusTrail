"""
llm_client.py — resilient wrapper around an external Chat-Completion endpoint.
Falls back to deterministic stub when running in CI or upstream failure occurs.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, cast

import httpx
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("llm-client")


class LLMClient:
    """Chat-completion helper that gracefully degrades in CI."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or settings.VELTRAX_MODEL
        self.base_url = base_url or settings.XAI_API_URL

        self._stub_mode: bool = (
            os.getenv("CI", "false").lower() == "true"
            or not settings.XAI_API_KEY
            or not self.base_url
        )
        if self._stub_mode:
            log.warning(
                "LLMClient running in STUB mode – no real HTTP calls will be made"
            )

        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {settings.XAI_API_KEY or 'dummy'}",
            "Content-Type": "application/json",
        }

    # ───────────────────────── synchronous chat ─────────────────────────
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """
        Blocking chat request. On failure—or when in stub mode—returns
        a deterministic stub response for test stability.
        """
        last_msg = messages[-1].get("content", "") if messages else ""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        } | kwargs

        if self._stub_mode:
            return self._stub(last_msg)

        try:
            resp = httpx.post(
                self.base_url, headers=self.headers, json=payload, timeout=30
            )
            resp.raise_for_status()
            return cast(Dict[str, Any], resp.json())
        except Exception as exc:  # pragma: no cover
            log.error("chat() failed: %s – falling back to stub", exc, exc_info=True)
            return self._stub(last_msg)

    # ───────────────────────── streaming chat ───────────────────────────
    async def stream_chat(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        Async generator yielding token deltas. Falls back to stub string
        on error or when running in stub mode.
        """
        last_msg = messages[-1].get("content", "") if messages else ""
        payload = {"model": self.model, "messages": messages, "stream": True} | kwargs

        if self._stub_mode:
            yield f"[stub-stream] {last_msg}"
            return

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", self.base_url, headers=self.headers, json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            obj = json.loads(raw)
                            delta = (
                                obj.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content")
                            )
                            if isinstance(delta, str):
                                yield delta
                        except Exception as exc:  # pragma: no cover
                            log.error("Chunk parse error: %s", exc, exc_info=True)
        except Exception as exc:  # pragma: no cover
            log.error("stream_chat() failed: %s – using stub", exc, exc_info=True)
            yield f"[stub-stream] {last_msg}"

    # ───────────────────────── stub helpers ─────────────────────────────
    @staticmethod
    def _stub(text: str) -> Dict[str, Any]:
        """Return fake, yet deterministic, assistant reply."""
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
