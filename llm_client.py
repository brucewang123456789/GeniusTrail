"""
LLMClient – wrapper around an external Chat-Completion endpoint.

CI safety:  when CI=true or XAI_API_KEY is missing, network calls are
short-circuited to a local stub so that tests never hit the real service.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List

import httpx
from config import settings

# Logger setup
LOG_FORMAT = '{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("llm-client")


class LLMClient:
    """Robust chat-completion helper with CI stub."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or settings.VELTRAX_MODEL
        self.base_url = base_url or settings.XAI_API_URL

        # CI stub toggle
        self._stub_mode: bool = (
            os.getenv("CI", "false").lower() == "true"
            or not settings.XAI_API_KEY
            or not self.base_url
        )
        if self._stub_mode:
            log.warning("LLMClient running in STUB mode – no real HTTP calls will be made")

        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {settings.XAI_API_KEY or 'dummy'}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        last_msg = messages[-1].get("content") if messages else ""
        payload = {"model": self.model, "messages": messages, "stream": False} | kwargs
        log.debug("chat payload: %s", payload)

        if self._stub_mode:
            return self._stub(last_msg)

        try:
            response = httpx.post(
                self.base_url, headers=self.headers, json=payload, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            log.error("chat() failed: %s – falling back to stub", exc, exc_info=True)
            return self._stub(last_msg)

    async def stream_chat(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        last_msg = messages[-1].get("content") if messages else ""
        payload = {"model": self.model, "messages": messages, "stream": True} | kwargs
        log.debug("stream_chat payload: %s", payload)

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
                            chunk_obj = json.loads(raw)
                            delta = (
                                chunk_obj.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content")
                            )
                            if delta:
                                yield delta
                        except Exception as exc:
                            log.error("Chunk parse error: %s", exc, exc_info=True)
        except Exception as exc:
            log.error("stream_chat() failed: %s – fallback stub", exc, exc_info=True)
            yield f"[stub-stream] {last_msg}"

    @staticmethod
    def _stub(text: str) -> Dict[str, Any]:
        """Return a deterministic fake response for tests/CI."""
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
