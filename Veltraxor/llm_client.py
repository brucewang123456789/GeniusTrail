# llm_client.py

from __future__ import annotations

import json
import uuid
import logging
from typing import Any, Dict, List, AsyncIterator

import httpx
from config import settings

# Logger setup
LOG_FORMAT = '{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("llm-client")


class LLMClient:
    """Wrapper around chat completions with robust error handling."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # Load model name and base URL from settings if not provided
        self.model = model or settings.VELTRAX_MODEL
        self.base_url = base_url or settings.XAI_API_URL

        if not self.base_url:
            log.error("Initialization failed: XAI_API_URL is missing or empty")
        if not settings.XAI_API_KEY:
            log.error("Initialization warning: XAI_API_KEY is missing or empty")

        log.info("LLMClient init: model=%s, base_url=%s", self.model, self.base_url)

        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {settings.XAI_API_KEY}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        last_msg = messages[-1].get("content") if messages else ""
        log.info(">> chat() called; last user msg=%r", last_msg)
        payload = {"model": self.model, "messages": messages, "stream": False} | kwargs
        log.debug("chat payload: %s", payload)

        # HTTP request
        try:
            response = httpx.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error("HTTP status error in chat(): %s", exc, exc_info=True)
            return self._stub(last_msg)
        except httpx.RequestError as exc:
            log.error("Request error in chat(): %s", exc, exc_info=True)
            return self._stub(last_msg)
        except Exception as exc:
            log.error("Unexpected error in chat(): %s", exc, exc_info=True)
            return self._stub(last_msg)

        # JSON parse
        try:
            resp_json = response.json()
        except ValueError as exc:
            log.error("JSON decode failed in chat(): %s", exc, exc_info=True)
            return self._stub(last_msg)

        log.debug("chat response JSON: %s", resp_json)
        return resp_json

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        last_msg = messages[-1].get("content") if messages else ""
        log.info(">> stream_chat() called; last user msg=%r", last_msg)
        payload = {"model": self.model, "messages": messages, "stream": True} | kwargs
        log.debug("stream_chat payload: %s", payload)

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST", self.base_url, headers=self.headers, json=payload
                ) as resp:
                    resp.raise_for_status()
                    log.info("<< stream_chat() HTTP %s", resp.status_code)
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        # parse each chunk
                        try:
                            chunk_obj = json.loads(raw)
                        except Exception as exc:
                            log.error(
                                "Failed to parse chunk in stream_chat(): %s; raw=%r",
                                exc,
                                raw,
                                exc_info=True,
                            )
                            continue
                        choices = chunk_obj.get("choices")
                        if isinstance(choices, list) and choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
        except httpx.HTTPStatusError as exc:
            log.error("HTTP error in stream_chat(): %s", exc, exc_info=True)
            yield "[stub-stream] " + last_msg
        except Exception as exc:
            log.error("Unexpected error in stream_chat(): %s", exc, exc_info=True)
            yield "[stub-stream] " + last_msg

    @staticmethod
    def _stub(text: str) -> Dict[str, Any]:
        """Return a minimal stub response on failure."""
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
