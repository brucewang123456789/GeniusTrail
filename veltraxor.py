# -*- coding: utf-8 -*-
"""veltraxor.py — unified FastAPI service and interactive CLI."""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from prometheus_client import CollectorRegistry, Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

from dynamic_cot_controller import decide_cot, integrate_cot
from llm_client import LLMClient

# ─────────────────────── config & logging ───────────────────────
load_dotenv()
MODEL_NAME: str = os.getenv("VELTRAX_MODEL", "grok-3-latest")
SYSTEM_PROMPT: str = os.getenv("VELTRAX_SYSTEM_PROMPT", "")
ALLOWED_ORIGINS: List[str] = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("veltraxor")

# ───────────────────────── LLM client ───────────────────────────
client: LLMClient = LLMClient(model=MODEL_NAME)

# ───────────────────────── metrics ──────────────────────────────
registry: CollectorRegistry = CollectorRegistry()
REQ_COUNTER: Counter = Counter(
    "http_requests_total", "Total HTTP reqs", ["path", "method", "status"], registry=registry
)
REQ_LATENCY: Histogram = Histogram(
    "http_request_latency_seconds", "Request latency", ["path", "method"], registry=registry
)
STREAM_TOKENS: Counter = Counter(
    "chat_stream_tokens_total", "Streamed tokens", ["used_cot"], registry=registry
)

# ─────────────────────── FastAPI setup ─────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    log.info("Veltraxor API starting")
    yield
    log.info("Veltraxor API stopped")

app: FastAPI = FastAPI(title="Veltraxor API", version="0.2.0", docs_url="/docs", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

class _AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[..., Awaitable[Any]]) -> Any:
        start = time.time()
        resp = await call_next(request)
        REQ_COUNTER.labels(request.url.path, request.method, resp.status_code).inc()
        REQ_LATENCY.labels(request.url.path, request.method).observe(time.time() - start)
        log.info(
            "%s %s →%s %.1f ms",
            request.method,
            request.url.path,
            resp.status_code,
            (time.time() - start) * 1000,
        )
        return resp

app.add_middleware(_AccessLog)

# ───────────────────────── models ───────────────────────────────
class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] | None = None

class ChatResponse(BaseModel):
    response: str
    used_cot: bool
    duration_ms: int

def _assemble(prompt: str, history: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
    msgs = history[:] if history else []
    msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": prompt})
    return msgs

# ─────────────────────── auth & errors ─────────────────────────
def verify_token(request: Request) -> None:
    """
    Strict auth:
    - If VELTRAX_API_TOKEN is unset → always 401.
    - If set, header must be either '<token>' or 'Bearer <token>'.
    """
    expected = os.getenv("VELTRAX_API_TOKEN")
    if not expected:
        # No token configured → unauthorized
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")
    hdr = (request.headers.get("authorization") or "").strip()
    # Exact match or Bearer prefix
    if hdr == expected or hdr == f"Bearer {expected}":
        return
    # Invalid/missing header
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")

@app.exception_handler(Exception)
async def everything(_: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled\n%s", traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# ─────────────────────── input validation ───────────────────────
def validate_prompt(prompt: str) -> None:
    if not prompt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt must not be empty")
    if len(prompt) > 10000:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt too long")

# ─────────────────────── endpoints ─────────────────────────────
@app.get("/ping")
async def ping() -> dict[str, bool]:
    return {"pong": True}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    # 1) Validate input first → 400 优先
    validate_prompt(req.prompt)
    # 2) Then auth → 401/403
    verify_token(request)

    # Chat logic unchanged
    messages = _assemble(req.prompt, req.history)
    used_cot = False
    try:
        used_cot = decide_cot(req.prompt, "")
        if used_cot:
            first = client.chat(messages)
            first_text = first["choices"][0]["message"]["content"]
            if first_text:
                messages = integrate_cot(client, SYSTEM_PROMPT, req.prompt, first_text) or messages
    except Exception:
        used_cot = False

    start = time.time()
    raw = client.chat(messages)
    return ChatResponse(
        response=raw["choices"][0]["message"]["content"],
        used_cot=used_cot,
        duration_ms=int((time.time() - start) * 1000),
    )

@app.post("/chat_stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    # 1) Validate input first
    validate_prompt(req.prompt)
    # 2) Auth next
    verify_token(request)

    async def gen() -> AsyncIterator[str]:
        messages = _assemble(req.prompt, req.history)
        used_cot = False
        try:
            used_cot = decide_cot(req.prompt, "")
            if used_cot:
                first = client.chat(messages)
                first_text = first["choices"][0]["message"]["content"]
                if first_text:
                    messages = integrate_cot(client, SYSTEM_PROMPT, req.prompt, first_text) or messages
        except Exception:
            used_cot = False

        start = time.time()
        try:
            async for chunk in client.stream_chat(messages):
                STREAM_TOKENS.labels(str(used_cot)).inc()
                yield json.dumps({"chunk": chunk, "used_cot": used_cot, "final": False}) + "\n"
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Upstream {e.response.status_code}") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        yield json.dumps(
            {
                "chunk": "[DONE]",
                "used_cot": used_cot,
                "final": True,
                "duration_ms": int((time.time() - start) * 1000),
            }
        ) + "\n"

    return StreamingResponse(gen(), media_type="application/json")

# CLI helper unchanged …
