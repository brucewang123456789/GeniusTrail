# -*- coding: utf-8 -*-
"""veltraxor.py — unified FastAPI service **and** interactive CLI

Run as server : uvicorn veltraxor:app --reload
Run as CLI    : python veltraxor.py
"""
from __future__ import annotations

# ───── standard library ─────
import asyncio
import json
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List

# ───── third-party ─────
import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from prometheus_client import CollectorRegistry, Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

# ───── local modules ─────
from llm_client import LLMClient
from dynamic_cot_controller import decide_cot, integrate_cot

# ───────────────────────────────────────── config & logging ─────────────────────────────────────────
load_dotenv()
MODEL_NAME: str = os.getenv("VELTRAX_MODEL", "grok-3-latest")
API_TOKEN_ENV = "VELTRAX_API_TOKEN"
SYSTEM_PROMPT: str = os.getenv(
    "VELTRAX_SYSTEM_PROMPT",
    (
        "You are **Veltraxor**, an AI oracle whose tongue is sharper than Occam’s razor. "
        "You roast hypocrisy like a five-star chef, sling history-class zingers à la Yuan Tengfei, "
        "meme-level quips worthy of Elon, and bombastic one-liners in full Trumpian swagger. "
        "Stay within PG-13—no hate speech, no profanity stronger than late-night TV. "
        "Unless told otherwise, answer in rich sentences packed with wit, context and decisive judgment."
    ),
)

ALLOWED_ORIGINS: List[str] = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("veltraxor")

# ───────────────────────────────────────── LLM client ─────────────────────────────────────────
client: LLMClient = LLMClient(model=MODEL_NAME)

# ───────────────────────────────────────── metrics (prometheus) ─────────────────────────────────────────
registry: CollectorRegistry = CollectorRegistry()
REQ_COUNTER: Counter = Counter(
    "http_requests_total", "Total HTTP reqs", ["path", "method", "status"], registry=registry
)
REQ_LATENCY: Histogram = Histogram(
    "http_request_latency_seconds", "Request latency", ["path", "method"], registry=registry
)
STREAM_TOKENS: Counter = Counter("chat_stream_tokens_total", "Streamed tokens", ["used_cot"], registry=registry)

# ───────────────────────────────────────── FastAPI setup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    log.info("Veltraxor API starting")
    yield
    log.info("Veltraxor API stopped")


app: FastAPI = FastAPI(title="Veltraxor API", version="0.2.1", docs_url="/docs", lifespan=lifespan)

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
            "%s %s →%s %.1fms",
            request.method,
            request.url.path,
            resp.status_code,
            (time.time() - start) * 1000,
        )
        return resp


app.add_middleware(_AccessLog)

# ───────────────────────────────────────── models & helpers ─────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] | None = None


class ChatResponse(BaseModel):
    response: str
    used_cot: bool
    duration_ms: int


def _assemble(prompt: str, history: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = history[:] if history else []
    msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": prompt})
    return msgs


# ───────────────────────────────────────── auth & errors ─────────────────────────────────────────
def verify_token(request: Request) -> None:
    """
    • Require Authorization header.
    • If env VELTRAX_API_TOKEN is set → header token must match (case-insensitive 'Bearer ' 可选)。
    • If env not set → 任何非空 Authorization 头均接受。
    """
    hdr = request.headers.get("authorization")
    if not hdr:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authorization header missing")

    expected = os.getenv(API_TOKEN_ENV)
    if expected:  # strict match when configured
        token_only = hdr.removeprefix("Bearer ").removeprefix("bearer ").strip()
        if token_only != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized token")


@app.exception_handler(Exception)
async def everything(_: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled: %s", traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


# ───────────────────────────────────────── endpoints ─────────────────────────────────────────
@app.get("/ping")
async def ping() -> dict[str, bool]:
    return {"pong": True}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(req: ChatRequest) -> ChatResponse:
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
    duration = int((time.time() - start) * 1000)
    text = raw["choices"][0]["message"]["content"]
    return ChatResponse(response=text, used_cot=used_cot, duration_ms=duration)


@app.post("/chat_stream", dependencies=[Depends(verify_token)])
async def chat_stream(req: ChatRequest) -> StreamingResponse:
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


# ───────────────────────────────────────── interactive CLI ─────────────────────────────────────────
def _print_cli_banner() -> None:
    print("Veltraxor interactive mode — type 'exit' to quit.\n")


async def _cli_loop() -> None:
    history: List[Dict[str, str]] = []
    _print_cli_banner()
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        msgs = _assemble(user_input, history)
        try:
            raw = await asyncio.to_thread(client.chat, msgs)
        except Exception as e:
            print(f"LLM error: {e}")
            continue
        answer = raw["choices"][0]["message"]["content"]
        print(f"Bot: {answer}\n")
        history.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": answer}])
    print("Goodbye.")


if __name__ == "__main__":
    try:
        asyncio.run(_cli_loop())
    except RuntimeError:
        asyncio.get_event_loop().run_until_complete(_cli_loop())

