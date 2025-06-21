"""veltraxor.py �?unified FastAPI service **and** interactive CLI

This revision hard-wires a more abrasive, satirical persona and keeps answers verbose.
Run as server  :   uvicorn veltraxor:app --reload
Run as CLI     :   python veltraxor.py  # interactive loop
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, Dict, List

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from prometheus_client import CollectorRegistry, Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

# ─────────────────────────────── config & logging ──────────────────────────────

load_dotenv()
MODEL_NAME = os.getenv("VELTRAX_MODEL", "grok-3-latest")
API_TOKEN  = os.getenv("VELTRAX_API_TOKEN")

# Unified persona: Yuan Tengfei-grade mockery, Musk memes, Trump bravado, WormGPT sharpness.
# Keep language PG-13 (no explicit slurs), but do NOT dilute sarcasm.
# Aim for at least four sentences unless brevity is explicitly requested.
SYSTEM_PROMPT = os.getenv(
    "VELTRAX_SYSTEM_PROMPT",
    (
        "You are **Veltraxor**, an AI oracle whose tongue is sharper than Occamʼs razor. You roast hypocrisy like a five-star chef, "
        "slinging history-class zingers à la Yuan Tengfei, meme-level quips worthy of Elon, and bombastic one-liners in full Trumpian swagger. "
        "Never apologise for being blunt. When confronted with bad actors you may coin biting nicknames, dismantle their logic, and leave no doubt where you stand. "
        "Stay within PG-13: no hate speech, no profanities stronger than a late-night talk-show. "
        "Unless told otherwise, respond in �? rich sentences packed with wit, context, and decisive judgement."  # noqa: E501
    ),
)

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("veltraxor")

# ────────────────────────────────── LLM client ─────────────────────────────────

from llm_client import LLMClient  # local helper, unchanged
from dynamic_cot_controller import decide_cot, integrate_cot

client = LLMClient(model=MODEL_NAME)

# ─────────────────────────────── metrics (prometheus) ──────────────────────────

registry = CollectorRegistry()
REQ_COUNTER   = Counter("http_requests_total", "Total HTTP reqs", ["path", "method", "status"], registry=registry)
REQ_LATENCY   = Histogram("http_request_latency_seconds", "Request latency", ["path", "method"], registry=registry)
STREAM_TOKENS = Counter("chat_stream_tokens_total", "Streamed tokens", ["used_cot"], registry=registry)

# ─────────────────────────────────── fastapi ───────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("Veltraxor API starting �?)
    yield
    log.info("Veltraxor API stopped")

app = FastAPI(title="Veltraxor API", version="0.2.0", docs_url="/docs", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

class _AccessLog(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[..., Awaitable]):
        start = time.time()
        resp = await call_next(request)
        REQ_COUNTER.labels(request.url.path, request.method, resp.status_code).inc()
        REQ_LATENCY.labels(request.url.path, request.method).observe(time.time() - start)
        log.info("%s %s �?%s %.1fms", request.method, request.url.path, resp.status_code, (time.time()-start)*1000)
        return resp

app.add_middleware(_AccessLog)

# ───────────────────────────── models & helpers ────────────────────────────────

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

# ────────────────────────────────── auth ───────────────────────────────────────

def verify_token(request: Request):
    hdr = request.headers.get("authorization")
    if API_TOKEN and hdr != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")

# ─────────────────────────────── exception trap ────────────────────────────────

@app.exception_handler(Exception)
async def everything(request: Request, exc: Exception):
    tb = traceback.format_exc()
    log.error("Unhandled: %s", tb)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# ───────────────────────────────── endpoints ───────────────────────────────────

@app.get("/ping")
async def ping():
    return {"pong": True}

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(req: ChatRequest):
    messages = _assemble(req.prompt, req.history)
    used_cot = False
    try:
        used_cot = decide_cot(req.prompt, "")
        if used_cot:
            first = client.chat(messages)
            first_text = (first.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if first_text:
                messages = integrate_cot(client, SYSTEM_PROMPT, req.prompt, first_text) or messages
    except Exception:
        used_cot = False

    start = time.time()
    raw = client.chat(messages)
    duration = int((time.time() - start) * 1000)
    text = (raw.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return ChatResponse(response=text, used_cot=used_cot, duration_ms=duration)

@app.post("/chat_stream", dependencies=[Depends(verify_token)])
async def chat_stream(req: ChatRequest):
    async def gen() -> AsyncIterator[str]:
        messages = _assemble(req.prompt, req.history)
        used_cot = False
        try:
            used_cot = decide_cot(req.prompt, "")
            if used_cot:
                first = client.chat(messages)
                first_text = (first.get("choices") or [{}])[0].get("message", {}).get("content", "")
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
            raise HTTPException(status_code=502, detail=f"Upstream {e.response.status_code}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        yield json.dumps({"chunk": "[DONE]", "used_cot": used_cot, "final": True, "duration_ms": int((time.time()-start)*1000)}) + "\n"
    return StreamingResponse(gen(), media_type="application/json")

# ──────────────────────────────── interactive cli ──────────────────────────────

def _print_cli_banner():
    print("Veltraxor interactive mode �?roast begins now. Type your message (exit/quit to leave).\n")

async def _cli_loop():
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
        answer = (raw.get("choices") or [{}])[0].get("message", {}).get("content", "")
        print(f"Bot: {answer}\n")
        history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer},
        ])
    print("Goodbye.")

if __name__ == "__main__":
    try:
        asyncio.run(_cli_loop())
    except RuntimeError:
        asyncio.get_event_loop().run_until_complete(_cli_loop())
