from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, Dict, List

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from prometheus_client import CollectorRegistry, Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("api-server")

# Configuration
MODEL_NAME = os.getenv("VELTRAX_MODEL", "grok-3-latest")
API_TOKEN = os.getenv("VELTRAX_API_TOKEN")
SYSTEM_PROMPT = os.getenv("VELTRAX_SYSTEM_PROMPT", "")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

# Import LLM client and CoT
from llm_client import LLMClient
from dynamic_cot_controller import decide_cot, integrate_cot

client = LLMClient(model=MODEL_NAME)

# Metrics definitions (registry, counters, histograms)
registry = CollectorRegistry()
REQ_COUNTER = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["path", "method", "status"],
    registry=registry
)
REQ_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Request latency",
    ["path", "method"],
    registry=registry
)
STREAM_TOKENS = Counter(
    "chat_stream_tokens_total",
    "Streamed tokens",
    ["used_cot"],
    registry=registry
)

# Middleware for request IDs and access logs
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[..., Awaitable]):
        rid = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.rid = rid
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[..., Awaitable]):
        start = time.time()
        resp = await call_next(request)
        latency = time.time() - start
        REQ_COUNTER.labels(request.url.path, request.method, resp.status_code).inc()
        REQ_LATENCY.labels(request.url.path, request.method).observe(latency)
        log.info(f"{request.method} {request.url.path} -> {resp.status_code} {latency*1000:.1f}ms")
        return resp

@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("API server starting.")
    yield
    log.info("API server stopped")

# FastAPI app setup
app = FastAPI(title="Veltraxor Chat API", version="0.3.0", lifespan=lifespan)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Explicit export for `from api_server import app`
__all__ = ["app"]

# Pydantic models
class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] | None = None

class ChatResponse(BaseModel):
    response: str
    used_cot: bool
    duration_ms: int

# Helper to assemble messages
def assemble(prompt: str, history: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
    msgs = history[:] if history else []
    msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": prompt})
    return msgs

# Token verification dependency
def verify_token(request: Request):
    hdr = request.headers.get("authorization")
    if API_TOKEN and hdr != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")

# Prompt validation
def validate_prompt(prompt: str):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty")
    if len(prompt) > 10000:
        raise HTTPException(status_code=400, detail="Prompt too long")

# Chat endpoint returning full ChatResponse
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(req: ChatRequest, request: Request):
    validate_prompt(req.prompt)
    used_cot = False
    messages = assemble(req.prompt, req.history)
    # Decide CoT
    try:
        used_cot = decide_cot(req.prompt, "")
        if used_cot:
            first = client.chat(messages)
            cot_text = first.get("choices", [{}])[0].get("message", {}).get("content", "")
            if cot_text:
                messages = integrate_cot(client, SYSTEM_PROMPT, req.prompt, cot_text) or messages
    except Exception:
        used_cot = False

    start = time.time()
    try:
        raw = client.chat(messages)
    except Exception as e:
        log.error("LLM failure: %s", str(e))
        raise HTTPException(status_code=500, detail="LLM backend failure")

    duration_ms = int((time.time() - start) * 1000)
    text = raw.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return ChatResponse(response=text, used_cot=used_cot, duration_ms=duration_ms)

# Ping endpoint
@app.get("/ping")
async def ping():
    return {"pong": True}

# Streaming endpoint with validation
@app.post("/chat_stream", dependencies=[Depends(verify_token)])
async def chat_stream(req: ChatRequest, request: Request):
    validate_prompt(req.prompt)
    async def gen() -> AsyncIterator[str]:
        msgs = assemble(req.prompt, req.history)
        used_cot = False
        try:
            used_cot = decide_cot(req.prompt, "")
            if used_cot:
                first = client.chat(msgs)
                cot_text = first.get("choices", [{}])[0].get("message", {}).get("content", "")
                if cot_text:
                    msgs = integrate_cot(client, SYSTEM_PROMPT, req.prompt, cot_text) or msgs
        except Exception:
            used_cot = False

        start = time.time()
        try:
            async for chunk in client.stream_chat(msgs):
                STREAM_TOKENS.labels(str(used_cot)).inc()
                yield json.dumps({"chunk": chunk, "used_cot": used_cot, "final": False}) + "\n"
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Upstream {e.response.status_code}")
        except Exception:
            raise HTTPException(status_code=500, detail="Streaming failure")
        yield json.dumps({
            "chunk": "[DONE]",
            "used_cot": used_cot,
            "final": True,
            "duration_ms": int((time.time()-start)*1000)
        }) + "\n"

    return StreamingResponse(gen(), media_type="application/json")
