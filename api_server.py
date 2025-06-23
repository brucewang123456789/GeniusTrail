from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, Dict, List, Any

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from prometheus_client import CollectorRegistry, Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

# Import LLM client + CoT logic
from llm_client import LLMClient
from dynamic_cot_controller import decide_cot, integrate_cot

# Load .env
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("api-server")

# Config
MODEL_NAME: str = os.getenv("VELTRAX_MODEL", "grok-3-latest")
API_TOKEN: str | None = os.getenv("VELTRAX_API_TOKEN")
SYSTEM_PROMPT: str = os.getenv("VELTRAX_SYSTEM_PROMPT", "")
CORS_ORIGINS: List[str] = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

client: LLMClient = LLMClient(model=MODEL_NAME)

# Metrics
registry: CollectorRegistry = CollectorRegistry()
REQ_COUNTER: Counter = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["path", "method", "status"],
    registry=registry,
)
REQ_LATENCY: Histogram = Histogram(
    "http_request_latency_seconds",
    "Request latency",
    ["path", "method"],
    registry=registry,
)
STREAM_TOKENS: Counter = Counter(
    "chat_stream_tokens_total", "Streamed tokens", ["used_cot"], registry=registry
)


# Middleware classes
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[..., Awaitable[Any]]
    ) -> Any:
        rid = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.rid = rid
        resp = await call_next(request)
        resp.headers["x-request-id"] = rid
        return resp


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[..., Awaitable[Any]]
    ) -> Any:
        start = time.time()
        resp = await call_next(request)
        latency = time.time() - start
        REQ_COUNTER.labels(request.url.path, request.method, resp.status_code).inc()
        REQ_LATENCY.labels(request.url.path, request.method).observe(latency)
        log.info(
            f"{request.method} {request.url.path} → {resp.status_code} {latency*1000:.1f}ms"
        )
        return resp


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("API server starting")
    yield
    log.info("API server stopped")


app: FastAPI = FastAPI(title="Veltraxor Chat API", version="0.3.0", lifespan=lifespan)
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

__all__ = ["app"]


# Models
class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] | None = None


class ChatResponse(BaseModel):
    response: str
    used_cot: bool
    duration_ms: int


# Helpers
def assemble(prompt: str, history: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = history[:] if history else []
    msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def verify_token(request: Request) -> None:
    hdr: str | None = request.headers.get("authorization")
    if not hdr:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token"
        )
    if not API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server token not configured",
        )
    expected = f"Bearer {API_TOKEN}"
    if hdr != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token"
        )


def validate_prompt(prompt: str) -> None:
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt must not be empty"
        )
    if len(prompt) > 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Prompt too long"
        )


# Endpoints
@app.get("/ping")
async def ping() -> dict[str, bool]:
    return {"pong": True}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(req: ChatRequest) -> ChatResponse:
    validate_prompt(req.prompt)
    used_cot: bool = False
    msgs = assemble(req.prompt, req.history)
    try:
        used_cot = decide_cot(req.prompt, "")
        if used_cot:
            first = client.chat(msgs)
            cot = first["choices"][0]["message"]["content"]
            if cot:
                msgs = integrate_cot(client, SYSTEM_PROMPT, req.prompt, cot) or msgs
    except Exception:
        used_cot = False

    start = time.time()
    try:
        raw = client.chat(msgs)
    except Exception as e:
        log.error("LLM failure: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM backend failure",
        )

    duration_ms = int((time.time() - start) * 1000)
    text = raw["choices"][0]["message"]["content"].strip()
    return ChatResponse(response=text, used_cot=used_cot, duration_ms=duration_ms)


@app.post("/chat_stream", dependencies=[Depends(verify_token)])
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        messages = assemble(req.prompt, req.history)
        used_cot: bool = False
        try:
            used_cot = decide_cot(req.prompt, "")
            if used_cot:
                first = client.chat(messages)
                cot = first["choices"][0]["message"]["content"]
                if cot:
                    messages = (
                        integrate_cot(client, SYSTEM_PROMPT, req.prompt, cot)
                        or messages
                    )
        except Exception:
            used_cot = False

        start = time.time()
        try:
            async for chunk in client.stream_chat(messages):
                STREAM_TOKENS.labels(str(used_cot)).inc()
                yield json.dumps(
                    {"chunk": chunk, "used_cot": used_cot, "final": False}
                ) + "\n"
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Upstream {e.response.status_code}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Streaming failure",
            )

        yield json.dumps(
            {
                "chunk": "[DONE]",
                "used_cot": used_cot,
                "final": True,
                "duration_ms": int((time.time() - start) * 1000),
            }
        ) + "\n"

    return StreamingResponse(gen(), media_type="application/json")
