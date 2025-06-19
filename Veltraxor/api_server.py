from dotenv import load_dotenv
from langdetect import detect  # Import language detection

# ────────────────────────────── Environment Setup ───────────────────────────
load_dotenv()

# ───────────────────────────── Standard Library ──────────────────────────────
import os
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, AsyncIterator, Callable, Awaitable
from contextlib import asynccontextmanager

# ─────────────────────────── Third-Party Packages ────────────────────────────
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from llm_client import LLMClient
from dynamic_cot_controller import decide_cot, integrate_cot

# ───────────────────────────── Configuration ────────────────────────────────
MODEL_NAME = os.getenv("VELTRAX_MODEL", "grok-3-latest")
API_TOKEN = os.getenv("VELTRAX_API_TOKEN")  # Optional bearer auth
SYSTEM_PROMPT = os.getenv(
    "VELTRAX_SYSTEM_PROMPT",
    "You are Veltraxor, a sharp-tongued assistant. Answer with witty, satirical, and occasionally biting humor—"
    "think punchlines like a stand-up comedy of history, memes like Elon Musk, and taglines like Trump. "
    "Remain clever and non-offensive."
)
ALLOWED_ORIG = os.getenv("CORS_ORIGINS", "*").split(",")
LISTEN_PORT = int(os.getenv("PORT", 8000))  # Default to 8000

# ───────────────────────────── Logging Setup ─────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("veltraxor-api")

# ─────────────────────────── Prometheus Metrics ─────────────────────────────
METRICS_REGISTRY = CollectorRegistry()
REQ_COUNTER = Counter("http_requests_total", "Total HTTP requests",
                      ["path", "method", "status"], registry=METRICS_REGISTRY)
REQ_LATENCY = Histogram("http_request_latency_seconds", "Request latency",
                        ["path", "method"], registry=METRICS_REGISTRY)
STREAM_TOKENS = Counter("chat_stream_tokens_total", "Streamed tokens",
                        ["used_cot"], registry=METRICS_REGISTRY)

# ───────────────────────────── FastAPI App ──────────────────────────────────
client = LLMClient(model=MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Veltraxor API started")
    log.info("Registered routes: %s", [route.path for route in app.routes])
    yield
    log.info("Veltraxor API stopped")


app = FastAPI(
    title="Veltraxor API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

# ─────────────────────────────── Middlewares ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIG,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[..., Awaitable[Response]]) -> Response:
        start = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - start) * 1000
        REQ_COUNTER.labels(request.url.path, request.method, response.status_code).inc()
        REQ_LATENCY.labels(request.url.path, request.method).observe(latency_ms / 1000)
        log.info("%s %s -> %s %.1fms", request.method, request.url.path, response.status_code, latency_ms)
        return response


app.add_middleware(AccessLogMiddleware)


# ───────────────────────────── Pydantic Models ──────────────────────────────
class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] | None = None


class ChatResponse(BaseModel):
    response: str
    used_cot: bool
    duration_ms: int


# ───────────────────────────── Auth Helper ──────────────────────────────────
def verify_token(request: Request):
    if API_TOKEN and request.headers.get("authorization") != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


# ────────────────────────────── Helper Functions ─────────────────────────────
def assemble_messages(prompt: str, history: List[Dict[str, str]] | None):
    user_language = detect(prompt)  # Detect the language of user input
    language_rule = {
        "zh-cn": "请用中文回答，保持袁腾飞式毒舌。",
        "en": "Answer in English with Musk/Trump-style sarcasm.",
    }.get(user_language, f"Answer in {user_language} with the same sharp tone.")

    msgs = history[:] if history else []
    msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT + "\n" + language_rule})
    msgs.append({"role": "user", "content": prompt})
    return msgs


# ─────────────────────────────── Endpoints ───────────────────────────────────
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}


@app.get("/metrics")
async def metrics():
    data = generate_latest(METRICS_REGISTRY)
    return Response(data, media_type=CONTENT_TYPE_LATEST)


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(req: ChatRequest, request: Request):
    """Assemble messages, optionally apply CoT, call LLMClient.chat, and return response."""
    raw_body = await request.body()
    log.info("[DEBUG] /chat raw body: %s", raw_body.decode(errors="ignore"))

    # 1. Build message list
    msgs = assemble_messages(req.prompt, req.history)
    log.info("[DEBUG] /chat before CoT messages: %s", msgs)

    # 2. Decide whether to use Chain-of-Thought
    used_cot = decide_cot(req.prompt, "")  # history not used for decision
    if used_cot:
        # Perform iterative CoT with the initial client.chat reply
        first = client.chat(msgs)["choices"][0]["message"]["content"].strip()
        msgs = integrate_cot(client, SYSTEM_PROMPT, req.prompt, first)
        log.info("[DEBUG] /chat after CoT used, final msg: %s", msgs)

    # 3. Call LLM for final response
    start_ts = time.time()
    raw = client.chat(msgs)
    duration_ms = int((time.time() - start_ts) * 1000)
    log.info("[DEBUG] /chat raw LLM response: %s", raw)

    # 4. Extract content
    if isinstance(raw, dict) and raw.get("choices"):
        choice = raw["choices"][0]
        content = choice.get("message", {}).get("content", "") or choice.get("text", "") or ""
    else:
        content = str(raw)

    return ChatResponse(response=content, used_cot=used_cot, duration_ms=duration_ms)


@app.post("/chat_stream", dependencies=[Depends(verify_token)])
async def chat_stream(req: ChatRequest, request: Request):
    """Streamed variant of /chat using LLMClient.stream_chat."""

    async def generator() -> AsyncIterator[str]:
        raw_body = await request.body()
        log.info("[DEBUG] /chat_stream raw body: %s", raw_body.decode(errors="ignore"))

        msgs = assemble_messages(req.prompt, req.history)
        used_cot = decide_cot(req.prompt, "")
        if used_cot:
            first = await client.chat(msgs)["choices"][0]["message"]["content"].strip()
            msgs = integrate_cot(client, SYSTEM_PROMPT, req.prompt, first)

        start_ts = time.time()
        async for chunk in client.stream_chat(msgs):
            data = {"chunk": chunk, "used_cot": used_cot, "final": False}
            STREAM_TOKENS.labels(str(used_cot)).inc()
            yield json.dumps(data) + "\n"

        total_ms = int((time.time() - start_ts) * 1000)
        yield json.dumps({"chunk": "[DONE]", "used_cot": used_cot, "final": True, "duration_ms": total_ms}) + "\n"

    return StreamingResponse(generator(), media_type="application/json")