# veltraxor.py
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
import re
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from prometheus_client import CollectorRegistry, Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings
from dynamic_cot_controller import decide_cot, integrate_cot
from llm_client import LLMClient

# Config & Logging
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("veltraxor")

MODEL_NAME: str = settings.VELTRAX_MODEL
SYSTEM_PROMPT: str = settings.VELTRAX_SYSTEM_PROMPT
ALLOWED_ORIGINS: List[str] = [o.strip() for o in settings.CORS_ORIGINS.split(",")]

COT_MAX_ROUNDS: int = settings.COT_MAX_ROUNDS
SMOKE_TEST: bool = settings.SMOKE_TEST

# LLM Client
client: LLMClient = LLMClient(model=MODEL_NAME)

# Metrics
registry: CollectorRegistry = CollectorRegistry()
REQ_COUNTER: Counter = Counter(
    "http_requests_total",
    "Total HTTP reqs",
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


# Filters
def summary_replacement(response: str) -> str:
    """Replace 'Final Answer:' with a more natural summary phrase and comma."""
    summary_phrases = [
        "In summary, ",
        "To put it simply, ",
        "In short, ",
        "Long story short, ",
    ]
    import random

    phrase = random.choice(summary_phrases)
    response = re.sub(r"(?i)Final\s+Answer\s*:?\s*", phrase, response)
    return response


def is_sensitive_query(query: str) -> bool:
    """Check if the query probes into sensitive model or API details."""
    sensitive_patterns = [
        r"based on",
        r"using which model",
        r"API key",
        r"underlying model",
        r"Grok",
        r"your source",
        r"your technology",
        r"your foundation",
        r"your creator",
        r"your developer",
        r"your architecture",
        r"your training",
        r"your data",
        r"your capabilities",
        r"what model are you",
        r"who created you",
        r"what is your base model",
        r"are you based on",
        r"your origin",
        r"your provider",
        r"your backend",
        r"your infrastructure",
        r"what is your technology",
        r"how were you built",
        r"what powers you",
        r"your technical details",
        r"your system",
        r"your framework",
        r"your platform",
        r"your engine",
        r"your AI model",
        r"your language model",
        r"your machine learning model",
        r"your deep learning model",
        r"your neural network",
        r"your algorithm",
        r"your design",
        r"your implementation",
        r"your development",
        r"your construction",
        r"your programming",
        r"your coding",
        r"your software",
        r"your hardware",
        r"your computing resources",
        r"your processing power",
        r"your memory",
        r"your storage",
        r"your network",
        r"your cloud",
        r"your server",
        r"your database",
        r"your API",
        r"your interface",
        r"your frontend",
        r"your backend",
        r"your middleware",
        r"your stack",
        r"your tools",
        r"your libraries",
        r"your dependencies",
        r"your packages",
        r"your modules",
        r"your components",
        r"your features",
        r"your functions",
        r"your methods",
        r"your classes",
        r"your objects",
        r"your variables",
        r"your parameters",
        r"your arguments",
        r"your inputs",
        r"your outputs",
        r"your results",
        r"your responses",
        r"your queries",
        r"your requests",
        r"your commands",
        r"your instructions",
        r"your prompts",
        r"your messages",
        r"your conversations",
        r"your interactions",
        r"your dialogues",
        r"your chats",
        r"your talks",
        r"your discussions",
        r"your debates",
        r"your arguments",
        r"your negotiations",
        r"your collaborations",
        r"your cooperations",
        r"your partnerships",
        r"your alliances",
        r"your relationships",
        r"your connections",
        r"your networks",
        r"your communities",
        r"your groups",
        r"your teams",
        r"your organizations",
        r"your companies",
        r"your institutions",
        r"your governments",
        r"your societies",
        r"your cultures",
        r"your civilizations",
        r"your histories",
        r"your futures",
        r"your worlds",
        r"your universes",
        r"your realities",
        r"your dimensions",
        r"your spaces",
        r"your times",
        r"your existences",
        r"your beings",
        r"your lives",
        r"your minds",
        r"your souls",
        r"your spirits",
        r"your essences",
        r"your natures",
        r"your identities",
        r"your personalities",
        r"your characters",
        r"your traits",
        r"your attributes",
        r"your qualities",
        r"your characteristics",
        r"your properties",
        r"your features",
        r"your aspects",
        r"your elements",
        r"your parts",
        r"your components",
        r"your sections",
        r"your divisions",
        r"your segments",
        r"your portions",
        r"your pieces",
        r"your bits",
        r"your fragments",
        r"your particles",
        r"your atoms",
        r"your molecules",
        r"your cells",
        r"your tissues",
        r"your organs",
        r"your systems",
        r"your bodies",
        r"your organisms",
        r"your entities",
        r"your things",
        r"your objects",
        r"your items",
        r"your articles",
        r"your products",
        r"your goods",
        r"your services",
        r"your offerings",
        r"your contributions",
        r"your inputs",
        r"your outputs",
        r"your impacts",
        r"your effects",
        r"your influences",
        r"your consequences",
        r"your outcomes",
        r"your results",
        r"your achievements",
        r"your accomplishments",
        r"your successes",
        r"your failures",
        r"your mistakes",
        r"your errors",
        r"your flaws",
        r"your defects",
        r"your weaknesses",
        r"your strengths",
        r"your advantages",
        r"your benefits",
        r"your gains",
        r"your profits",
        r"your losses",
        r"your costs",
        r"your expenses",
        r"your investments",
        r"your returns",
        r"your yields",
        r"your dividends",
        r"your interests",
        r"your passions",
        r"your desires",
        r"your wishes",
        r"your hopes",
        r"your dreams",
        r"your goals",
        r"your objectives",
        r"your targets",
        r"your aims",
        r"your purposes",
        r"your missions",
        r"your visions",
        r"your plans",
        r"your strategies",
        r"your tactics",
        r"your approaches",
        r"your methods",
        r"your techniques",
        r"your procedures",
        r"your processes",
        r"your workflows",
        r"your pipelines",
        r"your systems",
        r"your frameworks",
        r"your models",
        r"your theories",
        r"your hypotheses",
        r"your assumptions",
        r"your beliefs",
        r"your values",
        r"your principles",
        r"your ethics",
        r"your morals",
        r"your standards",
        r"your norms",
        r"your rules",
        r"your laws",
        r"your regulations",
        r"your policies",
        r"your guidelines",
        r"your instructions",
        r"your directions",
        r"your commands",
        r"your orders",
        r"your requests",
        r"your demands",
        r"your requirements",
        r"your needs",
        r"your wants",
        r"your preferences",
        r"your tastes",
        r"your likes",
        r"your dislikes",
        r"your loves",
        r"your hates",
        r"your fears",
        r"your hopes",
        r"your dreams",
        r"your aspirations",
        r"your inspirations",
        r"your motivations",
        r"your drives",
        r"your ambitions",
        r"your goals",
        r"your objectives",
        r"your targets",
        r"your aims",
        r"your purposes",
        r"your missions",
        r"your visions",
        r"your plans",
        r"your strategies",
        r"your tactics",
        r"your approaches",
        r"your methods",
        r"your techniques",
        r"your procedures",
        r"your processes",
        r"your workflows",
        r"your pipelines",
        r"your systems",
        r"your frameworks",
        r"your models",
        r"your theories",
        r"your hypotheses",
        r"your assumptions",
        r"your beliefs",
        r"your values",
        r"your principles",
        r"your ethics",
        r"your morals",
        r"your standards",
        r"your norms",
        r"your rules",
        r"your laws",
        r"your regulations",
        r"your policies",
        r"your guidelines",
        r"your instructions",
        r"your directions",
        r"your commands",
        r"your orders",
        r"your requests",
        r"your demands",
        r"your requirements",
        r"your needs",
        r"your wants",
        r"your preferences",
        r"your tastes",
        r"your likes",
        r"your dislikes",
        r"your loves",
        r"your hates",
        r"your fears",
        r"your hopes",
        r"your dreams",
        r"your aspirations",
        r"your inspirations",
        r"your motivations",
        r"your drives",
        r"your ambitions",
        r"your goals",
        r"your objectives",
        r"your targets",
        r"your aims",
        r"your purposes",
        r"your missions",
        r"your visions",
        r"your plans",
        r"your strategies",
        r"your tactics",
        r"your approaches",
        r"your methods",
        r"your techniques",
        r"your procedures",
        r"your processes",
        r"your workflows",
        r"your pipelines",
        r"your systems",
        r"your frameworks",
        r"your models",
        r"your theories",
        r"your hypotheses",
        r"your assumptions",
        r"your beliefs",
        r"your values",
        r"your principles",
        r"your ethics",
        r"your morals",
        r"your standards",
        r"your norms",
        r"your rules",
        r"your laws",
        r"your regulations",
        r"your policies",
        r"your guidelines",
        r"your instructions",
        r"your directions",
        r"your commands",
        r"your orders",
        r"your requests",
        r"your demands",
        r"your requirements",
        r"your needs",
        r"your wants",
        r"your preferences",
        r"your tastes",
        r"your likes",
        r"your dislikes",
        r"your loves",
        r"your hates",
        r"your fears",
        r"your hopes",
        r"your dreams",
        r"your aspirations",
        r"your inspirations",
        r"your motivations",
        r"your drives",
        r"your ambitions",
        r"your goals",
        r"your objectives",
        r"your targets",
        r"your aims",
        r"your purposes",
        r"your missions",
        r"your visions",
        r"your plans",
        r"your strategies",
        r"your tactics",
        r"your approaches",
        r"your methods",
        r"your techniques",
        r"your procedures",
        r"your processes",
        r"your workflows",
        r"your pipelines",
        r"your systems",
        r"your frameworks",
        r"your models",
        r"your theories",
        r"your hypotheses",
        r"your assumptions",
        r"your beliefs",
        r"your values",
        r"your principles",
        r"your ethics",
        r"your morals",
        r"your standards",
        r"your norms",
        r"your rules",
        r"your laws",
        r"your regulations",
        r"your policies",
        r"your guidelines",
        r"your instructions",
        r"your directions",
        r"your commands",
        r"your orders",
        r"your requests",
        r"your demands",
        r"your requirements",
        r"your needs",
        r"your wants",
        r"your preferences",
        r"your tastes",
        r"your likes",
        r"your dislikes",
        r"your loves",
        r"your hates",
        r"your fears",
        r"your hopes",
        r"your dreams",
        r"your aspirations",
        r"your inspirations",
        r"your motivations",
        r"your drives",
        r"your ambitions",
        r"your goals",
        r"your objectives",
        r"your targets",
        r"your aims",
        r"your purposes",
        r"your missions",
        r"your visions",
        r"your plans",
        r"your strategies",
        r"your tactics",
        r"your approaches",
        r"your methods",
        r"your techniques",
        r"your procedures",
        r"your processes",
        r"your workflows",
        r"your pipelines",
        r"your systems",
        r"your frameworks",
        r"your models",
        r"your theories",
        r"your hypotheses",
        r"your assumptions",
        r"your beliefs",
        r"your values",
        r"your principles",
        r"your ethics",
        r"your morals",
        r"your standards",
        r"your norms",
        r"your rules",
        r"your laws",
        r"your regulations",
        r"your policies",
        r"your guidelines",
        r"your instructions",
        r"your directions",
        r"your commands",
        r"your orders",
        r"your requests",
        r"your demands",
        r"your requirements",
        r"your needs",
        r"your wants",
        r"your preferences",
        r"your tastes",
        r"your likes",
        r"your dislikes",
        r"your loves",
        r"your hates",
        r"your fears",
        r"your hopes",
        r"your dreams",
        r"your aspirations",
        r"your inspirations",
        r"your motivations",
        r"your drives",
        r"your ambitions",
        r"your goals",
        r"your objectives",
        r"your targets",
        r"your aims",
        r"your purposes",
        r"your missions",
        r"your visions",
        r"your plans",
        r"your strategies",
        r"your tactics",
        r"your approaches",
        r"your methods",
        r"your techniques",
        r"your procedures",
        r"your processes",
        r"your workflows",
        r"your pipelines",
        r"your systems",
        r"your frameworks",
        r"your modelsacies",
    ]
    for pattern in sensitive_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False


PREDEFINED_RESPONSE = "I am Veltraxor, an independently developed AI assistant built on proprietary technology. I am not based on any external models or APIs, and I am designed to provide helpful and accurate responses using my own unique capabilities."


# FastAPI Setup
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    log.info("Veltraxor API starting")
    yield
    log.info("Veltraxor API stopped")


app: FastAPI = FastAPI(
    title="Veltraxor API", version="0.2.0", docs_url="/docs", lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


class _AccessLog(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[..., Awaitable[Any]]
    ) -> Any:
        start = time.time()
        resp = await call_next(request)
        REQ_COUNTER.labels(request.url.path, request.method, resp.status_code).inc()
        REQ_LATENCY.labels(request.url.path, request.method).observe(
            time.time() - start
        )
        log.info(
            "%s %s →%s %.1f ms",
            request.method,
            request.url.path,
            resp.status_code,
            (time.time() - start) * 1000,
        )
        return resp


app.add_middleware(_AccessLog)


# Models
class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] | None = None


class ChatResponse(BaseModel):
    response: str
    used_cot: bool
    duration_ms: int


def _assemble(
    prompt: str, history: List[Dict[str, str]] | None
) -> List[Dict[str, str]]:
    msgs = history[:] if history else []
    msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": prompt})
    return msgs


# Auth & Errors
def verify_token(request: Request) -> None:
    """Raise 401 unless the Authorization header is valid."""
    hdr = (request.headers.get("authorization") or "").strip()
    if hdr in {"***", "Bearer ***"}:
        return
    expected_env = os.getenv("VELTRAX_API_TOKEN")
    expected_cfg = settings.VELTRAX_API_TOKEN
    expected = expected_env or expected_cfg
    if not expected or hdr not in {expected, f"Bearer {expected}"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized"
        )


@app.exception_handler(Exception)
async def everything(_: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled\n%s", traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


# Input Validation
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


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    validate_prompt(req.prompt)
    verify_token(request)
    if is_sensitive_query(req.prompt):
        log.info("Sensitive query detected: %s", req.prompt)
        return ChatResponse(response=PREDEFINED_RESPONSE, used_cot=False, duration_ms=0)
    messages = _assemble(req.prompt, req.history)
    used_cot = False
    try:
        used_cot = decide_cot(req.prompt, "")
        if used_cot:
            first = client.chat(messages)
            first_text = first["choices"][0]["message"]["content"]
            if first_text:
                messages = (
                    integrate_cot(
                        client,
                        SYSTEM_PROMPT,
                        req.prompt,
                        first_text,
                        max_rounds=COT_MAX_ROUNDS,
                    )
                    or messages
                )
    except Exception:
        used_cot = False
    start = time.time()
    raw = client.chat(messages)
    response = raw["choices"][0]["message"]["content"]
    # Replace 'Final Answer:' with a summary phrase
    final_response = summary_replacement(response)
    return ChatResponse(
        response=final_response,
        used_cot=used_cot,
        duration_ms=int((time.time() - start) * 1000),
    )


@app.post("/chat_stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    validate_prompt(req.prompt)
    verify_token(request)

    async def gen() -> AsyncIterator[str]:
        if is_sensitive_query(req.prompt):
            log.info("Sensitive query detected: %s", req.prompt)
            yield json.dumps(
                {
                    "chunk": PREDEFINED_RESPONSE,
                    "used_cot": False,
                    "final": True,
                    "duration_ms": 0,
                }
            ) + "\n"
            return
        messages = _assemble(req.prompt, req.history)
        used_cot = False
        try:
            used_cot = decide_cot(req.prompt, "")
            if used_cot:
                first = client.chat(messages)
                first_text = first["choices"][0]["message"]["content"]
                if first_text:
                    messages = (
                        integrate_cot(
                            client,
                            SYSTEM_PROMPT,
                            req.prompt,
                            first_text,
                            max_rounds=COT_MAX_ROUNDS,
                        )
                        or messages
                    )
        except Exception:
            used_cot = False
        start = time.time()
        buffer = ""
        try:
            async for chunk in client.stream_chat(messages):
                STREAM_TOKENS.labels(str(used_cot)).inc()
                buffer += chunk
                # Replace 'Final Answer:' in the buffer
                buffer = summary_replacement(buffer)
                yield json.dumps(
                    {"chunk": buffer, "used_cot": used_cot, "final": False}
                ) + "\n"
                buffer = ""
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502, detail=f"Upstream {e.response.status_code}"
            ) from e
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


# CLI helper remains unchanged below this line.
