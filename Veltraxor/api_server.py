"""
FastAPI server exposing Veltraxor as /chat.

Run:  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import logging
from datetime import datetime
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm_client import LLMClient
from dynamic_cot_controller import decide_cot, integrate_cot

# ─────────────────────────────── configuration ────────────────────────────────
MODEL_NAME = os.getenv("VELTRAX_MODEL", "grok-3-latest")
API_TOKEN  = os.getenv("VELTRAX_API_TOKEN")   # optional
SYSTEM_PROMPT = "You are Veltraxor, a concise yet rigorous reasoning assistant."

# basic JSON logger
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("veltraxor-api")

client = LLMClient(model=MODEL_NAME)
app = FastAPI(title="Veltraxor API", version="0.1.0")

# ──────────────────────────────── models ──────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, str]] | None = None  # optional

class ChatResponse(BaseModel):
    response: str
    used_cot: bool
    duration_ms: int

# ───────────────────────────── auth dependency ────────────────────────────────
def verify_token(request: Request):
    if API_TOKEN is None:       # auth disabled
        return
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing token")

# ───────────────────────────── endpoints ───────────────────────────────────────
@app.get("/healthz")
async def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(req: ChatRequest):
    t0 = datetime.utcnow()

    # dialogue context
    messages: List[Dict[str, str]] = req.history[:] if req.history else []
    messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": req.prompt})

    first = client.chat(messages)["choices"][0]["message"]["content"].strip()

    if decide_cot(req.prompt, first):
        log.info("Deep reasoning triggered")
        final = integrate_cot(client, SYSTEM_PROMPT, req.prompt, first)
        used = True
    else:
        final = first
        used = False

    dt = int((datetime.utcnow() - t0).total_seconds() * 1000)
    log.info(f"latency_ms={dt} used_cot={used}")

    return ChatResponse(response=final, used_cot=used, duration_ms=dt)

# ────────────────────────────── entry helper ──────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))