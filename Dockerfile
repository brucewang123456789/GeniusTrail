# syntax=docker/dockerfile:1
# ------------------------------------------------------------
# CiteVizor – Production-friendly Dockerfile
# - Python 3.11 slim base (no CUDA; LLM served externally via vLLM)
# - System deps for OCR (tesseract) and pdffigures2 (Java + JAR)
# - Wheels-friendly pins via requirements.txt
# ------------------------------------------------------------

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System dependencies (runtime only; no compilers needed) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      tesseract-ocr \
      openjdk-17-jre-headless \
      fonts-dejavu-core \
      libglib2.0-0 libjpeg62-turbo libpng16-16 libopenjp2-7 \
      libxml2 libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

# Optional: install extra tesseract language packs (uncomment if needed)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       tesseract-ocr-eng tesseract-ocr-chi-sim \
#     && rm -rf /var/lib/apt/lists/*

# ---- pdffigures2 (for figure/table detection) ----
RUN mkdir -p /opt/pdffigures2 && \
    curl -fsSL -o /opt/pdffigures2/pdffigures2-assembly-0.2.2.jar \
      https://github.com/allenai/pdffigures2/releases/download/v0.2.2/pdffigures2-assembly-0.2.2.jar

# Default envs (can be overridden at runtime)
ENV PDFFIGURES2_JAR=/opt/pdffigures2/pdffigures2-assembly-0.2.2.jar \
    FITZ_DPI=200 \
    SERPER_ENDPOINT=https://google.serper.dev/scholar \
    VLLM_ENDPOINT=http://host.docker.internal:8000/v1

# ---- App layout ----
WORKDIR /app

# Copy & install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Create storage skeleton (matches schemas.py conventions)
RUN mkdir -p /app/storage/docs /app/storage/evidence/_rank /app/storage/renders /app/storage/reports

# Expose preview server port (optional human-in-the-loop)
EXPOSE 8787

# Healthcheck (no-op if preview server not running; returns healthy)
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8787/health || exit 0

# Default command: run the one-shot pipeline; override for other entrypoints
CMD ["python", "main.py"]
