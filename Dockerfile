# Dockerfile — full environment for local & CI test runs

FROM python:3.10-slim

# 1. Set working directory
WORKDIR /app

# 2. Install system-level tools & clients needed for testing
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       netcat-openbsd \
       redis-tools \
       postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy dependency definitions
COPY requirements.txt setup.py pytest.ini ./

# 4. Install Python dependencies (runtime + dev/test)
RUN python -m pip install --upgrade pip \
    && pip install -e .[dev] \
    && python -m pip cache purge

# 5. Copy application source
COPY . .

# 6. Default command: run pytest under test/ directory
CMD ["pytest", "test", "-q"]
