# ---------- build stage ----------
FROM python:3.13-slim AS builder

WORKDIR /app

# Install runtime dependencies first (leverage cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Install package in editable mode inside the build image
RUN pip install --no-cache-dir .

# ---------- runtime stage ----------
FROM python:3.13-slim
WORKDIR /app

# Copy everything we just built, including site-packages and source tree
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

# Copy entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# Use the script above; default to CLI, override with api
ENTRYPOINT ["/entrypoint.sh"]
CMD []