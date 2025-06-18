#!/usr/bin/env bash
set -e

# If the first argument is "api", run the FastAPI server
if [[ "$1" == "api" ]]; then
  uvicorn api_server:app --host 0.0.0.0 --port 8000
# Otherwise treat every argument as a one-shot prompt to the CLI
else
  # Ensure /app is on PYTHONPATH so veltraxor module is importable
  export PYTHONPATH="/app:$PYTHONPATH"
  python -m veltraxor "$@"
fi