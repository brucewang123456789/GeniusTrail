#!/usr/bin/env bash
# Ephemeral, no-on-disk-credential model launcher. Run script with `bash model-env.sh -- <command> [args]`.
set -euo pipefail
if [[ ${1:-} != '--' || $# -lt 2 ]]; then
  echo 'Usage: bash model-env.sh -- <command> [args...]' >&2
  exit 2
fi
shift
if [[ -z ${VODER_MODEL_ID:-} ]]; then
  if [[ ! -t 0 ]]; then echo 'VODER_MODEL_ID missing; interactive terminal or environment required' >&2; exit 2; fi
  read -r -p 'Model ID: ' VODER_MODEL_ID
fi
if [[ -z ${VODER_MODEL_API_KEY:-} ]]; then
  if [[ ! -t 0 ]]; then echo 'VODER_MODEL_API_KEY missing; interactive terminal or environment required' >&2; exit 2; fi
  read -rs -p 'Model API Key (hidden): ' VODER_MODEL_API_KEY
  printf '\n' >&2
fi
if [[ -z "$VODER_MODEL_ID" || -z "$VODER_MODEL_API_KEY" ]]; then echo 'MODEL_CREDENTIALS_REQUIRED' >&2; exit 2; fi
export VODER_MODEL_ID VODER_MODEL_API_KEY
exec "$@"
