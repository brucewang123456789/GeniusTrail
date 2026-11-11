#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$root"
python3 scripts/verify-source.py
node --test packages/voder-providers/test/*.test.mjs
node --test packages/voder-cli/test/*.test.mjs
node --test voder-agent/packages/voder-acceptance/test/*.test.mjs
# Slow archive/image suite is separate; no multi-hundred-MiB stress task in the fast smoke.
