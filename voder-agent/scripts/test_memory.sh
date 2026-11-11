#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
node --check "$HERE/packages/voder-memory/src/index.mjs"
node --check "$HERE/packages/voder-memory/bin/voder-memory.mjs"
node --check "$HERE/packages/voder-sdk/src/index.mjs"
node --check "$HERE/packages/voder-agent-core/src/kernel.mjs"
node --check "$HERE/packages/voder-agent-core/src/pi-host.mjs"
node --test "$HERE/packages/voder-memory/test/memory.test.mjs"
echo 'VODER_MEMORY_OFFLINE_GATE=PASS (not a model/VS Code/VGR production gate)'
