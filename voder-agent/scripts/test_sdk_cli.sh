#!/usr/bin/env bash
# Exact release smoke/contract suite; no paid model calls or VGR commit claims.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
NODE="${VODER_NODE_BINARY:-$(command -v node)}"
"$NODE" --check "$HERE/packages/voder-sdk/src/index.mjs"
"$NODE" --check "$HERE/packages/voder-sdk/src/control-plane.mjs"
"$NODE" --check "$HERE/packages/voder-agent-core/src/kernel.mjs"
"$NODE" --check "$HERE/packages/voder-cli/bin/voder-cli.mjs"
bash -n "$ROOT/bin/voder"
test -x "$ROOT/bin/voder" || { echo 'VODER_BIN_EXECUTABLE_GATE=FAIL' >&2; exit 1; }
# Use a compatible runtime for actual executable entrypoint (not a node-only syntax check).
"$ROOT/bin/voder" help >/dev/null
"$NODE" --test "$HERE/packages/voder-sdk/test/control-plane.test.mjs"
VODER_NODE_BINARY="$NODE" bash "$HERE/scripts/test_memory.sh"
"$NODE" "$HERE/packages/voder-cli/bin/voder-cli.mjs" help >/dev/null
printf 'VODER_SDK_CLI_OFFLINE_GATE=PASS (mock sessions, real file IO, live process protocol; NOT model or native VGR E2E)\n'
