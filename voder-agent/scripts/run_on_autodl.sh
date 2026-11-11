#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="${1:-/root/autodl-tmp}"
PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
echo '===== Native protocol bridge: isolated real-daemon smoke ====='
for tool in cargo node git; do command -v "$tool" >/dev/null || { echo "MISSING_TOOL=$tool"; exit 10; }; done
cd "$ROOT"
if [[ ! -x "$ROOT/target/debug/voderd" ]]; then
  echo '===== Building unchanged VGR daemon to debug target ====='
  cargo build -p voderd
fi
# Smoke script expects the standalone observer module under the addon directory.
node "$PACKAGE/scripts/live_cloud_probe.mjs" "$ROOT" "$ROOT/target/debug/voderd"
echo 'CURRENT_TERMINAL_REMAINS_OPEN=True'
