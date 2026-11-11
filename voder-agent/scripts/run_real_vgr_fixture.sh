#!/usr/bin/env bash
# Operator-only, existing Rust VGR Phase6 deterministic fixture: NOT model-driven integration.
set -Eeuo pipefail
ROOT="${1:-/root/autodl-tmp}"
if [[ "${VODER_RUN_NATIVE_MODAL_FIXTURE:-}" != "YES" ]]; then
  echo "NATIVE_MODAL_FIXTURE=NOT_RUN: set VODER_RUN_NATIVE_MODAL_FIXTURE=YES to approve paid remote sandbox calls" >&2
  exit 2
fi
[[ -f "$ROOT/voder-env.sh" ]] || { echo "MISSING_VODER_ENV" >&2; exit 2; }
# shellcheck disable=SC1090
source "$ROOT/voder-env.sh"
command -v cargo >/dev/null && command -v node >/dev/null || { echo "MISSING_NODE_OR_CARGO" >&2; exit 2; }
[[ -n "${VODER_MODAL_CLI:-}" && -f "$VODER_MODAL_CLI" ]] || { echo "VODER_MODAL_CLI_NOT_CONFIGURED" >&2; exit 2; }
export VODER_NODE_BINARY="$(command -v node)"
export VODER_VGR6_EVIDENCE_DIR="$ROOT/voder-agent/evidence/native-fixture"
mkdir -p "$VODER_VGR6_EVIDENCE_DIR"
cd "$ROOT"
echo '===== REAL PINNED RUST VGR FIXTURE (NOT MODEL DRIVEN) ====='
cargo run --locked --release --offline -p voderd --example vgr_modal_write_closure -- --self-check
cargo run --locked --release --offline -p voderd --example vgr_modal_write_closure
printf 'VODER_RUST_VGR_PINNED_FIXTURE_RC=0\n'
printf 'PI_MODEL_TO_NATIVE_EXECUTION_GATE=NOT_TESTED\n'
printf 'PRODUCTION_GATE=NOT_PASSED\n'
