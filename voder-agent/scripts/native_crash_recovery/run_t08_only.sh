#!/usr/bin/env bash
# Crash test is only on temporary worktrees. Root sync is strictly after real Rust success.
set -Eeuo pipefail
umask 077
ROOT=/root/autodl-tmp
PKG="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"
(cd "$PKG" && sha256sum -c SHA256SUMS)
python3 -B "$PKG/test_cargo_proxy.py"
python3 -B "$PKG/test_node_realpath.py"
[[ "${VODER_ALLOW_MODAL_CHARGES:-}" == YES && "${VODER_ENABLE_NATIVE_SIGKILL:-}" == YES ]] || { echo 'EXPLICIT_MODAL_AND_SIGKILL_CONSENT_REQUIRED'; exit 4; }
export VODER_ROOT="$ROOT"
export VODER_NODE_BINARY="${VODER_NODE_BINARY:-$ROOT/.voder-tools/node/bin/node}"
export VODER_MODAL_CLI="${VODER_MODAL_CLI:-$ROOT/packages/voder-sandbox-modal/src/cli.ts}"
EVID="$(mktemp -d "$ROOT/repair/.voder-t08-node-realpath-XXXXXXXX")"
chmod 700 "$EVID"
export VODER_TEST_EVIDENCE="$EVID"
echo "EVIDENCE_ROOT=$EVID"
set +e
python3 -B "$PKG/test_native_crash.py" 2>&1 | tee "$EVID/T08-run.log"
rc=${PIPESTATUS[0]}
set -e
if [[ "$rc" != 0 ]]; then
  echo 'T08_NATIVE_RUST_CRASH_ROLLBACK=NOT_PASSED'
  echo 'ROOT_T08_SCRIPT_SYNC=SKIPPED_VALIDATION_FAILED'
  echo 'T08_SDK_AUTOMATIC_SAFE_CONTINUATION=NOT_SUPPORTED'
  exit 4
fi
# Never update formal root scripts on the basis of a text-only PASS line:
# sync_to_root.py independently parses and verifies both recorded Rust cases.
python3 -B "$PKG/sync_to_root.py" --evidence "$EVID" --source "$PKG"
echo 'T08_NATIVE_RUST_CRASH_ROLLBACK=PASS'
echo 'T08_SDK_AUTOMATIC_SAFE_CONTINUATION=NOT_SUPPORTED'
echo 'VODER_T08_REALPATH_AND_ROOT_SYNC=PASS'
