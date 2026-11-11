#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$root"
export CARGO_TARGET_DIR="$root/.voder-build"
export VODER_AGENT_TEST_ROOT="$root/voder-agent"
export VODER_NATIVE_TEST_GATEWAY="$CARGO_TARGET_DIR/release/voder-gateway"
export VODER_CLI_TEST_PROVIDER_ROOT="$root"
export VODER_ARCHIVE_FIXTURES="$VODER_AGENT_TEST_ROOT/packages/voder-vscode/test/fixtures/archives"
export VODER_DOCUMENT_FIXTURES="$VODER_AGENT_TEST_ROOT/packages/voder-vscode/test/fixtures/documents"
export VODER_PDF_VISION_FIXTURES="$VODER_DOCUMENT_FIXTURES"
unset VODER_ALLOW_MODEL_API VODER_ALLOW_MODAL_CHARGES
unset OPENAI_API_KEY ANTHROPIC_API_KEY DEEPSEEK_API_KEY MODAL_TOKEN_ID MODAL_TOKEN_SECRET VODER_MODEL_API_KEY
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
test_home="$(mktemp -d)"
trap 'rm -rf -- "$test_home"' EXIT
export HOME="$test_home"
export XDG_CONFIG_HOME="$test_home/.config"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 scripts/verify-source.py
test -x "$VODER_NATIVE_TEST_GATEWAY"
test -x "$VODER_AGENT_TEST_ROOT/packages/voder-search/native/bin/voder-retrieval"
# The editor suite includes heavy Office/PDF/archive jobs. A low worker count
# avoids exhausting the memory of a clean Linux build host.
mapfile -d '' core_tests < <(find packages voder-agent/packages \
  -type d \( -name node_modules -o -name dist -o -name target -o -name voder-vscode \) -prune \
  -o -type f -path '*/test/*' \( -name '*.test.mjs' -o -name '*.test.cjs' \) -print0 | sort -z)
node --experimental-strip-types --test --test-reporter=tap --test-concurrency=4 "${core_tests[@]}"
node --test --test-reporter=tap --test-concurrency=2 voder-agent/packages/voder-vscode/test/*.test.{cjs,mjs}
cargo test --locked --workspace
cargo test --locked --manifest-path voder-agent/crates/voder-gateway/Cargo.toml
cargo test --locked --manifest-path voder-agent/packages/voder-search/native/Cargo.toml
