#!/usr/bin/env bash
# Voder Core Rust quality gate. No source mutations; fails closed.
set -Eeuo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LOG_DIR="${VODER_AUDIT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/voder-core-verify.XXXXXXXX")}" 
mkdir -p -- "$LOG_DIR"
cd "$PROJECT"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PROJECT/target}"
fail() { echo "VODER_CORE_VERIFY=FAIL gate=$1 log=$LOG_DIR/$1.log" >&2; exit 1; }
for tool in cargo rustc sha256sum; do command -v "$tool" >/dev/null 2>&1 || { echo "MISSING_TOOL=$tool" >&2; exit 2; }; done
printf '%s\n' "$(rustc --version)" "$(cargo --version)" > "$LOG_DIR/toolchain.txt"
find src tests -type f -name '*.rs' -print0 | sort -z | xargs -0 sha256sum > "$LOG_DIR/before.sha256"
for f in types contract budget retry obligation evidence validation action_commit consequence decision verdict transition; do
    test -f "src/$f.rs" || { echo "MISSING_MODULE=$f" >&2; exit 3; }
    grep -Eq "^pub mod $f;" src/lib.rs || { echo "UNEXPORTED_MODULE=$f" >&2; exit 3; }
done
run_gate() {
    local name="$1"; shift
    if "$@" >"$LOG_DIR/$name.log" 2>&1; then echo "${name}=PASS"; else tail -40 "$LOG_DIR/$name.log" >&2; fail "$name"; fi
}
run_gate FORMAT cargo fmt --all -- --check
run_gate CHECK cargo check --locked --offline --all-targets
run_gate TESTS cargo test --locked --offline --all-targets
run_gate DOCTESTS cargo test --locked --offline --doc
run_gate CLIPPY cargo clippy --locked --offline --all-targets -- -D warnings
run_gate RELEASE cargo build --locked --offline --release
run_gate TEST_DISCOVERY cargo test --locked --offline --all-targets -- --list
TEST_COUNT="$(grep -c ': test$' "$LOG_DIR/TEST_DISCOVERY.log" || true)"
echo "DISCOVERED_TESTS=$TEST_COUNT"
test "$TEST_COUNT" -ge 167 || fail TEST_COUNT
find src tests -type f -name '*.rs' -print0 | sort -z | xargs -0 sha256sum > "$LOG_DIR/after.sha256"
cmp -s "$LOG_DIR/before.sha256" "$LOG_DIR/after.sha256" || fail SOURCE_UNCHANGED
printf 'DISCOVERED_TESTS=%s\nVODER_CORE_VERIFY=PASS\n' "$TEST_COUNT" > "$LOG_DIR/result.txt"
echo "SOURCE_UNCHANGED=PASS"
echo "VODER_CORE_VERIFY=PASS"
echo "AUDIT_DIR=$LOG_DIR"
