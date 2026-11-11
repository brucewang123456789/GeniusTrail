#!/usr/bin/env bash
# Module 08 deliberately defaults to read-only evidence mode.
set -uo pipefail
voder_module8_accept() {
  local ROOT
  if ! ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"; then
    printf '%s\n' 'VODER_MODULE8=BLOCKED reason=invalid_agent_root' >&2
    return 4
  fi
  local PKG="$ROOT/packages/voder-acceptance"
  if [[ ! -f "$PKG/src/audit.mjs" || ! -f "$PKG/src/live.mjs" ]]; then
    printf '%s\n' 'VODER_MODULE8=BLOCKED reason=acceptance_module_not_installed' >&2
    return 4
  fi
  local NODE="${VODER_NODE_BINARY:-}"
  if [[ -z "$NODE" ]]; then NODE="$(command -v node || true)"; fi
  if [[ -z "$NODE" ]]; then
    printf '%s\n' 'VODER_MODULE8=BLOCKED reason=node_missing' >&2
    return 4
  fi
  export VODER_AGENT_ROOT="$ROOT"
  local operation="${1:-}"
  case "$operation" in
    audit) "$NODE" "$PKG/src/audit.mjs"; return $? ;;
    selftest) "$NODE" --test "$PKG/test/fixture.test.mjs"; return $? ;;
    live) "$NODE" "$PKG/src/live.mjs"; return $? ;;
    release) shift; "$NODE" "$PKG/src/release.mjs" "$@"; return $? ;;
    *) printf '%s\n' 'Usage: bash module8_accept.sh {selftest|audit|live|release}' >&2; return 2 ;;
  esac
}
voder_module8_accept "$@"
