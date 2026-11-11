#!/usr/bin/env bash
# Install one PINNED Node binary in Voder's tool directory; never modify system Node or Pi.
set -Eeuo pipefail
ROOT="${1:-/root/autodl-tmp}"
if [[ ! -d "$ROOT" || -L "$ROOT" ]]; then echo 'VODER_NODE_ROOT_INVALID' >&2; exit 2; fi
ROOT="$(cd "$ROOT" && pwd -P)"
VERSION='22.23.3'
ARCHIVE="node-v${VERSION}-linux-x64.tar.xz"
TOOL_PARENT="$ROOT/.voder-tools"
TARGET="$TOOL_PARENT/node-v${VERSION}-linux-x64"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
 echo 'VODER_NODE_UNSUPPORTED_PLATFORM: Linux x86_64 required for this bootstrap' >&2;exit 2
fi
valid_node() {
 [[ -x "$1" ]] || return 1
 local ver major minor patch
 ver="$("$1" --version 2>/dev/null || true)"
 [[ "$ver" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || return 1
 major="${BASH_REMATCH[1]}";minor="${BASH_REMATCH[2]}";patch="${BASH_REMATCH[3]}"
 (( major > 22 || (major == 22 && minor >= 19) ))
}
if [[ -x "$TARGET/bin/node" ]]; then
 [[ "$("$TARGET/bin/node" --version)" == "v${VERSION}" ]] || { echo 'VODER_NODE_MANAGED_VERSION_CONFLICT' >&2; exit 1; }
 echo "VODER_NODE_RUNTIME=PASS path=$TARGET/bin/node version=v${VERSION}"; exit 0
fi
if [[ -e "$TARGET" || -L "$TARGET" ]]; then echo 'VODER_NODE_TARGET_CONFLICT' >&2; exit 1; fi
if [[ -n "${VODER_NODE_BINARY:-}" ]] && valid_node "$VODER_NODE_BINARY"; then
 echo "VODER_NODE_RUNTIME=PASS existing=$VODER_NODE_BINARY version=$("$VODER_NODE_BINARY" --version)";exit 0
fi
if command -v node >/dev/null && valid_node "$(command -v node)"; then
 echo "VODER_NODE_RUNTIME=PASS system=$(command -v node) version=$(node --version)";exit 0
fi
if [[ -L "$TOOL_PARENT" ]];then echo 'VODER_NODE_UNSAFE_TOOLS_SYMLINK' >&2;exit 1;fi
mkdir -p "$TOOL_PARENT"
if [[ -L "$TOOL_PARENT" || ! -d "$TOOL_PARENT" ]];then echo 'VODER_NODE_UNSAFE_TOOLS_PATH' >&2;exit 1;fi
STAGE="$(mktemp -d "$TOOL_PARENT/.node-v22.23.3-stage-XXXXXXXX")"
clean_stage() {
 # Only delete our freshly allocated staging directory and never follow a changed path.
 if [[ -n "${STAGE:-}" && "$STAGE" == "$TOOL_PARENT"/.node-v22.23.3-stage-* && -d "$STAGE" && ! -L "$STAGE" ]];then
  rm -rf -- "$STAGE"
 fi
}
trap clean_stage EXIT
SOURCE_URL="https://nodejs.org/dist/v${VERSION}"
if [[ ! -x "$(command -v curl || true)" ]];then echo 'VODER_NODE_CURL_MISSING' >&2;exit 1;fi
curl --proto '=https' --tlsv1.2 -fsSL --retry 2 --connect-timeout 12 --max-time 180 "$SOURCE_URL/SHASUMS256.txt" -o "$STAGE/SHASUMS256.txt"
expected="$(awk -v name="$ARCHIVE" '$2==name && $1 ~ /^[0-9a-f]{64}$/ {print $1}' "$STAGE/SHASUMS256.txt")"
# Portable awk variants may not support {64}; validate explicitly instead.
if [[ ! "$expected" =~ ^[[:xdigit:]]{64}$ ]];then
 expected="$(awk -v name="$ARCHIVE" '$2==name{print $1}' "$STAGE/SHASUMS256.txt")"
fi
if [[ ! "$expected" =~ ^[[:xdigit:]]{64}$ ]];then echo 'VODER_NODE_OFFICIAL_CHECKSUM_MISSING' >&2;exit 1;fi
curl --proto '=https' --tlsv1.2 -fsSL --retry 2 --connect-timeout 12 --max-time 600 "$SOURCE_URL/$ARCHIVE" -o "$STAGE/$ARCHIVE"
printf '%s  %s\n' "$expected" "$STAGE/$ARCHIVE" | sha256sum -c -
mkdir -p "$STAGE/runtime/bin"
tar -xJf "$STAGE/$ARCHIVE" -C "$STAGE/runtime/bin" --strip-components=2 "node-v${VERSION}-linux-x64/bin/node"
if [[ "$("$STAGE/runtime/bin/node" --version)" != "v${VERSION}" ]];then echo 'VODER_NODE_BINARY_VERSION_MISMATCH' >&2;exit 1;fi
chmod 755 "$STAGE/runtime/bin/node"
[[ ! -e "$TARGET" && ! -L "$TARGET" ]] || { echo 'VODER_NODE_CONCURRENT_TARGET_EXISTS' >&2;exit 1; }
mv -T "$STAGE/runtime" "$TARGET"
"$TARGET/bin/node" --version
printf 'VODER_NODE_RUNTIME=PASS installed=%s/bin/node official_archive_sha256=%s\n' "$TARGET" "$expected"
