#!/usr/bin/env bash
# Build source checkout tools without modifying a global installation.
set -euo pipefail
umask 022
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 is required by this bootstrap.' >&2; exit 2; }
node -e 'const [major,minor]=process.versions.node.split(".").map(Number);if(major<22||(major===22&&minor<19))process.exit(1)' ||
  { echo 'Node 22.19 or newer is required.' >&2; exit 2; }
cargo --version | grep -q '^cargo 1\.96\.0 ' ||
  { echo 'Select Rust/Cargo 1.96.0 before running this script.' >&2; exit 2; }

npm ci --ignore-scripts --no-audit --no-fund
npm run build
for pkg in packages/voder-cli packages/voder-sandbox-modal; do
  npm ci --prefix "$pkg" --ignore-scripts --no-audit --no-fund
done
npm --prefix packages/voder-cli run build
for provider in cloud open; do
  pkg="packages/voder-provider-$provider"
  packages/voder-cli/node_modules/.bin/tsc -p "$pkg/tsconfig.json"
done
node --input-type=module <<'NODE'
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
const vendor = 'packages/voder-cli/vendor/';
const manifest = JSON.parse(readFileSync(vendor + 'MANIFEST.json'));
for (const [relative, expected] of Object.entries(manifest)) {
  const file = relative.includes('/dist/') ? 'packages/' + relative : vendor + relative;
  if (createHash('sha256').update(readFileSync(file)).digest('hex') !== expected) {
    throw new Error('Provider distribution does not match vendor manifest: ' + relative);
  }
}
NODE
for provider in cloud open; do
  dest="packages/voder-cli/vendor/voder-provider-$provider/dist"
  mkdir -p "$dest"
  cp "packages/voder-provider-$provider/dist/"*.js "$dest/"
done
playwright_stage="$(mktemp -d)"
trap 'rm -rf -- "$playwright_stage"' EXIT
npm pack playwright-core@1.61.1 --ignore-scripts --silent --pack-destination "$playwright_stage"
playwright="$playwright_stage/playwright-core-1.61.1.tgz"
printf '%s  %s\n' 3740bba3a3e93b4c7bd7c0f6e3c0e5f8cdce498bef13511e4d534cc99a8ce198 "$playwright" | sha256sum --check --status
install -m 644 "$playwright" voder-agent/packages/voder-vscode/src/vendor/playwright-core-1.61.1.tgz
engine="voder-agent/engine/voder-agent-engine"
npm ci --prefix "$engine" --ignore-scripts --no-audit --no-fund
npm --prefix "$engine" run build:offline
pi_modules="voder-agent/packages/voder-agent-core/node_modules"
pi_target="../../engine/voder-agent-engine/node_modules"
if [[ ! -e "$pi_modules" && ! -L "$pi_modules" ]]; then
  ln -s "$pi_target" "$pi_modules"
elif [[ "$(readlink "$pi_modules" || true)" != "$pi_target" ]]; then
  echo "Existing $pi_modules is managed separately. Run this bootstrap in a clean checkout." >&2
  exit 2
fi
node --input-type=module -e 'import {resolvePiImports} from "./voder-agent/packages/voder-agent-core/src/pi-imports.mjs"; await resolvePiImports()'

export CARGO_TARGET_DIR="$root/.voder-build"
cargo build --locked --release -p voderd
cargo build --locked --release --manifest-path voder-agent/crates/voder-gateway/Cargo.toml
cargo build --locked --release --manifest-path voder-agent/packages/voder-search/native/Cargo.toml
install -D -m 755 "$CARGO_TARGET_DIR/release/voder-retrieval" voder-agent/packages/voder-search/native/bin/voder-retrieval
install -D -m 755 "$(node -p 'process.execPath')" .voder-tools/node/bin/node
bash bin/voder --help
