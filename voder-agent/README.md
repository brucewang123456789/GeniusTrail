# Voder Agent (development integration)

Canonical Voder Agent project root. Active JavaScript sources are under `packages/`:
`voder-agent-core`, `voder-sdk`, `voder-native-bridge`.
The official Pi 1.1.0 npm installation is local to `packages/voder-agent-core/node_modules`.
Native Rust VGR remains in the parent Voder workspace at `../crates`.
The `repair/` directory is for downloadable archives, fix packages and test materials only.
Run the offline tests from the root with:
`node --test packages/{voder-agent-core,voder-sdk,voder-native-bridge}/test/*.test.mjs`.
Real daemon smoke: `bash scripts/run_on_autodl.sh /root/autodl-tmp`.
This is not yet a validated model-to-VGR authorized execution/commit integration.
