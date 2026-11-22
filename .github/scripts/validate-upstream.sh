#!/usr/bin/env bash
set -euo pipefail
expected=$(cat .github/scripts/wasi-input-digest.sha256)
actual=$(bash .github/scripts/wasi-input-digest.sh)
if [[ "$expected" != "$actual" ]]; then
    echo 'OCCT WASI build inputs or output changed. Reconcile and regenerate the embedded kernel before tracking this snapshot.' >&2
    exit 1
fi
cargo xtask codegen
cargo fmt --all
git diff --exit-code -- facade/generated/ crate/src/kernel_generated.rs
# Never let the upstream placeholder/debug test skips pass this gate.
OCCT_WASM_REQUIRE_KERNEL=1 cargo test --locked --release -p occt-wasm --test integration --test validation_bindings -- --test-threads=4
