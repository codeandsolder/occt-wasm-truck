#!/usr/bin/env bash
set -euo pipefail
# In release mode the tag is a version contract, not a branch name.
if [[ -n ${3:-} ]]; then
    version=$(cargo metadata --locked --no-deps --format-version 1 | jq -er '.packages[] | select(.name == "occt-wasm") | .version')
    test "$3" = "crate-v$version"
fi
cargo run --locked --package xtask -- codegen
cargo fmt --all
git diff --exit-code -- facade/generated/ crate/src/kernel_generated.rs
# Never let the upstream placeholder/debug test skips pass this gate.
OCCT_WASM_REQUIRE_KERNEL=1 cargo test --locked --release -p occt-wasm --test integration --test validation_bindings -- --test-threads=4
