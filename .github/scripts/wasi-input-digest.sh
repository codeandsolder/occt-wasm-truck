#!/usr/bin/env bash
set -euo pipefail
# Commit-tree identity, independent of local timestamps and clone location.
# Include the native source pin, facade/code generator, build recipe and
# embedded output; a source change must have an explicitly rebuilt artifact.
git ls-tree -r HEAD -- occt .gitmodules Dockerfile.builder facade xtask .cargo/config.toml crate/src/kernel_generated.rs crate/src/occt-wasm.wasm.br |
    sha256sum | cut -d ' ' -f 1
