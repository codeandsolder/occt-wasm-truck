#!/usr/bin/env bash
# Rebuild the facade and embedded module in the pinned native builder image.
set -euo pipefail
if [[ $(git rev-parse HEAD:Dockerfile.builder) != 0441ede58532884ba88c00a8b98f01726221d6ab || $(git rev-parse HEAD:scripts/fetch-rapidjson.sh) != 6ae38c36c61da3b9a412281529f0f940b73eed6e ]]; then
    echo 'OCCT native builder recipe changed: review the builder image digest and build prerequisites.' >&2
    exit 1
fi
ln -s /workspace/3rdparty 3rdparty
if [[ $(git rev-parse HEAD:occt) == c7f60ffe85f28c67358d1771454044f0521f2fdd ]]; then
    # Leave the tracked gitlink as a directory, not a symlink.
    mkdir -p occt
    ln -s /workspace/occt/build occt/build
else
    # A new native OCCT commit can use the same reviewed toolchain recipe.
    # Never link static libraries built from a different native revision.
    git submodule update --init --depth 1 occt
    cargo run --locked --package xtask -- build-occt
fi
test -d occt/build/include/opencascade
cargo run --locked --package xtask -- codegen
cargo fmt --all
cargo run --locked --package xtask -- build-wasi --release
# No changes inside the native submodule may leak into the candidate.
if [[ -f occt/.git ]]; then
    git -C occt diff --exit-code HEAD --
    git config submodule.occt.ignore untracked
fi
