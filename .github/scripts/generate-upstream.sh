#!/usr/bin/env bash
# Rebuild the facade and embedded module in the pinned native builder image.
set -euo pipefail
if [[ $(git rev-parse HEAD:Dockerfile.builder) != 7b93a81bc841fda9e7c62f91f0c7e36aaca9bc30 || $(git rev-parse HEAD:scripts/fetch-rapidjson.sh) != 6ae38c36c61da3b9a412281529f0f940b73eed6e ]]; then
    echo 'OCCT native builder recipe changed: review the builder image digest and build prerequisites.' >&2
    exit 1
fi
# The immutable builder includes native source patches that are not in the
# unmodified gitlink. A changed native revision must be rebuilt and repinned
# explicitly; compiling it here would silently discard those patches.
expected_native=$(cat patches/occt/gitlink)
if [[ $(git rev-parse HEAD:occt) != "$expected_native" ]]; then
    echo 'OCCT native revision changed: reconstruct/review source overlays, rebuild and repin the native image.' >&2
    exit 1
fi
# The image's actual patched source must match the checked-in source manifest.
root=$(pwd)
(cd /workspace/occt && sha256sum --check "$root/patches/occt/source.sha256")
ln -s /workspace/3rdparty 3rdparty
# Leave the tracked gitlink as a directory, not a symlink.
mkdir -p occt
ln -s /workspace/occt/build occt/build
test -d occt/build/include/opencascade
cargo run --locked --package xtask -- codegen
cargo fmt --all
cargo run --locked --package xtask -- build-wasi --release
# No changes inside the native submodule may leak into the candidate.
if [[ -f occt/.git ]]; then
    git -C occt diff --exit-code HEAD --
    git config submodule.occt.ignore untracked
fi
