# Reviewed native OCCT source overlays

These three patches are the **exact differences** between the unmodified
`c7f60ffe85f28c67358d1771454044f0521f2fdd` OCCT source and the reviewed
`ghcr.io/andymai/occt-wasm-builder` native libraries. They reproduce the
patched source files byte-for-byte, checked against `source.sha256`.

The image digest, package outputs, reasoning and validation corpus are tracked
in the [native builder provenance record](https://app.notion.com/p/421f74be90208159bcdce02f50782dd0).
The numeric fixes are already submitted upstream:
[OCCT #1793](https://github.com/Open-Cascade-SAS/OCCT/pull/1793)
and [OCCT #1794](https://github.com/Open-Cascade-SAS/OCCT/pull/1794).
`BOPAlgo_PaveFiller_6.patch` overlaps the separate, broader
[OCCT #1750](https://github.com/Open-Cascade-SAS/OCCT/pull/1750).

To validate the exact source without compiling the entire native library:

```sh
git submodule update --init --depth 1 occt
test "$(git rev-parse HEAD:occt)" = "$(cat patches/occt/gitlink)"
(
  cd occt
  git apply --check ../patches/occt/*.patch
  git apply ../patches/occt/*.patch
  sha256sum --check ../patches/occt/source.sha256
)
```

Do this only in a clean, disposable checkout: the example deliberately modifies
the checked-out submodule source. CI uses an ephemeral checkout. A full native
image rebuild is still required before changing the pinned immutable OCI digest.
An upstream submodule change must be paired with a reviewed patch rebase or
removal, updated checksums, a new native image, and regenerated WASI guest.
