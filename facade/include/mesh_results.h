#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

// Counts and offsets exposed to JavaScript and WASI are signed 32-bit scalar
// counts. Check before accumulating vertices, triangles, edges, or shapes.
template <int Components>
int addMeshElements(int previous, std::size_t count) {
    static_assert(Components > 0);
    constexpr int limit = std::numeric_limits<int>::max() / Components;
    if (previous < 0 || previous > limit ||
        count > static_cast<std::size_t>(limit - previous)) {
        throw std::length_error("mesh result exceeds scalar count limit");
    }
    return previous + static_cast<int>(count);
}

template <int Components>
int meshScalarCount(std::size_t count) {
    return addMeshElements<Components>(0, count) * Components;
}

// All elements are written by the tessellator. Default-initialize primitive
// arrays (not value-initialize) to avoid clearing a large buffer twice.
template <typename T>
std::unique_ptr<T[]> allocateMeshBuffer(int count) {
    static_assert(std::is_trivial_v<T>);
    if (count < 0 || static_cast<std::size_t>(count) >
                         std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::length_error("mesh buffer exceeds addressable size");
    }
    if (count == 0) return {};
    return std::unique_ptr<T[]>(new T[static_cast<std::size_t>(count)]);
}

inline uint32_t wasmBufferAddress(const void* data) noexcept {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(data));
}

/// Mesh data returned from tessellation.
struct MeshData {
    std::unique_ptr<float[]> positions;
    std::unique_ptr<float[]> normals;
    std::unique_ptr<float[]> uvs;
    std::unique_ptr<uint32_t[]> indices;
    std::unique_ptr<int32_t[]> faceGroups; // [triStart, triCount, faceHash] per face
    int positionCount = 0;
    int normalCount = 0;
    int uvCount = 0;
    int indexCount = 0;
    int faceGroupCount = 0; // number of int32s (faceCount * 3)

    uint32_t getPositionsPtr() const { return wasmBufferAddress(positions.get()); }
    uint32_t getNormalsPtr() const { return wasmBufferAddress(normals.get()); }
    uint32_t getUvsPtr() const { return wasmBufferAddress(uvs.get()); }
    uint32_t getIndicesPtr() const { return wasmBufferAddress(indices.get()); }
    uint32_t getFaceGroupsPtr() const { return wasmBufferAddress(faceGroups.get()); }
};

/// Edge line data for wireframe rendering.
struct EdgeData {
    std::unique_ptr<float[]> points;
    std::unique_ptr<int32_t[]> edgeGroups; // [pointStart, pointCount, edgeHash] per edge
    int pointCount = 0;
    int edgeGroupCount = 0; // number of int32s (edgeCount * 3)

    uint32_t getPointsPtr() const { return wasmBufferAddress(points.get()); }
    uint32_t getEdgeGroupsPtr() const { return wasmBufferAddress(edgeGroups.get()); }
};

/// Batch mesh data: concatenated positions/normals/indices with per-shape offsets.
struct MeshBatchData {
    std::unique_ptr<float[]> positions;
    std::unique_ptr<float[]> normals;
    std::unique_ptr<uint32_t[]> indices;
    std::unique_ptr<int32_t[]> shapeOffsets; // [posStart, posCount, idxStart, idxCount] per shape
    int positionCount = 0;
    int normalCount = 0;
    int indexCount = 0;
    int shapeCount = 0; // number of shapes (shapeOffsets has shapeCount * 4 int32s)

    uint32_t getPositionsPtr() const { return wasmBufferAddress(positions.get()); }
    uint32_t getNormalsPtr() const { return wasmBufferAddress(normals.get()); }
    uint32_t getIndicesPtr() const { return wasmBufferAddress(indices.get()); }
    uint32_t getShapeOffsetsPtr() const { return wasmBufferAddress(shapeOffsets.get()); }
};
