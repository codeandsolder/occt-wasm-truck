#include "mesh_results.h"

#include <cassert>
#include <utility>

template <typename Result>
void checkOwnership(std::unique_ptr<float[]> Result::* buffer) {
    static_assert(std::is_nothrow_move_constructible_v<Result>);
    static_assert(std::is_nothrow_move_assignable_v<Result>);
    static_assert(!std::is_copy_constructible_v<Result>);
    static_assert(!std::is_copy_assignable_v<Result>);

    Result current;
    current.*buffer = allocateMeshBuffer<float>(3);
    (current.*buffer)[0] = 1.25f;
    const auto original = (current.*buffer).get();

    // If the operation throws, assignment has not started and the previous
    // complete result is still owned by the live destination.
    const auto failed = [buffer]() -> Result {
        Result temporary;
        temporary.*buffer = allocateMeshBuffer<float>(6);
        throw std::runtime_error("operation failed");
    };
    try {
        current = failed();
        assert(false);
    } catch (const std::runtime_error&) {
        assert((current.*buffer).get() == original);
        assert((current.*buffer)[0] == 1.25f);
    }

    Result moved = std::move(current);
    assert(!(current.*buffer));
    assert((moved.*buffer).get() == original);
    current = std::move(moved);
    assert(!(moved.*buffer));
    assert((current.*buffer).get() == original);

    current = Result{};
    assert(!(current.*buffer));
}

int main() {
    checkOwnership(&MeshData::positions);
    checkOwnership(&MeshBatchData::positions);
    checkOwnership(&EdgeData::points);

    constexpr int max = std::numeric_limits<int>::max();
    assert(meshScalarCount<3>(max / 3) == (max / 3) * 3);
    assert(addMeshElements<4>(max / 4 - 1, 1) == max / 4);
    try {
        (void)addMeshElements<3>(max / 3, 1);
        assert(false);
    } catch (const std::length_error&) {}
    try {
        (void)meshScalarCount<4>(std::numeric_limits<std::size_t>::max());
        assert(false);
    } catch (const std::length_error&) {}
    try {
        (void)allocateMeshBuffer<float>(-1);
        assert(false);
    } catch (const std::length_error&) {}
    assert(!allocateMeshBuffer<float>(0));
}
