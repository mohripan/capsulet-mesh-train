#pragma once

#include <meshtrain/buffer.hpp>
#include <meshtrain/config.hpp>
#include <meshtrain/transport.hpp>

#include <functional>

namespace meshtrain::detail {

struct Chunk {
    std::size_t offset;
    std::size_t count;
};

[[nodiscard]] Chunk ring_chunk(std::size_t count, std::uint32_t world, std::uint32_t chunk);
void ring_all_reduce(const RunConfig& config, ByteTransport& transport, HostFloat32Buffer buffer,
                     Deadline deadline, ErrorContext& context, const std::function<void()>& check);

} // namespace meshtrain::detail
