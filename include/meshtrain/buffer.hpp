#pragma once

#include <cstddef>
#include <span>

#include <meshtrain/error.hpp>

namespace meshtrain {

// Borrowed writable CPU storage. The caller owns its lifetime and prevents
// concurrent mutation until the synchronous operation completes.
using HostFloat32Buffer = std::span<float>;

// Checks size arithmetic before any allocation or pointer access. Zero elements
// are valid; a zero limit, overflow, or an exceeded limit throws invalid_buffer.
[[nodiscard]] std::size_t checked_payload_bytes(std::size_t element_count,
                                                std::size_t max_payload_bytes,
                                                ErrorContext context = {});

} // namespace meshtrain
