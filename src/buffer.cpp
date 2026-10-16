#include <meshtrain/buffer.hpp>

#include <limits>
#include <utility>

namespace meshtrain {

std::size_t checked_payload_bytes(std::size_t element_count, std::size_t max_payload_bytes,
                                  ErrorContext context) {
    if (context.operation.empty()) {
        context.operation = "validate_buffer";
    }
    if (max_payload_bytes == 0U) {
        throw EngineError(ErrorCode::invalid_buffer, "payload limit must be positive",
                          std::move(context));
    }
    if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw EngineError(ErrorCode::invalid_buffer, "FP32 payload byte count overflows size_t",
                          std::move(context));
    }
    const auto bytes = element_count * sizeof(float);
    if (bytes > max_payload_bytes) {
        throw EngineError(ErrorCode::invalid_buffer, "FP32 payload exceeds configured limit",
                          std::move(context));
    }
    return bytes;
}

} // namespace meshtrain
