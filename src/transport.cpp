#include <meshtrain/error.hpp>
#include <meshtrain/transport.hpp>

namespace meshtrain {

void ByteTransport::exchange_bytes(std::uint32_t, std::span<const std::byte>, std::uint32_t,
                                   std::span<std::byte>, Deadline) {
    throw EngineError(ErrorCode::invalid_configuration,
                      "transport does not support duplex exchange");
}

} // namespace meshtrain
