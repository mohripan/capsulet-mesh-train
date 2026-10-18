#pragma once

#include <memory>

#include <meshtrain/config.hpp>
#include <meshtrain/transport.hpp>

namespace meshtrain {

struct TcpOptions {
    // Zero preserves the OS default. Positive values request kernel buffer sizes.
    int send_buffer_bytes{};
    int receive_buffer_bytes{};
    bool no_delay{true};
};

[[nodiscard]] bool tcp_enabled() noexcept;

// Joins a Linux TCP star or ring selected by config. One operation at a time;
// close cancels active I/O. Ownership, topology and deadlines: docs/TCP_PROTOCOL.md.
[[nodiscard]] std::unique_ptr<ByteTransport> connect_tcp(const RunConfig& config,
                                                         TcpOptions options = {});

} // namespace meshtrain
