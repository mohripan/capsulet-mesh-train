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

// Linux implementation. Blocks until every rank joins or startup fails.
// Rank 0 accepts every other rank; nonzero ranks connect only to rank 0.
// Calls perform exact byte transfers, not collective computation or framing.
// One send/receive operation at a time; concurrent operations are rejected.
// close() may run concurrently with I/O and is idempotent. Join active callers
// before destroying the transport. Failed I/O permanently closes the group.
[[nodiscard]] std::unique_ptr<ByteTransport> connect_tcp(const RunConfig& config,
                                                         TcpOptions options = {});

} // namespace meshtrain
