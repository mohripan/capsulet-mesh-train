#pragma once

#include "linux/socket.hpp"

#include <memory>
#include <vector>

namespace meshtrain::net {

using SocketTable = std::vector<std::shared_ptr<Socket>>;

struct JoinedGroup {
    SocketTable control;
    SocketTable data;
};

[[nodiscard]] ErrorContext peer_context(const RunConfig& config, std::uint32_t peer,
                                        std::string operation);
[[nodiscard]] JoinedGroup join_tcp(const RunConfig& config, const TcpOptions& options,
                                   Deadline deadline);

} // namespace meshtrain::net
