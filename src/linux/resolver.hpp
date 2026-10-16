#pragma once

#include <netinet/in.h>

#include <meshtrain/config.hpp>
#include <meshtrain/error.hpp>
#include <meshtrain/transport.hpp>

#include <vector>

namespace meshtrain::net {

[[nodiscard]] std::vector<sockaddr_in> resolve_ipv4(const Endpoint& endpoint, Deadline deadline,
                                                    const ErrorContext& context,
                                                    bool allow_any = false);

} // namespace meshtrain::net
