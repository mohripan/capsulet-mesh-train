#include <meshtrain/tcp.hpp>

#include <meshtrain/error.hpp>

namespace meshtrain {

bool tcp_enabled() noexcept { return false; }

std::unique_ptr<ByteTransport> connect_tcp(const RunConfig&, TcpOptions) {
    throw EngineError(ErrorCode::invalid_configuration, "TCP support is disabled in this build");
}

} // namespace meshtrain
