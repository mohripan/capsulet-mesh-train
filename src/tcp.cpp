#include <meshtrain/tcp.hpp>

#include "linux/socket.hpp"
#include "protocol.hpp"

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace meshtrain {
namespace {

ErrorContext context_for(const RunConfig& config, std::string operation) {
    ErrorContext context;
    context.rank = config.rank;
    context.run_id = config.run_id;
    context.operation = std::move(operation);
    return context;
}

wire::Header read_header(const net::Socket& socket, const RunConfig& config, Deadline deadline,
                         const ErrorContext& context) {
    std::array<std::byte, wire::header_bytes> bytes{};
    net::receive_exact(socket, bytes, deadline, context);
    return wire::decode_header(bytes, config.max_payload_bytes, context);
}

std::vector<std::shared_ptr<net::Socket>> join(const RunConfig& config, const TcpOptions& options,
                                               Deadline deadline) {
    std::vector<std::shared_ptr<net::Socket>> peers(config.world_size);
    if (config.world_size == 1U) {
        return peers;
    }
    if (config.rank == 0U) {
        auto context = context_for(config, "listen");
        context.peer_endpoint =
            config.listen_address + ':' + std::to_string(config.endpoints[0].port);
        const auto listener = net::listen(config, options, deadline, context);
        for (std::uint32_t joined = 1; joined < config.world_size; ++joined) {
            context.operation = "accept_hello";
            context.peer.reset();
            context.peer_endpoint =
                config.listen_address + ':' + std::to_string(config.endpoints[0].port);
            auto socket =
                std::make_shared<net::Socket>(net::accept(listener, options, deadline, context));
            const auto header = read_header(*socket, config, deadline, context);
            context.peer = header.source;
            if (header.source < config.world_size) {
                const auto& endpoint = config.endpoints[header.source];
                context.peer_endpoint = endpoint.host + ':' + std::to_string(endpoint.port);
            }
            if (header.operation != wire::Operation::hello || header.source >= config.world_size ||
                peers[header.source] != nullptr) {
                throw EngineError(ErrorCode::protocol_mismatch,
                                  "expected HELLO from a distinct group rank", context);
            }
            std::vector<std::byte> payload(static_cast<std::size_t>(header.payload_bytes));
            net::receive_exact(*socket, payload, deadline, context);
            const auto hello = wire::decode_hello(payload, context);
            if (hello.world_size != config.world_size || hello.run_id != config.run_id ||
                hello.fingerprint != config.context_fingerprint ||
                hello.max_payload_bytes != config.max_payload_bytes) {
                throw EngineError(ErrorCode::protocol_mismatch,
                                  "HELLO group, run, context, or payload limit mismatch", context);
            }
            peers[header.source] = std::move(socket);
        }
        for (std::uint32_t rank = 1; rank < config.world_size; ++rank) {
            context.operation = "send_ready";
            context.peer = rank;
            const auto& endpoint = config.endpoints[rank];
            context.peer_endpoint = endpoint.host + ':' + std::to_string(endpoint.port);
            wire::Header ready;
            ready.operation = wire::Operation::ready;
            ready.destination = rank;
            const auto bytes = wire::encode_header(ready, config.max_payload_bytes, context);
            net::send_exact(*peers[rank], bytes, deadline, context);
        }
    } else {
        auto context = context_for(config, "connect_root");
        context.peer = 0;
        context.peer_endpoint =
            config.endpoints[0].host + ':' + std::to_string(config.endpoints[0].port);
        auto socket = std::make_shared<net::Socket>(
            net::connect(config.endpoints[0], options, deadline, context));
        const auto payload = wire::encode_hello(config);
        wire::Header hello;
        hello.operation = wire::Operation::hello;
        hello.source = config.rank;
        hello.payload_bytes = payload.size();
        context.operation = "send_hello";
        const auto bytes = wire::encode_header(hello, config.max_payload_bytes, context);
        net::send_exact(*socket, bytes, deadline, context);
        net::send_exact(*socket, payload, deadline, context);
        context.operation = "receive_ready";
        const auto ready = read_header(*socket, config, deadline, context);
        if (ready.operation != wire::Operation::ready || ready.destination != config.rank) {
            throw EngineError(ErrorCode::protocol_mismatch, "expected READY addressed to this rank",
                              context);
        }
        peers[0] = std::move(socket);
    }
    return peers;
}

class TcpTransport final : public ByteTransport {
  public:
    TcpTransport(RunConfig config, std::vector<std::shared_ptr<net::Socket>> peers)
        : config_(std::move(config)), peers_(std::move(peers)) {}
    ~TcpTransport() override { close(); }

    void send_bytes(std::uint32_t peer, std::span<const std::byte> bytes,
                    Deadline deadline) override {
        operate(peer, "send_bytes", [&](const net::Socket& socket, const ErrorContext& context) {
            net::send_exact(socket, bytes, deadline, context);
        });
    }
    void receive_bytes(std::uint32_t peer, std::span<std::byte> bytes, Deadline deadline) override {
        operate(peer, "receive_bytes", [&](const net::Socket& socket, const ErrorContext& context) {
            net::receive_exact(socket, bytes, deadline, context);
        });
    }
    void close() noexcept override {
        std::vector<std::shared_ptr<net::Socket>> closing;
        {
            const std::lock_guard lock(lifecycle_);
            if (closed_) {
                return;
            }
            closed_ = true;
            closing.swap(peers_);
        }
        for (const auto& socket : closing) {
            if (socket != nullptr) {
                socket->shutdown();
            }
        }
        // An active operation retains its own lease. Its descriptor cannot be
        // recycled until I/O exits, even if close races with poll or recv.
    }

  private:
    template <typename Function>
    void operate(std::uint32_t peer, std::string operation, Function&& function) {
        auto context = context_for(config_, std::move(operation));
        context.peer = peer;
        if (peer < config_.endpoints.size()) {
            const auto& endpoint = config_.endpoints[peer];
            context.peer_endpoint = endpoint.host + ':' + std::to_string(endpoint.port);
        }
        std::unique_lock operation_lock(operation_, std::try_to_lock);
        if (!operation_lock.owns_lock()) {
            throw EngineError(ErrorCode::invalid_state,
                              "concurrent transport operations are unsupported", context);
        }
        std::shared_ptr<net::Socket> socket;
        {
            const std::lock_guard lock(lifecycle_);
            if (closed_ || peer >= peers_.size() || peers_[peer] == nullptr) {
                throw EngineError(ErrorCode::invalid_state,
                                  "transport is closed or peer is not connected", context);
            }
            socket = peers_[peer];
        }
        try {
            std::forward<Function>(function)(*socket, context);
        } catch (...) {
            close();
            throw;
        }
    }

    const RunConfig config_;
    std::vector<std::shared_ptr<net::Socket>> peers_;
    std::mutex lifecycle_;
    std::mutex operation_;
    bool closed_{};
};

} // namespace

bool tcp_enabled() noexcept { return true; }

std::unique_ptr<ByteTransport> connect_tcp(const RunConfig& config, TcpOptions options) {
    const auto context = context_for(config, "startup");
    const auto deadline = net::deadline_after(config.startup_timeout, context);
    validate(config);
    if (options.send_buffer_bytes < 0 || options.receive_buffer_bytes < 0) {
        throw EngineError(ErrorCode::invalid_configuration,
                          "socket buffer requests must be nonnegative", context);
    }
    auto peers = join(config, options, deadline);
    net::check_deadline(deadline, context);
    return std::make_unique<TcpTransport>(config, std::move(peers));
}

} // namespace meshtrain
