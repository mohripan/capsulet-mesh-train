#include <meshtrain/tcp.hpp>

#include "linux/socket.hpp"
#include "tcp_startup.hpp"

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

class TcpTransport final : public ByteTransport {
  public:
    TcpTransport(RunConfig config, net::JoinedGroup group)
        : config_(std::move(config)), group_(std::move(group)) {}
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
    void exchange_bytes(std::uint32_t send_peer, std::span<const std::byte> outgoing,
                        std::uint32_t receive_peer, std::span<std::byte> incoming,
                        Deadline deadline) override {
        const auto sending = net::peer_context(config_, send_peer, "exchange.send");
        const auto receiving = net::peer_context(config_, receive_peer, "exchange.receive");
        std::unique_lock operation_lock(operation_, std::try_to_lock);
        if (!operation_lock.owns_lock()) {
            throw EngineError(ErrorCode::invalid_state,
                              "concurrent transport operations are unsupported", sending);
        }
        std::shared_ptr<net::Socket> sender;
        std::shared_ptr<net::Socket> receiver;
        {
            const std::lock_guard lock(lifecycle_);
            sender = lease(send_peer, sending);
            receiver = lease(receive_peer, receiving);
        }
        try {
            net::exchange_exact(*sender, outgoing, *receiver, incoming, deadline, sending,
                                receiving);
        } catch (...) {
            close();
            throw;
        }
    }
    void close() noexcept override {
        net::JoinedGroup closing;
        {
            const std::lock_guard lock(lifecycle_);
            if (closed_) {
                return;
            }
            closed_ = true;
            closing.control.swap(group_.control);
            closing.data.swap(group_.data);
        }
        for (const auto* table : {&closing.control, &closing.data}) {
            for (const auto& socket : *table) {
                if (socket != nullptr) {
                    socket->shutdown();
                }
            }
        }
    }

  private:
    std::shared_ptr<net::Socket> lease(std::uint32_t peer, const ErrorContext& context) const {
        if (closed_ || peer >= group_.data.size() || group_.data[peer] == nullptr) {
            throw EngineError(ErrorCode::invalid_state,
                              "transport is closed or peer is not connected", context);
        }
        return group_.data[peer];
    }
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
            socket = lease(peer, context);
        }
        try {
            std::forward<Function>(function)(*socket, context);
        } catch (...) {
            close();
            throw;
        }
    }

    const RunConfig config_;
    net::JoinedGroup group_;
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
    auto group = net::join_tcp(config, options, deadline);
    net::check_deadline(deadline, context);
    return std::make_unique<TcpTransport>(config, std::move(group));
}

} // namespace meshtrain
