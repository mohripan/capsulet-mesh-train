#pragma once

#include <meshtrain/config.hpp>
#include <meshtrain/error.hpp>
#include <meshtrain/tcp.hpp>

#include <span>
#include <utility>

namespace meshtrain::net {

class Socket {
  public:
    Socket() = default;
    explicit Socket(int descriptor) noexcept : descriptor_(descriptor) {}
    ~Socket();
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& other) noexcept : descriptor_(std::exchange(other.descriptor_, -1)) {}
    Socket& operator=(Socket&& other) noexcept;

    [[nodiscard]] int descriptor() const noexcept { return descriptor_; }
    void shutdown() noexcept;

  private:
    int descriptor_{-1};
};

[[nodiscard]] Deadline deadline_after(std::chrono::milliseconds timeout,
                                      const ErrorContext& context);
void check_deadline(Deadline deadline, const ErrorContext& context);
[[nodiscard]] Socket listen(const RunConfig& config, const TcpOptions& options, Deadline deadline,
                            const ErrorContext& context);
[[nodiscard]] Socket accept(const Socket& listener, const TcpOptions& options, Deadline deadline,
                            const ErrorContext& context);
[[nodiscard]] Socket connect(const Endpoint& endpoint, const TcpOptions& options, Deadline deadline,
                             const ErrorContext& context);
void send_exact(const Socket& socket, std::span<const std::byte> bytes, Deadline deadline,
                const ErrorContext& context);
void receive_exact(const Socket& socket, std::span<std::byte> bytes, Deadline deadline,
                   const ErrorContext& context);

} // namespace meshtrain::net
