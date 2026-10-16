#include "socket.hpp"

#include "resolver.hpp"

#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <limits>
#include <string>
#include <system_error>

namespace meshtrain::net {
namespace {

[[noreturn]] void system_error(int error, const ErrorContext& context) {
    const auto code = error == EPIPE || error == ECONNRESET || error == ENOTCONN
                          ? ErrorCode::peer_closed
                          : ErrorCode::transport_failure;
    throw EngineError(code, std::error_code(error, std::generic_category()).message(), context);
}

void wait_for(int descriptor, short events, Deadline deadline, const ErrorContext& context) {
    while (true) {
        check_deadline(deadline, context);
        const auto left = std::chrono::ceil<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        const auto timeout = static_cast<int>(std::clamp<std::int64_t>(left.count(), 1, INT_MAX));
        pollfd watched{descriptor, events, 0};
        const auto result = ::poll(&watched, 1, timeout);
        if (result > 0) {
            if ((watched.revents & POLLNVAL) != 0) {
                throw EngineError(ErrorCode::transport_failure, "invalid socket descriptor",
                                  context);
            }
            return; // Let recv/send/SO_ERROR distinguish data, closure, and errors.
        }
        if (result < 0 && errno != EINTR) {
            system_error(errno, context);
        }
    }
}

void options_for(const Socket& socket, const TcpOptions& options, const ErrorContext& context) {
    const auto set = [&](int level, int option, int value) {
        if (::setsockopt(socket.descriptor(), level, option, &value, sizeof(value)) != 0) {
            system_error(errno, context);
        }
    };
    set(IPPROTO_TCP, TCP_NODELAY, options.no_delay ? 1 : 0);
    if (options.send_buffer_bytes > 0) {
        set(SOL_SOCKET, SO_SNDBUF, options.send_buffer_bytes);
    }
    if (options.receive_buffer_bytes > 0) {
        set(SOL_SOCKET, SO_RCVBUF, options.receive_buffer_bytes);
    }
}

Socket make_socket(const TcpOptions& options, const ErrorContext& context) {
    const auto descriptor =
        ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, IPPROTO_TCP);
    if (descriptor < 0) {
        system_error(errno, context);
    }
    Socket socket(descriptor);
    options_for(socket, options, context);
    return socket;
}

} // namespace

Socket::~Socket() {
    if (descriptor_ >= 0) {
        static_cast<void>(::close(descriptor_));
    }
}

Socket& Socket::operator=(Socket&& other) noexcept {
    if (this != &other) {
        if (descriptor_ >= 0) {
            static_cast<void>(::close(descriptor_));
        }
        descriptor_ = std::exchange(other.descriptor_, -1);
    }
    return *this;
}

void Socket::shutdown() noexcept {
    if (descriptor_ >= 0) {
        static_cast<void>(::shutdown(descriptor_, SHUT_RDWR));
    }
}

Deadline deadline_after(std::chrono::milliseconds timeout, const ErrorContext& context) {
    using Clock = std::chrono::steady_clock;
    if (timeout.count() <= 0 ||
        timeout > std::chrono::duration_cast<std::chrono::milliseconds>(Clock::duration::max())) {
        throw EngineError(ErrorCode::invalid_configuration,
                          "timeout cannot be represented by the monotonic clock", context);
    }
    const auto duration = std::chrono::duration_cast<Clock::duration>(timeout);
    const auto now = Clock::now();
    if (now.time_since_epoch() > Clock::duration::max() - duration) {
        throw EngineError(ErrorCode::invalid_configuration,
                          "deadline would overflow the monotonic clock", context);
    }
    return now + duration;
}

void check_deadline(Deadline deadline, const ErrorContext& context) {
    if (std::chrono::steady_clock::now() >= deadline) {
        throw EngineError(ErrorCode::timeout, "operation deadline expired", context);
    }
}

Socket listen(const RunConfig& config, const TcpOptions& options, Deadline deadline,
              const ErrorContext& context) {
    const auto addresses =
        resolve_ipv4({config.listen_address, config.endpoints[0].port}, deadline, context, true);
    int last_error = EADDRNOTAVAIL;
    for (const auto& address : addresses) {
        check_deadline(deadline, context);
        auto socket = make_socket(options, context);
        const int reuse = 1;
        if (::setsockopt(socket.descriptor(), SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) !=
            0) {
            system_error(errno, context);
        }
        if (::bind(socket.descriptor(), reinterpret_cast<const sockaddr*>(&address),
                   sizeof(address)) != 0) {
            last_error = errno;
            continue;
        }
        if (::listen(socket.descriptor(), SOMAXCONN) != 0) {
            system_error(errno, context);
        }
        return socket;
    }
    system_error(last_error, context);
}

Socket accept(const Socket& listener, const TcpOptions& options, Deadline deadline,
              const ErrorContext& context) {
    while (true) {
        check_deadline(deadline, context);
        const auto descriptor =
            ::accept4(listener.descriptor(), nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (descriptor >= 0) {
            Socket socket(descriptor);
            options_for(socket, options, context);
            return socket;
        }
        if (errno == EINTR || errno == ECONNABORTED) {
            continue;
        }
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            system_error(errno, context);
        }
        wait_for(listener.descriptor(), POLLIN, deadline, context);
    }
}

Socket connect(const Endpoint& endpoint, const TcpOptions& options, Deadline deadline,
               const ErrorContext& context) {
    const auto addresses = resolve_ipv4(endpoint, deadline, context);
    while (true) {
        for (const auto& address : addresses) {
            check_deadline(deadline, context);
            auto socket = make_socket(options, context);
            const auto result = ::connect(
                socket.descriptor(), reinterpret_cast<const sockaddr*>(&address), sizeof(address));
            if (result == 0) {
                return socket;
            }
            auto error = errno;
            if (error == EINPROGRESS || error == EINTR) {
                const auto attempt_deadline = std::min(
                    deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds{250});
                try {
                    wait_for(socket.descriptor(), POLLOUT, attempt_deadline, context);
                } catch (const EngineError& failure) {
                    if (failure.code() != ErrorCode::timeout) {
                        throw;
                    }
                    continue;
                }
                socklen_t size = sizeof(error);
                if (::getsockopt(socket.descriptor(), SOL_SOCKET, SO_ERROR, &error, &size) != 0) {
                    system_error(errno, context);
                }
                if (error == 0) {
                    return socket;
                }
            }
            if (error != ECONNREFUSED && error != ETIMEDOUT && error != EHOSTUNREACH &&
                error != ENETUNREACH && error != ECONNRESET) {
                system_error(error, context);
            }
        }
        check_deadline(deadline, context);
        const auto left = std::chrono::ceil<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        const auto pause = static_cast<int>(std::clamp<std::int64_t>(left.count(), 1, 10));
        if (::poll(nullptr, 0, pause) < 0 && errno != EINTR) {
            system_error(errno, context);
        }
    }
}

void send_exact(const Socket& socket, std::span<const std::byte> bytes, Deadline deadline,
                const ErrorContext& context) {
    check_deadline(deadline, context);
    while (!bytes.empty()) {
        check_deadline(deadline, context);
        const auto chunk = std::min(bytes.size(), static_cast<std::size_t>(SSIZE_MAX));
        const auto written = ::send(socket.descriptor(), bytes.data(), chunk, MSG_NOSIGNAL);
        if (written > 0) {
            bytes = bytes.subspan(static_cast<std::size_t>(written));
            continue;
        }
        if (written == 0) {
            throw EngineError(ErrorCode::peer_closed, "socket made no send progress", context);
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            system_error(errno, context);
        }
        wait_for(socket.descriptor(), POLLOUT, deadline, context);
    }
}

void receive_exact(const Socket& socket, std::span<std::byte> bytes, Deadline deadline,
                   const ErrorContext& context) {
    check_deadline(deadline, context);
    while (!bytes.empty()) {
        check_deadline(deadline, context);
        const auto chunk = std::min(bytes.size(), static_cast<std::size_t>(SSIZE_MAX));
        const auto received = ::recv(socket.descriptor(), bytes.data(), chunk, 0);
        if (received > 0) {
            bytes = bytes.subspan(static_cast<std::size_t>(received));
            continue;
        }
        if (received == 0) {
            throw EngineError(ErrorCode::peer_closed, "peer closed the byte stream", context);
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            system_error(errno, context);
        }
        wait_for(socket.descriptor(), POLLIN, deadline, context);
    }
}

} // namespace meshtrain::net
