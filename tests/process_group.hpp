#pragma once

#include "linux/socket.hpp"
#include "tcp_fixture.hpp"

#include <arpa/inet.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <string>
#include <thread>
#include <utility>

extern char** environ;

namespace test {
inline std::uint16_t unused_port() {
    meshtrain::net::Socket socket(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
    require(socket.descriptor() >= 0, "cannot create port reservation");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    require(::bind(socket.descriptor(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
                0,
            "cannot reserve ephemeral port");
    socklen_t size = sizeof(address);
    require(::getsockname(socket.descriptor(), reinterpret_cast<sockaddr*>(&address), &size) == 0,
            "cannot inspect ephemeral port");
    // The kernel owns allocation; release immediately before spawning the group.
    return ntohs(address.sin_port);
}

class Children {
  public:
    explicit Children(std::string executable) : executable_(std::move(executable)) {}
    Children(const Children&) = delete;
    Children& operator=(const Children&) = delete;
    Children(Children&&) = delete;
    Children& operator=(Children&&) = delete;
    ~Children() {
        for (const auto child : children_) {
            if (child.pid > 0) {
                static_cast<void>(::kill(child.pid, SIGKILL));
                while (::waitpid(child.pid, nullptr, 0) < 0 && errno == EINTR) {
                }
            }
        }
    }
    void start(std::string_view scenario, std::uint32_t rank, std::uint32_t world,
               std::uint16_t port, int expected_signal = 0) {
        std::array<std::string, 8> arguments{executable_,           "--worker",
                                             std::string(scenario), std::to_string(rank),
                                             std::to_string(world), std::to_string(port),
                                             "localhost",           ""};
        std::array<char*, 8> raw{};
        for (std::size_t index = 0; index + 1U < arguments.size(); ++index) {
            raw[index] = arguments[index].data();
        }
        children_.push_back({-1, expected_signal}); // Allocate tracking before spawning.
        auto& child = children_.back();
        pid_t spawned = -1;
        const auto result =
            ::posix_spawn(&spawned, executable_.c_str(), nullptr, nullptr, raw.data(), environ);
        require(result == 0, "posix_spawn failed");
        child.pid = spawned; // Only a successful spawn establishes ownership.
    }
    void finish(meshtrain::Deadline deadline = after(8s)) {
        std::size_t remaining = children_.size();
        while (remaining > 0) {
            for (auto& child : children_) {
                if (child.pid <= 0) {
                    continue;
                }
                int status = 0;
                const auto result = ::waitpid(child.pid, &status, WNOHANG);
                if (result == child.pid) {
                    child.pid = -1;
                    --remaining;
                    require(child.expected_signal == 0
                                ? WIFEXITED(status) && WEXITSTATUS(status) == 0
                                : WIFSIGNALED(status) && WTERMSIG(status) == child.expected_signal,
                            "worker failed or died from a signal");
                } else {
                    require(result == 0 || errno == EINTR, "waitpid failed");
                }
            }
            require(std::chrono::steady_clock::now() < deadline, "subprocess group hung");
            std::this_thread::sleep_for(2ms);
        }
    }

  private:
    std::string executable_;
    struct Child {
        pid_t pid;
        int expected_signal;
    };
    std::vector<Child> children_;
};

} // namespace test
