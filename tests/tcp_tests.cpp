#include "tcp_fixture.hpp"

#include "linux/socket.hpp"

#include <arpa/inet.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <utility>

extern char** environ;

namespace {
using namespace test;

std::uint16_t unused_port() {
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
    ~Children() {
        for (const auto child : children_) {
            if (child > 0) {
                static_cast<void>(::kill(child, SIGKILL));
                while (::waitpid(child, nullptr, 0) < 0 && errno == EINTR) {
                }
            }
        }
    }
    void start(std::string_view scenario, std::uint32_t rank, std::uint32_t world,
               std::uint16_t port) {
        std::array<std::string, 8> arguments{executable_,           "--worker",
                                             std::string(scenario), std::to_string(rank),
                                             std::to_string(world), std::to_string(port),
                                             "localhost",           ""};
        std::array<char*, 8> raw{};
        for (std::size_t index = 0; index + 1U < arguments.size(); ++index) {
            raw[index] = arguments[index].data();
        }
        pid_t child = -1;
        const auto result =
            ::posix_spawn(&child, executable_.c_str(), nullptr, nullptr, raw.data(), environ);
        require(result == 0, "posix_spawn failed");
        children_.push_back(child);
    }
    void finish() {
        const auto deadline = after(8s);
        std::size_t remaining = children_.size();
        while (remaining > 0) {
            for (auto& child : children_) {
                if (child <= 0) {
                    continue;
                }
                int status = 0;
                const auto result = ::waitpid(child, &status, WNOHANG);
                if (result == child) {
                    child = -1;
                    --remaining;
                    require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
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
    std::vector<pid_t> children_;
};

void run(const std::string& executable, std::string_view scenario, std::uint32_t world) {
    Children children(executable);
    const auto port = unused_port();
    if (scenario == "late_root") {
        children.start(scenario, 1, world, port);
        std::this_thread::sleep_for(100ms);
        children.start(scenario, 0, world, port);
    } else {
        children.start(scenario, 0, world, port);
        if (scenario != "missing") {
            for (std::uint32_t rank = 1; rank < world; ++rank) {
                if (scenario == "incomplete" && rank == 2) {
                    continue;
                }
                children.start(scenario, rank, world, port);
            }
        }
    }
    children.finish();
    std::cout << "PASS " << scenario << " workers=" << world << std::endl;
}

void socket_contracts() {
    using namespace meshtrain;
    std::array<int, 2> descriptors{};
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0,
                         descriptors.data()) == 0,
            "socketpair failed");
    net::Socket sender(descriptors[0]);
    {
        const net::Socket receiver(descriptors[1]);
    }
    const std::array<std::byte, 1> byte{};
    expect_error([&] { net::send_exact(sender, byte, after(), {}); }, ErrorCode::peer_closed);
    // A broken send must throw rather than terminating this process with SIGPIPE.
    expect_error([&] { net::send_exact(sender, byte, after(-1ms), {}); }, ErrorCode::timeout);
    expect_error([&] { static_cast<void>(net::deadline_after(0ms, {})); },
                 ErrorCode::invalid_configuration);
    auto config = tcp_group(0, 1, unused_port(), "localhost", "single");
    const auto transport = connect_tcp(config);
    expect_error([&] { transport->send_bytes(0, byte, after()); }, ErrorCode::invalid_state);
    transport->close();
    transport->close();
    // Failed attempts must release the listener too, including immediate rebinding.
    config.world_size = 2;
    config.endpoints.push_back({"unused-worker", 45000});
    config.startup_timeout = 5ms;
    for (int attempt = 0; attempt < 20; ++attempt) {
        expect_error([&] { static_cast<void>(connect_tcp(config)); }, ErrorCode::timeout);
    }
    std::cout << "PASS socket error/deadline/single-rank contracts\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 7 && std::string_view(argv[1]) == "--worker") {
            return test::tcp_worker(argv[2], static_cast<std::uint32_t>(std::stoul(argv[3])),
                                    static_cast<std::uint32_t>(std::stoul(argv[4])),
                                    static_cast<std::uint16_t>(std::stoul(argv[5])), argv[6]);
        }
        test::require(argc == 1, "expected --worker CASE RANK WORLD PORT HOST");
        const auto executable = std::filesystem::canonical("/proc/self/exe").string();
        socket_contracts();
        for (const auto world : {1U, 2U, 3U, 4U, 5U}) {
            run(executable, "echo", world);
        }
        for (const auto scenario :
             {"late_root",   "fragment",   "bad_magic",    "bad_version",   "oversize",
              "bad_repr",    "truncated",  "wrong_run",    "wrong_context", "wrong_world",
              "wrong_limit", "bad_ready",  "missing",      "stall",         "drip",
              "peer_close",  "io_timeout", "send_timeout", "bad_rank",      "cancel",
              "eintr",       "repeat"}) {
            run(executable, scenario, 2);
        }
        run(executable, "duplicate", 3);
        run(executable, "incomplete", 3);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
