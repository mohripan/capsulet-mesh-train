#pragma once

#include "tcp_fixture.hpp"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace fault {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
using test::require;

class File {
  public:
    explicit File(int descriptor = -1) : descriptor_(descriptor) {}
    ~File() { reset(); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    File(File&& other) noexcept : descriptor_(std::exchange(other.descriptor_, -1)) {}
    File& operator=(File&& other) noexcept {
        if (this != &other) {
            reset(std::exchange(other.descriptor_, -1));
        }
        return *this;
    }
    void reset(int descriptor = -1) noexcept {
        if (descriptor_ >= 0) {
            static_cast<void>(::close(descriptor_));
        }
        descriptor_ = descriptor;
    }
    [[nodiscard]] int get() const { return descriptor_; }

  private:
    int descriptor_;
};

struct Pipe {
    File read;
    File write;
    Pipe() {
        int descriptors[2]{};
        require(::pipe2(descriptors, O_CLOEXEC) == 0, "pipe2 failed");
        read.reset(descriptors[0]);
        write.reset(descriptors[1]);
    }
};

inline void send(int descriptor, char value) {
    ssize_t result;
    do {
        result = ::write(descriptor, &value, 1);
    } while (result < 0 && errno == EINTR);
    require(result == 1, "control write failed");
}

inline char receive(int descriptor, Clock::time_point deadline) {
    while (Clock::now() < deadline) {
        pollfd item{descriptor, POLLIN, 0};
        const auto left = std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now());
        const auto result =
            ::poll(&item, 1, static_cast<int>(std::max<std::int64_t>(1, left.count())));
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result > 0) {
            char value{};
            require(::read(descriptor, &value, 1) == 1, "control pipe closed");
            return value;
        }
    }
    throw std::runtime_error("control deadline expired");
}

inline std::size_t descriptors() {
    std::size_t count = 0;
    for ([[maybe_unused]] const auto& entry :
         std::filesystem::directory_iterator("/proc/self/fd")) {
        ++count;
    }
    return count;
}

inline std::string quote(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const auto character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (byte < 32) {
            output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                   << static_cast<unsigned>(byte) << std::dec;
        } else {
            if (character == '"' || character == '\\') {
                output << '\\';
            }
            output << character;
        }
    }
    return output.str() + '"';
}

inline void save(const std::filesystem::path& path, std::string_view value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << value << '\n';
    require(output.good(), "cannot write fault evidence");
}

class Watchdog {
  public:
    Watchdog() {
        if (::getpgrp() != ::getpid()) {
            require(::setpgid(0, 0) == 0, "cannot isolate fault controller group");
        }
        const auto controller = ::getpid();
        require(controller > 1, "run the fault fixture under CTest or docker --init");
        pid_ = ::fork();
        require(pid_ >= 0, "cannot fork watchdog");
        if (pid_ == 0) {
            static_cast<void>(::setpgid(0, 0));
            channel_.write.reset();
            bool active = false;
            auto deadline = Clock::now() + 120s;
            try {
                for (;;) {
                    const auto command = receive(channel_.read.get(), deadline);
                    if (command == 'Q') {
                        ::_exit(0);
                    }
                    if (command == 'B') {
                        active = true;
                        deadline = Clock::now() + 5s;
                    } else if (command == 'E') {
                        active = false;
                        deadline = Clock::now() + 120s;
                    } else {
                        throw std::runtime_error("invalid watchdog event");
                    }
                }
            } catch (...) {
                std::cerr << "FAIL independent watchdog expired active=" << active << std::endl;
                static_cast<void>(::kill(-controller, SIGKILL));
                ::_exit(124);
            }
        }
        channel_.read.reset();
    }
    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;
    ~Watchdog() {
        const char command = 'Q';
        static_cast<void>(::write(channel_.write.get(), &command, 1));
        int status{};
        while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
        }
    }
    void begin() { send(channel_.write.get(), 'B'); }
    void end() { send(channel_.write.get(), 'E'); }

  private:
    Pipe channel_;
    pid_t pid_{};
};

struct Settings {
    std::string name;
    bool startup;
    bool stop;
    unsigned target;
    bool omit_phase{};
    bool ring{};
    bool gather{};
};

inline Settings settings(std::string name) {
    const bool ring = name.starts_with("ring-");
    const bool gather = name.starts_with("ring-gather-");
    require(name.starts_with("startup-") || name.starts_with("collective-") ||
                name.starts_with("ring-startup-") || name.starts_with("ring-reduce-") || gather,
            "unknown phase");
    const bool root = name.find("-root-") != std::string::npos;
    const bool stop = name.ends_with("-stop");
    require(root || name.find("-worker-") != std::string::npos, "unknown target");
    require(stop || name.ends_with("-kill"), "unknown action");
    const bool startup = name.find("startup-") != std::string::npos;
    return {std::move(name), startup, stop, root ? 0U : 1U, false, ring, gather};
}

struct Result {
    std::uint32_t rank{};
    std::size_t before{};
    std::size_t after{};
    meshtrain::ErrorCode code{};
    bool valid{};
    bool has_sequence{};
    bool terminal{};
    bool has_peer{};
    std::array<char, 128> operation{};
    std::array<char, 128> endpoint{};
};

template <std::size_t Count> void copy(std::array<char, Count>& output, std::string_view value) {
    require(value.size() < Count, "fixture context exceeds report field");
    std::copy(value.begin(), value.end(), output.begin());
}

void gate(char marker, Clock::time_point deadline);
int worker(const Settings& scenario, std::uint32_t rank, std::uint16_t port, std::string run);
void run_case(const std::string& executable, Settings scenario, unsigned iteration,
              const std::filesystem::path& output, Watchdog& watchdog);
} // namespace fault
