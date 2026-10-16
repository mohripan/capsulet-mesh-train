#include "linux/resolver.hpp"
#include "test_support.hpp"

#include <netdb.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

namespace {
using namespace std::chrono_literals;
std::mutex gate;
std::condition_variable changed;
bool released{};
std::atomic<bool> entered{};
std::atomic<bool> retained{};
std::atomic<unsigned> freed{};

void release_lookup() {
    const std::lock_guard lock(gate);
    released = true;
    changed.notify_all();
}
} // namespace

extern "C" int __real_getaddrinfo(const char*, const char*, const addrinfo*, addrinfo**);
extern "C" void __real_freeaddrinfo(addrinfo*);

// Linker wrappers are confined to this executable. The production adapter has
// no test hooks; normal subprocess/container tests use the actual OS resolver.
extern "C" int __wrap_getaddrinfo(const char* host, const char* service, const addrinfo* hints,
                                  addrinfo** output) {
    entered.store(true);
    std::unique_lock lock(gate);
    changed.wait(lock, [] { return released; });
    retained.store(std::strcmp(host, "held.invalid") == 0 && std::strcmp(service, "45000") == 0);
    return __real_getaddrinfo("127.0.0.1", service, hints, output);
}

extern "C" void __wrap_freeaddrinfo(addrinfo* address) {
    __real_freeaddrinfo(address);
    freed.fetch_add(1);
}

int main() {
    try {
        using namespace meshtrain;
        const auto before = std::chrono::steady_clock::now();
        test::expect_error(
            [&] {
                static_cast<void>(net::resolve_ipv4({"held.invalid", 45000}, before + 50ms, {}));
            },
            ErrorCode::timeout);
        const auto elapsed = std::chrono::steady_clock::now() - before;
        test::require(entered.load() && elapsed < 1s && freed.load() == 0,
                      "slow resolution was not bounded while preserving request state");
        release_lookup();
        const auto cleanup_deadline = std::chrono::steady_clock::now() + 1s;
        while (freed.load() == 0 && std::chrono::steady_clock::now() < cleanup_deadline) {
            std::this_thread::sleep_for(1ms);
        }
        test::require(retained.load() && freed.load() == 1,
                      "timed-out resolver lost request storage or failed to release its result");
        test::expect_error(
            [&] { static_cast<void>(net::resolve_ipv4({"0.0.0.0", 45000}, cleanup_deadline, {})); },
            ErrorCode::invalid_configuration);
        std::cout << "PASS slow DNS deadline elapsed_ms="
                  << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
                  << "; late completion retained input and freed output exactly once\n";
        return 0;
    } catch (const std::exception& error) {
        release_lookup();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
