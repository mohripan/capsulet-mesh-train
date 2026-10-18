#include "ring_fixture.hpp"

#include <pthread.h>

#include <array>
#include <exception>
#include <iostream>
#include <latch>
#include <thread>

namespace ringtest {
namespace {
std::vector<std::byte> bytes(std::uint32_t rank, std::size_t base) {
    std::vector<std::byte> result(base == 0U ? 0U : base + rank * 31U);
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<std::byte>((index + rank) % 251U);
    }
    return result;
}

void handler(int) {}
} // namespace

int duplex(RunConfig settings, std::string_view scenario) {
    const auto baseline = descriptors();
    const auto successor = (settings.rank + 1U) % settings.world_size;
    const auto predecessor = (settings.rank + settings.world_size - 1U) % settings.world_size;
    {
        TcpOptions options;
        options.send_buffer_bytes = 4096;
        options.receive_buffer_bytes = scenario == "duplex_small" ? 4096 : 0;
        auto transport = connect_tcp(settings, options);
        if (scenario == "duplex_cancel") {
            if (settings.rank == 0U) {
                std::array<std::byte, 1> input{};
                std::exception_ptr error;
                std::latch started(1);
                std::jthread worker([&] {
                    started.count_down();
                    try {
                        transport->exchange_bytes(successor, {}, predecessor, input, after(2s));
                    } catch (...) {
                        error = std::current_exception();
                    }
                });
                started.wait();
                std::this_thread::sleep_for(50ms);
                expect_error(
                    [&] { transport->exchange_bytes(successor, {}, predecessor, {}, after()); },
                    ErrorCode::invalid_state);
                const auto before = std::chrono::steady_clock::now();
                transport->close();
                worker.join();
                require(error != nullptr && std::chrono::steady_clock::now() - before < 500ms,
                        "duplex close did not wake active I/O");
                expect_error([&] { std::rethrow_exception(error); }, ErrorCode::peer_closed);
                expect_error(
                    [&] { transport->exchange_bytes(successor, {}, predecessor, {}, after()); },
                    ErrorCode::invalid_state);
            } else {
                std::this_thread::sleep_for(250ms);
            }
        } else if (scenario == "duplex_timeout" || scenario == "duplex_drip") {
            if (settings.rank == 0U) {
                auto outgoing = bytes(0, scenario == "duplex_timeout" ? 4U * 1024U * 1024U : 0U);
                std::array<std::byte, 8> incoming{};
                const auto before = std::chrono::steady_clock::now();
                expect_error(
                    [&] {
                        transport->exchange_bytes(successor, outgoing, predecessor, incoming,
                                                  after(300ms));
                    },
                    ErrorCode::timeout);
                require(std::chrono::steady_clock::now() - before < 800ms,
                        "duplex partial progress renewed deadline");
            } else if (scenario == "duplex_drip") {
                const std::array<std::byte, 1> marker{};
                try {
                    for (unsigned index = 0; index < 8; ++index) {
                        transport->send_bytes(0, marker, after());
                        std::this_thread::sleep_for(90ms);
                    }
                } catch (const EngineError& error) {
                    require(error.code() == ErrorCode::peer_closed, error.what());
                }
            } else {
                std::this_thread::sleep_for(500ms);
            }
        } else if (scenario == "duplex_eof") {
            if (settings.rank == 0U) {
                std::array<std::byte, 1> input{};
                expect_error(
                    [&] { transport->exchange_bytes(successor, {}, predecessor, input, after()); },
                    ErrorCode::peer_closed);
            } else {
                transport->close();
            }
        } else {
            struct sigaction action{};
            action.sa_handler = handler;
            require(::sigemptyset(&action.sa_mask) == 0 &&
                        ::sigaction(SIGUSR1, &action, nullptr) == 0,
                    "cannot install test signal handler");
            const auto caller = ::pthread_self();
            std::jthread interrupter([caller](std::stop_token stop) {
                while (!stop.stop_requested()) {
                    static_cast<void>(::pthread_kill(caller, SIGUSR1));
                    std::this_thread::sleep_for(2ms);
                }
            });
            const std::array<std::size_t, 4> sizes{0, 1, 65537,
                                                   scenario == "duplex_small" ? 65539U : 1048583U};
            for (const auto base : sizes) {
                const auto outgoing = bytes(settings.rank, base);
                const auto expected = bytes(predecessor, base);
                std::vector<std::byte> incoming(expected.size());
                if (settings.rank == settings.world_size - 1U) {
                    std::this_thread::sleep_for(30ms);
                }
                transport->exchange_bytes(successor, outgoing, predecessor, incoming, after(3s));
                require(incoming == expected, "duplex changed asymmetric bytes");
            }
            const auto outgoing = bytes(settings.rank, settings.rank == 0U ? 0U : 32771U);
            const auto expected = bytes(predecessor, predecessor == 0U ? 0U : 32771U);
            std::vector<std::byte> incoming(expected.size());
            transport->exchange_bytes(successor, outgoing, predecessor, incoming, after());
            require(incoming == expected, "one empty duplex direction changed data");
            interrupter.request_stop();
        }
        transport->close();
    }
    require(descriptors() == baseline, "duplex leaked descriptors");
    std::cout << "duplex scenario=" << scenario << " rank=" << settings.rank
              << " fds_before=" << baseline << " fds_after=" << descriptors() << '\n';
    return 0;
}
} // namespace ringtest
