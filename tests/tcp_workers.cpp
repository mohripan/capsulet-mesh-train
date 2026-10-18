#include "tcp_fixture.hpp"

#include "linux/socket.hpp"
#include "protocol.hpp"

#include <pthread.h>
#include <signal.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <filesystem>
#include <iostream>
#include <latch>
#include <thread>

namespace test {
namespace {
using namespace meshtrain;

constexpr std::array<std::byte, 1> marker{std::byte{42}};

void echo(ByteTransport& transport, const RunConfig& config) {
    if (config.rank == 0) {
        for (std::uint32_t peer = 1; peer < config.world_size; ++peer) {
            std::vector<std::byte> bytes(pattern(peer).size());
            transport.receive_bytes(peer, bytes, after());
            require(bytes == pattern(peer), "root received corrupt payload");
            transport.send_bytes(peer, bytes, after());
        }
    } else {
        const auto original = pattern(config.rank);
        transport.send_bytes(0, original, after());
        std::vector<std::byte> received(original.size());
        transport.receive_bytes(0, received, after());
        require(received == original, "echo changed payload");
    }
}

bool raw_scenario(std::string_view scenario) {
    return scenario == "fragment" || scenario == "bad_magic" || scenario == "bad_version" ||
           scenario == "oversize" || scenario == "bad_repr" || scenario == "bad_rank" ||
           scenario == "truncated" || scenario == "wrong_run" || scenario == "wrong_context" ||
           scenario == "wrong_world" || scenario == "wrong_limit" || scenario == "duplicate" ||
           scenario == "stall" || scenario == "drip";
}

void raw_peer(std::string_view scenario, RunConfig config) {
    auto socket = net::connect(config.endpoints[0], {}, after(), {});
    if (scenario == "stall") {
        std::this_thread::sleep_for(800ms);
        return;
    }
    if (scenario == "wrong_run") {
        config.run_id = "another-run";
    } else if (scenario == "wrong_context") {
        config.context_fingerprint = "another-layout";
    }
    auto payload = wire::encode_hello(config);
    if (scenario == "bad_repr") {
        payload[15] = std::byte{};
    } else if (scenario == "wrong_world") {
        payload[3] = std::byte{3};
    } else if (scenario == "wrong_limit") {
        payload[11] ^= std::byte{1};
    }
    wire::Header hello;
    hello.source = scenario == "duplicate" ? 1U : config.rank;
    if (scenario == "bad_rank") {
        hello.source = config.world_size;
    }
    hello.payload_bytes = payload.size();
    auto header = wire::encode_header(hello, config.max_payload_bytes);
    if (scenario == "bad_magic") {
        header[0] = std::byte{};
    } else if (scenario == "bad_version") {
        header[5] = std::byte{1};
    } else if (scenario == "oversize") {
        header[32] = std::byte{1}; // Rejected before allocating the claimed payload.
    }
    if (scenario == "truncated") {
        net::send_exact(socket, std::span(header).first(7), after(), {});
        return;
    }
    if (scenario == "drip") {
        try {
            for (const auto byte : header) {
                net::send_exact(socket, std::span(&byte, 1), after(), {});
                std::this_thread::sleep_for(60ms);
            }
        } catch (const EngineError& error) {
            require(error.code() == ErrorCode::peer_closed, "unexpected drip send failure");
            return;
        }
        throw std::runtime_error("root accepted an unbounded slow header");
    }
    if (scenario == "fragment") {
        for (const auto& bytes :
             {std::span<const std::byte>(header), std::span<const std::byte>(payload)}) {
            for (const auto byte : bytes) {
                net::send_exact(socket, std::span(&byte, 1), after(), {});
                std::this_thread::sleep_for(1ms);
            }
        }
    } else {
        // Coalesce so a rejecting root cannot close between the two writes.
        std::vector<std::byte> frame(header.begin(), header.end());
        frame.insert(frame.end(), payload.begin(), payload.end());
        net::send_exact(socket, frame, after(), {});
    }
    std::array<std::byte, wire::header_bytes> response{};
    if (scenario != "fragment") {
        expect_error([&] { net::receive_exact(socket, response, after(), {}); },
                     ErrorCode::peer_closed);
        return;
    }
    net::receive_exact(socket, response, after(), {});
    const auto ready = wire::decode_header(response, config.max_payload_bytes);
    require(ready.operation == wire::Operation::ready && ready.destination == config.rank,
            "root sent wrong READY");
    const auto bytes = pattern(config.rank);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 997U) {
        net::send_exact(socket,
                        std::span(bytes).subspan(offset, std::min(997UL, bytes.size() - offset)),
                        after(), {});
    }
    std::vector<std::byte> received(bytes.size());
    net::receive_exact(socket, received, after(), {});
    require(bytes == received, "fragmented echo changed payload");
}

void bad_ready(const RunConfig& config) {
    if (config.rank != 0) {
        expect_error([&] { static_cast<void>(connect_tcp(config)); }, ErrorCode::protocol_mismatch);
        return;
    }
    auto listener = net::listen(config, {}, after(), {});
    const auto socket = net::accept(listener, {}, after(), {});
    std::array<std::byte, wire::header_bytes> bytes{};
    net::receive_exact(socket, bytes, after(), {});
    const auto header = wire::decode_header(bytes, config.max_payload_bytes);
    std::vector<std::byte> payload(static_cast<std::size_t>(header.payload_bytes));
    net::receive_exact(socket, payload, after(), {});
    wire::Header ready;
    ready.operation = wire::Operation::ready;
    ready.destination = 1;
    bytes = wire::encode_header(ready, config.max_payload_bytes);
    bytes[5] = std::byte{1};
    net::send_exact(socket, bytes, after(), {});
}

void cancel(ByteTransport& transport, const RunConfig& config) {
    std::array<std::byte, 1> received{};
    if (config.rank != 0) {
        expect_error([&] { transport.receive_bytes(0, received, after()); },
                     ErrorCode::peer_closed);
        return;
    }
    std::exception_ptr result;
    std::latch entered(1);
    std::jthread reader([&] {
        entered.count_down();
        try {
            transport.receive_bytes(1, received, after());
        } catch (...) {
            result = std::current_exception();
        }
    });
    entered.wait();
    std::this_thread::sleep_for(50ms);
    expect_error([&] { transport.send_bytes(1, marker, after()); }, ErrorCode::invalid_state);
    const auto before = std::chrono::steady_clock::now();
    transport.close();
    transport.close();
    reader.join();
    require(std::chrono::steady_clock::now() - before < 1s, "close did not wake blocked I/O");
    require(result != nullptr, "cancelled receive succeeded");
    expect_error([&] { std::rethrow_exception(result); }, ErrorCode::peer_closed);
    expect_error([&] { transport.receive_bytes(1, received, after()); }, ErrorCode::invalid_state);
}

void signal_handler(int) {}

void interrupted(ByteTransport& transport, const RunConfig& config) {
    if (config.rank != 0) {
        std::this_thread::sleep_for(200ms);
        transport.send_bytes(0, marker, after());
        return;
    }
    struct sigaction action{};
    action.sa_handler = signal_handler;
    require(::sigemptyset(&action.sa_mask) == 0 && ::sigaction(SIGUSR1, &action, nullptr) == 0,
            "cannot install test-only interrupt handler");
    const auto waiting_thread = ::pthread_self();
    std::atomic<bool> stop{};
    std::jthread interrupter([&] {
        while (!stop.load()) {
            static_cast<void>(::pthread_kill(waiting_thread, SIGUSR1));
            std::this_thread::sleep_for(2ms);
        }
    });
    std::array<std::byte, 1> received{};
    try {
        transport.receive_bytes(1, received, after(1s));
    } catch (...) {
        stop.store(true);
        throw;
    }
    stop.store(true);
    interrupter.join();
    require(received == marker, "interrupted receive changed data");
}

std::size_t descriptor_count() {
    return static_cast<std::size_t>(
        std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                      std::filesystem::directory_iterator{}));
}

void repeat(RunConfig config) {
    const auto before = descriptor_count();
    for (int iteration = 0; iteration < 20; ++iteration) {
        config.run_id = "repeat-" + std::to_string(iteration);
        const auto transport = connect_tcp(config);
        std::array<std::byte, 1> received{};
        if (config.rank == 0) {
            transport->receive_bytes(1, received, after());
            transport->send_bytes(1, received, after());
        } else {
            transport->send_bytes(0, marker, after());
            transport->receive_bytes(0, received, after());
        }
        require(received == marker, "repeated group changed data");
        transport->close();
    }
    require(descriptor_count() == before, "repeated group leaked descriptors");
}
} // namespace

int tcp_worker(std::string_view scenario, std::uint32_t rank, std::uint32_t world,
               std::uint16_t port, std::string_view host) {
    try {
        auto config = tcp_group(rank, world, port, host, scenario);
        if (scenario == "container") {
            config.startup_timeout = 15s;
        }
        if (scenario == "bad_ready") {
            bad_ready(config);
            return 0;
        }
        if (scenario == "incomplete" && rank != 0) {
            expect_error([&] { static_cast<void>(connect_tcp(config)); }, ErrorCode::peer_closed);
            return 0;
        }
        if (raw_scenario(scenario) && rank != 0) {
            raw_peer(scenario, config);
            return 0;
        }
        const bool startup_timeout = scenario == "missing" || scenario == "stall" ||
                                     scenario == "drip" || scenario == "incomplete";
        if (startup_timeout || (raw_scenario(scenario) && scenario != "fragment")) {
            config.startup_timeout = 500ms;
            const auto before = std::chrono::steady_clock::now();
            expect_error([&] { static_cast<void>(connect_tcp(config)); },
                         startup_timeout           ? ErrorCode::timeout
                         : scenario == "truncated" ? ErrorCode::peer_closed
                                                   : ErrorCode::protocol_mismatch);
            const auto elapsed = std::chrono::steady_clock::now() - before;
            require(elapsed < 1500ms, "startup exceeded total deadline tolerance");
            std::cout << "startup case=" << scenario << " elapsed_ms="
                      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
                      << '\n';
            return 0;
        }
        if (scenario == "repeat") {
            repeat(config);
            return 0;
        }
        TcpOptions options;
        options.send_buffer_bytes = 4096;
        const auto transport = connect_tcp(config, options);
        if (scenario == "cancel") {
            cancel(*transport, config);
        } else if (scenario == "eintr") {
            interrupted(*transport, config);
        } else if (scenario == "send_timeout") {
            if (rank == 0) {
                const std::vector<std::byte> blocked(16U * 1024U * 1024U);
                expect_error([&] { transport->send_bytes(1, blocked, after(250ms)); },
                             ErrorCode::timeout);
                expect_error([&] { transport->send_bytes(1, marker, after()); },
                             ErrorCode::invalid_state);
            } else {
                std::this_thread::sleep_for(800ms);
            }
        } else if (scenario == "peer_close" || scenario == "io_timeout") {
            if (rank != 0 && scenario == "peer_close") {
                transport->close();
            } else {
                std::array<std::byte, 1> received{};
                expect_error(
                    [&] {
                        transport->receive_bytes(rank == 0 ? 1U : 0U, received,
                                                 after(rank == 0 ? 250ms : 3s));
                    },
                    rank == 0 && scenario == "io_timeout" ? ErrorCode::timeout
                                                          : ErrorCode::peer_closed);
                expect_error([&] { transport->send_bytes(rank == 0 ? 1U : 0U, marker, after()); },
                             ErrorCode::invalid_state);
            }
        } else {
            echo(*transport, config);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "worker " << rank << " case=" << scenario << ": " << error.what() << '\n';
        return 1;
    }
}
} // namespace test
