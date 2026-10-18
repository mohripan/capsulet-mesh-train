#include "ring_fixture.hpp"

#include "protocol.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <iostream>
#include <thread>

namespace {
std::string_view active_case;
std::uint32_t active_rank{};
bool edge_payload{};
bool injected{};
bool edge_header{};
} // namespace

extern "C" ssize_t __real_send(int, const void*, std::size_t, int);

extern "C" ssize_t __wrap_send(int descriptor, const void* data, std::size_t count, int flags) {
    const auto* bytes = static_cast<const std::byte*>(data);
    if (active_rank != 1U || active_case.empty() || injected) {
        return __real_send(descriptor, data, count, flags);
    }
    if (count == meshtrain::wire::prefix_bytes && bytes[0] == std::byte{0x4d} &&
        bytes[7] == std::byte{7}) {
        if (active_case == "startup_edge_version") {
            std::array<std::byte, meshtrain::wire::prefix_bytes> changed{};
            std::copy_n(bytes, count, changed.begin());
            changed[5] = std::byte{1};
            injected = true;
            return __real_send(descriptor, changed.data(), count, flags);
        }
        edge_header = true;
    } else if (edge_header &&
               count == meshtrain::wire::header_bytes - meshtrain::wire::prefix_bytes) {
        edge_header = false;
        if (active_case == "startup_wrong_edge") {
            std::array<std::byte, meshtrain::wire::header_bytes> changed{};
            std::copy_n(bytes, count, changed.begin());
            changed[3] = std::byte{255};
            injected = true;
            return __real_send(descriptor, changed.data(), count, flags);
        }
        if (active_case.starts_with("startup_edge_")) {
            edge_payload = true;
        }
        return __real_send(descriptor, data, count, flags);
    }
    if (edge_payload && count >= 22U && count <= 404U) {
        std::array<std::byte, 404> changed{};
        std::copy_n(bytes, count, changed.begin());
        if (active_case == "startup_edge_run") {
            changed[20] ^= std::byte{1};
        } else if (active_case == "startup_edge_limit") {
            changed[11] ^= std::byte{1};
        } else if (active_case == "startup_edge_repr") {
            changed[15] = std::byte{};
        }
        injected = true;
        return __real_send(descriptor, changed.data(), count, flags);
    }
    if (count == meshtrain::wire::header_bytes && bytes[0] == std::byte{0x4d} &&
        bytes[1] == std::byte{0x45}) {
        const auto operation = std::to_integer<unsigned>(bytes[7]);
        if (operation == 8U && active_case == "startup_missing_ready") {
            injected = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(1200));
            errno = EPIPE;
            return -1;
        }
        if (operation == 7U && active_case.starts_with("startup_edge_")) {
            edge_payload = true;
        }
        if ((operation == 7U && active_case == "startup_wrong_edge") ||
            (operation == 1U && active_case == "startup_duplicate_rank")) {
            std::array<std::byte, meshtrain::wire::header_bytes> changed{};
            std::copy_n(bytes, count, changed.begin());
            changed[11] = active_case == "startup_wrong_edge" ? std::byte{255} : std::byte{2};
            injected = true;
            return __real_send(descriptor, changed.data(), count, flags);
        }
    }
    return __real_send(descriptor, data, count, flags);
}

namespace ringtest {
int startup(RunConfig settings, std::string_view scenario) {
    active_case = scenario;
    active_rank = settings.rank;
    const auto baseline = descriptors();
    const auto before = std::chrono::steady_clock::now();
    if (scenario == "startup_v1" && settings.rank == 1U) {
        {
            auto socket = net::connect(settings.endpoints[0], {}, after(), {});
            wire::Header header;
            header.source = 1;
            header.algorithm = AllReduceAlgorithm::ring;
            header.payload_bytes = 22;
            auto encoded = wire::encode_header(header, settings.max_payload_bytes);
            encoded[5] = std::byte{1};
            net::send_exact(socket, std::span(encoded).first(wire::prefix_bytes), after(), {});
            std::array<std::byte, 1> response{};
            expect_error([&] { net::receive_exact(socket, response, after(300ms), {}); },
                         ErrorCode::peer_closed);
        }
        require(descriptors() == baseline, "raw v1 peer leaked descriptors");
        return 0;
    }
    if (scenario == "startup_delay" && settings.rank == 2U) {
        std::this_thread::sleep_for(100ms);
    }
    if (settings.rank == 1U) {
        if (scenario == "startup_algorithm") {
            settings.all_reduce_algorithm = AllReduceAlgorithm::centralized;
        } else if (scenario == "startup_run") {
            settings.run_id += "-wrong";
        } else if (scenario == "startup_context") {
            settings.context_fingerprint += "-wrong";
        } else if (scenario == "startup_world") {
            settings.world_size = 2;
            settings.endpoints.resize(2);
        } else if (scenario == "startup_limit") {
            settings.max_payload_bytes /= 2;
        } else if (scenario == "startup_missing_edge") {
            settings.endpoints[2].port = ports(1);
        }
    }
    bool failed = false;
    try {
        auto transport = connect_tcp(settings);
        require(scenario == "startup_delay", "invalid topology reached final READY");
        Communicator communicator(settings, std::move(transport));
        communicator.all_reduce_sum({});
        require(communicator.next_sequence() == 1,
                "delayed ring did not complete empty participation");
    } catch (const EngineError& error) {
        failed = true;
        require(scenario != "startup_delay" &&
                    (error.code() == ErrorCode::protocol_mismatch ||
                     error.code() == ErrorCode::peer_closed || error.code() == ErrorCode::timeout),
                error.what());
        require(error.context().rank == settings.rank && !error.context().operation.empty(),
                "startup failure lost context");
        const bool edge_mismatch =
            scenario == "startup_wrong_edge" || scenario.starts_with("startup_edge_");
        const bool root_mismatch = scenario == "startup_v1" || scenario == "startup_algorithm" ||
                                   scenario == "startup_run" || scenario == "startup_context" ||
                                   scenario == "startup_world" || scenario == "startup_limit" ||
                                   scenario == "startup_duplicate_rank";
        if ((settings.rank == 2U && edge_mismatch) || (settings.rank == 0U && root_mismatch)) {
            require(error.code() == ErrorCode::protocol_mismatch,
                    "identity mismatch was not rejected as a protocol error");
        }
        std::cout << "startup case=" << scenario << " rank=" << settings.rank
                  << " category=" << to_string(error.code()) << '\n';
    }
    require(failed == (scenario != "startup_delay"), "startup fault did not fail");
    const bool syscall_fault =
        scenario == "startup_wrong_edge" || scenario == "startup_duplicate_rank" ||
        scenario == "startup_missing_ready" || scenario.starts_with("startup_edge_");
    require(settings.rank != 1U || !syscall_fault || injected,
            "startup injection was not exercised");
    require(descriptors() == baseline && std::chrono::steady_clock::now() - before < 1800ms,
            "startup leak or renewed deadline");
    return 0;
}
} // namespace ringtest
