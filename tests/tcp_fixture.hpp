#pragma once

#include "test_support.hpp"

#include <meshtrain/tcp.hpp>

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace test {
using namespace std::chrono_literals;

inline meshtrain::RunConfig tcp_group(std::uint32_t rank, std::uint32_t world, std::uint16_t port,
                                      std::string_view host, std::string_view run) {
    meshtrain::RunConfig config;
    config.rank = rank;
    config.world_size = world;
    config.run_id = std::string(run);
    config.context_fingerprint = "tcp-fixture-v1";
    config.startup_timeout = 3s;
    config.collective_timeout = 3s;
    config.endpoints.push_back({std::string(host), port});
    for (std::uint32_t peer = 1; peer < world; ++peer) {
        config.endpoints.push_back({"unused-worker-" + std::to_string(peer), 45000});
    }
    return config;
}

inline meshtrain::Deadline after(std::chrono::milliseconds timeout = 3s) {
    return std::chrono::steady_clock::now() + timeout;
}

inline std::vector<std::byte> pattern(std::uint32_t rank) {
    std::vector<std::byte> bytes(512U * 1024U + 7U);
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::byte>((index + rank) % 251U);
    }
    return bytes;
}

int tcp_worker(std::string_view scenario, std::uint32_t rank, std::uint32_t world,
               std::uint16_t port, std::string_view host);
} // namespace test
