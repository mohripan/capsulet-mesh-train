#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace meshtrain {

struct Endpoint {
    std::string host;
    std::uint16_t port{};
};

enum class AllReduceAlgorithm : std::uint16_t { centralized = 1, ring = 2 };

struct RunConfig {
    std::uint32_t rank{};
    std::uint32_t world_size{};
    std::string run_id;
    std::string context_fingerprint;
    std::vector<Endpoint> endpoints;
    std::string listen_address{"0.0.0.0"};
    std::chrono::milliseconds startup_timeout{30'000};
    std::chrono::milliseconds collective_timeout{30'000};
    std::size_t max_payload_bytes{256U * 1024U * 1024U};
    AllReduceAlgorithm all_reduce_algorithm{AllReduceAlgorithm::centralized};
};

// Validates values without resolving names, opening sockets, or changing config.
// Throws EngineError with invalid_configuration on unsupported/invalid input.
void validate(const RunConfig& config);

} // namespace meshtrain
