#pragma once

#include "process_group.hpp"

#include <meshtrain/communicator.hpp>

#include <filesystem>

namespace ringtest {
using namespace meshtrain;
using namespace test;

inline RunConfig config(std::uint32_t rank, std::uint32_t world, std::uint16_t port,
                        std::string_view host, std::string_view scenario) {
    RunConfig result;
    result.rank = rank;
    result.world_size = world;
    result.run_id = std::string(scenario);
    result.context_fingerprint = "ring-conformance-v2";
    result.startup_timeout = 1000ms;
    result.collective_timeout = 1000ms;
    result.all_reduce_algorithm = AllReduceAlgorithm::ring;
    for (std::uint32_t peer = 0; peer < world; ++peer) {
        const bool container = scenario == "container";
        result.endpoints.push_back(
            {container ? std::string(host) + std::to_string(peer) : std::string(host),
             static_cast<std::uint16_t>(container ? port : port + peer)});
    }
    return result;
}

inline std::uint16_t ports(std::uint32_t world) {
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
        const auto port = unused_port();
        if (static_cast<unsigned>(port) + world >= 65536U) {
            continue;
        }
        std::vector<net::Socket> reserved;
        bool available = true;
        for (std::uint32_t rank = 0; rank < world; ++rank) {
            net::Socket socket(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
            require(socket.descriptor() >= 0, "port reservation socket failed");
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            address.sin_port = htons(static_cast<std::uint16_t>(port + rank));
            if (::bind(socket.descriptor(), reinterpret_cast<sockaddr*>(&address),
                       sizeof(address)) != 0) {
                available = false;
                break;
            }
            reserved.push_back(std::move(socket));
        }
        if (available) {
            return port;
        }
    }
    throw std::runtime_error("cannot reserve rank endpoint ports");
}

inline std::size_t descriptors() {
    return static_cast<std::size_t>(
        std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                      std::filesystem::directory_iterator{}));
}

int numerical(RunConfig config);
int duplex(RunConfig config, std::string_view scenario);
int failure(RunConfig config, std::string_view scenario);
int startup(RunConfig config, std::string_view scenario);
} // namespace ringtest
