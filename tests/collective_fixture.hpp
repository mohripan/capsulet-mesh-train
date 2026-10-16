#pragma once

#include "tcp_fixture.hpp"

#include <meshtrain/communicator.hpp>

#include <cmath>
#include <iostream>

namespace test {

int collective_worker(std::string_view scenario, std::uint32_t rank, std::uint32_t world,
                      std::uint16_t port, std::string_view host);
void collective_failure(std::string_view scenario, meshtrain::RunConfig config);

template <typename Function>
void failed_collective(meshtrain::Communicator& communicator, const meshtrain::RunConfig& config,
                       Function&& operation, meshtrain::ErrorCode expected,
                       std::uint64_t successful = 0) {
    const auto before = std::chrono::steady_clock::now();
    bool caught = false;
    try {
        std::forward<Function>(operation)();
    } catch (const meshtrain::EngineError& error) {
        caught = true;
        require(error.code() == expected, error.what());
        require(error.context().rank == config.rank && error.context().run_id == config.run_id &&
                    error.context().sequence == successful &&
                    error.context().operation.starts_with("all_reduce_sum"),
                "collective failure lacks structured context");
        if (expected != meshtrain::ErrorCode::invalid_buffer) {
            require(error.context().peer.has_value() && !error.context().peer_endpoint.empty(),
                    "network failure lacks peer/endpoint context");
        }
        std::cout << "failure case=" << config.run_id << " rank=" << config.rank
                  << " category=" << meshtrain::to_string(error.code()) << " elapsed_ms="
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - before)
                         .count()
                  << " sequence=" << *error.context().sequence << '\n';
    }
    require(caught && communicator.state() == meshtrain::CommunicatorState::failed &&
                communicator.next_sequence() == successful,
            "failure did not preserve sequence and terminal state");
    require(std::chrono::steady_clock::now() - before < 2500ms, "failure exceeded operation bound");
    expect_error([&] { communicator.all_reduce_sum({}); }, meshtrain::ErrorCode::invalid_state);
    communicator.close();
    communicator.close();
}

} // namespace test
