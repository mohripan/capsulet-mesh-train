#include "collective_fixture.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iomanip>
#include <random>

namespace test {
namespace {
using namespace meshtrain;

std::vector<float> values(std::uint32_t rank, std::size_t count, bool random, unsigned iteration) {
    std::vector<float> result(count);
    std::mt19937 engine(1234U + rank * 101U + iteration);
    for (std::size_t index = 0; index < count; ++index) {
        if (random) {
            const auto sample = static_cast<std::int32_t>(engine() % 200001U) - 100000;
            result[index] = static_cast<float>(sample) / 100003.0F;
        } else {
            result[index] = static_cast<float>(rank * 4U + index % 7U);
        }
    }
    return result;
}

void numerical(Communicator& communicator, const RunConfig& config) {
    double max_absolute = 0;
    double max_relative = 0;
    std::uint64_t sequence = 0;
    for (const auto count : {0U, 1U, 3U, 1024U, 1025U, 16384U, 131073U}) {
        for (const auto random : {false, true}) {
            const auto input = values(config.rank, count, random, static_cast<unsigned>(sequence));
            auto actual = input;
            std::vector<double> expected(count);
            for (std::uint32_t peer = 0; peer < config.world_size; ++peer) {
                const auto contribution =
                    values(peer, count, random, static_cast<unsigned>(sequence));
                for (std::size_t index = 0; index < count; ++index) {
                    expected[index] += contribution[index];
                }
            }
            const auto* storage = actual.data();
            communicator.all_reduce_sum(actual);
            require(actual.data() == storage, "in-place collective replaced caller storage");
            for (std::size_t index = 0; index < count; ++index) {
                const auto absolute =
                    std::abs(static_cast<double>(actual[index]) - expected[index]);
                const auto relative = absolute / std::max(std::abs(expected[index]), 1e-12);
                if (random) {
                    require(std::isfinite(actual[index]) &&
                                absolute <= 1e-6 + 1e-5 * std::abs(expected[index]),
                            "random SUM exceeded atol=1e-6, rtol=1e-5");
                    max_absolute = std::max(max_absolute, absolute);
                    max_relative = std::max(max_relative, relative);
                } else {
                    require(absolute == 0, "exact SUM failed (or result was implicitly averaged)");
                }
            }
            ++sequence;
            require(communicator.next_sequence() == sequence &&
                        communicator.state() == CommunicatorState::ready,
                    "successful call did not advance exactly once");
        }
    }
    std::cout << std::setprecision(12) << "numerical rank=" << config.rank
              << " world=" << config.world_size << " collectives=" << sequence
              << " max_abs=" << max_absolute << " max_rel=" << max_relative << '\n';
}

void rank_order(Communicator& communicator, const RunConfig& config) {
    const std::array<float, 3> contributions{1e20F, -1e20F, 3.0F};
    std::array<float, 1> value{contributions[config.rank]};
    communicator.all_reduce_sum(value);
    require(value[0] == 3.0F, "root did not add FP32 contributions in ascending rank order");
    const std::array<float, 3> rounding{16777216.0F, 1.0F, -16777216.0F};
    value[0] = rounding[config.rank];
    communicator.all_reduce_sum(value);
    require(value[0] == 0.0F, "accumulation did not round in FP32 after each contribution");
    // Follow with an empty collective so zero-byte calls also negotiate sequence.
    communicator.all_reduce_sum({});
    require(communicator.next_sequence() == 3, "empty participation did not advance sequence");
}

std::size_t descriptors() {
    return static_cast<std::size_t>(
        std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                      std::filesystem::directory_iterator{}));
}

void reconnect(RunConfig config) {
    const auto before = descriptors();
    for (unsigned iteration = 0; iteration < 20; ++iteration) {
        config.run_id = "collective-attempt-" + std::to_string(iteration);
        Communicator communicator(config, connect_tcp(config));
        std::array<float, 1> value{static_cast<float>(config.rank + 1U)};
        communicator.all_reduce_sum(value);
        require(value[0] == 3.0F, "recreated communicator produced wrong sum");
    }
    require(descriptors() == before, "communicator lifecycle leaked descriptors");
}
} // namespace

int collective_worker(std::string_view scenario, std::uint32_t rank, std::uint32_t world,
                      std::uint16_t port, std::string_view host) {
    try {
        auto config = tcp_group(rank, world, port, host, scenario);
        if (scenario == "reconnect") {
            reconnect(config);
        } else if (scenario == "numerical" || scenario == "container" || scenario == "rank_order") {
            if (scenario == "container") {
                config.startup_timeout = 20s;
                config.collective_timeout = 5s;
            }
            TcpOptions options;
            options.send_buffer_bytes = 4096;
            Communicator communicator(config, connect_tcp(config, options));
            if (scenario == "rank_order") {
                rank_order(communicator, config);
            } else {
                numerical(communicator, config);
            }
        } else {
            collective_failure(scenario, config);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "worker " << rank << " case=" << scenario << ": " << error.what() << '\n';
        return 1;
    }
}
} // namespace test
