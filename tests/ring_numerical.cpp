#include "ring_fixture.hpp"

#include "protocol.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <set>

namespace ringtest {
namespace {
float value(std::uint32_t rank, std::size_t index, bool random, unsigned iteration) {
    if (!random) {
        return static_cast<float>(rank * 4U + index % 7U);
    }
    const auto hashed =
        static_cast<std::uint32_t>(index) * 2654435761U + rank * 1013904223U + iteration * 12345U;
    const auto sample = static_cast<std::int32_t>(hashed % 200001U) - 100000;
    return static_cast<float>(sample) / 100003.0F;
}

class Observed final : public ByteTransport {
  public:
    Observed(const RunConfig& config, std::unique_ptr<ByteTransport> inner)
        : config_(config), inner_(std::move(inner)) {}
    void send_bytes(std::uint32_t, std::span<const std::byte>, Deadline) override {
        throw std::runtime_error("ring attempted a standalone send");
    }
    void receive_bytes(std::uint32_t, std::span<std::byte>, Deadline) override {
        throw std::runtime_error("ring attempted a standalone receive");
    }
    void exchange_bytes(std::uint32_t send, std::span<const std::byte> outgoing,
                        std::uint32_t receive, std::span<std::byte> incoming,
                        Deadline deadline) override {
        require(send == (config_.rank + 1U) % config_.world_size &&
                    receive == (config_.rank + config_.world_size - 1U) % config_.world_size,
                "ring payload did not use direct neighbors");
        if (part_ < 2U) {
            std::copy(outgoing.begin(), outgoing.end(), encoded_.begin() + (part_ == 0U ? 0 : 8));
        }
        if (part_ == 1U) {
            const auto frame = wire::decode_header(encoded_, config_.max_payload_bytes);
            require(
                frame.sequence == sequence && frame.step == rounds % (config_.world_size - 1U) &&
                    frame.phase == (rounds < config_.world_size - 1U ? wire::Phase::reduce_scatter
                                                                     : wire::Phase::all_gather),
                "ring phase or step order differs");
            ++rounds;
        } else if (part_ == 2U) {
            payload += outgoing.size();
        }
        part_ = (part_ + 1U) % 3U;
        inner_->exchange_bytes(send, outgoing, receive, incoming, deadline);
    }
    void close() noexcept override { inner_->close(); }
    void completed(std::size_t count) {
        const auto chunk_size = [count, this](std::uint32_t chunk) {
            return count / config_.world_size + (chunk < count % config_.world_size ? 1U : 0U);
        };
        const auto excluded = chunk_size((config_.rank + 1U) % config_.world_size) +
                              chunk_size((config_.rank + 2U) % config_.world_size);
        require(part_ == 0U && rounds == 2U * (config_.world_size - 1U) &&
                    payload == (2U * count - excluded) * sizeof(float),
                "ring round or payload byte count differs");
        rounds = 0;
        payload = 0;
        ++sequence;
    }
    unsigned rounds{};
    std::size_t payload{};
    std::uint64_t sequence{};

  private:
    const RunConfig config_;
    std::unique_ptr<ByteTransport> inner_;
    unsigned part_{};
    std::array<std::byte, wire::header_bytes> encoded_{};
};

void conformance(const RunConfig& config, double& max_absolute, double& max_cross) {
    const auto baseline = descriptors();
    TcpOptions options;
    options.send_buffer_bytes = config.all_reduce_algorithm == AllReduceAlgorithm::ring ? 4096 : 0;
    auto transport = connect_tcp(config, options);
    const bool ring = config.all_reduce_algorithm == AllReduceAlgorithm::ring;
    Observed* observation = nullptr;
    if (ring) {
        auto observed = std::make_unique<Observed>(config, std::move(transport));
        observation = observed.get();
        transport = std::move(observed);
    }
    const auto degree = config.world_size == 1U ? 0U : (config.world_size == 2U ? 1U : 2U);
    const auto controls =
        config.world_size == 1U ? 0U : (config.rank == 0U ? config.world_size - 1U : 1U);
    require(descriptors() == baseline + controls + (ring ? degree : 0U),
            "startup descriptor topology differs");
    Communicator communicator(config, std::move(transport));
    std::set<std::size_t> counts{0,
                                 1,
                                 config.world_size - 1U,
                                 config.world_size,
                                 config.world_size + 1U,
                                 2U * config.world_size + 1U,
                                 1024,
                                 1025,
                                 131073,
                                 1048579};
    for (const auto count : counts) {
        for (const auto random : {false, true}) {
            const auto iteration = static_cast<unsigned>(communicator.next_sequence());
            std::vector<float> actual(count);
            for (std::size_t index = 0; index < count; ++index) {
                actual[index] = value(config.rank, index, random, iteration);
            }
            const auto* storage = actual.data();
            communicator.all_reduce_sum(actual);
            if (observation != nullptr) {
                observation->completed(count);
            }
            require(actual.data() == storage && communicator.next_sequence() == iteration + 1U,
                    "ring replaced storage or advanced sequence incorrectly");
            for (std::size_t index = 0; index < count; ++index) {
                double expected = 0;
                float centralized = 0;
                std::uint32_t chunk = 0;
                std::size_t end = 0;
                while (end <= index) {
                    end +=
                        count / config.world_size + (chunk < count % config.world_size ? 1U : 0U);
                    ++chunk;
                }
                --chunk;
                float ordered = value(chunk, index, random, iteration);
                for (std::uint32_t peer = 0; peer < config.world_size; ++peer) {
                    const auto contribution = value(peer, index, random, iteration);
                    expected += contribution;
                    centralized += contribution;
                    if (peer > 0) {
                        ordered =
                            value((chunk + peer) % config.world_size, index, random, iteration) +
                            ordered;
                    }
                }
                const auto absolute = std::abs(static_cast<double>(actual[index]) - expected);
                const auto cross = std::abs(static_cast<double>(actual[index]) - centralized);
                require(std::isfinite(actual[index]) &&
                            (random ? absolute <= 1e-6 + 1e-5 * std::abs(expected) : absolute == 0),
                        "ring SUM exceeded fixed reference tolerance");
                require(cross <= 1e-6 + 1e-5 * std::abs(static_cast<double>(centralized)),
                        "ring/centralized arithmetic mismatch");
                require(std::bit_cast<std::uint32_t>(actual[index]) ==
                            std::bit_cast<std::uint32_t>(ring ? ordered : centralized),
                        "ring rank differs from independently ordered FP32 result");
                max_absolute = std::max(max_absolute, absolute);
                max_cross = std::max(max_cross, cross);
            }
        }
    }
    std::cout << "algorithm=" << (ring ? "ring" : "centralized") << " rank=" << config.rank
              << " world=" << config.world_size << " counts=" << counts.size()
              << " calls=" << communicator.next_sequence() << " max_abs=" << std::setprecision(12)
              << max_absolute << " max_cross=" << max_cross << '\n';
}
} // namespace

int numerical(RunConfig settings) {
    settings.startup_timeout = 5s;
    settings.collective_timeout = 5s;
    const auto baseline = descriptors();
    double max_absolute = 0;
    double max_cross = 0;
    conformance(settings, max_absolute, max_cross);
    require(descriptors() == baseline, "ring numerical destruction leaked descriptors");
    settings.all_reduce_algorithm = AllReduceAlgorithm::centralized;
    settings.run_id += "-centralized";
    max_absolute = 0;
    max_cross = 0;
    conformance(settings, max_absolute, max_cross);
    require(descriptors() == baseline, "centralized numerical destruction leaked descriptors");
    return 0;
}
} // namespace ringtest
