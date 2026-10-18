#include "ring_fixture.hpp"

#include "protocol.hpp"
#include "tcp_startup.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <latch>
#include <thread>

namespace ringtest {
namespace {
class FaultTransport final : public ByteTransport {
  public:
    FaultTransport(const RunConfig& config, std::unique_ptr<ByteTransport> inner,
                   std::string_view scenario, std::latch* entered = nullptr)
        : config_(config), inner_(std::move(inner)), scenario_(scenario), entered_(entered) {}
    void send_bytes(std::uint32_t peer, std::span<const std::byte> bytes,
                    Deadline deadline) override {
        inner_->send_bytes(peer, bytes, deadline);
    }
    void receive_bytes(std::uint32_t peer, std::span<std::byte> bytes, Deadline deadline) override {
        inner_->receive_bytes(peer, bytes, deadline);
    }
    void exchange_bytes(std::uint32_t send, std::span<const std::byte> outgoing,
                        std::uint32_t receive, std::span<std::byte> incoming,
                        Deadline deadline) override {
        ++calls;
        if (entered_ != nullptr) {
            entered_->count_down();
            entered_ = nullptr;
        }
        if (scenario_ == "failure_cancel" && config_.rank == 2U && calls == 1U) {
            std::this_thread::sleep_for(300ms);
        }
        const bool gather = scenario_.starts_with("gather_");
        const auto field = gather                          ? scenario_.substr(7)
                           : scenario_.starts_with("bad_") ? scenario_.substr(4)
                                                           : std::string_view{};
        if (part_ == 0U) {
            std::copy(outgoing.begin(), outgoing.end(), encoded_.begin());
            selected_ = headers_++ % (config_.world_size - 1U) == 0U &&
                        std::to_integer<unsigned>(outgoing[7]) == (gather ? 10U : 9U);
            if (config_.rank == 1U) {
                if (scenario_ == "failure_cumulative") {
                    std::this_thread::sleep_for(400ms);
                } else if (scenario_ == "failure_delay" && calls == 1U) {
                    std::this_thread::sleep_for(80ms);
                }
                if (selected_ && (field == "version" || field == "operation")) {
                    auto changed = std::vector<std::byte>(outgoing.begin(), outgoing.end());
                    changed[field == "version" ? 5U : 7U] =
                        static_cast<std::byte>(field == "version" ? 1U
                                               : gather           ? 9U
                                                                  : 10U);
                    injected = true;
                    inner_->exchange_bytes(send, changed, receive, incoming, deadline);
                    part_ = 1;
                    return;
                }
            }
        }
        if (part_ == 1U) {
            std::copy(outgoing.begin(), outgoing.end(), encoded_.begin() + 8);
            const auto frame = wire::decode_header(encoded_, config_.max_payload_bytes);
            if (config_.rank == 1U && selected_ && !field.empty() && !injected) {
                auto changed = std::vector<std::byte>(outgoing.begin(), outgoing.end());
                const auto change = [&](std::size_t offset, unsigned value) {
                    changed[offset - wire::prefix_bytes] = static_cast<std::byte>(value);
                };
                if (field == "phase") {
                    change(51, 0);
                } else if (field == "step") {
                    change(55, 1);
                } else if (field == "chunk") {
                    change(59, (frame.chunk + 1U) % 3U);
                } else if (field == "offset") {
                    change(79, 1);
                } else if (field == "total") {
                    change(71, 14);
                } else if (field == "count") {
                    change(31, 1);
                    change(39, 4);
                } else if (field == "algorithm") {
                    change(49, 1);
                } else if (field == "dtype") {
                    change(41, 2);
                } else if (field == "reduction") {
                    change(43, 2);
                } else if (field == "source") {
                    change(11, 4);
                } else if (field == "destination") {
                    change(15, 4);
                } else if (field == "sequence") {
                    change(23, 1);
                } else if (field == "oversize") {
                    change(64, 255);
                } else if (field == "reserved") {
                    change(63, 1);
                } else {
                    throw std::runtime_error("unknown ring corruption");
                }
                injected = true;
                inner_->exchange_bytes(send, changed, receive, incoming, deadline);
                part_ = 2;
                return;
            }
        }
        if (config_.rank == 1U && part_ == 2U && calls == 3U) {
            if (scenario_ == "failure_eof") {
                inner_->close();
                throw EngineError(ErrorCode::peer_closed, "injected payload closure",
                                  net::peer_context(config_, send, "exchange.send"));
            }
            if (scenario_ == "failure_drip") {
                inner_->exchange_bytes(send, outgoing.first(1), receive, incoming, deadline);
                for (std::size_t offset = 1; offset < outgoing.size(); ++offset) {
                    std::this_thread::sleep_for(90ms);
                    inner_->send_bytes(send, outgoing.subspan(offset, 1), deadline);
                }
                part_ = 0;
                return;
            }
        }
        inner_->exchange_bytes(send, outgoing, receive, incoming, deadline);
        part_ = (part_ + 1U) % 3U;
    }
    void close() noexcept override { inner_->close(); }
    std::size_t calls{};
    bool injected{};

  private:
    const RunConfig config_;
    std::unique_ptr<ByteTransport> inner_;
    std::string_view scenario_;
    std::latch* entered_;
    unsigned part_{};
    unsigned headers_{};
    bool selected_{};
    std::array<std::byte, wire::header_bytes> encoded_{};
};

void one(const RunConfig& config, std::string_view scenario) {
    std::latch entered(1);
    auto observed = std::make_unique<FaultTransport>(
        config, connect_tcp(config), scenario,
        scenario == "failure_cancel" && config.rank == 0U ? &entered : nullptr);
    auto* observation = observed.get();
    Communicator communicator(config, std::move(observed));
    const std::size_t count = scenario == "failure_limit"  ? 5U
                              : scenario == "failure_tiny" ? 1U
                                                           : 13U;
    std::vector<float> values(count, static_cast<float>(config.rank + 1U));
    if (scenario == "failure_order") {
        const std::array<float, 3> contribution{1e20F, -1e20F, 3.0F};
        std::fill(values.begin(), values.end(), contribution[config.rank]);
    }
    const auto before = std::chrono::steady_clock::now();
    if (scenario == "failure_cancel" && config.rank == 0U) {
        std::exception_ptr exception;
        std::jthread worker([&] {
            try {
                communicator.all_reduce_sum(values);
            } catch (...) {
                exception = std::current_exception();
            }
        });
        entered.wait();
        expect_error([&] { communicator.all_reduce_sum({}); }, ErrorCode::invalid_state);
        require(communicator.state() == CommunicatorState::ready,
                "concurrent rejection damaged active ring");
        communicator.close();
        communicator.close();
        worker.join();
        require(exception != nullptr && communicator.state() == CommunicatorState::closed &&
                    communicator.next_sequence() == 0,
                "ring cancellation lost terminal state or sequence");
        return;
    }
    bool failed = false;
    try {
        communicator.all_reduce_sum(values);
    } catch (const EngineError& error) {
        failed = true;
        require(error.context().rank == config.rank && error.context().sequence == 0 &&
                    error.context().operation.starts_with("all_reduce_sum"),
                "ring failure lost context");
        require(communicator.state() == CommunicatorState::failed &&
                    communicator.next_sequence() == 0,
                "ring failure advanced sequence or allowed reuse");
        expect_error([&] { communicator.all_reduce_sum({}); }, ErrorCode::invalid_state);
        if (scenario.starts_with("bad_") || scenario.starts_with("gather_")) {
            require(config.rank == 1U ? observation->injected : true,
                    "corrupt frame was not exercised");
            if (config.rank == 2U) {
                require(error.code() == ErrorCode::protocol_mismatch, error.what());
            }
        } else if (scenario == "failure_limit") {
            require(error.code() == ErrorCode::invalid_buffer && observation->calls == 0,
                    "logical cap was not checked before chunk traffic");
        } else {
            require(error.code() == ErrorCode::timeout || error.code() == ErrorCode::peer_closed,
                    error.what());
        }
        std::cout << "ring fault=" << scenario << " rank=" << config.rank
                  << " category=" << to_string(error.code()) << " elapsed_ms="
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - before)
                         .count()
                  << " peer=" << error.context().peer.value_or(99) << '\n';
    }
    const bool success = scenario == "failure_delay" || scenario == "failure_order" ||
                         scenario == "failure_recreate" || scenario == "failure_tiny";
    require(failed != success, "ring fault returned an unexpected success/failure");
    if (success) {
        require(communicator.next_sequence() == 1, "successful delayed ring lost sequence");
        for (std::size_t index = 0; index < count; ++index) {
            if (scenario != "failure_order") {
                require(values[index] == 6, "delayed ring changed sum");
            } else {
                const std::uint32_t chunk = index < 5U ? 0U : (index < 9U ? 1U : 2U);
                const std::array<float, 3> contribution{1e20F, -1e20F, 3.0F};
                float expected = contribution[chunk];
                for (std::uint32_t step = 1; step < 3; ++step) {
                    expected = contribution[(chunk + step) % 3U] + expected;
                }
                require(values[index] == expected,
                        "ring cancellation fixture changed addition order");
            }
        }
    }
    require(std::chrono::steady_clock::now() - before < 1800ms,
            "ring steps renewed original deadline");
}
} // namespace

int failure(RunConfig settings, std::string_view scenario) {
    const auto baseline = descriptors();
    if (scenario == "failure_limit") {
        settings.max_payload_bytes = 16;
    } else if (scenario == "failure_tiny") {
        settings.max_payload_bytes = 4;
    }
    const unsigned attempts = scenario == "failure_recreate" ? 20U : 1U;
    for (unsigned iteration = 0; iteration < attempts; ++iteration) {
        auto attempt = settings;
        attempt.run_id += '-' + std::to_string(iteration);
        one(attempt, scenario);
        require(descriptors() == baseline, "ring recreate leaked descriptors");
    }
    return 0;
}
} // namespace ringtest
