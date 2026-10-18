#include "test_support.hpp"

#include <meshtrain/communicator.hpp>

#include <array>
#include <bit>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using namespace meshtrain;

struct Lifetime {
    bool closed{};
    bool destroyed{};
    unsigned calls{};
};

class TestTransport final : public ByteTransport {
  public:
    explicit TestTransport(std::shared_ptr<Lifetime> lifetime, bool allocation_failure = false)
        : lifetime_(std::move(lifetime)), allocation_failure_(allocation_failure) {}
    ~TestTransport() override { lifetime_->destroyed = true; }
    void send_bytes(std::uint32_t, std::span<const std::byte>, Deadline) override { fail(); }
    void receive_bytes(std::uint32_t, std::span<std::byte>, Deadline) override { fail(); }
    void close() noexcept override { lifetime_->closed = true; }

  private:
    [[noreturn]] void fail() {
        ++lifetime_->calls;
        if (allocation_failure_) {
            throw std::bad_alloc();
        }
        throw EngineError(ErrorCode::peer_closed, "injected\nclosure");
    }
    std::shared_ptr<Lifetime> lifetime_;
    bool allocation_failure_;
};

RunConfig config(std::uint32_t world = 1) {
    RunConfig result;
    result.world_size = world;
    result.run_id = "communicator-contract";
    result.context_fingerprint = "layout";
    for (std::uint32_t rank = 0; rank < world; ++rank) {
        result.endpoints.push_back({"worker-" + std::to_string(rank), 45000});
    }
    return result;
}

void identity_and_ownership(AllReduceAlgorithm algorithm) {
    const auto lifetime = std::make_shared<Lifetime>();
    {
        auto settings = config();
        settings.all_reduce_algorithm = algorithm;
        Communicator communicator(settings, std::make_unique<TestTransport>(lifetime));
        const std::array<std::uint32_t, 6> bits{0,           0x80000000U, 0x7f800000U,
                                                0xff800000U, 0x7fc01234U, 0x00800000U};
        std::vector<float> values;
        for (const auto value : bits) {
            values.push_back(std::bit_cast<float>(value));
        }
        communicator.all_reduce_sum(values);
        for (std::size_t index = 0; index < bits.size(); ++index) {
            test::require(std::bit_cast<std::uint32_t>(values[index]) == bits[index],
                          "one-rank identity changed storage bits");
        }
        communicator.all_reduce_sum({});
        test::require(communicator.state() == CommunicatorState::ready &&
                          communicator.next_sequence() == 2 && lifetime->calls == 0,
                      "identity/empty sequence or no-I/O contract failed");
    }
    test::require(lifetime->closed && lifetime->destroyed, "owned transport was not released");
}

void local_failure_and_close() {
    auto settings = config();
    settings.max_payload_bytes = 4;
    const auto lifetime = std::make_shared<Lifetime>();
    Communicator communicator(settings, std::make_unique<TestTransport>(lifetime));
    std::array<float, 2> values{1, 2};
    test::expect_error([&] { communicator.all_reduce_sum(values); }, ErrorCode::invalid_buffer);
    test::require(communicator.state() == CommunicatorState::failed &&
                      communicator.next_sequence() == 0 && lifetime->closed && lifetime->calls == 0,
                  "local validation did not fail the group before I/O");
    test::expect_error([&] { communicator.all_reduce_sum({}); }, ErrorCode::invalid_state);
    test::require(communicator.state() == CommunicatorState::failed, "reuse changed failed state");
    communicator.close();
    communicator.close();
    test::require(communicator.state() == CommunicatorState::closed, "explicit close not terminal");
    test::expect_error([&] { communicator.all_reduce_sum({}); }, ErrorCode::invalid_state);
    test::expect_error([&] { Communicator invalid(config(), nullptr); },
                       ErrorCode::invalid_configuration);
}

void error_context_and_exception_cleanup() {
    const auto lifetime = std::make_shared<Lifetime>();
    Communicator communicator(config(2), std::make_unique<TestTransport>(lifetime));
    std::array<float, 1> value{1};
    bool caught = false;
    try {
        communicator.all_reduce_sum(value);
    } catch (const EngineError& error) {
        caught = true;
        test::require(error.code() == ErrorCode::peer_closed && error.context().rank == 0 &&
                          error.context().sequence == 0 && error.context().peer == 1 &&
                          error.context().run_id == "communicator-contract" &&
                          error.context().peer_endpoint == "worker-1:45000" &&
                          error.context().operation == "all_reduce_sum.receive_header" &&
                          error.message() == "injected\nclosure",
                      "transport failure lost cause or collective context");
    }
    test::require(caught && lifetime->closed && communicator.next_sequence() == 0 &&
                      communicator.state() == CommunicatorState::failed,
                  "transport failure did not make communicator unusable");
    const auto allocation = std::make_shared<Lifetime>();
    Communicator another(config(2), std::make_unique<TestTransport>(allocation, true));
    caught = false;
    try {
        another.all_reduce_sum(value);
    } catch (const std::bad_alloc&) {
        caught = true;
    }
    test::require(caught && allocation->closed && another.state() == CommunicatorState::failed,
                  "non-engine exception skipped terminal cleanup");
}

void ring_adoption() {
    for (const auto world : {1U, 2U}) {
        auto settings = config(world);
        settings.all_reduce_algorithm = AllReduceAlgorithm::ring;
        const auto lifetime = std::make_shared<Lifetime>();
        Communicator communicator(settings, std::make_unique<TestTransport>(lifetime));
        std::array<float, 1> value{3};
        if (world == 1U) {
            communicator.all_reduce_sum(value);
            test::require(value[0] == 3 && communicator.next_sequence() == 1,
                          "ring identity failed");
        } else {
            test::expect_error([&] { communicator.all_reduce_sum(value); },
                               ErrorCode::invalid_configuration);
            test::require(communicator.state() == CommunicatorState::failed && lifetime->closed &&
                              communicator.next_sequence() == 0 && lifetime->calls == 0,
                          "unsupported duplex adapter did not fail terminally");
        }
    }
}
} // namespace

int main() {
    try {
        identity_and_ownership(AllReduceAlgorithm::centralized);
        identity_and_ownership(AllReduceAlgorithm::ring);
        local_failure_and_close();
        error_context_and_exception_cleanup();
        ring_adoption();
        std::cout << "PASS communicator identity, ownership, lifecycle, and contextual failures\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
