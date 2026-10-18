#include "collective_fixture.hpp"

#include "protocol.hpp"

#include <signal.h>

#include <array>
#include <exception>
#include <latch>
#include <thread>

namespace test {
namespace {
using namespace meshtrain;

wire::Header data_header(wire::Operation operation, std::uint32_t source, std::uint32_t destination,
                         std::uint64_t sequence = 0, std::uint64_t count = 3) {
    wire::Header result{operation, source,    destination,          sequence,
                        count,     count * 4, wire::Dtype::float32, wire::Reduction::sum};
    result.total_count = count;
    return result;
}

void send_header(ByteTransport& transport, const RunConfig& config, const wire::Header& header) {
    const auto bytes = wire::encode_header(header, config.max_payload_bytes);
    transport.send_bytes(header.destination, bytes, after());
}

void await_closure(ByteTransport& transport, std::uint32_t peer) {
    std::array<std::byte, 1> byte{};
    expect_error([&] { transport.receive_bytes(peer, byte, after()); }, ErrorCode::peer_closed);
}

void raw_request(ByteTransport& transport, const RunConfig& config) {
    std::array<std::byte, wire::header_bytes> bytes{};
    transport.receive_bytes(1, bytes, after());
    const auto header = wire::decode_header(bytes, config.max_payload_bytes);
    require(header.operation == wire::Operation::all_reduce_request && header.count == 3,
            "fault fixture received unexpected request");
    std::array<float, 3> payload{};
    transport.receive_bytes(1, std::as_writable_bytes(std::span(payload)), after());
}

void raw_round_trip(ByteTransport& transport, const RunConfig& config) {
    const auto header = data_header(wire::Operation::all_reduce_request, config.rank, 0);
    send_header(transport, config, header);
    std::array<float, 3> payload{1, 2, 3};
    transport.send_bytes(0, std::as_bytes(std::span(payload)), after());
    std::array<std::byte, wire::header_bytes> response{};
    transport.receive_bytes(0, response, after());
    const auto result = wire::decode_header(response, config.max_payload_bytes);
    require(result.operation == wire::Operation::all_reduce_result && result.sequence == 0 &&
                result.count == 3,
            "priming collective returned wrong metadata");
    transport.receive_bytes(0, std::as_writable_bytes(std::span(payload)), after());
    require(payload == std::array<float, 3>{2, 4, 6}, "priming collective returned wrong SUM");
}

void malicious_worker(std::string_view scenario, const RunConfig& config) {
    const auto transport = connect_tcp(config);
    if (scenario == "cancel") {
        await_closure(*transport, 0);
        return;
    }
    if (scenario == "stale_sequence") {
        raw_round_trip(*transport, config);
    }
    auto header = data_header(wire::Operation::all_reduce_request, config.rank, 0);
    if (scenario == "future_sequence") {
        header.sequence = 1;
    } else if (scenario == "wrong_source") {
        header.source = config.world_size;
    } else if (scenario == "wrong_destination") {
        header.destination = config.world_size;
    } else if (scenario == "wrong_operation") {
        header.operation = wire::Operation::all_reduce_result;
    }
    auto bytes = wire::encode_header(header, config.max_payload_bytes);
    if (scenario == "bad_dtype") {
        bytes[41] = std::byte{2};
    } else if (scenario == "bad_reduction") {
        bytes[43] = std::byte{2};
    } else if (scenario == "bad_count_bytes") {
        bytes[39] = std::byte{4};
    } else if (scenario == "oversize_data") {
        bytes[24] = std::byte{1};
        bytes[32] = std::byte{4}; // Consistent huge count/bytes, above the agreed bound.
    }
    // Destination mutations are wire errors; send them on the actual root stream.
    transport->send_bytes(0, bytes, after());
    if (scenario == "kill_worker") {
        static_cast<void>(::raise(SIGKILL));
        throw std::runtime_error("SIGKILL unexpectedly returned");
    }
    if (scenario == "cumulative_deadline" || scenario == "slow_payload") {
        try {
            std::array<float, 3> payload{1, 2, 3};
            const auto body = std::as_bytes(std::span(payload));
            if (scenario == "cumulative_deadline") {
                std::this_thread::sleep_for(config.rank == 1 ? 650ms : 1300ms);
                transport->send_bytes(0, body, after());
            } else {
                for (const auto byte : body) {
                    transport->send_bytes(0, std::span(&byte, 1), after());
                    std::this_thread::sleep_for(350ms);
                }
            }
        } catch (const EngineError& error) {
            require(error.code() == ErrorCode::peer_closed, error.what());
            return;
        }
    }
    await_closure(*transport, 0);
}

void malicious_root(std::string_view scenario, const RunConfig& config) {
    const auto transport = connect_tcp(config);
    raw_request(*transport, config);
    if (scenario == "kill_root") {
        static_cast<void>(::raise(SIGKILL));
        throw std::runtime_error("SIGKILL unexpectedly returned");
    }
    if (scenario == "stalled_result") {
        await_closure(*transport, 1);
        return;
    }
    auto header = data_header(wire::Operation::all_reduce_result, 0, 1);
    if (scenario == "result_sequence") {
        header.sequence = 1;
    } else if (scenario == "result_count") {
        header.count = 1;
        header.total_count = 1;
        header.payload_bytes = 4;
    } else if (scenario == "result_source") {
        header.source = config.world_size;
    } else if (scenario == "result_destination") {
        header.destination = config.world_size;
    }
    auto bytes = wire::encode_header(header, config.max_payload_bytes);
    if (scenario == "result_dtype") {
        bytes[41] = std::byte{2};
    }
    transport->send_bytes(1, bytes, after());
    await_closure(*transport, 1);
}

class ObservedTransport final : public ByteTransport {
  public:
    ObservedTransport(std::unique_ptr<ByteTransport> transport, std::latch& entered)
        : transport_(std::move(transport)), entered_(entered) {}
    void send_bytes(std::uint32_t peer, std::span<const std::byte> bytes,
                    Deadline deadline) override {
        transport_->send_bytes(peer, bytes, deadline);
    }
    void receive_bytes(std::uint32_t peer, std::span<std::byte> bytes, Deadline deadline) override {
        if (!notified_) {
            notified_ = true;
            entered_.count_down(); // The communicator has admitted its first call.
        }
        transport_->receive_bytes(peer, bytes, deadline);
    }
    void close() noexcept override { transport_->close(); }

  private:
    std::unique_ptr<ByteTransport> transport_;
    std::latch& entered_;
    bool notified_{};
};

void cancel(const RunConfig& config) {
    std::latch entered(1);
    Communicator communicator(config,
                              std::make_unique<ObservedTransport>(connect_tcp(config), entered));
    std::exception_ptr result;
    std::array<float, 3> values{};
    std::jthread caller([&] {
        try {
            communicator.all_reduce_sum(values);
        } catch (...) {
            result = std::current_exception();
        }
    });
    entered.wait();
    expect_error([&] { communicator.all_reduce_sum({}); }, ErrorCode::invalid_state);
    require(communicator.state() == CommunicatorState::ready,
            "concurrent-call rejection failed the active group");
    const auto before = std::chrono::steady_clock::now();
    communicator.close();
    communicator.close();
    caller.join();
    require(result != nullptr && communicator.state() == CommunicatorState::closed &&
                communicator.next_sequence() == 0 && std::chrono::steady_clock::now() - before < 1s,
            "close did not cancel active collective with closed state and unchanged sequence");
    try {
        std::rethrow_exception(result);
    } catch (const EngineError& error) {
        require(error.code() == ErrorCode::peer_closed || error.code() == ErrorCode::invalid_state,
                error.what());
    }
    expect_error([&] { communicator.all_reduce_sum({}); }, ErrorCode::invalid_state);
}
} // namespace

void collective_failure(std::string_view scenario, RunConfig config) {
    config.collective_timeout = 1s;
    config.max_payload_bytes = 12;
    const bool root_fault =
        scenario.starts_with("result_") || scenario == "stalled_result" || scenario == "kill_root";
    if (root_fault && config.rank == 0) {
        malicious_root(scenario, config);
        return;
    }
    const bool ordinary_worker = scenario == "count" || scenario == "local_limit" ||
                                 (scenario == "kill_worker" && config.rank == 2);
    if (!root_fault && config.rank != 0 && !ordinary_worker) {
        malicious_worker(scenario, config);
        return;
    }
    if (scenario == "cancel") {
        cancel(config);
        return;
    }
    Communicator communicator(config, connect_tcp(config));
    std::vector<float> values{1, 2, 3};
    if (scenario == "stale_sequence") {
        communicator.all_reduce_sum(values);
    } else if (scenario == "count" && config.rank != 0) {
        values.resize(1);
    } else if (scenario == "local_limit" && config.rank != 0) {
        values.resize(4);
    }
    const auto original = values;
    const bool timeout = scenario == "stall_payload" || scenario == "slow_payload" ||
                         scenario == "cumulative_deadline" || scenario == "stalled_result";
    const auto expected =
        timeout                                         ? ErrorCode::timeout
        : scenario == "local_limit" && config.rank != 0 ? ErrorCode::invalid_buffer
        : scenario == "local_limit" || scenario == "kill_root" || scenario == "kill_worker" ||
                (scenario == "count" && config.rank != 0)
            ? ErrorCode::peer_closed
            : ErrorCode::protocol_mismatch;
    failed_collective(
        communicator, config, [&] { communicator.all_reduce_sum(values); }, expected,
        scenario == "stale_sequence" ? 1U : 0U);
    if (scenario.starts_with("result_")) {
        require(values == original, "invalid result header modified the caller buffer");
    }
}
} // namespace test
