#include "failure_control.hpp"
#include "protocol.hpp"
#include "ring_fixture.hpp"

#include <meshtrain/communicator.hpp>

#include <sys/socket.h>

#include <memory>

namespace {
fault::Settings active{};
std::uint32_t current_rank{};
bool hello_sent{};
fault::Clock::time_point startup_gate;

class ObservedTransport final : public meshtrain::ByteTransport {
  public:
    explicit ObservedTransport(std::unique_ptr<ByteTransport> transport)
        : transport_(std::move(transport)) {}
    void send_bytes(std::uint32_t peer, std::span<const std::byte> bytes,
                    meshtrain::Deadline deadline) override {
        observe(current_rank == 1 ? 'S' : 'T', deadline);
        transport_->send_bytes(peer, bytes, deadline);
    }
    void receive_bytes(std::uint32_t peer, std::span<std::byte> bytes,
                       meshtrain::Deadline deadline) override {
        observe('R', deadline);
        transport_->receive_bytes(peer, bytes, deadline);
    }
    void close() noexcept override { transport_->close(); }

    void exchange_bytes(std::uint32_t send, std::span<const std::byte> outgoing,
                        std::uint32_t receive, std::span<std::byte> incoming,
                        meshtrain::Deadline deadline) override {
        if (part_ < 2U) {
            std::copy(outgoing.begin(), outgoing.end(), encoded_.begin() + (part_ == 0U ? 0 : 8));
        }
        if (part_ == 1U) {
            const auto frame = meshtrain::wire::decode_header(encoded_, 1024);
            if (frame.phase == (active.gather ? meshtrain::wire::Phase::all_gather
                                              : meshtrain::wire::Phase::reduce_scatter)) {
                observe(current_rank == 0 ? 'R' : (current_rank == 1 ? 'S' : 'T'), deadline);
            }
        }
        transport_->exchange_bytes(send, outgoing, receive, incoming, deadline);
        part_ = (part_ + 1U) % 3U;
    }

  private:
    void observe(char marker, meshtrain::Deadline deadline) {
        if (!observed_) {
            observed_ = true;
            fault::gate(marker, deadline);
        }
    }
    std::unique_ptr<ByteTransport> transport_;
    bool observed_{};
    unsigned part_{};
    std::array<std::byte, meshtrain::wire::header_bytes> encoded_{};
};
} // namespace

extern "C" int __real_listen(int, int);
extern "C" ssize_t __real_send(int, const void*, std::size_t, int);

extern "C" int __wrap_listen(int descriptor, int backlog) {
    const auto result = __real_listen(descriptor, backlog);
    if (result == 0 && active.startup && active.target == 0 && current_rank == 0) {
        fault::gate('L', startup_gate);
    }
    return result;
}

extern "C" ssize_t __wrap_send(int descriptor, const void* bytes, std::size_t count, int flags) {
    if (active.startup && active.target == 1 && current_rank == 1 && !hello_sent) {
        hello_sent = true;
        fault::gate('H', startup_gate);
    }
    return __real_send(descriptor, bytes, count, flags);
}

namespace fault {
void gate(char marker, Clock::time_point deadline) {
    if (!active.omit_phase) {
        send(3, marker);
    }
    require(receive(4, deadline) == 'G', "invalid phase release");
}

int worker(const Settings& scenario, std::uint32_t rank, std::uint16_t port, std::string run) {
    using namespace meshtrain;
    active = scenario;
    current_rank = rank;
    auto config = test::tcp_group(rank, 3, port, "127.0.0.1", run);
    if (scenario.ring) {
        config = ringtest::config(rank, 3, port, "127.0.0.1", run);
    }
    config.startup_timeout = 1000ms;
    config.collective_timeout = 1000ms;
    Result report;
    report.rank = rank;
    report.before = descriptors();
    startup_gate = Clock::now() + 900ms;
    bool joined = false;
    try {
        auto transport = connect_tcp(config);
        joined = true;
        require(!scenario.startup, "startup fault unexpectedly created a ready group");
        Communicator communicator(config,
                                  std::make_unique<ObservedTransport>(std::move(transport)));
        std::array<float, 3> values{1, 2, 3};
        try {
            communicator.all_reduce_sum(values);
            throw std::runtime_error("faulted collective unexpectedly succeeded");
        } catch (const EngineError& error) {
            require(communicator.state() == CommunicatorState::failed &&
                        communicator.next_sequence() == 0,
                    "failed communicator lost state/sequence");
            test::expect_error([&] { communicator.all_reduce_sum(values); },
                               ErrorCode::invalid_state);
            report.terminal = true;
            communicator.close();
            communicator.close();
            throw EngineError(error.code(), error.message(), error.context());
        }
    } catch (const EngineError& error) {
        require(error.context().rank == rank && error.context().run_id == run,
                "failure lacks matching rank/run context");
        require(error.code() == ErrorCode::peer_closed || error.code() == ErrorCode::timeout ||
                    (scenario.startup && error.code() == ErrorCode::transport_failure),
                error.what());
        require(scenario.startup ? !joined : report.terminal, "failure at wrong native phase");
        report.has_sequence = error.context().sequence == 0;
        report.has_peer = error.context().peer.has_value();
        require(scenario.startup || (report.has_sequence && report.has_peer),
                "collective failure lacks sequence/peer");
        report.code = error.code();
        copy(report.operation, error.context().operation);
        copy(report.endpoint, error.context().peer_endpoint);
        require(!error.context().operation.empty() && !error.context().peer_endpoint.empty(),
                "failure lacks operation/endpoint");
        std::cout << "rank=" << rank << " expected_engine_failure=" << error.what() << std::endl;
    }
    report.after = descriptors();
    require(report.before == report.after, "surviving worker leaked native descriptors");
    report.valid = true;
    require(::write(5, &report, sizeof(report)) == static_cast<ssize_t>(sizeof(report)),
            "cleanup report write failed");
    require(receive(4, Clock::now() + 3s) == 'E', "missing live-cleanup acknowledgement");
    return 1;
}
} // namespace fault
