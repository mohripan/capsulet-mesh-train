#include <meshtrain/communicator.hpp>

#include "deadline.hpp"
#include "protocol.hpp"
#include "ring.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace meshtrain {

class Communicator::Impl {
  public:
    Impl(RunConfig config, std::unique_ptr<ByteTransport> transport)
        : config_(std::move(config)), transport_(std::move(transport)) {
        validate(config_);
        if (transport_ == nullptr) {
            throw EngineError(ErrorCode::invalid_configuration,
                              "communicator requires an already joined transport", context(0));
        }
    }

    void all_reduce_sum(HostFloat32Buffer buffer) {
        std::unique_lock operation_lock(operation_, std::try_to_lock);
        if (!operation_lock.owns_lock()) {
            throw EngineError(ErrorCode::invalid_state, "concurrent collectives are unsupported",
                              context(next_sequence()));
        }
        ErrorContext current;
        try {
            // Context construction can allocate too. Every admitted exception
            // must close the group, including failures before the first I/O.
            current = context(next_sequence());
            require_ready(current); // Terminal states are preserved by fail().
            const auto deadline = detail::deadline_after(config_.collective_timeout, current);
            const auto bytes =
                checked_payload_bytes(buffer.size(), config_.max_payload_bytes, current);
            if (*current.sequence == std::numeric_limits<std::uint64_t>::max()) {
                throw EngineError(ErrorCode::invalid_state, "collective sequence exhausted",
                                  current);
            }
            if (config_.world_size > 1U) {
                if (config_.all_reduce_algorithm == AllReduceAlgorithm::ring) {
                    detail::ring_all_reduce(config_, *transport_, buffer, deadline, current,
                                            [&] { check_active(deadline, current); });
                } else if (config_.rank == 0U) {
                    reduce_and_distribute(buffer, bytes, deadline, current);
                } else {
                    exchange(buffer, bytes, deadline, current);
                }
            }
            detail::check_deadline(deadline, current);
            const std::lock_guard lock(lifecycle_);
            if (state_ != CommunicatorState::ready) {
                throw EngineError(ErrorCode::invalid_state, "communicator closed during collective",
                                  current);
            }
            ++sequence_;
        } catch (const EngineError& error) {
            fail();
            // Socket errors know the peer but not this collective's sequence.
            // Preserve the original cause without nesting formatted diagnostics.
            if (error.context().peer.has_value()) {
                current.peer = error.context().peer;
                current.peer_endpoint = error.context().peer_endpoint;
            }
            throw EngineError(error.code(), error.message(), std::move(current));
        } catch (...) {
            fail();
            throw;
        }
    }

    void close() noexcept {
        {
            const std::lock_guard lock(lifecycle_);
            state_ = CommunicatorState::closed;
        }
        transport_->close();
    }
    [[nodiscard]] CommunicatorState state() const noexcept {
        const std::lock_guard lock(lifecycle_);
        return state_;
    }
    [[nodiscard]] std::uint64_t next_sequence() const noexcept {
        const std::lock_guard lock(lifecycle_);
        return sequence_;
    }

  private:
    ErrorContext context(std::uint64_t sequence) const {
        ErrorContext result;
        result.rank = config_.rank;
        result.run_id = config_.run_id;
        result.sequence = sequence;
        result.operation = "all_reduce_sum";
        return result;
    }
    void target(ErrorContext& current, std::uint32_t peer, std::string operation) const {
        current.peer = peer;
        current.peer_endpoint =
            config_.endpoints[peer].host + ':' + std::to_string(config_.endpoints[peer].port);
        current.operation = "all_reduce_sum." + std::move(operation);
    }
    void require_ready(const ErrorContext& current) const {
        if (state() != CommunicatorState::ready) {
            throw EngineError(ErrorCode::invalid_state, "communicator is failed or closed",
                              current);
        }
    }
    void check_active(Deadline deadline, const ErrorContext& current) const {
        detail::check_deadline(deadline, current);
        require_ready(current);
    }
    void fail() noexcept {
        {
            const std::lock_guard lock(lifecycle_);
            if (state_ == CommunicatorState::ready) {
                state_ = CommunicatorState::failed;
            }
        }
        transport_->close();
    }
    wire::Header header(wire::Operation operation, std::uint32_t peer, std::size_t count,
                        std::size_t bytes, const ErrorContext& current) const {
        wire::Header result{operation,
                            config_.rank,
                            peer,
                            *current.sequence,
                            count,
                            bytes,
                            wire::Dtype::float32,
                            wire::Reduction::sum};
        result.total_count = count;
        return result;
    }
    void send_header(wire::Operation operation, std::uint32_t peer, std::size_t count,
                     std::size_t bytes, Deadline deadline, ErrorContext& current) {
        target(current, peer, "send_header");
        check_active(deadline, current);
        const auto encoded = wire::encode_header(header(operation, peer, count, bytes, current),
                                                 config_.max_payload_bytes, current);
        transport_->send_bytes(peer, encoded, deadline);
    }
    void receive_header(wire::Operation expected, std::uint32_t peer, std::size_t count,
                        std::size_t bytes, Deadline deadline, ErrorContext& current) {
        target(current, peer, "receive_header");
        check_active(deadline, current);
        std::array<std::byte, wire::header_bytes> encoded{};
        transport_->receive_bytes(peer, std::span(encoded).first(wire::prefix_bytes), deadline);
        wire::validate_prefix(std::span(encoded).first(wire::prefix_bytes), current);
        transport_->receive_bytes(peer, std::span(encoded).subspan(wire::prefix_bytes), deadline);
        const auto received = wire::decode_header(encoded, config_.max_payload_bytes, current);
        if (received.operation != expected || received.source != peer ||
            received.destination != config_.rank || received.sequence != *current.sequence ||
            received.count != count || received.payload_bytes != bytes) {
            throw EngineError(ErrorCode::protocol_mismatch,
                              "collective operation, ranks, sequence, or count mismatch", current);
        }
    }
    void send_payload(std::uint32_t peer, HostFloat32Buffer buffer, Deadline deadline,
                      ErrorContext& current) {
        target(current, peer, "send_payload");
        check_active(deadline, current);
        transport_->send_bytes(peer, std::as_bytes(buffer), deadline);
    }
    void receive_payload(std::uint32_t peer, HostFloat32Buffer buffer, Deadline deadline,
                         ErrorContext& current) {
        target(current, peer, "receive_payload");
        check_active(deadline, current);
        transport_->receive_bytes(peer, std::as_writable_bytes(buffer), deadline);
    }
    void reduce_and_distribute(HostFloat32Buffer buffer, std::size_t bytes, Deadline deadline,
                               ErrorContext& current) {
        std::vector<float> scratch(buffer.size());
        for (std::uint32_t peer = 1; peer < config_.world_size; ++peer) {
            receive_header(wire::Operation::all_reduce_request, peer, buffer.size(), bytes,
                           deadline, current);
            receive_payload(peer, scratch, deadline, current);
            current.operation = "all_reduce_sum.reduce";
            for (std::size_t offset = 0; offset < buffer.size();) {
                check_active(deadline, current);
                const auto end = offset + std::min<std::size_t>(4096, buffer.size() - offset);
                for (; offset < end; ++offset) {
                    buffer[offset] += scratch[offset];
                }
            }
        }
        for (std::uint32_t peer = 1; peer < config_.world_size; ++peer) {
            send_header(wire::Operation::all_reduce_result, peer, buffer.size(), bytes, deadline,
                        current);
            send_payload(peer, buffer, deadline, current);
        }
    }
    void exchange(HostFloat32Buffer buffer, std::size_t bytes, Deadline deadline,
                  ErrorContext& current) {
        send_header(wire::Operation::all_reduce_request, 0, buffer.size(), bytes, deadline,
                    current);
        send_payload(0, buffer, deadline, current);
        receive_header(wire::Operation::all_reduce_result, 0, buffer.size(), bytes, deadline,
                       current);
        receive_payload(0, buffer, deadline, current);
    }

    const RunConfig config_;
    const std::unique_ptr<ByteTransport> transport_;
    mutable std::mutex lifecycle_;
    std::mutex operation_;
    CommunicatorState state_{CommunicatorState::ready};
    std::uint64_t sequence_{};
};

Communicator::Communicator(RunConfig config, std::unique_ptr<ByteTransport> transport)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(transport))) {}
Communicator::~Communicator() { close(); }
void Communicator::all_reduce_sum(HostFloat32Buffer buffer) { impl_->all_reduce_sum(buffer); }
void Communicator::close() noexcept { impl_->close(); }
CommunicatorState Communicator::state() const noexcept { return impl_->state(); }
std::uint64_t Communicator::next_sequence() const noexcept { return impl_->next_sequence(); }

} // namespace meshtrain
