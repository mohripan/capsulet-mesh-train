#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>

namespace meshtrain {

using Deadline = std::chrono::steady_clock::time_point;

// Exact-byte, synchronous host transport. Implementations own their resources;
// buffers are borrowed only for the duration of a call. The same absolute
// deadline applies across all partial transfers and retries in that call.
// Socket descriptors and framework/device types never cross this boundary.
class ByteTransport {
  public:
    virtual ~ByteTransport() = default;

    virtual void send_bytes(std::uint32_t peer, std::span<const std::byte> bytes,
                            Deadline deadline) = 0;
    virtual void receive_bytes(std::uint32_t peer, std::span<std::byte> bytes,
                               Deadline deadline) = 0;

    // One synchronous operation progressing both directions; nonempty spans must
    // be disjoint. Adapters without duplex support throw invalid_configuration.
    virtual void exchange_bytes(std::uint32_t send_peer, std::span<const std::byte> outgoing,
                                std::uint32_t receive_peer, std::span<std::byte> incoming,
                                Deadline deadline);

    // Idempotent, noexcept, and safe concurrently with an operation. An
    // implementation must unblock outstanding I/O. Join active callers before
    // destroying the transport; the owner retains its lifetime during close.
    virtual void close() noexcept = 0;

  protected:
    ByteTransport() = default;
    ByteTransport(const ByteTransport&) = delete;
    ByteTransport& operator=(const ByteTransport&) = delete;
    ByteTransport(ByteTransport&&) = delete;
    ByteTransport& operator=(ByteTransport&&) = delete;
};

} // namespace meshtrain
