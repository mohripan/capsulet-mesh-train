#pragma once

#include <meshtrain/buffer.hpp>
#include <meshtrain/config.hpp>
#include <meshtrain/transport.hpp>

#include <cstdint>
#include <memory>

namespace meshtrain {

enum class CommunicatorState { ready, failed, closed };

// Owns an already joined transport configured for this same fixed group.
// The transport must not be accessed through another alias after adoption.
class Communicator final {
  public:
    Communicator(RunConfig config, std::unique_ptr<ByteTransport> transport);
    ~Communicator();
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;
    Communicator(Communicator&&) = delete;
    Communicator& operator=(Communicator&&) = delete;

    // Blocking in-place FP32 SUM, with no implicit averaging. The caller owns
    // storage and prevents concurrent access. Empty buffers still participate.
    // One collective at a time; rejection of a concurrent call leaves the active
    // call intact. An admitted failure makes this group unusable and may leave
    // the input partially modified. Never apply an optimizer step after failure.
    void all_reduce_sum(HostFloat32Buffer buffer);

    // Safe concurrently with a collective; join callers before destruction.
    void close() noexcept;
    [[nodiscard]] CommunicatorState state() const noexcept;
    // Starts at zero, advances only on local success, and never wraps.
    [[nodiscard]] std::uint64_t next_sequence() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace meshtrain
