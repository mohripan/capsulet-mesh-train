#include "deadline.hpp"

namespace meshtrain::detail {

Deadline deadline_after(std::chrono::milliseconds timeout, const ErrorContext& context) {
    using Clock = std::chrono::steady_clock;
    if (timeout.count() <= 0 ||
        timeout > std::chrono::duration_cast<std::chrono::milliseconds>(Clock::duration::max())) {
        throw EngineError(ErrorCode::invalid_configuration,
                          "timeout cannot be represented by the monotonic clock", context);
    }
    const auto duration = std::chrono::duration_cast<Clock::duration>(timeout);
    const auto now = Clock::now();
    if (now.time_since_epoch() > Clock::duration::max() - duration) {
        throw EngineError(ErrorCode::invalid_configuration,
                          "deadline would overflow the monotonic clock", context);
    }
    return now + duration;
}

void check_deadline(Deadline deadline, const ErrorContext& context) {
    if (std::chrono::steady_clock::now() >= deadline) {
        throw EngineError(ErrorCode::timeout, "operation deadline expired", context);
    }
}

} // namespace meshtrain::detail
