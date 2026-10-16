#pragma once

#include <meshtrain/error.hpp>
#include <meshtrain/transport.hpp>

namespace meshtrain::detail {

[[nodiscard]] Deadline deadline_after(std::chrono::milliseconds timeout,
                                      const ErrorContext& context);
void check_deadline(Deadline deadline, const ErrorContext& context);

} // namespace meshtrain::detail
