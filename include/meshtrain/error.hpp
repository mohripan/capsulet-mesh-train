#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace meshtrain {

enum class ErrorCode {
    invalid_configuration,
    protocol_mismatch,
    peer_closed,
    timeout,
    invalid_buffer,
    invalid_state,
    transport_failure,
};

[[nodiscard]] std::string_view to_string(ErrorCode code) noexcept;

struct ErrorContext {
    std::optional<std::uint32_t> rank;
    std::string run_id;
    std::string operation;
    std::optional<std::uint64_t> sequence;
    std::optional<std::uint32_t> peer;
    std::string peer_endpoint;
};

class EngineError : public std::runtime_error {
  public:
    EngineError(ErrorCode code, std::string message, ErrorContext context = {});

    [[nodiscard]] ErrorCode code() const noexcept { return code_; }
    // Original text without diagnostic formatting, for adding layer context.
    [[nodiscard]] const std::string& message() const noexcept { return message_; }
    [[nodiscard]] const ErrorContext& context() const noexcept { return context_; }

  private:
    ErrorCode code_;
    ErrorContext context_;
    std::string message_;
};

} // namespace meshtrain
