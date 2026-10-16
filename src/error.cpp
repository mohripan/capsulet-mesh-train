#include <meshtrain/error.hpp>

#include <sstream>
#include <utility>

namespace meshtrain {
namespace {

std::string escaped(std::string_view value) {
    constexpr char hex[] = "0123456789ABCDEF";
    std::string result;
    for (const char character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (byte >= 32U && byte <= 126U && character != '"' && character != '\\') {
            result += character;
        } else {
            result += "\\x";
            result += hex[byte >> 4U];
            result += hex[byte & 15U];
        }
    }
    return result;
}

std::string diagnostic(ErrorCode code, std::string_view message, const ErrorContext& context) {
    std::ostringstream stream;
    stream << "meshtrain/" << to_string(code);
    if (!context.operation.empty()) {
        stream << " operation=\"" << escaped(context.operation) << '"';
    }
    if (!context.run_id.empty()) {
        stream << " run=\"" << escaped(context.run_id) << '"';
    }
    if (context.rank) {
        stream << " rank=" << *context.rank;
    }
    if (context.sequence) {
        stream << " sequence=" << *context.sequence;
    }
    if (context.peer) {
        stream << " peer=" << *context.peer;
    }
    if (!context.peer_endpoint.empty()) {
        stream << " endpoint=\"" << escaped(context.peer_endpoint) << '"';
    }
    stream << ": " << escaped(message);
    return stream.str();
}

} // namespace

std::string_view to_string(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::invalid_configuration:
        return "invalid_configuration";
    case ErrorCode::protocol_mismatch:
        return "protocol_mismatch";
    case ErrorCode::peer_closed:
        return "peer_closed";
    case ErrorCode::timeout:
        return "timeout";
    case ErrorCode::invalid_buffer:
        return "invalid_buffer";
    case ErrorCode::invalid_state:
        return "invalid_state";
    case ErrorCode::transport_failure:
        return "transport_failure";
    }
    return "unknown";
}

EngineError::EngineError(ErrorCode code, std::string message, ErrorContext context)
    : std::runtime_error(diagnostic(code, message, context)), code_(code),
      context_(std::move(context)), message_(std::move(message)) {}

} // namespace meshtrain
