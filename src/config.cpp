#include <meshtrain/config.hpp>

#include <meshtrain/error.hpp>

#include <algorithm>
#include <bit>
#include <limits>
#include <set>
#include <string_view>
#include <utility>

namespace meshtrain {
namespace {

bool printable_ascii(std::string_view value) {
    return std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return character >= 32U && character <= 126U;
    });
}

bool host_syntax(std::string_view host) {
    if (host.empty() || host.size() > 253U) {
        return false;
    }
    if (host.back() == '.') {
        host.remove_suffix(1);
    }
    if (host.empty()) {
        return false;
    }
    const auto alphanumeric = [](unsigned char character) {
        return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
               (character >= '0' && character <= '9');
    };
    while (!host.empty()) {
        const auto dot = host.find('.');
        const auto label = host.substr(0, dot);
        if (label.empty() || label.size() > 63U ||
            !alphanumeric(static_cast<unsigned char>(label.front())) ||
            !alphanumeric(static_cast<unsigned char>(label.back())) ||
            !std::all_of(label.begin(), label.end(), [&alphanumeric](unsigned char character) {
                return alphanumeric(character) || character == '-';
            })) {
            return false;
        }
        if (dot == std::string_view::npos) {
            break;
        }
        host.remove_prefix(dot + 1U);
        if (host.empty()) {
            return false;
        }
    }
    return true;
}

std::string normalized_host(std::string host) {
    for (char& character : host) {
        if (character >= 'A' && character <= 'Z') {
            character = static_cast<char>(character - 'A' + 'a');
        }
    }
    if (!host.empty() && host.back() == '.') {
        host.pop_back();
    }
    return host;
}

} // namespace

void validate(const RunConfig& config) {
    ErrorContext context;
    context.rank = config.rank;
    context.run_id = config.run_id;
    context.operation = "validate_config";
    const auto reject = [&context](std::string message) {
        throw EngineError(ErrorCode::invalid_configuration, std::move(message), context);
    };

    if (config.world_size == 0U) {
        reject("world_size must be positive");
    }
    if (config.rank >= config.world_size) {
        reject("rank must be less than world_size");
    }
    if (config.endpoints.size() != config.world_size) {
        reject("endpoints must contain exactly one entry per rank");
    }
    if (config.run_id.empty() || config.run_id.size() > 128U || !printable_ascii(config.run_id)) {
        reject("run_id must contain 1 to 128 printable ASCII bytes");
    }
    if (config.context_fingerprint.empty() || config.context_fingerprint.size() > 256U ||
        !printable_ascii(config.context_fingerprint)) {
        reject("context_fingerprint must contain 1 to 256 printable ASCII bytes");
    }
    if (config.startup_timeout.count() <= 0 || config.collective_timeout.count() <= 0) {
        reject("startup and collective timeouts must be positive");
    }
    if (config.max_payload_bytes == 0U) {
        reject("max_payload_bytes must be positive");
    }
    if (config.all_reduce_algorithm != AllReduceAlgorithm::centralized &&
        config.all_reduce_algorithm != AllReduceAlgorithm::ring) {
        reject("unsupported all_reduce_algorithm");
    }
    if (!host_syntax(config.listen_address)) {
        reject("listen_address must be an IPv4 address or hostname");
    }

    std::set<std::pair<std::string, std::uint16_t>> seen;
    for (std::size_t rank = 0; rank < config.endpoints.size(); ++rank) {
        const auto& endpoint = config.endpoints[rank];
        const auto host = normalized_host(endpoint.host);
        if (!host_syntax(endpoint.host) || host == "0.0.0.0" || host == "0" ||
            endpoint.port == 0U) {
            context.peer = static_cast<std::uint32_t>(rank);
            context.peer_endpoint = endpoint.host + ':' + std::to_string(endpoint.port);
            reject("peer endpoint requires a non-wildcard IPv4 address or hostname and a nonzero "
                   "port");
        }
        if (!seen.emplace(host, endpoint.port).second) {
            context.peer = static_cast<std::uint32_t>(rank);
            context.peer_endpoint = endpoint.host + ':' + std::to_string(endpoint.port);
            reject(
                "rank endpoints must be distinct (hostname comparison ignores case and final dot)");
        }
    }
    if constexpr (sizeof(float) != 4U || !std::numeric_limits<float>::is_iec559 ||
                  std::endian::native != std::endian::little) {
        reject("the initial protocol requires little-endian IEEE-754 binary32 hosts");
    }
}

} // namespace meshtrain
