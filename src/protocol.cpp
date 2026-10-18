#include "protocol.hpp"

#include <algorithm>
#include <limits>
#include <string_view>

namespace meshtrain::wire {
namespace {

constexpr std::uint32_t magic = 0x4D455348U; // MESH

void put(std::span<std::byte> bytes, std::size_t offset, std::uint64_t value, std::size_t width) {
    for (std::size_t index = 0; index < width; ++index) {
        bytes[offset + index] =
            static_cast<std::byte>((value >> (8U * (width - index - 1U))) & 255U);
    }
}

std::uint64_t get(std::span<const std::byte> bytes, std::size_t offset, std::size_t width) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < width; ++index) {
        value = (value << 8U) | std::to_integer<std::uint64_t>(bytes[offset + index]);
    }
    return value;
}

[[noreturn]] void reject(std::string message, const ErrorContext& context) {
    throw EngineError(ErrorCode::protocol_mismatch, std::move(message), context);
}

bool printable(std::string_view text) {
    return std::all_of(text.begin(), text.end(),
                       [](unsigned char byte) { return byte >= 32U && byte <= 126U; });
}

} // namespace

void validate_prefix(std::span<const std::byte> bytes, const ErrorContext& context) {
    if (bytes.size() != prefix_bytes || get(bytes, 0, 4) != magic || get(bytes, 4, 2) != version ||
        get(bytes, 6, 2) < 1U || get(bytes, 6, 2) > 10U) {
        reject("invalid frame magic, version, or operation", context);
    }
}

void validate_header(const Header& header, std::size_t max_payload, const ErrorContext& context) {
    if (header.source == header.destination ||
        (header.algorithm != AllReduceAlgorithm::centralized &&
         header.algorithm != AllReduceAlgorithm::ring)) {
        reject("invalid frame ranks or algorithm", context);
    }
    switch (header.operation) {
    case Operation::hello:
    case Operation::edge_hello:
    case Operation::ready:
    case Operation::ring_connect:
    case Operation::edges_ready:
    case Operation::error: {
        if (header.sequence != 0U || header.count != 0U || header.dtype != Dtype::none ||
            header.reduction != Reduction::none || header.phase != Phase::none ||
            header.step != 0U || header.chunk != 0U || header.total_count != 0U ||
            header.element_offset != 0U) {
            reject("invalid control-frame metadata", context);
        }
        const bool identity =
            header.operation == Operation::hello || header.operation == Operation::edge_hello;
        if (identity && (header.payload_bytes < 22U || header.payload_bytes > 404U)) {
            reject("invalid identity frame length", context);
        }
        if ((header.operation == Operation::hello || header.operation == Operation::edges_ready) &&
            header.destination != 0U) {
            reject("invalid root control destination", context);
        }
        if ((header.operation == Operation::ready || header.operation == Operation::ring_connect) &&
            header.source != 0U) {
            reject("invalid root control source", context);
        }
        if ((header.operation == Operation::edge_hello ||
             header.operation == Operation::ring_connect ||
             header.operation == Operation::edges_ready) &&
            header.algorithm != AllReduceAlgorithm::ring) {
            reject("ring control requires ring algorithm", context);
        }
        if (header.operation == Operation::error) {
            if (header.payload_bytes > 1024U) {
                reject("oversized ERROR frame", context);
            }
        } else if (!identity && header.payload_bytes != 0U) {
            reject("unexpected control payload", context);
        }
        return;
    }
    case Operation::all_reduce_request:
    case Operation::all_reduce_result:
    case Operation::ring_reduce:
    case Operation::ring_gather:
        if (header.dtype != Dtype::float32 || header.reduction != Reduction::sum ||
            header.count > std::numeric_limits<std::uint64_t>::max() / 4U ||
            header.payload_bytes != header.count * 4U ||
            header.total_count > std::numeric_limits<std::uint64_t>::max() / 4U ||
            header.total_count * 4U > max_payload || header.element_offset > header.total_count ||
            header.count > header.total_count - header.element_offset) {
            reject("invalid or oversized FP32 SUM frame", context);
        }
        if (header.operation == Operation::all_reduce_request ||
            header.operation == Operation::all_reduce_result) {
            if (header.algorithm != AllReduceAlgorithm::centralized ||
                header.phase != Phase::none || header.step != 0U || header.chunk != 0U ||
                header.element_offset != 0U || header.total_count != header.count) {
                reject("invalid centralized metadata", context);
            }
        } else if (header.algorithm != AllReduceAlgorithm::ring ||
                   header.phase != (header.operation == Operation::ring_reduce
                                        ? Phase::reduce_scatter
                                        : Phase::all_gather)) {
            reject("invalid ring phase or algorithm", context);
        }
        return;
    }
    reject("unknown frame operation", context);
}

std::array<std::byte, header_bytes> encode_header(const Header& header, std::size_t max_payload,
                                                  const ErrorContext& context) {
    validate_header(header, max_payload, context);
    std::array<std::byte, header_bytes> bytes{};
    put(bytes, 0, magic, 4);
    put(bytes, 4, version, 2);
    put(bytes, 6, static_cast<std::uint16_t>(header.operation), 2);
    put(bytes, 8, header.source, 4);
    put(bytes, 12, header.destination, 4);
    put(bytes, 16, header.sequence, 8);
    put(bytes, 24, header.count, 8);
    put(bytes, 32, header.payload_bytes, 8);
    put(bytes, 40, static_cast<std::uint16_t>(header.dtype), 2);
    put(bytes, 42, static_cast<std::uint16_t>(header.reduction), 2);
    put(bytes, 48, static_cast<std::uint16_t>(header.algorithm), 2);
    put(bytes, 50, static_cast<std::uint16_t>(header.phase), 2);
    put(bytes, 52, header.step, 4);
    put(bytes, 56, header.chunk, 4);
    put(bytes, 64, header.total_count, 8);
    put(bytes, 72, header.element_offset, 8);
    return bytes;
}

Header decode_header(std::span<const std::byte> bytes, std::size_t max_payload,
                     const ErrorContext& context) {
    if (bytes.size() != header_bytes) {
        reject("frame header must contain 80 bytes", context);
    }
    validate_prefix(bytes.first(prefix_bytes), context);
    if (get(bytes, 44, 4) != 0U || get(bytes, 60, 4) != 0U) {
        reject("invalid frame magic, version, or reserved bits", context);
    }
    Header header;
    header.operation = static_cast<Operation>(get(bytes, 6, 2));
    header.source = static_cast<std::uint32_t>(get(bytes, 8, 4));
    header.destination = static_cast<std::uint32_t>(get(bytes, 12, 4));
    header.sequence = get(bytes, 16, 8);
    header.count = get(bytes, 24, 8);
    header.payload_bytes = get(bytes, 32, 8);
    header.dtype = static_cast<Dtype>(get(bytes, 40, 2));
    header.reduction = static_cast<Reduction>(get(bytes, 42, 2));
    header.algorithm = static_cast<AllReduceAlgorithm>(get(bytes, 48, 2));
    header.phase = static_cast<Phase>(get(bytes, 50, 2));
    header.step = static_cast<std::uint32_t>(get(bytes, 52, 4));
    header.chunk = static_cast<std::uint32_t>(get(bytes, 56, 4));
    header.total_count = get(bytes, 64, 8);
    header.element_offset = get(bytes, 72, 8);
    validate_header(header, max_payload, context);
    return header;
}

std::vector<std::byte> encode_hello(const RunConfig& config) {
    validate(config);
    std::vector<std::byte> bytes(20U + config.run_id.size() + config.context_fingerprint.size());
    put(bytes, 0, config.world_size, 4);
    put(bytes, 4, config.max_payload_bytes, 8);
    put(bytes, 12, representation, 4);
    put(bytes, 16, config.run_id.size(), 2);
    put(bytes, 18, config.context_fingerprint.size(), 2);
    std::size_t offset = 20;
    for (const auto& text :
         {std::string_view(config.run_id), std::string_view(config.context_fingerprint)}) {
        for (const char character : text) {
            bytes[offset++] = static_cast<std::byte>(static_cast<unsigned char>(character));
        }
    }
    return bytes;
}

Hello decode_hello(std::span<const std::byte> bytes, const ErrorContext& context) {
    if (bytes.size() < 22U || bytes.size() > 404U) {
        reject("invalid HELLO payload length", context);
    }
    const auto run_size = static_cast<std::size_t>(get(bytes, 16, 2));
    const auto context_size = static_cast<std::size_t>(get(bytes, 18, 2));
    if (run_size == 0U || run_size > 128U || context_size == 0U || context_size > 256U ||
        20U + run_size + context_size != bytes.size()) {
        reject("invalid HELLO identifier lengths", context);
    }
    Hello hello;
    hello.world_size = static_cast<std::uint32_t>(get(bytes, 0, 4));
    hello.max_payload_bytes = get(bytes, 4, 8);
    hello.float_representation = static_cast<std::uint32_t>(get(bytes, 12, 4));
    const auto* characters = reinterpret_cast<const char*>(bytes.data());
    hello.run_id.assign(characters + 20, run_size);
    hello.fingerprint.assign(characters + 20 + run_size, context_size);
    if (!printable(hello.run_id) || !printable(hello.fingerprint) || hello.world_size == 0U ||
        hello.max_payload_bytes == 0U || hello.float_representation != representation) {
        reject("invalid HELLO identifiers, group, limit, or float representation", context);
    }
    return hello;
}

} // namespace meshtrain::wire
