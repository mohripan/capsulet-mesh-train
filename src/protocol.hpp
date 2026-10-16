#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <meshtrain/config.hpp>
#include <meshtrain/error.hpp>

namespace meshtrain::wire {

inline constexpr std::size_t header_bytes = 48;
inline constexpr std::uint16_t version = 1;
inline constexpr std::uint32_t representation = 0x00010020U; // little-endian IEEE binary32

enum class Operation : std::uint16_t {
    hello = 1,
    ready = 2,
    all_reduce_request = 3,
    all_reduce_result = 4,
    error = 5
};
enum class Dtype : std::uint16_t { none = 0, float32 = 1 };
enum class Reduction : std::uint16_t { none = 0, sum = 1 };

struct Header {
    Operation operation{Operation::hello};
    std::uint32_t source{};
    std::uint32_t destination{};
    std::uint64_t sequence{};
    std::uint64_t count{};
    std::uint64_t payload_bytes{};
    Dtype dtype{Dtype::none};
    Reduction reduction{Reduction::none};
};

struct Hello {
    std::uint32_t world_size{};
    std::uint64_t max_payload_bytes{};
    std::uint32_t float_representation{};
    std::string run_id;
    std::string fingerprint;
};

void validate_header(const Header& header, std::size_t max_payload,
                     const ErrorContext& context = {});
[[nodiscard]] std::array<std::byte, header_bytes>
encode_header(const Header& header, std::size_t max_payload, const ErrorContext& context = {});
[[nodiscard]] Header decode_header(std::span<const std::byte> bytes, std::size_t max_payload,
                                   const ErrorContext& context = {});
[[nodiscard]] std::vector<std::byte> encode_hello(const RunConfig& config);
[[nodiscard]] Hello decode_hello(std::span<const std::byte> bytes,
                                 const ErrorContext& context = {});

} // namespace meshtrain::wire
