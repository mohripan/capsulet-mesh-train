#include "protocol.hpp"
#include "test_support.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <span>

namespace {
using namespace meshtrain;

void header_contracts() {
    wire::Header header;
    header.operation = wire::Operation::all_reduce_request;
    header.source = 1;
    header.sequence = 0x0102030405060708ULL;
    header.count = 3;
    header.payload_bytes = 12;
    header.dtype = wire::Dtype::float32;
    header.reduction = wire::Reduction::sum;
    const std::array<unsigned char, 48> expected{
        0x4d, 0x45, 0x53, 0x48, 0, 1, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0,  1, 2, 3, 4, 5, 6, 7, 8,
        0,    0,    0,    0,    0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 12, 0, 1, 0, 1, 0, 0, 0, 0};
    const auto bytes = wire::encode_header(header, 12);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        test::require(bytes[index] == static_cast<std::byte>(expected[index]),
                      "header differs from network-order golden bytes");
    }
    const auto decoded = wire::decode_header(bytes, 12);
    test::require(decoded.sequence == header.sequence && decoded.count == 3 &&
                      decoded.payload_bytes == 12 && decoded.source == 1,
                  "header fields did not survive decoding");
    for (const auto offset : {0U, 5U, 7U, 31U, 39U, 41U, 43U, 47U}) {
        auto corrupt = bytes;
        corrupt[offset] = std::byte{0xff};
        test::expect_error([&] { static_cast<void>(wire::decode_header(corrupt, 12)); },
                           ErrorCode::protocol_mismatch);
    }
    test::expect_error([&] { static_cast<void>(wire::decode_header(bytes, 11)); },
                       ErrorCode::protocol_mismatch);
    test::expect_error(
        [&] { static_cast<void>(wire::decode_header(std::span(bytes).first(47), 12)); },
        ErrorCode::protocol_mismatch);
    header.count = std::numeric_limits<std::uint64_t>::max();
    header.payload_bytes = header.count - 3U;
    test::expect_error([&] { static_cast<void>(wire::encode_header(header, 12)); },
                       ErrorCode::protocol_mismatch);
    header.count = 0;
    header.payload_bytes = 0;
    static_cast<void>(wire::encode_header(header, 1)); // Empty tensors are legal.
}

void hello_contracts() {
    RunConfig config;
    config.rank = 1;
    config.world_size = 2;
    config.run_id = "run";
    config.context_fingerprint = "layout";
    config.endpoints = {{"localhost", 45000}, {"worker", 45000}};
    config.max_payload_bytes = 1; // Control messages have their own small bounds.
    const auto bytes = wire::encode_hello(config);
    test::require(bytes.size() == 29 && bytes[3] == std::byte{2} && bytes[11] == std::byte{1} &&
                      bytes[14] == std::byte{0} && bytes[15] == std::byte{32} &&
                      bytes[17] == std::byte{3} && bytes[19] == std::byte{6},
                  "HELLO layout is not stable");
    const auto hello = wire::decode_hello(bytes);
    test::require(hello.world_size == 2 && hello.max_payload_bytes == 1 && hello.run_id == "run" &&
                      hello.fingerprint == "layout",
                  "HELLO fields did not survive decoding");
    for (const auto offset : {3U, 11U, 15U, 17U, 19U, 20U}) {
        auto corrupt = bytes;
        corrupt[offset] = std::byte{0};
        test::expect_error([&] { static_cast<void>(wire::decode_hello(corrupt)); },
                           ErrorCode::protocol_mismatch);
    }
    auto extra = bytes;
    extra.push_back(std::byte{});
    test::expect_error([&] { static_cast<void>(wire::decode_hello(extra)); },
                       ErrorCode::protocol_mismatch);
    wire::Header header;
    header.source = 1;
    header.payload_bytes = bytes.size();
    static_cast<void>(wire::encode_header(header, 1));
    for (const auto size : {0ULL, 21ULL, 405ULL, 0xffffffffffffffffULL}) {
        header.payload_bytes = size;
        test::expect_error([&] { static_cast<void>(wire::encode_header(header, 1)); },
                           ErrorCode::protocol_mismatch);
    }
    header.operation = wire::Operation::ready;
    header.source = 0;
    header.destination = 1;
    header.payload_bytes = 0;
    static_cast<void>(wire::encode_header(header, 1));
    header.sequence = 1;
    test::expect_error([&] { static_cast<void>(wire::encode_header(header, 1)); },
                       ErrorCode::protocol_mismatch);
}
} // namespace

int main() {
    try {
        header_contracts();
        hello_contracts();
        std::cout << "PASS wire header and HELLO contracts\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
