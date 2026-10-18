#include <meshtrain/buffer.hpp>
#include <meshtrain/config.hpp>
#include <meshtrain/error.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

meshtrain::RunConfig group(std::uint32_t world_size = 4U) {
    meshtrain::RunConfig config;
    config.world_size = world_size;
    config.run_id = "test-attempt-1";
    config.context_fingerprint = "test-layout";
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        config.endpoints.push_back({"worker-" + std::to_string(rank) + ".example", 45000});
    }
    return config;
}

template <typename Function>
void expect_error(Function&& operation, meshtrain::ErrorCode code, std::string_view fragment) {
    try {
        std::forward<Function>(operation)();
    } catch (const meshtrain::EngineError& error) {
        require(error.code() == code, "wrong error category");
        require(std::string_view(error.what()).find(fragment) != std::string_view::npos,
                "missing diagnostic detail");
        return;
    }
    throw std::runtime_error("operation unexpectedly succeeded");
}

void valid_groups() {
    for (const auto count : {1U, 2U, 3U, 4U, 5U}) {
        auto config = group(count);
        config.rank = count - 1U;
        meshtrain::validate(config);
    }
}

void invalid_membership() {
    auto config = group(0U);
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "world_size");
    config = group();
    config.rank = config.world_size;
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "rank");
    config = group();
    config.endpoints.pop_back();
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "endpoints");
}

void endpoint_validation() {
    for (const std::string host : {"", ".", "bad host", "bad\nhost", "::1", "bad..name", "-host",
                                   "host-", "0.0.0.0", "0.0.0.0.", "0"}) {
        auto config = group();
        config.endpoints[1].host = host;
        expect_error([&] { meshtrain::validate(config); },
                     meshtrain::ErrorCode::invalid_configuration, "peer endpoint");
    }
    auto config = group();
    config.endpoints[2].port = 0;
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "nonzero port");
    config = group();
    config.endpoints[0].host = std::string(64U, 'a') + ".example";
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "peer endpoint");
    config = group();
    config.listen_address = "invalid address";
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "listen_address");
}

void duplicate_endpoints() {
    auto config = group();
    config.endpoints[0].host = "WORKER.example.";
    config.endpoints[1].host = "worker.example";
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "distinct");
    config.endpoints[1].port = 45001;
    meshtrain::validate(config);
}

void identity_validation() {
    for (const std::string& identity :
         {std::string{}, std::string(129U, 'a'), std::string("bad\nrun"), std::string(1U, '\x7f'),
          std::string(1U, '\xff')}) {
        auto config = group();
        config.run_id = identity;
        expect_error([&] { meshtrain::validate(config); },
                     meshtrain::ErrorCode::invalid_configuration, "run_id");
    }
    auto config = group();
    config.run_id = std::string(128U, 'a');
    config.context_fingerprint = std::string(256U, 'b');
    meshtrain::validate(config);
    config.context_fingerprint += 'b';
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "context_fingerprint");
    config.context_fingerprint.clear();
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "context_fingerprint");
}

void limits_validation() {
    auto config = group();
    config.all_reduce_algorithm = static_cast<meshtrain::AllReduceAlgorithm>(99);
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "all_reduce_algorithm");
    config = group();
    config.startup_timeout = std::chrono::milliseconds{0};
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "timeouts");
    config = group();
    config.collective_timeout = std::chrono::milliseconds{-1};
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "timeouts");
    config = group();
    config.max_payload_bytes = 0;
    expect_error([&] { meshtrain::validate(config); }, meshtrain::ErrorCode::invalid_configuration,
                 "max_payload_bytes");
}

void payload_boundaries() {
    require(meshtrain::checked_payload_bytes(0U, 16U) == 0U, "empty payload rejected");
    require(meshtrain::checked_payload_bytes(4U, 16U) == 16U, "exact payload limit rejected");
    expect_error([] { static_cast<void>(meshtrain::checked_payload_bytes(5U, 16U)); },
                 meshtrain::ErrorCode::invalid_buffer, "exceeds");
    expect_error([] { static_cast<void>(meshtrain::checked_payload_bytes(0U, 0U)); },
                 meshtrain::ErrorCode::invalid_buffer, "positive");
    expect_error(
        [] {
            static_cast<void>(meshtrain::checked_payload_bytes(
                std::numeric_limits<std::size_t>::max() / sizeof(float) + 1U,
                std::numeric_limits<std::size_t>::max()));
        },
        meshtrain::ErrorCode::invalid_buffer, "overflows");
}

void error_context() {
    meshtrain::ErrorContext context;
    context.rank = 0;
    context.run_id = "attempt\n2";
    context.operation = "receive";
    context.sequence = 0;
    context.peer = 0;
    context.peer_endpoint = "worker.example:45000";
    const meshtrain::EngineError error(meshtrain::ErrorCode::timeout, "deadline expired", context);
    const std::string_view diagnostic(error.what());
    require(error.context().run_id == context.run_id, "original context was not retained");
    for (const auto fragment : {"meshtrain/timeout", "rank=0", "sequence=0", "peer=0", "receive",
                                "worker.example:45000", "attempt\\x0A2"}) {
        require(diagnostic.find(fragment) != std::string_view::npos,
                "missing context in diagnostic");
    }
    require(diagnostic.find('\n') == std::string_view::npos, "diagnostic permits log injection");
}

} // namespace

int main() {
    using Test = std::pair<std::string_view, void (*)()>;
    constexpr std::array tests{
        Test{"valid groups", valid_groups},
        Test{"invalid membership", invalid_membership},
        Test{"endpoint validation", endpoint_validation},
        Test{"duplicate endpoints", duplicate_endpoints},
        Test{"identity validation", identity_validation},
        Test{"limits validation", limits_validation},
        Test{"payload boundaries", payload_boundaries},
        Test{"error context", error_context},
    };
    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    }
    return failures == 0 ? 0 : 1;
}
