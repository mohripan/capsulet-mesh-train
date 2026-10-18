#include "ring_fixture.hpp"

#include <iostream>

namespace {
void run(const std::string& executable, std::string_view scenario, std::uint32_t world) {
    const auto port = ringtest::ports(world);
    test::Children children(executable);
    const auto started = std::chrono::steady_clock::now();
    for (std::uint32_t rank = 0; rank < world; ++rank) {
        children.start(scenario, rank, world, port);
    }
    children.finish(started +
                    (scenario == "numerical" ? std::chrono::seconds(20) : std::chrono::seconds(5)));
    std::cout << "PASS ring " << scenario << " workers=" << world << " elapsed_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - started)
                     .count()
              << std::endl;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 7 && std::string_view(argv[1]) == "--worker") {
            auto config =
                ringtest::config(static_cast<std::uint32_t>(std::stoul(argv[3])),
                                 static_cast<std::uint32_t>(std::stoul(argv[4])),
                                 static_cast<std::uint16_t>(std::stoul(argv[5])), argv[6], argv[2]);
            if (std::string_view(argv[2]).starts_with("duplex")) {
                return ringtest::duplex(std::move(config), argv[2]);
            }
            if (std::string_view(argv[2]).starts_with("startup")) {
                return ringtest::startup(std::move(config), argv[2]);
            }
            if (std::string_view(argv[2]).starts_with("failure") ||
                std::string_view(argv[2]).starts_with("bad_") ||
                std::string_view(argv[2]).starts_with("gather_")) {
                return ringtest::failure(std::move(config), argv[2]);
            }
            return ringtest::numerical(std::move(config));
        }
        test::require(argc == 1 || (argc == 2 && (std::string_view(argv[1]) == "--transport" ||
                                                  (std::string_view(argv[1]) == "--startup" ||
                                                   std::string_view(argv[1]) == "--failures"))),
                      "unexpected ring test arguments");
        const auto executable = std::filesystem::canonical("/proc/self/exe").string();
        if (argc == 1) {
            for (const auto world : {1U, 2U, 3U, 4U, 5U}) {
                run(executable, "numerical", world);
            }
        } else if (std::string_view(argv[1]) == "--transport") {
            run(executable, "duplex", 2);
            run(executable, "duplex", 3);
            run(executable, "duplex_small", 3);
            for (const auto scenario :
                 {"duplex_cancel", "duplex_timeout", "duplex_drip", "duplex_eof"}) {
                run(executable, scenario, 2);
            }
        } else if (std::string_view(argv[1]) == "--failures") {
            for (const auto field : {"phase", "step", "chunk", "offset", "total", "count",
                                     "algorithm", "dtype", "reduction", "source", "destination",
                                     "sequence", "oversize", "version", "operation", "reserved"}) {
                run(executable, std::string("bad_") + field, 3);
                run(executable, std::string("gather_") + field, 3);
            }
            for (const auto scenario : {"failure_delay", "failure_cumulative", "failure_drip",
                                        "failure_eof", "failure_cancel", "failure_limit",
                                        "failure_tiny", "failure_order", "failure_recreate"}) {
                run(executable, scenario, 3);
            }
        } else {
            run(executable, "startup_v1", 2);
            for (const auto scenario :
                 {"startup_delay", "startup_algorithm", "startup_run", "startup_context",
                  "startup_world", "startup_limit", "startup_wrong_edge", "startup_duplicate_rank",
                  "startup_missing_edge", "startup_missing_ready", "startup_edge_run",
                  "startup_edge_limit", "startup_edge_repr", "startup_edge_version"}) {
                run(executable, scenario, 3);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
