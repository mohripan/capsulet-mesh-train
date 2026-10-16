#include "collective_fixture.hpp"
#include "process_group.hpp"

#include <filesystem>
#include <iostream>

namespace {
void run(const std::string& executable, std::string_view scenario, std::uint32_t world) {
    const auto port = test::unused_port();
    test::Children children(executable);
    const auto before = std::chrono::steady_clock::now();
    for (std::uint32_t rank = 0; rank < world; ++rank) {
        const bool killed =
            (scenario == "kill_worker" && rank == 1) || (scenario == "kill_root" && rank == 0);
        children.start(scenario, rank, world, port, killed ? SIGKILL : 0);
    }
    children.finish(before + std::chrono::seconds(5));
    std::cout << "PASS " << scenario << " workers=" << world << " group_elapsed_ms="
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - before)
                     .count()
              << std::endl;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 7 && std::string_view(argv[1]) == "--worker") {
            return test::collective_worker(argv[2], static_cast<std::uint32_t>(std::stoul(argv[3])),
                                           static_cast<std::uint32_t>(std::stoul(argv[4])),
                                           static_cast<std::uint16_t>(std::stoul(argv[5])),
                                           argv[6]);
        }
        test::require(argc == 1, "expected --worker CASE RANK WORLD PORT HOST");
        const auto executable = std::filesystem::canonical("/proc/self/exe").string();
        for (const auto world : {1U, 2U, 3U, 4U, 5U}) {
            run(executable, "numerical", world);
        }
        run(executable, "rank_order", 3);
        run(executable, "reconnect", 2);
        for (const auto scenario :
             {"count",           "local_limit",   "stale_sequence",     "future_sequence",
              "bad_dtype",       "bad_reduction", "bad_count_bytes",    "oversize_data",
              "wrong_operation", "wrong_source",  "wrong_destination",  "result_sequence",
              "result_count",    "result_source", "result_destination", "result_dtype",
              "stall_payload",   "slow_payload",  "stalled_result",     "kill_root",
              "cancel"}) {
            run(executable, scenario, 2);
        }
        run(executable, "cumulative_deadline", 3);
        run(executable, "kill_worker", 3);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
