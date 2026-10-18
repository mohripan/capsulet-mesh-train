#include "failure_control.hpp"

#include "process_group.hpp"
#include "ring_fixture.hpp"

#include <spawn.h>

#include <cstdlib>
#include <vector>

namespace fault {
namespace {
constexpr std::array cases{"startup-root-kill",      "startup-root-stop",
                           "startup-worker-kill",    "startup-worker-stop",
                           "collective-root-kill",   "collective-root-stop",
                           "collective-worker-kill", "collective-worker-stop"};

constexpr std::array ring_cases{
    "ring-startup-root-kill",   "ring-startup-root-stop",  "ring-startup-worker-kill",
    "ring-startup-worker-stop", "ring-reduce-root-kill",   "ring-reduce-root-stop",
    "ring-reduce-worker-kill",  "ring-reduce-worker-stop", "ring-gather-root-kill",
    "ring-gather-root-stop",    "ring-gather-worker-kill", "ring-gather-worker-stop"};

struct Child {
    pid_t pid{-1};
    Pipe phase;
    Pipe control;
    Pipe result;
    Result report;
    std::size_t received{};
    bool reported{};
    bool marked{};
    bool reaped{};
    int status{};
};

class Group {
  public:
    Group() { children.reserve(3); }
    ~Group() {
        for (auto& child : children) {
            if (child.pid > 0 && !child.reaped) {
                static_cast<void>(::kill(child.pid, SIGKILL));
                while (::waitpid(child.pid, nullptr, 0) < 0 && errno == EINTR) {
                }
            }
        }
    }
    std::vector<Child> children;
};

void nonblocking(int descriptor) {
    const auto flags = ::fcntl(descriptor, F_GETFL);
    require(flags >= 0 && ::fcntl(descriptor, F_SETFL, flags | O_NONBLOCK) == 0,
            "cannot make control pipe nonblocking");
}

void spawn(Group& group, const std::string& executable, const Settings& scenario, unsigned rank,
           std::uint16_t port, const std::string& run, const std::filesystem::path& output) {
    group.children.emplace_back();
    auto& child = group.children.back();
    posix_spawn_file_actions_t actions;
    require(::posix_spawn_file_actions_init(&actions) == 0, "cannot initialize spawn actions");
    try {
        std::array<File, 3> sources;
        const std::array descriptors{child.phase.write.get(), child.control.read.get(),
                                     child.result.write.get()};
        for (std::size_t index = 0; index < descriptors.size(); ++index) {
            sources[index].reset(::fcntl(descriptors[index], F_DUPFD_CLOEXEC, 20));
            require(sources[index].get() >= 0, "cannot duplicate child pipe");
            require(::posix_spawn_file_actions_adddup2(&actions, sources[index].get(),
                                                       static_cast<int>(index) + 3) == 0,
                    "cannot map child control descriptors");
        }
        for (const auto stream : {STDOUT_FILENO, STDERR_FILENO}) {
            const auto name = std::to_string(rank) + (stream == STDOUT_FILENO ? ".out" : ".err");
            require(::posix_spawn_file_actions_addopen(&actions, stream, (output / name).c_str(),
                                                       O_WRONLY | O_CREAT | O_TRUNC, 0600) == 0,
                    "cannot open rank log");
        }
        std::array<std::string, 8> arguments{executable,
                                             "--worker",
                                             scenario.name,
                                             std::to_string(rank),
                                             std::to_string(port),
                                             run,
                                             scenario.omit_phase ? "omit" : "normal",
                                             ""};
        std::array<char*, 8> raw{};
        for (std::size_t index = 0; index + 1 < raw.size(); ++index) {
            raw[index] = arguments[index].data();
        }
        pid_t spawned = -1;
        require(::posix_spawn(&spawned, executable.c_str(), &actions, nullptr, raw.data(),
                              environ) == 0,
                "cannot spawn fault worker");
        child.pid = spawned;
    } catch (...) {
        ::posix_spawn_file_actions_destroy(&actions);
        throw;
    }
    ::posix_spawn_file_actions_destroy(&actions);
    child.phase.write.reset();
    child.control.read.reset();
    child.result.write.reset();
    nonblocking(child.phase.read.get());
    nonblocking(child.result.read.get());
}

void wait_markers(Group& group, const Settings& scenario, Clock::time_point deadline) {
    const auto expected = scenario.startup ? (scenario.target == 0 ? 'L' : 'H') : 'R';
    while (Clock::now() < deadline) {
        bool ready = true;
        for (unsigned rank = 0; rank < group.children.size(); ++rank) {
            if (scenario.startup && rank != scenario.target) {
                continue;
            }
            auto& child = group.children[rank];
            char marker{};
            const auto received = ::read(child.phase.read.get(), &marker, 1);
            if (received == 1) {
                const auto wanted =
                    scenario.startup || rank == 0 ? expected : (rank == 1 ? 'S' : 'T');
                require(marker == wanted, "wrong fault phase marker");
                child.marked = true;
            }
            ready = ready && child.marked;
        }
        if (ready) {
            return;
        }
        std::this_thread::sleep_for(1ms);
    }
    throw std::runtime_error("required native phase marker missing");
}

std::string worker_json(const Child& child, unsigned rank, unsigned target) {
    std::ostringstream output;
    output << "{\"rank\":" << rank << ",\"pid\":" << child.pid
           << ",\"reaped\":true,\"raw_wait_status\":" << child.status;
    if (rank == target) {
        output << ",\"cleanup\":\"OS_reclamation\",\"signal\":" << WTERMSIG(child.status);
    } else {
        const auto& report = child.report;
        output << ",\"cleanup\":\"verified_while_alive\",\"fds_before\":" << report.before
               << ",\"fds_after\":" << report.after
               << ",\"code\":" << quote(meshtrain::to_string(report.code))
               << ",\"operation\":" << quote(report.operation.data())
               << ",\"endpoint\":" << quote(report.endpoint.data())
               << ",\"terminal\":" << (report.terminal ? "true" : "false")
               << ",\"sequence\":" << (report.has_sequence ? "0" : "null")
               << ",\"exit_code\":" << WEXITSTATUS(child.status);
    }
    return output.str() + '}';
}
} // namespace

void run_case(const std::string& executable, Settings scenario, unsigned iteration,
              const std::filesystem::path& output, Watchdog& watchdog) {
    const auto directory = output / (scenario.name + '-' + std::to_string(iteration));
    std::filesystem::create_directories(directory);
    const auto baseline = descriptors();
    const auto port = scenario.ring ? ringtest::ports(3) : test::unused_port();
    const auto run =
        scenario.name + '-' + std::to_string(::getpid()) + '-' + std::to_string(iteration);
    std::string reports;
    watchdog.begin();
    const auto started = Clock::now();
    std::size_t stopped_sockets = 0;
    try {
        {
            Group group;
            for (unsigned rank = 0; rank < 3; ++rank) {
                spawn(group, executable, scenario, rank, port, run, directory);
            }
            wait_markers(group, scenario, started + 800ms);
            auto& target = group.children[scenario.target];
            require(::kill(target.pid, scenario.stop ? SIGSTOP : SIGKILL) == 0,
                    "fault signal failed");
            if (scenario.stop) {
                bool stopped = false;
                while (Clock::now() < started + 1s) {
                    int status{};
                    if (::waitpid(target.pid, &status, WNOHANG | WUNTRACED) == target.pid) {
                        stopped = WIFSTOPPED(status) && WSTOPSIG(status) == SIGSTOP;
                        break;
                    }
                    std::this_thread::sleep_for(1ms);
                }
                require(stopped, "target did not actually stop");
                for (const auto& entry : std::filesystem::directory_iterator(
                         "/proc/" + std::to_string(target.pid) + "/fd")) {
                    if (std::filesystem::read_symlink(entry.path())
                            .string()
                            .starts_with("socket:")) {
                        ++stopped_sockets;
                    }
                }
                require(stopped_sockets > 0, "stopped victim did not retain a real socket");
            }
            if (!scenario.startup) {
                for (unsigned rank = 0; rank < 3; ++rank) {
                    if (rank != scenario.target) {
                        send(group.children[rank].control.write.get(), 'G');
                    }
                }
            }
            bool all_reported = false;
            bool saw_timeout = false;
            while (!all_reported && Clock::now() < started + 4s) {
                all_reported = true;
                for (unsigned rank = 0; rank < 3; ++rank) {
                    if (rank == scenario.target) {
                        continue;
                    }
                    auto& child = group.children[rank];
                    if (!child.reported) {
                        auto* bytes = reinterpret_cast<std::byte*>(&child.report);
                        const auto count = ::read(child.result.read.get(), bytes + child.received,
                                                  sizeof(Result) - child.received);
                        if (count > 0) {
                            child.received += static_cast<std::size_t>(count);
                        }
                        if (child.received == sizeof(Result)) {
                            require(child.report.valid && child.report.rank == rank &&
                                        child.report.before == child.report.after &&
                                        ::kill(child.pid, 0) == 0,
                                    "survivor cleanup was not verified while alive");
                            saw_timeout =
                                saw_timeout || child.report.code == meshtrain::ErrorCode::timeout;
                            child.reported = true;
                            send(child.control.write.get(), 'E');
                        }
                    }
                    all_reported = all_reported && child.reported;
                }
                std::this_thread::sleep_for(1ms);
            }
            require(all_reported && (!scenario.stop || saw_timeout),
                    "survivors did not detect native failure before cleanup");
            if (scenario.stop) {
                require(::kill(target.pid, SIGKILL) == 0, "cannot reclaim stopped target");
            }
            for (unsigned rank = 0; rank < 3; ++rank) {
                auto& child = group.children[rank];
                while (Clock::now() < started + 4s) {
                    if (::waitpid(child.pid, &child.status, WNOHANG) == child.pid) {
                        child.reaped = true;
                        break;
                    }
                    std::this_thread::sleep_for(1ms);
                }
                require(child.reaped, "owned worker not reaped");
                require(rank == scenario.target
                            ? WIFSIGNALED(child.status) && WTERMSIG(child.status) == SIGKILL
                            : WIFEXITED(child.status) && WEXITSTATUS(child.status) == 1,
                        "unexpected native worker exit status");
                reports += (rank == 0 ? "" : ",") + worker_json(child, rank, scenario.target);
            }
        }
        {
            for (std::uint32_t rank = 0; rank < (scenario.ring ? 3U : 1U); ++rank) {
                auto config = scenario.ring ? ringtest::config(rank, 3, port, "127.0.0.1", "rebind")
                                            : test::tcp_group(0, 3, port, "127.0.0.1", "rebind");
                const auto listener = meshtrain::net::listen(config, {}, Clock::now() + 100ms, {});
            }
        }
        require(descriptors() == baseline, "controller leaked descriptors");
        const auto elapsed =
            std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        require(elapsed < 5000, "case exceeded five-second overall bound");
        std::ostringstream result;
        result << "{\"schema\":1,\"status\":\"passed\",\"case\":" << quote(scenario.name)
               << ",\"run_id\":" << quote(run) << ",\"operation_timeout_ms\":1000"
               << ",\"case_elapsed_ms\":" << elapsed << ",\"group_exit_code\":1"
               << ",\"stopped_confirmed\":" << (scenario.stop ? "true" : "false")
               << ",\"stopped_socket_count\":" << stopped_sockets
               << ",\"controller_fds_before\":" << baseline
               << ",\"controller_fds_after\":" << descriptors()
               << ",\"listener_reused\":true,\"workers\":[" << reports << "]}";
        save(directory / "result.json", result.str());
        require(Clock::now() - started < 5s, "evidence publication exceeded case bound");
        watchdog.end();
        std::cout << result.str() << std::endl;
    } catch (const std::exception& error) {
        watchdog.end();
        std::ostringstream failed;
        failed << "{\"status\":\"failed\",\"error\":" << quote(error.what())
               << ",\"case_elapsed_ms\":"
               << std::chrono::duration<double, std::milli>(Clock::now() - started).count()
               << ",\"controller_fds_before\":" << baseline
               << ",\"controller_fds_after\":" << descriptors() << '}';
        save(directory / "result.json", failed.str());
        throw;
    }
}
} // namespace fault

int main(int argc, char** argv) {
    try {
        if (argc == 7 && std::string_view(argv[1]) == "--worker") {
            auto scenario = fault::settings(argv[2]);
            scenario.omit_phase = std::string_view(argv[6]) == "omit";
            return fault::worker(scenario, static_cast<std::uint32_t>(std::stoul(argv[3])),
                                 static_cast<std::uint16_t>(std::stoul(argv[4])), argv[5]);
        }
        fault::require(argc >= 2, "expected --case NAME or --repeat");
        const auto executable = std::filesystem::canonical("/proc/self/exe").string();
        const auto mode = std::string_view(argv[1]);
        const bool ring =
            mode == "--ring-repeat" ||
            (mode == "--case" && argc >= 3 && std::string_view(argv[2]).starts_with("ring-"));
        auto output =
            std::filesystem::current_path() / (ring ? "ring-fault-evidence" : "fault-evidence");
        if (const auto* environment =
                std::getenv(ring ? "MESHTRAIN_RING_FAULT_OUTPUT" : "MESHTRAIN_FAULT_OUTPUT")) {
            output = environment;
        }
        fault::Watchdog watchdog;
        if (mode == "--watchdog-expiry") {
            watchdog.begin();
            std::this_thread::sleep_for(std::chrono::seconds(6));
            throw std::runtime_error("independent watchdog failed to fire");
        }
        if (mode == "--repeat" || mode == "--ring-repeat") {
            for (unsigned iteration = 0; iteration < 20; ++iteration) {
                fault::run_case(
                    executable,
                    fault::settings(mode == "--ring-repeat"
                                        ? fault::ring_cases[iteration % fault::ring_cases.size()]
                                        : fault::cases[iteration % fault::cases.size()]),
                    iteration + 1, output / "repeat", watchdog);
            }
        } else {
            fault::require(mode == "--case" && argc >= 3, "expected --case NAME");
            auto scenario = fault::settings(argv[2]);
            scenario.omit_phase = argc == 4 && std::string_view(argv[3]) == "--omit-phase";
            fault::run_case(executable, scenario, 0, output, watchdog);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << std::endl;
        return 2;
    }
}
