#include <meshtrain/buffer.hpp>
#include <meshtrain/communicator.hpp>
#include <meshtrain/config.hpp>
#include <meshtrain/error.hpp>
#include <meshtrain/tcp.hpp>
#include <meshtrain/transport.hpp>

#include <cstddef>
#include <iostream>
#include <memory>

namespace {
class NoIo final : public meshtrain::ByteTransport {
  public:
    void send_bytes(std::uint32_t, std::span<const std::byte>, meshtrain::Deadline) override {
        throw meshtrain::EngineError(meshtrain::ErrorCode::invalid_state,
                                     "unexpected identity I/O");
    }
    void receive_bytes(std::uint32_t, std::span<std::byte>, meshtrain::Deadline) override {
        throw meshtrain::EngineError(meshtrain::ErrorCode::invalid_state,
                                     "unexpected identity I/O");
    }
    void close() noexcept override {}
};
} // namespace

int main() {
    std::cout << "TCP enabled: " << meshtrain::tcp_enabled() << '\n';
    meshtrain::RunConfig config;
    config.world_size = 1;
    config.run_id = "installed-consumer";
    config.context_fingerprint = "layout";
    config.endpoints = {{"127.0.0.1", 45000}};
    meshtrain::validate(config);
    float storage[]{1.0F, 2.0F};
    const meshtrain::HostFloat32Buffer buffer(storage);
    for (const auto algorithm :
         {meshtrain::AllReduceAlgorithm::centralized, meshtrain::AllReduceAlgorithm::ring}) {
        config.all_reduce_algorithm = algorithm;
        std::unique_ptr<meshtrain::ByteTransport> transport =
            meshtrain::tcp_enabled() ? meshtrain::connect_tcp(config) : std::make_unique<NoIo>();
        meshtrain::Communicator communicator(config, std::move(transport));
        communicator.all_reduce_sum(buffer);
        if (storage[0] != 1.0F || storage[1] != 2.0F || communicator.next_sequence() != 1) {
            return 4;
        }
    }
    if (meshtrain::checked_payload_bytes(buffer.size(), config.max_payload_bytes) != 8U) {
        return 1;
    }
    try {
        static_cast<void>(meshtrain::checked_payload_bytes(buffer.size(), 4U));
    } catch (const meshtrain::EngineError& error) {
        if (error.code() != meshtrain::ErrorCode::invalid_buffer) {
            return 2;
        }
        std::cout << "Installed core configuration, headers, and errors work.\n";
        return 0;
    }
    return 3;
}
