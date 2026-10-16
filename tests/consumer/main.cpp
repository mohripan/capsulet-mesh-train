#include <meshtrain/buffer.hpp>
#include <meshtrain/communicator.hpp>
#include <meshtrain/config.hpp>
#include <meshtrain/error.hpp>
#include <meshtrain/tcp.hpp>
#include <meshtrain/transport.hpp>

#include <cstddef>
#include <iostream>

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
    if (meshtrain::tcp_enabled()) {
        meshtrain::Communicator communicator(config, meshtrain::connect_tcp(config));
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
