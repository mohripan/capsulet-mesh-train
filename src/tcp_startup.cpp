#include "tcp_startup.hpp"

#include "protocol.hpp"

#include <array>
#include <utility>

namespace meshtrain::net {

ErrorContext peer_context(const RunConfig& config, std::uint32_t peer, std::string operation) {
    ErrorContext result;
    result.rank = config.rank;
    result.run_id = config.run_id;
    result.peer = peer;
    result.operation = std::move(operation);
    if (peer < config.endpoints.size()) {
        result.peer_endpoint =
            config.endpoints[peer].host + ':' + std::to_string(config.endpoints[peer].port);
    }
    return result;
}

namespace {
wire::Header read_header(const Socket& socket, const RunConfig& config, Deadline deadline,
                         const ErrorContext& context) {
    std::array<std::byte, wire::header_bytes> bytes{};
    receive_exact(socket, std::span(bytes).first(wire::prefix_bytes), deadline, context);
    wire::validate_prefix(std::span(bytes).first(wire::prefix_bytes), context);
    receive_exact(socket, std::span(bytes).subspan(wire::prefix_bytes), deadline, context);
    return wire::decode_header(bytes, config.max_payload_bytes, context);
}

wire::Header control(const RunConfig& config, wire::Operation operation,
                     std::uint32_t destination) {
    wire::Header header;
    header.operation = operation;
    header.source = config.rank;
    header.destination = destination;
    header.algorithm = config.all_reduce_algorithm;
    return header;
}

void send_control(const Socket& socket, const RunConfig& config, wire::Operation operation,
                  std::uint32_t destination, Deadline deadline) {
    const auto context = peer_context(config, destination, "startup.send_control");
    const auto encoded = wire::encode_header(control(config, operation, destination),
                                             config.max_payload_bytes, context);
    send_exact(socket, encoded, deadline, context);
}

void expect_control(const Socket& socket, const RunConfig& config, wire::Operation operation,
                    std::uint32_t peer, Deadline deadline) {
    const auto context = peer_context(config, peer, "startup.receive_control");
    const auto actual = read_header(socket, config, deadline, context);
    auto expected = control(config, operation, config.rank);
    expected.source = peer;
    if (actual != expected) {
        throw EngineError(ErrorCode::protocol_mismatch, "unexpected startup control", context);
    }
}

void validate_identity(const wire::Header& header, std::span<const std::byte> payload,
                       const RunConfig& config, const ErrorContext& context) {
    const auto hello = wire::decode_hello(payload, context);
    if (hello.world_size != config.world_size || hello.run_id != config.run_id ||
        hello.fingerprint != config.context_fingerprint ||
        hello.max_payload_bytes != config.max_payload_bytes ||
        header.algorithm != config.all_reduce_algorithm) {
        throw EngineError(ErrorCode::protocol_mismatch,
                          "HELLO group, run, context, limit, or algorithm mismatch", context);
    }
}

SocketTable join_control(const RunConfig& config, const TcpOptions& options, const Socket& listener,
                         Deadline deadline) {
    SocketTable peers(config.world_size);
    if (config.rank == 0U) {
        for (std::uint32_t joined = 1; joined < config.world_size; ++joined) {
            auto context = peer_context(config, 0, "startup.accept_hello");
            context.peer.reset();
            auto socket = std::make_shared<Socket>(accept(listener, options, deadline, context));
            const auto header = read_header(*socket, config, deadline, context);
            context = peer_context(config, header.source, "startup.receive_hello");
            if (header.operation != wire::Operation::hello || header.source >= config.world_size ||
                peers[header.source] != nullptr ||
                header.algorithm != config.all_reduce_algorithm) {
                throw EngineError(ErrorCode::protocol_mismatch,
                                  "expected HELLO from a distinct matching rank", context);
            }
            std::array<std::byte, 404> payload{};
            const auto bytes =
                std::span(payload).first(static_cast<std::size_t>(header.payload_bytes));
            receive_exact(*socket, bytes, deadline, context);
            validate_identity(header, bytes, config, context);
            peers[header.source] = std::move(socket);
        }
    } else {
        const auto context = peer_context(config, 0, "startup.connect_root");
        auto socket =
            std::make_shared<Socket>(connect(config.endpoints[0], options, deadline, context));
        const auto payload = wire::encode_hello(config);
        auto header = control(config, wire::Operation::hello, 0);
        header.payload_bytes = payload.size();
        const auto bytes = wire::encode_header(header, config.max_payload_bytes, context);
        send_exact(*socket, bytes, deadline, context);
        send_exact(*socket, payload, deadline, context);
        peers[0] = std::move(socket);
    }
    return peers;
}

SocketTable join_edges(const RunConfig& config, const TcpOptions& options, const Socket& listener,
                       Deadline deadline) {
    SocketTable peers(config.world_size);
    const auto successor = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(config.rank) + 1U) % config.world_size);
    const auto predecessor = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(config.rank) + config.world_size - 1U) % config.world_size);
    const auto sending = peer_context(config, successor, "startup.connect_successor");
    const auto receiving = peer_context(config, predecessor, "startup.accept_predecessor");
    if (config.world_size > 2U || config.rank == 0U) {
        peers[successor] = std::make_shared<Socket>(
            connect(config.endpoints[successor], options, deadline, sending));
    }
    if (config.world_size > 2U || config.rank == 1U) {
        peers[predecessor] =
            std::make_shared<Socket>(accept(listener, options, deadline, receiving));
    }
    auto header = control(config, wire::Operation::edge_hello, successor);
    const auto payload = wire::encode_hello(config);
    header.payload_bytes = payload.size();
    const auto encoded = wire::encode_header(header, config.max_payload_bytes, sending);
    std::array<std::byte, wire::header_bytes> received{};
    exchange_exact(*peers[successor], std::span(encoded).first(wire::prefix_bytes),
                   *peers[predecessor], std::span(received).first(wire::prefix_bytes), deadline,
                   sending, receiving);
    wire::validate_prefix(std::span(received).first(wire::prefix_bytes), receiving);
    exchange_exact(*peers[successor], std::span(encoded).subspan(wire::prefix_bytes),
                   *peers[predecessor], std::span(received).subspan(wire::prefix_bytes), deadline,
                   sending, receiving);
    const auto hello = wire::decode_header(received, config.max_payload_bytes, receiving);
    if (hello.operation != wire::Operation::edge_hello || hello.source != predecessor ||
        hello.destination != config.rank || hello.algorithm != config.all_reduce_algorithm) {
        throw EngineError(ErrorCode::protocol_mismatch, "unexpected ring edge identity", receiving);
    }
    std::array<std::byte, 404> identity{};
    const auto incoming = std::span(identity).first(static_cast<std::size_t>(hello.payload_bytes));
    exchange_exact(*peers[successor], payload, *peers[predecessor], incoming, deadline, sending,
                   receiving);
    validate_identity(hello, incoming, config, receiving);
    return peers;
}
} // namespace

JoinedGroup join_tcp(const RunConfig& config, const TcpOptions& options, Deadline deadline) {
    JoinedGroup group{SocketTable(config.world_size), SocketTable(config.world_size)};
    if (config.world_size == 1U) {
        return group;
    }
    const bool ring = config.all_reduce_algorithm == AllReduceAlgorithm::ring;
    Socket listener;
    if (ring || config.rank == 0U) {
        auto context = peer_context(config, config.rank, "startup.listen");
        context.peer.reset();
        context.peer_endpoint =
            config.listen_address + ':' + std::to_string(config.endpoints[config.rank].port);
        listener = listen(config, options, deadline, context);
    }
    group.control = join_control(config, options, listener, deadline);
    const auto release = ring ? wire::Operation::ring_connect : wire::Operation::ready;
    if (config.rank == 0U) {
        for (std::uint32_t peer = 1; peer < config.world_size; ++peer) {
            send_control(*group.control[peer], config, release, peer, deadline);
        }
    } else {
        expect_control(*group.control[0], config, release, 0, deadline);
    }
    if (!ring) {
        group.data = std::move(group.control);
        return group;
    }
    group.data = join_edges(config, options, listener, deadline);
    if (config.rank == 0U) {
        for (std::uint32_t peer = 1; peer < config.world_size; ++peer) {
            expect_control(*group.control[peer], config, wire::Operation::edges_ready, peer,
                           deadline);
        }
        for (std::uint32_t peer = 1; peer < config.world_size; ++peer) {
            send_control(*group.control[peer], config, wire::Operation::ready, peer, deadline);
        }
    } else {
        send_control(*group.control[0], config, wire::Operation::edges_ready, 0, deadline);
        expect_control(*group.control[0], config, wire::Operation::ready, 0, deadline);
    }
    check_deadline(deadline, peer_context(config, config.rank, "startup.complete"));
    return group;
}

} // namespace meshtrain::net
