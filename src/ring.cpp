#include "ring.hpp"

#include "protocol.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace meshtrain::detail {

Chunk ring_chunk(std::size_t count, std::uint32_t world, std::uint32_t chunk) {
    if (world == 0U || chunk >= world) {
        throw EngineError(ErrorCode::invalid_configuration, "invalid ring partition index");
    }
    const auto quotient = count / world;
    const auto remainder = count % world;
    return {chunk * quotient + std::min<std::size_t>(chunk, remainder),
            quotient + (chunk < remainder ? 1U : 0U)};
}

namespace {
wire::Header frame(wire::Phase phase, std::uint32_t step, std::uint32_t chunk, std::uint32_t source,
                   std::uint32_t destination, std::size_t total, std::uint32_t world,
                   std::uint64_t sequence) {
    const auto range = ring_chunk(total, world, chunk);
    wire::Header result;
    result.operation = phase == wire::Phase::reduce_scatter ? wire::Operation::ring_reduce
                                                            : wire::Operation::ring_gather;
    result.source = source;
    result.destination = destination;
    result.sequence = sequence;
    result.count = range.count;
    result.payload_bytes = range.count * sizeof(float);
    result.dtype = wire::Dtype::float32;
    result.reduction = wire::Reduction::sum;
    result.algorithm = AllReduceAlgorithm::ring;
    result.phase = phase;
    result.step = step;
    result.chunk = chunk;
    result.total_count = total;
    result.element_offset = range.offset;
    return result;
}
} // namespace

void ring_all_reduce(const RunConfig& config, ByteTransport& transport, HostFloat32Buffer buffer,
                     Deadline deadline, ErrorContext& context, const std::function<void()>& check) {
    const auto world = config.world_size;
    const auto rank = static_cast<std::uint64_t>(config.rank);
    const auto index = [world](std::uint64_t value) {
        return static_cast<std::uint32_t>(value % world);
    };
    const auto successor = index(rank + 1U);
    const auto predecessor = index(rank + world - 1U);
    context.peer = predecessor;
    context.peer_endpoint = config.endpoints[predecessor].host + ':' +
                            std::to_string(config.endpoints[predecessor].port);
    std::vector<float> scratch(ring_chunk(buffer.size(), world, 0).count);
    for (const auto phase : {wire::Phase::reduce_scatter, wire::Phase::all_gather}) {
        for (std::uint32_t step = 0; step < world - 1U; ++step) {
            const auto outgoing =
                index(rank + world + (phase == wire::Phase::all_gather ? 1U : 0U) - step);
            const auto incoming =
                index(rank + world - (phase == wire::Phase::reduce_scatter ? 1U : 0U) - step);
            const auto prefix =
                std::string("all_reduce_sum.ring.") +
                (phase == wire::Phase::reduce_scatter ? "reduce_scatter." : "all_gather.") +
                std::to_string(step);
            context.operation = prefix + ".header";
            check();
            const auto sent = frame(phase, step, outgoing, config.rank, successor, buffer.size(),
                                    world, *context.sequence);
            const auto encoded = wire::encode_header(sent, config.max_payload_bytes, context);
            std::array<std::byte, wire::header_bytes> received{};
            transport.exchange_bytes(successor, std::span(encoded).first(wire::prefix_bytes),
                                     predecessor, std::span(received).first(wire::prefix_bytes),
                                     deadline);
            wire::validate_prefix(std::span(received).first(wire::prefix_bytes), context);
            check();
            transport.exchange_bytes(successor, std::span(encoded).subspan(wire::prefix_bytes),
                                     predecessor, std::span(received).subspan(wire::prefix_bytes),
                                     deadline);
            const auto expected = frame(phase, step, incoming, predecessor, config.rank,
                                        buffer.size(), world, *context.sequence);
            if (wire::decode_header(received, config.max_payload_bytes, context) != expected) {
                throw EngineError(ErrorCode::protocol_mismatch, "ring step metadata mismatch",
                                  context);
            }
            const auto send_range = ring_chunk(buffer.size(), world, outgoing);
            const auto receive_range = ring_chunk(buffer.size(), world, incoming);
            const auto send_buffer = buffer.subspan(send_range.offset, send_range.count);
            const auto local = buffer.subspan(receive_range.offset, receive_range.count);
            const auto receive_buffer = phase == wire::Phase::reduce_scatter
                                            ? HostFloat32Buffer(scratch).first(receive_range.count)
                                            : local;
            context.operation = prefix + ".payload";
            check();
            transport.exchange_bytes(successor, std::as_bytes(send_buffer), predecessor,
                                     std::as_writable_bytes(receive_buffer), deadline);
            if (phase == wire::Phase::reduce_scatter) {
                context.operation = prefix + ".reduce";
                for (std::size_t offset = 0; offset < local.size();) {
                    check();
                    const auto end = offset + std::min<std::size_t>(4096, local.size() - offset);
                    for (; offset < end; ++offset) {
                        local[offset] += scratch[offset];
                    }
                }
            }
        }
    }
    check();
}

} // namespace meshtrain::detail
