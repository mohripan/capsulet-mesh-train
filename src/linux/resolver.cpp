#include "resolver.hpp"

#include "socket.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <pthread.h>

#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>

namespace meshtrain::net {
namespace {

struct Lookup {
    std::string host;
    std::string service;
    addrinfo hints{};
    addrinfo* addresses{};
    int result{};
    std::mutex mutex;
    std::condition_variable complete;
    bool done{};

    ~Lookup() {
        if (addresses != nullptr) {
            ::freeaddrinfo(addresses);
        }
    }
};

void* resolve(void* argument) noexcept {
    // The detached worker owns the request even if its caller times out.
    const std::unique_ptr<std::shared_ptr<Lookup>> owner(
        static_cast<std::shared_ptr<Lookup>*>(argument));
    const auto lookup = *owner;
    const auto result = ::getaddrinfo(lookup->host.c_str(), lookup->service.c_str(), &lookup->hints,
                                      &lookup->addresses);
    {
        const std::lock_guard lock(lookup->mutex);
        lookup->result = result;
        lookup->done = true;
    }
    lookup->complete.notify_all();
    return nullptr;
}

class ThreadAttributes {
  public:
    explicit ThreadAttributes(const ErrorContext& context) {
        const auto result = ::pthread_attr_init(&attributes_);
        if (result != 0) {
            throw EngineError(ErrorCode::transport_failure,
                              std::error_code(result, std::generic_category()).message(), context);
        }
    }
    ~ThreadAttributes() { static_cast<void>(::pthread_attr_destroy(&attributes_)); }
    ThreadAttributes(const ThreadAttributes&) = delete;
    ThreadAttributes& operator=(const ThreadAttributes&) = delete;
    [[nodiscard]] pthread_attr_t* get() noexcept { return &attributes_; }

  private:
    pthread_attr_t attributes_{};
};

void start_lookup(const std::shared_ptr<Lookup>& lookup, const ErrorContext& context) {
    ThreadAttributes attributes(context);
    auto result = ::pthread_attr_setdetachstate(attributes.get(), PTHREAD_CREATE_DETACHED);
    if (result != 0) {
        throw EngineError(ErrorCode::transport_failure,
                          std::error_code(result, std::generic_category()).message(), context);
    }
    auto owner = std::make_unique<std::shared_ptr<Lookup>>(lookup);
    pthread_t thread{};
    // Release before creation: a successful worker can finish before create returns.
    auto* argument = owner.release();
    result = ::pthread_create(&thread, attributes.get(), resolve, argument);
    if (result != 0) {
        owner.reset(argument); // No thread was created, so ownership remains here.
        throw EngineError(ErrorCode::transport_failure,
                          "cannot start name resolution: " +
                              std::error_code(result, std::generic_category()).message(),
                          context);
    }
}

} // namespace

std::vector<sockaddr_in> resolve_ipv4(const Endpoint& endpoint, Deadline deadline,
                                      const ErrorContext& context, bool allow_any) {
    check_deadline(deadline, context);
    sockaddr_in numeric{};
    numeric.sin_family = AF_INET;
    numeric.sin_port = htons(endpoint.port);
    if (::inet_pton(AF_INET, endpoint.host.c_str(), &numeric.sin_addr) == 1) {
        if (!allow_any && numeric.sin_addr.s_addr == htonl(INADDR_ANY)) {
            throw EngineError(ErrorCode::invalid_configuration,
                              "peer resolves to a wildcard address", context);
        }
        return {numeric};
    }

    const auto lookup = std::make_shared<Lookup>();
    lookup->host = endpoint.host;
    lookup->service = std::to_string(endpoint.port);
    lookup->hints.ai_family = AF_INET;
    lookup->hints.ai_socktype = SOCK_STREAM;
    lookup->hints.ai_protocol = IPPROTO_TCP;
    lookup->hints.ai_flags = AI_NUMERICSERV;
    start_lookup(lookup, context);
    std::unique_lock lock(lookup->mutex);
    if (!lookup->complete.wait_until(lock, deadline, [&lookup] { return lookup->done; })) {
        throw EngineError(ErrorCode::timeout, "name resolution deadline expired", context);
    }
    const auto result = lookup->result;
    if (result != 0) {
        throw EngineError(ErrorCode::transport_failure,
                          "name resolution failed: " + std::string(::gai_strerror(result)),
                          context);
    }
    std::vector<sockaddr_in> addresses;
    for (const auto* address = lookup->addresses; address != nullptr; address = address->ai_next) {
        if (address->ai_family == AF_INET && address->ai_addrlen >= sizeof(sockaddr_in)) {
            sockaddr_in copy{};
            std::memcpy(&copy, address->ai_addr, sizeof(copy));
            if (allow_any || copy.sin_addr.s_addr != htonl(INADDR_ANY)) {
                addresses.push_back(copy);
            }
        }
    }
    if (addresses.empty()) {
        throw EngineError(ErrorCode::transport_failure, "name has no usable IPv4 address", context);
    }
    check_deadline(deadline, context);
    return addresses;
}

} // namespace meshtrain::net
