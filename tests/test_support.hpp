#pragma once

#include <meshtrain/error.hpp>

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace test {

inline void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Function> void expect_error(Function&& operation, meshtrain::ErrorCode code) {
    try {
        std::forward<Function>(operation)();
    } catch (const meshtrain::EngineError& error) {
        require(error.code() == code, std::string("wrong error: ") + error.what());
        return;
    }
    throw std::runtime_error("operation unexpectedly succeeded");
}

} // namespace test
