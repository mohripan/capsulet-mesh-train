#include "bindings.hpp"

#include <meshtrain/config.hpp>

#include <pybind11/stl.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace meshtrain::python {
namespace {

template <typename Integer> Integer integer(py::handle value, const char* name) {
    if (PyBool_Check(value.ptr()) || !PyLong_Check(value.ptr())) {
        throw py::type_error(std::string(name) + " must be an integer, not a boolean");
    }
    const auto extracted = [&] {
        if constexpr (std::is_signed_v<Integer>) {
            return PyLong_AsLongLong(value.ptr());
        } else {
            return PyLong_AsUnsignedLongLong(value.ptr());
        }
    }();
    if (PyErr_Occurred() != nullptr) {
        throw py::error_already_set();
    }
    if (!std::in_range<Integer>(extracted)) {
        throw std::overflow_error(std::string(name) + " is outside the native integer range");
    }
    return static_cast<Integer>(extracted);
}

RunConfig make_config(py::object rank, py::object world_size, std::string run_id,
                      std::string context_fingerprint, std::vector<Endpoint> endpoints,
                      std::string listen_address, py::object startup_ms, py::object collective_ms,
                      py::object max_payload_bytes) {
    RunConfig config;
    config.rank = integer<std::uint32_t>(rank, "rank");
    config.world_size = integer<std::uint32_t>(world_size, "world_size");
    config.run_id = std::move(run_id);
    config.context_fingerprint = std::move(context_fingerprint);
    config.endpoints = std::move(endpoints);
    config.listen_address = std::move(listen_address);
    config.startup_timeout = std::chrono::milliseconds(
        integer<std::chrono::milliseconds::rep>(startup_ms, "startup_timeout_ms"));
    config.collective_timeout = std::chrono::milliseconds(
        integer<std::chrono::milliseconds::rep>(collective_ms, "collective_timeout_ms"));
    config.max_payload_bytes = integer<std::size_t>(max_payload_bytes, "max_payload_bytes");
    validate(config);
    return config;
}

} // namespace

void bind_config(py::module_& module) {
    py::class_<Endpoint>(module, "Endpoint")
        .def(py::init([](std::string host, py::object port) {
                 return Endpoint{std::move(host), integer<std::uint16_t>(port, "port")};
             }),
             py::arg("host").noconvert(), py::arg("port"))
        .def_readonly("host", &Endpoint::host)
        .def_readonly("port", &Endpoint::port);

    py::class_<RunConfig>(module, "RunConfig")
        .def(py::init(&make_config), py::kw_only(), py::arg("rank"), py::arg("world_size"),
             py::arg("run_id").noconvert(), py::arg("context_fingerprint").noconvert(),
             py::arg("endpoints"), py::arg("listen_address").noconvert() = "0.0.0.0",
             py::arg("startup_timeout_ms") = 30'000, py::arg("collective_timeout_ms") = 30'000,
             py::arg("max_payload_bytes") = 256U * 1024U * 1024U)
        .def_readonly("rank", &RunConfig::rank)
        .def_readonly("world_size", &RunConfig::world_size)
        .def_readonly("run_id", &RunConfig::run_id)
        .def_readonly("context_fingerprint", &RunConfig::context_fingerprint)
        .def_readonly("listen_address", &RunConfig::listen_address)
        .def_readonly("max_payload_bytes", &RunConfig::max_payload_bytes)
        .def_property_readonly("endpoints",
                               [](const RunConfig& config) { return config.endpoints; })
        .def_property_readonly(
            "startup_timeout_ms",
            [](const RunConfig& config) { return config.startup_timeout.count(); })
        .def_property_readonly("collective_timeout_ms", [](const RunConfig& config) {
            return config.collective_timeout.count();
        });
}

} // namespace meshtrain::python
