#include "bindings.hpp"

#include <meshtrain/error.hpp>

#include <pybind11/stl.h>

#include <exception>
#include <string>

namespace py = pybind11;

namespace {

void translate_engine_error(std::exception_ptr pending) {
    try {
        if (pending) {
            std::rethrow_exception(pending);
        }
    } catch (const meshtrain::EngineError& error) {
        const auto type = py::module_::import("meshtrain._native").attr("EngineError");
        const auto instance = type(error.what());
        const auto& context = error.context();
        instance.attr("code") = std::string(meshtrain::to_string(error.code()));
        instance.attr("rank") = py::cast(context.rank);
        instance.attr("run_id") = context.run_id;
        instance.attr("operation") = context.operation;
        instance.attr("sequence") = py::cast(context.sequence);
        instance.attr("peer") = py::cast(context.peer);
        instance.attr("peer_endpoint") = context.peer_endpoint;
        py::set_error(type, instance);
    }
}

} // namespace

PYBIND11_MODULE(_native, module) {
    module.attr("__version__") = MESHTRAIN_VERSION;
    py::exception<meshtrain::EngineError>(module, "EngineError", PyExc_RuntimeError);
    py::register_local_exception_translator(translate_engine_error);
    meshtrain::python::bind_config(module);
    meshtrain::python::bind_communicator(module);
}
