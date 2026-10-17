#include "bindings.hpp"

#include <meshtrain/communicator.hpp>
#include <meshtrain/tcp.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace py = pybind11;

namespace meshtrain::python {
namespace {

void all_reduce_sum(std::shared_ptr<Communicator> communicator, py::object array) {
    const auto buffer = borrow_float32(array);
    const HostFloat32Buffer span(static_cast<float*>(buffer->buf),
                                 static_cast<std::size_t>(buffer->shape[0]));
    {
        const py::gil_scoped_release release;
        communicator->all_reduce_sum(span);
    }
}

void close(std::shared_ptr<Communicator> communicator) {
    const py::gil_scoped_release release;
    communicator->close();
}

const char* state(const Communicator& communicator) {
    switch (communicator.state()) {
    case CommunicatorState::ready:
        return "ready";
    case CommunicatorState::failed:
        return "failed";
    case CommunicatorState::closed:
        return "closed";
    }
    throw std::logic_error("unknown communicator state");
}

} // namespace

void bind_communicator(py::module_& module) {
    py::class_<Communicator, std::shared_ptr<Communicator>>(module, "Communicator")
        .def(py::init([](RunConfig config) {
                 const py::gil_scoped_release release;
                 auto transport = connect_tcp(config);
                 return std::make_shared<Communicator>(std::move(config), std::move(transport));
             }),
             py::arg("config").noconvert())
        .def("all_reduce_sum", &all_reduce_sum, py::arg("array"))
        .def("close", &close)
        .def_property_readonly("state", &state)
        .def_property_readonly("next_sequence", &Communicator::next_sequence)
        .def("__enter__", [](std::shared_ptr<Communicator> communicator) { return communicator; })
        .def("__exit__", [](std::shared_ptr<Communicator> communicator, py::object, py::object,
                            py::object) { close(std::move(communicator)); });
}

} // namespace meshtrain::python
