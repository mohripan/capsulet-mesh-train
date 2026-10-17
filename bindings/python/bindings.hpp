#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace meshtrain::python {

void bind_config(pybind11::module_& module);
void bind_communicator(pybind11::module_& module);
struct BufferReleaser {
    void operator()(Py_buffer* buffer) const noexcept;
};

using BufferExport = std::unique_ptr<Py_buffer, BufferReleaser>;
[[nodiscard]] BufferExport borrow_float32(const pybind11::object& object);

} // namespace meshtrain::python
