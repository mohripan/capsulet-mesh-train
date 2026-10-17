#include "bindings.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace py = pybind11;

namespace meshtrain::python {
namespace {

py::ssize_t validate_array(const py::array& array) {
    if (!array.dtype().equal(py::dtype::of<float>())) {
        throw py::type_error("all_reduce_sum requires native-endian float32 dtype");
    }
    if (array.ndim() != 1 || (array.flags() & py::array::c_style) == 0 || !array.writeable()) {
        throw py::value_error(
            "all_reduce_sum requires a writable one-dimensional C-contiguous array");
    }
    const auto count = array.shape(0);
    if (count < 0 || !std::in_range<std::size_t>(count) ||
        static_cast<std::size_t>(count) > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        count > std::numeric_limits<py::ssize_t>::max() / static_cast<py::ssize_t>(sizeof(float))) {
        throw py::value_error("array length is not representable as a native FP32 buffer");
    }
    if (count > 0 && (array.data() == nullptr ||
                      reinterpret_cast<std::uintptr_t>(array.data()) % alignof(float) != 0U)) {
        throw py::value_error("nonempty float32 storage must be aligned for float");
    }

    return count;
}

} // namespace

void BufferReleaser::operator()(Py_buffer* buffer) const noexcept {
    PyBuffer_Release(buffer);
    delete buffer;
}

BufferExport borrow_float32(const py::object& object) {
    if (!py::isinstance<py::array>(object)) {
        throw py::type_error("all_reduce_sum requires a NumPy ndarray; conversion is disabled");
    }
    const auto array = py::reinterpret_borrow<py::array>(object);
    const auto count = validate_array(array);
    auto allocation = std::make_unique<Py_buffer>();
    if (PyObject_GetBuffer(array.ptr(), allocation.get(),
                           PyBUF_STRIDES | PyBUF_FORMAT | PyBUF_WRITABLE) != 0) {
        throw py::error_already_set();
    }
    BufferExport buffer(allocation.release());
    if (validate_array(array) != count || buffer->ndim != 1 || buffer->shape == nullptr ||
        buffer->shape[0] != count || buffer->readonly != 0 || buffer->format == nullptr ||
        buffer->format != py::format_descriptor<float>::format() ||
        buffer->itemsize != static_cast<py::ssize_t>(sizeof(float)) ||
        buffer->buf != array.data() ||
        buffer->len != count * static_cast<py::ssize_t>(sizeof(float)) ||
        PyBuffer_IsContiguous(buffer.get(), 'C') == 0) {
        throw py::value_error("exported buffer does not match the accepted NumPy array");
    }
    return buffer;
}

} // namespace meshtrain::python
