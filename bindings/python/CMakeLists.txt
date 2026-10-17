find_package(Python 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.1.0 EXACT CONFIG REQUIRED)

pybind11_add_module(_native MODULE module.cpp config.cpp communicator.cpp buffer.cpp)
target_link_libraries(_native PRIVATE meshtrain::core)
target_compile_definitions(_native PRIVATE MESHTRAIN_VERSION="${PROJECT_VERSION}")
set_target_properties(_native PROPERTIES CXX_EXTENSIONS OFF)
meshtrain_project_warnings(_native)
install(TARGETS _native LIBRARY DESTINATION meshtrain COMPONENT python)
