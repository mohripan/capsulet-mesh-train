cmake_minimum_required(VERSION 3.24)

project(meshtrain VERSION 0.1.0 LANGUAGES CXX)

option(MESHTRAIN_BUILD_TESTS "Build the native core tests" ${PROJECT_IS_TOP_LEVEL})
option(MESHTRAIN_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(meshtrain_core STATIC
    src/buffer.cpp
    src/config.cpp
    src/error.cpp
)
add_library(meshtrain::core ALIAS meshtrain_core)
set_target_properties(meshtrain_core PROPERTIES
    EXPORT_NAME core
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_compile_features(meshtrain_core PUBLIC cxx_std_20)
target_include_directories(meshtrain_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

function(meshtrain_project_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
        if(MESHTRAIN_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow
        )
        if(MESHTRAIN_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()
meshtrain_project_warnings(meshtrain_core)

install(TARGETS meshtrain_core EXPORT meshtrainTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/meshtrain DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT meshtrainTargets
    NAMESPACE meshtrain::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshtrain
)
configure_package_config_file(cmake/meshtrainConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/meshtrainConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshtrain
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/meshtrainConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/meshtrainConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/meshtrainConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/meshtrain
)

if(MESHTRAIN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
