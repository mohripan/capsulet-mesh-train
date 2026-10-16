foreach(required MESHTRAIN_SOURCE_DIR MESHTRAIN_BINARY_DIR MESHTRAIN_TEST_GENERATOR MESHTRAIN_TEST_COMPILER)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Missing ${required}")
    endif()
endforeach()

string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef attempt)
set(smoke_root "${MESHTRAIN_BINARY_DIR}/install-smoke/${attempt}")
cmake_path(IS_PREFIX MESHTRAIN_BINARY_DIR "${smoke_root}" NORMALIZE within_build)
if(NOT within_build)
    message(FATAL_ERROR "Install test paths must stay inside the build directory")
endif()
file(MAKE_DIRECTORY "${smoke_root}")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${MESHTRAIN_BINARY_DIR}"
    --prefix "${smoke_root}/original" --config "${MESHTRAIN_TEST_CONFIG}"
    COMMAND_ERROR_IS_FATAL ANY)
file(RENAME "${smoke_root}/original" "${smoke_root}/relocated")
execute_process(COMMAND "${CMAKE_COMMAND}"
    -S "${MESHTRAIN_SOURCE_DIR}/tests/consumer" -B "${smoke_root}/consumer"
    -G "${MESHTRAIN_TEST_GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${smoke_root}/relocated"
    "-DMESHTRAIN_EXPECTED_PREFIX=${smoke_root}/relocated"
    "-DCMAKE_CXX_COMPILER=${MESHTRAIN_TEST_COMPILER}"
    "-DCMAKE_BUILD_TYPE=${MESHTRAIN_TEST_CONFIG}"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${smoke_root}/consumer"
    --config "${MESHTRAIN_TEST_CONFIG}" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${smoke_root}/consumer"
    -C "${MESHTRAIN_TEST_CONFIG}" --output-on-failure COMMAND_ERROR_IS_FATAL ANY)
