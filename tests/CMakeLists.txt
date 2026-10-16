add_executable(meshtrain_core_tests core_tests.cpp)
set_target_properties(meshtrain_core_tests PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(meshtrain_core_tests PRIVATE meshtrain::core)
meshtrain_project_warnings(meshtrain_core_tests)
add_test(NAME core_contracts COMMAND meshtrain_core_tests)
set_tests_properties(core_contracts PROPERTIES TIMEOUT 10)

add_test(NAME installed_consumer
    COMMAND ${CMAKE_COMMAND}
        -DMESHTRAIN_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DMESHTRAIN_BINARY_DIR=${PROJECT_BINARY_DIR}
        -DMESHTRAIN_TEST_GENERATOR=${CMAKE_GENERATOR}
        -DMESHTRAIN_TEST_COMPILER=${CMAKE_CXX_COMPILER}
        -DMESHTRAIN_TEST_CONFIG=$<CONFIG>
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_consumer.cmake
)
set_tests_properties(installed_consumer PROPERTIES TIMEOUT 90)
