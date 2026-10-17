add_executable(meshtrain_core_tests core_tests.cpp)
set_target_properties(meshtrain_core_tests PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(meshtrain_core_tests PRIVATE meshtrain::core)
meshtrain_project_warnings(meshtrain_core_tests)
add_test(NAME core_contracts COMMAND meshtrain_core_tests)
set_tests_properties(core_contracts PROPERTIES TIMEOUT 10)

add_executable(meshtrain_protocol_tests protocol_tests.cpp)
set_target_properties(meshtrain_protocol_tests PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(meshtrain_protocol_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(meshtrain_protocol_tests PRIVATE meshtrain::core)
meshtrain_project_warnings(meshtrain_protocol_tests)
add_test(NAME wire_protocol COMMAND meshtrain_protocol_tests)
set_tests_properties(wire_protocol PROPERTIES TIMEOUT 10)

add_executable(meshtrain_communicator_tests communicator_tests.cpp)
set_target_properties(meshtrain_communicator_tests PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(meshtrain_communicator_tests PRIVATE meshtrain::core)
meshtrain_project_warnings(meshtrain_communicator_tests)
add_test(NAME communicator_contracts COMMAND meshtrain_communicator_tests)
set_tests_properties(communicator_contracts PROPERTIES TIMEOUT 10)

if(MESHTRAIN_ENABLE_TCP)
    add_executable(meshtrain_collective_tests collective_tests.cpp collective_workers.cpp
        collective_failures.cpp)
    set_target_properties(meshtrain_collective_tests PROPERTIES CXX_EXTENSIONS OFF)
    target_include_directories(meshtrain_collective_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(meshtrain_collective_tests PRIVATE meshtrain::core Threads::Threads)
    meshtrain_project_warnings(meshtrain_collective_tests)
    add_test(NAME centralized_collectives COMMAND meshtrain_collective_tests)
    set_tests_properties(centralized_collectives PROPERTIES TIMEOUT 90)
    add_executable(meshtrain_tcp_tests tcp_tests.cpp tcp_workers.cpp)
    set_target_properties(meshtrain_tcp_tests PROPERTIES CXX_EXTENSIONS OFF)
    target_include_directories(meshtrain_tcp_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(meshtrain_tcp_tests PRIVATE meshtrain::core Threads::Threads)
    meshtrain_project_warnings(meshtrain_tcp_tests)
    add_test(NAME tcp_processes COMMAND meshtrain_tcp_tests)
    set_tests_properties(tcp_processes PROPERTIES TIMEOUT 60)
    add_executable(meshtrain_resolver_tests resolver_tests.cpp)
    set_target_properties(meshtrain_resolver_tests PROPERTIES CXX_EXTENSIONS OFF)
    target_include_directories(meshtrain_resolver_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(meshtrain_resolver_tests PRIVATE meshtrain::core Threads::Threads)
    target_link_options(meshtrain_resolver_tests PRIVATE
        -Wl,--wrap=getaddrinfo -Wl,--wrap=freeaddrinfo)
    meshtrain_project_warnings(meshtrain_resolver_tests)
    add_test(NAME resolver_deadline COMMAND meshtrain_resolver_tests)
    set_tests_properties(resolver_deadline PROPERTIES TIMEOUT 10)

    add_executable(meshtrain_failure_tests failure_tests.cpp failure_worker.cpp)
    set_target_properties(meshtrain_failure_tests PROPERTIES CXX_EXTENSIONS OFF)
    target_include_directories(meshtrain_failure_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(meshtrain_failure_tests PRIVATE meshtrain::core Threads::Threads)
    target_link_options(meshtrain_failure_tests PRIVATE -Wl,--wrap=listen -Wl,--wrap=send)
    meshtrain_project_warnings(meshtrain_failure_tests)
    foreach(phase startup collective)
        foreach(target root worker)
            foreach(action kill stop)
                set(fault_case "${phase}-${target}-${action}")
                add_test(NAME "fault_${fault_case}" COMMAND meshtrain_failure_tests --case "${fault_case}")
                set_tests_properties("fault_${fault_case}" PROPERTIES TIMEOUT 5 LABELS m1_integration RUN_SERIAL TRUE)
            endforeach()
        endforeach()
    endforeach()
    add_test(NAME fault_repeated_cleanup COMMAND meshtrain_failure_tests --repeat)
    set_tests_properties(fault_repeated_cleanup PROPERTIES TIMEOUT 110 LABELS m1_integration RUN_SERIAL TRUE)
endif()

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
