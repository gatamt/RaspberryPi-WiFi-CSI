# Unit tests for pi_streamer_c. Each test is a standalone executable linked
# against pi_streamer_core + unity. Register with CTest so `ctest` runs them
# all from any build dir.

function(pi_add_unit_test name)
    add_executable(${name} ${name}.c)
    target_link_libraries(${name}
        PRIVATE
            pi_streamer_core_mock
            unity
    )
    target_include_directories(${name}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/include
    )
    # Keep Unity's vendored code from flagging our -Werror.
    target_compile_options(${name} PRIVATE -Wno-error)
    # Enable test-only hooks in the mock backends (e.g. RX injection for
    # pipeline control-message tests). Mock sources guard their hook
    # bodies behind PI_ENABLE_TEST_HOOKS.
    target_compile_definitions(${name} PRIVATE PI_ENABLE_TEST_HOOKS=1)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
        TIMEOUT 30
        ENVIRONMENT "UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1"
    )
endfunction()

pi_add_unit_test(test_logger)
pi_add_unit_test(test_ring_buffer)
pi_add_unit_test(test_wire_format)
pi_add_unit_test(test_state_machine)
pi_add_unit_test(test_camera)
pi_add_unit_test(test_encoder)
pi_add_unit_test(test_udp_sender)
pi_add_unit_test(test_inference)
pi_add_unit_test(test_inference_state)
pi_add_unit_test(test_overlay)
pi_add_unit_test(test_postprocess)
pi_add_unit_test(test_pipeline)
pi_add_unit_test(test_rt)
