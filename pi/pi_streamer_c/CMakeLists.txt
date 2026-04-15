cmake_minimum_required(VERSION 3.22)
project(pi_streamer_c
    VERSION 0.1.0
    DESCRIPTION "Pi C11 low-level video streamer"
    LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Standards
# -----------------------------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)     # _GNU_SOURCE-style Linux extensions allowed

set(CMAKE_CXX_STANDARD 17)     # libcamera's C++ API requires C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# Target flags
# -----------------------------------------------------------------------------
# PI_TARGET=ON emits Pi 5 Cortex-A76 specific flags. OFF keeps the build host-
# portable (Mac arm64 / Linux x86_64) for TDD cycles.
option(PI_TARGET "Enable Cortex-A76 / ARMv8.2-A specific compile flags" OFF)
option(PI_ENABLE_WERROR "Treat warnings as errors" ON)

# Per-module real-backend toggles (T0). Default ON when PI_TARGET is set AND
# the corresponding library is found by pkg_check_modules / find_package
# further down. Each can be explicitly flipped OFF to fall back to the mock
# even on Pi (useful for bisecting backend bugs).
option(PI_USE_REAL_CAMERA    "Use libcamera backend (requires PI_TARGET)" ${PI_TARGET})
option(PI_USE_REAL_ENCODER   "Use libx264 backend (requires PI_TARGET)"   ${PI_TARGET})
option(PI_USE_REAL_UDP       "Use io_uring backend (requires PI_TARGET)"  ${PI_TARGET})
option(PI_USE_REAL_INFERENCE "Use HailoRT backend (requires PI_TARGET)"   ${PI_TARGET})

if(PI_TARGET)
    add_compile_options(
        -mcpu=cortex-a76
        -mtune=cortex-a76
        -march=armv8.2-a+crc+simd
    )
    # NB: PI_TARGET=1 is NOT set globally here. It is applied per-target
    # below so `pi_streamer_core_mock` (the test-only library) compiles
    # without PI_TARGET and therefore without the Pi threading path,
    # keeping `ctest` deterministic even on a native Pi build.
    message(STATUS "pi_streamer_c: PI_TARGET build (Cortex-A76, ARMv8.2-A)")
else()
    message(STATUS "pi_streamer_c: host build (TDD / mock)")
endif()

add_compile_options(
    -O3
    -ftree-vectorize
    -Wall
    -Wextra
    -Wshadow
    -Wdouble-promotion
    -Wformat=2
    -Wmissing-prototypes
    -Wstrict-prototypes
    -Wpointer-arith
    -Wcast-qual
    -Wunused
    -fstack-protector-strong
    -fno-common
    -fvisibility=hidden
)

if(PI_ENABLE_WERROR)
    add_compile_options(-Werror)
endif()

# Sanitizer convenience flags (used by scripts/run_tests.sh)
option(PI_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(PI_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

if(PI_ENABLE_ASAN OR PI_ENABLE_UBSAN)
    set(_san_list "")
    if(PI_ENABLE_ASAN)
        list(APPEND _san_list "address")
    endif()
    if(PI_ENABLE_UBSAN)
        list(APPEND _san_list "undefined")
    endif()
    string(JOIN "," _san_joined ${_san_list})
    add_compile_options(-fsanitize=${_san_joined} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${_san_joined})
    message(STATUS "pi_streamer_c: sanitizers enabled: ${_san_joined}")
endif()

# -----------------------------------------------------------------------------
# Pi-side optional dependencies (libcamera / libx264 / liburing / HailoRT)
# -----------------------------------------------------------------------------
# These are not required to build — the host TDD path uses mocks and skips
# them. The flags below detect each library so we can fail early with a
# helpful message if someone runs build.sh on a host that doesn't have them.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(LIBCAMERA QUIET IMPORTED_TARGET libcamera)
    pkg_check_modules(LIBX264   QUIET IMPORTED_TARGET x264)
    pkg_check_modules(LIBURING  QUIET IMPORTED_TARGET liburing)
endif()
find_package(HailoRT CONFIG QUIET)

# -----------------------------------------------------------------------------
# Unity test framework via FetchContent
# -----------------------------------------------------------------------------
# Reference: https://github.com/ThrowTheSwitch/Unity  (v2.6.0 tag)
# We vendor via FetchContent so offline builds are OK after the first fetch.
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(
    unity
    GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
    GIT_TAG        v2.6.0
    GIT_SHALLOW    TRUE
)
# Unity's CMake emits -Werror which conflicts with ours when the vendored
# copy warns; disable Werror just for the subdirectory.
set(UNITY_EXTENSION_FIXTURE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(unity)

# -----------------------------------------------------------------------------
# Public include directory
# -----------------------------------------------------------------------------
include_directories(include)

# -----------------------------------------------------------------------------
# Core library
# -----------------------------------------------------------------------------
# Per-module real-backend selection. The mock source stays in the build
# when the real backend is disabled or the library is not found, so the
# host TDD build is always complete and self-contained.
set(PI_CORE_SOURCES
    src/logger.c
    src/ring_buffer.c
    src/wire_format.c
    src/state_machine.c
    # RT helpers (mlockall + CPU affinity + SCHED_FIFO + thread naming).
    # Compiles on host as a no-op so TDD stays green; real behavior is
    # gated behind PI_TARGET in src/rt.c.
    src/rt.c
    # Seqlock-published inference shared state. Host-portable so the
    # test_inference_state unit test can exercise it directly.
    src/inference_state.c
    # YUV420 Y-plane bbox overlay. Pure C, host-portable, covered by
    # test_overlay.
    src/overlay.c
    # YOLOv8 pose postprocess (DFL + NMS + letterbox reverse). Host-
    # portable, covered by test_postprocess.
    src/postprocess.c
    # Main event loop wiring (camera → inference + encoder → wire_format
    # → UDP sender). Single-threaded run-to-completion is the ground
    # truth; thread pinning + SCHED_FIFO are layered on optionally.
    src/pipeline.c
)

if(PI_USE_REAL_CAMERA AND LIBCAMERA_FOUND)
    list(APPEND PI_CORE_SOURCES src/camera_libcamera.cpp)
    message(STATUS "pi_streamer_core: camera backend = libcamera (real)")
else()
    list(APPEND PI_CORE_SOURCES src/camera_mock.c)
    message(STATUS "pi_streamer_core: camera backend = mock")
endif()

if(PI_USE_REAL_ENCODER AND LIBX264_FOUND)
    list(APPEND PI_CORE_SOURCES src/encoder_x264.c)
    message(STATUS "pi_streamer_core: encoder backend = libx264 (real)")
else()
    list(APPEND PI_CORE_SOURCES src/encoder_mock.c)
    message(STATUS "pi_streamer_core: encoder backend = mock")
endif()

if(PI_USE_REAL_UDP AND LIBURING_FOUND)
    list(APPEND PI_CORE_SOURCES src/udp_sender_iouring.c)
    message(STATUS "pi_streamer_core: udp backend = io_uring (real)")
else()
    list(APPEND PI_CORE_SOURCES src/udp_sender_mock.c)
    message(STATUS "pi_streamer_core: udp backend = mock")
endif()

if(PI_USE_REAL_INFERENCE AND HailoRT_FOUND)
    # HailoRT 5.1.1 on Hailo-10H exposes the async InferModel API only via
    # C++ headers (infer_model.hpp). We therefore build a thin C++ shim that
    # exports extern "C" functions matching include/pi_streamer/inference.h.
    list(APPEND PI_CORE_SOURCES src/inference_hailort.cpp)
    message(STATUS "pi_streamer_core: inference backend = HailoRT (real, C++ shim)")
else()
    list(APPEND PI_CORE_SOURCES src/inference_mock.c)
    message(STATUS "pi_streamer_core: inference backend = mock")
endif()

add_library(pi_streamer_core STATIC ${PI_CORE_SOURCES})

# Production executable — thin wrapper that calls pi_pipeline_run.
add_executable(pi_streamer_c_bin src/main.c)
set_target_properties(pi_streamer_c_bin PROPERTIES OUTPUT_NAME pi_streamer_c)
target_link_libraries(pi_streamer_c_bin PRIVATE pi_streamer_core)
target_include_directories(pi_streamer_core PUBLIC include)
target_compile_definitions(pi_streamer_core PRIVATE _GNU_SOURCE)
# PI_TARGET lives on the production lib + bin only. pi_streamer_core_mock
# stays PI_TARGET-less so the unit tests always exercise the deterministic
# single-threaded host loop even on a native aarch64 build.
if(PI_TARGET)
    target_compile_definitions(pi_streamer_core  PRIVATE PI_TARGET=1)
    target_compile_definitions(pi_streamer_c_bin PRIVATE PI_TARGET=1)
endif()
set_property(TARGET pi_streamer_core PROPERTY C_VISIBILITY_PRESET hidden)
# pthread is always required. librt is only needed on Linux (older glibc
# routes clock_gettime through it). On macOS/BSD, clock_gettime lives in libc.
# libm is needed on Linux for expf/sinf/etc used by src/postprocess.c.
# On macOS libm symbols live in libSystem so no explicit link is required.
target_link_libraries(pi_streamer_core PUBLIC pthread)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(pi_streamer_core PUBLIC rt m)
endif()

# Real-backend link guards (each only fires when the corresponding option
# is ON and the library was actually located).
if(PI_USE_REAL_CAMERA AND LIBCAMERA_FOUND)
    target_link_libraries(pi_streamer_core PRIVATE PkgConfig::LIBCAMERA)
endif()
if(PI_USE_REAL_ENCODER AND LIBX264_FOUND)
    target_link_libraries(pi_streamer_core PRIVATE PkgConfig::LIBX264)
endif()
if(PI_USE_REAL_UDP AND LIBURING_FOUND)
    target_link_libraries(pi_streamer_core PRIVATE PkgConfig::LIBURING)
endif()
if(PI_USE_REAL_INFERENCE AND HailoRT_FOUND)
    target_link_libraries(pi_streamer_core PRIVATE HailoRT::libhailort)
endif()

# -----------------------------------------------------------------------------
# Mock-only library: unconditional mock backends for test targets.
# -----------------------------------------------------------------------------
# Tests link against this target instead of pi_streamer_core so they are
# immune to PI_USE_REAL_* flag combinations. This lets us exercise the real
# backends on Pi without breaking the TDD loop.
add_library(pi_streamer_core_mock STATIC
    src/logger.c
    src/ring_buffer.c
    src/wire_format.c
    src/state_machine.c
    src/rt.c
    src/inference_state.c
    src/overlay.c
    src/postprocess.c
    src/pipeline.c
    src/camera_mock.c
    src/encoder_mock.c
    src/udp_sender_mock.c
    src/inference_mock.c
)
target_include_directories(pi_streamer_core_mock PUBLIC include)
target_compile_definitions(pi_streamer_core_mock PRIVATE _GNU_SOURCE)
set_property(TARGET pi_streamer_core_mock PROPERTY C_VISIBILITY_PRESET hidden)
target_link_libraries(pi_streamer_core_mock PUBLIC pthread)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(pi_streamer_core_mock PUBLIC rt m)
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests/unit)
