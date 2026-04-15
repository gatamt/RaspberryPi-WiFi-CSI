#!/usr/bin/env bash
# Configure + build pi_streamer_c in Release mode and run unit tests.
#
# Usage:
#   scripts/build.sh              # host build (TDD / mocks)
#   PI_TARGET=1 scripts/build.sh  # Pi-native build (Cortex-A76 flags)
#
# The script auto-detects aarch64 and enables PI_TARGET by default in that
# case. On Mac / x86_64 Linux it defaults to host mode.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${HERE}/build"

CMAKE_ARGS=(
    -S "${HERE}"
    -B "${BUILD_DIR}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

# Auto-enable PI_TARGET on aarch64 unless the env var explicitly disables it.
if [[ "${PI_TARGET:-}" == "1" ]] || { [[ -z "${PI_TARGET:-}" ]] && [[ "$(uname -m)" == "aarch64" ]]; }; then
    CMAKE_ARGS+=(-DPI_TARGET=ON)
    echo "[build] PI_TARGET=ON"
else
    CMAKE_ARGS+=(-DPI_TARGET=OFF)
    echo "[build] PI_TARGET=OFF (host build)"
fi

echo "[build] configure"
cmake "${CMAKE_ARGS[@]}"

echo "[build] compile"
cmake --build "${BUILD_DIR}" --parallel

echo "[build] ctest"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo "[build] compile_commands.json -> ${BUILD_DIR}/compile_commands.json"
echo "[build] done."
