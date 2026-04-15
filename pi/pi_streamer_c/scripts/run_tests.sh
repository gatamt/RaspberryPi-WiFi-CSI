#!/usr/bin/env bash
# Exhaustive test runner. Builds pi_streamer_c three ways and runs ctest
# on each:
#   1. build-san       : ASAN + UBSAN enabled, Debug -O1
#   2. build-valgrind  : no sanitizers, Debug -O1 (valgrind can't coexist with ASAN)
#   3. (optional) coverage : gcov/lcov line coverage report
#
# Run on the Pi or on host — both work. Host build uses mocks.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"

# ---------- 1. ASAN + UBSAN ----------
SAN_DIR="${HERE}/build-san"
echo "[run_tests] ASAN+UBSAN configure"
cmake \
    -S "${HERE}" \
    -B "${SAN_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPI_ENABLE_ASAN=ON \
    -DPI_ENABLE_UBSAN=ON \
    -DPI_ENABLE_WERROR=ON

echo "[run_tests] ASAN+UBSAN build + ctest"
cmake --build "${SAN_DIR}" --parallel
ctest --test-dir "${SAN_DIR}" --output-on-failure

# ---------- 2. valgrind ----------
VG_DIR="${HERE}/build-valgrind"
echo "[run_tests] valgrind configure"
cmake \
    -S "${HERE}" \
    -B "${VG_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPI_ENABLE_ASAN=OFF \
    -DPI_ENABLE_UBSAN=OFF \
    -DPI_ENABLE_WERROR=ON \
    -DCMAKE_C_FLAGS_DEBUG="-g -O1"
cmake --build "${VG_DIR}" --parallel

if command -v valgrind >/dev/null 2>&1; then
    echo "[run_tests] valgrind sweep"
    while IFS= read -r -d '' test_bin; do
        echo "[valgrind] ${test_bin}"
        valgrind \
            --leak-check=full \
            --show-leak-kinds=all \
            --error-exitcode=1 \
            --track-origins=yes \
            --quiet \
            "${test_bin}"
    done < <(find "${VG_DIR}/tests/unit" -type f -perm -u+x -name 'test_*' -print0)
else
    echo "[run_tests] valgrind not installed — skipping valgrind sweep"
fi

echo "[run_tests] all green."
