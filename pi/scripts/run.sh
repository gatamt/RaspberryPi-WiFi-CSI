#!/usr/bin/env bash
#
# Convenience launcher for pi_streamer.
# Assumes install.sh has been run.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${PI_DIR}/.venv"

if [[ ! -d "${VENV}" ]]; then
    echo "venv not found at ${VENV} — run scripts/install.sh first"
    exit 1
fi

# shellcheck disable=SC1090
source "${VENV}/bin/activate"
cd "${PI_DIR}"
exec python -m pi_streamer.main "$@"
