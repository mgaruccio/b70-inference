#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
mkdir -p "$HERE/worker-scratch"
SCRATCH=$(mktemp -d "$HERE/worker-scratch/cpu-XXXXXX")
trap 'rm -rf "$SCRATCH"' EXIT
export BOLT_SINGLETONS_W3_CPU_SCRATCH="$SCRATCH"
python3 -B "$HERE/test-timing.py"
