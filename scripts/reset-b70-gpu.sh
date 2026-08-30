#!/usr/bin/env bash
# Request a PCI function reset for the Intel Arc Pro B70 after its fan controller
# becomes stuck. Run on inference-host, only while no workload is using the GPU.
set -euo pipefail

BDF="${B70_PCI_BDF:-0000:0b:00.0}"
TIMEOUT_SECONDS="${B70_RESET_TIMEOUT_SECONDS:-30}"
DEVICE_DIR="/sys/bus/pci/devices/${BDF}"
RESET_PATH="${DEVICE_DIR}/reset"

usage() {
    cat <<'EOF'
Usage: reset-b70-gpu.sh --status | --reset

--status  Print the B70's current Xe driver, power mode, fan RPM, and temperatures.
--reset   Require root, refuse if a process is using /dev/dri, then request the
          kernel's default PCI reset method for the B70. Wait for the xe driver
          and DRM render node to return, then print the post-reset status.

Environment:
  B70_PCI_BDF               PCI BDF (default: 0000:0b:00.0)
  B70_RESET_TIMEOUT_SECONDS Reset recovery timeout in seconds (default: 30)

This does not set a fan speed. The Xe hwmon interface exposes only fan RPM on
this host. A successful PCI reset is not proof that a physically faulty fan is
fixed.
EOF
}

fail() {
    echo "error: $*" >&2
    exit 1
}

require_b70() {
    [[ -d "$DEVICE_DIR" ]] || fail "PCI device ${BDF} is not present"
    [[ "$(<"${DEVICE_DIR}/vendor")" == "0x8086" ]] || fail "${BDF} is not an Intel device"
    [[ "$(<"${DEVICE_DIR}/device")" == "0xe223" ]] || fail "${BDF} is not an Arc Pro B70"
}

hwmon_dir() {
    local candidate
    for candidate in "${DEVICE_DIR}"/hwmon/hwmon*; do
        [[ -r "${candidate}/name" ]] || continue
        [[ "$(<"${candidate}/name")" == "xe" ]] || continue
        printf '%s\n' "$candidate"
        return 0
    done
    return 1
}

print_status() {
    local driver power hwmon path label value
    driver="$(readlink -f "${DEVICE_DIR}/driver" 2>/dev/null || true)"
    power="$(<"${DEVICE_DIR}/power/control")"
    echo "B70 ${BDF}"
    echo "driver: ${driver:-unbound}"
    echo "power/control: ${power}"
    echo "reset methods: $(<"${DEVICE_DIR}/reset_method")"

    if ! hwmon="$(hwmon_dir)"; then
        echo "xe hwmon: unavailable"
        return
    fi

    if [[ -r "${hwmon}/fan1_input" ]]; then
        echo "fan RPM: $(<"${hwmon}/fan1_input")"
    fi
    for path in "${hwmon}"/temp*_input; do
        [[ -r "$path" ]] || continue
        label="${path%_input}_label"
        value="$(<"$path")"
        printf 'temperature %-16s %s m°C\n' "$(cat "$label" 2>/dev/null || basename "${path%_input}")" "$value"
    done
}

wait_for_rebind() {
    local deadline=$((SECONDS + TIMEOUT_SECONDS))
    while (( SECONDS < deadline )); do
        if [[ -L "${DEVICE_DIR}/driver" ]] && [[ -e /dev/dri/renderD128 ]]; then
            return 0
        fi
        sleep 1
    done
    return 1
}

ensure_unused() {
    command -v fuser >/dev/null 2>&1 || fail "fuser is required to verify that the GPU is unused"
    if fuser -s /dev/dri/renderD* 2>/dev/null; then
        fail "a process is using /dev/dri; stop GPU workloads before resetting ${BDF}"
    fi
}

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
require_b70

case "$1" in
    --status)
        print_status
        ;;
    --reset)
        [[ $EUID -eq 0 ]] || fail "--reset must be run as root (for example: sudo $0 --reset)"
        [[ -w "$RESET_PATH" ]] || fail "${RESET_PATH} is not writable; this kernel does not support a PCI reset here"
        ensure_unused
        echo "Requesting the kernel default PCI reset for ${BDF}..."
        printf '1\n' >"$RESET_PATH"
        wait_for_rebind || fail "the xe driver or /dev/dri/renderD128 did not return within ${TIMEOUT_SECONDS}s"
        echo "PCI reset completed. Verify that the fan RPM returns to normal after the card cools."
        print_status
        ;;
    --help|-h)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
