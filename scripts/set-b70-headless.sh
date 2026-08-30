#!/usr/bin/env bash
# Configure inference-host to boot without the Plasma login manager using the B70.
set -euo pipefail

LOGIN_SERVICE="${B70_LOGIN_SERVICE:-plasmalogin.service}"
DRM_RELEASE_TIMEOUT_SECONDS="${B70_DRM_RELEASE_TIMEOUT_SECONDS:-15}"

usage() {
    cat <<'EOF'
Usage: sudo set-b70-headless.sh --apply

Sets multi-user.target as the default boot target, then disables and stops the
Plasma login manager so it no longer holds /dev/dri/renderD128. SSH stays
available. To restore a graphical login later, set graphical.target as the
default and enable/start the login service again.

Environment:
  B70_LOGIN_SERVICE  Login-manager systemd service (default: plasmalogin.service)
  B70_DRM_RELEASE_TIMEOUT_SECONDS  Time to wait for the login session to release DRM (default: 15)
EOF
}

[[ $# -eq 1 && "$1" == "--apply" ]] || { usage >&2; exit 2; }
[[ $EUID -eq 0 ]] || { echo "error: --apply must run as root" >&2; exit 1; }

systemctl cat "$LOGIN_SERVICE" >/dev/null 2>&1 || {
    echo "error: login service ${LOGIN_SERVICE} is not installed" >&2
    exit 1
}

wait_for_drm_release() {
    local deadline=$((SECONDS + DRM_RELEASE_TIMEOUT_SECONDS))
    while (( SECONDS < deadline )); do
        if ! fuser -s /dev/dri/renderD* 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

systemctl set-default multi-user.target
systemctl disable --now "$LOGIN_SERVICE"

[[ "$(systemctl get-default)" == "multi-user.target" ]] || {
    echo "error: multi-user.target was not set as the default" >&2
    exit 1
}
if systemctl is-active --quiet "$LOGIN_SERVICE"; then
    echo "error: ${LOGIN_SERVICE} is still active" >&2
    exit 1
fi
if ! wait_for_drm_release; then
    echo "error: a process still holds /dev/dri; inspect it before resetting the B70" >&2
    exit 1
fi

echo "Headless mode enabled: default=$(systemctl get-default), ${LOGIN_SERVICE}=inactive."
echo "The B70 is now available for sudo scripts/reset-b70-gpu.sh --reset."
