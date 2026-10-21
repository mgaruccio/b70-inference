#!/usr/bin/env bash
set -euo pipefail

# This is the only remote launcher.  It intentionally uses an explicit IPv6
# endpoint while pinning the SSH host-key alias used by inference-host.
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SSH_HOST='mike@fd7a:115c:a1e0::c933:4f38'
REMOTE='mike@[fd7a:115c:a1e0::c933:4f38]'
HOST_ALIAS='inference-host'
REMOTE_ROOT='/home/mike/b70-evals/20261021-lilicorr-serving-profile'
RESULT_DIR='capture-r1'

SSH_OPTIONS=(
  -6
  -T
  -o BatchMode=yes
  -o StrictHostKeyChecking=yes
  -o HostKeyAlias="${HOST_ALIAS}"
  -o ControlMaster=no
  -o ControlPath=none
  -o ForwardAgent=no
  -o ForwardX11=no
  -o ClearAllForwardings=yes
  -o PermitLocalCommand=no
  -o RequestTTY=no
)
SSH=(ssh "${SSH_OPTIONS[@]}" "${SSH_HOST}")
RSYNC_SSH="ssh -6 -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o HostKeyAlias=${HOST_ALIAS} -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no -o ForwardX11=no -o ClearAllForwardings=yes -o PermitLocalCommand=no -o RequestTTY=no"
FILES=(README.md run-profile.py test-profile.py lead-run.sh .gitignore analyze-trace.py test-analyze-trace.py kernel-rules.json)

"${SSH[@]}" "mkdir -p -- '${REMOTE_ROOT}'"
for file in "${FILES[@]}"; do
  rsync -a --protect-args -e "${RSYNC_SSH}" \
    "${HERE}/${file}" "${REMOTE}:${REMOTE_ROOT}/${file}"
done

capture_rc=0
"${SSH[@]}" "cd -- '${REMOTE_ROOT}' && python3 -B run-profile.py --lead-capture >capture.stdout.log 2>capture.stderr.log" || capture_rc=$?
printf '%s\n' "$capture_rc" >"${HERE}/capture.exit-code"

# This is deliberately a new SSH/Python process and runs even when capture or
# its remote connection failed.  The capture process already has its own
# finally restoration; this is the independent second restoration check.
verify_rc=0
"${SSH[@]}" "cd -- '${REMOTE_ROOT}' && python3 -B run-profile.py --lead-verify-restored >restore.stdout.log 2>restore.stderr.log" || verify_rc=$?
printf '%s\n' "$verify_rc" >"${HERE}/restore.exit-code"

sync_rc=0
mkdir -p -- "${HERE}/${RESULT_DIR}"
rsync -a --protect-args -e "${RSYNC_SSH}" \
  "${REMOTE}:${REMOTE_ROOT}/${RESULT_DIR}/" "${HERE}/${RESULT_DIR}/" || sync_rc=$?
for file in capture.stdout.log capture.stderr.log restore.stdout.log restore.stderr.log; do
  rsync -a --protect-args -e "${RSYNC_SSH}" "${REMOTE}:${REMOTE_ROOT}/${file}" "${HERE}/" || sync_rc=$?
done

if (( capture_rc != 0 || verify_rc != 0 || sync_rc != 0 )); then
  printf 'serving profile failed: capture=%d verify=%d sync=%d\n' \
    "${capture_rc}" "${verify_rc}" "${sync_rc}" >&2
  exit 1
fi
printf 'serving profile captured, independently restored, and synced to %s\n' "${HERE}/${RESULT_DIR}"
