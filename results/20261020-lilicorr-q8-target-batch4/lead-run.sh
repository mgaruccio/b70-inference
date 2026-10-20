#!/usr/bin/env bash
set -euo pipefail
local_case=/home/mike/code/b70-inference/results/20261020-lilicorr-q8-target-batch4
remote_case=/home/mike/b70-evals/20261020-lilicorr-q8-target-batch4
args=(-6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no)
transport='ssh -6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no'
# Do not sync/mutate candidate inputs after the completed build; verify them fresh.
if ssh "${args[@]}" mike@fd7a:115c:a1e0::c933:4f38 'bash -s' <<'CAPTURE'
set -euo pipefail
cd /home/mike/b70-evals/20261020-lilicorr-q8-target-batch4
python3 -B source-proof.py --built >pre-serving-build-proof.json
if python3 -B run-performance.py --lead-run-timing >capture-lead.stdout.log 2>capture-lead.stderr.log; then rc=0; else rc=$?; fi
printf '%s\n' "$rc" >capture-lead.exit-code
tail -n 30 capture-lead.stdout.log
tail -n 30 capture-lead.stderr.log
exit "$rc"
CAPTURE
then capture_rc=0; else capture_rc=$?; fi
printf 'Q8_TARGET_CAPTURE_SSH_EXIT=%s\n' "$capture_rc"
# A different SSH connection and Python process, including capture failures.
if ssh "${args[@]}" mike@fd7a:115c:a1e0::c933:4f38 'bash -s' <<'VERIFY'
set -euo pipefail
cd /home/mike/b70-evals/20261020-lilicorr-q8-target-batch4
if python3 -B run-performance.py --lead-after-timing-restore >after-lead.stdout.log 2>after-lead.stderr.log; then rc=0; else rc=$?; fi
printf '%s\n' "$rc" >after-lead.exit-code
tail -n 30 after-lead.stdout.log
tail -n 30 after-lead.stderr.log
exit "$rc"
VERIFY
then after_rc=0; else after_rc=$?; fi
printf 'Q8_TARGET_SEPARATE_RESTORATION_AND_ANALYSIS_SSH_EXIT=%s\n' "$after_rc"
rsync -r -e "$transport" "mike@[fd7a:115c:a1e0::c933:4f38]:$remote_case/" "$local_case/"
test "$capture_rc" = 0
test "$after_rc" = 0
