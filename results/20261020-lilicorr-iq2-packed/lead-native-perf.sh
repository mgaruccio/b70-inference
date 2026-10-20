#!/usr/bin/env bash
set -euo pipefail
local_case=/home/mike/code/b70-inference/results/20261020-lilicorr-iq2-packed
remote_case=/home/mike/b70-evals/20261020-lilicorr-iq2-packed-r2
ssh_args=(-6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no)
transport='ssh -6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no'
remote='mike@[fd7a:115c:a1e0::c933:4f38]'
for name in run-native-perf.py test-native-perf.py native-perf-cases.txt NATIVE_PERF_PLAN.md; do
 rsync -r -e "$transport" "$local_case/$name" "$remote:$remote_case/"
done
# First SSH process owns capture; the inherited signal/finally lifecycle restores.
if ssh "${ssh_args[@]}" mike@fd7a:115c:a1e0::c933:4f38 'bash -s' <<'CAPTURE'
set -euo pipefail
cd /home/mike/b70-evals/20261020-lilicorr-iq2-packed-r2
python3 -B test-native-perf.py -v
if python3 -B run-native-perf.py --lead-run-native-perf >native-perf-lead-run.stdout.log 2>native-perf-lead-run.stderr.log; then rc=0; else rc=$?; fi
printf '%s\n' "$rc" >native-perf-lead-run.exit-code
cat native-perf-lead-run.stdout.log
tail -n 30 native-perf-lead-run.stderr.log
exit "$rc"
CAPTURE
then capture_rc=0; else capture_rc=$?; fi
printf 'NATIVE_PERF_CAPTURE_SSH_EXIT=%s\n' "$capture_rc"
# Deliberately independent SSH connection and Python process, even on failure.
if ssh "${ssh_args[@]}" mike@fd7a:115c:a1e0::c933:4f38 'bash -s' <<'VERIFY'
set -euo pipefail
cd /home/mike/b70-evals/20261020-lilicorr-iq2-packed-r2
if python3 -B run-native-perf.py --lead-verify-restored >native-perf-lead-restore.stdout.log 2>native-perf-lead-restore.stderr.log; then rc=0; else rc=$?; fi
printf '%s\n' "$rc" >native-perf-lead-restore.exit-code
cat native-perf-lead-restore.stdout.log
tail -n 30 native-perf-lead-restore.stderr.log
exit "$rc"
VERIFY
then restore_rc=0; else restore_rc=$?; fi
printf 'NATIVE_PERF_SEPARATE_RESTORATION_SSH_EXIT=%s\n' "$restore_rc"
# Retain failures too. Do not delete or overwrite completed run directories.
rsync -r -e "$transport" "$remote:$remote_case/native-perf-r1/" "$local_case/native-perf-r1/"
for name in native-perf-lead-run.stdout.log native-perf-lead-run.stderr.log native-perf-lead-run.exit-code native-perf-lead-restore.stdout.log native-perf-lead-restore.stderr.log native-perf-lead-restore.exit-code; do
 rsync -r -e "$transport" "$remote:$remote_case/$name" "$local_case/"
done
test "$capture_rc" = 0
test "$restore_rc" = 0
