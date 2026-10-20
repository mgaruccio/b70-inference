#!/usr/bin/env bash
set -euo pipefail
local_case=/home/mike/code/b70-inference/results/20261020-lilicorr-q8-head-screen
remote_case=/home/mike/b70-evals/20261020-lilicorr-q8-head-screen
ssh_args=(-6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no)
transport='ssh -6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no'
remote='mike@[fd7a:115c:a1e0::c933:4f38]'
ssh "${ssh_args[@]}" mike@fd7a:115c:a1e0::c933:4f38 "test ! -e '$remote_case/native-perf-r1' && mkdir -p '$remote_case'"
for name in run-native-perf.py test-native-perf.py native-perf-cases.txt NATIVE_PERF_PLAN.md; do
 rsync -r -e "$transport" "$local_case/$name" "$remote:$remote_case/"
done
if ssh "${ssh_args[@]}" mike@fd7a:115c:a1e0::c933:4f38 'bash -s' <<'CAPTURE'
set -euo pipefail
cd /home/mike/b70-evals/20261020-lilicorr-q8-head-screen
python3 -B test-native-perf.py -v >cpu-checks.txt 2>&1
cat cpu-checks.txt
if python3 -B run-native-perf.py --lead-run-native-perf >native-perf-lead-run.stdout.log 2>native-perf-lead-run.stderr.log; then rc=0; else rc=$?; fi
printf '%s\n' "$rc" >native-perf-lead-run.exit-code
cat native-perf-lead-run.stdout.log
tail -n 30 native-perf-lead-run.stderr.log
exit "$rc"
CAPTURE
then capture_rc=0; else capture_rc=$?; fi
printf 'Q8_HEAD_SCREEN_CAPTURE_SSH_EXIT=%s\n' "$capture_rc"
# Separate SSH and Python process, including when the capture fails.
if ssh "${ssh_args[@]}" mike@fd7a:115c:a1e0::c933:4f38 'bash -s' <<'VERIFY'
set -euo pipefail
cd /home/mike/b70-evals/20261020-lilicorr-q8-head-screen
if python3 -B run-native-perf.py --lead-verify-restored >native-perf-lead-restore.stdout.log 2>native-perf-lead-restore.stderr.log; then rc=0; else rc=$?; fi
printf '%s\n' "$rc" >native-perf-lead-restore.exit-code
cat native-perf-lead-restore.stdout.log
tail -n 30 native-perf-lead-restore.stderr.log
exit "$rc"
VERIFY
then restore_rc=0; else restore_rc=$?; fi
printf 'Q8_HEAD_SCREEN_SEPARATE_RESTORATION_SSH_EXIT=%s\n' "$restore_rc"
rsync -r -e "$transport" "$remote:$remote_case/" "$local_case/"
test "$capture_rc" = 0
test "$restore_rc" = 0
