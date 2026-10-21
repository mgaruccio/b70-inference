#!/usr/bin/env bash
set -euo pipefail
local_case=/home/mike/code/b70-inference/results/20261021-lilicorr-serving-profile
remote_case=/home/mike/b70-evals/20261021-lilicorr-serving-profile
ssh_args=(-6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no)
transport='ssh -6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no'
remote='mike@[fd7a:115c:a1e0::c933:4f38]'
for file in native-pilot-240.py test-native-pilot-240.py NATIVE_PILOT_240.md lead-native-pilot-240.sh; do
 rsync -r -e "$transport" "$local_case/$file" "$remote:$remote_case/"
done
if ssh "${ssh_args[@]}" mike@fd7a:115c:a1e0::c933:4f38 'bash -s' <<'CAPTURE'
set -euo pipefail
cd /home/mike/b70-evals/20261021-lilicorr-serving-profile
python3 -B test-native-pilot-240.py -v >pilot240-cpu-checks.txt 2>&1
python3 -B native-pilot-240.py --lead-run >pilot240-capture.stdout.log 2>pilot240-capture.stderr.log
CAPTURE
then capture_rc=0; else capture_rc=$?; fi
printf '%s\n' "$capture_rc" >"$local_case/pilot240-capture.exit-code"
if ssh "${ssh_args[@]}" mike@fd7a:115c:a1e0::c933:4f38 'bash -s' <<'VERIFY'
set -euo pipefail
cd /home/mike/b70-evals/20261021-lilicorr-serving-profile
python3 -B native-pilot-240.py --lead-verify-restored >pilot240-restore.stdout.log 2>pilot240-restore.stderr.log
VERIFY
then restore_rc=0; else restore_rc=$?; fi
printf '%s\n' "$restore_rc" >"$local_case/pilot240-restore.exit-code"
rsync -r -e "$transport" "$remote:$remote_case/native-pilot-240-r1/" "$local_case/native-pilot-240-r1/"
for file in pilot240-cpu-checks.txt pilot240-capture.stdout.log pilot240-capture.stderr.log pilot240-restore.stdout.log pilot240-restore.stderr.log; do
 rsync -r -e "$transport" "$remote:$remote_case/$file" "$local_case/"
done
printf 'PROFILER240_PILOT_CAPTURE=%s INDEPENDENT_RESTORE=%s\n' "$capture_rc" "$restore_rc"
tail -25 "$local_case/pilot240-capture.stderr.log"
tail -20 "$local_case/pilot240-restore.stderr.log"
test "$capture_rc" = 0
test "$restore_rc" = 0
