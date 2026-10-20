#!/usr/bin/env bash
set -euo pipefail
local_case=/home/mike/code/b70-inference/results/20261020-lilicorr-q8-target-batch4
remote_case=/home/mike/b70-evals/20261020-lilicorr-q8-target-batch4
transport='ssh -6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no'
rsync -r -e "$transport" "$local_case/" "mike@[fd7a:115c:a1e0::c933:4f38]:$remote_case/"
if ssh -6 -o HostKeyAlias=inference-host -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@fd7a:115c:a1e0::c933:4f38 'bash -s' <<'BUILD'
set -euo pipefail
cd /home/mike/b70-evals/20261020-lilicorr-q8-target-batch4
python3 -B test-bindings.py >remote-bindings-cpu-checks.txt 2>&1
test ! -e build-attempt-r1
test ! -e /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-q8-target-batch4-r1
if python3 -B build-candidate.py >build-lead.stdout.log 2>build-lead.stderr.log; then rc=0; else rc=$?; fi
printf '%s\n' "$rc" >build-lead.exit-code
tail -n 40 build-lead.stdout.log
tail -n 40 build-lead.stderr.log
exit "$rc"
BUILD
then rc=0; else rc=$?; fi
# Preserve failed build output too; no candidate/seed cleanup or retry in-place.
rsync -r -e "$transport" "mike@[fd7a:115c:a1e0::c933:4f38]:$remote_case/" "$local_case/"
exit "$rc"
