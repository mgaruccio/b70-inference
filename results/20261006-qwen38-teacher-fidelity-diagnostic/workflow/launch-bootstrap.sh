#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/home/mike/b70-evals/20261006-qwen38-teacher-fidelity-diagnostic
CODE=/home/mike/code/b70-inference
HOST=qwen38-teacher-fidelity-check-20261006
CONNECTED=0
finish() {
  local status=$?
  trap - EXIT
  set +e
  printf '%s\n' "$status" > "$ROOT/bootstrap-workflow-exit.txt"
  if (( CONNECTED )); then
    timeout 120 scp -q "$HOST:qwen-bootstrap.log" "$ROOT/remote-bootstrap.log"
    timeout 180 rclone copyto "$(cat "$ROOT/r2-location.txt")/bootstrap.tgz" "$ROOT/bootstrap.tgz" --contimeout 15s --timeout 60s
  fi
  exit "$status"
}
trap finish EXIT
python3 "$ROOT/check-ready.py"
ssh -o ConnectTimeout=20 "$HOST" 'mkdir -p ~/qwen-mtp-run ~/qwen-mtp-code/scripts/experiments ~/qwen-mtp-code/tests ~/qwen-mtp-code/patches ~/qwen-mtp-upload; chmod 700 ~/qwen-mtp-run ~/qwen-mtp-code ~/qwen-mtp-upload'
CONNECTED=1
for file in qwen38_train_mtp.py qwen38_teacher_fidelity.py qwen38_mtp_live_parity.py qwen38_mtp_native_corpus.py qwen38_lossy_probe.py qwen38_dflash2_probe.py qwen38_mtp_reference.py qwen38_mtp_tune_probe.py; do
  rsync -a "$CODE/scripts/experiments/$file" "$HOST:qwen-mtp-code/scripts/experiments/"
done
rsync -a "$CODE/tests/test_qwen38_train_mtp.py" "$CODE/tests/test_qwen38_mtp_live_parity.py" "$CODE/tests/test_qwen38_teacher_fidelity.py" "$HOST:qwen-mtp-code/tests/"
PATCHES="$CODE/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b"
rsync -a "$PATCHES/patch_mtp_training.py" "$PATCHES/b70_mtp_training.py" "$PATCHES/patch_mtp_native_capture.py" "$PATCHES/b70_mtp_native_capture.py" "$PATCHES/qwen38_mtp_parity.py" "$HOST:qwen-mtp-code/patches/"
ssh "$HOST" 'mkdir -p ~/qwen-mtp-code/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b; cp ~/qwen-mtp-code/patches/qwen38_mtp_parity.py ~/qwen-mtp-code/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/qwen38_mtp_parity.py'
scp -q "$ROOT/run-pilot.py" "$ROOT/baseline-api.py" "$ROOT/diagnose-fidelity.py" "$ROOT/verifier-histories.py" "$ROOT/r2-transfer.py" "$ROOT/upload-archive.py" "$ROOT/bootstrap.sh" "$HOST:qwen-mtp-code/"
scp -pq "$ROOT/backup-private.json" "$HOST:qwen-mtp-upload/backup-private.json"
scp -q "$ROOT/pilot-protocol.json" "$ROOT/archived-teacher-inputs.json" "$ROOT/train-requests.jsonl" "$ROOT/dev-requests.jsonl" "$ROOT/dev-checks.jsonl" "$HOST:qwen-mtp-run/"
FIXTURES=/home/mike/code/.pi-worktrees/b70-inference/qwen-teacher-fidelity/results/20260909-qwen38-dflash2-rtn-standard/mtp4-clients/reference-source/patches
ssh "$HOST" 'mkdir -p ~/qwen-mtp-code/results/20260909-qwen38-dflash2-rtn-standard/mtp4-clients/reference-source/patches; chmod 600 ~/qwen-mtp-upload/backup-private.json; cp ~/qwen-mtp-code/diagnose-fidelity.py ~/qwen-mtp-run/executed-fidelity.py; cp ~/qwen-mtp-code/run-pilot.py ~/qwen-mtp-run/executed-serving-library.py; sha256sum ~/qwen-mtp-code/scripts/experiments/*.py ~/qwen-mtp-code/patches/*.py > ~/qwen-mtp-run/source-sha256.txt'
rsync -a "$FIXTURES/patch_draft_lmhead_int4.py" "$FIXTURES/patch_draft_mtp_int4.py" "$HOST:qwen-mtp-code/results/20260909-qwen38-dflash2-rtn-standard/mtp4-clients/reference-source/patches/"
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$HOST" 'bash -lc '\''exec nohup timeout --signal=TERM --kill-after=120s 1800s bash ~/qwen-mtp-code/bootstrap.sh > ~/qwen-bootstrap.log 2>&1'\'''
printf 'PINNED_H100_FIDELITY_BOOTSTRAP_READY_ZERO_UPDATES\n'
