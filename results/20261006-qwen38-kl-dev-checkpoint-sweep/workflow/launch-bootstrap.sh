#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/home/mike/b70-evals/20261006-qwen38-kl-dev-checkpoint-sweep
CODE=/home/mike/code/b70-inference
HOST=qwen38-mtp-h100-dev-sweep-20261006
SIGNER=/home/mike/b70-evals/20261002-glimmer-mtp-training/stage1-50k/r2-sdk-venv/bin/python
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
  if (( status != 0 )); then
    printf 'Bootstrap failed; same owned lease capped4h/$13.20 retained for immediate correction; no training claimed.\n'
  fi
  exit "$status"
}
trap finish EXIT
if ! test -s "$ROOT/backup-private.json"; then "$SIGNER" "$ROOT/prepare-r2.py"; else printf 'Reusing still-valid private R2 upload URLs for the same expanded run.\n'; fi
python3 "$ROOT/check-ready.py"
# Provider messages were streamed above; no duplicate create.
ssh -o ConnectTimeout=20 "$HOST" 'mkdir -p ~/qwen-mtp-run ~/qwen-mtp-code/scripts/experiments ~/qwen-mtp-code/tests ~/qwen-mtp-code/patches ~/qwen-mtp-upload; chmod 700 ~/qwen-mtp-run ~/qwen-mtp-code ~/qwen-mtp-upload'
CONNECTED=1
rsync -a "$CODE/scripts/experiments/qwen38_train_mtp.py" "$CODE/scripts/experiments/qwen38_mtp_native_corpus.py" "$CODE/scripts/experiments/qwen38_lossy_probe.py" "$CODE/scripts/experiments/qwen38_dflash2_probe.py" "$CODE/scripts/experiments/qwen38_mtp_reference.py" "$CODE/scripts/experiments/qwen38_mtp_tune_probe.py" "$HOST:qwen-mtp-code/scripts/experiments/"
rsync -a "$CODE/tests/test_qwen38_train_mtp.py" "$HOST:qwen-mtp-code/tests/"
rsync -a "$CODE/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_mtp_training.py" "$CODE/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/b70_mtp_training.py" "$CODE/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_mtp_native_capture.py" "$CODE/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/b70_mtp_native_capture.py" "$HOST:qwen-mtp-code/patches/"
rsync -a "$CODE/scripts/experiments/qwen38_mtp_live_parity.py" "$HOST:qwen-mtp-code/scripts/experiments/"
rsync -a "$CODE/tests/test_qwen38_mtp_live_parity.py" "$HOST:qwen-mtp-code/tests/"
rsync -a "$CODE/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/qwen38_mtp_parity.py" "$HOST:qwen-mtp-code/patches/"
ssh "$HOST" 'mkdir -p ~/qwen-mtp-code/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b; cp ~/qwen-mtp-code/patches/qwen38_mtp_parity.py ~/qwen-mtp-code/patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/qwen38_mtp_parity.py' 
scp -q "$ROOT/run-pilot.py" "$ROOT/sweep.py" "$ROOT/baseline-api.py" "$HOST:qwen-mtp-code/"
ssh "$HOST" 'cp ~/qwen-mtp-code/run-pilot.py ~/qwen-mtp-run/executed-serving-library.py; cp ~/qwen-mtp-code/sweep.py ~/qwen-mtp-run/executed-sweep.py'
scp -q "$ROOT/r2-transfer.py" "$HOST:qwen-mtp-code/r2-transfer.py"
scp -q "$ROOT/upload-archive.py" "$HOST:qwen-mtp-code/upload-archive.py"
scp -pq "$ROOT/backup-private.json" "$HOST:qwen-mtp-upload/backup-private.json"
scp -pq "$ROOT/restore-private.json" "$HOST:qwen-mtp-upload/restore-private.json"
scp -q "$ROOT/restore-cache.py" "$ROOT/restore-checkpoints.py" "$HOST:qwen-mtp-code/"
scp -q "$ROOT/pilot-protocol.json" "$ROOT/dev-requests.jsonl" "$ROOT/dev-checks.jsonl" "$HOST:qwen-mtp-run/"
scp -q "$ROOT/bootstrap.sh" "$HOST:qwen-mtp-code/bootstrap.sh"
ssh "$HOST" 'chmod 600 ~/qwen-mtp-upload/backup-private.json ~/qwen-mtp-upload/restore-private.json; sha256sum ~/qwen-mtp-code/scripts/experiments/*.py ~/qwen-mtp-code/patches/*.py > ~/qwen-mtp-run/source-sha256.txt'
FIXTURES=/home/mike/code/.pi-worktrees/b70-inference/qwen-mtp-cuda/results/20260909-qwen38-dflash2-rtn-standard/mtp4-clients/reference-source/patches
ssh "$HOST" 'mkdir -p ~/qwen-mtp-code/results/20260909-qwen38-dflash2-rtn-standard/mtp4-clients/reference-source/patches'
rsync -a "$FIXTURES/patch_draft_lmhead_int4.py" "$FIXTURES/patch_draft_mtp_int4.py" "$HOST:qwen-mtp-code/results/20260909-qwen38-dflash2-rtn-standard/mtp4-clients/reference-source/patches/"
printf 'Physically verified Scaleway H100 bootstrap: frozen BF16 model and saved-head evaluation; zero training authorized.\n'
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$HOST" 'bash -lc '\''exec nohup timeout --signal=TERM --kill-after=120s 3600s bash ~/qwen-mtp-code/bootstrap.sh > ~/qwen-bootstrap.log 2>&1'\'''
printf 'CUDA_BOOTSTRAP_AND_SAVED_CHECKPOINTS_READY_ZERO_TRAINING\n'
