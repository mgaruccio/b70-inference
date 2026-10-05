#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/home/mike/b70-evals/20261005-qwen38-full-corpus-cuda
HOST=qwen38-mtp-h100-full-20261005
finish() {
  local result=$?
  trap - EXIT
  set +e
  printf '%s\n' "$result" > "$ROOT/training-workflow-exit.txt"
  timeout 120 scp -q "$HOST:qwen-pilot.log" "$ROOT/remote-pilot.log"
  timeout 120 scp -q "$HOST:qwen-mtp-run/comparison.json" "$HOST:qwen-mtp-run/stock-baseline.json" "$HOST:qwen-mtp-run/state-drift.json" "$ROOT/"
  timeout 120 scp -q "$HOST:qwen-mtp-run/training/tuned-mtp.json" "$ROOT/training-report.json"
  if (( result == 0 )); then
    python3 "$ROOT/verify-final.py"
    backup=$?
    if (( backup == 0 )); then
      timeout 120 python3 "$ROOT/provider.py" --delete
      result=$?
    else
      printf 'Full archive readback verification failed; the same capped lease remains available for rescue.\n'
      result=1
    fi
  else
    printf 'Full-corpus pipeline failed; same capped lease retained for immediate correction. No pilot fallback or duplicate rental.\n'
  fi
  exit "$result"
}
trap finish EXIT
ssh -o ConnectTimeout=20 "$HOST" 'bash -lc '\''set -e; test "$(cat ~/qwen-mtp-run/bootstrap-exit.txt)" = 0; test -f ~/qwen-model/config.json'\'''
scp -q "$ROOT/run-pilot.py" "$ROOT/resume-after-train-capture.py" "$ROOT/upload-archive.py" "$ROOT/r2-transfer.py" "$HOST:qwen-mtp-code/"
scp -q /home/mike/b70-evals/20261005-qwen38-native-mtp-baseline/run.py "$HOST:qwen-mtp-code/baseline-api.py"
ssh "$HOST" 'python3 -m py_compile ~/qwen-mtp-code/run-pilot.py ~/qwen-mtp-code/resume-after-train-capture.py ~/qwen-mtp-code/baseline-api.py ~/qwen-mtp-code/upload-archive.py; cp ~/qwen-mtp-code/run-pilot.py ~/qwen-mtp-run/executed-workflow.py; cp ~/qwen-mtp-code/resume-after-train-capture.py ~/qwen-mtp-run/resume-workflow.py; sha256sum ~/qwen-mtp-code/run-pilot.py ~/qwen-mtp-code/resume-after-train-capture.py ~/qwen-mtp-code/baseline-api.py ~/qwen-mtp-code/upload-archive.py ~/qwen-mtp-code/scripts/experiments/*.py ~/qwen-mtp-code/patches/*.py > ~/qwen-mtp-run/workflow-source-sha256.txt'
printf 'Starting full native BF16 corpus374/64;10 complete epochs3740 updates; dev selection; untouched64-test MTP4/8 comparisons.\n'
if [[ ${1:-} == resume ]]; then
  ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$HOST" 'bash -lc '\''exec nohup timeout --signal=TERM --kill-after=120s 10500s env LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} OMP_NUM_THREADS=8 ~/qwen-mtp-env/bin/python -u ~/qwen-mtp-code/resume-after-train-capture.py > ~/qwen-pilot.log 2>&1'\'''
else
  ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$HOST" 'bash -lc '\''exec nohup timeout --signal=TERM --kill-after=120s 10500s env LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} OMP_NUM_THREADS=8 ~/qwen-mtp-env/bin/python -u ~/qwen-mtp-code/run-pilot.py > ~/qwen-pilot.log 2>&1'\'''
fi
printf 'FULL_CORPUS_NATIVE_PIPELINE_FINISHED_BACKING_UP_BEFORE_GPU_DELETION\n'
