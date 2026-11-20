#!/usr/bin/env bash
# Teacher-forced decode capture for each build on the build-moe reference trajectory.
set -uo pipefail
cd ~/flint-vllm-spike
F=.venv/lib/python3.12/site-packages/vllm/models/flint
cat swap-build.sh | grep -q xpu_decode.py || { echo swap-build lacks xpu_decode; exit 2; }
for b in ${TF_BUILDS:-moe dec3 dec4}; do
  bash swap-build.sh $b || exit 3
  OFFLINE_SCRIPT=ktest3/tf.py OFFLINE_MAXSEC=1500 bash maintenance.sh --offline >/dev/null 2>&1
  r=$(ls -td http-run.* | head -1)
  if [[ -s $r/tf.jsonl ]]; then cp $r/tf.jsonl ktest3/tf-$b.jsonl; cp $r/free0.json ktest3/free0-$b.json; echo "TF $b $r ok";
  else echo "TF $b $r FAILED"; grep -v '^$' $r/server.log | tail -25; fi
done
bash swap-build.sh dec4
cmp -s build-dec4/model.py $F/model.py && cmp -s build-dec4/xpu_decode.py $F/ops/xpu_decode.py && echo "venv=build-dec4"
