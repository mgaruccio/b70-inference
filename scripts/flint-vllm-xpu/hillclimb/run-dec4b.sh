#!/usr/bin/env bash
set -uo pipefail
cd ~/flint-vllm-spike
F=.venv/lib/python3.12/site-packages/vllm/models/flint
cmp -s build-dec4/model.py $F/model.py || { echo venv differs from build-dec4; exit 5; }
.venv/bin/python -I ktest3/fix_dec4_patch.py $F/model.py || exit 6
cp $F/model.py build-dec4/model.py
OFFLINE_SCRIPT=ktest3/shdbg.py OFFLINE_MAXSEC=900 bash maintenance.sh --offline >/dev/null 2>&1
r=$(ls -td http-run.* | head -1); cat $r/shdbg.json
grep -q '"attr": "(tensor' $r/shdbg.json || { echo PACK NOT BUILT AT LOAD; exit 7; }
for i in 1 2; do
  echo "== graph bench $i"
  EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096" bash maintenance.sh --bench
  echo "GRAPH_RUN_$i=$(ls -td http-run.* | head -1)"
done
echo "== eager bench"
READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096" bash maintenance.sh --bench
echo "EAGER_RUN=$(ls -td http-run.* | head -1)"
