#!/usr/bin/env bash
set -euo pipefail
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-qwen38-four-way-speed
for cell in mtp4-01 custom-01 dflash-01 dspark-01 dspark-02 dflash-02 custom-02 mtp4-02; do
  arm=${cell%-*}
  python3 -u run-comparison.py --cell "$arm" --out "$PWD/$cell" > "$cell.console.log" 2>&1 || { rc=$?; echo "$rc" > "$cell.exit-code.txt"; tail -60 "$cell.console.log"; exit "$rc"; }
  echo 0 > "$cell.exit-code.txt"
  echo "COMPLETED $cell"
done
python3 analyze.py > comparison.json
