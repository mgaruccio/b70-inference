#!/usr/bin/env bash
# Plugin path: P (published plugin on vLLM main + #69186) vs PC (+ INK2_CUTLASS_MOE=1). Gates then 2 rounds.
set -uo pipefail
c=$HOME/cmp
export TF_REF=$c/runs/20261122-221209-A-s0-qgate
echo "start $(date -Is)"
for cfg in P PC; do $c/cmp.sh $cfg 0 qgate smoke parity tf || echo "FAIL $cfg qgate"; done
for r in 1 2; do
  for cfg in P PC; do
    $c/cmp.sh $cfg 0 pr$r smoke decode depth6k depth30k prefill || echo "FAIL $cfg s0 pr$r"
    $c/cmp.sh $cfg 1 pr$r smoke decode depth6k depth30k || echo "FAIL $cfg s1 pr$r"
  done
done
echo "done $(date -Is)"
