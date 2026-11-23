#!/usr/bin/env bash
# Hybrid follow-up: quality gates for H1/H2, then timed A B H1 H2 x2 (same session, interleaved).
set -uo pipefail
c=$HOME/cmp
export TF_REF=$c/runs/20261122-221209-A-s0-qgate
echo "start $(date -Is)"
for cfg in H1 H2; do $c/cmp.sh $cfg 0 qgate smoke parity tf || echo "FAIL $cfg qgate"; done
for r in 1 2; do
  for cfg in A B H1 H2; do
    $c/cmp.sh $cfg 0 hr$r smoke decode depth6k depth30k prefill || echo "FAIL $cfg s0 hr$r"
    $c/cmp.sh $cfg 1 hr$r smoke decode depth6k depth30k || echo "FAIL $cfg s1 hr$r"
  done
done
echo "done $(date -Is)"
