#!/usr/bin/env bash
# Timed development-tier rounds: ABC order, two rounds, spec 0/1. Failures are logged and the round continues.
set -uo pipefail
c=$HOME/cmp
TF_REF=$c/runs/20261122-221209-A-s0-qgate
export TF_REF
echo "start $(date -Is)"
# Quality gate for C with PR #69186 (the C0 gate ran on the unpatched image).
$c/cmp.sh C 0 qgate smoke parity tf decode || echo "FAIL C qgate"
for r in 1 2; do
  for cfg in A B C; do
    $c/cmp.sh $cfg 0 r$r smoke decode depth6k depth30k prefill || echo "FAIL $cfg s0 r$r"
    $c/cmp.sh $cfg 1 r$r smoke decode depth6k depth30k || echo "FAIL $cfg s1 r$r"
  done
done
echo "done $(date -Is)"
