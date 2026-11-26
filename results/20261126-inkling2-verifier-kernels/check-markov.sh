#!/usr/bin/env bash
# Compare only the DSpark recurrent input-pack operation on the staged plugin.
set -euo pipefail
bash probe-python.sh markov -m pytest -q -rs -p no:cacheprovider markov/tests/test_dspark_markov_inputs.py | tee "$PRETEST_DIR/markov-tests.txt"
grep -Eq '^8 passed' "$PRETEST_DIR/markov-tests.txt"
bash probe-python.sh markov markov/scripts/bench_dspark_markov_inputs.py --check-dispatch -o "$PRETEST_DIR/markov-prefix.json"
