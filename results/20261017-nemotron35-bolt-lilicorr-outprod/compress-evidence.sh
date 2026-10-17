#!/usr/bin/env bash
set -euo pipefail
cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-outprod
[[ "$(hostname)" == inference-host ]]
while IFS= read -r -d '' path; do
    test ! -e "$path.gz"
    gzip -n -c "$path" > "$path.gz"
    gzip -t "$path.gz"
    cmp "$path" <(gzip -dc "$path.gz")
done < <(find . -type f -name '*.log' -not -path './draft/*' -print0)
find . -type f -name '*.gz' -print0 | sort -z | xargs -0 sha256sum > compressed.sha256
printf 'BOLT_LILICORR_LOSSLESS_COMPRESSION_PASSED\n'
