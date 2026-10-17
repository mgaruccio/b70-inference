#!/usr/bin/env bash
set -euo pipefail
cd /home/mike/b70-evals/20261017-nemotron35-bolt-decode-profile
while IFS= read -r -d '' file; do
    gzip -n -c -- "$file" > "$file.gz"
    gzip -t -- "$file.gz"
done < <(find run-r1 -type f \( -name '*.log' -o -name 'llama-server.1.json' -o -name 'unitrace.1.txt' \) -print0)
printf 'COMPRESSED_EVIDENCE_INTEGRITY_PASS\n'
