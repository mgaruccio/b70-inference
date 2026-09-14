#!/usr/bin/env bash
set -euo pipefail
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-grouped-quality
R=$PWD
IMAGE=$(docker image inspect --format '{{.Id}}' qwen38-quality:20260913)
mkdir -p preparation-02
printf '%s\n' "$IMAGE" > preparation-02/image-id.txt
docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --cpus 4 --memory 8g --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g --user "$(id -u):$(id -g)" --entrypoint python "$IMAGE" -c 'import torch; from importlib.metadata import version; assert torch.version.cuda is None; print(torch.__version__); print(version("lm-eval")); print(version("evalplus"))' > preparation-02/versions.txt
docker run --rm --network bridge --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --cpus 4 --memory 8g --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g --user "$(id -u):$(id -g)" -v "$R/preparation-02:/input:rw" "$IMAGE" prepare --out /input/prepared > preparation-02/prepare.log 2>&1
python3 - <<'PY'
import json,pathlib
root=pathlib.Path('preparation-02/prepared')
rows=[json.loads(line) for line in (root/'prepared.jsonl').read_text().splitlines()]
seen=set(); chosen=[]
for row in rows:
    if row['task'] not in seen and row.get('sample_role','primary')=='primary':
        seen.add(row['task']);chosen.append(row)
assert len(chosen)==4
(root/'smoke.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in chosen))
print('Prepared four-task smoke set', [row['id'] for row in chosen])
PY
