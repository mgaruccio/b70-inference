#!/usr/bin/env bash
set -euo pipefail
umask 077
p=$HOME/inkling-vllm-xpu
stage=$p/sparse-69444
live=$p/serve/serve-inkling.sh
svc=inkling-inference.service
candidate=$1
out=$(mktemp -d "$stage/promote-dspark-schedule.XXXXXX")
backup=$live.pre-dspark-schedule-$(date -u +%Y%m%d-%H%M%S)
started=0
roll_back() {
    local rc=$?
    trap - EXIT
    trap '' INT TERM HUP
    if ((rc != 0 && started)); then
        set +e
        sudo -n journalctl -u "$svc" --since "$since" --no-pager -o cat | tee "$out/candidate.log" >/dev/null
        grep -E 'Available KV cache|KV cache size|ValueError:|AssertionError:|Error:' "$out/candidate.log" | head -15 >&2
        if sudo -n systemctl stop "$svc" && cp -p "$backup" "$live.rollback" && mv -f "$live.rollback" "$live" && sudo -n systemctl start "$svc" && timeout 300 bash "$HOME/cmp/live/live_check.sh" rollback-dspark-schedule >"$out/rollback.log" 2>&1; then
            echo "Restored and smoke-tested the prior launcher: $backup"
        else
            echo "ERROR: rollback did not verify; backup is $backup" >&2
        fi
    fi
    exit "$rc"
}
trap roll_back EXIT
trap 'exit 143' INT TERM HUP
exec 9>"$stage/maintenance.lock"
flock -n 9
[[ $(hostname -s) == b70-dev && $(id -un) == mike ]]
systemctl is-active --quiet "$svc"
sudo -n true
bash -n "$candidate"
test -x "$stage/.venv/bin/python"
test -s "$stage/sparse/vllm/v1/core/single_type_kv_cache_manager.py"
compgen -G "$stage/plugin/vllm_inkling2_xpu/ops/_inkling_nvfp4_xe2*.so" >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8080/metrics | awk '/^vllm:num_requests_(running|waiting)\{/ { seen++; if ($NF != 0) busy=1 } END { exit(seen < 2 || busy) }'
cp -p "$live" "$backup"
echo "Artifacts: $out; rollback launcher: $backup"
since=$(date '+%F %T')
started=1
sudo -n systemctl stop "$svc"
install -m "$(stat -c %a "$backup")" "$candidate" "$live.next"
mv -f "$live.next" "$live"
sudo -n systemctl start "$svc"
ready=0
for ((i=0; i<120; i++)); do
    if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then ready=1; break; fi
    if [[ $(systemctl show "$svc" -p SubState --value) != running ]]; then break; fi
    sleep 1
done
((ready))
timeout 300 bash "$HOME/cmp/live/live_check.sh" promoted-dspark-schedule | tee "$out/smoke.log"
[[ $(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models) == 401 ]]
"$stage/.venv/bin/python" -I - "$out" <<'PY'
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from urllib.request import Request, urlopen

out = Path(sys.argv[1])
key = subprocess.check_output(["sudo", "-n", "cat", "/etc/b70-flint/api-key"], text=True).strip()
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
url = "http://127.0.0.1:8080"

def call(path, opts=None, timeout=10):
    data = None if opts is None else json.dumps(opts).encode()
    with urlopen(Request(url + path, data=data, headers=headers), timeout=timeout) as r:
        return r.read()

def metrics():
    out = {}
    for line in call("/metrics").decode().splitlines():
        if line.startswith("vllm:"):
            name, val = line.rsplit(maxsplit=1)
            m = name.split("{", 1)[0].removeprefix("vllm:")
            out[m] = out.get(m, 0) + float(val)
    assert "num_preemptions_total" in out, "preemption metric missing"
    return out

models = json.loads(call("/v1/models"))
assert any(m["id"] == "inkling-2-mini" and m.get("max_model_len") == 409000 for m in models["data"]), models
(out / "models.json").write_text(json.dumps(models) + "\n")
before = metrics()
rng = random.Random(408744)
start = time.monotonic()
res = json.loads(call("/v1/completions", dict(
    model="inkling-2-mini", prompt=[rng.randrange(1024, 30000) for _ in range(408744)],
    max_tokens=128, temperature=0, seed=42, ignore_eos=True, return_token_ids=True,
), timeout=480))
after = metrics()
preemptions = after["num_preemptions_total"] - before["num_preemptions_total"]
drafts = after.get("spec_decode_num_drafts_total", 0) - before.get("spec_decode_num_drafts_total", 0)
accepted = after.get("spec_decode_num_accepted_tokens_total", 0) - before.get("spec_decode_num_accepted_tokens_total", 0)
usage = res["usage"]
report = dict(usage=usage, seconds=round(time.monotonic() - start, 2), preemptions=preemptions,
              drafts=drafts, accepted_draft_tokens=accepted)
(out / "near-limit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
assert (usage["prompt_tokens"], usage["completion_tokens"], preemptions) == (408744, 128, 0), report
assert drafts > 0 and accepted > 0, report
PY
systemctl is-active --quiet "$svc"
cmp "$candidate" "$live"
sudo -n journalctl -u "$svc" --since "$since" --no-pager -o cat | tee "$out/candidate.log" >/dev/null
grep -E 'Available KV cache|XPU KV cache size' "$out/candidate.log"
sha256sum "$live"
echo '409K/C32 production verification passed.'
