#!/usr/bin/env bash
# Live inkling-inference (:8080, API key) check: health, chat smoke, tool call, DSpark decode short + 30K.
set -euo pipefail
label=$1
art=$HOME/cmp/live/$(date +%Y%m%d-%H%M%S)-$label; mkdir -p "$art"
export LIVE_KEY=$(sudo cat /etc/b70-flint/api-key)
url=http://127.0.0.1:8080; H=(-H "Authorization: Bearer $LIVE_KEY" -H 'Content-Type: application/json')
for _ in $(seq 1 300); do curl -fsS -o /dev/null "${H[@]}" "$url/health" 2>/dev/null && break; sleep 2; done
curl -fsS "${H[@]}" "$url/v1/models" | jq -c '[.data[] | {id, max_model_len}]' | tee "$art/models.json"
curl -sS --fail-with-body --max-time 300 "${H[@]}" -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 37 plus 58? Reply with only the number."}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"reasoning_effort":"low"}}' "$url/v1/chat/completions" > "$art/chat.json"
jq -e '.choices[0] | .finish_reason == "stop" and (.message.content | test("^\\s*95\\s*$"))' "$art/chat.json"
curl -sS --fail-with-body --max-time 300 "${H[@]}" -d '{"model":"inkling-2-mini","temperature":0,"max_tokens":256,"chat_template_kwargs":{"reasoning_effort":"low"},"messages":[{"role":"user","content":"Call echo_value with the exact string B70_TOOL_OK. Do not answer in prose."}],"tools":[{"type":"function","function":{"name":"echo_value","description":"Echo a supplied value.","parameters":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}}}]}' "$url/v1/chat/completions" > "$art/tool.json"
jq -e '.choices[0] | .finish_reason == "tool_calls" and any(.message.tool_calls[]; .function.name == "echo_value" and (.function.arguments|fromjson|.value == "B70_TOOL_OK"))' "$art/tool.json"
py=$HOME/flint-vllm-spike/.venv/bin/python
"$py" -B $HOME/cmp/live/live_bench.py "$art" --port 8080 --tokens 256 | tail -1
"$py" -B $HOME/cmp/live/live_bench.py "$art" --port 8080 --tokens 256 --depth 30000 --cases 3 | tail -1
echo "ART $art"
