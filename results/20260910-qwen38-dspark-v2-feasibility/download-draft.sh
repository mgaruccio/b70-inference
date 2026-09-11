#!/usr/bin/env bash
set -euo pipefail
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility
mkdir draft
cd draft
revision=b9a5dbdf03bc999c6c73c426b19c2d9041cea393
base="https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/resolve/$revision"
printf 'repository=RadixArk/Qwen3.8-27B-DSpark\nrevision=%s\n' "$revision"
curl --fail --location --retry 2 --max-time 900 --output config.json.part "$base/config.json"
printf '%s\n' 'dd65fb1b01c2adea69512ff2990a79d58eb7fe2c7ea97375aa66f657a29a5bfd  config.json.part' | sha256sum --check
mv config.json.part config.json
curl --fail --location --retry 2 --max-time 900 --output model.safetensors.part "$base/model.safetensors"
printf '%s\n' '2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7  model.safetensors.part' | sha256sum --check
mv model.safetensors.part model.safetensors
stat -c '%n %s bytes' config.json model.safetensors
sha256sum config.json model.safetensors
