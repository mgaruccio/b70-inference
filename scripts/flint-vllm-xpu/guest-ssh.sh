#!/usr/bin/env bash
set -euo pipefail
kh=$(mktemp)
trap 'rm -f "$kh"' EXIT
ssh -n -o BatchMode=yes b70-host 'ssh-keygen -F 192.168.122.70 -f ~/.config/thermalwriter/b70-display/known_hosts' > "$kh"
ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$kh" -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -J b70-host 192.168.122.70 "$@"
