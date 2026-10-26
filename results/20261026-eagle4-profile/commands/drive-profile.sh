#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/tmp/20261026-eagle4-profile
REMOTE=/home/mike/b70-evals/20261026-eagle4-profile
SSH=(ssh -6 -o IPQoS=none -o BatchMode=yes -o ConnectTimeout=10 -o 'HostName=fe80::c234:583f:5eca:d498%%wlan0' -o HostKeyAlias=inference-host)
"${SSH[@]}" inference-host "bash -c 'mkdir $REMOTE'"
tar -C "$ROOT" -cf - control launch-profile.sh profile-request.py run-profile.sh | "${SSH[@]}" inference-host "tar -xf - -C $REMOTE"
"${SSH[@]}" inference-host "bash $REMOTE/run-profile.sh"
rsync -a -e "ssh -6 -o IPQoS=none -o BatchMode=yes -o ConnectTimeout=10 -o HostName=fe80::c234:583f:5eca:d498%%wlan0 -o HostKeyAlias=inference-host" "inference-host:$REMOTE/control/" "$ROOT/control/"
