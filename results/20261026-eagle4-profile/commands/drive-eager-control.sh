#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/tmp/20261026-eagle4-profile
REMOTE=/home/mike/b70-evals/20261026-eagle4-profile
SSH=(ssh -6 -o IPQoS=none -o BatchMode=yes -o ConnectTimeout=10 -o 'HostName=fe80::c234:583f:5eca:d498%%wlan0' -o HostKeyAlias=inference-host)
"${SSH[@]}" inference-host "test ! -e $REMOTE/eager-control"
tar -C "$ROOT" -cf - eager-control launch-eager-control.sh launch-native-mtp.sh run-arm.sh | "${SSH[@]}" inference-host "tar -xf - -C $REMOTE"
"${SSH[@]}" inference-host "bash $REMOTE/run-arm.sh eager-control eagle 6 0"
rsync -a -e "ssh -6 -o IPQoS=none -o BatchMode=yes -o ConnectTimeout=10 -o HostName=fe80::c234:583f:5eca:d498%%wlan0 -o HostKeyAlias=inference-host" "inference-host:$REMOTE/eager-control/measure/" "$ROOT/eager-control/measure/"
