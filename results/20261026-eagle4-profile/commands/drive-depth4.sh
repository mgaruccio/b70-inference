#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/tmp/20261026-eagle4-profile
REMOTE=/home/mike/b70-evals/20261026-eagle4-profile
SSH=(ssh -6 -o IPQoS=none -o BatchMode=yes -o ConnectTimeout=10 -o 'HostName=fe80::c234:583f:5eca:d498%%wlan0' -o HostKeyAlias=inference-host)
collect() {
 rc=$?
 trap - EXIT
 rsync -a -e "ssh -6 -o IPQoS=none -o BatchMode=yes -o ConnectTimeout=10 -o HostName=fe80::c234:583f:5eca:d498%%wlan0 -o HostKeyAlias=inference-host" "inference-host:$REMOTE/depth4/measure/" "$ROOT/depth4/measure/" || true
 exit "$rc"
}
"${SSH[@]}" inference-host "test ! -e $REMOTE/depth4"
tar -C "$ROOT" -cf - depth4 launch-depth4.sh | "${SSH[@]}" inference-host "tar -xf - -C $REMOTE"
trap collect EXIT
"${SSH[@]}" inference-host "bash $REMOTE/run-arm.sh depth4 eagle 6 1"
