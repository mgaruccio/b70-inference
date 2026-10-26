#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/tmp/20261026-eagle4-profile
REMOTE=/home/mike/b70-evals/20261026-eagle4-profile
SSH=(ssh -6 -o IPQoS=none -o BatchMode=yes -o ConnectTimeout=10 -o 'HostName=fe80::c234:583f:5eca:d498%%wlan0' -o HostKeyAlias=inference-host)
collect() {
 rc=$?
 trap - EXIT
 rsync -a -e "ssh -6 -o IPQoS=none -o BatchMode=yes -o ConnectTimeout=10 -o HostName=fe80::c234:583f:5eca:d498%%wlan0 -o HostKeyAlias=inference-host" "inference-host:$REMOTE/mtp-final/measure/" "$ROOT/mtp-final/measure/" || true
 exit "$rc"
}
"${SSH[@]}" inference-host "bash -c 'mkdir $REMOTE/mtp-final'"
trap collect EXIT
"${SSH[@]}" inference-host "bash $REMOTE/run-arm.sh mtp-final mtp 6 1"
