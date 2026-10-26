#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/tmp/20261026-eagle4-profile
REMOTE=/home/mike/b70-evals/20261026-eagle4-profile
SSH=(ssh -6 -o IPQoS=none -o BatchMode=yes -o ConnectTimeout=10 -o 'HostName=fe80::c234:583f:5eca:d498%%wlan0' -o HostKeyAlias=inference-host)
collect() {
 rc=$?
 trap - EXIT
 rsync -a -e "ssh -6 -o IPQoS=none -o BatchMode=yes -o ConnectTimeout=10 -o HostName=fe80::c234:583f:5eca:d498%%wlan0 -o HostKeyAlias=inference-host" "inference-host:$REMOTE/draft-graphs/" "$ROOT/draft-graphs/" || true
 exit "$rc"
}
"${SSH[@]}" inference-host "test ! -e $REMOTE/draft-graphs"
tar -C "$ROOT" -cf - draft-graphs launch-draft-graphs.sh run-native-graphs.sh run-candidate.sh profile-candidate.py | "${SSH[@]}" inference-host "tar -xf - -C $REMOTE"
trap collect EXIT
"${SSH[@]}" inference-host "bash $REMOTE/run-native-graphs.sh > $REMOTE/draft-graphs/native.log 2>&1"
"${SSH[@]}" inference-host "bash $REMOTE/run-candidate.sh"
