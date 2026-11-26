#!/usr/bin/env bash
# Execute inside a hillclimb.sh PRETEST window. Use the runtime driver, not the compiler's libsycl.
set -euo pipefail
: "${PRETEST_DIR:?requires the maintenance harness}"
p=$HOME/inkling-vllm-xpu
stage=$p/sparse-69444
runtime=$HOME/flint-vllm-spike
driver=$runtime/driver-26.40
here=$(cd "$(dirname "$0")" && pwd)
case $1 in
    stock) src=$stage ;;
    pf6|pf3|pf9|static|layout|attn|router|wide-n|narrow-n|dequant|paged|markov|prefix|merge|paired) src=$here/$1 ;;
    *) exit 2 ;;
esac
shift
export PYTHONPATH="$src/plugin:$p/vocab-probe/pytest-overlay:$here/tests:$stage/sparse:$runtime/serve"
export LD_LIBRARY_PATH="$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib:$driver/usr/local/lib"
export LIBRARY_PATH="$driver/usr/lib/x86_64-linux-gnu:$driver/usr/lib" CPATH="$driver/usr/include"
export OCL_ICD_VENDORS="$runtime/opencl-vendors" TRITON_CACHE_DIR="$stage/triton" HF_HUB_OFFLINE=1
export NEOReadDebugKeys=1 ForcePreemptionMode=3
unset VLLM_API_KEY
cd "$here"
exec "$stage/.venv/bin/python" -B "$@"
