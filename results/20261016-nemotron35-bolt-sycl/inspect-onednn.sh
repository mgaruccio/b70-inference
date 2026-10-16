#!/usr/bin/env bash
# Host-only library identification; no GPU device mounted, Qwen stays live.
set -euo pipefail
docker run --rm --name bolt-onednn-version \
  -v /home/mike/inference/src/oneDNN/install:/opt/onednn:ro \
  intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04 bash -lc '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
icx -x c - -I/opt/onednn/include -L/opt/onednn/lib -Wl,-rpath,/opt/onednn/lib -ldnnl -o /tmp/dnnl-version <<"C"
#include <stdio.h>
#include <oneapi/dnnl/dnnl.h>
int main(void) {
    const dnnl_version_t *v = dnnl_version();
    printf("oneDNN %d.%d.%d hash=%s cpu_runtime=%u gpu_runtime=%u\n",
           v->major, v->minor, v->patch, v->hash, v->cpu_runtime, v->gpu_runtime);
    return 0;
}
C
/tmp/dnnl-version
sha256sum /opt/onednn/lib/libdnnl.so.3
'
df -B1 /home/mike/inference/models
