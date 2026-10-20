# Experimental full-trace Q8 ABBA adapter

This package is a lead-only diagnostic adapter for the existing Q8 drafter runner. It
adds only a private `/capture` read/write bind, `BOLT_VERIFY_CAPTURE_DIR=/capture`,
`LLAMA_TRACE=1`, and strict positive checks for the actual trace container, PID1,
raw startup/server logs, and 524288-byte `row-*` CPU sampler files. The qualified
source, Q8 model bind, IQ2/Q8 singleton arithmetic, full-block-8/lattice gates,
rejected-noncausal-KV behavior, W7 exclusion, public requests, ABBA order, and
inherited restoration proof remain unchanged. No trace rows are read back by the
backend or interpreted by this adapter.

The output is fresh and diagnostic only:

```text
/home/mike/b70-evals/20261019-lilicorr-q8-drafter/verification-routes-20261020-r2
```

On `inference-host`, after staging this directory beside the existing Q8 package:

```bash
python3 -B test-performance.py -v
python3 -B run-performance.py --lead-run-timing
python3 -B run-performance.py --lead-after-timing-restore
```

The lead owns the real B70 journey, storage, and any final interpretation. Host-side
launch/route records are diagnostic rather than GPU event times; see Intel's kernel
launch guidance: <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/kernel-launch.html>.
