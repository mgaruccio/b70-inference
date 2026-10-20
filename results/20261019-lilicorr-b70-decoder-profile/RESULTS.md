# Decoder-marker continuation — intended coverage expansion not achieved

**No optimization, speedup, or promotion.** The added exact `inp_noise_embd`
marker did **not** expand observed operator coverage beyond the same 167-node
lattice-head segment. Do not describe this as a full decoder profile.

The actual capture and independent analysis report `status=analyzed`,
`errors=[]`, and `independently_restored=true`. All 100 public prevalidation
jobs, 24 measured responses, and inherited native/IQ2/Q8 gates passed. Q8
proposals/acceptance remained short 129/84 and long 133/82. These checks establish
fidelity/restoration, **not the intended profiler coverage**.

Six measured requests per point, median host wall milliseconds per response:

- Short: outer draft submission 310.633; instrumented head core 58.390;
  post-loop logging 8.350. All 43 draft owners still have only 167 observed nodes.
- Long: outer draft submission 340.537; instrumented head core 61.477;
  post-loop logging 8.785. All 45 draft owners still have only 167 observed nodes.

The remaining interval is not uniquely attributed. Source inspection confirms
that the drafter can share host-resident target token embeddings, and the backend
scheduler names cross-backend tensor copies `%s#%s#%d`. Renaming is a plausible
explanation for the ineffective input-name gate, not demonstrated causality in
this run's uninstrumented segment. Further name-gate deployments are rejected.
The continuation instead needs an explicit drafter context execution scope,
validated on the actual scheduler path before another full capture.

## Actual execution on inference-host

From `/home/mike/b70-evals/20261019-lilicorr-b70-decoder-profile`:

```bash
python3 -B test-profile.py
python3 -B source-proof.py
bash build-candidate.sh
python3 -B source-proof.py --built
python3 -B run-profile.py --lead-run-profile
# Separate process, after capture/restoration:
python3 -B run-profile.py --lead-after-profile-restore
```

The frozen-SDK build exited 0 and retained the exact expected three-library
delta. The completed SYCL seed's source/binary proof was unchanged before/after
the reflink/incremental build. All 6 continuation tests passed locally and on
host, including the inherited 18-test suite (which includes the 13 phase tests).
Before build, the lead corrected build/dependency success markers to match the
inherited proof and added a regression; the real `--built` proof passed on host.
No host packages, protected sources, model bytes, arithmetic, kernel operations,
acceptance policy, or persistent launcher were changed.

`profile-r1/analysis-r1/{analysis,profile}.json` retains actual public/phase/operator
results; `profile-r1/analysis-live-restoration.json` records independent live
restoration. Build manifests, source maps, seed-before/after proofs and raw logs
are retained alongside them. The original service was restored and no slower
candidate was promoted. A successful partial analysis is not a successful
coverage-expansion experiment; both observations are preserved unchanged.


All 2,481 transferred raw profile files matched remote SHA256. Git retention uses
79 verified lossless log copies (1,022,675,112 raw bytes → 65,220,954 gzip bytes)
and `profile.json.gz` (128,778,690 → 6,389,881 bytes), whose decompressed SHA256 is
`12dc929b8a4a14159c947e7497226b1af2cca3dc0fc3efaaaa4ec25910133daf`.
All local and remote raw originals remain; decompress for original filenames.
