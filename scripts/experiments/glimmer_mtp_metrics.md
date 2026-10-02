# Live Glimmer MTP monitoring

Dashboard: https://hermes.tailc35014.ts.net:3000/d/glimmer-mtp/

Reuses the `model-router` Needle monitoring path: a temporary stdlib desktop
bridge, existing desktop Prometheus, remote_write to existing Hermes Prometheus,
and provisioned Hermes Grafana. No new metrics database or boot service.

```sh
scp scripts/experiments/glimmer_mtp_metrics.py \
  glimmer-mtp-train-20261002:~/mtp-training-code/scripts/experiments/
python3 scripts/experiments/glimmer_mtp_metrics.py \
  --ssh glimmer-mtp-train-20261002 \
  --run-dir /home/shadeform/mtp-training-run \
  --listen 100.109.144.103 --port 9110
curl -fsS http://100.109.144.103:9110/metrics
```

Configuration source: `~/code/glinet/metrics/prometheus.yml` and
`grafana/dashboards/glimmer-mtp.json`. Scrape/refresh every 20 seconds.
Only `job=glimmer-mtp` is added to the existing remote_write allowlist.
The bridge is read-only, bounded SSH observations, and never reads checkpoint
weights or installs ML dependencies. The desktop must remain running.
Stop the bridge when monitoring ends; do not stop the trainer to stop monitoring.

## Meaning of the panels

Process phase comes from actual `/proc` command lines, not successful submission.
Idle/stopped is not proof of failure. A stopped process with a traceback in its
latest component log is separately identified. An inaccessible cloud source
removes live training/GPU gauges; it never replays cached values. Exporter failure
is distinguished by Prometheus `up`. Observation age and log age are separate:
a fresh observation does not imply an optimizer update.

Live split counts come from actual capture-progress log events; complete totals
require the written index. Sequence tokens include prompts; eligible response
roots do not. Loss/updates
come from complete JSONL records; partial trailing writes are ignored. Tiny-fit
teacher agreement is explicitly training-set diagnostic, not heldout acceptance.
Stage-0 complete is not substantive experiment completion. Historical training
curves remain visible when a process stops; the run-state panel indicates activity.

## Verification / cleanup

Environment: approved Shadeform A100 host, existing desktop Prometheus, Hermes
Grafana 12.1 / Prometheus, Tailscale. Real input: live capture/training logs.

1. Run `python3 -m unittest discover -s tests -p test_glimmer_mtp_metrics.py -v`.
2. Read the remote snapshot and compare phase, update/loss, GPU and captured
   counts to the actual process/logs. `curl` the public metrics boundary above.
3. Validate with `docker exec metrics-prometheus-1 promtool check config
   /etc/prometheus/prometheus.yml`; reload existing Prometheus with SIGHUP.
4. Query Hermes `/api/v1/query` for `up{job="glimmer-mtp"}`,
   `mtp_source_reachable{job="glimmer-mtp"}`, `mtp_phase{job="glimmer-mtp"}` and
   real training gauges. Expect scrape/source 1 and values matching actual logs.
5. Open the dashboard in a real browser. Verify no panel query errors, real state,
   missing-data explanations, stage selector and advancing real updates/loss.
   Retain screenshots and observed metrics under
   `/home/mike/b70-evals/20261002-glimmer-mtp-training/monitoring/` (not Git).
6. After training ends, stop only the temporary bridge. Preserve existing Needle
   metrics, dashboards and Prometheus storage. No experimental weights deleted.

Observed 2026-10-02: the four targeted test files passed **165 tests** in the
isolated CPU Docker runtime. `promtool check config` passed. Actual `/metrics`
and Hermes reported `up=1`, source reachable 1 and phase 1 (teacher capture).
Every dashboard PromQL expression returned success. Browser verification showed
real Stage-0 losses/updates, selected-head validation acceptance, advancing main
capture counters, working stage filtering and the 20-second refresh control.
Artifacts: `monitoring/live-metrics.prom`, `panel-query-verification.json`,
`grafana-live.png`, and `grafana-live-final.png` under the path above. The desktop
Prometheus needed recreation (same storage/config) because its single-file bind
mount retained the old inode after the configuration edit; SIGHUP alone did not
load the new job. Existing Grafana/Needle services were not restarted.
Official research: Grafana file provisioning loads version-controlled dashboards
into the existing service, avoiding a parallel UI/store:
https://grafana.com/tutorials/provision-dashboards-and-data-sources/
Prometheus exposition defines the scrape boundary and missing metric semantics:
https://prometheus.io/docs/instrumenting/exposition_formats/
