# B70 host CPU boost-off persistence

On 2026-10-09, persisted the existing supported mitigation on `inference-host`
(CachyOS, systemd 261.2, Limine 12.6.0, `amd-pstate-epp`). This is not a CPU
failure diagnosis or a firmware Curve Optimizer change. No EFI, SMU, MSR,
bootloader, or production inference launcher edits were made.

Source: `configs/systemd/b70-cpu-boost-off.service`.
Installed: `/etc/systemd/system/b70-cpu-boost-off.service`, root-owned mode 0644,
enabled by `multi-user.target.wants/b70-cpu-boost-off.service`.
No previous unit existed. `cpupower.service` was disabled and remains disabled.

## Research and test process

Fresh official references:
- <https://docs.kernel.org/admin-guide/pm/cpufreq.html#the-boost-file-in-sysfs>:
  writing 0 disables the global supported boost mechanism; not a BIOS edit.
- <https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html>:
  `Type=oneshot` waits for successful command exit; `RemainAfterExit=yes` records
  successful activation without a resident process.

Preconditions: reachable SSH, passwordless sudo, no running containers, existing
boost=0 and sixteen maximum-frequency caps of 3801000 kHz. Validate through the
real systemd manager and CPUFreq sysfs interface, not a mocked filesystem.

Executed after copying the source bytes to the installed path (refusing an
existing unit):

```sh
sudo chmod 644 /etc/systemd/system/b70-cpu-boost-off.service
sudo systemd-analyze verify /etc/systemd/system/b70-cpu-boost-off.service
sudo systemctl daemon-reload
sudo systemctl enable --now b70-cpu-boost-off.service
sudo systemctl restart b70-cpu-boost-off.service
systemctl is-enabled b70-cpu-boost-off.service
systemctl show b70-cpu-boost-off.service -p ActiveState -p SubState -p Result -p ExecMainStatus
cat /sys/devices/system/cpu/cpufreq/boost
cat /sys/devices/system/cpu/cpufreq/policy*/scaling_max_freq | sort | uniq -c
cat /proc/sys/kernel/random/boot_id
sha256sum /home/mike/inference/launchers/start-qwen38.sh
```

Observed: verification exit 0; enabled; `ActiveState=active`, `SubState=exited`,
`Result=success`, `ExecMainStatus=0`; boost 0; `16 3801000` caps.
Boot ID remained `c93aea98-bb22-4cf6-879f-42d2980cb726`.
Launcher SHA256 remained
`63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.

**No reboot was performed.** This verifies enabled configuration and actual
service execution, not successful application during a new boot. Re-run the
readback commands after the next planned reboot. The service fails visibly if
the CPUFreq interface is missing; it does not silently skip the mitigation.
Stopping it deliberately does not re-enable boost.

## Rollback

Remove only this owned unit and symlink, retaining current runtime boost-off:

```sh
sudo systemctl disable --now b70-cpu-boost-off.service
sudo rm /etc/systemd/system/b70-cpu-boost-off.service
sudo systemctl daemon-reload
```

This removes boot persistence; it does not write boost=1 or restore the previous
4968237 kHz runtime caps. Re-enabling boost is a separate decision given the
unresolved historical CPU MCA failures. No reboot/re-enable is part of rollback.

Lab native preview failed with `ENOENT` for its workspace socket. Installation
used ordinary SSH and the existing user authorization; no substitute Lab service
was created.
