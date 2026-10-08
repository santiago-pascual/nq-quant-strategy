# Linux ARM64 deployment preparation (PAPER replay only)

This is a deployment checklist, not an authorization to route orders. The
checked-in service launches the deterministic replay source only. No realtime
MNQ market-data adapter or provider credentials are present, and the reviewed
CME snapshot covers October 8--31, 2026 only.

## ARM64 compatibility evidence and blockers

The pinned versions in `requirements-realtime-paper-py313.lock` have CPython
3.13 ARM64 Linux wheels for NumPy, pandas, SciPy, scikit-learn, hmmlearn,
PyArrow, and XGBoost on PyPI. The pure-Python runtime dependencies use
architecture-independent wheels. Select an Oracle Linux image with glibc
2.28 or newer; the PyArrow/scikit-learn ARM64 wheels use the manylinux glibc
2.28 baseline. XGBoost's AArch64 wheel may require the OS `libgomp` runtime.
This is wheel-availability evidence only; no Linux ARM64 installation or
runtime test has been performed.

Oracle Linux 10 documents Python 3.12 as its default; Oracle Linux 9's
documented standard is Python 3.9 with optional Python 3.11/3.12. The lock was
created with CPython 3.13.15. Build that exact interpreter from the official
source tarball and verify its published SHA-256 before proceeding; do not
silently switch the runtime to 3.12. Python.org publishes the tarball and
checksum on the [3.13.15 release page](https://www.python.org/downloads/release/python-31315/).
The lock pins versions but does not include artifact hashes and is not a
platform-specific lock file. Verify the resolved environment with `pip check`
and the import/smoke checks below on the actual ARM64 VM before calling it
reproducible.

## Create an Always Free A1 VM in Oracle Cloud Console

These are manual instructions. No Oracle account has been accessed and no VM
has been created.

1. Sign in to Oracle Cloud Console and select the tenancy's **Home region**.
   Open **Compute → Instances → Create instance**.
2. Name it (for example, `mnq-paper-pilot`) and select a Linux **ARM64/AArch64**
   image. Verify the image architecture before continuing.
3. In **Shape**, select **Ampere → VM.Standard.A1.Flex**. Set **2 OCPUs** and
   **12 GB memory**. Confirm the shape is marked **Always Free eligible** in the
   final eligibility/cost panel. Do not select a paid shape or exceed remaining
   free quotas.
4. Keep the boot volume at its 50 GB default. Do not add paid block storage.
   Oracle's home-region Always Free block-volume pool is 200 GB combined.
5. Add only your SSH public key. If a public IP is needed, restrict VCN ingress
   TCP/22 to your current IP. Do not open a monitoring API port. If the region
   reports A1 capacity unavailable, stop; do not switch to a paid shape/region.
6. Review the final cost/eligibility summary and create only when it shows
   Always Free resources.

Always Free A1 is a home-region allowance, not a guarantee of capacity or
durability. Oracle documents possible idle-instance reclamation; treat a VM as
a test host until recovery and backups are exercised.

## Install on an ARM64 VM

1. Use a dedicated non-root account and durable local block storage. Do not
   place SQLite/WAL/checkpoints on NFS or another network filesystem.
2. Build CPython 3.13.15 from the official source tarball. The source SHA-256
   below is published by Python.org; this deliberately pins the same runtime
   patch observed on Windows:

   ```sh
   sudo dnf install -y gcc make xz openssl-devel bzip2-devel libffi-devel \
     zlib-devel xz-devel readline-devel sqlite-devel ncurses-devel gdbm-devel \
     libuuid-devel
   mkdir -p "$HOME/src"
   sudo mkdir -p /opt/cpython
   cd "$HOME/src"
   curl -fLO https://www.python.org/ftp/python/3.13.15/Python-3.13.15.tar.xz
   echo '1e66a7945a48390ee4c2a4268a0e4185884059a13c4aab6d148aa208deea4a76  Python-3.13.15.tar.xz' | sha256sum -c -
   tar -xf Python-3.13.15.tar.xz
   cd Python-3.13.15
   ./configure --prefix=/opt/cpython/3.13.15 --with-ensurepip=install
   make -j2
   sudo make install
   /opt/cpython/3.13.15/bin/python3.13 --version
   ```

3. Clone the public repository and record the exact source revision:

   ```sh
   sudo mkdir -p /opt/mnq-paper
   sudo chown "$USER":"$USER" /opt/mnq-paper
   git clone https://github.com/santiago-pascual/nq-quant-strategy.git /opt/mnq-paper/app
   cd /opt/mnq-paper/app
   git rev-parse HEAD
   ```

   If authentication is required, use a read-only deploy key on the VM; never
   put a token in the clone URL or shell history. Raw market data is not in Git
   and must be transferred separately to the configured data location. Do not
   copy credentials, Windows database/WAL files, checkpoints, logs, or result
   output into the source checkout.

   To transfer a raw data file from Windows, use key-based SSH and a staging
   path, then move it with restricted ownership on the VM:

   ```powershell
   scp -i $env:USERPROFILE\.ssh\mnq_paper_ed25519 `
     C:\path\to\mnq_raw_data.parquet `
     opc@VM_PUBLIC_IP:/var/tmp/mnq_raw_data.parquet
   ```

4. Install the platform runtime dependency (including `libgomp` for XGBoost if
   required), create the virtual environment and install the pinned packages.
   `--only-binary` prevents an unnoticed C/C++ source build:

   ```sh
   cd /opt/mnq-paper/app
   /opt/cpython/3.13.15/bin/python3.13 -m venv .venv
   # Keep ensurepip's pip version bundled with the hash-verified CPython source.
   .venv/bin/python -m pip --version
   .venv/bin/python -m pip install --only-binary=:all: -r requirements-realtime-paper-py313.lock
   .venv/bin/python -m pip check
   .venv/bin/python -c 'import platform; print(platform.platform(), platform.machine()); import hmmlearn, numpy, pandas, pyarrow, scipy, sklearn, xgboost; print(numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__, hmmlearn.__version__, pyarrow.__version__, xgboost.__version__)'
   ```

5. Create the service account and persistent directories; install the example
   environment and systemd unit. Edit the environment file to select a data
   interval that exists locally. The checked-in example data is not included,
   and the reviewed calendar covers October 8--31, 2026 only:

   ```sh
   sudo useradd --system --home-dir /var/lib/mnq-paper --shell /sbin/nologin mnqpaper
   sudo install -d -o mnqpaper -g mnqpaper -m 0700 /var/lib/mnq-paper /var/lib/mnq-paper-backups
   sudo install -d -o root -g mnqpaper -m 0750 /etc/mnq-paper
   sudo install -o root -g mnqpaper -m 0640 deploy/systemd/paper.env.example /etc/mnq-paper/paper.env
   sudo install -o root -g root -m 0644 deploy/systemd/mnq-paper-replay.service.example /etc/systemd/system/mnq-paper-replay.service
   ```

6. Start and inspect the PAPER replay service:

   ```sh
   sudo systemctl daemon-reload
   sudo systemctl enable --now mnq-paper-replay.service
   sudo systemctl status mnq-paper-replay.service
   journalctl -u mnq-paper-replay.service -f
   ```

### Bounded smoke and resource capture

Before enabling the service, use an empty disposable output directory and a
short interval present in the local data. `/usr/bin/time -v` reports peak RSS;
`pidstat -ru 1` or `systemd-cgtop` samples CPU and memory. Save measurements
outside the source tree. Do not run a multi-year pseudo-live just to benchmark
a refit.

```sh
mkdir -p /var/lib/mnq-paper-smoke
/usr/bin/time -v .venv/bin/python -m src.paper.run_realtime_paper \
  --command run --mode PAPER --replay-start 2026-10-08T13:30:00Z \
  --replay-end 2026-10-08T14:00:00Z \
  --output-dir /var/lib/mnq-paper-smoke \
  --cost-config src/paper/config/topstepx_mnq_fees_2026-07.json
```

The example only works if the data contains that interval and the installed
calendar covers every evaluated date.

### Recovery, backup and private monitoring

Verify resume in the disposable smoke directory with the same command and
`--resume`; check that the last processed timestamp is unchanged for already
committed bars and no duplicate bar/trade/state-transition records appear. A
crash test must also use a disposable output directory. Do not interrupt an
active production process to test recovery.

Back up SQLite online and retain the matching checkpoint, event/refit ledgers,
and cost/calendar files:

```sh
.venv/bin/python -m src.paper.run_realtime_paper --command backup-database \
  --output-dir /var/lib/mnq-paper \
  --backup-path /var/lib/mnq-paper-backups/analytics-$(date -u +%Y%m%dT%H%M%SZ).sqlite3
```

The monitoring API is read-only. If started, bind it to `127.0.0.1` and access
it through an SSH tunnel. Do not add a public API ingress rule.

The systemd unit stores state under `/var/lib/mnq-paper`, restarts failures,
and sends READY/WATCHDOG/STOPPING notifications. `TimeoutStopSec=900` allows
the observed 590-second MR refit to finish on the development machine. The
wrapper resumes only from a checkpoint and fails closed if existing state has
none. `paper_writer.lock` only protects one output directory on one host; it
does not fence a second host or another directory.

## Health, storage, backup and logs

Inspect health with:

```sh
python -m src.paper.run_realtime_paper --command status --output-dir /var/lib/mnq-paper
python -m src.paper.run_realtime_paper --command analytics --output-dir /var/lib/mnq-paper
systemctl show mnq-paper-replay.service -p ActiveState -p SubState -p NRestarts
```

The runner writes `status.json`, SQLite analytics/WAL, checksummed checkpoints,
`events.jsonl`, and `hmm_refits.jsonl`. Use SQLite's online backup command;
also back up the matching checkpoint, event/refit ledgers, and cost/calendar
profiles together:

```sh
python -m src.paper.run_realtime_paper --command backup-database \
  --output-dir /var/lib/mnq-paper \
  --backup-path /var/lib/mnq-paper-backups/analytics-$(date -u +%Y%m%dT%H%M%SZ).sqlite3
```

Standard output/error go to journald. Optional host-wide retention settings are
in `deploy/systemd/journald-mnq-paper.conf.example`; review before installing
because journald retention is system-wide. Keep the Paper JSONL audit files
append-only while the process runs. Archive a completed run as a whole after a
clean stop; do not `copytruncate` or rotate its JSONL while active.

The read-only monitoring API has no order routes and binds to loopback by
default. Do not expose it publicly. No API service is enabled by this replay
unit.

## Windows fallback

The local deterministic replay fallback is documented in
`src/paper/REALTIME_PAPER.md`. Use the same CPython 3.13.15 environment and
dependency file, start with `scripts/start_paper_replay.ps1`, request a graceful
stop with the Paper CLI or Ctrl+C, and use `-AutoResume` for a Task Scheduler
restart action. Task Scheduler must be configured for one instance only and a
bounded retry count. Monitor `status.json` plus the Python process working set
in Task Manager or `Get-Process`; no Windows service manager is installed by
this repository.

## Free-tier / readiness limits

Oracle's current Always Free resources documentation describes an A1 allowance
equivalent to 2 OCPUs and 12 GB RAM for a continuously allocated VM, and warns
that Always Free capacity can be unavailable in a home region. Oracle also
documents idle-instance reclamation criteria. A low-activity Paper process
could meet those criteria; do not count the VM as a durable trading host
without a tested recovery/backup plan. The measured Windows replay and
historical HMM fit times are not ARM performance measurements. No VM has been
provisioned, no Linux tests have run, and the actual live-feed adapter is
missing. This package is therefore not ready for unattended Paper operation.
