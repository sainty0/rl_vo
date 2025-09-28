# env/episode_orchestrator.py
import os
import uuid
import json
import math
import socket
import time
import shlex
import signal
import random
import subprocess
import threading
import logging
import shutil
from urllib.parse import urlsplit
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple, IO

"""
EpisodeOrchestrator
===================

Adds:
- Live tee of child stdout/stderr to console + files (with timestamps)
- Preflight command checks and environment snapshot
- rosbridge port wait based on rosbridge_url
- Clear return metadata (run_dir, PIDs, exit codes)
- Tail of recent child logs on failure/timeout
"""

class EpisodeOrchestrator:
    def __init__(
        self,
        roslaunch_cmd: List[str],
        seq_root: str,
        rosbridge_url: str,
        est_out: str = "/tmp/est.tum",
        rate_hz: int = 1,
        run_root: str = "/tmp/rlvo",
        odom_topic: str = "/lio_sam/mapping/odometry_incremental",
        stream_to_stdout: bool = True,
        tail_lines_on_fail: int = 200,
    ):
        self.roslaunch_cmd = list(roslaunch_cmd)
        self.seq_root = seq_root
        self.rosbridge_url = rosbridge_url
        self.est_out = est_out
        self.rate_hz = int(rate_hz)
        self.run_root = run_root
        self.odom_topic = odom_topic
        self.stream_to_stdout = stream_to_stdout
        self.tail_lines_on_fail = tail_lines_on_fail

        os.makedirs(self.run_root, exist_ok=True)

        self._ros_proc: Optional[subprocess.Popen] = None
        self._odom_proc: Optional[subprocess.Popen] = None
        self._player_proc: Optional[subprocess.Popen] = None
        self._tee_threads: List[threading.Thread] = []

        # External overrides (optional) via env:
        self._odom_to_tum_cmd = os.environ.get("ODOM_TO_TUM_CMD", "/odom_to_tum.py")
        self._file_player_cmd = os.environ.get("FILE_PLAYER_CMD", "rosrun file_player file_player_headless")

        # Dry run flag
        self._dry_run = (os.environ.get("DRY_RUN", "0") == "1")

        # Logger
        self._logger = logging.getLogger("EpisodeOrchestrator")
        self._logger.setLevel(logging.DEBUG)
        # Console handler (info+)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        if not any(isinstance(h, logging.StreamHandler) for h in self._logger.handlers):
            self._logger.addHandler(ch)

    # ------------------------------
    # Public API
    # ------------------------------
    def probe(self, duration_s: float, safe_leaf: float, topics: Dict[str, str]) -> Dict[str, float]:
        probe_steps = max(1, int(round(self.rate_hz * duration_s)))
        if self._dry_run:
            self._logger.info("[probe] DRY_RUN=1 → returning synthetic probe stats")
            return self._synthetic_probe(duration_s)

        run_dir = self._new_run_dir(tag="probe")
        self._attach_file_logger(run_dir)

        self._logger.info("[probe] run_dir=%s steps=%d duration_s=%.3f", run_dir, probe_steps, duration_s)
        self._log_system_context()

        try:
            self._preflight_checks()
            self._ros_proc = self._pg_spawn(self.roslaunch_cmd, run_dir, name="roslaunch")
            self._logger.info("[probe] roslaunch pid=%d", self._ros_proc.pid)

            self._ensure_parent_dir(self.est_out)
            odom_cmd = f'{self._odom_to_tum_cmd} --topic {shlex.quote(self.odom_topic)} --out {shlex.quote(self.est_out)}'
            self._odom_proc = self._pg_spawn(shlex.split(odom_cmd), run_dir, name="odom_to_tum")
            self._logger.info("[probe] odom_to_tum pid=%d -> %s", self._odom_proc.pid, self.est_out)

            self._wait_for_rosbridge_port(run_dir)

            time.sleep(max(0.2, duration_s))
            self._logger.info("[probe] done; returning synthetic stats placeholder")
            return self._synthetic_probe(duration_s)

        except Exception as e:
            self._logger.exception("[probe] FAILED: %s", e)
            self._tail_child_logs(run_dir)
            raise
        finally:
            self._teardown_all()

    def run_window(
        self,
        seq_name: str,
        start_percent: float,
        warmup_s: float,
        score_s: float,
        timeout_s: Optional[float] = None,
        before_play_fn: Optional[Callable] = None,
    ) -> Dict[str, object]:
        # steps = int(round(self.rate_hz * (warmup_s + score_s)))
        # steps = max(1, steps)
        steps = 30000
        if timeout_s is None:
            timeout_s = warmup_s + score_s + 10.0

        seq_dir = os.path.join(self.seq_root, seq_name)

        if self._dry_run:
            run_dir = self._new_run_dir(tag="dryrun")
            self._attach_file_logger(run_dir)
            self._logger.info("[run] DRY_RUN=1 seq=%s start=%.3f warmup=%.2f score=%.2f steps=%d",
                              seq_name, start_percent, warmup_s, score_s, steps)
            self._ensure_parent_dir(self.est_out)
            self._write_synthetic_tum(self.est_out, n=int(score_s * self.rate_hz))
            return {
                "est_tum": self.est_out,
                "runtime_s": float(warmup_s + score_s),
                "timeout": False,
                "diverged": False,
                "run_dir": run_dir,
                "pids": {},
                "exit_codes": {},
            }

        run_dir = self._new_run_dir(tag="run")
        self._attach_file_logger(run_dir)
        self._logger.info("[run] run_dir=%s seq=%s start=%.3f warmup=%.2f score=%.2f steps=%d timeout_s=%.2f",
                          run_dir, seq_name, start_percent, warmup_s, score_s, steps, timeout_s)
        self._log_system_context()

        t0 = time.time()
        timeout = False
        pids = {}
        exit_codes = {}

        try:
            self._preflight_checks()

            self._ros_proc = self._pg_spawn(self.roslaunch_cmd, run_dir, name="roslaunch")
            pids["roslaunch"] = self._ros_proc.pid
            self._logger.info("[run] roslaunch pid=%d", self._ros_proc.pid)

            self._ensure_parent_dir(self.est_out)
            odom_cmd = f'{self._odom_to_tum_cmd} --topic {shlex.quote(self.odom_topic)} --out {shlex.quote(self.est_out)}'
            self._odom_proc = self._pg_spawn(shlex.split(odom_cmd), run_dir, name="odom_to_tum")
            pids["odom_to_tum"] = self._odom_proc.pid
            self._logger.info("[run] odom_to_tum pid=%d -> %s", self._odom_proc.pid, self.est_out)

            self._wait_for_rosbridge_port(run_dir)

            if before_play_fn is not None:
                try:
                    self._logger.info("[run] before_play_fn() starting…")
                    before_play_fn()
                    self._logger.info("[run] before_play_fn() done")
                except Exception as e:
                    self._logger.warning("[run] before_play_fn raised: %s", e, exc_info=True)

            player_cmd = f'{self._file_player_cmd} --dir {shlex.quote(seq_dir)} --rate {self.rate_hz} --start-percent {float(start_percent):.3f} --step {steps}'
            self._player_proc = self._pg_spawn(shlex.split(player_cmd), run_dir, name="file_player")
            pids["file_player"] = self._player_proc.pid
            self._logger.info("[run] file_player pid=%d", self._player_proc.pid)

            self._wait_with_timeout(self._player_proc, timeout_s)

        except TimeoutError:
            timeout = True
            self._logger.error("[run] TIMEOUT after %.2fs waiting for file_player to finish", timeout_s)
            self._tail_child_logs(run_dir)
        except Exception as e:
            self._logger.exception("[run] FAILED: %s", e)
            self._tail_child_logs(run_dir)
            raise
        finally:
            # Record exit codes before teardown if possible
            for name, proc in (("file_player", self._player_proc),
                               ("odom_to_tum", self._odom_proc),
                               ("roslaunch", self._ros_proc)):
                if proc is not None:
                    exit_codes[name] = proc.poll()
            self._teardown_all()

        runtime = time.time() - t0
        self._logger.info("[run] finished in %.2fs timeout=%s", runtime, timeout)
        self._logger.info("[run] exit_codes=%s", exit_codes)

        return {
            "est_tum": self.est_out,
            "runtime_s": float(runtime),
            "timeout": bool(timeout),
            "diverged": False,
            "run_dir": run_dir,
            "pids": pids,
            "exit_codes": exit_codes,
        }

    # ------------------------------
    # Internals / helpers
    # ------------------------------
    def _new_run_dir(self, tag: str) -> str:
        rid = f"{tag}-{uuid.uuid4().hex[:8]}"
        d = os.path.join(self.run_root, rid)
        os.makedirs(d, exist_ok=True)
        return d

    def _attach_file_logger(self, run_dir: str):
        """Attach a file handler to the orchestrator logger for this run_dir."""
        fh_path = os.path.join(run_dir, "orchestrator.log")
        fh = logging.FileHandler(fh_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        # Avoid duplicate handlers across runs
        # Remove existing FileHandlers first
        self._logger.handlers = [h for h in self._logger.handlers if not isinstance(h, logging.FileHandler)]
        self._logger.addHandler(fh)
        self._logger.debug("[ctx] attached file logger at %s", fh_path)

    def _pg_spawn(self, cmd: List[str], cwd: Optional[str], name: str) -> subprocess.Popen:
        """Spawn a command in a new process group with live tee to <name>.log in cwd."""
        assert cwd, "cwd (run_dir) must be provided"
        log_path = os.path.join(cwd, f"{name}.log")
        self._logger.info("[spawn] %s: %s", name, shlex.join(cmd))

        # Unbuffer python children for real-time logs
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        # Open log file in binary append mode
        lf = open(log_path, "ab", buffering=0)

        # Start process with stdout=PIPE, stderr=STDOUT for tee
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # new process group
            env=env,
            bufsize=0,
        )

        # Tee thread to mirror child's stdout to file (and optionally console)
        t = threading.Thread(target=self._tee_stream, args=(proc.stdout, lf, name), daemon=True)
        t.start()
        self._tee_threads.append(t)
        return proc

    def _tee_stream(self, stream: IO[bytes], logfile: IO[bytes], name: str):
        """Prefix each line with timestamp and process name; write to file and maybe console."""
        while True:
            if stream is None:
                break
            line = stream.readline()
            if not line:
                break
            ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            framed = f"[{ts}] [{name}] ".encode("utf-8") + line
            try:
                logfile.write(framed)
            except Exception:
                pass
            if self.stream_to_stdout:
                try:
                    # Write to console without double newlines
                    print(framed.decode("utf-8").rstrip("\n"))
                except Exception:
                    pass

    def _wait_for_rosbridge_port(self, run_dir: str):
        """Parse ws/wss URL and wait for TCP port to accept connections."""
        try:
            u = urlsplit(self.rosbridge_url)
            host = u.hostname or "localhost"
            port = u.port or (443 if u.scheme == "wss" else 80)
        except Exception:
            # Fallback: best-effort parse "host:port"
            parts = self.rosbridge_url.rsplit(":", 1)
            host = parts[0]
            port = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 9090
        self._logger.info("[net] waiting for rosbridge %s:%d …", host, port)
        try:
            self._wait_for_port(host, port, timeout_s=15.0)
            self._logger.info("[net] rosbridge ready")
        except Exception as e:
            self._logger.error("[net] rosbridge NOT reachable: %s", e)
            raise

    def _wait_for_port(self, host: str, port: int, timeout_s: float = 10.0):
        t0 = time.time()
        last_err = None
        while time.time() - t0 < timeout_s:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return True
            except Exception as e:
                last_err = e
                time.sleep(0.25)
        raise RuntimeError(f"Timeout waiting for {host}:{port} ({last_err})")

    def kill_pg(self, proc: Optional[subprocess.Popen], sig=signal.SIGTERM, wait_s: float = 3.0):
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, sig)
            t0 = time.time()
            while proc.poll() is None and (time.time() - t0) < wait_s:
                time.sleep(0.05)
            if proc.poll() is None:
                os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                pass

    def _teardown_all(self):
        # Kill order: player → odom → ros
        self.kill_pg(self._player_proc)
        self._player_proc = None
        self.kill_pg(self._odom_proc)
        self._odom_proc = None
        self.kill_pg(self._ros_proc)
        self._ros_proc = None

    def _wait_with_timeout(self, proc: subprocess.Popen, timeout_s: float):
        t0 = time.time()
        while proc.poll() is None:
            if (time.time() - t0) > timeout_s:
                raise TimeoutError("file_player_headless timed out")
            time.sleep(0.05)

    @staticmethod
    def _ensure_parent_dir(path: str):
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def _write_synthetic_tum(path: str, n: int = 180, dt: float = 0.1, drift: float = 0.02):
        import numpy as np
        t = np.arange(n) * dt
        xyz = np.cumsum(np.random.randn(n, 3) * drift, axis=0)
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        arr = np.column_stack([t, xyz, quat])
        np.savetxt(path, arr, fmt="%.9f")

    @staticmethod
    def _synthetic_probe(duration_s: float) -> Dict[str, float]:
        rng = random.Random(42)
        return {
            "pts_per_scan_mean": rng.uniform(3e4, 8e4),
            "pts_per_scan_std": rng.uniform(5e2, 3e3),
            "scan_rate_hz": rng.uniform(8, 12),
            "surf_pts_mean": rng.uniform(2e4, 7e4),
            "surf_pts_std": rng.uniform(5e2, 2e3),
            "corner_pts_mean": rng.uniform(5e3, 2e4),
            "corner_pts_std": rng.uniform(3e2, 1e3),
            "odom_rate_hz": rng.uniform(10, 30),
            "pose_dropouts_s": rng.uniform(0.0, 0.2),
            "vel_norm_mean": rng.uniform(0.0, 5.0),
            "vel_norm_std": rng.uniform(0.0, 1.0),
            "acc_jolt_mean": rng.uniform(0.0, 10.0),
            "imu_ang_vel_rms": rng.uniform(0.0, 0.5),
            "imu_lin_acc_rms": rng.uniform(0.0, 1.0),
            "planarity_ratio_mean": rng.uniform(0.1, 0.9),
            "action_last": 0.0,
        }

    # ---------- Diagnostics ----------
    def _preflight_checks(self):
        """Resolve and log actual binaries; warn if missing/non-executable."""
        def _which(x: str) -> Optional[str]:
            if " " in x:
                # compound command like "rosrun file_player file_player_headless"
                return shutil.which(x.split()[0])
            return shutil.which(x)

        checks = {
            "roslaunch": _which(self.roslaunch_cmd[0]),
            "odom_to_tum": shutil.which(self._odom_to_tum_cmd.split()[0]) if " " in self._odom_to_tum_cmd else (self._odom_to_tum_cmd if os.path.exists(self._odom_to_tum_cmd) else shutil.which(self._odom_to_tum_cmd)),
            "rosrun": shutil.which("rosrun"),
            "file_player": _which(self._file_player_cmd),
        }
        for name, resolved in checks.items():
            self._logger.info("[preflight] %-12s -> %s", name, resolved or "<not found>")
            if resolved is None:
                self._logger.warning("[preflight] %s NOT found on PATH; launch may fail", name)

    def _log_system_context(self):
        ctx = {
            "cwd": os.getcwd(),
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "PATH": os.environ.get("PATH", ""),
            "ROS_MASTER_URI": os.environ.get("ROS_MASTER_URI", ""),
            "ROS_IP": os.environ.get("ROS_IP", ""),
            "ROS_PACKAGE_PATH": os.environ.get("ROS_PACKAGE_PATH", ""),
        }
        self._logger.debug("[ctx] %s", json.dumps(ctx, indent=2))

    def _tail_child_logs(self, run_dir: str):
        """Tail recent lines of child logs into orchestrator.log and console."""
        names = ["roslaunch.log", "odom_to_tum.log", "file_player.log"]
        for n in names:
            p = os.path.join(run_dir, n)
            if not os.path.exists(p):
                continue
            try:
                with open(p, "rb") as f:
                    lines = f.readlines()[-self.tail_lines_on_fail:]
                self._logger.error("----- tail %s (last %d lines) -----", n, self.tail_lines_on_fail)
                for b in lines:
                    try:
                        s = b.decode("utf-8", errors="replace").rstrip("\n")
                        print(s)
                    except Exception:
                        pass
            except Exception as e:
                self._logger.warning("[tail] could not read %s: %s", n, e)
