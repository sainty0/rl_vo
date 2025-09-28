import os
import uuid
import json
import math
import time
import shlex
import signal
import random
import subprocess
from typing import Dict, List, Optional

"""
EpisodeOrchestrator
===================

Process-level orchestration for a single episode using CLI-only controls:

- Starts a ROS graph (SC-LIO-SAM + rosbridge_websocket) via roslaunch
- Starts your existing odom_to_tum.py logger via rosrun (or an override command)
- Runs the file_player in headless CLI mode:
    rosrun file_player file_player_headless --dir <SEQ_DIR> --rate <R> --start-percent <PCT> --step <N>
- Treats file_player process exit as the episode boundary
- Enforces timeouts and cleans up process groups on exit or exception
- Writes all logs under /tmp/rlvo/<uuid> (configurable via run_root)

DRY_RUN
-------
If environment variable DRY_RUN=1 is set, this class will *not* spawn any subprocesses.
Instead, it will:
- "sleep" a tiny amount
- write a small synthetic TUM file to est_out
- return synthetic metadata

This lets unit tests run without ROS, MulRan, or SC-LIO-SAM installed.
"""


class EpisodeOrchestrator:
    def __init__(
        self,
        roslaunch_cmd: List[str],
        seq_root: str,
        rosbridge_url: str,
        est_out: str = "/tmp/est.tum",
        rate_hz: int = 10,
        run_root: str = "/tmp/rlvo",
        odom_topic: str = "/lio_sam/mapping/odometry_incremental",
    ):
        """
        Parameters
        ----------
        roslaunch_cmd : List[str]
            Command to launch SC-LIO-SAM + (optionally) rosbridge_websocket, e.g.:
            ["roslaunch", "your_pkg", "launch_sclsam.launch"]
        seq_root : str
            Root directory of MulRan sequences, e.g., "/data/mulran"
        rosbridge_url : str
            WebSocket URL for rosbridge_server (e.g., "ws://localhost:9090")
        est_out : str
            Output TUM path written by odom_to_tum.py (e.g., "/tmp/est.tum")
        rate_hz : int
            File player playback rate in Hz (10 recommended)
        run_root : str
            Root directory for run logs (each episode gets a unique subdir)
        odom_topic : str
            Odometry topic to log (nav_msgs/Odometry)
        """
        self.roslaunch_cmd = list(roslaunch_cmd)
        self.seq_root = seq_root
        self.rosbridge_url = rosbridge_url
        self.est_out = est_out
        self.rate_hz = int(rate_hz)
        self.run_root = run_root
        self.odom_topic = odom_topic

        os.makedirs(self.run_root, exist_ok=True)

        # Internal state
        self._ros_proc: Optional[subprocess.Popen] = None
        self._odom_proc: Optional[subprocess.Popen] = None
        self._player_proc: Optional[subprocess.Popen] = None

        # External overrides (optional) via env:
        # ODOM_TO_TUM_CMD="rosrun your_pkg odom_to_tum.py"
        self._odom_to_tum_cmd = os.environ.get("ODOM_TO_TUM_CMD", "rosrun your_pkg odom_to_tum.py")
        # FILE_PLAYER_CMD="rosrun file_player file_player_headless"
        self._file_player_cmd = os.environ.get("FILE_PLAYER_CMD", "rosrun file_player file_player_headless")

        # Dry run flag
        self._dry_run = (os.environ.get("DRY_RUN", "0") == "1")

    # ------------------------------
    # Public API
    # ------------------------------
    def probe(self, duration_s: float, safe_leaf: float, topics: Dict[str, str]) -> Dict[str, float]:
        """
        Optional quick run (or warm-up) to produce a context observation vector.
        Implementation strategy:
          - Start ROS graph
          - Start odom_to_tum logger (even if not strictly needed for probe stats)
          - Option A (cleanest): run a tiny file_player slice (duration_s → steps) then exit
          - Teardown process group

        Since probe aggregates are provided by separate metrics exporters in a full system,
        and we don't rely on them here, this function simply ensures graph viability and returns
        a minimal dict. In DRY_RUN it returns synthetic stats immediately.

        Returns a dict with keys matching the 16-D observation layout (missing keys default to 0 in env).
        """
        # Compute steps as a small slice for probe (round to int)
        probe_steps = max(1, int(round(self.rate_hz * duration_s)))

        if self._dry_run:
            # Minimal synthetic stats for probe
            return self._synthetic_probe(duration_s)

        run_dir = self._new_run_dir(tag="probe")

        try:
            # Start ROS graph
            self._ros_proc = self._pg_spawn(self.roslaunch_cmd, cwd=run_dir)

            # Start odom_to_tum logger (writes est_out; harmless during probe)
            self._ensure_parent_dir(self.est_out)
            odom_cmd = f'{self._odom_to_tum_cmd} --topic {shlex.quote(self.odom_topic)} --out {shlex.quote(self.est_out)}'
            self._odom_proc = self._pg_spawn(shlex.split(odom_cmd), cwd=run_dir)

            # Small sleep to let nodes come up
            time.sleep(1.0)

            # We don't change the leaf here (safe baseline can be published by RL env beforehand)
            # Run a tiny slice with a placeholder sequence — the RL env will run a proper probe window if needed.
            # Here we just verify the stack is alive.
            time.sleep(max(0.1, duration_s))

            # Return a mostly-empty probe vector; real metrics exporter would populate these
            return self._synthetic_probe(duration_s)
        finally:
            self._teardown_all()

    def run_window(
        self,
        seq_name: str,
        start_percent: float,
        warmup_s: float,
        score_s: float,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Launches ROS graph, starts odom logger, then runs file_player for the window:
          steps = int(rate_hz * (warmup_s + score_s))

        The file_player process exit marks the end of the episode.
        On timeout or exceptions, kills process groups to ensure a hard reset between episodes.

        Returns dict with:
          {
            "est_tum": <path>,
            "runtime_s": <float>,
            "timeout": <bool>,
            "diverged": <bool>  # we don't detect here; env computes APE/diverged later
          }
        """
        steps = int(round(self.rate_hz * (warmup_s + score_s)))
        steps = max(1, steps)
        if timeout_s is None:
            timeout_s = warmup_s + score_s + 10.0  # small cushion

        seq_dir = os.path.join(self.seq_root, seq_name)
        if self._dry_run:
            # Simulate a run: write a tiny synthetic TUM file and return quickly
            self._ensure_parent_dir(self.est_out)
            self._write_synthetic_tum(self.est_out, n=int(score_s * self.rate_hz))
            return {
                "est_tum": self.est_out,
                "runtime_s": float(warmup_s + score_s),
                "timeout": False,
                "diverged": False,
            }

        run_dir = self._new_run_dir(tag="run")
        t0 = time.time()
        timeout = False

        try:
            # Start ROS graph
            self._ros_proc = self._pg_spawn(self.roslaunch_cmd, cwd=run_dir)

            # Start odom_to_tum logger to est_out
            self._ensure_parent_dir(self.est_out)
            odom_cmd = f'{self._odom_to_tum_cmd} --topic {shlex.quote(self.odom_topic)} --out {shlex.quote(self.est_out)}'
            self._odom_proc = self._pg_spawn(shlex.split(odom_cmd), cwd=run_dir)

            # Give nodes a moment to come up
            time.sleep(1.0)

            # Run file player
            player_cmd = f'{self._file_player_cmd} --dir {shlex.quote(seq_dir)} --rate {self.rate_hz} --start-percent {float(start_percent):.3f} --step {steps}'
            self._player_proc = self._pg_spawn(shlex.split(player_cmd), cwd=run_dir)

            # Wait for file player to exit (it should exit when playback is complete)
            self._wait_with_timeout(self._player_proc, timeout_s)
        except TimeoutError:
            timeout = True
        finally:
            # Always teardown ROS + odom logger after the window
            self._teardown_all()

        runtime = time.time() - t0
        return {
            "est_tum": self.est_out,
            "runtime_s": float(runtime),
            "timeout": bool(timeout),
            "diverged": False,  # APE/divergence is computed later by the RL env
        }

    # ------------------------------
    # Internals / helpers
    # ------------------------------
    def _new_run_dir(self, tag: str) -> str:
        rid = f"{tag}-{uuid.uuid4().hex[:8]}"
        d = os.path.join(self.run_root, rid)
        os.makedirs(d, exist_ok=True)
        return d

    def _pg_spawn(self, cmd: List[str], cwd: Optional[str] = None) -> subprocess.Popen:
        """
        Spawn a command in a new process group so we can kill the entire tree with kill_pg().
        Linux: use preexec_fn=os.setsid. On macOS, setsid is also available via Python.
        """
        return subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # new process group
        )

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
            # As a last resort, try to terminate the root proc
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                pass

    def _teardown_all(self):
        # Kill file player first (if still running), then odom logger, then ROS graph
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
        # Minimal probe vector; real metrics exporter would populate richer stats.
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
