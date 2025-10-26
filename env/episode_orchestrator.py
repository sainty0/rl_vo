# env/episode_orchestrator.py
import os
import uuid
import json
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
from typing import Dict, List, Optional, Callable, IO

from .rosbridge_client import RosbridgeClient

class EpisodeOrchestrator:
    """
    Orchestrates a long-lived ROS core + roslaunch stack and short-lived episode
    actors (fileplayer + odom_to_tum). Core stays up across episodes.

    Lifecycle:
      begin_episode()  -> ensure core up, reset metrics, start fileplayer
      end_episode()    -> stop fileplayer + odom_to_tum only (core stays)
      close_all()      -> stop everything (core + episode actors)
    """

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
            lio_launch_cmd: Optional[List[str]] = None,
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

        # Core
        self._ros_proc: Optional[subprocess.Popen] = None
        # Episode actors
        self._odom_proc: Optional[subprocess.Popen] = None
        self._player_proc: Optional[subprocess.Popen] = None
        self._lio_proc: Optional[subprocess.Popen] = None

        self._tee_threads: List[threading.Thread] = []
        self._rb: Optional[RosbridgeClient] = None

        # Externals (env overrides)
        self._odom_to_tum_cmd = os.environ.get("ODOM_TO_TUM_CMD", "/odom_to_tum.py")
        self._file_player_cmd = os.environ.get("FILE_PLAYER_CMD", "rosrun file_player file_player_headless")
        self._metrics_out = os.environ.get("METRICS_OUT_FILE", "/tmp/rlvo/metrics.json")
        # Optional override for LIO-SAM stack launch (list of argv), or via env LIO_SAM_LAUNCH_CMD
        # Default to local lio_stack.launch so preflight reports a sensible status.
        self._lio_launch_cmd = (
            list(lio_launch_cmd)
            if lio_launch_cmd is not None
            else (shlex.split(os.environ.get("LIO_SAM_LAUNCH_CMD", "")) or ["roslaunch", "/rl_vo/launch/lio_stack.launch"])
        )

        self._dry_run = (os.environ.get("DRY_RUN", "0") == "1")
        self._ros_down = False

        # Logging
        self._logger = logging.getLogger("EpisodeOrchestrator")
        self._logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        if not any(isinstance(h, logging.StreamHandler) for h in self._logger.handlers):
            self._logger.addHandler(ch)

        # Book-keeping
        self._stream_run_dir: Optional[str] = None
        self._t_stream_start: Optional[float] = None
        self._last_commit_ts: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def begin_episode(self, seq_name: str, start_percent: float, safe_leaf: float) -> Dict[str, object]:
        seq_dir = os.path.join(self.seq_root, seq_name)
        if self._stream_run_dir is None:
            run_dir = self._new_run_dir(tag="stream")
            self._attach_file_logger(run_dir)
            self._log_system_context()
            self._stream_run_dir = run_dir
        run_dir = self._stream_run_dir

        if self._dry_run:
            self._logger.info("[begin] DRY_RUN seq=%s start=%.3f", seq_name, start_percent)
            self._ensure_parent_dir(self.est_out)
            self._write_synthetic_tum(self.est_out, n=int(2 * self.rate_hz))
            self._t_stream_start = time.time()
            self._ros_down = False
            return {"run_dir": run_dir, "pids": {}, "est_tum": self.est_out}

        # 1) Ensure core (roslaunch + rosbridge + metrics + LIO-SAM) is alive/reused
        if not self.is_ros_ok():
            self._logger.info("[core] starting roslaunch core…")
            self._preflight_checks()
            self._ros_proc = self._pg_spawn(self.roslaunch_cmd, run_dir, name="roslaunch")
            self._logger.info("[core] roslaunch pid=%d", self._ros_proc.pid)
            self._wait_for_rosbridge_port(run_dir)
            self._rb = RosbridgeClient(self.rosbridge_url, lazy=True)
        else:
            if self._rb is None:
                self._rb = RosbridgeClient(self.rosbridge_url, lazy=True)

        # 2) Restart LIO stack every episode
        try:
            self._kill_pg(self._lio_proc)
        except Exception:
            pass
        self._lio_proc = None
        lio_cmd = self._lio_launch_cmd or ["roslaunch", "/rl_vo/launch/lio_stack.launch"]
        self._lio_proc = self._pg_spawn(lio_cmd, run_dir, name="lio_stack")
        self._logger.info("[begin] lio_stack pid=%d", self._lio_proc.pid if self._lio_proc else -1)

        # 3) (Re)start odom_to_tum for this episode
        self._ensure_parent_dir(self.est_out)
        self._kill_pg(self._odom_proc)  # safety if stale
        odom_cmd = f'{self._odom_to_tum_cmd} --topic {shlex.quote(self.odom_topic)} --out {shlex.quote(self.est_out)}'
        self._odom_proc = self._pg_spawn(shlex.split(odom_cmd), run_dir, name="odom_to_tum")
        self._logger.info("[begin] odom_to_tum pid=%d -> %s", self._odom_proc.pid, self.est_out)

        # 3) Reset metrics + publish safe param
        try:
            self._rb.call_service("/rl_metrics/reset", {})
        except Exception as e:
            self._logger.warning("[begin] metrics reset failed: %s", e)
        try:
            self._rb.publish_float("/lio_sam/params/mapping_surf_leaf_size", float(safe_leaf))
        except Exception as e:
            self._logger.warning("[begin] publish safe leaf failed: %s", e)

        # 4) Start file player (fresh every episode)
        self._kill_pg(self._player_proc)  # safety if stale
        player_cmd = f'{self._file_player_cmd} --dir {shlex.quote(seq_dir)} --rate {self.rate_hz} --start-percent {float(start_percent):.3f}'
        self._player_proc = self._pg_spawn(shlex.split(player_cmd), run_dir, name="file_player")
        self._logger.info("[begin] file_player pid=%d", self._player_proc.pid)

        # Optional readiness gate on metrics file (non-fatal)
        ready = self._wait_for_metrics_ready(timeout_s=8.0)
        if not ready:
            self._logger.warning("[begin] metrics readiness timed out; proceeding anyway")

        self._t_stream_start = time.time()
        return {
            "run_dir": run_dir,
            "pids": {
                "roslaunch-core": self._ros_proc.pid if self._ros_proc else None,
                "roslaunch": self._ros_proc.pid if self._ros_proc else None,
                "lio_stack": self._lio_proc.pid if self._lio_proc else None,
                "odom_to_tum": self._odom_proc.pid if self._odom_proc else None,
                "file_player": self._player_proc.pid if self._player_proc else None,
            },
            "est_tum": self.est_out,
        }

    def set_leaf(self, leaf_topic: str, leaf: float):
        if self._dry_run:
            return
        if self._rb is None:
            self._rb = RosbridgeClient(self.rosbridge_url, lazy=True)
        try:
            self._rb.publish_float(leaf_topic, float(leaf))
        except Exception as e:
            self._ros_down = True
            self._logger.warning("[set_leaf] publish failed: %s", e)

    def tick(self, step_len_s: float):
        t0 = time.time()
        while (time.time() - t0) < step_len_s:
            if not self.is_player_alive() or not self.is_ros_ok():
                break
            time.sleep(0.02)

    def commit_metrics(self) -> Dict[str, float]:
        if self._dry_run:
            rng = random.Random(time.time())
            return {
                "pts_per_scan_mean": rng.uniform(3e4, 8e4),
                "pts_per_scan_std": rng.uniform(5e2, 3e3),
                "scan_rate_hz": float(self.rate_hz),
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

        try:
            if self._rb is None:
                self._rb = RosbridgeClient(self.rosbridge_url, lazy=True)
            self._rb.call_service("/rl_metrics/commit", {})
        except Exception as e:
            self._ros_down = True
            self._logger.warning("[metrics] commit failed: %s", e)
            return {}

        try:
            with open(self._metrics_out, "r") as f:
                stats = json.load(f)
            self._last_commit_ts = time.time()
            return stats
        except Exception as e:
            self._logger.warning("[metrics] read failed (%s): %s", self._metrics_out, e)
            return {}

    # Episode ends: stop only the short-lived actors
    def end_episode(self):
        self._kill_pg(self._player_proc)
        self._player_proc = None
        self._kill_pg(self._odom_proc)
        self._odom_proc = None
        # keep core (roslaunch) running

    # Full shutdown (used by env.close())
    def close_all(self):
        self._kill_pg(self._player_proc); self._player_proc = None
        self._kill_pg(self._odom_proc);   self._odom_proc   = None
        self._kill_pg(self._lio_proc);    self._lio_proc    = None
        self._kill_pg(self._ros_proc);    self._ros_proc    = None
        self._stream_run_dir = None
        self._t_stream_start = None
        self._rb = None

    # ------------------------------------------------------------------ #
    # Status helpers
    # ------------------------------------------------------------------ #
    def is_ros_ok(self) -> bool:
        if self._dry_run:
            return True
        return (self._ros_proc is not None) and (self._ros_proc.poll() is None)

    def comms_alive(self) -> bool:
        return (not self._ros_down) and self.is_ros_ok()

    def is_player_alive(self) -> bool:
        if self._dry_run:
            return True
        if not self.is_ros_ok():
            return False
        if self._player_proc is None:
            return False
        return self._player_proc.poll() is None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _new_run_dir(self, tag: str) -> str:
        rid = f"{tag}-{uuid.uuid4().hex[:8]}"
        d = os.path.join(self.run_root, rid)
        os.makedirs(d, exist_ok=True)
        return d

    def _attach_file_logger(self, run_dir: str):
        fh_path = os.path.join(run_dir, "orchestrator.log")
        fh = logging.FileHandler(fh_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        self._logger.handlers = [h for h in self._logger.handlers if not isinstance(h, logging.FileHandler)]
        self._logger.addHandler(fh)
        self._logger.debug("[ctx] attached file logger at %s", fh_path)

    def _pg_spawn(self, cmd: List[str], cwd: Optional[str], name: str) -> subprocess.Popen:
        assert cwd, "cwd (run_dir) must be provided"
        log_path = os.path.join(cwd, f"{name}.log")
        self._logger.info("[spawn] %s: %s", name, shlex.join(cmd))
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        lf = open(log_path, "ab", buffering=0)
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # new process group
            env=env,
            bufsize=0,
        )
        t = threading.Thread(target=self._tee_stream, args=(proc.stdout, lf, name), daemon=True)
        t.start()
        self._tee_threads.append(t)
        return proc

    def _tee_stream(self, stream: IO[bytes], logfile: IO[bytes], name: str):
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
                    print(framed.decode("utf-8").rstrip("\n"))
                except Exception:
                    pass

    def _wait_for_rosbridge_port(self, run_dir: str):
        try:
            u = urlsplit(self.rosbridge_url)
            host = u.hostname or "localhost"
            port = u.port or (443 if u.scheme == "wss" else 80)
        except Exception:
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

    def _kill_pg(self, proc: Optional[subprocess.Popen], sig=signal.SIGTERM, wait_s: float = 3.0):
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
        EpisodeOrchestrator._ensure_parent_dir(path)
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

    def _wait_for_metrics_ready(self, timeout_s: float = 8.0) -> bool:
        """
        Poll metrics JSON until odom_rate_hz>0 and variable_tokens_n>0, or timeout.
        Returns True if ready else False.
        """
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                with open(self._metrics_out, "r") as f:
                    d = json.load(f)
                if float(d.get("odom_rate_hz", 0.0)) > 0.0 and int(d.get("variable_tokens_n", 0)) > 0:
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def _preflight_checks(self):
        def _which(x: str) -> Optional[str]:
            if " " in x:
                return shutil.which(x.split()[0])
            return shutil.which(x)
        checks = {
            "roslaunch": _which(self.roslaunch_cmd[0]),
            "odom_to_tum": shutil.which(self._odom_to_tum_cmd.split()[0]) if " " in self._odom_to_tum_cmd
            else (self._odom_to_tum_cmd if os.path.exists(self._odom_to_tum_cmd)
                  else shutil.which(self._odom_to_tum_cmd)),
            "rosrun": shutil.which("rosrun"),
            "file_player": _which(self._file_player_cmd),
            "lio_stack": _which(self._lio_launch_cmd[0]) if self._lio_launch_cmd else None,
        }
        for name, resolved in checks.items():
            self._logger.info("[preflight] %-12s -> %s", name, resolved or "<not found>")
            if name == "lio_stack" and self._lio_launch_cmd:
                # If using 'roslaunch <file.launch>', also verify the .launch file path
                launch_args = self._lio_launch_cmd[1:]
                launch_file = None
                if launch_args:
                    # roslaunch [<pkg> <file>] OR [<file>]; detect path-like .launch
                    if len(launch_args) == 1 and launch_args[0].endswith(".launch"):
                        launch_file = launch_args[0]
                    elif len(launch_args) >= 2 and launch_args[1].endswith(".launch"):
                        launch_file = launch_args[1]
                if launch_file:
                    self._logger.info("[preflight] %-12s file -> %s (exists=%s)", "lio_stack", launch_file, os.path.exists(launch_file))
            if resolved is None and name != "lio_stack":
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
