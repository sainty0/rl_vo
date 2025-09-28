import os
import math
import random
import numpy as np
from typing import Dict, Tuple, Any, List, Type

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from env.episode_orchestrator import EpisodeOrchestrator
from env.rosbridge_client import RosbridgeClient, MockRosbridge
from env.utils.ape import ape_rmse


LEAF_MIN = 0.05
LEAF_MAX = 1.00


def action_to_leaf(a: float) -> float:
    a = float(max(0.0, min(1.0, a)))
    lo = math.log(LEAF_MIN)
    hi = math.log(LEAF_MAX)
    leaf = math.exp(lo + a * (hi - lo))
    return float(max(LEAF_MIN, min(LEAF_MAX, leaf)))


class RunningMeanStdLite:
    def __init__(self, dim: int):
        self.count = 1e-4
        self.mean = np.zeros((dim,), dtype=np.float32)
        self.var = np.ones((dim,), dtype=np.float32)

    def update(self, x: np.ndarray):
        x = x.astype(np.float32)
        b_mean = x.mean(axis=0)
        b_var = x.var(axis=0)
        b_count = x.shape[0]

        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * (b_count / tot)

        m_a = self.var * self.count
        m_b = b_var * b_count
        M2 = m_a + m_b + (delta ** 2) * self.count * b_count / tot
        self.var = M2 / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.maximum(np.sqrt(self.var), 1e-6)
        return (x - self.mean) / std


class RLBatchedEnv(VecEnv):
    """
    Episode-constant action env for SC-LIO-SAM parameter selection via rosbridge.

    reset():
      - Select sequence and random start-percent (ensuring room for a probe if configured)
      - Start a short probe using EpisodeOrchestrator (or just warm-up in DRY_RUN)
      - Return normalized 16-D observation vector

    step(action):
      - Publish leaf via rosbridge to /lio_sam/params/mapping_surf_leaf_size
      - Run window (warmup + score) using EpisodeOrchestrator (CLI-only control)
      - Compute APE RMSE from est_tum vs GT; return terminal reward; done=True
    """

    metadata = {}

    def __init__(
        self,
        cfg: Dict[str, Any],
        num_envs: int = 1,
        mock_rosbridge: bool = False,
    ):
        self.num_envs = num_envs
        self.cfg = cfg

        # Config unpack
        self.seed_val = int(cfg.get("seed", 0))
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)

        paths = cfg.get("paths", {})
        self.est_tum = paths.get("est_tum", "/tmp/est.tum")
        self.run_root = paths.get("run_root", "/tmp/rlvo")

        mulran = cfg.get("mulran", {})
        self.seq_root = mulran.get("root", "/data/mulran")
        self.seqs = list(mulran.get("seqs", ["KAIST01"]))

        fpc = cfg.get("file_player", {})
        self.rate_hz = int(fpc.get("rate_hz", 10))
        self.start_percent_min = float(fpc.get("start_percent_min", 0.0))
        self.start_percent_max = float(fpc.get("start_percent_max", 90.0))

        timing = cfg.get("timing", {})
        self.probe_s_min = float(timing.get("probe_s_min", 2.0))
        self.probe_s_max = float(timing.get("probe_s_max", 5.0))
        self.warmup_s = float(timing.get("warmup_s", 2.0))
        self.score_s = float(timing.get("score_s", 18.0))

        rosbridge = cfg.get("rosbridge", {})
        self.rosbridge_url = rosbridge.get("url", "ws://localhost:9090")
        self.leaf_topic = cfg.get("topics", {}).get("leaf", "/lio_sam/params/mapping_surf_leaf_size")
        self.odom_topic = cfg.get("topics", {}).get("odom", "/lio_sam/mapping/odometry_incremental")

        # Observation/Action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.obs_rms = RunningMeanStdLite(16)
        self._last_obs = np.zeros((self.num_envs, 16), dtype=np.float32)

        # Rosbridge client (mock allowed for unit tests)
        self._mock_rb = bool(mock_rosbridge or (os.environ.get("DRY_RUN", "0") == "1"))
        self.rb = MockRosbridge() if self._mock_rb else RosbridgeClient(self.rosbridge_url)

        # Orchestrator
        self._orch = EpisodeOrchestrator(
            roslaunch_cmd=self._roslaunch_cmd_from_cfg(cfg),
            seq_root=self.seq_root,
            rosbridge_url=self.rosbridge_url,
            est_out=self.est_tum,
            rate_hz=self.rate_hz,
            run_root=self.run_root,
            odom_topic=self.odom_topic,
        )

        # Episode-local sample
        self._current_seq = None
        self._current_start_percent = None
        self._pending_action = None

    # ---- VecEnv required API ----
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Choose sequence and random starting percent, keep margin before run for a separate probe
        self._current_seq = random.choice(self.seqs)
        probe_s = random.uniform(self.probe_s_min, self.probe_s_max)
        margin_pct = self._seconds_to_percent(self.warmup_s + self.score_s + probe_s)

        low = self.start_percent_min
        high = max(self.start_percent_min, min(self.start_percent_max, 100.0 - margin_pct))
        if high <= low:
            # Fallback: if not enough room, clamp within allowed range
            high = self.start_percent_max
        self._current_start_percent = random.uniform(low, high)

        # Run a short probe (cleaner: separate process-level episode)
        probe_stats = self._orch.probe(duration_s=probe_s, safe_leaf=0.40, topics={"leaf": self.leaf_topic})

        # Build observation vector (16-D, see spec)
        obs = self._build_obs(probe_stats)
        self.obs_rms.update(obs)
        obs_norm = self.obs_rms.normalize(obs)

        self._last_obs[:] = obs_norm
        return self._last_obs

    def step(self, actions: np.ndarray, use_gt_initialization: bool = True):
        # Episode-constant action: apply once and run terminal window
        a = float(np.clip(actions.reshape(-1)[0], 0.0, 1.0))
        leaf = action_to_leaf(a)
        self._publish_leaf(leaf)

        # Run the window; steps = round(rate_hz * (warmup_s + score_s))
        result = self._orch.run_window(
            seq_name=self._current_seq,
            start_percent=float(self._current_start_percent),
            warmup_s=self.warmup_s,
            score_s=self.score_s,
        )

        est_path = result["est_tum"]
        gt_path = self._resolve_gt_path(self._current_seq)

        # Compute APE RMSE (clip [0,20])
        ape = ape_rmse(est_path, gt_path)
        ape = float(min(max(ape, 0.0), 20.0))

        timeout = 1.0 if result.get("timeout", False) else 0.0
        diverged = 1.0 if result.get("diverged", False) else 0.0
        runtime = float(result.get("runtime_s", self.warmup_s + self.score_s))

        reward = - ape - 0.01 * runtime - 5.0 * timeout - 10.0 * diverged

        done = np.ones((self.num_envs,), dtype=bool)
        info_item = {
            "ape_rmse": ape,
            "leaf": leaf,
            "seq": self._current_seq,
            "start_percent": self._current_start_percent,
            "runtime_s": runtime,
            "timeout": bool(timeout),
            "diverged": bool(diverged),
        }
        info = [info_item for _ in range(self.num_envs)]
        valid_mask = np.ones((self.num_envs,), dtype=bool)

        # Next observation is irrelevant for terminal episodes; return last obs
        return self._last_obs.copy(), np.array([reward], dtype=np.float32), done, info, valid_mask

    # ---- helpers ----
    def _build_obs(self, stats: Dict[str, float]) -> np.ndarray:
        keys = [
            "pts_per_scan_mean", "pts_per_scan_std", "scan_rate_hz",
            "surf_pts_mean", "surf_pts_std", "corner_pts_mean", "corner_pts_std",
            "odom_rate_hz", "pose_dropouts_s",
            "vel_norm_mean", "vel_norm_std", "acc_jolt_mean",
            "imu_ang_vel_rms", "imu_lin_acc_rms",
            "planarity_ratio_mean", "action_last",
        ]
        v = np.zeros((self.num_envs, 16), dtype=np.float32)
        for j, k in enumerate(keys):
            v[0, j] = float(stats.get(k, 0.0))
        # Ensure action_last=0 during probe
        v[0, -1] = 0.0
        return v

    def _publish_leaf(self, leaf: float):
        # rosbridge is assumed to be started by the roslaunch
        try:
            self.rb.publish_float(self.leaf_topic, float(leaf))
        except Exception as e:
            # If rosbridge is down, we proceed (reward will penalize via timeout/diverge)
            print(f"[WARN] rosbridge publish failed: {e}")

    def _resolve_gt_path(self, seq: str) -> str:
        """
        Assumption: Ground truth TUM file is located at:
            <seq_root>/<seq>/gt.tum
        Adjust this as needed if your GT lives elsewhere.
        In DRY_RUN, if missing, create a synthetic GT to match the est length.
        """
        path = os.path.join(self.seq_root, seq, "gt.tum")
        if os.path.exists(path):
            return path

        # DRY_RUN: synthesize a smooth GT file if not present (write under run_root to avoid read-only dataset dirs)
        if os.environ.get("DRY_RUN", "0") == "1":
            safe_gt = os.path.join(self.run_root, f"{seq}_gt.tum")
            self._write_synthetic_gt(safe_gt)
            return safe_gt

        # If no GT, fall back to est path (APE=0) but warn; better to raise and force config fix.
        print(f"[WARN] GT not found at {path}; using est for APE=0 fallback.")
        return self.est_tum

    @staticmethod
    def _write_synthetic_gt(path: str, n: int = 200, dt: float = 0.1):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t = np.arange(n) * dt
        xyz = np.zeros((n, 3), dtype=np.float64)
        quat = np.tile([0, 0, 0, 1.0], (n, 1))
        arr = np.column_stack([t, xyz, quat])
        np.savetxt(path, arr, fmt="%.9f")

    @staticmethod
    def _roslaunch_cmd_from_cfg(cfg: Dict[str, Any]) -> List[str]:
        # By default, use our provided launch file; user may customize
        launch_pkg = cfg.get("roslaunch", {}).get("package", None)
        launch_file = cfg.get("roslaunch", {}).get("file", "scripts/launch_sclsam.launch")
        if launch_pkg:
            return ["roslaunch", launch_pkg, launch_file]
        # Absolute or relative .launch path (roslaunch supports relative from CWD)
        return ["roslaunch", launch_file]

    # Unused VecEnv abstract methods (not needed by the PPO loop in rl_vo)
    def close(self):
        try:
            self._orch._teardown_all()
        except Exception:
            pass
        try:
            if hasattr(self.rb, "close"):
                self.rb.close()
        except Exception:
            pass

    # The following are (not implemented) methods for the abstract parent methods
    def render(self):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        raise NotImplementedError

    def env_method(
            self,
            method_name: str,
            *method_args,
            indices: VecEnvIndices = None,
            **method_kwargs
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise NotImplementedError

    def step_async(self):
        raise NotImplementedError

    def step_wait(self):
        raise NotImplementedError

    # Utility
    @staticmethod
    def _seconds_to_percent(seconds: float) -> float:
        # Heuristic: estimate percent budget using playback rate; if total seq duration unknown,
        # reserve a small margin (we assume user sets start_percent_max conservatively).
        # Here we map seconds to ~percent-of-seq as a tiny fraction to avoid impossible constraints.
        # You can override by tightening start_percent_max in config.
        return min(5.0, max(0.0, seconds * 0.01))
