# env/rl_env.py
import os
import time
import math
import random
import numpy as np
from typing import Dict, Any, List, Type

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices

from env.episode_orchestrator import EpisodeOrchestrator
from env.rosbridge_client import RosbridgeClient, MockRosbridge
from env.utils.ape import ape_rmse

LEAF_MIN = 0.05
LEAF_MAX = 1.00

def action_to_leaf(a: float) -> float:
    a = float(max(0.0, min(1.0, a)))
    lo = math.log(LEAF_MIN); hi = math.log(LEAF_MAX)
    bounded_a = float(max(LEAF_MIN, min(LEAF_MAX, math.exp(lo + a * (hi - lo)))))
    print(f"[DEBUG] bounded_action = {bounded_a}, action = {a}")
    return bounded_a

class RunningMeanStdLite:
    def __init__(self, dim: int):
        self.count = 1e-4
        self.mean = np.zeros((dim,), dtype=np.float32)
        self.var = np.ones((dim,), dtype=np.float32)
    def update(self, x: np.ndarray):
        x = x.astype(np.float32)
        b_mean = x.mean(axis=0); b_var = x.var(axis=0); b_count = x.shape[0]
        delta = b_mean - self.mean; tot = self.count + b_count
        self.mean = self.mean + delta * (b_count / tot)
        m_a = self.var * self.count; m_b = b_var * b_count
        M2 = m_a + m_b + (delta ** 2) * self.count * b_count / tot
        self.var = M2 / tot; self.count = tot
    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.maximum(np.sqrt(self.var), 1e-6); return (x - self.mean) / std

class RLBatchedEnv(VecEnv):
    """
    Keep core ROS alive across episodes. End-of-file on the player ends the
    episode cleanly; next reset restarts only the player at a new start-percent.
    """

    metadata = {}

    def __init__(self, cfg: Dict[str, Any], num_envs: int = 1, mock_rosbridge: bool = False):
        self.num_envs = num_envs
        self.cfg = cfg

        # Config
        self.seed_val = int(cfg.get("seed", 0))
        random.seed(self.seed_val); np.random.seed(self.seed_val)

        paths = cfg.get("paths", {})
        self.est_tum = paths.get("est_tum", "/tmp/est.tum")
        self.run_root = paths.get("run_root", "/tmp/rlvo")

        mulran = cfg.get("mulran", {})
        self.seq_root = mulran.get("root", "/data/mulran")
        self.seqs = list(mulran.get("seqs", ["KAIST01"]))

        fpc = cfg.get("file_player", {})
        self.rate_hz = int(fpc.get("rate_hz", 10))
        self.start_percent_min = float(fpc.get("start_percent_min", 0.0))
        self.start_percent_max = float(fpc.get("start_percent_max", 80.0))

        timing = cfg.get("timing", {})
        self.step_len_s = float(timing.get("step_len_s", 1.0))
        self.score_win_s = float(timing.get("score_win_s", 3.0))
        self.max_steps = int(timing.get("max_steps", 100000))
        self.probe_s_min = float(timing.get("probe_s_min", 2.0))
        self.probe_s_max = float(timing.get("probe_s_max", 5.0))
        self.warmup_s = float(timing.get("warmup_s", 2.0))
        self.score_s = float(timing.get("score_s", 18.0))

        tokens_cfg = cfg.get("tokens", {})
        self.max_tokens = int(tokens_cfg.get("max_tokens", 64))
        self.variable_feature_dim = int(tokens_cfg.get("variable_feature_dim", 3))
        self.critique_dim = int(tokens_cfg.get("critique_dim", 4))
        self._fixed_keys = [
            "pts_per_scan_mean","pts_per_scan_std","scan_rate_hz",
            "surf_pts_mean","surf_pts_std","corner_pts_mean","corner_pts_std",
            "odom_rate_hz","pose_dropouts_s",
            "vel_norm_mean","vel_norm_std","acc_jolt_mean",
            "imu_ang_vel_rms","imu_lin_acc_rms",
            "planarity_ratio_mean","action_last",
        ]
        self.fixed_dim = len(self._fixed_keys)
        self.variable_block_dim = self.max_tokens * self.variable_feature_dim
        self.total_obs_dim = self.fixed_dim + self.variable_block_dim + self.critique_dim

        # Policy-facing dims
        self.agent_obs_dim_fixed = self.fixed_dim
        self.agent_obs_dim_variable = self.variable_block_dim

        # Reward shaping
        rew_cfg = cfg.get("reward", {})
        self.action_smooth_penalty = float(rew_cfg.get("action_smooth_penalty", 0.0))

        rb_port = int(cfg.get("rosbridge", {}).get("port", 9090))
        self.rosbridge_url = f"ws://localhost:{rb_port}"
        self.leaf_topic = cfg.get("topics", {}).get("leaf", "/lio_sam/params/mapping_surf_leaf_size")
        self.odom_topic = cfg.get("topics", {}).get("odom", "/lio_sam/mapping/odometry_incremental")

        # Spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.obs_rms = RunningMeanStdLite(self.total_obs_dim)
        self._last_obs = np.zeros((self.num_envs, self.total_obs_dim), dtype=np.float32)

        # Rosbridge client (mock allowed)
        self._mock_rb = bool(mock_rosbridge or (os.environ.get("DRY_RUN", "0") == "1"))
        self._rb = MockRosbridge() if self._mock_rb else RosbridgeClient(self.rosbridge_url, lazy=True)

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

        # Episode-local state
        self._current_seq = None
        self._current_start_percent = None
        self._steps = 0
        self._last_action = 0.0
        self._episode_alive = False
        self._pending_rms_update = False

    # ----------------- VecEnv API ----------------- #
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> np.ndarray:
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        self._current_seq = random.choice(self.seqs)
        self._current_start_percent = random.uniform(self.start_percent_min, self.start_percent_max)

        # Start/reuse core + start fresh player
        self._orch.begin_episode(
            seq_name=self._current_seq,
            start_percent=float(self._current_start_percent),
            safe_leaf=0.40,
        )
        self._episode_alive = True
        self._steps = 0
        self._last_action = 0.0

        t0 = time.time()
        stats = {}
        while time.time() - t0 < 1.0:
            stats = self._orch.commit_metrics()
            if stats:
                break
            time.sleep(0.05)
        obs = self._obs_from_stats(stats or {}, action_last=0.0, update_rms=bool(stats))
        self._pending_rms_update = not bool(stats)
        return obs

    def step(self, actions: np.ndarray, use_gt_initialization: bool = True):
        """
        Streaming step:
          - map action -> leaf, publish once
          - tick(step_len_s)
          - compute rolling APE over last score_win_s
          - build next obs from committed metrics
        """
        # Map action
        a = float(np.clip(actions.reshape(-1)[0], 0.0, 1.0))
        leaf = action_to_leaf(a)

        if not self._mock_rb:
            try:
                self._orch.set_leaf(self.leaf_topic, float(leaf))
                time.sleep(0.05)
            except Exception as e:
                print(f"[WARN] rosbridge publish failed: {e}")

        t0 = time.time()
        self._orch.tick(self.step_len_s)
        runtime = time.time() - t0

        player_alive = self._orch.is_player_alive()
        comms_ok = self._orch.comms_alive()
        will_hit_horizon = (self._steps + 1) >= self.max_steps
        done_flag = (not player_alive) or (not comms_ok) or will_hit_horizon

        if done_flag:
            # Keep core alive at normal episode end; restart only episode actors
            if not comms_ok:
                self._orch.close_all()
                done_reason = "ros_down"
            else:
                self._orch.end_episode()
                done_reason = "player_ended" if not player_alive else "max_steps"

            # Pick new random start percent for next episode and (re)start episode actors without killing core
            self._current_seq = random.choice(self.seqs)
            self._current_start_percent = random.uniform(self.start_percent_min, self.start_percent_max)
            self._orch.begin_episode(
                seq_name=self._current_seq,
                start_percent=float(self._current_start_percent),
                safe_leaf=0.40,
            )
            self._steps = 0
            self._last_action = 0.0

            # Warm-up: wait briefly for first non-empty metrics
            t0 = time.time()
            stats = {}
            while time.time() - t0 < 1.0:
                stats = self._orch.commit_metrics()
                if stats:
                    break
                time.sleep(0.05)
            obs = self._obs_from_stats(stats or {}, action_last=0.0, update_rms=bool(stats))
            self._pending_rms_update = not bool(stats)

            info_item = {
                "ape_rmse": None,
                "leaf": leaf,
                "seq": self._current_seq,
                "start_percent": self._current_start_percent,
                "runtime_s": float(runtime),
                "step_idx": int(self._steps),
                "player_alive": bool(player_alive),
                "done_reason": done_reason,
                "valid_mask": False,
            }
            valid_mask = np.zeros((self.num_envs,), dtype=bool)
            # Report that previous episode ended; return fresh obs for next episode and done=True
            return (
                obs,
                np.array([0.0], dtype=np.float32),
                np.array([True], dtype=bool),
                [info_item for _ in range(self.num_envs)],
                valid_mask
            )

        # Normal step
        est_path = self.est_tum
        gt_path = self._resolve_gt_path(self._current_seq)
        stats = self._orch.commit_metrics()

        have_stats = bool(stats) and (int(stats.get("variable_tokens_n", 0)) > 0)
        obs = self._obs_from_stats(
            stats if have_stats else {},
            action_last=a,
            update_rms=have_stats or getattr(self, "_pending_rms_update", False)
        )
        if have_stats and getattr(self, "_pending_rms_update", False):
            self._pending_rms_update = False

        # Default: invalid until GT confirmed and stats present
        valid_mask = np.array([have_stats], dtype=bool)

        reward = 0.0
        if gt_path and os.path.exists(gt_path):
            try:
                samefile = os.path.samefile(gt_path, est_path)
            except Exception:
                samefile = (os.path.abspath(gt_path) == os.path.abspath(est_path))
            if not samefile and have_stats:
                ape = float(ape_rmse(est_path, gt_path, score_last_seconds=self.score_win_s))
                runtime_pen = 0.001 * float(runtime)
                delta_pen = 0.01 * abs(a - float(self._last_action or 0.0))
                print(f"[DEBUG] ape: {ape}, runtime: {runtime_pen}, delta: {delta_pen}")
                reward = 0.1 * (-float(ape)) - runtime_pen - delta_pen
                valid_mask[:] = True
            else:
                # invalid if GT is same as est (fallback) or no stats
                valid_mask[:] = False
        else:
            valid_mask[:] = False

        # Step bookkeeping
        self._steps += 1
        self._last_action = a

        info_item = {
            "ape_rmse": float(ape) if 'ape' in locals() else None,
            "leaf": leaf,
            "seq": self._current_seq,
            "start_percent": self._current_start_percent,
            "runtime_s": float(runtime),
            "step_idx": int(self._steps),
            "player_alive": True,
            "valid_mask": bool(valid_mask[0]),
        }
        return obs, np.array([reward], dtype=np.float32), np.array([False], dtype=bool), [info_item], valid_mask

    def close(self):
        try:
            self._orch.close_all()
        except Exception:
            pass

    # ----------------- helpers ----------------- #
    def _obs_from_stats(self, stats: Dict[str, float], action_last: float, update_rms: bool) -> np.ndarray:
        fixed = np.zeros((self.num_envs, self.fixed_dim), dtype=np.float32)
        for j, k in enumerate(self._fixed_keys):
            fixed[0, j] = float(stats.get(k, 0.0))
        fixed[0, -1] = float(action_last)

        toks = stats.get("variable_tokens", [])
        vf = self.variable_feature_dim; mt = self.max_tokens
        toks_np = np.zeros((mt, vf), dtype=np.float32)
        for i in range(min(len(toks), mt)):
            tok = toks[i]
            for j in range(min(vf, len(tok))):
                toks_np[i, j] = float(tok[j])
        var_flat = toks_np.reshape(1, mt * vf)

        tail = stats.get("critique_tail", None)
        if tail is None or len(tail) < self.critique_dim:
            derived = [
                float(stats.get("odom_rate_hz", 0.0)),
                float(stats.get("pose_dropouts_s", 0.0)),
                float(stats.get("scan_rate_hz", 0.0)),
                float(stats.get("pts_per_scan_mean", 0.0)),
            ]
            tail = (derived + [0.0] * self.critique_dim)[:self.critique_dim]
        else:
            tail = tail[:self.critique_dim]
        tail_np = np.array(tail, dtype=np.float32).reshape(1, self.critique_dim)

        obs_raw = np.concatenate([fixed, var_flat, tail_np], axis=1).astype(np.float32)
        if update_rms:
            self.obs_rms.update(obs_raw)
        obs_norm = self.obs_rms.normalize(obs_raw)
        self._last_obs[:] = obs_norm
        return self._last_obs.copy()

    def _resolve_gt_path(self, seq: str) -> str:
        path = os.path.join(self.seq_root, seq, f"{seq}_gt.tum")
        if os.path.exists(path):
            return path
        if os.environ.get("DRY_RUN", "0") == "1":
            safe_gt = os.path.join(self.run_root, f"{seq}_gt.tum")
            self._write_synthetic_gt(safe_gt)
            return safe_gt
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
        launch_pkg = cfg.get("roslaunch", {}).get("package", None)
        launch_file = cfg.get("roslaunch", {}).get("file", "/rl_vo/scripts/launch_sclsam.launch")
        rb_port = int(cfg.get("rosbridge", {}).get("port", 9090))
        args = [f"rosbridge_port:={rb_port}"]
        if launch_pkg:
            return ["roslaunch", launch_pkg, launch_file] + args
        return ["roslaunch", launch_file] + args

    # Unused VecEnv hooks
    def render(self): raise NotImplementedError
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]: raise NotImplementedError
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]: raise NotImplementedError
    def get_attr(self, attr_name, indices=None): raise NotImplementedError
    def set_attr(self, attr_name, value, indices=None): raise NotImplementedError
    def step_async(self): raise NotImplementedError
    def step_wait(self): raise NotImplementedError
    def update_rms(self):
        """SB3 compatibility: RMS is updated opportunistically in step/reset."""
        return
