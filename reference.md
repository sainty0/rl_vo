# RL_VO Reference

Reinforcement Learning Meets Visual Odometry (ECCV 2024)
This document gives AI agents concise context about the rl_vo project: structure, purpose of major components, and tech stack.

## Folder/File Structure

Top-level layout (from tree.txt):
```
.
├── config
│   ├── __init__.py
│   ├── config_eval.yaml
│   └── config.yaml
├── dataloader
│   ├── __init__.py
│   ├── euroc_loader.py
│   ├── tartan_loader.py
│   └── tum_loader.py
├── doc
│   └── thumbnail.png
├── Dockerfile
├── env
│   ├── __init__.py
│   ├── svo_wrapper.py
│   └── utils
├── evaluation
│   └── evaluate_runs.py
├── launch_container.sh
├── LICENSE
├── play.py
├── policies
│   ├── __init__.py
│   └── attention_policy.py
├── README.md
├── requirements.txt
├── rl_algorithms
│   ├── __init__.py
│   ├── buffers.py
│   ├── on_policy_algorithm.py
│   └── ppo.py
├── svo-lib
│   ├── CMakeLists.txt
│   ├── Dockerfile
│   ├── fast_neon
│   ├── gflags
│   ├── glog
│   ├── launch_container.sh
│   ├── minkindr
│   ├── opengv
│   ├── requirements.txt
│   ├── rpg_common
│   ├── svo
│   ├── svo_cmake
│   ├── svo_common
│   ├── svo_direct
│   ├── svo_env
│   ├── svo_img_align
│   ├── svo_tracker
│   └── vikit
├── train.py
└── tree.txt
```

Other repository files present alongside rl_vo (outer project):
- docker-compose and various slam-related scripts/projects. These are not part of rl_vo itself; focus here is the rl_vo/ directory.

## Purpose of Each Major File/Directory

- Dockerfile
  - Container build for the RL VO framework on Ubuntu 20.04 with NVIDIA CUDA. Installs system deps, Python venv, PyTorch (CUDA wheels), and requirements.txt.
  - Creates a non-root sudo user via ENV USERNAME and configures GPU/container permissions.

- launch_container.sh
  - Convenience script to run the built image with GPU access, X11 forwarding, and bind mounts for:
    - Code: <path>/vo_rl/
    - Datasets: <path>/TartanAir/, <path>/EuRoC/, <path>/TUM-RGBD/
    - Logs: <path>/log_voRL/
  - Replace placeholders (<path>, etc.) with actual host paths before running.

- README.md
  - Paper links, citations, install/run instructions, dataset structure, and commands for training/evaluation.

- LICENSE
  - GPLv3 license for the project.

- requirements.txt
  - Python package requirements for the RL framework (unpinned except networkx in Dockerfile).

- tree.txt
  - Captured directory structure snapshot for quick reference.

- config/
  - config.yaml: Hydra config for training (dataset paths, VO parameters, agent/training hyperparameters, logging, wandb flags).
  - config_eval.yaml: Hydra config for evaluation/playback (paths, weight file(s), eval settings).
  - __init__.py: Marks package.

- dataloader/
  - tartan_loader.py, euroc_loader.py, tum_loader.py: Loaders for TartanAir, EuRoC, and TUM-RGBD datasets. Provide trajectory access, intrinsics handling (via calibration YAMLs), and per-sequence sampling.
  - __init__.py: Package marker.

- env/
  - svo_wrapper.py: Vectorized SVO-based Gymnasium-like environment (VecSVOEnv). Wraps the classical SVO VO pipeline as an RL environment. Exposes observation/action spaces, steps underlying VO per image pair, computes rewards, maintains normalization (RMS), utilities for GT-alignment, etc.
  - utils/: Helper modules for trajectory alignment, visualization overlays, etc. Used by env and evaluation workflows.
  - __init__.py: Package marker.

- policies/
  - attention_policy.py: Custom Actor-Critic policy architecture (CustomActorCriticPolicy) with an encoder tailored to variable-sized features from VO (e.g., keypoints/map stats). Provides stochastic policy head and value function for PPO.
  - __init__.py: Package marker.

- rl_algorithms/
  - ppo.py: PPO algorithm implementation customized for masked/variable-length rollouts and VecSVOEnv specifics. Handles learning loops, logging hooks, and checkpointing.
  - on_policy_algorithm.py: Base utilities for on-policy algorithms used by PPO.
  - buffers.py: MaskedRolloutBuffer supporting valid-state masking, GAE, returns, and multi-env episodes.
  - __init__.py: Package marker.

- train.py
  - Entrypoint for training via Hydra (config/config.yaml).
  - Builds VecSVOEnv (train/val), constructs CustomActorCriticPolicy, and runs PPO.learn with configured hyperparameters. Optionally resumes from a policy checkpoint, integrates with Weights & Biases (wandb).

- play.py
  - Entrypoint for evaluation/inference via Hydra (config/config_eval.yaml).
  - Loads trained policy weights and env RMS stats, steps the environment over evaluation sequences, saves per-step images/plots and results arrays, and prints aggregate statistics. Includes trajectory alignment (Umeyama) to compare predicted vs GT poses. Saves results for later aggregation.

- evaluation/evaluate_runs.py
  - Post-processing/evaluation script to aggregate results across runs. User populates METHODS dict with paths to run outputs and sets OUT_DIR, then runs as a module to compute metrics and produce plots/tables.

- svo-lib/
  - Third-party Semidirect Visual Odometry (SVO) sources and dependencies (C++/CMake). Built inside its own Dockerfile or via cmake .. && make (as in README).
  - Contains vendor libraries and SVO modules (tracker, img_align, etc.). Provides the classical VO backend wrapped by env/svo_wrapper.py for RL interaction.

- doc/thumbnail.png
  - Thumbnail used in README for video link.

## Tech Stack

System/Container
- Base image: nvidia/cuda:12.1.0-devel-ubuntu20.04
- OS: Ubuntu 20.04 (focal)
- GPU: NVIDIA required; container uses NVIDIA_VISIBLE_DEVICES and NVIDIA_DRIVER_CAPABILITIES. nvidia-container-toolkit installed for GPU passthrough in Docker.
- X11: Configured in launch_container.sh for GUI output (e.g., OpenCV windows/plots) inside container.

System Libraries (apt)
- Core: sudo, tmux, valgrind, libssl-dev
- Build: cmake, libboost-all-dev
- Math/Linear Algebra: libeigen3-dev, libsuitesparse-dev
- Vision/Graphics: libopencv-dev, libglew-dev
- Parsing/Config: libyaml-cpp-dev
- Python tooling: python3, python3-pip, python3.8-venv, python3-pybind11
- Python utility modules (Ubuntu packages): python3-importlib-metadata, python3-more-itertools, python3-zipp

Python
- Python environment: venv created via python3.8-venv in Dockerfile (PATH includes /venv/bin).
- PyTorch stack:
  - torch, torchvision, torchaudio installed from https://download.pytorch.org/whl/cu118 (CUDA 11.8 wheels). Exact versions resolve at build time based on index; wheels include CUDA runtime and are compatible in-container.
- Pinned:
  - networkx==3.1 (explicitly installed in Dockerfile).
- Requirements (requirements.txt; version-free unless noted):
  - ipython — dev REPL convenience
  - numpy — array ops
  - PyYAML — YAML parsing (configs/calibration)
  - gnupg — crypto tooling (if needed by evaluation/logging workflows)
  - opencv-python — image I/O and basic visualization
  - stable-baselines3 — RL algorithms (PPO backbone)
  - hydra-core — hierarchical config and CLI overrides
  - wandb — experiment logging/tracking
  - tqdm — progress bars
  - scipy — scientific utils
  - plotly — plotting
  - SciencePlots — matplotlib style sheets
  - pycryptodomex — crypto primitives
- Gymnasium and SB3:
  - Code references gymnasium.spaces; Gymnasium API is used via SB3 utilities (obs_as_tensor) and custom wrappers.

C++/CMake (SVO)
- Toolchain: CMake + GNU toolchain.
- Dependencies: Eigen3, OpenCV, Boost, SuiteSparse, YAML-CPP, GLEW, glog, gflags, and related vendor modules included under svo-lib/.
- Build:
  - As per README: mkdir build && cd build && cmake .. && make
  - Separate Dockerfile under svo-lib for standalone SVO builds if desired.

## Conceptual Overview / Data & Control Flow

- RL reframes VO as sequential decision-making:
  - Environment: The VO pipeline (SVO) + image sequence dataset acts as the environment.
  - Observations: Derived from VO internals (keypoints/map statistics/prior poses) and normalized via env RMS.
  - Actions: Policy outputs decisions (e.g., keyframe selection, grid-size selection); see play.py handling of 2-dim actions.
  - Rewards: Pose error, runtime, and other metrics configured under config.agent.reward.
- Training:
  - train.py (Hydra) -> constructs VecSVOEnv (train/val) -> CustomActorCriticPolicy -> PPO.learn
  - Logging: Optional W&B via config flags (wandb_logging, wandb_tag, wandb_group).
  - Checkpointing: PPO saves checkpoints; policy can be reloaded from config.policy_path.
- Evaluation:
  - play.py (Hydra) -> evaluate_policy loads policy weights and env RMS -> runs sequences up to max_eval_steps -> saves per-step images/npz and prints summary stats.
  - evaluation/evaluate_runs.py aggregates outputs from multiple runs to compare methods.

## Notes and Conventions

- Directory naming in README shows vo_rl while this repo directory is rl_vo; ensure paths in launch_container.sh and configs match your actual local path.
- Datasets require calibration YAMLs placed under dataset_root/calibration/ as documented in README.
- Placeholders:
  - Dockerfile uses ENV USERNAME and sets WORKDIR <path>/vo_rl; change to your actual username and path.
  - launch_container.sh contains <path> placeholders; replace with absolute host paths.
- GPU/Display:
  - launch_container.sh maps X11 sockets and sets DISPLAY for GUI output; ensure xhost local:root is allowed before running.
