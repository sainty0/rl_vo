#!/usr/bin/env python3
import os
import numpy as np
import torch
import hydra
from datetime import datetime
from omegaconf import OmegaConf
from stable_baselines3.common.utils import get_linear_fn

from rl_algorithms.ppo import PPO
from policies.attention_policy import CustomActorCriticPolicy
from env.rl_env import RLBatchedEnv  # for vo_algorithm == "SCLSAM"


def configure_random_seed(seed, env=None):
    if env is not None and hasattr(env, "seed"):
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config):
    torch.init_num_threads()

    # -------- Branch selection --------
    val_env = None
    eval_interval = -1
    policy = None
    policy_kwargs = {}

    if config.vo_algorithm == "SVO":
        # Lazily import, to avoid SVO deps on the SCLSAM branch
        from env.svo_wrapper import VecSVOEnv

        env = VecSVOEnv(
            config.svo_params_file, config.svo_calib_file,
            config.dataset_dir, config.n_envs, reward_config=config.agent.reward,
            mode="train", initialize_glog=True
        )
        val_env = VecSVOEnv(
            config.svo_params_file, config.svo_calib_file, config.dataset_dir, 32,
            reward_config=config.agent.reward, mode="val", initialize_glog=False
        )

        policy = CustomActorCriticPolicy
        encoder_kwargs = dict(
            variable_feature_dim=env.variable_feature_dim,
            obs_dim_variable=env.agent_obs_dim_variable,
            obs_dim_fixed=env.agent_obs_dim_fixed,
            critique_dim=env.critique_dim,
        )
        policy_kwargs = dict(
            encoder_kwargs=encoder_kwargs,
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            log_std_init=-0.0,
        )
        eval_interval = config.val_interval

    elif config.vo_algorithm == "SCLSAM":
        # New lidar path: attention policy with [fixed | variable tokens | critique tail]
        cfg_dict = OmegaConf.to_container(config, resolve=True)

        # Training env
        env = RLBatchedEnv(
            cfg_dict,
            num_envs=int(config.n_envs),
            mock_rosbridge=bool(os.environ.get("DRY_RUN", "0") == "1"),
        )

        # Validation env (same class; use your eval config keys if present)
        cfg_eval = dict(cfg_dict)
        # Optional: override sequences/ranges for validation if config provides them
        if "eval" in cfg_dict:
            cfg_eval["mulran"] = dict(cfg_eval.get("mulran", {}))
            if "sequences" in cfg_dict["eval"]:
                cfg_eval["mulran"]["seqs"] = list(cfg_dict["eval"]["sequences"])
        val_env = RLBatchedEnv(
            cfg_eval,
            num_envs=min(4, int(config.n_envs)),
            mock_rosbridge=bool(os.environ.get("DRY_RUN", "0") == "1"),
        )

        policy = CustomActorCriticPolicy
        encoder_kwargs = dict(
            variable_feature_dim=getattr(env, "variable_feature_dim", 3),
            obs_dim_variable=getattr(env, "agent_obs_dim_variable", 0),
            obs_dim_fixed=getattr(env, "agent_obs_dim_fixed", 16),
            critique_dim=getattr(env, "critique_dim", 0),
        )
        policy_kwargs = dict(
            encoder_kwargs=encoder_kwargs,
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            log_std_init=-0.0,
        )
        eval_interval = config.val_interval  # enable periodic eval with val_env
    else:
        raise AssertionError(f"Unknown vo_algorithm: {config.vo_algorithm}")

    # -------- Seeding --------
    configure_random_seed(config.seed, env=env)

    # -------- Logging dir --------
    if not config.wandb_logging:
        log_dir = os.path.join(config.log_path, datetime.now().strftime("%b%d_%H-%M-%S"))
    elif config.wandb_group is not None:
        log_dir = os.path.join(
            config.log_path,
            config.wandb_group,
            datetime.now().strftime("%b%d_%H-%M-%S") + "_" + config.wandb_tag,
            )
    else:
        log_dir = os.path.join(
            config.log_path,
            datetime.now().strftime("%b%d_%H-%M-%S") + "_" + config.wandb_tag,
            )
    if config.wandb_logging:
        os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)

    # -------- Device --------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -------- Optional RMS restore --------
    if getattr(config, "policy_path", None):
        rms_path = config.policy_path[:-4] + "_rms.npz"
        if hasattr(env, "load_rms") and os.path.exists(rms_path):
            env.load_rms(rms_path)
        if val_env is not None and hasattr(val_env, "load_rms") and os.path.exists(rms_path):
            val_env.load_rms(rms_path)

    # -------- Create PPO --------
    model = PPO(
        tensorboard_log=None,
        log_dir=log_dir,
        policy=policy,
        policy_kwargs=policy_kwargs,
        env=env,
        n_epochs=config.agent.n_epochs,
        gae_lambda=config.agent.gae_lambda,
        gamma=config.agent.gamma,
        n_steps=config.agent.n_steps,
        ent_coef=config.agent.ent_coef,
        vf_coef=config.agent.vf_coef,
        max_grad_norm=config.agent.max_grad_norm,
        batch_size=config.agent.batch_size,
        learning_rate=get_linear_fn(3e-4, 3e-5, 1.0),
        clip_range=0.2,
        use_sde=config.agent.use_sde,
        verbose=1,
        seed=config.seed,
        wandb_logging=config.wandb_logging,
        wandb_tag=config.wandb_tag,
        wandb_group=config.wandb_group,
        config=config,
        device=device,
    )

    # -------- Optional policy restore --------
    if getattr(config, "policy_path", None):
        state_dict = torch.load(config.policy_path, map_location=device, weights_only=False)["state_dict"]
        model.policy.load_state_dict(state_dict, strict=False)
        model.policy.to(device)

    # -------- Train --------
    model.learn(
        total_timesteps=int(config.total_timesteps),
        log_interval=100,
        eval_interval=eval_interval,
        val_env=val_env,
    )


if __name__ == "__main__":
    main()
