import os, sys, yaml, numpy as np
# Ensure project root (rl_vo) is on sys.path when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_algorithms.ppo import PPO
from env.rl_env import RLBatchedEnv

def main():
    os.environ.setdefault("DRY_RUN", "1")
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    env = RLBatchedEnv(cfg, num_envs=1, mock_rosbridge=True)
    obs = env.reset()
    assert obs.shape == (1,16)
    model = PPO("MlpPolicy", env, n_steps=128, batch_size=64, gamma=0.99,
                gae_lambda=0.95, learning_rate=3e-4, n_epochs=10, seed=0, verbose=1)
    model.learn(total_timesteps=512)
    print("Mock training OK")

if __name__ == "__main__":
    main()
