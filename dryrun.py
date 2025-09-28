import yaml, numpy as np
from env.rl_env import RLBatchedEnv
with open("config/config.yaml") as f: cfg=yaml.safe_load(f)
env = RLBatchedEnv(cfg, num_envs=1, mock_rosbridge=False)
obs = env.reset()
obs2, r, d, info, mask = env.step(np.array([[0.5]], dtype=np.float32))
print("reward:", r, "done:", d, "info:", info)
