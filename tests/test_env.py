import numpy as np, yaml, os
from env.rl_env import RLBatchedEnv

def test_env_smoke():
    os.environ["DRY_RUN"] = "1"
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    env = RLBatchedEnv(cfg, num_envs=1, mock_rosbridge=True)
    obs = env.reset()
    assert obs.shape == (1,16)
    a = np.array([[0.5]], dtype=np.float32)
    obs2, r, d, info, m = env.step(a)
    assert d.all() and isinstance(info, list)
