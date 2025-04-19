import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env import ABETReportEnv

def test_env_shapes():
    env = ABETReportEnv(seed=42)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    obs, r, done, trunc, info = env.step(env.action_space.sample())
    assert isinstance(r, (float, np.floating))
    assert env.observation_space.contains(obs)
