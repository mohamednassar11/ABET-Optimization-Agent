"""
train_dqn.py
Trains a DQN on ABETReportEnv, starting from offline synthetic data.
"""

import json, gzip, pathlib, random
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from env import ABETReportEnv


# -------- hyper‑parameters --------
TOTAL_STEPS          = 200_000          # on‑policy learning
BUFFER_SIZE          = 100_000
BATCH_SIZE           = 512
GAMMA                = 0.99
TARGET_NET_UPDATE    = 10_000
LEARNING_RATE        = 3e-4
SAVE_PATH            = "models/dqn_abet"
REPLAY_FILE          = "data/synthetic_replay.jsonl.gz"
SEED                 = 42
EXPLORATION_FRACTION = 0.5
EXPLORATION_FINAL_EPS= 0.05
# ----------------------------------

def load_offline_buffer(buf: ReplayBuffer, path: str, env):
    """Fill an SB3 replay buffer with transitions from JSONL‑gzip."""
    with gzip.open(path, "rt") as f:
        for line in f:
            t = json.loads(line)
            buf.add(
                obs               = np.array(t["obs"]),
                next_obs          = np.array(t["next_obs"]),
                action            = np.array([t["action"]]),
                reward            = np.array([t["reward"]]),
                done              = np.array([t["terminated"] or t["truncated"]]),
                infos             = [{}],
            )
    print(f"Loaded {buf.size()} transitions into buffer.")

def main():
    env = ABETReportEnv(seed=SEED)
    eval_env = ABETReportEnv(seed=SEED + 1)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate           = LEARNING_RATE,
        buffer_size             = BUFFER_SIZE,
        learning_starts         = 0,               # we already have data
        batch_size              = BATCH_SIZE,
        gamma                   = GAMMA,
        target_update_interval  = TARGET_NET_UPDATE,
        policy_kwargs           = dict(net_arch=[256, 256]),
        verbose                 = 1,
        seed                    = SEED,
        exploration_fraction    = EXPLORATION_FRACTION,
        exploration_final_eps   = EXPLORATION_FINAL_EPS,
    )

    # pre‑fill replay buffer
    load_offline_buffer(model.replay_buffer, REPLAY_FILE, env)

    # callbacks
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes   = 10,
        eval_freq         = 5_000,
        deterministic     = True,
        verbose           = 1,
    )
    pbar_cb = ProgressBarCallback()

    # train
    model.learn(total_timesteps=TOTAL_STEPS,
                progress_bar=False,
                callback=[eval_cb, pbar_cb])

    pathlib.Path(SAVE_PATH).parent.mkdir(exist_ok=True, parents=True)
    model.save(SAVE_PATH)
    print(f"\n  Model saved to {SAVE_PATH}.zip")

if __name__ == "__main__":
    # CUDA if available
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    main()
