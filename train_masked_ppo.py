"""
train_masked_ppo.py
Train a Maskable‑PPO agent on the ABETReportEnv using action masking.
"""

import pathlib, json, gzip, numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from env import ABETReportEnv, mask_fn   # mask_fn returns the action mask

# ------------------------ hyper‑params ---------------------------------------
TOTAL_STEPS   = 300_000        # on‑policy steps
N_STEPS_RO    = 2048           # rollout length
BATCH_SIZE    = 256            # minibatch size
GAMMA         = 0.99
LR            = 3e-4
SAVE_PATH     = "models/masked_ppo_abet"
SEED          = 0
# -----------------------------------------------------------------------------

# ---------- create masked VecEnvs -------------------------------------------
train_env = ActionMasker(ABETReportEnv(seed=SEED),   mask_fn)
eval_env  = ActionMasker(ABETReportEnv(seed=SEED+1), mask_fn)
# -----------------------------------------------------------------------------

model = MaskablePPO(
    "MlpPolicy",
    train_env,
    n_steps       = N_STEPS_RO,
    batch_size    = BATCH_SIZE,
    gamma         = GAMMA,
    learning_rate = LR,
    clip_range    = 0.2,
    gae_lambda    = 0.95,
    policy_kwargs = dict(net_arch=[256, 256]),
    verbose       = 1,
    seed          = SEED,
)

# ---------------- evaluation callback with masking ---------------------------
eval_cb = MaskableEvalCallback(
    eval_env,
    n_eval_episodes = 10,
    eval_freq       = 20_000,
    deterministic   = True,
    verbose         = 1,
)

# optional CLI progress bar
pbar_cb = ProgressBarCallback()

# --------------------------- TRAIN -------------------------------------------
model.learn(
    total_timesteps = TOTAL_STEPS,
    callback        = [pbar_cb, eval_cb],
)

# -------------------------- SAVE ---------------------------------------------
pathlib.Path(SAVE_PATH).parent.mkdir(exist_ok=True, parents=True)
model.save(SAVE_PATH)
print(f"\n Masked PPO model saved to {SAVE_PATH}.zip")
