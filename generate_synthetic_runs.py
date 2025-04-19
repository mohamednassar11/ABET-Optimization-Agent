"""
generate_synthetic_runs.py
Simulates accreditation episodes with random / heuristic actions
and stores transitions for offline RL pre‑training.
"""

import json, gzip, random
import numpy as np
from tqdm import trange
from env import ABETReportEnv

N_EPISODES   = 5000          # change as you like
MAX_STEPS    = 24
OUT_PATH     = "data/synthetic_replay.jsonl.gz"
SEED         = 42

rng = np.random.default_rng(SEED)
env = ABETReportEnv(seed=SEED)

def choose_action(observation):
    """
    Simple heuristic: 20 % random, else act on lowest‑scoring criterion.
    """
    if rng.random() < 0.2:
        return env.action_space.sample()

    # obs layout: [criterion_scores(8), doc_complete, feedback(300), weeks, last_action(7)]
    scores = observation[:8]
    lowest = int(scores.argmin())
    # map lowest‑index to action id (1‑6) or noop if already high
    mapping = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:1, 7:2}
    return mapping[lowest]

def main():
    transitions = 0
    with gzip.open(OUT_PATH, "wt") as f:
        for ep in trange(N_EPISODES, desc="Simulating"):
            obs, _ = env.reset()
            for _ in range(MAX_STEPS):
                act  = choose_action(obs)
                nxt, rew, term, trunc, info = env.step(act)

                sample = {
                    "obs":        obs.tolist(),
                    "action":     int(act),
                    "reward":     float(rew),
                    "next_obs":   nxt.tolist(),
                    "terminated": term,
                    "truncated":  trunc,
                }
                f.write(json.dumps(sample) + "\n")
                transitions += 1
                obs = nxt
                if term or trunc:
                    break
    print(f"Saved {transitions:,} transitions → {OUT_PATH}")

if __name__ == "__main__":
    main()
