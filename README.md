# Optimization & Reporting Agent for ABET Accreditation (RL)

This agent uses reinforcement learning (Masked PPO) to optimize ABET self-study reports by:
- Improving compliance scores
- Suggesting targeted report edits
- Reducing time-to-completion

## Highlights
- Custom Gymnasium environment modeling ABET criteria
- Reward shaping and action masking
- Synthetic data simulation and offline warm-start
- Masked PPO policy training via `sb3-contrib`

## Structure
- `env.py` — Gym environment + reward logic
- `generate_synthetic_runs.py` — create replay buffer data
- `train_masked_ppo.py` — train RL model
- `models/` — saved PPO checkpoints
- `data/` — synthetic episodes
- `tests/` — env unit tests

## Next steps
- Integrate with Compliance Checker API
- Auto-generate full self-study sections
- Deploy `POST /ora/suggest` FastAPI service
