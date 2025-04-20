#Agent 2 – ABET Compliance Checker

Part of the ABET Automation with MARL suite

This repository contains the Compliance Checking Agent (Agent 2). The agent analyses ABET self‑study documents (PDF/DOCX) and produces per‑criterion compliance scores that feed the multi‑agent reinforcement‑learning (MARL) pipeline.

Key Features

Stage

Description

Implementation

Parse

Extract paragraphs from the input report

ComplianceCheckingAgent._extract_paragraphs()

Classify

Rule‑based keyword matcher assigns each paragraph to one of the eight ABET criteria

RULES dictionary and _classify()

Score

Computes Coverage, Quality and a weighted Final Score for every criterion

_score()

State Space

Builds a nine‑element vector (eight criterion scores + overall score) for MARL consumption

ComplianceState.to_vector()

Reward

Returns supervised or heuristic reward for reinforcement learning

compute_reward()

Installation

python -m venv venv
source venv/bin/activate
pip install pdfplumber pandas numpy matplotlib

Quick Start

python agent2_pdf.py \
       --pdf "/path/to/ABET Self Study Report.pdf" \
       --out ./output \
       --viz          # optional – save bar‑chart PNG

Generated artefacts:

criteria_scores.json – grades for each criterion and overall status.

non_compliance_report.csv – gaps if any criterion score is below 0.70.

compliance_scores.png – bar‑chart visual summary (created only when --viz is set).

The console also prints the state vector and reward for MARL integration.

Directory Layout

.
├── agent2_pdf.py          main script / agent implementation
├── output/                generated reports
└── README.md              this file

Configuration

All rule parameters are in RULES inside agent2_pdf.py.

keywords – list of terms for each criterion.

min_ev – minimum evidence count expected.

imp – importance weight used when computing the overall score.

The weightings for coverage, quality and rule_score are set at the top of the file and can be adjusted.

MARL Integration

state_vec, reward = agent.run()

Pass state_vec to other agents or a coordinator and use reward for policy optimisation.

Road‑map

Replace the keyword matcher with a fine‑tuned BERT model for higher accuracy.

Support DOCX and plain‑text inputs.

Add Dockerfile and continuous‑integration tests.




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
