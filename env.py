# env.py  — ABET Optimisation & Reporting Environment
# -----------------------------------------------
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, List, Tuple


class ABETReportEnv(gym.Env):
    """
    Custom Gymnasium environment that models the optimisation cycle
    for an ABET self‑study report.

    Observation (flattened float32 vector):
        criterion_scores        – 8 floats in [0,1]
        doc_completeness        – 1 float in [0,1]
        feedback_embed          – 300‑d SBERT (PCA‑reduced) vector in [0,1]
        weeks_left_norm         – 1 float (weeks_left / MAX_WEEKS)
        last_action_one_hot     – 7 floats (|ACTIONS|)

    Action space (Discrete):
        0 : noop
        1 : update_faculty_cv
        2 : improve_course_syllabus
        3 : upload_student_assessment
        4 : clarify_outcome_mapping
        5 : add_continuous_improvement_evidence
        6 : request_missing_docs
    """

    metadata = {"render_modes": []}

    # ------------- static problem parameters ----------------------------------
    N_CRITERIA: int = 8
    EMBED_DIM: int = 300
    MAX_WEEKS: int = 8

    ACTIONS: List[str] = [
        "noop",
        "update_faculty_cv",
        "improve_course_syllabus",
        "upload_student_assessment",
        "clarify_outcome_mapping",
        "add_continuous_improvement_evidence",
        "request_missing_docs",
    ]

    # ------------- REWARD WEIGHTS — tweak to shape behaviour ------------------
    REWARD_CRIT_IMPROVE   = 15.0   # per‑point increase across all criteria
    REWARD_DOC_COMPLETE   = 3.0    # per‑point increase in doc completeness
    BONUS_ALL_SATISFIED   = 40.0   # once every criterion ≥ 0.9
    PENALTY_REDUNDANT     = -6.0   # noop / no improvement this step
    PENALTY_LATE_WEEK     = -2.0   # each late week if any criterion < 0.7
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # action & observation spaces
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        state_dim = (
            self.N_CRITERIA + 1 + self.EMBED_DIM + 1 + len(self.ACTIONS)
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )

        self.state: Dict[str, Any] = {}
        self.prev_scores: np.ndarray | None = None
        self.prev_doc: float | None = None

    # -------------------------------------------------------------------------
    # Gym API
    # -------------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # initialise state with mid‑range values
        self.state = {
            "criterion_scores": self.rng.uniform(0.3, 0.7, self.N_CRITERIA),
            "doc_completeness": float(self.rng.uniform(0.4, 0.8)),
            "feedback_embed": self.rng.uniform(0.0, 1.0, self.EMBED_DIM),
            "weeks_left": self.MAX_WEEKS,
            "last_action": 0,
        }
        self.prev_scores = self.state["criterion_scores"].copy()
        self.prev_doc    = self.state["doc_completeness"]
        return self._get_observation(), {"action_mask": compute_action_mask(self.state)}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), "Invalid action"

        self._apply_action(action)
        reward = self._compute_reward(action)

        # advance one week
        self.state["weeks_left"] = max(0, self.state["weeks_left"] - 1)

        terminated = bool(
            self.state["weeks_left"] == 0
            or np.all(self.state["criterion_scores"] >= 0.9)
        )
        truncated = False  # not using truncation for now

        obs = self._get_observation()

        # prepare for next step’s delta calculations
        self.prev_scores = self.state["criterion_scores"].copy()
        self.prev_doc    = self.state["doc_completeness"]

        info = {"action_mask": compute_action_mask(self.state)}

        return obs, reward, terminated, truncated, info

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    def _apply_action(self, action: int) -> None:
        """Simulate the effect of *action* on the report state."""
        if action == 0:          # noop
            self.state["last_action"] = 0
            return

        # boost the lowest‑scoring criterion
        idx_low = int(np.argmin(self.state["criterion_scores"]))
        delta = float(self.rng.uniform(0.05, 0.15))
        self.state["criterion_scores"][idx_low] = float(
            np.clip(
                self.state["criterion_scores"][idx_low] + delta, 0.0, 1.0
            )
        )

        # some actions also raise doc completeness
        if action in {2, 3, 5}:
            self.state["doc_completeness"] = float(
                np.clip(
                    self.state["doc_completeness"]
                    + self.rng.uniform(0.02, 0.05),
                    0.0,
                    1.0,
                )
            )

        self.state["last_action"] = action

    def _compute_reward(self, action: int) -> float:
        delta_scores = self.state["criterion_scores"] - self.prev_scores
        delta_doc    = self.state["doc_completeness"] - self.prev_doc

        reward  = self.REWARD_CRIT_IMPROVE * float(delta_scores.sum())
        reward += self.REWARD_DOC_COMPLETE * float(delta_doc)

        if np.all(self.state["criterion_scores"] >= 0.9):
            reward += self.BONUS_ALL_SATISFIED

        # penalty if nothing improved or noop
        if action == 0 or (abs(delta_scores.sum()) < 1e-4 and abs(delta_doc) < 1e-4):
            reward += self.PENALTY_REDUNDANT

        # urgency penalty near deadline if still weak
        if (
            self.state["weeks_left"] <= self.MAX_WEEKS * 0.25
            and np.any(self.state["criterion_scores"] < 0.7)
        ):
            reward += self.PENALTY_LATE_WEEK

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        action_one_hot = np.zeros(len(self.ACTIONS), dtype=np.float32)
        action_one_hot[self.state["last_action"]] = 1.0

        obs = np.concatenate(
            [
                self.state["criterion_scores"],
                [self.state["doc_completeness"]],
                self.state["feedback_embed"],
                [self.state["weeks_left"] / self.MAX_WEEKS],
                action_one_hot,
            ]
        ).astype(np.float32)
        return obs

    # render()/close() stubs could be added here if needed

def compute_action_mask(state: dict) -> np.ndarray:
    """
    Returns a Bool mask (True = legal) length == len(ACTIONS).
    Example heuristic:
      • Disallow edits if criterion already ≥ 0.9
      • Disallow doc‑upload if completeness ≥ 0.95
    """
    mask = np.ones(len(ABETReportEnv.ACTIONS), dtype=bool)

    # 1) If all criteria satisfied, only noop is legal
    if np.all(state["criterion_scores"] >= 0.9):
        mask[:] = False
        mask[0] = True
        return mask

    # 2) If doc completeness is high, hide actions 3 & 6
    if state["doc_completeness"] >= 0.95:
        mask[3] = mask[6] = False

    # 3) Hide action that just ran (avoid redundant repeat)
    mask[state["last_action"]] = False

    return mask


# ---------------------------------------------------------------
# Helper for sb3‑contrib ActionMasker wrapper
def mask_fn(env) -> np.ndarray:
    """Return a bool array; True where action is allowed."""
    return compute_action_mask(env.state)
# ---------------------------------------------------------------
