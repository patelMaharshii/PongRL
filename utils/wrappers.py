"""
Environment factory helpers.

- make_env()       → vectorised + frame-stacked training env (no rendering)
- make_eval_env()  → single env with rgb_array rendering for visualisation

Both accept ALE difficulty knobs:
  difficulty            int   0–3   opponent paddle speed (0 = easiest)
  mode                  int   0–1   game variant (0 = standard, 1 = squash)
  repeat_action_prob    float 0–1   sticky-action probability (0 = deterministic)
"""

import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)


def make_env(
    env_id: str,
    n_envs: int,
    n_stack: int,
    seed: int,
    difficulty: int = 0,
    mode: int = 0,
    repeat_action_probability: float = 0.25,
) -> VecFrameStack:
    """
    Creates a vectorised, preprocessed Atari environment for training.

    SB3's make_atari_env automatically applies:
      - NoopResetEnv     (random no-ops on reset for stochasticity)
      - MaxAndSkipEnv    (frame-skip k=4, max-pool last 2 frames)
      - EpisodicLifeEnv  (life loss = episode end for credit assignment)
      - FireResetEnv     (press FIRE on reset where required)
      - WarpFrame        (resize to 84x84 grayscale)
      - ClipRewardEnv    (clip rewards to [-1, +1])
    """
    env = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={
            "frameskip": 1,
            "difficulty": difficulty,
            "mode": mode,
            "repeat_action_probability": repeat_action_probability,
        },
    )
    return VecFrameStack(env, n_stack)


def make_eval_env(
    env_id: str,
    n_stack: int,
    seed: int,
    render: bool = True,
    difficulty: int = 0,
    mode: int = 0,
    repeat_action_probability: float = 0.25,
) -> VecFrameStack:
    """
    Creates a single, preprocessed Atari environment for evaluation.
    If render=True, uses rgb_array mode so frames can be grabbed by evaluate.py.
    """
    render_mode = "rgb_array" if render else None
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={
            "frameskip": 1,
            "render_mode": render_mode,
            "difficulty": difficulty,
            "mode": mode,
            "repeat_action_probability": repeat_action_probability,
        },
    )
    return VecFrameStack(env, n_stack)