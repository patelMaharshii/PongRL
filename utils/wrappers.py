"""
Environment factory helpers.

- make_env()       → vectorised + frame-stacked training env (no rendering)
- make_eval_env()  → single env with rgb_array rendering for visualisation
"""

import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

gym.register_envs(ale_py)


def make_env(env_id: str, n_envs: int, n_stack: int, seed: int) -> VecFrameStack:
    """
    Creates a vectorised, preprocessed Atari environment for training.

    SB3's make_atari_env automatically applies:
      - NoopResetEnv        (random no-ops on reset for stochasticity)
      - MaxAndSkipEnv       (frame-skip k=4, max-pool last 2 frames)
      - EpisodicLifeEnv     (life loss = episode end for credit assignment)
      - FireResetEnv        (press FIRE on reset where required)
      - WarpFrame            (resize to 84x84 grayscale)
      - ClipRewardEnv       (clip rewards to [-1, +1])
    """
    env = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"frameskip": 1}   # ← ADD THIS
    )
    env = VecFrameStack(env, n_stack)
    return env


def make_eval_env(env_id: str, n_stack: int, seed: int,
                  render: bool = True) -> VecFrameStack:
    """
    Creates a single, preprocessed Atari environment for evaluation.
    If render=True, uses rgb_array mode so frames can be grabbed by evaluate.py.
    """
    render_mode = "rgb_array" if render else None
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"frameskip": 1, "render_mode": render_mode}   # ← ADD frameskip: 1
    )
    env = VecFrameStack(env, n_stack)
    return env