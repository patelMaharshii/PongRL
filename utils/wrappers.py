"""Environment factory helpers for Atari Pong experiments."""

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
