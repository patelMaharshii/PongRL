"""
Train PPO on Atari Pong using either:
- direct training on a fixed difficulty
- adaptive curriculum learning with threshold-based progression

Outputs machine-readable CSV/JSON metrics for later comparison.
"""

import argparse
import csv
import glob
import json
import os
import signal
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from config import Config
from utils.wrappers import make_env, make_eval_env


@dataclass
class EvalRecord:
    train_step: int
    method: str
    seed: int
    current_difficulty: int
    target_difficulty_steps: int
    eval_difficulty: int
    eval_mode: int
    eval_rap: float
    mean_reward: float
    std_reward: float
    win_rate: float
    threshold_hit: int
    stage_steps: int
    stage_index: int


class AdaptiveCurriculumCallback(BaseCallback):
    def __init__(
        self,
        cfg: Config,
        method: str,
        seed: int,
        metrics_csv_path: str,
        eval_freq: int,
        n_eval_episodes: int,
        mode: int,
        rap: float,
        direct_difficulty: int,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.cfg = cfg
        self.method = method
        self.seed = seed
        self.metrics_csv_path = metrics_csv_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.mode = mode
        self.rap = rap
        self.direct_difficulty = direct_difficulty

        self.curriculum_levels = list(cfg.CURRICULUM_LEVELS)
        self.stage_index = 0
        self.current_difficulty = self.curriculum_levels[0] if method == "curriculum" else direct_difficulty
        self.stage_start_t = 0
        self.target_difficulty_steps = 0
        self.streak = 0
        self.best_mean = -float("inf")
        self.best_model_path = None
        self.records: list[EvalRecord] = []

        self.eval_env = None
        self.header_written = False
        self._last_eval_timestep = 0

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.metrics_csv_path), exist_ok=True)
        self.eval_env = make_eval_env(
            self.cfg.ENV_ID,
            self.cfg.N_STACK,
            seed=self.seed + 10_000,
            render=False,
            difficulty=self.current_difficulty,
            mode=self.mode,
            repeat_action_probability=self.rap,
        )
        self._write_header_if_needed()
        self._save_stage_manifest(event="training_start")

    def _write_header_if_needed(self):
        if os.path.exists(self.metrics_csv_path) and os.path.getsize(self.metrics_csv_path) > 0:
            self.header_written = True
            return
        with open(self.metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(EvalRecord(0,"",0,0,0,0,0,0.0,0.0,0.0,0.0,0,0,0)).keys()))
            writer.writeheader()
        self.header_written = True

    def _append_record(self, record: EvalRecord):
        with open(self.metrics_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(record).keys()))
            writer.writerow(asdict(record))
        self.records.append(record)

    def _set_train_env_difficulty(self, difficulty: int):
        new_env = make_env(
            self.cfg.ENV_ID,
            self.cfg.N_ENVS,
            self.cfg.N_STACK,
            self.seed,
            difficulty=difficulty,
            mode=self.mode,
            repeat_action_probability=self.rap,
        )
        old_env = self.model.env
        self.model.set_env(new_env)  # SB3 wraps in VecTransposeImage here
        old_env.close()

        # Reset through self.model.env so the obs goes through
        # VecTransposeImage and comes back as (N, C, H, W) = (8,4,84,84)
        reset_obs = self.model.env.reset()
        self.model._last_obs = reset_obs
        self.model._last_episode_starts = np.ones(
            (self.cfg.N_ENVS,), dtype=bool
        )

    def _set_eval_env_difficulty(self, difficulty: int):
        if self.eval_env is not None:
            self.eval_env.close()
        self.eval_env = make_eval_env(
            self.cfg.ENV_ID,
            self.cfg.N_STACK,
            seed=self.seed + 10_000,
            render=False,
            difficulty=difficulty,
            mode=self.mode,
            repeat_action_probability=self.rap,
        )

    def _run_eval(self, difficulty: int):
        rewards = []
        wins = 0
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, info = self.eval_env.step(action)
                total_reward += float(reward[0])
                done = bool(terminated[0])
            rewards.append(total_reward)
            wins += int(total_reward > 0)
        return float(np.mean(rewards)), float(np.std(rewards)), float(wins / len(rewards))

    def _save_stage_manifest(self, event: str):
        path = os.path.join(self.cfg.METRICS_DIR, f"{self.method}_seed{self.seed}_manifest.json")
        os.makedirs(self.cfg.METRICS_DIR, exist_ok=True)
        payload = {
            "event": event,
            "method": self.method,
            "seed": self.seed,
            "num_timesteps": int(self.model.num_timesteps) if self.model else 0,
            "current_difficulty": self.current_difficulty,
            "stage_index": self.stage_index,
            "stage_start_t": self.stage_start_t,
            "target_difficulty_steps": self.target_difficulty_steps,
            "threshold": self.cfg.CURRICULUM_THRESHOLD,
            "streak": self.streak,
            "levels": self.curriculum_levels,
            "mode": self.mode,
            "rap": self.rap,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_timestep < self.eval_freq:
            return True

        self._last_eval_timestep = self.num_timesteps 

        mean_reward, std_reward, win_rate = self._run_eval(self.current_difficulty)
        stage_steps = self.num_timesteps - self.stage_start_t
        if self.current_difficulty == self.cfg.CURRICULUM_TARGET_DIFF:
            self.target_difficulty_steps += self.eval_freq

        threshold_hit = int(mean_reward >= self.cfg.CURRICULUM_THRESHOLD)
        if threshold_hit:
            self.streak += 1
        else:
            self.streak = 0

        record = EvalRecord(
            train_step=int(self.num_timesteps),
            method=self.method,
            seed=self.seed,
            current_difficulty=self.current_difficulty,
            target_difficulty_steps=int(self.target_difficulty_steps),
            eval_difficulty=self.current_difficulty,
            eval_mode=self.mode,
            eval_rap=float(self.rap),
            mean_reward=float(mean_reward),
            std_reward=float(std_reward),
            win_rate=float(win_rate),
            threshold_hit=threshold_hit,
            stage_steps=int(stage_steps),
            stage_index=int(self.stage_index),
        )
        self._append_record(record)

        if mean_reward > self.best_mean:
            self.best_mean = mean_reward
            best_path = os.path.join(self.cfg.BEST_MODEL_DIR, f"{self.method}_seed{self.seed}_best")
            self.model.save(best_path)
            self.best_model_path = best_path

        if self.method == "curriculum":
            can_advance = (
                self.stage_index < len(self.curriculum_levels) - 1
                and stage_steps >= self.cfg.CURRICULUM_MIN_STEPS
                and self.streak >= self.cfg.CURRICULUM_STREAK
            )
            force_advance = (
                self.stage_index < len(self.curriculum_levels) - 1
                and stage_steps >= self.cfg.CURRICULUM_MAX_STEPS
            )
            if can_advance or force_advance:
                self.stage_index += 1
                self.current_difficulty = self.curriculum_levels[self.stage_index]
                self.stage_start_t = self.num_timesteps
                self.streak = 0
                self._set_train_env_difficulty(self.current_difficulty)
                self._set_eval_env_difficulty(self.current_difficulty)
                self._save_stage_manifest(event="advance")
                if self.verbose:
                    reason = "threshold" if can_advance else "max_steps"
                    print(f"[train] advance -> difficulty {self.current_difficulty} via {reason} at step {self.num_timesteps:,}")

        if self.verbose:
            print(
                f"[eval] step={self.num_timesteps:,} method={self.method} diff={self.current_difficulty} "
                f"mean={mean_reward:+.2f} std={std_reward:.2f} win_rate={win_rate:.2%} streak={self.streak}"
            )
        return True

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
        self._save_stage_manifest(event="training_end")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO Pong for curriculum/direct comparison")
    parser.add_argument("--method", choices=["curriculum", "direct"], required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--difficulty", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Used by direct training only")
    parser.add_argument("--mode", type=int, default=None, choices=[0, 1])
    parser.add_argument("--rap", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--streak", type=int, default=None)
    parser.add_argument("--min-stage-steps", type=int, default=None)
    parser.add_argument("--max-stage-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    total_steps = args.timesteps or cfg.TOTAL_TIMESTEPS
    seed = args.seed if args.seed is not None else cfg.SEED
    mode = args.mode if args.mode is not None else cfg.MODE
    rap = args.rap if args.rap is not None else cfg.REPEAT_ACTION_PROB

    if args.threshold is not None:
        cfg.CURRICULUM_THRESHOLD = args.threshold
    if args.streak is not None:
        cfg.CURRICULUM_STREAK = args.streak
    if args.min_stage_steps is not None:
        cfg.CURRICULUM_MIN_STEPS = args.min_stage_steps
    if args.max_stage_steps is not None:
        cfg.CURRICULUM_MAX_STEPS = args.max_stage_steps

    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.METRICS_DIR, exist_ok=True)

    initial_difficulty = cfg.CURRICULUM_LEVELS[0] if args.method == "curriculum" else args.difficulty
    train_env = make_env(cfg.ENV_ID, cfg.N_ENVS, cfg.N_STACK, seed,
                         difficulty=initial_difficulty, mode=mode, repeat_action_probability=rap)

    metrics_csv = os.path.join(cfg.METRICS_DIR, f"train_{args.method}_seed{seed}.csv")
    curriculum_callback = AdaptiveCurriculumCallback(
        cfg=cfg,
        method=args.method,
        seed=seed,
        metrics_csv_path=metrics_csv,
        eval_freq=cfg.EVAL_FREQ,
        n_eval_episodes=cfg.N_EVAL_EPISODES,
        mode=mode,
        rap=rap,
        direct_difficulty=args.difficulty,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.CHECKPOINT_FREQ,
        save_path=cfg.MODEL_DIR,
        save_replay_buffer=False,
        save_vecnormalize=False,
        name_prefix=f"ppo_pong_{args.method}_seed{seed}",
        verbose=1,
    )

    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env, tensorboard_log=cfg.LOG_DIR)
    else:
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            device="cuda" if torch.cuda.is_available() else "cpu",
            learning_rate=cfg.LEARNING_RATE,
            n_steps=cfg.N_STEPS,
            batch_size=cfg.BATCH_SIZE,
            n_epochs=cfg.N_EPOCHS,
            gamma=cfg.GAMMA,
            gae_lambda=cfg.GAE_LAMBDA,
            clip_range=cfg.CLIP_RANGE,
            ent_coef=cfg.ENT_COEF,
            vf_coef=cfg.VF_COEF,
            max_grad_norm=cfg.MAX_GRAD_NORM,
            tensorboard_log=cfg.LOG_DIR,
            verbose=1,
            seed=seed,
        )

    print(f"[train] method={args.method} seed={seed} total_steps={total_steps:,} init_diff={initial_difficulty} mode={mode} rap={rap}")

    def _save_and_exit(sig, frame):
        interrupted_path = os.path.join(cfg.MODEL_DIR, f"ppo_pong_{args.method}_seed{seed}_interrupted")
        model.save(interrupted_path)
        print(f"\n[Interrupted] saved -> {interrupted_path}.zip at {model.num_timesteps:,} steps")
        sys.exit(0)

    signal.signal(signal.SIGINT, _save_and_exit)

    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([curriculum_callback, checkpoint_callback]),
        reset_num_timesteps=(args.resume is None),
    )

    final_path = os.path.join(cfg.MODEL_DIR, f"ppo_pong_{args.method}_seed{seed}_final")
    model.save(final_path)
    print(f"[train] final model -> {final_path}.zip")

    ckpts = sorted(glob.glob(os.path.join(cfg.MODEL_DIR, f"ppo_pong_{args.method}_seed{seed}_*.zip")))
    for old in ckpts[:-5]:
        if "final" not in old and "best" not in old:
            try:
                os.remove(old)
            except OSError:
                pass

    train_env.close()


if __name__ == "__main__":
    main()
