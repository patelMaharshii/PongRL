"""Helper script to produce cross-difficulty / cross-RAP evaluation CSVs."""

import argparse
import csv
import os

import numpy as np
from stable_baselines3 import PPO

from config import Config
from utils.wrappers import make_eval_env


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation suite for a trained Pong model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for difficulty in cfg.CROSS_DIFFICULTIES:
        for mode in cfg.CROSS_MODES:
            for rap in cfg.ROBUST_RAPS:
                env = make_eval_env(cfg.ENV_ID, cfg.N_STACK, seed=args.seed + 50_000, render=False,
                                    difficulty=difficulty, mode=mode,
                                    repeat_action_probability=rap)
                model = PPO.load(args.model, env=env)
                for ep in range(1, args.episodes + 1):
                    obs = env.reset()
                    done = False
                    reward_total = 0.0
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, info = env.step(action)
                        reward_total += float(reward[0])
                        done = bool(terminated[0])
                    rows.append({
                        "method": args.method,
                        "seed": args.seed,
                        "difficulty": difficulty,
                        "mode": mode,
                        "rap": rap,
                        "episode": ep,
                        "reward": reward_total,
                    })
                env.close()
                print(f"[suite] done method={args.method} seed={args.seed} diff={difficulty} mode={mode} rap={rap}")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method","seed","difficulty","mode","rap","episode","reward"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
