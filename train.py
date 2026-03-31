"""
train.py — Train a PPO agent to play Atari Pong.

Usage:
    python train.py                         # train from scratch
    python train.py --resume models/best    # resume from a checkpoint
    python train.py --timesteps 5000000     # override total timesteps

Outputs:
    models/best/best_model.zip    best checkpoint (saved by EvalCallback)
    models/ppo_pong_final.zip     final weights after all timesteps
    logs/PPO_*/                   TensorBoard event files

Monitor training:
    tensorboard --logdir logs/
"""

import argparse
import os
import glob
import signal
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)

import torch
from config import Config
from utils.wrappers import make_env, make_eval_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Atari Pong")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .zip model to resume training from")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override Config.TOTAL_TIMESTEPS")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = Config()

    total_steps = args.timesteps or cfg.TOTAL_TIMESTEPS

    os.makedirs(cfg.MODEL_DIR,      exist_ok=True)
    os.makedirs(cfg.BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR,        exist_ok=True)

    # ── Environments ──────────────────────────────────────────────────────────
    train_env = make_env(cfg.ENV_ID, cfg.N_ENVS, cfg.N_STACK, cfg.SEED)
    eval_env  = make_eval_env(cfg.ENV_ID, cfg.N_STACK, seed=0, render=False)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = cfg.BEST_MODEL_DIR,
        eval_freq            = cfg.EVAL_FREQ,        # steps across all envs
        n_eval_episodes      = cfg.N_EVAL_EPISODES,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = cfg.CHECKPOINT_FREQ,
        save_path   = cfg.MODEL_DIR,
        save_replay_buffer = False,
        save_vecnormalize  = False,
        name_prefix = "ppo_pong_ckpt",
        verbose     = 1,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume:
        print(f"[train] Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env            = train_env,
            tensorboard_log = cfg.LOG_DIR,
        )
    else:
        model = PPO(
            policy         = "CnnPolicy",
            env            = train_env,
            device="cuda" if torch.cuda.is_available() else "cpu",
            learning_rate  = cfg.LEARNING_RATE,
            n_steps        = cfg.N_STEPS,
            batch_size     = cfg.BATCH_SIZE,
            n_epochs       = cfg.N_EPOCHS,
            gamma          = cfg.GAMMA,
            gae_lambda     = cfg.GAE_LAMBDA,
            clip_range     = cfg.CLIP_RANGE,
            ent_coef       = cfg.ENT_COEF,
            vf_coef        = cfg.VF_COEF,
            max_grad_norm  = cfg.MAX_GRAD_NORM,
            tensorboard_log = cfg.LOG_DIR,
            verbose        = 1,
            seed           = cfg.SEED,
        )

    print(f"[train] Policy network:\\n{model.policy}")
    print(f"[train] Training for {total_steps:,} timesteps with {cfg.N_ENVS} envs...")

    # ── Graceful shutdown on Ctrl+C ──────────────────────────────────────
    def _save_and_exit(sig, frame):
        print("\n\n[Interrupted] Saving current model before exit...")
        interrupted_path = os.path.join(cfg.MODEL_DIR, "ppo_pong_interrupted")
        model.save(interrupted_path)
        print(f"[Saved] → {interrupted_path}.zip")
        print(f"[Steps completed] {model.num_timesteps:,}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _save_and_exit)

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps = total_steps,
        callback        = [eval_callback, checkpoint_callback],
        reset_num_timesteps = (args.resume is None),
    )

    ckpts = sorted(glob.glob(os.path.join(cfg.MODEL_DIR, "ppo_pong_ckpt_*.zip")))
    keep_last_n = 5
    for old in ckpts[:-keep_last_n]:
        os.remove(old)
        print(f"[train] Pruned old checkpoint: {old}")

    # ── Save final ────────────────────────────────────────────────────────────
    final_path = os.path.join(cfg.MODEL_DIR, "ppo_pong_final")
    model.save(final_path)
    print(f"[train] Final model saved → {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()