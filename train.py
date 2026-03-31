"""
train.py — Train a PPO agent to play Atari Pong.

Usage:
    python train.py                                  # train from scratch
    python train.py --resume models/best             # resume from checkpoint
    python train.py --timesteps 5000000              # override total timesteps

    # Phase 2 — harder difficulty:
    python train.py --resume models/best/best_model_first_phase \\
                    --difficulty 3 --mode 0 --rap 0.25

    # Maximum chaos:
    python train.py --resume models/best/best_model_first_phase \\
                    --difficulty 3 --mode 1 --rap 0.5

Difficulty flags:
    --difficulty  INT    0–3  opponent paddle speed (default: config)
    --mode        INT    0–1  game variant: 0=standard, 1=squash (default: config)
    --rap         FLOAT  0–1  repeat-action probability / sticky actions (default: config)

Outputs:
    models/best/best_model.zip    best checkpoint (saved by EvalCallback)
    models/ppo_pong_final.zip     final weights after all timesteps
    logs/PPO_*/                   TensorBoard event files

Monitor training:
    tensorboard --logdir logs/
"""

import argparse
import glob
import os
import signal
import sys

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from config import Config
from utils.wrappers import make_env, make_eval_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Atari Pong")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .zip model to resume training from")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override Config.TOTAL_TIMESTEPS")

    # ── Difficulty flags ──────────────────────────────────────────────────────
    diff = parser.add_argument_group("difficulty modifiers")
    diff.add_argument("--difficulty", type=int, default=None,
                      choices=[0, 1, 2, 3],
                      help="Opponent paddle speed: 0=easiest … 3=hardest "
                           "(default: Config.DIFFICULTY)")
    diff.add_argument("--mode", type=int, default=None,
                      choices=[0, 1],
                      help="Game variant: 0=standard Pong, 1=squash/wall "
                           "(default: Config.MODE)")
    diff.add_argument("--rap", type=float, default=None,
                      metavar="FLOAT",
                      help="Repeat-action (sticky) probability 0–1 "
                           "(default: Config.REPEAT_ACTION_PROB)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = Config()

    total_steps = args.timesteps or cfg.TOTAL_TIMESTEPS

    # CLI overrides take priority over config defaults
    difficulty  = args.difficulty if args.difficulty is not None else cfg.DIFFICULTY
    mode        = args.mode       if args.mode       is not None else cfg.MODE
    rap         = args.rap        if args.rap        is not None else cfg.REPEAT_ACTION_PROB

    os.makedirs(cfg.MODEL_DIR,      exist_ok=True)
    os.makedirs(cfg.BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR,        exist_ok=True)

    # ── Print active settings ─────────────────────────────────────────────────
    print("[train] ── Environment settings ───────────────────────────────────")
    print(f"[train]   ENV_ID     : {cfg.ENV_ID}")
    print(f"[train]   difficulty : {difficulty}  (0=easiest paddle, 3=hardest)")
    print(f"[train]   mode       : {mode}  (0=standard, 1=squash)")
    print(f"[train]   sticky RAP : {rap}  (repeat-action probability)")
    print("[train] ────────────────────────────────────────────────────────────")

    # ── Environments ──────────────────────────────────────────────────────────
    train_env = make_env(
        cfg.ENV_ID, cfg.N_ENVS, cfg.N_STACK, cfg.SEED,
        difficulty=difficulty,
        mode=mode,
        repeat_action_probability=rap,
    )
    eval_env = make_eval_env(
        cfg.ENV_ID, cfg.N_STACK, seed=0, render=False,
        difficulty=difficulty,
        mode=mode,
        repeat_action_probability=rap,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=cfg.BEST_MODEL_DIR,
        eval_freq=cfg.EVAL_FREQ,
        n_eval_episodes=cfg.N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.CHECKPOINT_FREQ,
        save_path=cfg.MODEL_DIR,
        save_replay_buffer=False,
        save_vecnormalize=False,
        name_prefix="ppo_pong_ckpt",
        verbose=1,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.resume:
        print(f"[train] Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=cfg.LOG_DIR,
        )
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
            seed=cfg.SEED,
        )

    print(f"[train] Policy network:\n{model.policy}")
    print(f"[train] Training for {total_steps:,} timesteps with {cfg.N_ENVS} envs...")

    # ── Graceful shutdown on Ctrl+C ───────────────────────────────────────────
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
        total_timesteps=total_steps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=(args.resume is None),
    )

    # Prune old checkpoints, keep last 5
    ckpts = sorted(glob.glob(os.path.join(cfg.MODEL_DIR, "ppo_pong_ckpt_*.zip")))
    for old in ckpts[:-5]:
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