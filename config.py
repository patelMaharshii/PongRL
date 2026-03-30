"""
Central configuration for all hyperparameters and paths.
Edit this file before running train.py or evaluate.py.
"""


class Config:
    # ── Environment ───────────────────────────────────────────────────────────
    ENV_ID        = "ALE/Pong-v5"   # Legacy env with no auto frame-skip
    N_ENVS        = 8                      # Parallel environments for rollout
    N_STACK       = 4                      # Frames to stack per observation
    SEED          = 42

    # ── PPO Hyperparameters ───────────────────────────────────────────────────
    LEARNING_RATE = 2.5e-4
    N_STEPS       = 128      # Steps collected per env per rollout
    BATCH_SIZE    = 256      # Minibatch size (N_STEPS * N_ENVS must be divisible)
    N_EPOCHS      = 4        # Gradient update passes per rollout
    GAMMA         = 0.99     # Discount factor
    GAE_LAMBDA    = 0.95     # GAE bias-variance trade-off
    CLIP_RANGE    = 0.1      # PPO clipping epsilon (tighter than default 0.2)
    ENT_COEF      = 0.01     # Entropy bonus to encourage exploration
    VF_COEF       = 0.5      # Value-function loss weight
    MAX_GRAD_NORM = 0.5      # Gradient clipping threshold

    # ── Training ──────────────────────────────────────────────────────────────
    TOTAL_TIMESTEPS   = 10_000_000
    EVAL_FREQ         = 50_000     # Evaluate every N env steps
    N_EVAL_EPISODES   = 10         # Episodes per evaluation checkpoint
    CHECKPOINT_FREQ   = 50_000

    # ── Paths ──────────────────────────────────────────────────────────────────
    MODEL_DIR      = "./models"
    BEST_MODEL_DIR = "./models/best"
    LOG_DIR        = "./logs"

    # ── Evaluation ────────────────────────────────────────────────────────────
    EVAL_EPISODES  = 5             # Episodes to run during evaluate.py
    RECORD_VIDEO   = True          # Save an MP4 of the evaluation run
    VIDEO_PATH     = "./eval_video.mp4"
    VIDEO_FPS      = 30