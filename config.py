"""
Central configuration for PPO Pong curriculum/direct comparison experiments.
"""


class Config:
    # ── Environment ───────────────────────────────────────────────────────────
    ENV_ID  = "ALE/Pong-v5"
    N_ENVS  = 8
    N_STACK = 4
    SEED    = 42

    # ── PPO Hyperparameters ───────────────────────────────────────────────────
    LEARNING_RATE = 2.5e-4
    N_STEPS       = 128
    BATCH_SIZE    = 256
    N_EPOCHS      = 4
    GAMMA         = 0.99
    GAE_LAMBDA    = 0.95
    CLIP_RANGE    = 0.1
    ENT_COEF      = 0.01
    VF_COEF       = 0.5
    MAX_GRAD_NORM = 0.5

    # ── Training / evaluation cadence ────────────────────────────────────────
    TOTAL_TIMESTEPS   = 25_000_000
    EVAL_FREQ         = 50_000
    N_EVAL_EPISODES   = 10
    CHECKPOINT_FREQ   = 50_000

    # ── Difficulty defaults ───────────────────────────────────────────────────
    MODE                = 0
    REPEAT_ACTION_PROB  = 0.25

    # ── Adaptive curriculum settings ──────────────────────────────────────────
    CURRICULUM_LEVELS       = [0, 1, 2, 3]
    CURRICULUM_THRESHOLD    = 15.0
    CURRICULUM_STREAK       = 2          # consecutive evals above threshold
    CURRICULUM_MIN_STEPS    = 1_000_000  # minimum steps at a difficulty before advancing
    CURRICULUM_MAX_STEPS    = 10_000_000 # force-advance cap per difficulty
    CURRICULUM_TARGET_DIFF  = 3

    # ── Evaluation suite ──────────────────────────────────────────────────────
    EVAL_EPISODES      = 30
    RECORD_VIDEO       = True
    VIDEO_PATH         = "./eval_video.mp4"
    VIDEO_FPS          = 30
    ROBUST_RAPS        = [0.0, 0.25, 0.5]
    CROSS_DIFFICULTIES = [0, 1, 2, 3]
    CROSS_MODES        = [0, 1]

    # ── Paths ─────────────────────────────────────────────────────────────────
    MODEL_DIR      = "./models"
    BEST_MODEL_DIR = "./models/best"
    LOG_DIR        = "./logs"
    METRICS_DIR    = "./metrics"
    REPORT_DIR     = "./reports"
