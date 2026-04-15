# PPO Pong: Curriculum vs Direct Training

Contributors: Maharshii Patel, Jiwon Lee, Jacob Lee, Guojia La 

A PPO agent trained on Atari Pong to compare two training strategies:

- **Curriculum**: train on difficulty 0 first, then advance to difficulties 1, 2, and 3 once a performance threshold is met.
- **Direct**: train on difficulty 3 from the start.

***

## Project Structure

```
.
├── config.py            All hyperparameters and paths
├── train.py             Training script (curriculum or direct)
├── evaluate.py          Evaluation with live visualisation
├── compare_methods.py   Generate comparison plots and reports
├── run_eval_suite.py    Cross-difficulty/RAP evaluation helper
├── utils/
│   └── wrappers.py      Environment factory
├── metrics/             Per-seed training and evaluation CSVs
├── reports/             Aggregated plots and summary CSVs
├── models/              Model checkpoints
└── logs/                TensorBoard event files
```

***

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
AutoROM --accept-license
```

Requires Python 3.10+.

***

## Training

```bash
# Curriculum (difficulty 0 -> 3, threshold-based progression)
python train.py --method curriculum --seed 42

# Direct (difficulty 3 from the start)
python train.py --method direct --seed 42

# Resume from a checkpoint
python train.py --method curriculum --seed 42 --resume models/best/model
```

Training logs are written to `logs/`. Monitor with:

```bash
tensorboard --logdir logs
```

Per-step metrics are saved to `metrics/train_{method}seed{seed}.csv` as training runs.

***

## Evaluation

```bash
python evaluate.py --model models/best/best_model

# Options
python evaluate.py --model models/best/best_model --episodes 20
python evaluate.py --model models/best/best_model --difficulty 3 --no-render
```

Runs a live dashboard with the game feed, per-episode rewards, and run statistics. Saves an MP4 by default.

***

## Cross-Difficulty Evaluation Suite

After training, run the evaluation suite to collect results across all difficulty/mode/RAP combinations:

```bash
python run_eval_suite.py \
  --model models/best/best_model \
  --method curriculum \
  --seed 42 \
  --out metrics/eval_curriculum_seed42.csv
```

***

## Comparison Report

Once you have metrics from both methods across multiple seeds:

```bash
python compare_methods.py --outdir reports/
```

Outputs written to `reports/`:

| File | Description |
|---|---|
| `method_summary.csv` | Aggregated metrics per method |
| `seed_summary.csv` | Per-seed metrics |
| `training_curve_total_steps.png` | Learning curve over total steps |
| `target_curve_difficulty3_steps.png` | Learning curve over difficulty-3 steps only |
| `summary_bars.png` | Final reward, AUC, time-to-threshold, jumpstart |
| `cross_difficulty_heatmap.png` | Robustness across difficulty/mode/RAP |
| `eval_suite_summary.csv` | Aggregated evaluation suite results |
| `comparison_report.md` | Written summary |

***

## Configuration

All settings are in `config.py`. Key options:

| Setting | Default | Description |
|---|---|---|
| `N_ENVS` | 8 | Parallel training environments |
| `TOTAL_TIMESTEPS` | 25,000,000 | Total environment steps |
| `LEARNING_RATE` | 2.5e-4 | PPO learning rate |
| `CLIP_RANGE` | 0.1 | PPO clipping epsilon |
| `EVAL_FREQ` | 50,000 | Steps between evaluations |
| `N_EVAL_EPISODES` | 10 | Episodes per evaluation |
| `CURRICULUM_THRESHOLD` | 15.0 | Mean reward to advance difficulty |
| `CURRICULUM_STREAK` | 2 | Consecutive evals above threshold to advance |
| `CURRICULUM_MIN_STEPS` | 1,000,000 | Minimum steps per difficulty stage |
| `CURRICULUM_MAX_STEPS` | 10,000,000 | Maximum steps before forcing advancement |

***

## How It Works

**Environment**: `ALE/Pong-v5` with grayscale, 84x84 resize, frame-skip 4, reward clipping, and 4-frame stacking via `VecFrameStack`.

**Policy**: CNN policy (Nature DQN architecture — 3 conv layers, 512-unit FC, shared actor-critic heads).

**Curriculum logic**: The `AdaptiveCurriculumCallback` in `train.py` evaluates the agent every `EVAL_FREQ` steps. If the mean reward exceeds `CURRICULUM_THRESHOLD` for `CURRICULUM_STREAK` consecutive evaluations and at least `CURRICULUM_MIN_STEPS` have elapsed in the current stage, the environment is rebuilt at the next difficulty level.