# Pong RL — PPO Agent

Train and evaluate a Proximal Policy Optimization (PPO) agent to play Atari Pong
using Stable-Baselines3 and Gymnasium.

---

## Project Structure

```
pong-rl/
├── config.py          # All hyperparameters and paths (edit this)
├── train.py           # Training script
├── evaluate.py        # Evaluation with live visualisation
├── utils/
│   ├── __init__.py
│   └── wrappers.py    # Environment factory (make_env, make_eval_env)
├── models/            # Auto-created — model checkpoints saved here
│   └── best/          # Best checkpoint (by EvalCallback)
├── logs/              # TensorBoard event files
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Python 3.10+ recommended
pip install -r requirements.txt

# Install Atari ROMs (required by ALE)
python -m ale_py.scripts.install_roms --accept-license
# OR
AutoROM --accept-license
```

### 2. Train the Agent

```bash
python train.py
```

Optional flags:
```bash
python train.py --timesteps 5000000         # Shorter run (reaches ~+5 reward)
python train.py --resume models/best        # Resume from best checkpoint
```

Training logs are saved to `logs/`. Monitor with:
```bash
tensorboard --logdir logs/
```

### 3. Evaluate the Agent

```bash
python evaluate.py
```

This opens a **live dashboard** with:
- Left:  the game playing in real-time (with reward overlay)
- Top-right: bar chart of per-episode rewards (updates live)
- Bottom-right: run statistics table

Optional flags:
```bash
python evaluate.py --model models/ppo_pong_final   # Use final model
python evaluate.py --episodes 10                   # Run 10 episodes
python evaluate.py --no-video                      # Skip MP4 recording
python evaluate.py --no-render                     # Headless (video only)
```

---

## Expected Training Progress

| Timesteps | Mean Reward | Notes                                   |
|-----------|-------------|-----------------------------------------|
| ~1M       | −18 to −15  | Agent barely reacts                     |
| ~3M       | −10 to −5   | Agent starts tracking the ball          |
| ~5M       | 0 to +10    | Wins rallies consistently               |
| ~10M      | +15 to +21  | Near-optimal (perfect score = +21)      |

Training 10M steps with 8 parallel envs:
- **CPU only** (8-core): ~3–6 hours
- **Single GPU**:       ~1–2 hours

---

## Configuration

All settings live in `config.py`. Key options:

| Setting          | Default       | Description                        |
|------------------|---------------|------------------------------------|
| `N_ENVS`         | 8             | Parallel training environments     |
| `TOTAL_TIMESTEPS`| 10_000_000    | Total environment steps to train   |
| `LEARNING_RATE`  | 2.5e-4        | PPO learning rate                  |
| `CLIP_RANGE`     | 0.1           | PPO clipping epsilon               |
| `EVAL_FREQ`      | 50_000        | Steps between evaluations          |
| `EVAL_EPISODES`  | 5             | Episodes per evaluate.py run       |
| `RECORD_VIDEO`   | True          | Save MP4 of evaluation             |

---

## How It Works

1. **Env wrappers** (`utils/wrappers.py`):  
   SB3's `make_atari_env` auto-applies grayscale, 84×84 resize, frame-skip (k=4),
   reward clipping, and EpisodicLifeEnv. `VecFrameStack` stacks 4 frames as channels.

2. **PPO** (`train.py`):  
   Uses a CNN policy (Nature DQN architecture: 3 conv layers → 512-unit FC → actor/critic heads).
   8 parallel envs collect 128 steps each per rollout → 4 epochs of gradient updates.

3. **Visualisation** (`evaluate.py`):  
   Grabs `rgb_array` frames from the underlying ALE env, overlays stats with OpenCV,
   and updates a Matplotlib dashboard in real time. Optionally encodes frames to MP4.