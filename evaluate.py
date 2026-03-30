"""
evaluate.py — Run a trained PPO agent on Pong with live visualisation.

Usage:
    python evaluate.py                              # uses models/best/best_model
    python evaluate.py --model models/ppo_pong_final
    python evaluate.py --model models/best/best_model --episodes 10 --no-video

Layout:
  ┌──────────────────────────┬─────────────────────────┐
  │   GAME FRAME             │  Episode Reward History │
  │   (live, with overlays)  │  (bar chart, live)      │
  │                          ├─────────────────────────┤
  │                          │  Run Statistics         │
  └──────────────────────────┴─────────────────────────┘

Press Q in the matplotlib window to quit early.
"""

import argparse
import os
import sys
import time

import imageio
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO

from config import Config
from utils.wrappers import make_eval_env


# ── Colour palette (matches terminal aesthetic) ───────────────────────────────
C_BG      = "#0d1117"
C_SURFACE = "#161b22"
C_BORDER  = "#30363d"
C_TEXT    = "#e6edf3"
C_MUTED   = "#8b949e"
C_GREEN   = "#3fb950"
C_RED     = "#f85149"
C_TEAL    = "#4f98a3"
C_GOLD    = "#e8af34"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO Pong agent")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model .zip (default: models/best/best_model)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of episodes (default: Config.EVAL_EPISODES)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip saving an MP4 recording")
    parser.add_argument("--no-render", action="store_true",
                        help="Headless mode — no matplotlib window (video only)")
    return parser.parse_args()


def bar_color(reward: float) -> str:
    """Map reward to a colour: green (positive) / red (negative) / gold (zero)."""
    if reward > 0:
        return C_GREEN
    if reward < 0:
        return C_RED
    return C_GOLD


def draw_stats_panel(ax, episode_rewards: list[float], current_ep: int,
                     total_eps: int, current_reward: float,
                     step: int, start_time: float) -> None:
    """Render the bottom-right statistics panel."""
    ax.clear()
    ax.set_facecolor(C_SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    elapsed   = time.time() - start_time
    mean_rew  = np.mean(episode_rewards) if episode_rewards else 0.0
    best_rew  = max(episode_rewards)     if episode_rewards else 0.0
    worst_rew = min(episode_rewards)     if episode_rewards else 0.0

    lines = [
        ("Episode",       f"{current_ep} / {total_eps}"),
        ("Current Reward",f"{current_reward:+.0f}"),
        ("Mean Reward",   f"{mean_rew:+.1f}"),
        ("Best",          f"{best_rew:+.0f}"),
        ("Worst",         f"{worst_rew:+.0f}"),
        ("Steps (ep)",    f"{step:,}"),
        ("Elapsed",       f"{elapsed:.0f}s"),
    ]

    y = 0.88
    for label, value in lines:
        ax.text(0.08, y, label, color=C_MUTED,  fontsize=9,  va="top",
                fontfamily="monospace")
        ax.text(0.92, y, value, color=C_TEXT,   fontsize=9,  va="top",
                ha="right", fontfamily="monospace", fontweight="bold")
        y -= 0.13

    # Perfect-score indicator
    if episode_rewards and max(episode_rewards) >= 18:
        ax.text(0.5, 0.04, "★ Near-Perfect Score!", color=C_GOLD,
                fontsize=8, ha="center", va="bottom", fontweight="bold")


def draw_reward_chart(ax, episode_rewards: list[float],
                      current_reward: float, current_ep: int) -> None:
    """Live bar chart of per-episode rewards."""
    ax.clear()
    ax.set_facecolor(C_SURFACE)
    ax.spines[:].set_color(C_BORDER)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)

    all_rewards = episode_rewards + [current_reward]
    episodes    = list(range(1, len(all_rewards) + 1))
    colours     = [bar_color(r) for r in all_rewards]

    # Mark incomplete (current) episode differently
    if colours:
        colours[-1] = C_TEAL

    ax.bar(episodes, all_rewards, color=colours, width=0.7, zorder=2)

    # Zero line
    ax.axhline(0, color=C_BORDER, linewidth=0.8, zorder=1)

    # Mean line (completed episodes only)
    if episode_rewards:
        mean = np.mean(episode_rewards)
        ax.axhline(mean, color=C_GOLD, linewidth=1.2,
                   linestyle="--", zorder=3, alpha=0.8)
        ax.text(len(all_rewards) + 0.3, mean, f"μ={mean:+.1f}",
                color=C_GOLD, fontsize=7, va="center")

    ax.set_xlim(0.3, max(len(all_rewards) + 1.2, 6))
    ax.set_ylim(-22, 22)
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Reward",  fontsize=8)
    ax.set_title("Episode Rewards", color=C_TEXT, fontsize=9, pad=6)

    legend = [
        mpatches.Patch(color=C_TEAL,  label="Current"),
        mpatches.Patch(color=C_GREEN, label="Win (+)"),
        mpatches.Patch(color=C_RED,   label="Loss (−)"),
    ]
    ax.legend(handles=legend, fontsize=7, facecolor=C_BG,
              edgecolor=C_BORDER, labelcolor=C_TEXT,
              loc="lower right", framealpha=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def run_evaluation(model_path: str, n_episodes: int,
                   save_video: bool, headless: bool,
                   video_path: str, fps: int) -> list[float]:
    """
    Run the agent for n_episodes, displaying a live matplotlib dashboard
    and optionally saving an MP4 recording.
    """
    cfg = Config()
    env = make_eval_env(cfg.ENV_ID, cfg.N_STACK, seed=99, render=True)

    print(f"[eval] Loading model: {model_path}")
    model = PPO.load(model_path, env=env)

    # ── Figure layout ─────────────────────────────────────────────────────────
    if not headless:
        plt.ion()

    fig = plt.figure(figsize=(14, 7), facecolor=C_BG)
    fig.canvas.manager.set_window_title("Pong RL — Evaluation")  # type: ignore[attr-defined]

    gs = gridspec.GridSpec(
        2, 2,
        figure     = fig,
        width_ratios  = [1.1, 1],
        height_ratios = [2,   1],
        hspace = 0.35,
        wspace = 0.3,
        left   = 0.04, right = 0.97,
        top    = 0.95, bottom = 0.06,
    )

    ax_game  = fig.add_subplot(gs[:, 0])   # full left column — game frame
    ax_chart = fig.add_subplot(gs[0, 1])   # top-right — reward chart
    ax_stats = fig.add_subplot(gs[1, 1])   # bottom-right — stats table

    for ax in (ax_game, ax_chart, ax_stats):
        ax.set_facecolor(C_SURFACE)

    ax_game.axis("off")

    fig.suptitle("PPO Agent — Atari Pong Evaluation",
                 color=C_TEXT, fontsize=11, fontweight="bold", y=0.99)

    # ── Video writer ──────────────────────────────────────────────────────────
    video_writer = None
    if save_video:
        video_writer = imageio.get_writer(video_path, fps=fps, macro_block_size=1)
        print(f"[eval] Recording to: {video_path}")

    # ── Run episodes ──────────────────────────────────────────────────────────
    episode_rewards: list[float] = []
    start_time = time.time()
    quit_early = False

    for ep in range(1, n_episodes + 1):
        obs           = env.reset()
        episode_reward = 0.0
        step          = 0
        done          = False

        print(f"[eval] Episode {ep}/{n_episodes}", end="", flush=True)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)

            episode_reward += float(reward[0])
            step           += 1

            # Grab raw RGB frame from the underlying ALE env
            raw_frame = env.get_images()   # list of HxWx3 uint8 arrays
            frame     = raw_frame[0] if raw_frame else np.zeros((210, 160, 3), dtype=np.uint8)

            # ── Overlay text on the frame ──────────────────────────────────
            import cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, f"EP {ep}/{n_episodes}",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"Reward: {episode_reward:+.0f}",
                        (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,230,100), 1, cv2.LINE_AA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if video_writer is not None:
                video_writer.append_data(frame_rgb)

            # ── Update matplotlib every 4 frames ──────────────────────────
            if step % 4 == 0 and not headless:
                ax_game.clear()
                ax_game.imshow(frame_rgb, aspect="auto")
                ax_game.axis("off")
                ax_game.set_title(f"Episode {ep}  |  Step {step:,}",
                                  color=C_TEXT, fontsize=9, pad=4)

                draw_reward_chart(ax_chart, episode_rewards, episode_reward, ep)
                draw_stats_panel(ax_stats, episode_rewards, ep, n_episodes,
                                 episode_reward, step, start_time)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

                # Check for Q key press to quit
                if not plt.fignum_exists(fig.number):
                    quit_early = True
                    break

            if terminated[0]:
                done = True

        episode_rewards.append(episode_reward)
        print(f" → reward: {episode_reward:+.0f}")

        if quit_early:
            print("[eval] Window closed — stopping early.")
            break

    # ── Final frame ───────────────────────────────────────────────────────────
    if not headless and plt.fignum_exists(fig.number):
        draw_reward_chart(ax_chart, episode_rewards[:-1], episode_rewards[-1], ep)
        draw_stats_panel(ax_stats, episode_rewards[:-1], len(episode_rewards),
                         n_episodes, episode_rewards[-1], step, start_time)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.ioff()
        plt.show()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\\n" + "─" * 40)
    print(f"  Episodes:    {len(episode_rewards)}")
    print(f"  Mean reward: {np.mean(episode_rewards):+.2f}")
    print(f"  Std reward:  {np.std(episode_rewards):.2f}")
    print(f"  Best:        {max(episode_rewards):+.0f}")
    print(f"  Worst:       {min(episode_rewards):+.0f}")
    print("─" * 40)

    if video_writer is not None:
        video_writer.close()
        print(f"[eval] Video saved → {video_path}")

    env.close()
    return episode_rewards


def main() -> None:
    args = parse_args()
    cfg  = Config()

    model_path = args.model or os.path.join(cfg.BEST_MODEL_DIR, "best_model")
    n_episodes = args.episodes or cfg.EVAL_EPISODES
    save_video = (not args.no_video) and cfg.RECORD_VIDEO
    headless   = args.no_render

    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"[eval] ERROR: Model not found at '{model_path}'")
        print("[eval] Train first:  python train.py")
        sys.exit(1)

    run_evaluation(
        model_path = model_path,
        n_episodes = n_episodes,
        save_video = save_video,
        headless   = headless,
        video_path = cfg.VIDEO_PATH,
        fps        = cfg.VIDEO_FPS,
    )


if __name__ == "__main__":
    main()