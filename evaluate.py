"""Evaluate a trained PPO Pong agent under configurable difficulty/mode/RAP settings."""

import argparse
import csv
import os
import sys
import time

import cv2
import imageio
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from config import Config
from utils.wrappers import make_eval_env

C_BG = "#0d1117"
C_SURFACE = "#161b22"
C_BORDER = "#30363d"
C_TEXT = "#e6edf3"
C_MUTED = "#8b949e"
C_GREEN = "#3fb950"
C_RED = "#f85149"
C_TEAL = "#4f98a3"
C_GOLD = "#e8af34"
C_ORANGE = "#f0883e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO Pong agent")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--difficulty", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1])
    parser.add_argument("--rap", type=float, default=0.25)
    parser.add_argument("--method", type=str, default="unknown")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-out", type=str, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def bar_color(reward: float) -> str:
    if reward > 0:
        return C_GREEN
    if reward < 0:
        return C_RED
    return C_GOLD


def draw_reward_chart(ax, episode_rewards, current_reward):
    ax.clear()
    ax.set_facecolor(C_SURFACE)
    ax.spines[:].set_color(C_BORDER)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    all_rewards = episode_rewards + [current_reward]
    colours = [bar_color(r) for r in all_rewards]
    if colours:
        colours[-1] = C_TEAL
    ax.bar(range(1, len(all_rewards) + 1), all_rewards, color=colours, width=0.7)
    ax.axhline(0, color=C_BORDER, linewidth=0.8)
    if episode_rewards:
        mean = np.mean(episode_rewards)
        ax.axhline(mean, color=C_GOLD, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_ylim(-22, 22)
    ax.set_title("Episode Rewards", color=C_TEXT, fontsize=9)


def draw_stats_panel(ax, episode_rewards, current_reward, step, start_time, difficulty, mode, rap):
    ax.clear()
    ax.set_facecolor(C_SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    elapsed = time.time() - start_time
    completed = episode_rewards if episode_rewards else []
    mean_rew = np.mean(completed) if completed else 0.0
    lines = [
        ("Current", f"{current_reward:+.0f}"),
        ("Mean", f"{mean_rew:+.2f}"),
        ("Step", f"{step:,}"),
        ("Elapsed", f"{elapsed:.0f}s"),
        ("Difficulty", str(difficulty)),
        ("Mode", str(mode)),
        ("RAP", f"{rap:.2f}"),
    ]
    y = 0.9
    for label, value in lines:
        ax.text(0.08, y, label, color=C_MUTED, fontsize=8, va="top", fontfamily="monospace")
        ax.text(0.92, y, value, color=C_TEXT, fontsize=8, va="top", ha="right", fontfamily="monospace")
        y -= 0.12


def run_evaluation(model_path, episodes, difficulty, mode, rap, save_video, headless, video_path, fps):
    cfg = Config()
    env = make_eval_env(cfg.ENV_ID, cfg.N_STACK, seed=999, render=True,
                        difficulty=difficulty, mode=mode,
                        repeat_action_probability=rap)
    model = PPO.load(model_path, env=env)
    if not headless:
        plt.ion()
    fig = plt.figure(figsize=(14, 7), facecolor=C_BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.1, 1], height_ratios=[2, 1],
                           hspace=0.35, wspace=0.3, left=0.04, right=0.97, top=0.95, bottom=0.06)
    ax_game = fig.add_subplot(gs[:, 0])
    ax_chart = fig.add_subplot(gs[0, 1])
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_game.axis("off")
    fig.suptitle(f"PPO Pong Eval | D{difficulty} M{mode} RAP {rap:.2f}", color=C_TEXT, fontsize=10, fontweight="bold")

    writer = None
    if save_video:
        writer = imageio.get_writer(video_path, fps=fps, macro_block_size=1)

    rewards = []
    start_time = time.time()
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)
            ep_reward += float(reward[0])
            step += 1
            raw_frame = env.get_images()
            frame = raw_frame[0] if raw_frame else np.zeros((210, 160, 3), dtype=np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, f"EP {ep}/{episodes}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"Reward: {ep_reward:+.0f}", (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,230,100), 1, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"D{difficulty} M{mode} RAP{rap:.2f}", (4, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,180,80), 1, cv2.LINE_AA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if writer is not None:
                writer.append_data(frame_rgb)
            if step % 4 == 0 and not headless:
                ax_game.clear()
                ax_game.imshow(frame_rgb, aspect="auto")
                ax_game.axis("off")
                ax_game.set_title(f"Episode {ep} | Step {step:,}", color=C_TEXT, fontsize=9)
                draw_reward_chart(ax_chart, rewards, ep_reward)
                draw_stats_panel(ax_stats, rewards, ep_reward, step, start_time, difficulty, mode, rap)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
            done = bool(terminated[0])
        rewards.append(ep_reward)
        print(f"[eval] episode {ep}/{episodes} -> {ep_reward:+.0f}")

    if writer is not None:
        writer.close()
    env.close()

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    win_rate = float(np.mean(np.array(rewards) > 0))
    return rewards, mean_reward, std_reward, win_rate


def main() -> None:
    args = parse_args()
    cfg = Config()
    if not os.path.exists(args.model) and not os.path.exists(args.model + ".zip"):
        print(f"Model not found: {args.model}")
        sys.exit(1)
    episodes = args.episodes or cfg.EVAL_EPISODES
    rewards, mean_reward, std_reward, win_rate = run_evaluation(
        args.model, episodes, args.difficulty, args.mode, args.rap,
        save_video=(not args.no_video) and cfg.RECORD_VIDEO,
        headless=args.no_render,
        video_path=cfg.VIDEO_PATH,
        fps=cfg.VIDEO_FPS,
    )
    print(f"Mean reward: {mean_reward:+.2f}")
    print(f"Std reward : {std_reward:.2f}")
    print(f"Win rate   : {win_rate:.2%}")

    if args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method","seed","difficulty","mode","rap","episode","reward"])
            writer.writeheader()
            for i, reward in enumerate(rewards, start=1):
                writer.writerow({
                    "method": args.method,
                    "seed": args.seed,
                    "difficulty": args.difficulty,
                    "mode": args.mode,
                    "rap": args.rap,
                    "episode": i,
                    "reward": reward,
                })


if __name__ == "__main__":
    main()
