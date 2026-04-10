"""
Aggregate training/evaluation CSVs and produce metrics + comparison charts.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare curriculum vs direct Pong experiments")
    parser.add_argument("--metrics-dir", type=str, default="./metrics")
    parser.add_argument("--out-dir", type=str, default="./reports")
    parser.add_argument("--threshold", type=float, default=15.0)
    return parser.parse_args()


def auc(xs, ys):
    if len(xs) < 2:
        return np.nan
    return float(np.trapezoid(ys, xs))


def first_threshold_step(df: pd.DataFrame, threshold: float, step_col: str = "train_step"):
    hit = df[df["mean_reward"] >= threshold]
    if hit.empty:
        return np.nan
    return float(hit.sort_values(step_col).iloc[0][step_col])


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    train_files = glob.glob(os.path.join(args.metrics_dir, "train_*.csv"))
    eval_files = glob.glob(os.path.join(args.metrics_dir, "eval_*.csv"))
    if not train_files:
        raise SystemExit("No training metric CSVs found.")

    train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    eval_df = pd.concat([pd.read_csv(f) for f in eval_files], ignore_index=True) if eval_files else pd.DataFrame()

    train_df = train_df.sort_values(["method", "seed", "train_step"])

    # Target-only trajectory (difficulty 3)
    target_df = train_df[train_df["eval_difficulty"] == 3].copy()

    seed_rows = []
    for (method, seed), g in target_df.groupby(["method", "seed"]):
        g = g.sort_values("target_difficulty_steps")
        seed_rows.append({
            "method": method,
            "seed": seed,
            "auc_target_reward": auc(g["target_difficulty_steps"].values, g["mean_reward"].values),
            "time_to_threshold_total": first_threshold_step(g, args.threshold, "train_step"),
            "time_to_threshold_target": first_threshold_step(g, args.threshold, "target_difficulty_steps"),
            "final_target_reward": float(g.iloc[-1]["mean_reward"]) if len(g) else np.nan,
            "final_target_win_rate": float(g.iloc[-1]["win_rate"]) if len(g) else np.nan,
            "jumpstart_target_reward": float(g.iloc[0]["mean_reward"]) if len(g) else np.nan,
        })
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(os.path.join(args.out_dir, "seed_summary.csv"), index=False)

    method_summary = seed_summary.groupby("method").agg(["mean", "std", "count"])
    method_summary.to_csv(os.path.join(args.out_dir, "method_summary.csv"))

    # Training curves aggregated across seeds
    agg = train_df.groupby(["method", "train_step"], as_index=False).agg(
        mean_reward=("mean_reward", "mean"),
        reward_std=("mean_reward", "std"),
    )
    agg.to_csv(os.path.join(args.out_dir, "aggregated_training_curve.csv"), index=False)

    plt.style.use("dark_background")

    # 1) Overall training curves
    fig, ax = plt.subplots(figsize=(10, 6))
    for method, g in agg.groupby("method"):
        g = g.sort_values("train_step")
        ax.plot(g["train_step"], g["mean_reward"], label=method)
        std = g["reward_std"].fillna(0).values
        ax.fill_between(g["train_step"], g["mean_reward"] - std, g["mean_reward"] + std, alpha=0.2)
    ax.axhline(args.threshold, linestyle="--", alpha=0.6, color="gold", label=f"threshold {args.threshold}")
    ax.set_title("Training curve: mean eval reward vs total steps")
    ax.set_xlabel("Train steps")
    ax.set_ylabel("Mean eval reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "training_curve_total_steps.png"), dpi=180)
    plt.close(fig)

    # 2) Target-only curve by target difficulty steps
    if not target_df.empty:
        agg_target = target_df.groupby(["method", "target_difficulty_steps"], as_index=False).agg(
            mean_reward=("mean_reward", "mean"),
            reward_std=("mean_reward", "std"),
        )
        agg_target.to_csv(os.path.join(args.out_dir, "aggregated_target_curve.csv"), index=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        for method, g in agg_target.groupby("method"):
            g = g.sort_values("target_difficulty_steps")
            ax.plot(g["target_difficulty_steps"], g["mean_reward"], label=method)
            std = g["reward_std"].fillna(0).values
            ax.fill_between(g["target_difficulty_steps"], g["mean_reward"] - std, g["mean_reward"] + std, alpha=0.2)
        ax.axhline(args.threshold, linestyle="--", alpha=0.6, color="gold")
        ax.set_title("Target-task curve: mean eval reward vs difficulty-3 steps")
        ax.set_xlabel("Difficulty-3 training steps")
        ax.set_ylabel("Mean eval reward")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "target_curve_difficulty3_steps.png"), dpi=180)
        plt.close(fig)

    # 3) Summary bars
    if not seed_summary.empty:
        metrics = ["final_target_reward", "auc_target_reward", "time_to_threshold_total", "jumpstart_target_reward"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for ax, metric in zip(axes, metrics):
            stats = seed_summary.groupby("method")[metric].agg(["mean", "std"])
            ax.bar(stats.index, stats["mean"], yerr=stats["std"].fillna(0), capsize=5)
            ax.set_title(metric.replace("_", " "))
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "summary_bars.png"), dpi=180)
        plt.close(fig)

    # 4) Cross-difficulty robustness heatmap from eval suite
    if not eval_df.empty:
        summary_eval = eval_df.groupby(["method", "difficulty", "mode", "rap"], as_index=False).agg(
            mean_reward=("reward", "mean"),
            std_reward=("reward", "std"),
            win_rate=("reward", lambda s: float(np.mean(np.array(s) > 0))),
        )
        summary_eval.to_csv(os.path.join(args.out_dir, "eval_suite_summary.csv"), index=False)

        pivot = summary_eval[(summary_eval["mode"] == 0) & (summary_eval["rap"] == 0.25)].pivot(
            index="method", columns="difficulty", values="mean_reward"
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), labels=list(pivot.index))
        ax.set_title("Cross-difficulty mean reward (mode=0, RAP=0.25)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.values[i, j]:+.1f}", ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "cross_difficulty_heatmap.png"), dpi=180)
        plt.close(fig)

    # 5) Markdown summary
    with open(os.path.join(args.out_dir, "comparison_report.md"), "w") as f:
        f.write("# PPO Pong Curriculum vs Direct Comparison\n\n")
        f.write("## Generated outputs\n\n")
        f.write("- `seed_summary.csv`: per-seed metrics\n")
        f.write("- `method_summary.csv`: aggregated method metrics\n")
        f.write("- `training_curve_total_steps.png`: learning curve over total steps\n")
        f.write("- `target_curve_difficulty3_steps.png`: target-task-only curve\n")
        f.write("- `summary_bars.png`: final reward, AUC, time-to-threshold, jumpstart\n")
        if not eval_df.empty:
            f.write("- `cross_difficulty_heatmap.png`: cross-difficulty robustness\n")
            f.write("- `eval_suite_summary.csv`: aggregated evaluation suite results\n")


if __name__ == "__main__":
    main()
