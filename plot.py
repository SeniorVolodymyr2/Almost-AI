from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tracker import EpisodeRecord


def save_training_chart(records: list[EpisodeRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for ax, title in zip(axes, ("Reward per episode", "Score per episode"), strict=True):
            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return

    episodes = [record.episode for record in records]
    rewards = [record.total_reward for record in records]
    max_rewards = [record.max_reward_so_far for record in records]
    scores = [record.score for record in records]
    max_scores = [record.max_score_so_far for record in records]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(episodes, rewards, marker="o", markersize=3, linewidth=1, label="Reward / episode")
    axes[0].plot(episodes, max_rewards, linewidth=2, label="Max reward / episode")
    axes[0].set_title("Reward per episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes, scores, marker="o", markersize=3, linewidth=1, label="Score / episode")
    axes[1].plot(episodes, max_scores, linewidth=2, label="Max score / episode")
    axes[1].set_title("Score per episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
