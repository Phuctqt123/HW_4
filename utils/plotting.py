from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: List[float], window: int = 50) -> np.ndarray:
    if not values:
        return np.array([])
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_training_plots(
    rewards: List[float],
    losses: List[float],
    scores: List[float],
    epsilons: List[float],
    output_dir: str = "training_plots",
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    episodes = np.arange(1, len(rewards) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(episodes, rewards, alpha=0.45, label="Reward")
    avg_rewards = moving_average(rewards)
    if len(avg_rewards):
        axes[0, 0].plot(np.arange(len(avg_rewards)) + 1, avg_rewards, label="Moving avg")
    axes[0, 0].set_title("Reward per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].legend()

    axes[0, 1].plot(episodes, losses, color="tab:red")
    axes[0, 1].set_title("Training Loss")
    axes[0, 1].set_xlabel("Episode")

    axes[1, 0].plot(episodes, scores, color="tab:green", alpha=0.75)
    axes[1, 0].set_title("Score per Episode")
    axes[1, 0].set_xlabel("Episode")

    axes[1, 1].plot(episodes, epsilons, color="tab:purple")
    axes[1, 1].set_title("Epsilon Decay")
    axes[1, 1].set_xlabel("Episode")

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "training_metrics.png", dpi=160)
    plt.close(fig)
