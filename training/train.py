from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn, optim

from environment import TetrisEnv
from models import DQN, DuelingDQN
from training.replay_buffer import ReplayBuffer
from utils.plotting import save_training_plots


def select_action(model: nn.Module, state: np.ndarray, epsilon: float, action_size: int, device: torch.device) -> int:
    if random.random() < epsilon:
        return random.randrange(action_size)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(model(state_t).argmax(dim=1).item())


def optimize(
    policy_net: nn.Module,
    target_net: nn.Module,
    replay: ReplayBuffer,
    optimizer: optim.Optimizer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    double_dqn: bool,
) -> float:
    if len(replay) < batch_size:
        return 0.0

    batch = replay.sample(batch_size)
    states = torch.tensor(batch.states, device=device)
    actions = torch.tensor(batch.actions, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.rewards, device=device).unsqueeze(1)
    next_states = torch.tensor(batch.next_states, device=device)
    dones = torch.tensor(batch.dones, device=device).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    with torch.no_grad():
        if double_dqn:
            next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = target_net(next_states).gather(1, next_actions)
        else:
            next_q_values = target_net(next_states).max(dim=1, keepdim=True).values
        targets = rewards + gamma * next_q_values * (1.0 - dones)

    loss = nn.SmoothL1Loss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def train(args: argparse.Namespace) -> Dict[str, List[float]]:
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    env = TetrisEnv(use_hold=args.use_hold, seed=args.seed)
    model_cls = DuelingDQN if args.dueling else DQN
    policy_net = model_cls(env.state_size, env.action_space_n).to(device)
    target_net = model_cls(env.state_size, env.action_space_n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.memory_size)

    rewards: List[float] = []
    losses: List[float] = []
    scores: List[float] = []
    epsilons: List[float] = []
    epsilon = args.epsilon_start
    best_score = -1

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        episode_losses: List[float] = []

        for _ in range(args.max_steps):
            action = select_action(policy_net, state, epsilon, env.action_space_n, device)
            next_state, reward, done, info = env.step(action)
            replay.push(state, action, reward, next_state, done)
            loss = optimize(policy_net, target_net, replay, optimizer, args.batch_size, args.gamma, device, args.double_dqn)
            if loss:
                episode_losses.append(loss)
            state = next_state
            episode_reward += reward
            if done:
                break

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        rewards.append(episode_reward)
        losses.append(float(np.mean(episode_losses)) if episode_losses else 0.0)
        scores.append(float(env.score))
        epsilons.append(epsilon)

        if episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        checkpoint = {
            "episode": episode,
            "model_state_dict": policy_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epsilon": epsilon,
            "state_size": env.state_size,
            "action_size": env.action_space_n,
            "dueling": args.dueling,
            "use_hold": args.use_hold,
        }

        if env.score > best_score:
            best_score = env.score
            torch.save(checkpoint, Path(args.checkpoint_dir) / "best_model.pt")

        if episode % args.save_every == 0:
            torch.save(checkpoint, Path(args.checkpoint_dir) / f"dqn_tetris_ep{episode}.pt")
            save_training_plots(rewards, losses, scores, epsilons, args.plot_dir)

        if episode == 1 or episode % args.log_every == 0:
            avg_reward = float(np.mean(rewards[-args.log_every :]))
            avg_score = float(np.mean(scores[-args.log_every :]))
            print(
                f"Episode {episode:5d} | reward {episode_reward:8.2f} | "
                f"avg_reward {avg_reward:8.2f} | score {env.score:5d} | "
                f"avg_score {avg_score:7.1f} | epsilon {epsilon:.3f}"
            )

    save_training_plots(rewards, losses, scores, epsilons, args.plot_dir)
    final_checkpoint = {
        "episode": args.episodes,
        "model_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epsilon": epsilon,
        "state_size": env.state_size,
        "action_size": env.action_space_n,
        "dueling": args.dueling,
        "use_hold": args.use_hold,
    }
    torch.save(final_checkpoint, Path(args.checkpoint_dir) / "final_model.pt")
    return {"rewards": rewards, "losses": losses, "scores": scores, "epsilons": epsilons}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Tetris.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-size", type=int, default=50000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--plot-dir", type=str, default="training_plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-hold", action="store_true")
    parser.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dueling", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
