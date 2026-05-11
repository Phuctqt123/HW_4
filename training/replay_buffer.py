from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np


@dataclass
class Batch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Batch:
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return Batch(
            states=states.astype(np.float32),
            actions=actions.astype(np.int64),
            rewards=rewards.astype(np.float32),
            next_states=next_states.astype(np.float32),
            dones=dones.astype(np.float32),
        )

    def __len__(self) -> int:
        return len(self.memory)
