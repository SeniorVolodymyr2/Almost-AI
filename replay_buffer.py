from __future__ import annotations

import random
import threading
from collections import deque
from dataclasses import dataclass


@dataclass
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, transition: Transition) -> None:
        with self._lock:
            self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        with self._lock:
            return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
