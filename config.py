from __future__ import annotations

from dataclasses import dataclass

from environs import Env


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    model_name: str
    models_dir: str
    load_episode: int | None
    load_latest: bool
    train: bool
    baseline: bool
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    max_episodes: int
    reconnect_delay: float
    save_every_episodes: int
    evaluation_dir: str
    train_every_n_steps: int


def load_settings() -> Settings:
    env = Env()
    env.read_env()

    load_episode_raw = env.str("MODEL_LOAD_EPISODE", "")
    load_episode = int(load_episode_raw) if load_episode_raw.strip() else None

    return Settings(
        host=env.str("HOST", "127.0.0.1"),
        port=env.int("PORT", 5005),
        model_name=env.str("MODEL_NAME", "racing_dqn"),
        models_dir=env.str("MODELS_DIR", "models"),
        load_episode=load_episode,
        load_latest=env.bool("LOAD_LATEST", False),
        train=env.bool("TRAIN", False),
        baseline=env.bool("BASELINE", False),
        epsilon=env.float("EPSILON", 1.0),
        epsilon_min=env.float("EPSILON_MIN", 0.05),
        epsilon_decay=env.float("EPSILON_DECAY", 0.995),
        max_episodes=env.int("MAX_EPISODES", 500),
        reconnect_delay=env.float("RECONNECT_DELAY", 2.0),
        save_every_episodes=env.int("SAVE_EVERY_EPISODES", 25),
        evaluation_dir=env.str("EVALUATION_DIR", "evaluation"),
        train_every_n_steps=env.int("TRAIN_EVERY_N_STEPS", 4),
    )
