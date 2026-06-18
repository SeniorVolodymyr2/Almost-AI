from __future__ import annotations

from pathlib import Path


def ensure_models_dir(models_dir: str = "models") -> Path:
    path = Path(models_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def episode_model_path(model_name: str, episode: int, models_dir: str = "models") -> Path:
    ensure_models_dir(models_dir)
    return Path(models_dir) / f"{model_name}_{episode}.pt"


def find_latest_model(model_name: str, models_dir: str = "models") -> Path | None:
    directory = Path(models_dir)
    if not directory.exists():
        return None

    candidates = list(directory.glob(f"{model_name}_*.pt"))
    if not candidates:
        return None

    def episode_number(path: Path) -> int:
        suffix = path.stem.rsplit("_", 1)[-1]
        return int(suffix)

    return max(candidates, key=episode_number)


def resolve_load_path(
    model_name: str,
    models_dir: str = "models",
    load_episode: int | None = None,
) -> Path | None:
    if load_episode is not None:
        path = episode_model_path(model_name, load_episode, models_dir)
        return path if path.exists() else None
    return find_latest_model(model_name, models_dir)
