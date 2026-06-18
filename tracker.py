from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EpisodeRecord:
    episode: int
    unity_episode: int
    score: int
    total_reward: float
    epsilon: float
    max_score_so_far: int
    max_reward_so_far: float


@dataclass
class MetricsTracker:
    evaluation_dir: str
    model_name: str
    records: list[EpisodeRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.dir_path = Path(self.evaluation_dir)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.metrics_csv = self.dir_path / f"{self.model_name}.metrics.csv"
        self.chart_path = self.dir_path / f"{self.model_name}_training.png"
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.metrics_csv.exists():
            return
        with self.metrics_csv.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.records.append(
                    EpisodeRecord(
                        episode=int(row["episode"]),
                        unity_episode=int(row["unity_episode"]),
                        score=int(row["score"]),
                        total_reward=float(row["total_reward"]),
                        epsilon=float(row["epsilon"]),
                        max_score_so_far=int(row["max_score_so_far"]),
                        max_reward_so_far=float(row["max_reward_so_far"]),
                    )
                )

    def record_episode(
        self,
        unity_episode: int,
        score: int,
        total_reward: float,
        epsilon: float,
    ) -> EpisodeRecord:
        episode_number = len(self.records) + 1
        max_score = max((r.score for r in self.records), default=0)
        max_score_so_far = max(max_score, score)
        max_reward = max((r.total_reward for r in self.records), default=float("-inf"))
        max_reward_so_far = max(max_reward, total_reward)

        record = EpisodeRecord(
            episode=episode_number,
            unity_episode=unity_episode,
            score=score,
            total_reward=total_reward,
            epsilon=epsilon,
            max_score_so_far=max_score_so_far,
            max_reward_so_far=max_reward_so_far,
        )
        self.records.append(record)
        self._append_csv(record)
        return record

    def _append_csv(self, record: EpisodeRecord) -> None:
        write_header = not self.metrics_csv.exists()
        with self.metrics_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "episode",
                    "unity_episode",
                    "score",
                    "total_reward",
                    "epsilon",
                    "max_score_so_far",
                    "max_reward_so_far",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "episode": record.episode,
                    "unity_episode": record.unity_episode,
                    "score": record.score,
                    "total_reward": f"{record.total_reward:.4f}",
                    "epsilon": f"{record.epsilon:.4f}",
                    "max_score_so_far": record.max_score_so_far,
                    "max_reward_so_far": f"{record.max_reward_so_far:.4f}",
                }
            )

    def update_chart(self) -> Path:
        from plot import save_training_chart

        save_training_chart(self.records, self.chart_path)
        return self.chart_path
