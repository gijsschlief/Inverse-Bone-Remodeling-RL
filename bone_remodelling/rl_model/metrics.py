"""Container for storing training metrics."""

from dataclasses import dataclass


@dataclass
class ValidationSnapshot:
    """A single record of the agent's state at a specific point in time."""

    step: int
    ssim: float
    complexity: int
    learning_rate: float
    patience: int


class MetricsContainer:
    """The collection of all snapshots throughout training."""

    def __init__(self) -> None:
        """Initialize the metrics container."""
        self.history: list[ValidationSnapshot] = []
        self.complexity_jumps: list[tuple[int, int]] = []
        self.episode_rewards: list[float] = []
        self.episode_end_timesteps: list[int] = []

    def add_snapshot(self, snapshot: ValidationSnapshot) -> None:
        """Add a new snapshot to the history."""
        self.history.append(snapshot)

    def resume_from_history(self, stats_dict: dict) -> None:
        """Convert list of dicts back into ValidationSnapshot objects."""
        history_data = stats_dict.get("history", [])
        self.history = [ValidationSnapshot(**s) for s in history_data]

        jumps = stats_dict.get("complexity_jumps", [])
        self.complexity_jumps = [tuple(j) for j in jumps]

        self.episode_rewards = stats_dict.get("episode_rewards", [])
        self.episode_end_timesteps = stats_dict.get("episode_end_timesteps", [])

    @property
    def validation_steps(self) -> list[int]:
        """Get a list of validation steps from the history."""
        return [s.step for s in self.history]

    @property
    def validation_ssim(self) -> list[float]:
        """Get a list of validation SSIM values from the history."""
        return [s.ssim for s in self.history]
