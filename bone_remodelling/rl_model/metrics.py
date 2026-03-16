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

    def add_snapshot(self, snapshot: ValidationSnapshot) -> None:
        """Add a new snapshot to the history."""
        self.history.append(snapshot)

    def resume_from_history(self, history_data: list[dict], jumps: list) -> None:
        """Convert list of dicts back into ValidationSnapshot objects."""
        self.history = [ValidationSnapshot(**s) for s in history_data]
        self.complexity_jumps = [tuple(j) for j in jumps]

    @property
    def validation_steps(self) -> list[int]:
        """Get a list of validation steps from the history."""
        return [s.step for s in self.history]

    @property
    def validation_ssim(self) -> list[float]:
        """Get a list of validation SSIM values from the history."""
        return [s.ssim for s in self.history]
