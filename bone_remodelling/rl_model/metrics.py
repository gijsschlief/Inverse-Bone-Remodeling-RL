"""Container for storing training metrics."""


class MetricsContainer:
    """Container for storing training metrics."""

    def __init__(self) -> None:
        """Initialize the metrics container."""
        self.episode_rewards: list[float] = []
        self.validation_steps: list[int] = []
        self.validation_ssim: list[float] = []
        self.episode_indices: list[int] = []
        self.episode_end_timesteps: list[int] = []
        self.complexity_jumps: list[tuple[int, int]] = []  # List of (step, complexity) when curriculum complexity increases
