"""Parameters for the RL environement."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RLParameters:
    """Parameters used by the RL agent.

    Attributes
    ----------
        max_steps: int
        force_boundary: float
        density_constraint: float
        n_steps: int
        batch_size: int
        ent_coef: float
        learning_rate: float
        learning_rate_decay: float
        minimum_learning_rate: float
        seed: int
        verbose: int
        device: str
        patience: int
        per_step_dead_zone: float
        per_step_force_change: float
        per_step_location_change: float

    """

    max_steps: int = 25
    force_boundary: float = 200
    density_constraint: float = 1.73
    n_steps: int = 2048
    batch_size: int = 64
    ent_coef: float = 0.01
    learning_rate: float = 1e-3
    learning_rate_decay: float = 0.1
    minimum_learning_rate: float = 1e-5
    seed: int = 42
    verbose: int = 1
    device: str = "cpu"
    patience: int = 20
    per_step_dead_zone: float = 1.0
    per_step_force_change: float = 10.0
    per_step_location_change: float = 1.0

@dataclass
class RunConfiguration:
    """Configuration for running the RL training.

    Attributes
    ----------
        output_dir: Path
        forward_type: str
        total_timesteps: int
        render_freq: int
        validation_frequency: int
        reward_plot_path: Path | None

    """

    output_dir: Path
    forward_type: str = "surrogate"
    total_timesteps: int = 20_000_001
    render_frequency: int = 9999
    validation_frequency: int = 200_000
    validation_size: int = 100
    number_of_environments: int = 10
    random_state: int = 0
    reward_plot_path: Path | None = field(default=None)
    agent_path: Path | None = field(default=None)
    data_path: Path | None = field(default=None)
    surrogate_path: Path | None = field(default=None)

    def __post_init__(self) -> None:
        """Initialize path fields based on output_dir."""
        self.output_dir.resolve()
        if self.reward_plot_path is None:
            self.reward_plot_path = self.output_dir / Path("figures", "reward_curve_extra_action_ensemble_RL_20mil.png")
        if self.agent_path is None:
            self.agent_path = self.output_dir / Path("agents", "surrogate_agent_extra_action_best_20mil.zip")
        if self.data_path is None:
            self.data_path = self.output_dir / Path("raw", "triangular")
        if self.surrogate_path is None:
            self.surrogate_path = self.output_dir / Path("surrogate_models", "surrogate.pth")
