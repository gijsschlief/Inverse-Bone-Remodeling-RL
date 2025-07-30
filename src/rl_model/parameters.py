"""Parameters for the RL environement."""

from dataclasses import dataclass


@dataclass
class RLParameters:
    """Parameters used by the RL agent.

    Attributes
    ----------
        max_steps: int
        force_boundary: float
        density_constraint: float
        render_mode: str
        n_steps: int
        batch_size: int
        ent_coef: float
        learning_rate: float
        seed: int

    """

    max_steps: int = 50
    force_boundary: float = 30
    density_constraint: float = 1.73
    n_steps: int = 2048
    batch_size: int = 64
    ent_coef: float = 0.01
    learning_rate: float = 1e-3
    learning_rate_decay: float = 0.5
    minimum_learning_rate: float = 1e-6
    seed: int = 42
    verbose: int = 1
    device: str = "cpu"
    patience: int = 5
