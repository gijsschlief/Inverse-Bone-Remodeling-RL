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

    """

    max_steps: int = 50
    force_boundary: float = 30
    density_constraint: float = 1.73
    render_mode: str = "human"
