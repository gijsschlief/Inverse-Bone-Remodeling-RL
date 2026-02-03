"""Constants and parameters used by multiple modules in bone_remodelling package."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConfigurationParameters:
    """Parameters for bone remodelling simulations."""

    force_top_resolution: int = 10
    force_side_resolution: int = 10
    mesh_top_resolution: int = 10
    mesh_side_resolution: int = 10

    seed: int = 1
    min_density: float = 0.01
    max_density: float = 1.74
    start_density: float = 0.87

    output_dir: Path = Path("data")

    def __post_init__(self) -> None:
        """Validate the configuration parameters for bone remodelling simulations."""
        if self.min_density >= self.max_density:
            raise ValueError("min_density must be < max_density")
