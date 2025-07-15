"""The density updater is responsible for updating the density each timestep when running the forward simulation."""

import numpy as np
from bone_remodeling.forward_model.density_parameters import SimulationParameters


class DensityUpdater:
    """The DensityUpdater class manages the density update process during the simulation.

    It applies the density update rule based on the specific energy density (SED)
    and the parameters defined in SimulationParameters. The updater tracks the
    convergence of each cell and updates the density accordingly.
    """

    def __init__(self, simulation_parameters: SimulationParameters) -> None:
        """Initialize the DensityUpdater with simulation parameters.

        Args:
        ----
            simulation_parameters (SimulationParameters): Parameters for the simulation.

        """
        self.density = simulation_parameters.initial_density_field.copy()
        self.active_cells = np.ones_like(self.density, dtype=bool)
        self.convergence_counter = np.zeros_like(self.density, dtype=int)

        self.dt = simulation_parameters.dt
        self.remodeling_rate_coefficient = simulation_parameters.remodeling_rate_coefficient
        self.stimulus_threshold = simulation_parameters.stimulus_threshold
        self.min_density = simulation_parameters.min_density
        self.max_density = simulation_parameters.max_density
        self.convergence_tolerance = simulation_parameters.convergence_tolerance
        self.convergence_after_steps = simulation_parameters.convergence_after_steps

    def _update_active_cells(self, delta: np.ndarray) -> None:
        """Check which cells are still active based on changes in density."""
        cells_converged = np.abs(delta) < self.convergence_tolerance
        self.convergence_counter[self.active_cells & cells_converged] += 1
        self.convergence_counter[self.active_cells & ~cells_converged] = 0

        cells_converged = (
            self.convergence_counter >= self.convergence_after_steps
        )
        self.active_cells &= ~cells_converged

    def update(self, strain_energy_density: np.ndarray) -> np.ndarray:
        """Apply the density update rule to the current density.

        Args:
        ----
            strain_energy_density (np.ndarray): Specific energy density (SED) array, shape=(n_cells,).

        Returns:
        -------
            np.ndarray: Updated density array.

        """
        stimulus = np.zeros_like(self.density)
        stimulus[self.active_cells] = (strain_energy_density[self.active_cells] / self.density[self.active_cells])

        delta = self.remodeling_rate_coefficient * (stimulus - self.stimulus_threshold)

        self.density[self.active_cells] = self.density[self.active_cells] + self.dt * delta[self.active_cells]
        self.density = np.clip(self.density, self.min_density, self.max_density)

        self._update_active_cells(delta)
        return self.density
