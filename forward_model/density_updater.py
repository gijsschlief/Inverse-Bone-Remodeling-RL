"""The density updater is responsible for updating the density each timestep when running the forward simulation."""

import numpy as np


class DensityUpdater:
    """The DensityUpdater class manages the density update process during the simulation.

    It applies the density update rule based on the specific energy density (SED)
    and the parameters defined in SimulationParameters. The updater tracks the
    convergence of each cell and updates the density accordingly.
    """

    def __init__(self,
        initial_density: np.ndarray,
        dt: float,
        remodeling_rate_coefficient: float,
        stimulus_threshold: float,
        min_density: float,
        max_density: float,
        convergence_tolerance: float,
        convergence_after_steps: int,
    ) -> None:
        """Initialize the DensityUpdater with simulation parameters.

        Args:
        ----
            initial_density (np.ndarray): Initial density field.
            dt (float): Time step for the simulation.
            remodeling_rate_coefficient (float): Coefficient for the remodeling rate.
            stimulus_threshold (float): Threshold for the stimulus.
            min_density (float): Minimum allowed density.
            max_density (float): Maximum allowed density.
            convergence_tolerance (float): Tolerance for convergence checks.
            convergence_after_steps (int): Number of steps to consider for convergence.

        """
        self._initial_density = initial_density.copy()
        self.density = self._initial_density.copy()
        self.active_cells = np.ones_like(self.density, dtype=bool)
        self.convergence_counter = np.zeros_like(self.density, dtype=int)

        self.dt = dt
        self.remodeling_rate_coefficient = remodeling_rate_coefficient
        self.stimulus_threshold = stimulus_threshold
        self.min_density = min_density
        self.max_density = max_density
        self.convergence_tolerance = convergence_tolerance
        self.convergence_after_steps = convergence_after_steps

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
        np.clip(self.density, self.min_density, self.max_density, out=self.density)

        self._update_active_cells(delta)
        return self.density

    def convergenced(self) -> bool:
        """Check which cells have converged based on the convergence counter.

        Returns
        -------
            bool: True if all cells have converged, False otherwise.

        """
        return bool(np.all(~self.active_cells))

    def reset(self) -> None:
        """Reset the density updater to its initial state."""
        self.density[:] = self._initial_density
        self.active_cells[:] = True
        self.convergence_counter[:] = 0
