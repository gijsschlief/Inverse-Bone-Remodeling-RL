"""The density updater is responsible for updating the density each timestep when running the forward simulation."""

import numpy as np


class DensityUpdater:
    """The DensityUpdater class manages the density update process during the simulation.

    It applies the density update rule based on the specific energy density (SED)
    and the parameters defined in SimulationParameters. The updater tracks the
    convergence of each cell and updates the density accordingly.
    """

    def __init__(
        self,
        initial_density: np.ndarray,
        dt: float,
        remodeling_rate_coefficient: float,
        stimulus_threshold: float,
        min_density: float,
        max_density: float,
        convergence_tolerance: float = 1e-6,
        convergence_tolerance_decay: float = 1.06,
        convergence_after_steps: int = 10,
        convergence_steps_decay: float = 0.97,
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
            convergence_tolerance_decay (float): Decay factor for convergence tolerance.
            convergence_after_steps (int): Number of steps to consider for convergence.
            convergence_steps_decay (float): Decay factor for convergence.

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
        self.convergence_tolerance_decay = convergence_tolerance_decay
        self.convergence_after_steps = convergence_after_steps
        self.convergence_steps_decay = convergence_steps_decay

        self.tolerance_decayed = float(self.convergence_tolerance)
        self.convergence_decayed = float(self.convergence_after_steps)

    def _decay(self) -> None:
        """Decay the convergence counter for all active cells."""
        self.convergence_decayed = (
            self.convergence_decayed * self.convergence_steps_decay
        )
        self.tolerance_decayed = (
            self.tolerance_decayed * self.convergence_tolerance_decay
        )

    def _update_active_cells(self, delta: np.ndarray) -> None:
        """Check which cells are still active based on changes in density."""
        cells_converged = np.abs(delta) < self.tolerance_decayed
        self.convergence_counter[self.active_cells & cells_converged] += 1
        self.convergence_counter[self.active_cells & ~cells_converged] = 0

        cells_converged = self.convergence_counter >= np.ceil(self.convergence_decayed)
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
        self._decay()

        stimulus = np.zeros_like(self.density)
        stimulus[self.active_cells] = (
            strain_energy_density[self.active_cells] / self.density[self.active_cells]
        )

        delta = self.remodeling_rate_coefficient * (stimulus - self.stimulus_threshold)

        self.density[self.active_cells] = (
            self.density[self.active_cells] + self.dt * delta[self.active_cells]
        )
        np.clip(self.density, self.min_density, self.max_density, out=self.density)

        self._update_active_cells(delta)
        return self.density

    def __bool__(self) -> bool:
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
        self.tolerance_decayed = float(self.convergence_tolerance)
        self.convergence_decayed = float(self.convergence_after_steps)
