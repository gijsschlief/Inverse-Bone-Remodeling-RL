"""The density updater is responsible for updating the density each timestep when running the forward simulation."""

import numpy as np

from bone_remodelling.forward_model.parameters import SimulationParameters


class DensityUpdater:
    """The DensityUpdater class manages the density update process during the simulation.

    It applies the density update rule based on the specific energy density (SED)
    and the parameters defined in SimulationParameters. The updater tracks the
    convergence of each cell and updates the density accordingly.
    """

    def __init__(
        self,
        initial_density: np.ndarray,
        simulation_parameters: SimulationParameters,
    ) -> None:
        """Initialize the DensityUpdater with simulation parameters.

        Args:
        ----
            initial_density (np.ndarray): The initial density values for the simulation.
            simulation_parameters (SimulationParameters): The parameters for the simulation.

        """
        self._initial_density = initial_density.copy()
        self.density = self._initial_density.copy()
        self.active_cells = np.ones_like(self.density, dtype=bool)
        self.convergence_counter = np.zeros_like(self.density, dtype=int)
        self.maximum_delta = simulation_parameters.maximum_delta

        self.dt = simulation_parameters.dt
        self.remodeling_rate_coefficient = (
            simulation_parameters.remodeling_rate_coefficient
        )
        self.stimulus_threshold = simulation_parameters.stimulus_threshold
        self.min_density = simulation_parameters.min_density
        self.max_density = simulation_parameters.max_density
        self.convergence_tolerance = simulation_parameters.convergence_tolerance
        self.convergence_tolerance_decay = (
            simulation_parameters.convergence_tolerance_decay
        )
        self.convergence_after_steps = simulation_parameters.convergence_after_steps
        self.convergence_steps_decay = simulation_parameters.convergence_steps_decay

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

    def _update_active_cells(self, delta_rho: np.ndarray) -> None:
        """Check which cells are still active based on changes in density."""
        cells_converged = delta_rho < self.tolerance_decayed
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
        old_density = self.density.copy()
        stimulus = np.zeros_like(self.density)
        stimulus[self.active_cells] = (
            strain_energy_density[self.active_cells] / self.density[self.active_cells]
        )

        delta = self.remodeling_rate_coefficient * (stimulus - self.stimulus_threshold)
        delta = np.clip(delta, -self.maximum_delta, self.maximum_delta) # Limit density changes per timestep to avoid instability

        self.density[self.active_cells] = (
            self.density[self.active_cells] + self.dt * delta[self.active_cells]
        )
        np.clip(self.density, self.min_density, self.max_density, out=self.density)

        delta_rho = np.abs(self.density - old_density)
        self._update_active_cells(delta_rho)

        self._decay()
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
