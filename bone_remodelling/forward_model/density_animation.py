"""Create an animation of density changes over time using the forward model simulation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from pyvista import Plotter, UnstructuredGrid

from bone_remodelling.forward_data.force_profile_generator import (
    ForceProfileGenerator,
)
from bone_remodelling.forward_model.density_visualizer import (
    plot_density_matrix,
    render_density_pyvista_frame,
)
from bone_remodelling.forward_model.main import DensitySimulation
from bone_remodelling.forward_model.parameters import SimulationParameters

logger = logging.getLogger(__name__)


def animate_density_matplotlib(
    simulation: DensitySimulation,
    output_directory: Path = Path(__file__).parent.parent.parent / Path("data", "animations", "density_animation.gif"),
) -> FuncAnimation:
    """Create an animation of the density changes over time using Matplotlib.

    Args:
    ----
        simulation (DensitySimulation): The simulation object to run.
        output_directory (Path): The directory where the animation will be saved.

    """
    logger.info("Starting Matplotlib animation generation.")
    if not output_directory.exists():
        output_directory.parent.mkdir(parents=True, exist_ok=True)

    density_film: np.ndarray = np.zeros(
        (simulation.time_steps, simulation.n_rows, simulation.n_columns),
    )

    simulation.reset()
    for i in range(simulation.time_steps):
        simulation.step()
        density_film[i] = simulation.get_density()

    fig, ax = plt.subplots()

    def _update_to_next_frame(frame: int) -> list[Axes]:
        """Update the plot for the current frame."""
        ax.clear()
        plot_density_matrix(
            matrix=density_film[frame, :, :],
            force_data=(simulation.force_profile, simulation.force_mask),
            title=f"Step {frame + 1}",
            axis=ax,
        )
        return [ax]

    animation = FuncAnimation(
        fig,
        _update_to_next_frame,
        frames=simulation.time_steps,
        interval=100,
        repeat=True,
    )
    plt.tight_layout()
    try:
        animation.save(str(output_directory), writer="ffmpeg")
        logger.info(f"Matplotlib animation saved to {output_directory}")
    except ImportError:
        logger.warning("Could not save animation.")
    return animation

def animate_density_pyvista(
    simulation: DensitySimulation,
    output_directory: Path = Path(__file__).parent.parent.parent / Path("data", "animations", "density_animation_pyvista.gif"),
) -> None:
    """Create a PyVista animation using in-memory snapshots of FEniCS data.

    Args:
    ----
        simulation (DensitySimulation): The simulation object to run.
        output_directory (Path): The directory where the animation will be saved.

    """
    logger.info("Starting PyVista animation generation.")

    fenics_mesh = simulation.density_function.function_space().mesh()
    coords = fenics_mesh.coordinates()
    cells = fenics_mesh.cells()
    points_3d = np.zeros((coords.shape[0], 3))
    points_3d[:, :2] = coords
    cells_pv = np.column_stack([np.full(cells.shape[0], 3), cells])
    base_grid = UnstructuredGrid(cells_pv, np.full(cells.shape[0], 5, dtype=np.uint8), points_3d)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    pyvista_plotter = Plotter(off_screen=True)
    pyvista_plotter.open_gif(str(output_directory))
    step = 0

    try:
        simulation.reset()

        for step in range(simulation.time_steps):
            simulation.step()
            base_grid.cell_data["Density"] = simulation.get_density_function().vector().get_local()
            pyvista_plotter.clear()
            render_density_pyvista_frame(
                pyvista_plotter=pyvista_plotter,
                grid=base_grid,
                scalar_field_name="Density",
                step_title=f"Step {step + 1}",
                clim=(simulation.min_density, simulation.max_density),
            )
            pyvista_plotter.write_frame()
    except RuntimeError as e:
        logger.error(f"Animation interrupted at step {step}: {e}")
        raise # Re-raise after closing plotter to notify the caller
    finally:
        pyvista_plotter.close()

    logger.info(f"PyVista animation saved to {output_directory}")


def main() -> None:
    """Run the density animations to create to GIFS."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting density animation generation.")

    force_profile_generator = ForceProfileGenerator(
        profile_top_and_sides=(10, 10),
        force_bounds=(0.1, 10.0),
        energy_bounds=(1e3, 5e4),
        batch_seed=1,
    )
    force_profile = force_profile_generator.merger(num_samples=1).squeeze()
    force_mask = force_profile_generator.generate_force_mask()

    parameters = SimulationParameters(
        force_profile=force_profile,
        force_mask=force_mask,
        initial_density_field=np.ones((20, 20)) * 0.87,
    )

    # Alternative profile for Weinans model validation ----
    validation_force_maginitude = -25 # -5 for previous parameters
    n_points = 100
    scale_factors = np.linspace(1.8, 0, n_points+1)[:-1]

    validation_force_profile = np.array([
        scale_factors * validation_force_maginitude,
        np.zeros(n_points),
        np.zeros(n_points),
    ])

    validation_parameters = SimulationParameters(  # noqa: F841
        force_profile=validation_force_profile,
        force_mask=force_mask,
        initial_density_field=np.ones((10, 10)) * 0.87,
    )
    # To run change parameters to validation_parameters ----

    # Check the moment theory about the pillar.
    moment_force_profile = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, -5, 5],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     ])

    moment_parameters = SimulationParameters(  # noqa: F841
        force_profile=moment_force_profile,
        force_mask=force_mask,
        initial_density_field=np.ones((10, 10)) * 0.87,
    )
    # To run change parameters to moment_parameters ----

    simulation = DensitySimulation(parameters=parameters)

    animate_density_matplotlib(simulation=simulation)
    #animate_density_pyvista(simulation=simulation)


if __name__ == "__main__":
    main()
