"""Create an animation of density changes over time using the forward model simulation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.animation import FuncAnimation

from bone_remodeling.src.forward_data.force_profile_generator import (
    ForceProfileGenerator,
)
from bone_remodeling.src.forward_model.density_visualizer import (
    plot_density_matrix,
    render_density_pyvista_frame,
)
from bone_remodeling.src.forward_model.main import DensitySimulation
from bone_remodeling.src.forward_model.parameters import SimulationParameters

logger = logging.getLogger(__name__)


def animate_density_matplotlib(
    simulation: DensitySimulation,
    output_directory: Path,
) -> FuncAnimation:
    """Create an animation of the density changes over time using Matplotlib.

    Args:
    ----
        simulation (DensitySimulation): The simulation object to run.
        output_directory (Path): The directory where the animation will be saved.

    """
    logger.info("Starting Matplotlib animation generation.")
    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    density_film: np.ndarray = np.zeros(
        (simulation.time_steps, simulation.n_rows, simulation.n_columns),
    )

    simulation.reset()
    for i in range(simulation.time_steps):
        simulation.step()
        density_film[i] = simulation.get_density()

    fig, ax = plt.subplots()

    def _update_to_next_frame(frame: int) -> list[plt.Axes]:
        """Update the plot for the current frame."""
        ax.clear()
        plot_density_matrix(
            matrix=density_film[frame, :, :],
            force_profile=simulation.force_profile,
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
        animation.save(output_directory, writer="ffmpeg")
        logger.info(f"Matplotlib animation saved to {output_directory}")
    except ImportError:
        logger.warning("Could not save animation.")
    return animation


def _update_pyvista_files(simulation: DensitySimulation, file_pattern: str) -> None:
    """Update the PyVista files to ensure they are in the correct format."""
    simulation.reset()
    for i in range(simulation.time_steps):
        simulation.step()
        simulation.save(
            to_save_data=simulation.density_function,
            output_path=file_pattern + f"_{i}",
        )
        Path.rename(
            file_pattern + f"_{i}" + "000000.vtu",
            file_pattern + f"_{i}" + ".vtu",
        )
        Path.rmdir(file_pattern + f"_{i}")
        Path.unlink(file_pattern + f"_{i}" + ".pvd")


def animate_density_pyvista(
    simulation: DensitySimulation,
    output_directory: Path,
    file_pattern: str,
) -> None:
    """Create an animation of the density changes over time using PyVista.

    Args:
    ----
        simulation (DensitySimulation): The simulation object to run.
        output_directory (Path): The directory where the animation will be saved.
        file_pattern (Path): The pattern for the output files.

    """
    logger.info("Starting PyVista animation generation.")
    _update_pyvista_files(simulation, file_pattern)

    pv.OFF_SCREEN = True

    file_list = sorted(
        Path.glob(Path(file_pattern).parent, Path(file_pattern).name + "_*.vtu"),
        key=lambda fn: int(Path(fn).stem.split("_")[-1]),
    )

    # Initialize the Plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(output_directory.as_posix())

    for step, filename in enumerate(file_list):
        grid = pv.read(filename)
        if not isinstance(grid, pv.UnstructuredGrid):
            logger.warning(f"File {filename} is not an UnstructuredGrid; skipping.")
            continue
        if not grid.array_names:
            logger.warning(f"No scalar arrays in {filename}; skipping.")
            continue

        render_density_pyvista_frame(
            plotter=plotter,
            grid=grid,
            scalar_field_name=grid.array_names[0],
            step_title=f"Step {step + 1}",
            clim=(simulation.min_density, simulation.max_density),
        )
        plotter.write_frame()
    plotter.close()
    logger.info(f"PyVista animation saved to {output_directory}")


def main() -> None:
    """Run the density animations to create to GIFS."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting density animation generation.")

    force_profile_generator = ForceProfileGenerator(
        profile_length=10,
        batch_seed=10,
    )
    force_profile = force_profile_generator.merger(
        num_samples=1,
        scaling=20,
    ).squeeze()

    parameters = SimulationParameters(
        force_profile=force_profile,
        initial_density_field=np.ones((10, 10)) * 0.8,
        save_data=True,
        time_steps=50,
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
        initial_density_field=np.ones((n_points, n_points)) * 0.8,
        save_data=True,
    )
    # To run change parameters to validation_parameters ----

    # Check the moment theory about the pillar.
    moment_force_profile = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0, 0, -5, 5],
                                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     ])

    moment_parameters = SimulationParameters(  # noqa: F841
        force_profile=moment_force_profile,
        initial_density_field=np.ones((10, 10)) * 0.8,
        save_data=True,
    )
    # To run change parameters to moment_parameters ----


    simulation = DensitySimulation(parameters)

    animate_density_matplotlib(
        simulation=simulation,
        output_directory=Path(
            "/home/gijs/Desktop/Thesis/data/animations/0_density_animation.gif",
        ),
    )
    animate_density_pyvista(
        simulation=simulation,
        output_directory=Path(
            "/home/gijs/Desktop/Thesis/data/animations/0_density_animation_pyvista.gif",
        ),
        file_pattern="/home/gijs/Desktop/Thesis/data/animations/density_animation",
    )


if __name__ == "__main__":
    main()
