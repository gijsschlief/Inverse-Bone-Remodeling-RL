"""Visualizer for bone remodeling forward model. Containing functions to plot density matrices and force profiles."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv  # type: ignore
from bone_remodeling.forward_model.density_simulation import DensitySimulation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def plot_density_matrix(
    matrix: np.ndarray,
    force_profile: np.ndarray | None,
    title: str,
    axis: plt.Axes,
    color_scale_min: float = 0.01,
    color_scale_max: float = 1.73,
) -> None:
    """Plot a density matrix with annotations.

    Args:
    ----
        matrix (np.ndarray): Density matrix to plot.
        force_profile (np.ndarray | None): Force profile corresponding to the matrix.
        title (str): Title of the plot.
        axis: Matplotlib axis to plot on.
        color_scale_min (float): Minimum value for color scaling.
        color_scale_max (float): Maximum value for color scaling.

    """
    axis.imshow(
        matrix,
        cmap="viridis",
        interpolation="nearest",
        vmin=color_scale_min,
        vmax=color_scale_max,
    )
    axis.set_title(title)
    axis.set_xlabel("Columns")
    axis.set_ylabel("Rows")

    # Annotate matrix values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    # Plot force arrows if force_profile is provided
    if force_profile is not None:
        height, width = matrix.shape
        top_forces = force_profile[0]
        right_forces = force_profile[1]
        left_forces = force_profile[2]

        # Compute max magnitude for scaling (avoid division by zero)
        max_force = np.max(np.abs(force_profile))

        arrow_scale = 0.5  # Maximum arrow length

        # Top forces: draw downward arrows above row 0
        for j in range(width):
            if abs(top_forces[j]) > 1e-3:
                scaled_length = arrow_scale * abs(top_forces[j]) / max_force
                force_color = _get_force_color(top_forces[j])
                if top_forces[j] < 0:
                    axis.arrow(
                        j,
                        -0.5,
                        0,
                        scaled_length * np.sign(top_forces[j]),  # downward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )
                elif top_forces[j] > 0:
                    axis.arrow(
                        j,
                        -0.5 - scaled_length - 0.15,
                        0,
                        scaled_length * np.sign(top_forces[j]),  # upward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )

        # Left forces: draw rightward arrows left of column 0
        for i in range(height):
            if abs(left_forces[i]) > 1e-3:
                scaled_length = arrow_scale * abs(left_forces[i]) / max_force
                force_color = _get_force_color(left_forces[i])
                if left_forces[i] < 0:
                    axis.arrow(
                        -0.5,
                        height - 1 - i,
                        scaled_length * np.sign(left_forces[i]),
                        0,  # rightward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )
                elif left_forces[i] > 0:
                    axis.arrow(
                        -0.5 - scaled_length - 0.15,
                        height - 1 - i,
                        scaled_length * np.sign(left_forces[i]),
                        0,  # leftward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )

        # Right forces: draw leftward arrows right of last column
        for i in range(height):
            if abs(right_forces[i]) > 1e-3:
                scaled_length = arrow_scale * abs(right_forces[i]) / max_force
                force_color = _get_force_color(right_forces[i])
                if right_forces[i] < 0:
                    axis.arrow(
                        width - 0.5 + scaled_length + 0.15,
                        height - 1 - i,
                        scaled_length * np.sign(right_forces[i]),
                        0,  # rightward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )
                elif right_forces[i] > 0:
                    axis.arrow(
                        width - 0.5,
                        height - 1 - i,
                        scaled_length * np.sign(right_forces[i]),
                        0,  # leftward
                        head_width=0.2,
                        head_length=0.15,
                        fc=force_color,
                        ec=force_color,
                    )

def _get_force_color(
    force_val: float, min_force: float = 0.1, max_force: float = 10.0
) -> str:
    """Map a force magnitude to a grayscale hex color between light gray and black."""
    abs_force = abs(force_val)

    if abs_force < min_force:
        return "#000000"  # Light gray for very small forces
    if abs_force > max_force:
        return "#FF0000"  # Black for very large forces

    # Normalize force value between 0 and 1
    norm = min(max((abs_force - min_force) / (max_force - min_force), 0.0), 1.0)

    # Interpolate between black (0,0,0) and red (255,0,0)
    red = int(255 * norm)
    green = 0
    blue = 0

    hex_color = f"#{red:02x}{green:02x}{blue:02x}"
    return hex_color

def plot_density_pyvista(simulation: DensitySimulation, directory: Path = Path("/home/gijs/Desktop/Thesis/data/fenics/density_simulation.pvd")) -> None:
    """Plot the density simulation using PyVista.

    Args:
    ----
        simulation (DensitySimulation): The density simulation object.
        directory (Path): Directory where the simulation results are stored.

    """
    try:
        reader = pv.get_reader(directory)
        reader.set_active_time_point(0)
        grid = reader.read()[0]
        scalar_field_name = grid.array_names[0]
        clim = (simulation.min_density, simulation.max_density)

        plotter = pv.Plotter()
        render_density_pyvista_frame(plotter, grid, scalar_field_name, "Final Step", clim)
        plotter.show()
    except Exception as e:
        logging.exception(f"Failed to plot result with PyVista: {e}")

def render_density_pyvista_frame(
    plotter: pv.Plotter,
    grid: pv.UnstructuredGrid,
    scalar_field_name: str,
    step_title: str,
    clim: tuple[float, float],
) -> None:
    """Render a single frame of density in a PyVista Plotter."""
    plotter.clear()
    plotter.add_mesh(
        grid,
        scalars=scalar_field_name,
        show_edges=True,
        show_scalar_bar=True,
        clim=clim,
        cmap="viridis",
    )
    plotter.camera_position = 'xy'
    plotter.add_text(step_title, position="upper_left", font_size=14)

def main() -> None:
    """Run the density visualizer to create plots."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting density visualization.")

    force_profile = np.random.rand(3, 10) * 10 - 5

    simulation = DensitySimulation(
        force_profile=force_profile,
        initial_density_field=np.ones((10, 10)) * 0.8,
        time_steps=100,
        parameters={"save": False, "plot": True},
    )

    plot_density_pyvista(simulation=simulation)

if __name__ == "__main__":
    main()
