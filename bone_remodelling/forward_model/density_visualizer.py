"""Visualizer for bone remodeling forward model. Containing functions to plot density matrices and force profiles."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pyvista import Plotter, UnstructuredGrid

from bone_remodelling.forward_model.main import DensitySimulation
from bone_remodelling.forward_model.parameters import SimulationParameters

logger = logging.getLogger(__name__)


def plot_density_matrix(
    matrix: np.ndarray,
    force_profile: np.ndarray | None,
    title: str,
    axis: Axes,
    color_scale: tuple[float, float] = (0.01, 1.73),
) -> None:
    """Plot a density matrix with annotations.

    Args:
    ----
        matrix (np.ndarray): Density matrix to plot.
        force_profile (np.ndarray | None): Force profile corresponding to the matrix.
        title (str): Title of the plot.
        axis: Matplotlib axis to plot on.
        color_scale (tuple[float, float]): Contains the minimum and maximum color scale values

    """
    min_color_scale, max_color_scale = color_scale

    axis.imshow(
        matrix,
        cmap="viridis",
        interpolation="nearest",
        vmin=min_color_scale,
        vmax=max_color_scale,
    )
    axis.set_title(title)
    axis.set_xlabel("Columns")
    axis.set_ylabel("Rows")

    # Annotate matrix values
    # Only annotate if the matrix is not too large
    max_annotate_size = 10
    if matrix.shape[0] <= max_annotate_size and matrix.shape[1] <= max_annotate_size:
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

    if force_profile is not None:
        height, width = matrix.shape
        _plot_force_arrows(axis, force_profile, width, height)
        _plot_force_band(axis, force_profile, width, height)

def _plot_force_band(axis: Axes, force_profile: np.ndarray, width: int, height: int) -> None:
    """Plot a colored band above the density matrix to represent top forces."""
    top_forces = force_profile[1, :width]
    left_forces = force_profile[0, :height]
    right_forces = force_profile[2, :height]
    v_min = -np.max(np.abs(force_profile))
    v_max =  np.max(np.abs(force_profile))

    band_thickness = 0.1
    top_offset = 0.61
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()

    axis.imshow(
        top_forces[np.newaxis, :],
        extent=(-0.5, width - 0.5, - top_offset, band_thickness -top_offset),
        cmap="seismic",
        aspect="auto",
        vmin=v_min,
        vmax=v_max,
        alpha=0.8,
    )

    left_offset = 0.605
    axis.imshow(
        left_forces[:, np.newaxis],
        extent=(band_thickness -left_offset, -left_offset, -0.5, height - 0.5),
        cmap="seismic",
        aspect="auto",
        vmin=v_min,
        vmax=v_max,
        alpha=0.8,
    )

    right_offset = 0.4
    axis.imshow(
        right_forces[:, np.newaxis],
        extent=(width - band_thickness - right_offset, width - right_offset, -0.5, height - 0.5),
        cmap="seismic",
        aspect="auto",
        vmin=v_min,
        vmax=v_max,
        alpha=0.8,
    )

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)

def _get_quiver_data(side: str, forces: np.ndarray, height: int, width: int, threshold: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    mask = np.abs(forces) > threshold
    if not np.any(mask):
        return None
    indices = np.where(mask)[0]
    f_vals = forces[mask]
    offset = 1.5

    if side == 'top':
        x, y = indices, np.full_like(indices, -offset)
        u, v = np.zeros_like(f_vals), -f_vals
    elif side == 'left':
        x, y = np.full_like(indices, -offset), height - 1 - indices
        u, v = -f_vals, np.zeros_like(f_vals)
    else:
        x, y = np.full_like(indices, width -1 + offset), height - 1 - indices
        u, v = f_vals, np.zeros_like(f_vals)
    return x, y, u, v, f_vals

def _plot_force_arrows(axis: Axes, force_profile: np.ndarray, width: int, height: int) -> None:
    """Plot all force arrows (top, left, right) using a single quiver call per side.

    Args:
    ----
        axis: Matplotlib axis to plot on.
        force_profile (np.ndarray): 2D array with shape (3, N) containing forces for top, left, right sides.
        width (int): Width of the density matrix.
        height (int): Height of the density matrix.

    """
    left_forces  = force_profile[0, :height]
    top_forces   = force_profile[1, :width]
    right_forces = force_profile[2, :height]

    for side, forces in zip(['left', 'top', 'right'], [left_forces, top_forces, right_forces]):
        data = _get_quiver_data(side, forces, height, width)
        if data:
            x, y, u, v, magnitudes = data
            axis.quiver(
                x, y, u, v, magnitudes,
                cmap='seismic',
                angles='xy',
                scale_units='xy',
                scale=np.max(np.abs(force_profile)),
                pivot='middle',
                width=0.025,
                minshaft=2,
                minlength=1e-12,
            )

def plot_density_pyvista(

    simulation: DensitySimulation,
) -> None:
    """Plot the density simulation using PyVista.

    Args:
    ----
        simulation (DensitySimulation): The density simulation instance.

    """
    fenics_mesh = simulation.density_function.function_space().mesh()
    coords = fenics_mesh.coordinates()
    cells = fenics_mesh.cells()
    points_3d = np.zeros((coords.shape[0], 3))
    points_3d[:, :2] = coords
    cells_pv = np.column_stack([np.full(cells.shape[0], 3), cells])
    base_grid = UnstructuredGrid(cells_pv, np.full(cells.shape[0], 5, dtype=np.uint8), points_3d)
    base_grid.cell_data["Density"] = simulation.get_density_function().vector().get_local()

    try:
        pyvista_plotter = Plotter()
        render_density_pyvista_frame(
            pyvista_plotter,
            base_grid,
            "Density",
            "Initial density field",
            (simulation.min_density, simulation.max_density),
        )
        pyvista_plotter.show()
    except Exception as e:
        logger.exception(f"Failed to plot result with PyVista: {e}")


def render_density_pyvista_frame(
    pyvista_plotter: Plotter,
    grid: UnstructuredGrid,
    scalar_field_name: str,
    step_title: str,
    clim: tuple[float, float],
) -> None:
    """Render a single frame of density in a PyVista Plotter."""
    pyvista_plotter.clear()
    pyvista_plotter.add_mesh(
        grid,
        scalars=scalar_field_name,
        preference="cell",
        show_edges=True,
        show_scalar_bar=True,
        clim=clim,
        cmap="viridis",
    )
    pyvista_plotter.camera_position = "xy"
    pyvista_plotter.add_text(step_title, position="upper_left", font_size=14)


def example_usage() -> None:
    """Run the density visualizer to create plots."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting density visualization.")

    force_profile = np.random.rand(3, 50) * 10 - 5

    simulation_parameters = SimulationParameters(
        force_profile=force_profile,
        initial_density_field=np.ones((50, 10)) * 0.8,
        time_steps=100,
    )

    plot_density_matrix(
        matrix=simulation_parameters.initial_density_field,
        force_profile=force_profile,
        title="Initial Density Field",
        axis=plt.gca(),
    )
    plt.show()

    simulation = DensitySimulation(parameters=simulation_parameters)
    simulation.step()
    plot_density_pyvista(simulation=simulation)

if __name__ == "__main__":
    example_usage()
