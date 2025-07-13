"""Create a film of density changes over time using a forward model simulation."""

import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from bone_remodeling.forward_model.density_simulation import DensitySimulation
from bone_remodeling.forward_model.force_profile_generator import ForceProfileGenerator
from bone_remodeling.surrogate_model.visualizer import plot_density_matrix
from matplotlib.animation import FuncAnimation

SAVE_GIF = "/home/gijs/Desktop/Thesis/data/animations/9_density_animation.gif"
SAVE_GIF_PYVISTA = "/home/gijs/Desktop/Thesis/data/animations/9_density_animation_pyvista.gif"
FILE_PATTERN = "/home/gijs/Desktop/Thesis/data/animations/density_animation"
TIME_STEPS = 100

force_profile_generator = ForceProfileGenerator(
    profile_length=10,
    batch_seed=np.random.randint(0, 10000),
)
force_profile = force_profile_generator.merger(num_samples=1, force_max=20).squeeze()

density_start = np.ones((10, 10)) * 0.8
parameters: dict = {
    "save": True,
    "output_dir": "/home/gijs/Desktop/Thesis/data/fenics",
    "plot": True,
    "convergence_after_steps": 10,
}

simulation = DensitySimulation(
    force_profile=force_profile,
    initial_density_field=density_start,
    time_steps=TIME_STEPS,
    dt=1.0,
    parameters=parameters,
)

density_film: np.ndarray = np.zeros((TIME_STEPS, simulation.n_rows, simulation.n_columns))

for i in range(TIME_STEPS):
    simulation.step()
    density_film[i] = simulation.get_density()
    simulation.save(to_save_data=simulation.density_function, output_path=FILE_PATTERN + f"_{i}")
    os.rename(
        FILE_PATTERN + f"_{i}" + "000000.vtu",
        FILE_PATTERN + f"_{i}" + ".vtu",
    )
    os.rmdir(
        FILE_PATTERN + f"_{i}")
    os.remove(
        FILE_PATTERN + f"_{i}" + ".pvd"
    )

fig, ax = plt.subplots()

def update(frame: int) -> list[plt.Axes]:
    """Update the plot for the current frame."""
    ax.clear()
    plot_density_matrix(
        matrix=density_film[frame, :, :],
        force_profile=force_profile,
        title=f"Step {frame+1}",
        axis=ax,
    )
    # Return a list of artists for FuncAnimation
    return [ax]

animation = FuncAnimation(fig, update, frames=TIME_STEPS, interval=100, repeat=True)
plt.tight_layout()
try:
    animation.save(SAVE_GIF, writer="ffmpeg")
except ImportError:
    logging.info("Could not save animation.")


pv.OFF_SCREEN = True

# Collect your files, sorted by index
file_list = sorted(
    glob.glob(FILE_PATTERN + "_*.vtu"),
    key=lambda fn: int(os.path.splitext(fn)[0].split("_")[-1])
)

# Initialize the Plotter in off‐screen mode and open the GIF writer
plotter = pv.Plotter(off_screen=True)
plotter.open_gif(SAVE_GIF_PYVISTA)

# Loop through each timestep file and write a frame
for step, filename in enumerate(file_list):
    grid = pv.read(filename)
    if not grid.array_names:
        logging.warning(f"No scalar arrays in {filename}; skipping.")
        continue

    # Clear the previous frame
    plotter.clear()

    # Add the mesh; use the first array in the file as the scalar field
    cmap = "viridis"
    clim = [grid.get_data_range()[0], grid.get_data_range()[1]]
    plotter.add_mesh(
        grid,
        scalars=grid.array_names[0],
        show_edges=True,
        clim=clim,
        cmap=cmap,
        show_scalar_bar=True,
    )

    plotter.view_xy()
    plotter.add_text(f"Step {step+1}", position="upper_left", font_size=14)

    # Write the frame
    plotter.write_frame()

# Close and finalize the GIF
plotter.close()
logging.info(f"PyVista animation saved to {SAVE_GIF_PYVISTA}")