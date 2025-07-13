"""Create a film of density changes over time using a forward model simulation."""

import matplotlib.pyplot as plt
import numpy as np
from bone_remodeling.forward_model.density_simulation import DensitySimulation
from bone_remodeling.forward_model.force_profile_generator import ForceProfileGenerator
from bone_remodeling.surrogate_model.visualizer import plot_density_matrix
from matplotlib.animation import FuncAnimation

SAVE_PATH = "/home/gijs/Desktop/Thesis/data/fenics"
TIME_STEPS = 100

force_profile_generator = ForceProfileGenerator(
    profile_length=10,
    batch_seed=np.random.randint(0, 10000),
)
force_profile = force_profile_generator.impulse(
    num_samples=1,
    force_count_max=30,
    force_max=30.0,
)[0]

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
plt.show()
