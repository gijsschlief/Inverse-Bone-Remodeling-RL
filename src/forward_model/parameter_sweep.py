"""Parameter sweep tool for scientifically selecting convergence constants for the forward bone remodeling model.

This implements:
1. A high-quality reference run with very strict tolerances.
2. A sweep over multiple tolerance combinations.
3. L2 error vs. reference + runtime measurement.
4. CSV output + optional Pareto plot.
"""

import csv
import itertools
import logging
import time
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bone_remodeling.src.forward_data.force_profile_generator import (
    ForceProfileGenerator,  # type: ignore
)
from bone_remodeling.src.forward_model.main import DensitySimulation  # type: ignore
from bone_remodeling.src.forward_model.parameters import (
    SimulationParameters,  # type: ignore
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_simulation(params: SimulationParameters) -> np.ndarray:
    """Run a simulation once and return the final density field."""
    density_simulation = DensitySimulation(parameters=params)
    density_simulation.reset()
    density_simulation.update_force_profile(params.force_profile)
    density_simulation.run()
    return density_simulation.get_density()

def compute_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute relative L2 error."""
    return np.linalg.norm(a - b) / np.linalg.norm(b)

def run_reference_simulation(initial_simulation_parameters: SimulationParameters, force_profiles: np.ndarray) -> list[np.ndarray]:
    """Run a very tight simulation to get a high-quality reference solution."""
    logger.info("\n=== Running reference simulation (high-quality baseline) ===\n")
    initial_simulation_parameters = replace(
        initial_simulation_parameters,
        time_steps=1_000,
    )
    reference_run_parameters = replace(
        initial_simulation_parameters,
        convergence_tolerance_decay=1.01,
        convergence_after_steps=100,
        convergence_steps_decay=0.99,
    )

    t0 = time.time()
    reference_densities = [np.zeros_like(initial_simulation_parameters.initial_density_field) for _ in range(len(force_profiles))]
    for i in range(len(force_profiles)):
        logger.info(f"  → Running reference simulation {i + 1}/{len(force_profiles)}")
        reference_run = replace(
            reference_run_parameters,
            force_profile=force_profiles[i],
        )
        reference_densities[i] = run_simulation(reference_run)
    t1 = time.time()
    logger.info(f"Reference simulation runtime: {t1 - t0:.2f}s")
    return reference_densities

def parameter_grid() -> Generator[Any, Any, Any]:
    """Yield dictionaries of possible parameter combinations to test. Modify here to change sweep ranges."""
    # SINGLE LARGE SWEEP
    decay_tol_options = [1, 1.02, 1.04, 1.06, 1.08]
    ct0_options = [1, 5, 10]
    ct_decay_options = [1, 0.98, 0.96, 0.94]
    time_steps_options = [50, 100, 150, 200, 250, 300]

    for tol_decay, ct0, ct_decay, tsteps in itertools.product(
        decay_tol_options, ct0_options, ct_decay_options, time_steps_options,
    ):
        yield {
            "convergence_tolerance_decay": tol_decay,
            "convergence_after_steps": ct0,
            "convergence_steps_decay": ct_decay,
            "time_steps": tsteps,
        }

def run_parameter_sweep(base_params: SimulationParameters, force_profiles: np.ndarray, density_references: list[np.ndarray], output_csv: Path) -> None:
    """Run the full parameter sweep and save results to CSV."""
    logger.info("\n=== Starting parameter sweep ===\n")
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tol_decay", "C0", "C_decay", "time_steps",
            "runtime_s", "l2_error",
        ])

        for combo in parameter_grid():
            logger.info(f"Testing params: {combo}")

            params = replace(base_params, **combo)
            density = np.zeros((len(force_profiles),) + base_params.initial_density_field.shape)
            l2_error = np.zeros(len(force_profiles))

            t0 = time.time()
            try:
                for i in range(len(force_profiles)):
                    params_i = replace(params, force_profile=force_profiles[i])
                    density[i] = run_simulation(params_i)

                runtime = time.time() - t0
                for i in range(len(force_profiles)):
                    l2_error[i] = compute_l2_error(density[i], density_references[i])
                average_l2_error = np.mean(l2_error)
            except Exception as e:
                logger.error(f"A simulation failed: {e!s}")
                runtime = np.nan
                average_l2_error = np.nan

            writer.writerow([
                combo["convergence_tolerance_decay"],
                combo["convergence_after_steps"],
                combo["convergence_steps_decay"],
                combo["time_steps"],
                runtime,
                average_l2_error,
            ])
            logger.info(f"  → runtime={runtime:.2f}s, L2 error={average_l2_error:.3e}")
    logger.info(f"\nSweep complete. Results written to: {output_csv}")

def datasweep() -> None:
    """Define a representative loadcase and run a parameter sweep. Saves results to CSV."""
    initial_density = np.full((10, 10), 0.8)
    force_profile_generator = ForceProfileGenerator(
        profile_length=10,
        batch_seed=12345,
    )
    force_profiles = force_profile_generator.merger(num_samples=20, scaling=10.0)

    simulation_base_parameters = SimulationParameters(force_profile=force_profiles[0], initial_density_field=initial_density)
    output_csv = Path("parameter_sweep_results_final.csv")
    density_references = run_reference_simulation(simulation_base_parameters, force_profiles)
    run_parameter_sweep(simulation_base_parameters, force_profiles, density_references, output_csv)

def pareto_plot() -> None:
    """Generate a Pareto plot from the CSV results."""
    df1 = pd.read_csv("/home/gijs/Desktop/Thesis/data/sweeps/parameter_sweep_results_final2.csv")
    df2 = pd.read_csv("/home/gijs/Desktop/Thesis/data/sweeps/parameter_sweep_results_final3.csv")
    df = pd.concat([df1, df2], ignore_index=True)

    # Remove failed runs
    df = df.dropna(subset=["runtime_s", "l2_error"])
    df = df[df["l2_error"] > 0]

    pareto = []
    for _i, row_i in df.iterrows():
        dominated = False
        for _j, row_j in df.iterrows():
            if (
                (row_j["runtime_s"] <= row_i["runtime_s"]) and
                (row_j["l2_error"] <= row_i["l2_error"]) and
                ((row_j["runtime_s"] < row_i["runtime_s"]) or
                (row_j["l2_error"] < row_i["l2_error"]))
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(row_i)

    pareto = sorted(pareto, key=lambda x: x["runtime_s"])
    pareto_df = pd.DataFrame(pareto)
    logger.info(f"Pareto-optimal points on sweep:\n{pareto_df}")
    plt.scatter(df["runtime_s"], df["l2_error"], s=10)
    plt.xlabel("Runtime (s)")
    plt.ylabel("Relative L2 error")
    plt.yscale("log")
    plt.title("Parameter sweep — Runtime vs Error")
    plt.scatter(pareto_df["runtime_s"], pareto_df["l2_error"], color="red", s=30, label="Pareto front")
    # orinal model
    plt.scatter(26.849, 0.065, color="black", marker="x", s=100, label="Original model t = 100")

    # Chosen optimum
    plt.scatter(58.871966, 0.002908, color="red", marker="x", s=100, label="Chosen optimum t = 250")
    plt.grid(visible=True)
    plt.legend()
    plt.show()

def performance_comparison(num_samples: int = 100) -> None:
    """Compare performance of selected parameter sets. Logs runtime and L2 error vs. reference."""
    profile_length: int = 10

    initial_density = np.full((profile_length, profile_length), 0.8)
    force_profile_generator = ForceProfileGenerator(
        profile_length=profile_length,
        batch_seed=12345,
    )
    force_profiles = force_profile_generator.merger(num_samples=num_samples, scaling=20.0)

    simulation_base_parameters = SimulationParameters(force_profile=force_profiles[0], initial_density_field=initial_density)

    # Selected parameter sets from sweep
    parameter_reference = replace(
            simulation_base_parameters,
            convergence_tolerance_decay=1.01,
            convergence_after_steps=100,
            convergence_steps_decay=0.99,
            time_steps=1000,
        )
    parameter_sets = [
        {
            "convergence_tolerance_decay": 1.06,
            "convergence_after_steps": 20,
            "convergence_steps_decay": 0.96,
            "time_steps": 200,
        },
        {
            "convergence_tolerance_decay": 1,
            "convergence_after_steps": 1,
            "convergence_steps_decay": 1,
            "time_steps": 100,
        },
    ]

    _reference_density = np.zeros((num_samples, profile_length, profile_length))
    l2_error = np.zeros(num_samples)
    t0 = time.time()
    for i in range(num_samples):
        params_i = replace(parameter_reference, force_profile=force_profiles[i])
        _reference_density[i]  = run_simulation(params_i)
    reference_runtime = time.time() - t0
    logger.info(f"Reference simulation runtime for {num_samples} runs: {reference_runtime:.2f}s")
    logger.info(f"  → Example profiles: {_reference_density[0]}")

    for combo in parameter_sets:
        params = replace(simulation_base_parameters, **combo)
        logger.info(f"Running performance test with params: {combo}")
        t0 = time.time()
        density = np.zeros((num_samples, profile_length, profile_length))
        for i in range(num_samples):
            params_i = replace(params, force_profile=force_profiles[i])
            density[i]  = run_simulation(params_i)
            l2_error[i] = compute_l2_error(density[i], _reference_density[i])
        runtime = time.time() - t0
        combined_l2_error = np.mean(l2_error)

        logger.info(f"  → Total runtime for {num_samples} runs: {runtime:.2f}s")
        logger.info(f"  → L2 Error: {combined_l2_error:.4f}")
        logger.info(f"  → Example profile: {density[0]}")

if __name__ == "__main__":
    #datasweep()
    pareto_plot()
    #performance_comparison(num_samples=5)
