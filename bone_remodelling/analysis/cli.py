"""The parser for the analysis tools from the model."""

import argparse
from pathlib import Path

from bone_remodelling.parameters import ConfigurationParameters


def cli(config: ConfigurationParameters, argv: list[str]) -> None:
    """CLI entry point for training data generation."""
    parser = argparse.ArgumentParser(
        description="Generate training data for bone remodeling simulations."
    )
    parser.add_argument(
        "--type",
        type=str,
        default="max_force_sweep",
        choices=[
            "forward_parameter_sweep",
            "max_force_sweep",
            "visualise_forces",
            "visualise_database",
            "diversity_metrics",
        ],
        help="Analysis tools used for figure generation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples to generate for the forward parameter sweep.",
    )
    args = parser.parse_args(argv)
    if args.type == "forward_parameter_sweep":
        from bone_remodelling.analysis.forward_parameter_sweep import (
            run_forward_sweep as forward_parameter_sweep,
        )  # noqa: I001, PLC0415

        sweep_directory = config.output_dir / Path("forward_parameter_sweep")
        sweep_directory.mkdir(exist_ok=True)
        sweep_file = sweep_directory / Path("sweep_results.csv")
        forward_parameter_sweep(sweep_file, num_samples=args.num_samples)
    elif args.type == "max_force_sweep":
        from bone_remodelling.analysis.max_force_sweep import (
            run_sweep_and_analyse as max_force_sweep,
        )  # noqa: I001, PLC0415

        max_force_sweep(config.output_dir / Path("sweeps", "max_force_sweep"))
    elif args.type == "visualise_forces":
        from bone_remodelling.analysis.visualise_force_profiles import (
            analyse_force_generator as visualise_forces,
        )  # noqa: I001, PLC0415

        visualise_forces()
    elif args.type == "visualise_database":
        from bone_remodelling.analysis.visualise_database import (
            analyse_raw_data as visualise_database,
        )  # noqa: I001, PLC0415

        visualise_database(config.output_dir / Path("raw"))
    elif args.type == "diversity_metrics":
        from bone_remodelling.analysis.diversity_metrics import (
            run_diversity_metrics as diversity_metrics,
        )  # noqa: I001, PLC0415

        diversity_metrics(config.output_dir / Path("raw"))
    else:
        raise ValueError(f"Unknown analysis type: {args.type}")
