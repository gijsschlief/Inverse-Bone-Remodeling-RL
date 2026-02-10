"""The parser for the analysis tools from the model."""

import argparse

from bone_remodelling.parameters import ConfigurationParameters


def cli(config: ConfigurationParameters, argv: list[str]) -> None:
    """CLI entry point for training data generation."""
    parser = argparse.ArgumentParser(description="Generate training data for bone remodeling simulations.")
    parser.add_argument(
        "--type",
        type=str,
        default="max_force_sweep",
        choices=["forward_parameter_sweep", "max_force_sweep", "visualise_forces", "diversity_metrics"]
        help="Analysis tools used for figure generation.",
    )
    parser.add_argument(
        "--force_type",
        type=str,
        choices=["merger", "triangular"],
        default="triangular",
        help="Type of force profile to generate ('merger' or 'triangular').",
    )
    args = parser.parse_args(argv)
    if args.type == "forward_parameter_sweep":
        from bone_remodelling.analysis.forward_parameter_sweep import run as forward_parameter_sweep  # noqa: I001, PLC0415
        forward_parameter_sweep(config)
    elif args.type == "max_force_sweep":
        from bone_remodelling.analysis.max_force_sweep import run as max_force_sweep  # noqa: I001, PLC0415
        max_force_sweep(config, force_type=args.force_type)
    elif args.type == "visualise_forces":
        from bone_remodelling.analysis.visualise_forces import run as visualise_forces  # noqa: I001, PLC0415
        visualise_forces(config)
    elif args.type == "diversity_metrics":
        from bone_remodelling.analysis.diversity_metrics import run as diversity_metrics  # noqa: I001, PLC0415
        diversity_metrics(config)
