"""Unified CLI for Thesis Modules."""

import argparse
import logging
from pathlib import Path

from bone_remodelling.parameters import ConfigurationParameters


def setup_logging() -> logging.Logger:
    """Set up logging for the CLI."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def build_configuration_parameters() -> ConfigurationParameters:
    """Set up the data directory for simulations.

    Returns
    -------
    ConfigurationParameters
        The configuration parameters for bone remodelling simulations.

    """
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return ConfigurationParameters(output_dir=output_dir)

def main() -> None:
    """Handle CLI arguments and execute the appropriate module."""
    parser = argparse.ArgumentParser(description="Unified CLI for Thesis modules.")
    parser.add_argument(
        "module",
        type=str,
        choices=["animation", "generate_data", "surrogate_training", "surrogate_evaluation"],
        help="The module to run. Choices are: animation, generate_data, surrogate_training, surrogate_evaluation.",
    )
    args, remaining_args = parser.parse_known_args()

    logger = setup_logging()
    configuration_parameters =build_configuration_parameters()
    logger.info(f"Output directory set to: {configuration_parameters.output_dir}")

    # Dispatch to the appropriate module based on the argument
    if args.module == "animation":
        from bone_remodelling.forward_model.density_animation import cli as generate_animation  # noqa: I001, PLC0415
        generate_animation(configuration_parameters, remaining_args)

    elif args.module == "generate_data":
        from bone_remodelling.forward_data.generator import cli as generate_data  # noqa: I001, PLC0415
        generate_data(configuration_parameters, remaining_args)

    elif args.module == "surrogate_training":
        from bone_remodelling.surrogate_model.train_surrogate import cli as surrogate_training  # noqa: I001, PLC0415
        surrogate_training(configuration_parameters, remaining_args)

    elif args.module == "surrogate_evaluation":
        from bone_remodelling.surrogate_model.evaluate_surrogate import cli as surrogate_evaluation  # noqa: I001, PLC0415
        surrogate_evaluation(configuration_parameters, remaining_args)

    else:
        raise RuntimeError(f"Unknown module: {args.module}")

# TODO: Create a logfile which contains outputs of the log for each run.
# TODO: Append on the fly datageneration


if __name__ == "__main__":
    main()
