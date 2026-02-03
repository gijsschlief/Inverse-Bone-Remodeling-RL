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
    config = ConfigurationParameters(
        output_dir=Path(__file__).resolve().parent / "data"
    )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return config

def main() -> None:
    """Handle CLI arguments and execute the appropriate module."""
    parser = argparse.ArgumentParser(description="Unified CLI for Thesis modules.")
    parser.add_argument(
        "module",
        type=str,
        choices=["animation", "inverse_model", "data_processing"],
        help="The module to run.",
    )
    args, remaining_args = parser.parse_known_args()

    logger = setup_logging()
    configuration_parameters =build_configuration_parameters()
    logger.info(f"Output directory set to: {configuration_parameters.output_dir}")

    # Dispatch to the appropriate module based on the argument
    if args.module == "animation":
        from bone_remodelling.forward_model.density_animation import run as forward_main  # noqa: I001, PLC0415
        forward_main(configuration_parameters)

    elif args.module == "inverse_model":
        from bone_remodelling.inverse_model.cli import main as inverse_main
        inverse_main(configuration_parameters, remaining_args)

    elif args.module == "data_processing":
        from bone_remodelling.data_processing.cli import main as data_main  # noqa: PLC0415
        data_main(configuration_parameters, remaining_args)

    else:
        raise RuntimeError(f"Unknown module: {args.module}")



if __name__ == "__main__":
    main()
