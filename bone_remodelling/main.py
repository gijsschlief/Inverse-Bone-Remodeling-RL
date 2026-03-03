"""Unified CLI for Thesis Modules."""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from bone_remodelling.parameters import ConfigurationParameters


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to both console and a file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Console Handler (for the user)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler (for the logfile)
    logfile = output_dir / "logs"
    logfile.mkdir(parents=True, exist_ok=True)
    log_file = logfile.resolve()
    log_path = log_file / Path(
        f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
    )
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    return logger


def build_configuration_parameters() -> ConfigurationParameters:
    """Set up the data directory for simulations.

    Returns
    -------
    ConfigurationParameters
        The configuration parameters for bone remodelling simulations.

    """
    output_dir = Path(__file__).resolve().parent.parent / Path("data")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return ConfigurationParameters(output_dir=output_dir)


def main() -> None:  # noqa: C901, PLR0912
    """Handle CLI arguments and execute the appropriate module."""
    parser = argparse.ArgumentParser(description="Unified CLI for Thesis modules.")
    parser.add_argument(
        "module",
        type=str,
        choices=[
            "animate_forward_model",
            "generate_forward_data",
            "train_surrogate",
            "evaluate_surrogate",
            "train_inverse",
            "evaluate_inverse",
            "train_rl",
            "evaluate_rl",
            "analysis",
            "train_all",
            "evaluate_all",
        ],
        help="The module to run. Choices are: animate_forward_model, generate_forward_data, train_surrogate, evaluate_surrogate, train_inverse, evaluate_inverse, train_rl, evaluate_rl, analysis, train_all, evaluate_all.",
    )
    args, remaining_args = parser.parse_known_args()
    configuration_parameters = build_configuration_parameters()

    logger = setup_logging(configuration_parameters.output_dir)
    logger.info(f"--- Starting Module: {args.module} ---")
    logger.info(f"Output directory set to: {configuration_parameters.output_dir}")

    # Dispatch to the appropriate module based on the argument
    try:
        if args.module == "animate_forward_model":
            from bone_remodelling.forward_model.density_animation import (  # noqa: PLC0415
                cli as generate_animation,
            )

            generate_animation(configuration_parameters, remaining_args)

        elif args.module == "generate_forward_data":
            from bone_remodelling.forward_data.generator import cli as generate_data  # noqa: I001, PLC0415

            generate_data(configuration_parameters, remaining_args)

        elif args.module == "train_surrogate":
            from bone_remodelling.surrogate_model.train_surrogate import (  # noqa: PLC0415
                cli as surrogate_training,
            )

            surrogate_training(configuration_parameters, remaining_args)

        elif args.module == "evaluate_surrogate":
            from bone_remodelling.surrogate_model.evaluate_surrogate import (  # noqa: PLC0415
                cli as surrogate_evaluation,
            )

            surrogate_evaluation(configuration_parameters, remaining_args)

        elif args.module == "train_inverse":
            from bone_remodelling.inverse_model.train_inverse import (  # noqa: PLC0415
                cli as inverse_training,
            )

            inverse_training(configuration_parameters, remaining_args)

        elif args.module == "evaluate_inverse":
            from bone_remodelling.inverse_model.evaluate_inverse import (  # noqa: PLC0415
                cli as inverse_evaluation,
            )

            inverse_evaluation(configuration_parameters, remaining_args)

        elif args.module == "train_rl":
            import matplotlib  # noqa: ICN001, PLC0415

            matplotlib.use("TkAgg")
            from bone_remodelling.rl_model.run_rl_environment import cli as rl_training  # noqa: I001, PLC0415

            rl_training(configuration_parameters, remaining_args)

        elif args.module == "evaluate_rl":
            from bone_remodelling.rl_model.evaluate_agent import cli as rl_evaluation  # noqa: I001, PLC0415

            rl_evaluation(configuration_parameters)

        elif args.module == "analysis":
            from bone_remodelling.analysis.cli import cli as analysis  # noqa: I001, PLC0415

            analysis(configuration_parameters, remaining_args)

        elif args.module == "train_all":
            logger.info(
                f"Running all modules in sequence...{remaining_args} not used for train_all.",
            )
            train_all(configuration_parameters)

        elif args.module == "evaluate_all":
            logger.info(
                f"Evaluating all modules in sequence...{remaining_args} not used for evaluate_all.",
            )
            evaluate_all(configuration_parameters)

        else:
            raise RuntimeError(f"Unknown module: {args.module}")
    except Exception as e:
        logger.error(f"An error occurred while running the module {args.module}: {e}")
        raise
    logger.info(f"--- Finished Module: {args.module} ---")


def train_all(configuration_parameters: ConfigurationParameters) -> None:
    """Train all modules in sequence."""
    from bone_remodelling.forward_data.generator import cli as generate_data  # noqa: I001, PLC0415
    from bone_remodelling.surrogate_model.train_surrogate import (  # noqa: PLC0415
        cli as surrogate_training,
    )
    from bone_remodelling.inverse_model.train_inverse import cli as inverse_training  # noqa: PLC0415
    from bone_remodelling.rl_model.run_rl_environment import cli as rl_training  # noqa: PLC0415

    generate_data(configuration_parameters, ["--samples", "100000", "--type", "merger"])
    surrogate_training(configuration_parameters, [])
    generate_data(
        configuration_parameters,
        ["--samples", "50000", "--type", "triangular"],
    )
    inverse_training(configuration_parameters, [])
    rl_training(configuration_parameters, [])


def evaluate_all(configuration_parameters: ConfigurationParameters) -> None:
    """Evaluate all modules in sequence."""
    from bone_remodelling.forward_model.density_animation import (  # noqa: PLC0415
        cli as generate_animation,
    )
    from bone_remodelling.inverse_model.evaluate_inverse import (  # noqa: PLC0415
        cli as inverse_evaluation,
    )
    from bone_remodelling.rl_model.evaluate_agent import (  # noqa: PLC0415
        cli as rl_evaluation,
    )
    from bone_remodelling.surrogate_model.evaluate_surrogate import (  # noqa: PLC0415
        cli as surrogate_evaluation,
    )

    generate_animation(
        configuration_parameters,
        ["--type", "validation", "--animator", "matplotlib"],
    )
    surrogate_evaluation(configuration_parameters, [])
    inverse_evaluation(configuration_parameters, [])
    rl_evaluation(configuration_parameters)


# TODO(gijsschlief): Add parameters to the parser of the RL model to give more freedom to the user
# https://github.com/gijsschlief/Inverse-Bone-Remodeling-RL/issues/20

# TODO(gijsschlief): Create the parsing of the analysis module that can generate the different figures
# https://github.com/gijsschlief/Inverse-Bone-Remodeling-RL/issues/19

if __name__ == "__main__":
    main()
