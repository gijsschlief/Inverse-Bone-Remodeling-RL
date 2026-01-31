"""Unified CLI for Thesis Modules."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

CLI_MODULES = {
    "forward-model": "forward_model/forward_modeling_cli.py",
    "data-generator": "data_generator_cli.py",
}


def main() -> None:
    """Handle CLI arguments and execute the appropriate module."""
    parser = argparse.ArgumentParser(description="Unified CLI for Thesis modules.")
    parser.add_argument(
        "module",
        choices=CLI_MODULES.keys(),
        help="Which module to run",
    )
    args, remaining_args = parser.parse_known_args()

    module_path = Path(__file__).resolve().parent / CLI_MODULES[args.module]

    if not module_path.exists():
        logger.info(f"Module file not found: {module_path}")
        sys.exit(1)

    # Call the module as a subprocess with the remaining CLI args
    command = [sys.executable, str(module_path), *remaining_args]
    sys.exit(subprocess.call(command))


if __name__ == "__main__":
    main()
