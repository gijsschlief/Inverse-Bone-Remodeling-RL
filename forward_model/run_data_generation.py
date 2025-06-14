import argparse
from pathlib import Path

import numpy as np
from fenics import set_log_level, LogLevel

from data_generation import TrainingDataGenerator

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training or edge case data for the forward model.")
    default_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    parser.add_argument("--output_dir", type=str, default=str(default_dir), help="Directory to save output.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate.")
    parser.add_argument("--force_max", type=int, default=2, help="Maximum force magnitude.")
    parser.add_argument("--force_count_max", type=int, default=7, help="Max number of force applications.")
    parser.add_argument("--batch_seed", type=int, default=np.random.randint(0, 1_000_000), help="Random seed.")
    parser.add_argument("--mode", type=str, choices=["parallel", "sequential", "edge"], default="parallel",
                        help="Generation mode: 'parallel', 'sequential', or 'edge'.")

    args = parser.parse_args()
    set_log_level(LogLevel.ERROR)

    generator = TrainingDataGenerator(
        output_dir=args.output_dir,
        force_max=args.force_max,
        force_count_max=args.force_count_max,
        batch_seed=args.batch_seed
    )

    if args.mode == "parallel":
        generator.generate_parallel(args.num_samples)
    elif args.mode == "sequential":
        generator.generate_sequential(args.num_samples)
    elif args.mode == "edge":
        generator.generate_edge_cases(args.num_samples)


if __name__ == "__main__":
    main()