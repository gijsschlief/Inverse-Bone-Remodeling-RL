import argparse
import sys
from pathlib import Path

def forward_simulation(args):
    from forward_model.main import main as fenics_simulator
    fenics_simulator()

def generate_data(args):
    from Thesis_code.forward_model.data_generation import main as generate_data
    generate_data()

def train_surrogate(args):
    from surrogate_model.trainer import main as surrogate_trainer
    surrogate_trainer(epochs=args.epochs)

def load_surrogate(args):
    from surrogate_model.loader import main as surrogate_loader
    surrogate_loader()

def main():
    parser = argparse.ArgumentParser(description="Inverse Bone Remodeling Framework")
    subparsers = parser.add_subparsers(dest="command")

    # Forward simulation
    p_forward = subparsers.add_parser("forward", help="Run the forward simulation with FEniCS")
    p_forward.add_argument("--time_steps", type=int, default=100, help="Number of time steps for the simulation")
    p_forward.add_argument("--dt", type=float, default=0.01, help="Time step size")
    p_forward.add_argument("--min_density", type=float, default=0.01, help="Minimum bone density")
    p_forward.add_argument("--max_density", type=float, default=1.74, help="Maximum bone density")
    p_forward.add_argument("--file_location", type=str, default=None, help="Location of FEniCS files")
    p_forward.add_argument("--tolerance", type=float, default=1E-14, help="Tolerance for convergence criteria")
    p_forward.add_argument("--save", action="store_true", help="Save the simulation results")
    p_forward.add_argument("--plot", action="store_true", help="Plot the density simulation")
    p_forward.set_defaults(func=forward_simulation)

    # Data generation
    p_data = subparsers.add_parser("generate", help="Generate dataset with FEniCS")
    p_data.add_argument("--samples", type=int, default=100, help="Number of samples")
    p_data.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "raw"), help="Directory to save output.")
    p_data.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate.")
    p_data.add_argument("--force_max", type=int, default=2, help="Maximum force magnitude.")
    p_data.add_argument("--force_count_max", type=int, default=7, help="Max number of force applications.")
    p_data.add_argument("--batch_seed", type=int, default=np.random.randint(0, 1_000_000), help="Random seed.")
    p_data.add_argument("--mode", type=str, choices=["parallel", "sequential", "edge"], default="parallel", help="Generation mode: 'parallel', 'sequential', or 'edge'.")
    p_data.set_defaults(func=generate_data)

    # Surrogate training
    p_sur = subparsers.add_parser("train_surrogate", help="Train the surrogate model")
    p_sur.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p_sur.set_defaults(func=train_surrogate)

    # Surrogate loading
    p_load_sur = subparsers.add_parser("load_surrogate", help="Load the surrogate model")
    p_load_sur.set_defaults(func=load_surrogate)

    # Parse arguments
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    args.func(args)

if __name__ == "__main__":
    #default_dir = Path(__file__).resolve().parent
    #sys.path.insert(0, str(default_dir))
    #main()
    import numpy as np
    print(np.version.full_version)
