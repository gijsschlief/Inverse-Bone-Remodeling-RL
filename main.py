import argparse
import sys

def forward_simulation(args):
    from forward_model.main import main as fenics_simulator
    fenics_simulator()

def generate_data(args):
    from forward_model.data_generation import main as generate_training_data
    generate_training_data(samples=args.samples)

def generate_data_parallel(args):
    from forward_model.data_generation_parallel import main as generate_training_data_parallel
    generate_training_data_parallel(samples=args.samples)

def train_surrogate(args):
    from surrogate_model.trainer import main as surrogate_trainer
    surrogate_trainer(epochs=args.epochs)

def load_surrogate(args):
    from surrogate_model.loader import main as surrogate_loader
    surrogate_loader()

def main():
    parser = argparse.ArgumentParser(description="Inverse Bone Remodeling Framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Data generation
    p_data = subparsers.add_parser("generate", help="Generate dataset with FEniCS")
    p_data.add_argument("--samples", type=int, default=100, help="Number of samples")
    p_data.set_defaults(func=generate_data)

    # Surrogate training
    p_sur = subparsers.add_parser("train_surrogate", help="Train the surrogate model")
    p_sur.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p_sur.set_defaults(func=train_surrogate)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    sys.path.append("/home/gijs/Desktop/Thesis/Thesis_code")
    main()
