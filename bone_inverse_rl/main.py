import argparse

def generate_data(args):
    from forward_model import fenics_simulator
    fenics_simulator.generate_dataset(num_samples=args.samples)

def train_surrogate(args):
    from surrogate_model import cnn_surrogate
    cnn_surrogate.train_model(epochs=args.epochs)

def train_rl(args):
    from rl_model import train_agent
    train_agent.train(timesteps=args.timesteps)

def evaluate(args):
    from rl_model import train_agent
    train_agent.evaluate_policy()

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

    # RL training
    p_rl = subparsers.add_parser("train_rl", help="Train the RL agent")
    p_rl.add_argument("--timesteps", type=int, default=100000, help="RL training steps")
    p_rl.set_defaults(func=train_rl)

    # Evaluation
    p_eval = subparsers.add_parser("evaluate", help="Evaluate the RL agent")
    p_eval.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
