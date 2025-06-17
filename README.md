# Thesis Code Repository

Welcome to the repository for my thesis on **Using Reinforcement Learning to Solve the Inverse Bone Remodeling Problem**. This repository contains all the code, scripts, and resources developed and used during the research.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
The goal of this thesis is to explore how reinforcement learning can be applied to solve the inverse bone remodeling problem. This repository includes:
- Implementation of reinforcement learning algorithms.
- Simulation environments for testing and evaluation.
- Data preprocessing and analysis scripts.
- Visualization tools for results and findings.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/gijsschlief/thesis-code.git
    cd thesis-code
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run the main script to start the experiments:
```bash
python thesis_cli
```

## Project Structure
```
Thesis_code/
├── delft_blue/         # Contains .sh files for running the modules on a super computer cluster
├── forward_model/      # Simulation environment for the forward model and data generation
├── rl_model/           # RL models and training scripts
├── surrogate_model/    # Unsupervised learning neural network model to estimate the forward model
├── tests/              # Test module containing test functions for all code
├── thesis_cli.py       # Entry point for running experiments
└── README.md           # Project documentation
```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the codebase.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
Feel free to reach out if you have any questions or need further assistance!
