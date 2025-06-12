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
    git clone https://github.com/your-username/thesis-code.git
    cd thesis-code
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run the main script to start the experiments:
```bash
python main.py
```
Modify the configuration files in the `config/` directory to customize the experiments.

## Project Structure
```
bone_inverse_rl/
├── delft_blue/         # bash files for running simulations on the delft supercomputer cluster
├── forward_model/      # runs the forward bone remodelling finite element solver for datageneration
├── rl_model/           # RL models and training scripts
├── surrogate_model/    # Supervised learning model that estimates forward_model
├── utils/              # Utility functions and helpers
├── test/               # Test functions
├── main.py             # Entry point for running experiments
├── __init__.py         # Empty initilizer
└── README.md           # Project documentation
```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the codebase.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
Feel free to reach out if you have any questions or need further assistance!
