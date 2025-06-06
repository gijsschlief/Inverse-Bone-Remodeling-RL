#!/bin/bash
#SBATCH --job-name="cpd_loop"
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=compute
#SBATCH --account=research-me-bme
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Load modules:
module load 2024r1
module load miniconda3

# Activate conda environment:
conda activate handSSM

# Run job
srun python src/iterative_CPD_loop_0.py

# Deactivate conda environment:
conda deactivate
