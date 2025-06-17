#!/bin/bash
#SBATCH --job-name="example_job"
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=gpu-v100
#SBATCH --account=research-me-bme
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Load modules:
module load 2024r1
module load miniconda3
module load cuda/12.1

# Activate conda environment:
conda activate handSSM

# Run job
srun python train.py --model="2d" --config_file="/scratch/etwstay/selfsupLandmarks/configs/distal_phalange.json"

# Deactivate conda environment:
conda deactivate