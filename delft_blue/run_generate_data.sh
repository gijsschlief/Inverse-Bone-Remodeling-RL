#!/bin/bash
#SBATCH --job-name="gen_bone_remodelling_data"
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=compute
#SBATCH --account=Education-ME-MSc-ME
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Load modules:
module load 2024r1
module load python/3.10.12
module load miniconda3
conda activate /home/gschlief/ondemand/bone_remodel

# Create output directory
OUTPUT_DIR="/scratch/gschlief/handSSM_data_generation"
mkdir -p "${OUTPUT_DIR}"

# Run job
# Specify your output directory and number of samples:
srun python forward_model/run_data_generation.py \
    --output_dir "${OUTPUT_DIR}" \
    --num_samples 20 \
    -v

# Deactivate conda environment
conda deactivate
