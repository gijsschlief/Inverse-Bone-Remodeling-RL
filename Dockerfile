# Use a Miniconda base to handle the FEniCS installation
FROM continuumio/miniconda3:23.10.0-1

# 1. Install system dependencies
# libgl1 and libxrender1 are required for PyVista/VTK (headless rendering)
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libxrender1 \
    libfontconfig1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up the working directory
WORKDIR /app

# 3. Install FEniCS via Conda first (Legacy 2019.1.0)
# Note: We create the environment 'bone_env' with Python 3.11 as requested
RUN conda create -n bone_env python=3.11 -y && \
    conda install -n bone_env -c conda-forge fenics=2019.1.0 -y && \
    conda clean -afy

# Set the path so subsequent commands use the conda environment
ENV PATH="/opt/conda/envs/bone_env/bin:$PATH"

# 4. Install Poetry
RUN pip install --no-cache-dir poetry

# 5. Copy configuration files
COPY pyproject.toml poetry.lock* README.md ./

# 6. Install project dependencies via Poetry
# We use --no-interaction and tell poetry NOT to create its own virtualenv 
# because we are already inside the Conda environment.
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# 7. Copy the rest of the source code
COPY . .

# 8. Final installation of the package itself
RUN poetry install --no-interaction --no-ansi

# Set the default command to the CLI app defined in your toml
ENTRYPOINT ["python3", "bone_remodelling/main.py"]
