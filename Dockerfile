# DNA-Origami-AutoEncoder Production Container
# Multi-stage build for optimized production deployment

# =============================================================================
# Base Stage: CUDA-enabled Python environment
# =============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    pkg-config \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libeigen3-dev \
    libfftw3-dev \
    libboost-all-dev \
    graphviz \
    graphviz-dev \
    git \
    git-lfs \
    curl \
    wget \
    vim \
    htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r dnaorigami && useradd -r -g dnaorigami -d /app -s /bin/bash dnaorigami

# Set up Python
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# =============================================================================
# Dependencies Stage: Install Python dependencies
# =============================================================================
FROM base as dependencies

# Copy dependency files
COPY requirements.txt pyproject.toml ./
COPY requirements/ requirements/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional scientific computing packages
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    lightning \
    wandb \
    tensorboard \
    optuna \
    ray[tune] \
    biopython \
    mdanalysis \
    rdkit \
    openmm

# =============================================================================
# Build Stage: Compile native extensions and tools
# =============================================================================
FROM dependencies as build

# Install build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build oxDNA simulation engine
WORKDIR /tmp
RUN git clone https://github.com/lorenzo-rovigatti/oxDNA.git \
    && cd oxDNA \
    && mkdir build \
    && cd build \
    && cmake .. -DCUDA=ON -DCUDA_COMMON_ARCH=ON \
    && make -j$(nproc) \
    && make install \
    && cd / \
    && rm -rf /tmp/oxDNA

# Create application directory
WORKDIR /app

# Copy source code
COPY . .

# Install application in development mode
RUN pip install --no-cache-dir -e .

# =============================================================================
# Production Stage: Minimal runtime image
# =============================================================================
FROM base as production

# Copy Python environment from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy compiled tools from build stage
COPY --from=build /usr/local/bin/oxDNA /usr/local/bin/
COPY --from=build /usr/local/lib/liboxDNA* /usr/local/lib/

# Create necessary directories
RUN mkdir -p /app/{data,models,results,logs} \
    && chown -R dnaorigami:dnaorigami /app

# Copy application code
COPY --chown=dnaorigami:dnaorigami . /app/

# Set working directory
WORKDIR /app

# Switch to application user
USER dnaorigami

# Install application
RUN pip install --user --no-cache-dir -e .

# Expose ports
EXPOSE 8888 6006 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import dna_origami_ae; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Development Stage: Full development environment
# =============================================================================
FROM build as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-benchmark \
    black \
    flake8 \
    mypy \
    pre-commit \
    sphinx \
    nbsphinx

# Install Jupyter extensions
RUN pip install --no-cache-dir \
    jupyterlab \
    jupyter-widgets \
    ipywidgets \
    nglview

# Enable Jupyter extensions
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    && jupyter labextension install nglview-js-widgets

# Create development user
USER root
RUN usermod -a -G sudo dnaorigami \
    && echo "dnaorigami ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch back to application user
USER dnaorigami

# Set up development environment
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV DNA_ORIGAMI_AE_ENV=development

# Default development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Testing Stage: Optimized for CI/CD
# =============================================================================
FROM dependencies as testing

# Install testing dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-benchmark \
    pytest-timeout \
    coverage[toml] \
    black \
    flake8 \
    mypy \
    bandit \
    safety

# Copy source code
COPY . /app
WORKDIR /app

# Install application
RUN pip install --no-cache-dir -e .

# Run tests by default
CMD ["pytest", "--cov=dna_origami_ae", "--cov-report=xml", "--cov-report=term-missing"]

# =============================================================================
# GPU Stage: Optimized for GPU workloads
# =============================================================================
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as gpu

# Copy base setup
COPY --from=base /usr/local /usr/local
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Install CUDA-optimized packages
RUN pip install --no-cache-dir \
    cupy-cuda12x \
    rapids-cudf \
    rapids-cuml \
    && pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy application
COPY . /app
WORKDIR /app

# Install application
RUN pip install --no-cache-dir -e .

# Set GPU-specific environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default GPU command
CMD ["python", "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"]