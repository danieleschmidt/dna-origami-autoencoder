#!/bin/bash

# DNA-Origami-AutoEncoder Development Container Post-Create Script
# This script sets up the development environment after container creation

set -e

echo "🧬 Setting up DNA-Origami-AutoEncoder development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
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
    wget \
    curl \
    vim \
    htop \
    tree \
    jq \
    unzip

# Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "🐍 Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/conda
    rm /tmp/miniconda.sh
    echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
    export PATH="/opt/conda/bin:$PATH"
fi

# Initialize conda for bash and zsh
echo "🔧 Configuring conda..."
/opt/conda/bin/conda init bash
/opt/conda/bin/conda init zsh

# Create development environment
echo "🌱 Creating conda environment..."
if [ -f "environment-dev.yml" ]; then
    /opt/conda/bin/conda env create -f environment-dev.yml
else
    echo "⚠️  environment-dev.yml not found, creating basic environment..."
    /opt/conda/bin/conda create -n dna-origami-ae-dev python=3.11 -y
fi

# Activate environment and install core packages
echo "📚 Installing core Python packages..."
source /opt/conda/etc/profile.d/conda.sh
conda activate dna-origami-ae-dev

# Install core scientific computing stack
conda install -y -c conda-forge \
    numpy \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets \
    seaborn \
    plotly \
    bokeh \
    h5py \
    networkx \
    numba \
    cython

# Install PyTorch with CUDA support
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install additional ML/AI packages
pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    wandb \
    tensorboard \
    optuna \
    ray[tune] \
    lightning \
    torchmetrics

# Install bioinformatics packages
conda install -y -c conda-forge -c bioconda \
    biopython \
    mdanalysis \
    nglview \
    rdkit \
    openmm \
    pymol-open-source

# Install development tools
pip install --no-cache-dir \
    black \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-benchmark \
    pre-commit \
    ruff \
    isort \
    bandit \
    safety \
    sphinx \
    sphinx-rtd-theme \
    nbsphinx \
    myst-parser

# Install package in development mode
if [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
    echo "🔧 Installing package in development mode..."
    pip install -e ".[dev]"
fi

# Setup pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🎣 Installing pre-commit hooks..."
    pre-commit install
fi

# Install oxDNA (DNA simulation engine)
echo "🧬 Installing oxDNA simulation engine..."
cd /tmp
git clone https://github.com/lorenzo-rovigatti/oxDNA.git
cd oxDNA
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install
cd /workspaces/dna-origami-autoencoder

# Setup Jupyter extensions
echo "🪐 Configuring Jupyter..."
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install nglview-js-widgets

# Create common directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,external,synthetic}
mkdir -p experiments
mkdir -p models/{trained,checkpoints}
mkdir -p results/{figures,reports}
mkdir -p logs
mkdir -p notebooks/{exploratory,analysis,tutorials}
mkdir -p tests/{unit,integration,performance}
mkdir -p docs/{api,tutorials,examples}
mkdir -p scripts/{data,training,evaluation}

# Setup Git hooks and configuration
echo "🔧 Configuring Git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Create useful aliases
echo "⚡ Setting up aliases..."
cat >> ~/.bashrc << 'EOF'

# DNA-Origami-AutoEncoder Development Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Conda aliases
alias cenv='conda env list'
alias cact='conda activate'
alias cdeact='conda deactivate'

# Python/Development aliases
alias py='python'
alias ipy='ipython'
alias jlab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias nb='jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Testing aliases
alias pytest-cov='pytest --cov=dna_origami_ae --cov-report=html --cov-report=term-missing'
alias test-fast='pytest -x -v tests/'
alias test-all='pytest -v tests/'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
alias gd='git diff'

# Docker aliases (if needed)
alias dc='docker-compose'
alias dps='docker ps'
alias di='docker images'

# Project specific
alias cddata='cd /workspaces/dna-origami-autoencoder/data'
alias cdnb='cd /workspaces/dna-origami-autoencoder/notebooks'
alias cdexp='cd /workspaces/dna-origami-autoencoder/experiments'
EOF

# Setup ZSH aliases if ZSH is being used
if [ -f ~/.zshrc ]; then
    cat >> ~/.zshrc << 'EOF'

# DNA-Origami-AutoEncoder Development Aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Conda aliases
alias cenv='conda env list'
alias cact='conda activate'
alias cdeact='conda deactivate'

# Python/Development aliases
alias py='python'
alias ipy='ipython'
alias jlab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias nb='jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Testing aliases
alias pytest-cov='pytest --cov=dna_origami_ae --cov-report=html --cov-report=term-missing'
alias test-fast='pytest -x -v tests/'
alias test-all='pytest -v tests/'

# Project specific
alias cddata='cd /workspaces/dna-origami-autoencoder/data'
alias cdnb='cd /workspaces/dna-origami-autoencoder/notebooks'
alias cdexp='cd /workspaces/dna-origami-autoencoder/experiments'
EOF
fi

# Create default environment activation script
echo "🔄 Setting up automatic environment activation..."
cat > ~/.activate_conda_env << 'EOF'
#!/bin/bash
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    if conda env list | grep -q "dna-origami-ae-dev"; then
        conda activate dna-origami-ae-dev
        echo "🧬 Activated dna-origami-ae-dev environment"
    fi
fi
EOF

# Add to bashrc and zshrc
echo "source ~/.activate_conda_env" >> ~/.bashrc
if [ -f ~/.zshrc ]; then
    echo "source ~/.activate_conda_env" >> ~/.zshrc
fi

# Setup environment variables
cat >> ~/.bashrc << 'EOF'

# DNA-Origami-AutoEncoder Environment Variables
export DNA_ORIGAMI_AE_ROOT="/workspaces/dna-origami-autoencoder"
export DNA_ORIGAMI_AE_DATA_DIR="$DNA_ORIGAMI_AE_ROOT/data"
export DNA_ORIGAMI_AE_MODELS_DIR="$DNA_ORIGAMI_AE_ROOT/models"
export DNA_ORIGAMI_AE_RESULTS_DIR="$DNA_ORIGAMI_AE_ROOT/results"

# CUDA configuration
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

# Python optimization
export PYTHONPATH="$DNA_ORIGAMI_AE_ROOT:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8

# Development settings
export JUPYTER_CONFIG_DIR="$DNA_ORIGAMI_AE_ROOT/.jupyter"
export IPYTHONDIR="$DNA_ORIGAMI_AE_ROOT/.ipython"
EOF

if [ -f ~/.zshrc ]; then
    cat >> ~/.zshrc << 'EOF'

# DNA-Origami-AutoEncoder Environment Variables
export DNA_ORIGAMI_AE_ROOT="/workspaces/dna-origami-autoencoder"
export DNA_ORIGAMI_AE_DATA_DIR="$DNA_ORIGAMI_AE_ROOT/data"
export DNA_ORIGAMI_AE_MODELS_DIR="$DNA_ORIGAMI_AE_ROOT/models"
export DNA_ORIGAMI_AE_RESULTS_DIR="$DNA_ORIGAMI_AE_ROOT/results"

# CUDA configuration
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

# Python optimization
export PYTHONPATH="$DNA_ORIGAMI_AE_ROOT:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8

# Development settings
export JUPYTER_CONFIG_DIR="$DNA_ORIGAMI_AE_ROOT/.jupyter"
export IPYTHONDIR="$DNA_ORIGAMI_AE_ROOT/.ipython"
EOF
fi

# Create welcome message
cat > ~/.welcome_message << 'EOF'
🧬 DNA-Origami-AutoEncoder Development Environment

Quick Start Commands:
  jlab          - Start Jupyter Lab
  test-fast     - Run fast tests
  pytest-cov    - Run tests with coverage
  cenv          - List conda environments
  
Project Structure:
  📁 data/       - Datasets (raw, processed, synthetic)
  📁 notebooks/  - Jupyter notebooks
  📁 experiments/- Research experiments
  📁 models/     - Trained models and checkpoints
  📁 tests/      - Test suites
  📁 docs/       - Documentation

GPU Support: ✅ CUDA enabled
Package Status: Development installation ready

Happy coding! 🚀
EOF

echo ""
echo "$(cat ~/.welcome_message)"
echo ""

# Set executable permissions
chmod +x ~/.activate_conda_env

echo "✅ Development environment setup complete!"
echo "🔄 Please reload your shell or run 'source ~/.bashrc' to activate all changes."