# Repository Context

## Structure:
./API_DOCUMENTATION.md
./api_server.py
./ARCHITECTURE.md
./automation_system.py
./AUTONOMOUS_SDLC_COMPLETION_REPORT.md
./benchmark_results.md
./build_system.py
./CHANGELOG.md
./CODE_OF_CONDUCT.md
./complete_sdlc_demo.py
./COMPLETION_SUMMARY.md
./CONTRIBUTING.md
./demo_cli.py
./deployment/agent_mesh_xr_production.py
./deployment/autonomous_sdlc_deployment.py
./deployment/production_deployment.py
./DEPLOYMENT.md
./DEPLOYMENT_README.md
./deploy_production.py
./dna_origami_ae/api/middleware.py

## README (if exists):
# DNA-Origami-AutoEncoder

A groundbreaking "wet-lab ML" framework that encodes 8-bit images into self-assembling DNA origami structures and decodes them using transformer models trained on simulated base-pair kinetics. This project bridges synthetic biology and machine learning to demonstrate information storage and retrieval through biological self-assembly.

## Overview

DNA-Origami-AutoEncoder represents digital information as DNA sequences that fold into specific origami patterns. These patterns can be "read" by a neural network trained on molecular dynamics simulations, creating a biological autoencoder system. This opens new possibilities for ultra-dense data storage, biocomputing, and molecular information processing.

## Key Features

### 🧬 Core Scientific Capabilities
- **Image to DNA Encoding**: Convert 8-bit grayscale images to DNA sequences
- **Origami Design**: Automated scaffold/staple design for target shapes
- **Molecular Simulation**: GPU-accelerated DNA folding dynamics
- **Neural Decoding**: Transformer-based pattern recognition from AFM-like data
- **Wet-Lab Protocols**: Ready-to-use protocols for DNA synthesis and imaging
- **Error Correction**: Bio-compatible error correction codes

### 🚀 Production-Ready Infrastructure
- **CLI Interface**: Comprehensive command-line tools for all operations
- **REST API**: High-performance async web API with auto-scaling
- **GPU Acceleration**: CUDA/OpenCL support with automatic fallback
- **Distributed Processing**: Auto-scaling task management with load balancing
- **Advanced Error Handling**: Circuit breakers, retry logic, and recovery strategies
- **Comprehensive Monitoring**: Real-time metrics, health checks, and performance profiling

### 🌍 Global Compliance & Security
- **Multi-Region Support**: GDPR, CCPA, PDPA compliance built-in
- **Data Protection**: Automated anonymization and retention management
- **Audit Trail**: Comprehensive logging for regulatory compliance
- **Security Hardening**: Rate limiting, authentication, input validation

## Installation

```bash
# Basic installation
pip install dna-origami-autoencoder

# With molecular dynamics support
pip install dna-origami-autoencoder[md]

# With wet-lab protocol generation
pip install dna-origami-autoencoder[wetlab]

# Full installation
pip install dna-origami-autoencoder[full]

# Development
git clone https://github.com/yourusername/dna-origami-autoencoder
cd dna-origami-autoencoder

## Main files:
