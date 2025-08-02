# DNA-Origami-AutoEncoder Makefile
# Standardized build commands and development workflows

.PHONY: help install install-dev test lint format clean build docker-build docker-run docs

# Default target
help: ## Show this help message
	@echo "DNA-Origami-AutoEncoder Development Commands"
	@echo "============================================="
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# Development Setup
# =============================================================================

install: ## Install package for production
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

setup-conda: ## Create conda environment
	conda env create -f environment.yml
	conda activate dna-origami-ae
	$(MAKE) install-dev

update-deps: ## Update all dependencies
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -r requirements.txt
	pip install --upgrade -e ".[dev]"

# =============================================================================
# Code Quality
# =============================================================================

format: ## Format code with black and isort
	black dna_origami_ae/ tests/ scripts/
	isort dna_origami_ae/ tests/ scripts/

format-check: ## Check code formatting
	black --check dna_origami_ae/ tests/ scripts/
	isort --check-only dna_origami_ae/ tests/ scripts/

lint: ## Run linting checks
	flake8 dna_origami_ae/ tests/ scripts/
	mypy dna_origami_ae/
	bandit -r dna_origami_ae/

lint-fix: ## Fix linting issues automatically
	$(MAKE) format
	flake8 dna_origami_ae/ tests/ scripts/ --extend-ignore=E203,W503

security: ## Run security checks
	bandit -r dna_origami_ae/
	safety check

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest tests/unit/

test-integration: ## Run integration tests only
	pytest tests/integration/

test-performance: ## Run performance tests only
	pytest tests/performance/

test-coverage: ## Run tests with coverage report
	pytest --cov=dna_origami_ae --cov-report=html --cov-report=term-missing

test-fast: ## Run fast tests only (skip slow tests)
	pytest -m "not slow"

test-gpu: ## Run GPU tests only
	pytest -m gpu

test-watch: ## Run tests in watch mode
	pytest-watch

benchmark: ## Run benchmarks
	pytest --benchmark-only tests/performance/

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build
	cd docs && make clean

docs-auto: ## Auto-build documentation on changes
	sphinx-autobuild docs docs/_build/html

# =============================================================================
# Container Operations
# =============================================================================

docker-build: ## Build Docker image
	docker build -t dna-origami-ae:latest .

docker-build-dev: ## Build development Docker image
	docker build --target development -t dna-origami-ae:dev .

docker-build-gpu: ## Build GPU-enabled Docker image
	docker build --target gpu -t dna-origami-ae:gpu .

docker-run: ## Run Docker container
	docker run -it --rm -p 8888:8888 -v $(PWD):/app dna-origami-ae:latest

docker-run-dev: ## Run development Docker container
	docker run -it --rm -p 8888:8888 -v $(PWD):/app dna-origami-ae:dev

docker-run-gpu: ## Run GPU Docker container
	docker run -it --rm --gpus all -p 8888:8888 -v $(PWD):/app dna-origami-ae:gpu

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docker-compose-build: ## Build all Docker images
	docker-compose build

docker-shell: ## Open shell in running container
	docker exec -it dna-origami-ae-app bash

# =============================================================================
# Data and Model Management
# =============================================================================

download-data: ## Download sample datasets
	python scripts/download_sample_data.py

prepare-data: ## Prepare datasets for training
	python scripts/prepare_datasets.py

validate-data: ## Validate dataset integrity
	python scripts/validate_datasets.py

clean-data: ## Clean temporary data files
	find data/ -name "*.tmp" -delete
	find data/ -name "*.cache" -delete

backup-models: ## Backup trained models
	tar -czf models/backup_$(shell date +%Y%m%d_%H%M%S).tar.gz models/trained/

# =============================================================================
# Development Utilities
# =============================================================================

jupyter: ## Start Jupyter Lab
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

notebook: ## Start Jupyter Notebook
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

tensorboard: ## Start TensorBoard
	tensorboard --logdir=logs/tensorboard --host=0.0.0.0 --port=6006

profile: ## Profile application performance
	python -m cProfile -o profile.prof scripts/profile_main.py
	python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"

memory-profile: ## Profile memory usage
	python -m memory_profiler scripts/memory_profile_main.py

# =============================================================================
# Database Operations
# =============================================================================

db-init: ## Initialize database
	python scripts/init_database.py

db-migrate: ## Run database migrations
	python scripts/migrate_database.py

db-backup: ## Backup database
	python scripts/backup_database.py

db-restore: ## Restore database from backup
	python scripts/restore_database.py --backup=$(BACKUP_FILE)

# =============================================================================
# Experiment Management
# =============================================================================

experiment-new: ## Create new experiment
	python scripts/create_experiment.py --name=$(EXP_NAME)

experiment-run: ## Run experiment
	python scripts/run_experiment.py --config=$(CONFIG_FILE)

experiment-status: ## Check experiment status
	python scripts/check_experiments.py

experiment-clean: ## Clean old experiments
	python scripts/clean_experiments.py --days=30

# =============================================================================
# Deployment
# =============================================================================

deploy-staging: ## Deploy to staging environment
	docker-compose -f docker-compose.staging.yml up -d

deploy-prod: ## Deploy to production environment
	docker-compose -f docker-compose.prod.yml up -d

health-check: ## Check application health
	curl -f http://localhost:8080/health || exit 1

logs: ## View application logs
	docker-compose logs -f dna-origami-ae

logs-tail: ## Tail application logs
	tail -f logs/dna-origami-ae.log

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-all: clean ## Clean everything including data and models
	rm -rf data/processed/*
	rm -rf models/trained/*
	rm -rf results/*
	rm -rf logs/*.log

clean-docker: ## Clean Docker images and containers
	docker system prune -f
	docker image prune -f
	docker container prune -f
	docker volume prune -f

# =============================================================================
# CI/CD Support
# =============================================================================

ci-setup: ## Setup CI environment
	pip install --upgrade pip
	pip install -e ".[dev]"

ci-test: ## Run CI tests
	pytest --cov=dna_origami_ae --cov-report=xml --cov-report=term-missing

ci-lint: ## Run CI linting
	black --check dna_origami_ae/ tests/
	flake8 dna_origami_ae/ tests/
	mypy dna_origami_ae/

ci-security: ## Run CI security checks
	bandit -r dna_origami_ae/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

ci-build: ## Build for CI
	python -m build

# =============================================================================
# Release Management
# =============================================================================

version: ## Show current version
	python -c "import dna_origami_ae; print(dna_origami_ae.__version__)"

tag-release: ## Tag a new release
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)

build-release: ## Build release packages
	python -m build
	twine check dist/*

upload-release: ## Upload release to PyPI
	twine upload dist/*

# =============================================================================
# Monitoring and Observability
# =============================================================================

monitoring-up: ## Start monitoring stack
	docker-compose --profile monitoring up -d

monitoring-down: ## Stop monitoring stack
	docker-compose --profile monitoring down

metrics: ## View current metrics
	curl -s http://localhost:9090/api/v1/query?query=up

alerts: ## Check active alerts
	curl -s http://localhost:9093/api/v1/alerts

# =============================================================================
# Utilities
# =============================================================================

env-info: ## Show environment information
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Git branch: $(shell git branch --show-current)"
	@echo "Git commit: $(shell git rev-parse --short HEAD)"
	@echo "Working directory: $(PWD)"
	@echo "Virtual environment: $(VIRTUAL_ENV)"

check-deps: ## Check for dependency updates
	pip list --outdated

tree: ## Show project structure
	tree -I '__pycache__|*.pyc|.git|.pytest_cache|.mypy_cache|node_modules'

size: ## Show project size
	du -sh .
	du -sh data/ models/ results/ logs/ 2>/dev/null || true

# =============================================================================
# Variables
# =============================================================================

# Get variables from environment or use defaults
VERSION ?= $(shell python -c "import dna_origami_ae; print(dna_origami_ae.__version__)" 2>/dev/null || echo "unknown")
CONFIG_FILE ?= configs/default.yaml
EXP_NAME ?= experiment_$(shell date +%Y%m%d_%H%M%S)
BACKUP_FILE ?= $(shell ls -t backups/*.sql | head -1)