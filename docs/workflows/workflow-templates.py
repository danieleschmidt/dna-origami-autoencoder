#!/usr/bin/env python3
"""
GitHub Actions workflow template generator for DNA-Origami-AutoEncoder.

This script generates comprehensive workflow templates that repository maintainers
can manually create due to GitHub App permission limitations.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class WorkflowTemplate:
    """Configuration for a workflow template."""
    
    name: str
    filename: str
    description: str
    triggers: List[str]
    jobs: Dict[str, Any]
    env_vars: Dict[str, str] = None
    secrets_required: List[str] = None
    
    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}
        if self.secrets_required is None:
            self.secrets_required = []

class WorkflowGenerator:
    """Generate comprehensive GitHub Actions workflow templates."""
    
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.workflows_dir = project_root / "docs" / "workflows" / "examples"
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_ci_workflow(self) -> str:
        """Generate comprehensive CI workflow template."""
        workflow = {
            "name": "CI - Continuous Integration",
            "on": {
                "push": {
                    "branches": ["main", "develop"],
                    "paths-ignore": ["docs/**", "*.md"]
                },
                "pull_request": {
                    "branches": ["main", "develop"],
                    "paths-ignore": ["docs/**", "*.md"]
                },
                "workflow_dispatch": None
            },
            "env": {
                "PYTHON_DEFAULT_VERSION": "3.11",
                "CUDA_VERSION": "12.1",
                "CODECOV_TOKEN": "${{ secrets.CODECOV_TOKEN }}",
                "DNA_ORIGAMI_AE_ENV": "testing"
            },
            "jobs": {
                "code-quality": {
                    "name": "Code Quality Checks",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "${{ env.PYTHON_DEFAULT_VERSION }}"}
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -e .[dev]"
                        },
                        {
                            "name": "Run black formatting check",
                            "run": "black --check --diff dna_origami_ae/ tests/"
                        },
                        {
                            "name": "Run isort import sorting",
                            "run": "isort --check-only --diff dna_origami_ae/ tests/"
                        },
                        {
                            "name": "Run flake8 linting",
                            "run": "flake8 dna_origami_ae/ tests/"
                        },
                        {
                            "name": "Run mypy type checking",
                            "run": "mypy dna_origami_ae/"
                        },
                        {
                            "name": "Run bandit security scan",
                            "run": "bandit -r dna_origami_ae/ -f json -o bandit-report.json"
                        },
                        {
                            "name": "Upload bandit results",
                            "uses": "actions/upload-artifact@v3",
                            "if": "always()",
                            "with": {
                                "name": "bandit-report",
                                "path": "bandit-report.json"
                            }
                        }
                    ]
                },
                "test-matrix": {
                    "name": "Test Matrix",
                    "runs-on": "${{ matrix.os }}",
                    "strategy": {
                        "fail-fast": False,
                        "matrix": {
                            "os": ["ubuntu-latest", "windows-latest", "macos-latest"],
                            "python-version": ["3.9", "3.10", "3.11"],
                            "include": [
                                {
                                    "os": "ubuntu-latest",
                                    "python-version": "3.11",
                                    "coverage": True
                                }
                            ]
                        }
                    },
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python ${{ matrix.python-version }}",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "${{ matrix.python-version }}"}
                        },
                        {
                            "name": "Install system dependencies (Ubuntu)",
                            "if": "matrix.os == 'ubuntu-latest'",
                            "run": """
                                sudo apt-get update
                                sudo apt-get install -y libhdf5-dev libopenblas-dev gfortran
                            """
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -e .[dev,test]"
                        },
                        {
                            "name": "Run unit tests",
                            "run": "pytest tests/unit/ -v --tb=short"
                        },
                        {
                            "name": "Run integration tests",
                            "run": "pytest tests/integration/ -v --tb=short"
                        },
                        {
                            "name": "Run tests with coverage",
                            "if": "matrix.coverage",
                            "run": "pytest --cov=dna_origami_ae --cov-report=xml --cov-report=term-missing"
                        },
                        {
                            "name": "Upload coverage to Codecov",
                            "if": "matrix.coverage",
                            "uses": "codecov/codecov-action@v3",
                            "with": {
                                "file": "./coverage.xml",
                                "flags": "unittests",
                                "name": "codecov-umbrella"
                            }
                        }
                    ]
                },
                "gpu-tests": {
                    "name": "GPU Tests",
                    "runs-on": "self-hosted-gpu",
                    "if": "github.event_name != 'pull_request' || contains(github.event.pull_request.labels.*.name, 'test-gpu')",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "${{ env.PYTHON_DEFAULT_VERSION }}"}
                        },
                        {
                            "name": "Install CUDA dependencies",
                            "run": """
                                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
                                pip install -e .[dev,gpu]
                            """
                        },
                        {
                            "name": "Check GPU availability",
                            "run": "python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""
                        },
                        {
                            "name": "Run GPU tests",
                            "run": "pytest tests/ -m gpu -v --tb=short"
                        }
                    ]
                },
                "docker-build": {
                    "name": "Docker Build Test",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Docker Buildx",
                            "uses": "docker/setup-buildx-action@v3"
                        },
                        {
                            "name": "Build test image",
                            "uses": "docker/build-push-action@v5",
                            "with": {
                                "context": ".",
                                "target": "testing",
                                "push": False,
                                "tags": "dna-origami-ae:test-${{ github.sha }}"
                            }
                        },
                        {
                            "name": "Run tests in container",
                            "run": """
                                docker run --rm dna-origami-ae:test-${{ github.sha }} \
                                  pytest --cov=dna_origami_ae --cov-report=term-missing
                            """
                        }
                    ]
                },
                "security-scan": {
                    "name": "Security Scan",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Run Trivy vulnerability scanner",
                            "uses": "aquasecurity/trivy-action@master",
                            "with": {
                                "scan-type": "fs",
                                "format": "sarif",
                                "output": "trivy-results.sarif"
                            }
                        },
                        {
                            "name": "Upload Trivy scan results to GitHub Security tab",
                            "uses": "github/codeql-action/upload-sarif@v2",
                            "with": {"sarif_file": "trivy-results.sarif"}
                        }
                    ]
                }
            }
        }
        
        return self._save_workflow(workflow, "ci.yml")
    
    def generate_cd_workflow(self) -> str:
        """Generate deployment workflow template."""
        workflow = {
            "name": "CD - Continuous Deployment",
            "on": {
                "push": {"branches": ["main"]},
                "release": {"types": ["published"]},
                "workflow_dispatch": {
                    "inputs": {
                        "environment": {
                            "description": "Deployment environment",
                            "required": True,
                            "default": "staging",
                            "type": "choice",
                            "options": ["staging", "production"]
                        },
                        "version": {
                            "description": "Version to deploy",
                            "required": False,
                            "default": "latest"
                        }
                    }
                }
            },
            "env": {
                "REGISTRY": "ghcr.io",
                "IMAGE_NAME": "${{ github.repository }}",
                "PYTHON_VERSION": "3.11"
            },
            "jobs": {
                "build-and-publish": {
                    "name": "Build and Publish Images",
                    "runs-on": "ubuntu-latest",
                    "permissions": {
                        "contents": "read",
                        "packages": "write"
                    },
                    "outputs": {
                        "image": "${{ steps.image.outputs.image }}",
                        "digest": "${{ steps.build.outputs.digest }}"
                    },
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Docker Buildx",
                            "uses": "docker/setup-buildx-action@v3"
                        },
                        {
                            "name": "Log in to Container Registry",
                            "uses": "docker/login-action@v3",
                            "with": {
                                "registry": "${{ env.REGISTRY }}",
                                "username": "${{ github.actor }}",
                                "password": "${{ secrets.GITHUB_TOKEN }}"
                            }
                        },
                        {
                            "name": "Extract metadata",
                            "id": "meta",
                            "uses": "docker/metadata-action@v5",
                            "with": {
                                "images": "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}",
                                "tags": """
                                    type=ref,event=branch
                                    type=ref,event=pr
                                    type=sha,prefix={{branch}}-
                                    type=raw,value=latest,enable={{is_default_branch}}
                                """
                            }
                        },
                        {
                            "name": "Build and push Docker image",
                            "id": "build",
                            "uses": "docker/build-push-action@v5",
                            "with": {
                                "context": ".",
                                "target": "production",
                                "push": True,
                                "tags": "${{ steps.meta.outputs.tags }}",
                                "labels": "${{ steps.meta.outputs.labels }}",
                                "cache-from": "type=gha",
                                "cache-to": "type=gha,mode=max",
                                "platforms": "linux/amd64,linux/arm64"
                            }
                        },
                        {
                            "name": "Generate SBOM",
                            "uses": "anchore/sbom-action@v0",
                            "with": {
                                "image": "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.meta.outputs.version }}",
                                "format": "spdx-json",
                                "output-file": "sbom.spdx.json"
                            }
                        },
                        {
                            "name": "Upload SBOM",
                            "uses": "actions/upload-artifact@v3",
                            "with": {
                                "name": "sbom",
                                "path": "sbom.spdx.json"
                            }
                        }
                    ]
                },
                "deploy-staging": {
                    "name": "Deploy to Staging",
                    "runs-on": "ubuntu-latest",
                    "needs": "build-and-publish",
                    "if": "github.ref == 'refs/heads/main' || github.event.inputs.environment == 'staging'",
                    "environment": {
                        "name": "staging",
                        "url": "${{ vars.STAGING_URL }}"
                    },
                    "steps": [
                        {
                            "name": "Deploy to staging",
                            "run": """
                                echo "Deploying to staging environment..."
                                # Add actual deployment commands here
                            """
                        },
                        {
                            "name": "Run smoke tests",
                            "run": """
                                curl -f ${{ vars.STAGING_URL }}/health || exit 1
                                echo "Staging deployment successful"
                            """
                        }
                    ]
                },
                "deploy-production": {
                    "name": "Deploy to Production",
                    "runs-on": "ubuntu-latest",
                    "needs": ["build-and-publish", "deploy-staging"],
                    "if": "github.event_name == 'release' || github.event.inputs.environment == 'production'",
                    "environment": {
                        "name": "production",
                        "url": "${{ vars.PRODUCTION_URL }}"
                    },
                    "steps": [
                        {
                            "name": "Deploy to production",
                            "run": """
                                echo "Deploying to production environment..."
                                # Add actual production deployment commands here
                            """
                        },
                        {
                            "name": "Run production health checks",
                            "run": """
                                curl -f ${{ vars.PRODUCTION_URL }}/health || exit 1
                                echo "Production deployment successful"
                            """
                        },
                        {
                            "name": "Notify team",
                            "uses": "8398a7/action-slack@v3",
                            "if": "always()",
                            "with": {
                                "status": "${{ job.status }}",
                                "text": "Production deployment ${{ job.status }}",
                                "webhook_url": "${{ secrets.SLACK_WEBHOOK_URL }}"
                            }
                        }
                    ]
                }
            }
        }
        
        return self._save_workflow(workflow, "cd.yml")
    
    def generate_security_workflow(self) -> str:
        """Generate comprehensive security scanning workflow."""
        workflow = {
            "name": "Security Scan",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]},
                "schedule": [{"cron": "0 2 * * *"}],  # Daily at 2 AM
                "workflow_dispatch": None
            },
            "permissions": {
                "contents": "read",
                "security-events": "write",
                "actions": "read"
            },
            "jobs": {
                "codeql-analysis": {
                    "name": "CodeQL Analysis",
                    "runs-on": "ubuntu-latest",
                    "strategy": {
                        "fail-fast": False,
                        "matrix": {"language": ["python"]}
                    },
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Initialize CodeQL",
                            "uses": "github/codeql-action/init@v2",
                            "with": {"languages": "${{ matrix.language }}"}
                        },
                        {
                            "name": "Perform CodeQL Analysis",
                            "uses": "github/codeql-action/analyze@v2"
                        }
                    ]
                },
                "dependency-scan": {
                    "name": "Dependency Vulnerability Scan",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.11"}
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install safety pip-audit"
                        },
                        {
                            "name": "Run Safety check",
                            "run": "safety check --json --output safety-report.json || true"
                        },
                        {
                            "name": "Run pip-audit",
                            "run": "pip-audit --format=json --output=pip-audit-report.json || true"
                        },
                        {
                            "name": "Upload vulnerability reports",
                            "uses": "actions/upload-artifact@v3",
                            "with": {
                                "name": "vulnerability-reports",
                                "path": "*-report.json"
                            }
                        }
                    ]
                },
                "container-scan": {
                    "name": "Container Security Scan",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Build test image",
                            "run": "docker build -t security-test:latest ."
                        },
                        {
                            "name": "Run Trivy vulnerability scanner",
                            "uses": "aquasecurity/trivy-action@master",
                            "with": {
                                "image-ref": "security-test:latest",
                                "format": "sarif",
                                "output": "trivy-results.sarif"
                            }
                        },
                        {
                            "name": "Upload Trivy scan results",
                            "uses": "github/codeql-action/upload-sarif@v2",
                            "with": {"sarif_file": "trivy-results.sarif"}
                        }
                    ]
                },
                "secrets-scan": {
                    "name": "Secrets Detection",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "TruffleHog OSS",
                            "uses": "trufflesecurity/trufflehog@main",
                            "with": {
                                "path": "./",
                                "base": "${{ github.event.repository.default_branch }}",
                                "head": "HEAD",
                                "extra_args": "--debug --only-verified"
                            }
                        }
                    ]
                }
            }
        }
        
        return self._save_workflow(workflow, "security-scan.yml")
    
    def generate_model_training_workflow(self) -> str:
        """Generate ML model training workflow."""
        workflow = {
            "name": "Model Training",
            "on": {
                "workflow_dispatch": {
                    "inputs": {
                        "dataset_version": {
                            "description": "Dataset version to use",
                            "required": True,
                            "default": "latest"
                        },
                        "model_type": {
                            "description": "Model type to train",
                            "required": True,
                            "default": "transformer",
                            "type": "choice",
                            "options": ["transformer", "cnn", "hybrid"]
                        },
                        "epochs": {
                            "description": "Number of training epochs",
                            "required": False,
                            "default": "50"
                        }
                    }
                },
                "schedule": [{"cron": "0 3 * * 0"}]  # Weekly on Sunday
            },
            "env": {
                "WANDB_PROJECT": "dna-origami-autoencoder",
                "PYTHON_VERSION": "3.11",
                "CUDA_VERSION": "12.1"
            },
            "jobs": {
                "train-model": {
                    "name": "Train ML Model",
                    "runs-on": "self-hosted-gpu",
                    "timeout-minutes": 720,  # 12 hours
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "${{ env.PYTHON_VERSION }}"}
                        },
                        {
                            "name": "Install CUDA and PyTorch",
                            "run": """
                                pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
                                pip install -e .[ml,gpu]
                            """
                        },
                        {
                            "name": "Verify GPU availability",
                            "run": """
                                python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
                                nvidia-smi
                            """
                        },
                        {
                            "name": "Download dataset",
                            "run": """
                                python scripts/download_dataset.py \
                                  --version ${{ github.event.inputs.dataset_version || 'latest' }}
                            """
                        },
                        {
                            "name": "Train model",
                            "env": {
                                "WANDB_API_KEY": "${{ secrets.WANDB_API_KEY }}",
                                "MODEL_TYPE": "${{ github.event.inputs.model_type || 'transformer' }}",
                                "EPOCHS": "${{ github.event.inputs.epochs || '50' }}"
                            },
                            "run": """
                                python scripts/train_model.py \
                                  --model-type $MODEL_TYPE \
                                  --epochs $EPOCHS \
                                  --gpu \
                                  --wandb-logging \
                                  --save-checkpoints
                            """
                        },
                        {
                            "name": "Validate model",
                            "run": """
                                python scripts/validate_model.py \
                                  --model-path models/trained/latest.pth \
                                  --test-set data/test \
                                  --metrics-output validation-metrics.json
                            """
                        },
                        {
                            "name": "Upload model artifacts",
                            "uses": "actions/upload-artifact@v3",
                            "with": {
                                "name": "trained-model-${{ github.run_number }}",
                                "path": """
                                    models/trained/
                                    validation-metrics.json
                                    training-logs/
                                """
                            }
                        },
                        {
                            "name": "Create model release",
                            "if": "success() && github.event_name != 'schedule'",
                            "run": """
                                gh release create model-v${{ github.run_number }} \
                                  models/trained/latest.pth \
                                  --title "Model v${{ github.run_number }}" \
                                  --notes "Trained model artifacts from run ${{ github.run_number }}"
                            """,
                            "env": {"GITHUB_TOKEN": "${{ secrets.GITHUB_TOKEN }}"}
                        }
                    ]
                }
            }
        }
        
        return self._save_workflow(workflow, "model-training.yml")
    
    def generate_dependency_update_workflow(self) -> str:
        """Generate dependency update workflow."""
        workflow = {
            "name": "Dependency Updates",
            "on": {
                "schedule": [{"cron": "0 4 * * 1"}],  # Weekly on Monday
                "workflow_dispatch": None
            },
            "permissions": {
                "contents": "write",
                "pull-requests": "write"
            },
            "jobs": {
                "update-dependencies": {
                    "name": "Update Dependencies",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.11"}
                        },
                        {
                            "name": "Install pip-tools",
                            "run": "pip install pip-tools"
                        },
                        {
                            "name": "Update requirements",
                            "run": """
                                pip-compile --upgrade requirements/base.in
                                pip-compile --upgrade requirements/dev.in
                                pip-compile --upgrade requirements/ml.in
                            """
                        },
                        {
                            "name": "Check for changes",
                            "id": "changes",
                            "run": """
                                if git diff --quiet; then
                                  echo "has_changes=false" >> $GITHUB_OUTPUT
                                else
                                  echo "has_changes=true" >> $GITHUB_OUTPUT
                                fi
                            """
                        },
                        {
                            "name": "Run security audit",
                            "if": "steps.changes.outputs.has_changes == 'true'",
                            "run": """
                                pip install safety
                                safety check --json --output safety-report.json || true
                            """
                        },
                        {
                            "name": "Create Pull Request",
                            "if": "steps.changes.outputs.has_changes == 'true'",
                            "uses": "peter-evans/create-pull-request@v5",
                            "with": {
                                "token": "${{ secrets.GITHUB_TOKEN }}",
                                "commit-message": "chore: update dependencies",
                                "title": "Automated dependency updates",
                                "body": """
                                    This PR contains automated dependency updates.
                                    
                                    - Security audit results attached
                                    - All tests will run automatically
                                    - Review changes before merging
                                """,
                                "branch": "automated/dependency-updates",
                                "labels": "dependencies,automated"
                            }
                        }
                    ]
                }
            }
        }
        
        return self._save_workflow(workflow, "dependency-update.yml")
    
    def _save_workflow(self, workflow: Dict[str, Any], filename: str) -> str:
        """Save workflow to YAML file."""
        filepath = self.workflows_dir / filename
        
        with open(filepath, 'w') as f:
            # Add header comment
            f.write(f"# {workflow['name']}\n")
            f.write("# This is a template file that must be manually copied to .github/workflows/\n")
            f.write("# GitHub App permission limitations prevent automatic workflow creation\n\n")
            
            yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)
        
        return str(filepath)
    
    def generate_all_workflows(self) -> Dict[str, str]:
        """Generate all workflow templates."""
        workflows = {}
        
        print("📝 Generating GitHub Actions workflow templates...")
        
        workflows["ci"] = self.generate_ci_workflow()
        print("  ✅ CI workflow template generated")
        
        workflows["cd"] = self.generate_cd_workflow()
        print("  ✅ CD workflow template generated")
        
        workflows["security"] = self.generate_security_workflow()
        print("  ✅ Security workflow template generated")
        
        workflows["model_training"] = self.generate_model_training_workflow()
        print("  ✅ Model training workflow template generated")
        
        workflows["dependency_update"] = self.generate_dependency_update_workflow()
        print("  ✅ Dependency update workflow template generated")
        
        return workflows
    
    def generate_setup_documentation(self) -> str:
        """Generate comprehensive setup documentation."""
        setup_doc = f"""# GitHub Actions Workflow Setup

## Manual Setup Required

Due to GitHub App permission limitations, workflow files must be manually created by repository maintainers.

## Setup Steps

### 1. Copy Workflow Templates

Copy the following template files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy template files
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/model-training.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### 2. Configure Repository Secrets

Navigate to Repository Settings > Secrets and variables > Actions and add:

**Required Secrets:**
- `CODECOV_TOKEN` - For code coverage reporting
- `DOCKERHUB_USERNAME` - Docker Hub username (if using Docker Hub)
- `DOCKERHUB_TOKEN` - Docker Hub access token
- `WANDB_API_KEY` - Weights & Biases API key (for ML experiments)

**Optional Secrets:**
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `SONAR_TOKEN` - SonarCloud analysis token
- Cloud provider credentials (AWS_ACCESS_KEY_ID, etc.)

### 3. Configure Repository Variables

Add the following variables:

- `STAGING_URL` - Staging environment URL
- `PRODUCTION_URL` - Production environment URL
- `DOCKER_REGISTRY` - Container registry URL

### 4. Set Up Self-Hosted Runners (Optional)

For GPU-intensive workloads, configure self-hosted runners:

1. Go to Repository Settings > Actions > Runners
2. Click "New self-hosted runner"
3. Follow setup instructions
4. Add `self-hosted-gpu` label to GPU-enabled runners

### 5. Configure Branch Protection

Set up branch protection rules for `main` branch:

1. Go to Repository Settings > Branches
2. Add rule for `main` branch
3. Enable:
   - Require pull request reviews
   - Require status checks to pass
   - Require conversation resolution
   - Restrict pushes

### 6. Enable GitHub Security Features

1. Go to Repository Settings > Security & analysis
2. Enable:
   - Dependency graph
   - Dependabot alerts
   - Dependabot security updates
   - Secret scanning
   - Code scanning (CodeQL)

## Workflow Customization

### Environment-Specific Settings

Modify workflow files based on your environment:

- Update Python versions in matrices
- Adjust GPU requirements
- Configure deployment targets
- Set resource limits

### Security Configuration

- Review and adjust security scanning tools
- Configure notification channels
- Set vulnerability thresholds
- Customize compliance checks

## Troubleshooting

### Common Issues

1. **Workflow not triggering**
   - Check branch names in trigger conditions
   - Verify file permissions
   - Review repository settings

2. **GPU tests failing**
   - Ensure self-hosted runners have GPU access
   - Verify CUDA installation
   - Check resource availability

3. **Deployment failures**
   - Validate deployment credentials
   - Check target environment status
   - Review network connectivity

### Getting Help

- Check GitHub Actions logs for detailed error messages
- Review workflow status in the Actions tab
- Consult GitHub Actions documentation
- Contact the development team

## Maintenance

### Regular Tasks

- Review and update workflow templates monthly
- Monitor workflow execution metrics
- Update security scanning tools
- Review and rotate secrets quarterly

### Performance Optimization

- Monitor workflow execution times
- Optimize caching strategies
- Review resource usage
- Update runner configurations

## Security Best Practices

- Use official GitHub Actions when possible
- Pin action versions with commit SHA
- Minimize secret exposure
- Regular security audits of workflows
- Monitor for unauthorized changes

---

**Note**: These workflows are templates and may need customization based on your specific requirements and infrastructure setup.
"""
        
        setup_path = self.workflows_dir.parent / "SETUP_INSTRUCTIONS.md"
        with open(setup_path, 'w') as f:
            f.write(setup_doc)
        
        return str(setup_path)


def main():
    """Main entry point for workflow template generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate GitHub Actions workflow templates"
    )
    
    parser.add_argument("--output-dir", help="Output directory for templates")
    parser.add_argument("--workflow", choices=["ci", "cd", "security", "training", "deps", "all"], 
                       default="all", help="Specific workflow to generate")
    
    args = parser.parse_args()
    
    generator = WorkflowGenerator()
    
    if args.workflow == "all":
        workflows = generator.generate_all_workflows()
        setup_doc = generator.generate_setup_documentation()
        
        print(f"\n✅ Generated {len(workflows)} workflow templates")
        print(f"📄 Setup documentation: {setup_doc}")
        
        print("\n📋 Next steps:")
        print("1. Review generated templates in docs/workflows/examples/")
        print("2. Copy templates to .github/workflows/ manually")
        print("3. Configure repository secrets and variables")
        print("4. Set up branch protection rules")
        print("5. Test workflows with a pull request")
        
    else:
        # Generate specific workflow
        workflow_methods = {
            "ci": generator.generate_ci_workflow,
            "cd": generator.generate_cd_workflow,
            "security": generator.generate_security_workflow,
            "training": generator.generate_model_training_workflow,
            "deps": generator.generate_dependency_update_workflow
        }
        
        if args.workflow in workflow_methods:
            workflow_path = workflow_methods[args.workflow]()
            print(f"✅ Generated {args.workflow} workflow: {workflow_path}")


if __name__ == "__main__":
    main()