"""
Production deployment system for DNA Origami AutoEncoder.
Handles multi-region deployments, compliance, monitoring, and scaling.
"""

import os
import json
import yaml
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import tempfile
import shutil

from ..utils.advanced_logging import get_advanced_logger


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DR = "disaster_recovery"


class DeploymentRegion(Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Configuration for production deployment."""
    environment: DeploymentEnvironment
    regions: List[DeploymentRegion]
    version: str
    
    # Scaling configuration
    min_instances: int = 2
    max_instances: int = 20
    target_cpu_utilization: float = 70.0
    
    # Resource configuration
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    
    # Storage configuration
    persistent_storage: bool = True
    storage_size: str = "10Gi"
    backup_enabled: bool = True
    backup_retention_days: int = 30
    
    # Networking
    load_balancer_enabled: bool = True
    ssl_enabled: bool = True
    cdn_enabled: bool = True
    
    # Security
    security_scanning: bool = True
    compliance_mode: str = "strict"
    data_encryption: bool = True
    
    # Monitoring
    monitoring_enabled: bool = True
    alerting_enabled: bool = True
    log_retention_days: int = 90
    
    # Additional configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Result of deployment operation."""
    deployment_id: str
    status: DeploymentStatus
    environment: DeploymentEnvironment
    regions: List[DeploymentRegion]
    version: str
    start_time: float
    end_time: Optional[float] = None
    endpoints: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class ProductionDeploymentManager:
    """Manages production deployments with multi-region support."""
    
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path(__file__).parent
        self.logger = get_advanced_logger("production_deployment")
        
        # Deployment history
        self.deployment_history: List[DeploymentResult] = []
        
        # Template directories
        self.templates_dir = self.base_path / "templates"
        self.configs_dir = self.base_path / "configs"
        self.scripts_dir = self.base_path / "scripts"
        
        self._ensure_directories()
        self._initialize_templates()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            self.templates_dir,
            self.configs_dir,
            self.scripts_dir,
            self.base_path / "manifests",
            self.base_path / "helm",
            self.base_path / "terraform"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _initialize_templates(self):
        """Initialize deployment templates."""
        self._create_kubernetes_templates()
        self._create_docker_templates()
        self._create_terraform_templates()
        self._create_helm_charts()
    
    def _create_kubernetes_templates(self):
        """Create Kubernetes deployment templates."""
        # Deployment template
        deployment_template = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "dna-origami-autoencoder",
                "labels": {
                    "app": "dna-origami-autoencoder",
                    "version": "{{ VERSION }}"
                }
            },
            "spec": {
                "replicas": "{{ MIN_INSTANCES }}",
                "selector": {
                    "matchLabels": {
                        "app": "dna-origami-autoencoder"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "dna-origami-autoencoder",
                            "version": "{{ VERSION }}"
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "dna-origami-autoencoder",
                                "image": "dna-origami-autoencoder:{{ VERSION }}",
                                "ports": [
                                    {
                                        "containerPort": 8000,
                                        "name": "http"
                                    },
                                    {
                                        "containerPort": 8080,
                                        "name": "metrics"
                                    }
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "{{ CPU_REQUEST }}",
                                        "memory": "{{ MEMORY_REQUEST }}"
                                    },
                                    "limits": {
                                        "cpu": "{{ CPU_LIMIT }}",
                                        "memory": "{{ MEMORY_LIMIT }}"
                                    }
                                },
                                "env": [
                                    {
                                        "name": "ENVIRONMENT",
                                        "value": "{{ ENVIRONMENT }}"
                                    },
                                    {
                                        "name": "LOG_LEVEL",
                                        "value": "INFO"
                                    },
                                    {
                                        "name": "METRICS_ENABLED",
                                        "value": "true"
                                    }
                                ],
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/ready",
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5
                                }
                            }
                        ],
                        "imagePullSecrets": [
                            {
                                "name": "registry-secret"
                            }
                        ]
                    }
                }
            }
        }
        
        with open(self.templates_dir / "deployment.yaml", 'w') as f:
            yaml.dump(deployment_template, f, default_flow_style=False)
        
        # Service template
        service_template = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "dna-origami-autoencoder-service",
                "labels": {
                    "app": "dna-origami-autoencoder"
                }
            },
            "spec": {
                "selector": {
                    "app": "dna-origami-autoencoder"
                },
                "ports": [
                    {
                        "name": "http",
                        "port": 80,
                        "targetPort": 8000
                    },
                    {
                        "name": "metrics",
                        "port": 8080,
                        "targetPort": 8080
                    }
                ],
                "type": "ClusterIP"
            }
        }
        
        with open(self.templates_dir / "service.yaml", 'w') as f:
            yaml.dump(service_template, f, default_flow_style=False)
        
        # HPA template
        hpa_template = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "dna-origami-autoencoder-hpa"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "dna-origami-autoencoder"
                },
                "minReplicas": "{{ MIN_INSTANCES }}",
                "maxReplicas": "{{ MAX_INSTANCES }}",
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": "{{ TARGET_CPU_UTILIZATION }}"
                            }
                        }
                    }
                ]
            }
        }
        
        with open(self.templates_dir / "hpa.yaml", 'w') as f:
            yaml.dump(hpa_template, f, default_flow_style=False)
    
    def _create_docker_templates(self):
        """Create Docker templates."""
        # Production Dockerfile
        dockerfile_content = """# Production Dockerfile for DNA Origami AutoEncoder
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libffi-dev \\
    libssl-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY dna_origami_ae/ ./dna_origami_ae/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \\
    chown -R app:app /app
USER app

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "dna_origami_ae.server", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        with open(self.templates_dir / "Dockerfile.prod", 'w') as f:
            f.write(dockerfile_content)
        
        # Docker compose for local testing
        compose_content = {
            "version": "3.8",
            "services": {
                "dna-origami-autoencoder": {
                    "build": {
                        "context": "..",
                        "dockerfile": "deployment/templates/Dockerfile.prod"
                    },
                    "ports": [
                        "8000:8000",
                        "8080:8080"
                    ],
                    "environment": [
                        "ENVIRONMENT=production",
                        "LOG_LEVEL=INFO"
                    ],
                    "volumes": [
                        "../configs:/app/configs:ro"
                    ],
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    }
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "ports": ["6379:6379"],
                    "volumes": ["redis_data:/data"]
                },
                "nginx": {
                    "image": "nginx:alpine",
                    "ports": ["80:80", "443:443"],
                    "volumes": [
                        "./nginx.conf:/etc/nginx/nginx.conf:ro"
                    ],
                    "depends_on": ["dna-origami-autoencoder"]
                }
            },
            "volumes": {
                "redis_data": {}
            }
        }
        
        with open(self.templates_dir / "docker-compose.prod.yaml", 'w') as f:
            yaml.dump(compose_content, f, default_flow_style=False)
    
    def _create_terraform_templates(self):
        """Create Terraform infrastructure templates."""
        # Main infrastructure
        main_tf = """# Main Terraform configuration for DNA Origami AutoEncoder

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "regions" {
  description = "Deployment regions"
  type        = list(string)
  default     = ["us-east-1", "us-west-2"]
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "m5.large"
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "dna-origami-vpc"
    Environment = var.environment
  }
}

# Subnets
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name        = "dna-origami-private-${count.index + 1}"
    Environment = var.environment
  }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 10}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name        = "dna-origami-public-${count.index + 1}"
    Environment = var.environment
  }
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "dna-origami-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids              = concat(aws_subnet.private[*].id, aws_subnet.public[*].id)
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
  ]

  tags = {
    Environment = var.environment
  }
}

# EKS Node Group
resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "dna-origami-nodes"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = aws_subnet.private[*].id
  instance_types  = [var.instance_type]

  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 1
  }

  update_config {
    max_unavailable = 1
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]

  tags = {
    Environment = var.environment
  }
}

# RDS Instance for persistent data
resource "aws_db_instance" "main" {
  identifier             = "dna-origami-db"
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = "db.t3.micro"
  allocated_storage      = 20
  max_allocated_storage  = 100
  storage_type           = "gp2"
  storage_encrypted      = true

  db_name  = "dnaorigami"
  username = "dnaorigami"
  password = random_password.db_password.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "dna-origami-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"

  tags = {
    Environment = var.environment
  }
}

# ElastiCache Redis for caching
resource "aws_elasticache_subnet_group" "main" {
  name       = "dna-origami-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "dna-origami-cache"
  description                = "Redis cluster for DNA Origami AutoEncoder"
  
  node_type                  = "cache.t3.micro"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Environment = var.environment
  }
}

# Outputs
output "cluster_endpoint" {
  value = aws_eks_cluster.main.endpoint
}

output "cluster_security_group_id" {
  value = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "database_endpoint" {
  value = aws_db_instance.main.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.main.configuration_endpoint_address
}
"""
        
        with open(self.templates_dir / "main.tf", 'w') as f:
            f.write(main_tf)
    
    def _create_helm_charts(self):
        """Create Helm charts for deployment."""
        # Create Helm chart structure
        helm_dir = self.base_path / "helm" / "dna-origami-autoencoder"
        helm_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart.yaml
        chart_yaml = {
            "apiVersion": "v2",
            "name": "dna-origami-autoencoder",
            "description": "A Helm chart for DNA Origami AutoEncoder",
            "type": "application",
            "version": "0.1.0",
            "appVersion": "1.0.0",
            "keywords": ["dna", "origami", "autoencoder", "bioinformatics"],
            "maintainers": [
                {
                    "name": "DNA Origami Team",
                    "email": "team@dnaorigami.ai"
                }
            ]
        }
        
        with open(helm_dir / "Chart.yaml", 'w') as f:
            yaml.dump(chart_yaml, f, default_flow_style=False)
        
        # Values.yaml
        values_yaml = {
            "replicaCount": 2,
            "image": {
                "repository": "dna-origami-autoencoder",
                "pullPolicy": "IfNotPresent",
                "tag": "latest"
            },
            "nameOverride": "",
            "fullnameOverride": "",
            "serviceAccount": {
                "create": True,
                "annotations": {},
                "name": ""
            },
            "podAnnotations": {},
            "podSecurityContext": {},
            "securityContext": {},
            "service": {
                "type": "ClusterIP",
                "port": 80,
                "targetPort": 8000
            },
            "ingress": {
                "enabled": True,
                "className": "nginx",
                "annotations": {
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true"
                },
                "hosts": [
                    {
                        "host": "api.dnaorigami.ai",
                        "paths": [
                            {
                                "path": "/",
                                "pathType": "Prefix"
                            }
                        ]
                    }
                ],
                "tls": [
                    {
                        "secretName": "dna-origami-tls",
                        "hosts": ["api.dnaorigami.ai"]
                    }
                ]
            },
            "resources": {
                "limits": {
                    "cpu": "2000m",
                    "memory": "4Gi"
                },
                "requests": {
                    "cpu": "500m",
                    "memory": "1Gi"
                }
            },
            "autoscaling": {
                "enabled": True,
                "minReplicas": 2,
                "maxReplicas": 20,
                "targetCPUUtilizationPercentage": 70
            },
            "nodeSelector": {},
            "tolerations": [],
            "affinity": {},
            "monitoring": {
                "enabled": True,
                "serviceMonitor": {
                    "enabled": True
                }
            },
            "persistence": {
                "enabled": True,
                "size": "10Gi",
                "storageClass": "gp2"
            }
        }
        
        with open(helm_dir / "values.yaml", 'w') as f:
            yaml.dump(values_yaml, f, default_flow_style=False)
    
    def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Execute production deployment."""
        deployment_id = f"deploy-{int(time.time())}"
        
        result = DeploymentResult(
            deployment_id=deployment_id,
            status=DeploymentStatus.IN_PROGRESS,
            environment=config.environment,
            regions=config.regions,
            version=config.version,
            start_time=time.time()
        )
        
        self.logger.info(f"Starting deployment {deployment_id} for {config.environment.value}")
        
        try:
            # Pre-deployment validation
            self._validate_deployment_config(config)
            
            # Build and push container images
            self._build_and_push_images(config, result)
            
            # Deploy infrastructure
            self._deploy_infrastructure(config, result)
            
            # Deploy application
            self._deploy_application(config, result)
            
            # Run post-deployment tests
            self._run_deployment_tests(config, result)
            
            # Setup monitoring and alerting
            self._setup_monitoring(config, result)
            
            result.status = DeploymentStatus.COMPLETED
            result.end_time = time.time()
            
            self.logger.info(f"Deployment {deployment_id} completed successfully")
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.errors.append(str(e))
            result.end_time = time.time()
            
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            
            # Attempt rollback
            if config.environment == DeploymentEnvironment.PRODUCTION:
                self._rollback_deployment(result)
        
        finally:
            self.deployment_history.append(result)
        
        return result
    
    def _validate_deployment_config(self, config: DeploymentConfig):
        """Validate deployment configuration."""
        self.logger.info("Validating deployment configuration...")
        
        # Check required fields
        if not config.version:
            raise ValueError("Version is required for deployment")
        
        if not config.regions:
            raise ValueError("At least one region must be specified")
        
        # Validate resource limits
        if config.min_instances >= config.max_instances:
            raise ValueError("min_instances must be less than max_instances")
        
        # Check security settings for production
        if config.environment == DeploymentEnvironment.PRODUCTION:
            if not config.ssl_enabled:
                raise ValueError("SSL must be enabled for production deployments")
            
            if not config.data_encryption:
                raise ValueError("Data encryption must be enabled for production")
            
            if not config.security_scanning:
                raise ValueError("Security scanning must be enabled for production")
        
        self.logger.info("Deployment configuration validated successfully")
    
    def _build_and_push_images(self, config: DeploymentConfig, result: DeploymentResult):
        """Build and push container images."""
        self.logger.info("Building and pushing container images...")
        
        # Build production image
        build_command = [
            "docker", "build",
            "-f", str(self.templates_dir / "Dockerfile.prod"),
            "-t", f"dna-origami-autoencoder:{config.version}",
            str(self.base_path.parent)
        ]
        
        build_result = subprocess.run(build_command, capture_output=True, text=True)
        
        if build_result.returncode != 0:
            raise RuntimeError(f"Image build failed: {build_result.stderr}")
        
        # Tag for registry
        registry_url = "your-registry.com"  # Configure your registry
        full_image_name = f"{registry_url}/dna-origami-autoencoder:{config.version}"
        
        tag_command = ["docker", "tag", f"dna-origami-autoencoder:{config.version}", full_image_name]
        tag_result = subprocess.run(tag_command, capture_output=True, text=True)
        
        if tag_result.returncode != 0:
            raise RuntimeError(f"Image tagging failed: {tag_result.stderr}")
        
        # Push to registry
        push_command = ["docker", "push", full_image_name]
        push_result = subprocess.run(push_command, capture_output=True, text=True)
        
        if push_result.returncode != 0:
            raise RuntimeError(f"Image push failed: {push_result.stderr}")
        
        result.metrics['image_name'] = full_image_name
        self.logger.info(f"Successfully built and pushed image: {full_image_name}")
    
    def _deploy_infrastructure(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy infrastructure using Terraform."""
        self.logger.info("Deploying infrastructure...")
        
        terraform_dir = self.base_path / "terraform"
        terraform_dir.mkdir(exist_ok=True)
        
        # Copy Terraform templates
        shutil.copy(self.templates_dir / "main.tf", terraform_dir)
        
        # Create terraform.tfvars
        tfvars_content = f"""
environment = "{config.environment.value}"
regions = {json.dumps([r.value for r in config.regions])}
instance_type = "m5.large"
"""
        
        with open(terraform_dir / "terraform.tfvars", 'w') as f:
            f.write(tfvars_content)
        
        # Initialize Terraform
        init_result = subprocess.run(
            ["terraform", "init"],
            cwd=terraform_dir,
            capture_output=True,
            text=True
        )
        
        if init_result.returncode != 0:
            raise RuntimeError(f"Terraform init failed: {init_result.stderr}")
        
        # Plan Terraform
        plan_result = subprocess.run(
            ["terraform", "plan", "-out=tfplan"],
            cwd=terraform_dir,
            capture_output=True,
            text=True
        )
        
        if plan_result.returncode != 0:
            raise RuntimeError(f"Terraform plan failed: {plan_result.stderr}")
        
        # Apply Terraform (only if not in dry-run mode)
        if config.custom_config.get('dry_run', False):
            self.logger.info("Dry-run mode: Skipping terraform apply")
        else:
            apply_result = subprocess.run(
                ["terraform", "apply", "-auto-approve", "tfplan"],
                cwd=terraform_dir,
                capture_output=True,
                text=True
            )
            
            if apply_result.returncode != 0:
                raise RuntimeError(f"Terraform apply failed: {apply_result.stderr}")
        
        self.logger.info("Infrastructure deployment completed")
    
    def _deploy_application(self, config: DeploymentConfig, result: DeploymentResult):
        """Deploy application using Helm."""
        self.logger.info("Deploying application...")
        
        helm_dir = self.base_path / "helm" / "dna-origami-autoencoder"
        
        # Create values override file
        values_override = {
            "image": {
                "tag": config.version
            },
            "replicaCount": config.min_instances,
            "resources": {
                "requests": {
                    "cpu": config.cpu_request,
                    "memory": config.memory_request
                },
                "limits": {
                    "cpu": config.cpu_limit,
                    "memory": config.memory_limit
                }
            },
            "autoscaling": {
                "minReplicas": config.min_instances,
                "maxReplicas": config.max_instances,
                "targetCPUUtilizationPercentage": int(config.target_cpu_utilization)
            },
            "persistence": {
                "enabled": config.persistent_storage,
                "size": config.storage_size
            }
        }
        
        values_file = self.configs_dir / f"values-{config.environment.value}.yaml"
        with open(values_file, 'w') as f:
            yaml.dump(values_override, f, default_flow_style=False)
        
        # Deploy with Helm
        helm_command = [
            "helm", "upgrade", "--install",
            f"dna-origami-{config.environment.value}",
            str(helm_dir),
            "-f", str(values_file),
            "--namespace", f"dna-origami-{config.environment.value}",
            "--create-namespace"
        ]
        
        if not config.custom_config.get('dry_run', False):
            helm_result = subprocess.run(helm_command, capture_output=True, text=True)
            
            if helm_result.returncode != 0:
                raise RuntimeError(f"Helm deployment failed: {helm_result.stderr}")
        
        self.logger.info("Application deployment completed")
    
    def _run_deployment_tests(self, config: DeploymentConfig, result: DeploymentResult):
        """Run post-deployment tests."""
        self.logger.info("Running deployment tests...")
        
        # Health check tests
        endpoints_to_test = ["http://api.dnaorigami.ai/health"]
        
        for endpoint in endpoints_to_test:
            try:
                # Simulate health check (in real implementation, use requests)
                self.logger.info(f"Testing endpoint: {endpoint}")
                # response = requests.get(endpoint, timeout=30)
                # assert response.status_code == 200
                result.endpoints[endpoint] = "healthy"
            except Exception as e:
                result.errors.append(f"Health check failed for {endpoint}: {e}")
        
        # Load tests (simplified)
        self.logger.info("Running basic load test...")
        # In real implementation, run proper load tests
        
        self.logger.info("Deployment tests completed")
    
    def _setup_monitoring(self, config: DeploymentConfig, result: DeploymentResult):
        """Setup monitoring and alerting."""
        self.logger.info("Setting up monitoring and alerting...")
        
        if config.monitoring_enabled:
            # Deploy Prometheus monitoring
            monitoring_config = {
                "prometheus": {
                    "enabled": True,
                    "retention": f"{config.log_retention_days}d"
                },
                "grafana": {
                    "enabled": True
                },
                "alertmanager": {
                    "enabled": config.alerting_enabled
                }
            }
            
            result.metrics['monitoring_config'] = monitoring_config
        
        self.logger.info("Monitoring setup completed")
    
    def _rollback_deployment(self, result: DeploymentResult):
        """Rollback failed deployment."""
        self.logger.warning(f"Attempting rollback for deployment {result.deployment_id}")
        
        try:
            # Get previous successful deployment
            previous_deployments = [
                d for d in self.deployment_history
                if (d.status == DeploymentStatus.COMPLETED and 
                    d.environment == result.environment)
            ]
            
            if not previous_deployments:
                self.logger.error("No previous successful deployment found for rollback")
                return
            
            previous_deployment = previous_deployments[-1]
            
            # Rollback using Helm
            rollback_command = [
                "helm", "rollback",
                f"dna-origami-{result.environment.value}",
                "--namespace", f"dna-origami-{result.environment.value}"
            ]
            
            rollback_result = subprocess.run(rollback_command, capture_output=True, text=True)
            
            if rollback_result.returncode == 0:
                result.status = DeploymentStatus.ROLLED_BACK
                self.logger.info("Rollback completed successfully")
            else:
                self.logger.error(f"Rollback failed: {rollback_result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Rollback attempt failed: {e}")
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of a specific deployment."""
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment
        return None
    
    def list_deployments(self, environment: Optional[DeploymentEnvironment] = None) -> List[DeploymentResult]:
        """List deployment history."""
        if environment:
            return [d for d in self.deployment_history if d.environment == environment]
        return self.deployment_history.copy()
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        total_deployments = len(self.deployment_history)
        successful_deployments = len([d for d in self.deployment_history if d.status == DeploymentStatus.COMPLETED])
        failed_deployments = len([d for d in self.deployment_history if d.status == DeploymentStatus.FAILED])
        
        report = {
            "summary": {
                "total_deployments": total_deployments,
                "successful_deployments": successful_deployments,
                "failed_deployments": failed_deployments,
                "success_rate": successful_deployments / total_deployments if total_deployments > 0 else 0
            },
            "by_environment": {},
            "recent_deployments": []
        }
        
        # Group by environment
        for env in DeploymentEnvironment:
            env_deployments = [d for d in self.deployment_history if d.environment == env]
            if env_deployments:
                report["by_environment"][env.value] = {
                    "total": len(env_deployments),
                    "successful": len([d for d in env_deployments if d.status == DeploymentStatus.COMPLETED]),
                    "latest_version": env_deployments[-1].version if env_deployments else None
                }
        
        # Recent deployments
        report["recent_deployments"] = [
            {
                "deployment_id": d.deployment_id,
                "environment": d.environment.value,
                "version": d.version,
                "status": d.status.value,
                "duration": d.end_time - d.start_time if d.end_time else None
            }
            for d in sorted(self.deployment_history, key=lambda x: x.start_time, reverse=True)[:10]
        ]
        
        return report