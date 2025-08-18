#!/usr/bin/env python3
"""
Observability stack management for DNA-Origami-AutoEncoder.

This module provides comprehensive observability setup including metrics,
logging, tracing, and monitoring dashboard management.
"""

import os
import sys
import subprocess
import json
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class ObservabilityComponent:
    """Configuration for an observability component."""
    
    name: str
    service_type: str  # prometheus, grafana, jaeger, loki, etc.
    enabled: bool = True
    port: Optional[int] = None
    health_check_url: Optional[str] = None
    dependencies: List[str] = None
    configuration: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.configuration is None:
            self.configuration = {}

class ObservabilityStack:
    """Comprehensive observability stack management."""
    
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.monitoring_dir = project_root / "monitoring"
        self.components = self._initialize_components()
        
    def _initialize_components(self) -> Dict[str, ObservabilityComponent]:
        """Initialize observability components configuration."""
        return {
            "prometheus": ObservabilityComponent(
                name="prometheus",
                service_type="metrics",
                port=9090,
                health_check_url="http://localhost:9090/-/healthy",
                configuration={
                    "retention_time": "15d",
                    "scrape_interval": "15s",
                    "evaluation_interval": "15s"
                }
            ),
            "grafana": ObservabilityComponent(
                name="grafana",
                service_type="visualization",
                port=3000,
                health_check_url="http://localhost:3000/api/health",
                dependencies=["prometheus"],
                configuration={
                    "admin_password": "admin123",
                    "anonymous_access": False,
                    "plugins": [
                        "grafana-piechart-panel",
                        "grafana-worldmap-panel",
                        "grafana-clock-panel"
                    ]
                }
            ),
            "jaeger": ObservabilityComponent(
                name="jaeger",
                service_type="tracing",
                port=16686,
                health_check_url="http://localhost:16686/api/health",
                configuration={
                    "sampling_strategy": "probabilistic",
                    "sampling_rate": 0.1,
                    "storage_type": "elasticsearch"
                }
            ),
            "loki": ObservabilityComponent(
                name="loki",
                service_type="logging",
                port=3100,
                health_check_url="http://localhost:3100/ready",
                configuration={
                    "retention_period": "30d",
                    "chunk_store": "filesystem",
                    "max_query_length": "12000h"
                }
            ),
            "alertmanager": ObservabilityComponent(
                name="alertmanager",
                service_type="alerting",
                port=9093,
                health_check_url="http://localhost:9093/-/healthy",
                dependencies=["prometheus"],
                configuration={
                    "smtp_smarthost": "localhost:587",
                    "smtp_from": "alerts@dna-origami-ae.local",
                    "webhook_url": "http://slack-webhook:8080/hooks"
                }
            ),
            "node_exporter": ObservabilityComponent(
                name="node_exporter",
                service_type="system_metrics",
                port=9100,
                health_check_url="http://localhost:9100/metrics",
                configuration={
                    "collectors": [
                        "cpu", "diskstats", "filesystem", "loadavg",
                        "meminfo", "netdev", "netstat", "stat", "time"
                    ]
                }
            ),
            "cadvisor": ObservabilityComponent(
                name="cadvisor",
                service_type="container_metrics",
                port=8080,
                health_check_url="http://localhost:8080/healthz",
                configuration={
                    "storage_duration": "2m",
                    "housekeeping_interval": "10s"
                }
            )
        }
    
    def generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration file."""
        config = {
            "global": {
                "scrape_interval": self.components["prometheus"].configuration["scrape_interval"],
                "evaluation_interval": self.components["prometheus"].configuration["evaluation_interval"],
                "external_labels": {
                    "monitor": "dna-origami-ae-monitor",
                    "environment": os.getenv("DNA_ORIGAMI_AE_ENV", "development")
                }
            },
            "rule_files": [
                "alert_rules.yml",
                "recording_rules.yml"
            ],
            "alerting": {
                "alertmanagers": [
                    {
                        "static_configs": [
                            {"targets": ["alertmanager:9093"]}
                        ]
                    }
                ]
            },
            "scrape_configs": self._generate_scrape_configs()
        }
        
        config_path = self.monitoring_dir / "prometheus" / "prometheus_generated.yml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(config_path)
    
    def _generate_scrape_configs(self) -> List[Dict[str, Any]]:
        """Generate Prometheus scrape configurations."""
        scrape_configs = [
            {
                "job_name": "prometheus",
                "static_configs": [{"targets": ["localhost:9090"]}],
                "scrape_interval": "30s"
            },
            {
                "job_name": "dna-origami-ae-app",
                "static_configs": [{"targets": ["dna-origami-ae:8080"]}],
                "scrape_interval": "15s",
                "metrics_path": "/metrics"
            },
            {
                "job_name": "node-exporter",
                "static_configs": [{"targets": ["node-exporter:9100"]}],
                "scrape_interval": "15s"
            },
            {
                "job_name": "cadvisor",
                "static_configs": [{"targets": ["cadvisor:8080"]}],
                "scrape_interval": "15s"
            }
        ]
        
        # Add GPU metrics if available
        if self._gpu_available():
            scrape_configs.append({
                "job_name": "nvidia-gpu",
                "static_configs": [{"targets": ["gpu-exporter:9445"]}],
                "scrape_interval": "15s"
            })
        
        # Add application-specific metrics
        app_metrics = [
            ("dna-encoding", "/metrics/encoding"),
            ("simulation", "/metrics/simulation"),
            ("ml-training", "/metrics/training"),
            ("origami-design", "/metrics/design")
        ]
        
        for job_name, metrics_path in app_metrics:
            scrape_configs.append({
                "job_name": job_name,
                "static_configs": [{"targets": ["dna-origami-ae:8080"]}],
                "scrape_interval": "15s",
                "metrics_path": metrics_path
            })
        
        return scrape_configs
    
    def generate_grafana_dashboards(self) -> List[str]:
        """Generate Grafana dashboard configurations."""
        dashboards = []
        
        # Main application dashboard
        app_dashboard = self._create_application_dashboard()
        app_path = self.monitoring_dir / "grafana" / "dashboards" / "dna-origami-ae-app.json"
        app_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(app_path, 'w') as f:
            json.dump(app_dashboard, f, indent=2)
        dashboards.append(str(app_path))
        
        # Scientific computing dashboard
        science_dashboard = self._create_scientific_dashboard()
        science_path = self.monitoring_dir / "grafana" / "dashboards" / "scientific-computing.json"
        
        with open(science_path, 'w') as f:
            json.dump(science_dashboard, f, indent=2)
        dashboards.append(str(science_path))
        
        # Infrastructure dashboard
        infra_dashboard = self._create_infrastructure_dashboard()
        infra_path = self.monitoring_dir / "grafana" / "dashboards" / "infrastructure.json"
        
        with open(infra_path, 'w') as f:
            json.dump(infra_dashboard, f, indent=2)
        dashboards.append(str(infra_path))
        
        return dashboards
    
    def _create_application_dashboard(self) -> Dict[str, Any]:
        """Create main application dashboard configuration."""
        return {
            "dashboard": {
                "id": None,
                "title": "DNA-Origami-AutoEncoder Application",
                "description": "Main application metrics and health monitoring",
                "tags": ["dna-origami-ae", "application"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Application Status",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "up{job='dna-origami-ae-app'}",
                                "refId": "A"
                            }
                        ],
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "thresholds"},
                                "thresholds": {
                                    "steps": [
                                        {"color": "red", "value": 0},
                                        {"color": "green", "value": 1}
                                    ]
                                }
                            }
                        },
                        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{job='dna-origami-ae-app'}[5m])",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 6, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job='dna-origami-ae-app'}[5m]))",
                                "refId": "A",
                                "legendFormat": "95th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{job='dna-origami-ae-app'}[5m]))",
                                "refId": "B",
                                "legendFormat": "50th percentile"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            }
        }
    
    def _create_scientific_dashboard(self) -> Dict[str, Any]:
        """Create scientific computing dashboard configuration."""
        return {
            "dashboard": {
                "id": None,
                "title": "Scientific Computing Metrics",
                "description": "DNA encoding, simulation, and ML training metrics",
                "tags": ["dna-origami-ae", "scientific"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "DNA Encoding Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(dna_sequences_encoded_total[5m])",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Simulation Success Rate",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "rate(simulations_successful_total[5m]) / rate(simulations_total[5m]) * 100",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "GPU Utilization",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "nvidia_gpu_utilization_gpu",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 0}
                    }
                ],
                "time": {"from": "now-6h", "to": "now"},
                "refresh": "1m"
            }
        }
    
    def _create_infrastructure_dashboard(self) -> Dict[str, Any]:
        """Create infrastructure monitoring dashboard."""
        return {
            "dashboard": {
                "id": None,
                "title": "Infrastructure Monitoring",
                "description": "System resources, containers, and infrastructure health",
                "tags": ["dna-origami-ae", "infrastructure"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "CPU Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            }
        }
    
    def setup_logging_stack(self) -> Dict[str, str]:
        """Setup comprehensive logging stack with Loki and Promtail."""
        logging_configs = {}
        
        # Loki configuration
        loki_config = {
            "auth_enabled": False,
            "server": {
                "http_listen_port": 3100
            },
            "ingester": {
                "lifecycler": {
                    "address": "127.0.0.1",
                    "ring": {
                        "kvstore": {"store": "inmemory"},
                        "replication_factor": 1
                    }
                }
            },
            "schema_config": {
                "configs": [
                    {
                        "from": "2020-05-15",
                        "store": "boltdb",
                        "object_store": "filesystem",
                        "schema": "v11",
                        "index": {
                            "prefix": "index_",
                            "period": "168h"
                        }
                    }
                ]
            },
            "storage_config": {
                "boltdb": {"directory": "/loki/index"},
                "filesystem": {"directory": "/loki/chunks"}
            },
            "limits_config": {
                "enforce_metric_name": False,
                "reject_old_samples": True,
                "reject_old_samples_max_age": "168h"
            }
        }
        
        loki_path = self.monitoring_dir / "loki" / "loki.yml"
        loki_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(loki_path, 'w') as f:
            yaml.dump(loki_config, f, default_flow_style=False)
        
        logging_configs["loki"] = str(loki_path)
        
        # Promtail configuration
        promtail_config = {
            "server": {
                "http_listen_port": 9080,
                "grpc_listen_port": 0
            },
            "positions": {
                "filename": "/tmp/positions.yaml"
            },
            "clients": [
                {"url": "http://loki:3100/loki/api/v1/push"}
            ],
            "scrape_configs": [
                {
                    "job_name": "dna-origami-ae-logs",
                    "static_configs": [
                        {
                            "targets": ["localhost"],
                            "labels": {
                                "job": "dna-origami-ae",
                                "__path__": "/app/logs/*.log"
                            }
                        }
                    ]
                },
                {
                    "job_name": "docker-logs",
                    "docker_sd_configs": [
                        {"host": "unix:///var/run/docker.sock"}
                    ],
                    "relabel_configs": [
                        {
                            "source_labels": ["__meta_docker_container_name"],
                            "target_label": "container"
                        }
                    ]
                }
            ]
        }
        
        promtail_path = self.monitoring_dir / "promtail" / "promtail.yml"
        promtail_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(promtail_path, 'w') as f:
            yaml.dump(promtail_config, f, default_flow_style=False)
        
        logging_configs["promtail"] = str(promtail_path)
        
        return logging_configs
    
    def deploy_observability_stack(self, environment: str = "development") -> bool:
        """Deploy the complete observability stack."""
        print("🔍 Deploying observability stack...")
        
        try:
            # Generate configurations
            self.generate_prometheus_config()
            self.generate_grafana_dashboards()
            self.setup_logging_stack()
            
            # Deploy using docker-compose
            cmd = [
                "docker-compose",
                "-f", "docker-compose.yml",
                "--profile", "monitoring",
                "up", "-d"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✅ Observability stack deployed successfully")
                self._wait_for_services()
                return True
            else:
                print(f"❌ Failed to deploy observability stack: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error deploying observability stack: {e}")
            return False
    
    def _wait_for_services(self, timeout: int = 300):
        """Wait for observability services to become healthy."""
        print("⏳ Waiting for services to become healthy...")
        
        start_time = time.time()
        
        for name, component in self.components.items():
            if not component.enabled or not component.health_check_url:
                continue
            
            print(f"  Checking {name}...")
            
            while time.time() - start_time < timeout:
                try:
                    result = subprocess.run(
                        ["curl", "-f", component.health_check_url],
                        capture_output=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        print(f"  ✅ {name} is healthy")
                        break
                        
                except subprocess.TimeoutExpired:
                    pass
                
                time.sleep(5)
            else:
                print(f"  ⚠️ {name} health check timed out")
    
    def _gpu_available(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def generate_observability_report(self) -> Dict[str, Any]:
        """Generate comprehensive observability setup report."""
        report = {
            "observability_stack": {
                "components": {
                    name: {
                        "enabled": comp.enabled,
                        "service_type": comp.service_type,
                        "port": comp.port,
                        "dependencies": comp.dependencies
                    }
                    for name, comp in self.components.items()
                },
                "configurations_generated": [
                    "prometheus.yml",
                    "alert_rules.yml", 
                    "grafana_dashboards",
                    "loki.yml",
                    "promtail.yml"
                ]
            },
            "monitoring_capabilities": [
                "Application metrics collection",
                "Infrastructure monitoring",
                "GPU utilization tracking",
                "Scientific computing metrics",
                "Distributed tracing",
                "Centralized logging",
                "Real-time alerting",
                "Performance dashboards"
            ],
            "deployment_status": "configured",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return report


def main():
    """Main entry point for observability stack management."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DNA-Origami-AutoEncoder Observability Stack"
    )
    
    parser.add_argument("--generate-configs", action="store_true", help="Generate all configurations")
    parser.add_argument("--deploy", action="store_true", help="Deploy observability stack")
    parser.add_argument("--environment", default="development", help="Target environment")
    
    args = parser.parse_args()
    
    stack = ObservabilityStack()
    
    if args.generate_configs:
        print("📝 Generating observability configurations...")
        stack.generate_prometheus_config()
        stack.generate_grafana_dashboards()
        stack.setup_logging_stack()
        print("✅ Configurations generated")
    
    if args.deploy:
        success = stack.deploy_observability_stack(args.environment)
        if not success:
            sys.exit(1)
    
    # Generate and print report
    report = stack.generate_observability_report()
    print("\n📊 Observability Stack Report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()