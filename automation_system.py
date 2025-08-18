#!/usr/bin/env python3
"""
Comprehensive automation system for DNA-Origami-AutoEncoder.

This system provides automated maintenance, metrics collection, and 
repository health monitoring capabilities.
"""

import os
import sys
import subprocess
import json
import time
import schedule
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class MetricData:
    """Container for collected metrics."""
    
    timestamp: str
    metric_type: str
    value: float
    unit: str
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

@dataclass
class AutomationTask:
    """Configuration for an automation task."""
    
    name: str
    description: str
    schedule: str  # cron-like schedule
    command: str
    timeout_minutes: int = 30
    retry_count: int = 2
    enabled: bool = True
    last_run: Optional[str] = None
    success_count: int = 0
    failure_count: int = 0

class MetricsCollector:
    """Comprehensive metrics collection system."""
    
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.metrics_dir = project_root / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
    def collect_repository_metrics(self) -> List[MetricData]:
        """Collect repository health and activity metrics."""
        metrics = []
        timestamp = datetime.now().isoformat()
        
        # Code quality metrics
        metrics.extend(self._collect_code_quality_metrics(timestamp))
        
        # Repository size and structure metrics
        metrics.extend(self._collect_repository_structure_metrics(timestamp))
        
        # Git activity metrics
        metrics.extend(self._collect_git_activity_metrics(timestamp))
        
        # Test coverage metrics
        metrics.extend(self._collect_test_coverage_metrics(timestamp))
        
        # Dependency metrics
        metrics.extend(self._collect_dependency_metrics(timestamp))
        
        return metrics
    
    def _collect_code_quality_metrics(self, timestamp: str) -> List[MetricData]:
        """Collect code quality metrics."""
        metrics = []
        
        try:
            # Count lines of code
            result = subprocess.run(
                ["find", "dna_origami_ae", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    total_lines = int(lines[-1].split()[0])
                    metrics.append(MetricData(
                        timestamp=timestamp,
                        metric_type="code_lines_total",
                        value=total_lines,
                        unit="lines",
                        labels={"component": "main_package"}
                    ))
        except Exception:
            pass
        
        # Count Python files
        try:
            result = subprocess.run(
                ["find", "dna_origami_ae", "-name", "*.py", "-type", "f"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                file_count = len(result.stdout.strip().split('\n'))
                metrics.append(MetricData(
                    timestamp=timestamp,
                    metric_type="python_files_count",
                    value=file_count,
                    unit="files",
                    labels={"component": "main_package"}
                ))
        except Exception:
            pass
        
        # Complexity metrics (if available)
        try:
            result = subprocess.run(
                ["python3", "-m", "radon", "cc", "dna_origami_ae", "--average"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0 and "Average complexity" in result.stdout:
                # Parse average complexity
                for line in result.stdout.split('\n'):
                    if "Average complexity" in line:
                        complexity = float(line.split()[-1].rstrip('()'))
                        metrics.append(MetricData(
                            timestamp=timestamp,
                            metric_type="code_complexity_average",
                            value=complexity,
                            unit="complexity_score",
                            labels={"component": "main_package"}
                        ))
                        break
        except Exception:
            pass
        
        return metrics
    
    def _collect_repository_structure_metrics(self, timestamp: str) -> List[MetricData]:
        """Collect repository structure metrics."""
        metrics = []
        
        # Directory sizes
        for directory in ["data", "models", "results", "logs", "tests"]:
            dir_path = self.project_root / directory
            if dir_path.exists():
                try:
                    result = subprocess.run(
                        ["du", "-sb", str(dir_path)],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        size_bytes = int(result.stdout.split()[0])
                        metrics.append(MetricData(
                            timestamp=timestamp,
                            metric_type="directory_size_bytes",
                            value=size_bytes,
                            unit="bytes",
                            labels={"directory": directory}
                        ))
                except Exception:
                    pass
        
        # File counts by type
        file_types = {
            "python": "*.py",
            "jupyter": "*.ipynb",
            "yaml": "*.yml",
            "json": "*.json",
            "markdown": "*.md"
        }
        
        for file_type, pattern in file_types.items():
            try:
                result = subprocess.run(
                    ["find", ".", "-name", pattern, "-type", "f"],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                
                if result.returncode == 0:
                    count = len([line for line in result.stdout.strip().split('\n') if line])
                    metrics.append(MetricData(
                        timestamp=timestamp,
                        metric_type="file_count_by_type",
                        value=count,
                        unit="files",
                        labels={"file_type": file_type}
                    ))
            except Exception:
                pass
        
        return metrics
    
    def _collect_git_activity_metrics(self, timestamp: str) -> List[MetricData]:
        """Collect Git repository activity metrics."""
        metrics = []
        
        try:
            # Commit count in last 30 days
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            result = subprocess.run(
                ["git", "rev-list", "--count", f"--since={thirty_days_ago}", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                commit_count = int(result.stdout.strip())
                metrics.append(MetricData(
                    timestamp=timestamp,
                    metric_type="git_commits_30d",
                    value=commit_count,
                    unit="commits",
                    labels={"period": "30_days"}
                ))
        except Exception:
            pass
        
        try:
            # Total commit count
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                total_commits = int(result.stdout.strip())
                metrics.append(MetricData(
                    timestamp=timestamp,
                    metric_type="git_commits_total",
                    value=total_commits,
                    unit="commits",
                    labels={"scope": "all_time"}
                ))
        except Exception:
            pass
        
        try:
            # Branch count
            result = subprocess.run(
                ["git", "branch", "-r"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                branch_count = len(result.stdout.strip().split('\n'))
                metrics.append(MetricData(
                    timestamp=timestamp,
                    metric_type="git_branches_count",
                    value=branch_count,
                    unit="branches",
                    labels={"type": "remote"}
                ))
        except Exception:
            pass
        
        return metrics
    
    def _collect_test_coverage_metrics(self, timestamp: str) -> List[MetricData]:
        """Collect test coverage metrics."""
        metrics = []
        
        # Run coverage analysis
        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", "--cov=dna_origami_ae", 
                 "--cov-report=json", "--cov-report=term-missing", 
                 "tests/", "-q"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300
            )
            
            # Parse coverage JSON if available
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                metrics.append(MetricData(
                    timestamp=timestamp,
                    metric_type="test_coverage_percent",
                    value=total_coverage,
                    unit="percent",
                    labels={"scope": "total"}
                ))
                
                # Coverage by file/module
                for filename, file_data in coverage_data.get("files", {}).items():
                    if "dna_origami_ae" in filename:
                        module_coverage = file_data.get("summary", {}).get("percent_covered", 0)
                        module_name = filename.replace("/", ".").replace(".py", "")
                        
                        metrics.append(MetricData(
                            timestamp=timestamp,
                            metric_type="test_coverage_by_module",
                            value=module_coverage,
                            unit="percent",
                            labels={"module": module_name}
                        ))
        except Exception:
            pass
        
        return metrics
    
    def _collect_dependency_metrics(self, timestamp: str) -> List[MetricData]:
        """Collect dependency and security metrics."""
        metrics = []
        
        # Count dependencies
        try:
            with open(self.project_root / "requirements.txt") as f:
                deps = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                
            metrics.append(MetricData(
                timestamp=timestamp,
                metric_type="dependencies_count",
                value=len(deps),
                unit="packages",
                labels={"type": "production"}
            ))
        except Exception:
            pass
        
        # Security vulnerabilities (if safety is available)
        try:
            result = subprocess.run(
                ["python3", "-m", "safety", "check", "--json"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                safety_data = json.loads(result.stdout)
                vuln_count = len(safety_data)
                
                metrics.append(MetricData(
                    timestamp=timestamp,
                    metric_type="security_vulnerabilities",
                    value=vuln_count,
                    unit="vulnerabilities",
                    labels={"scanner": "safety"}
                ))
        except Exception:
            pass
        
        return metrics
    
    def save_metrics(self, metrics: List[MetricData], filename: str = None) -> str:
        """Save metrics to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.metrics_dir / filename
        
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": [asdict(metric) for metric in metrics],
            "count": len(metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        return str(filepath)

class AutomationEngine:
    """Automated maintenance and operations engine."""
    
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.config_file = project_root / "automation_config.json"
        self.log_file = project_root / "logs" / "automation.log"
        self.log_file.parent.mkdir(exist_ok=True)
        
        self.tasks = self._load_automation_config()
        self.metrics_collector = MetricsCollector(project_root)
        
    def _load_automation_config(self) -> Dict[str, AutomationTask]:
        """Load automation configuration."""
        default_tasks = {
            "metrics_collection": AutomationTask(
                name="metrics_collection",
                description="Collect repository and code quality metrics",
                schedule="daily",
                command="python3 automation_system.py --collect-metrics",
                timeout_minutes=15
            ),
            "dependency_audit": AutomationTask(
                name="dependency_audit",
                description="Check for security vulnerabilities in dependencies",
                schedule="weekly",
                command="python3 -m safety check --json --output security_audit.json",
                timeout_minutes=10
            ),
            "code_quality_check": AutomationTask(
                name="code_quality_check",
                description="Run comprehensive code quality analysis",
                schedule="daily",
                command="python3 run_comprehensive_tests.py --quality",
                timeout_minutes=20
            ),
            "cleanup_artifacts": AutomationTask(
                name="cleanup_artifacts",
                description="Clean up old build artifacts and temporary files",
                schedule="weekly",
                command="find . -name '__pycache__' -type d -exec rm -rf {} +; find . -name '*.pyc' -delete",
                timeout_minutes=5
            ),
            "backup_models": AutomationTask(
                name="backup_models",
                description="Backup trained models and important artifacts",
                schedule="weekly",
                command="tar -czf backups/models_$(date +%Y%m%d).tar.gz models/trained/",
                timeout_minutes=30
            ),
            "update_documentation": AutomationTask(
                name="update_documentation",
                description="Regenerate API documentation",
                schedule="weekly",
                command="sphinx-build -b html docs/ docs/_build/",
                timeout_minutes=15
            ),
            "performance_benchmarks": AutomationTask(
                name="performance_benchmarks",
                description="Run performance benchmarks and collect metrics",
                schedule="weekly",
                command="python3 run_comprehensive_tests.py --performance",
                timeout_minutes=60
            )
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    config_data = json.load(f)
                
                tasks = {}
                for name, task_data in config_data.get("tasks", {}).items():
                    tasks[name] = AutomationTask(**task_data)
                
                return tasks
            except Exception:
                pass
        
        return default_tasks
    
    def _save_automation_config(self):
        """Save automation configuration."""
        config_data = {
            "tasks": {
                name: asdict(task) for name, task in self.tasks.items()
            },
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def run_task(self, task_name: str) -> bool:
        """Run a specific automation task."""
        if task_name not in self.tasks:
            self._log(f"Task {task_name} not found")
            return False
        
        task = self.tasks[task_name]
        
        if not task.enabled:
            self._log(f"Task {task_name} is disabled")
            return False
        
        self._log(f"Starting task: {task_name}")
        start_time = time.time()
        
        for attempt in range(task.retry_count + 1):
            try:
                result = subprocess.run(
                    task.command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=task.timeout_minutes * 60
                )
                
                if result.returncode == 0:
                    duration = time.time() - start_time
                    task.last_run = datetime.now().isoformat()
                    task.success_count += 1
                    
                    self._log(f"Task {task_name} completed successfully in {duration:.1f}s")
                    self._save_automation_config()
                    return True
                else:
                    self._log(f"Task {task_name} failed (attempt {attempt + 1}): {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                self._log(f"Task {task_name} timed out (attempt {attempt + 1})")
            except Exception as e:
                self._log(f"Task {task_name} error (attempt {attempt + 1}): {e}")
        
        # All attempts failed
        task.failure_count += 1
        self._save_automation_config()
        self._log(f"Task {task_name} failed after {task.retry_count + 1} attempts")
        return False
    
    def run_scheduled_tasks(self):
        """Run tasks based on their schedule."""
        current_time = datetime.now()
        
        for task_name, task in self.tasks.items():
            if not task.enabled:
                continue
            
            should_run = False
            
            # Simple schedule parsing
            if task.schedule == "daily":
                if not task.last_run or self._days_since_last_run(task) >= 1:
                    should_run = True
            elif task.schedule == "weekly":
                if not task.last_run or self._days_since_last_run(task) >= 7:
                    should_run = True
            elif task.schedule == "hourly":
                if not task.last_run or self._hours_since_last_run(task) >= 1:
                    should_run = True
            
            if should_run:
                self.run_task(task_name)
    
    def _days_since_last_run(self, task: AutomationTask) -> float:
        """Calculate days since last run."""
        if not task.last_run:
            return float('inf')
        
        last_run = datetime.fromisoformat(task.last_run)
        return (datetime.now() - last_run).total_seconds() / 86400
    
    def _hours_since_last_run(self, task: AutomationTask) -> float:
        """Calculate hours since last run."""
        if not task.last_run:
            return float('inf')
        
        last_run = datetime.fromisoformat(task.last_run)
        return (datetime.now() - last_run).total_seconds() / 3600
    
    def _log(self, message: str):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def generate_automation_report(self) -> Dict[str, Any]:
        """Generate comprehensive automation report."""
        report = {
            "automation_status": {
                "total_tasks": len(self.tasks),
                "enabled_tasks": sum(1 for task in self.tasks.values() if task.enabled),
                "disabled_tasks": sum(1 for task in self.tasks.values() if not task.enabled)
            },
            "task_statistics": {},
            "recent_activity": [],
            "health_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Task statistics
        for name, task in self.tasks.items():
            total_runs = task.success_count + task.failure_count
            success_rate = (task.success_count / total_runs * 100) if total_runs > 0 else 0
            
            report["task_statistics"][name] = {
                "enabled": task.enabled,
                "total_runs": total_runs,
                "success_count": task.success_count,
                "failure_count": task.failure_count,
                "success_rate": success_rate,
                "last_run": task.last_run,
                "schedule": task.schedule
            }
        
        # Health metrics
        report["health_metrics"] = {
            "overall_success_rate": sum(task.success_count for task in self.tasks.values()) / 
                                  max(sum(task.success_count + task.failure_count for task in self.tasks.values()), 1) * 100,
            "tasks_overdue": sum(1 for task in self.tasks.values() if self._is_task_overdue(task)),
            "last_metrics_collection": self._get_last_metrics_timestamp()
        }
        
        return report
    
    def _is_task_overdue(self, task: AutomationTask) -> bool:
        """Check if a task is overdue based on its schedule."""
        if not task.enabled or not task.last_run:
            return False
        
        if task.schedule == "daily" and self._days_since_last_run(task) > 1.5:
            return True
        elif task.schedule == "weekly" and self._days_since_last_run(task) > 8:
            return True
        elif task.schedule == "hourly" and self._hours_since_last_run(task) > 2:
            return True
        
        return False
    
    def _get_last_metrics_timestamp(self) -> Optional[str]:
        """Get timestamp of last metrics collection."""
        metrics_files = list(self.metrics_collector.metrics_dir.glob("metrics_*.json"))
        if not metrics_files:
            return None
        
        latest_file = max(metrics_files, key=lambda x: x.stat().st_mtime)
        return datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()

class RepositoryHealthMonitor:
    """Monitor overall repository health and generate reports."""
    
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.metrics_collector = MetricsCollector(project_root)
        self.automation_engine = AutomationEngine(project_root)
        
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive repository health report."""
        # Collect current metrics
        metrics = self.metrics_collector.collect_repository_metrics()
        
        # Get automation status
        automation_report = self.automation_engine.generate_automation_report()
        
        # Calculate health scores
        health_scores = self._calculate_health_scores(metrics)
        
        report = {
            "repository_health": {
                "overall_score": health_scores["overall"],
                "component_scores": health_scores["components"],
                "status": self._get_health_status(health_scores["overall"])
            },
            "metrics_summary": {
                "total_metrics": len(metrics),
                "collection_timestamp": datetime.now().isoformat(),
                "key_metrics": self._extract_key_metrics(metrics)
            },
            "automation_status": automation_report["automation_status"],
            "recommendations": self._generate_recommendations(metrics, automation_report),
            "alerts": self._generate_alerts(metrics, automation_report),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_health_scores(self, metrics: List[MetricData]) -> Dict[str, Any]:
        """Calculate health scores based on metrics."""
        scores = {
            "code_quality": 85,  # Default baseline
            "test_coverage": 80,
            "security": 90,
            "automation": 85,
            "maintenance": 80
        }
        
        # Adjust scores based on actual metrics
        for metric in metrics:
            if metric.metric_type == "test_coverage_percent":
                scores["test_coverage"] = min(100, metric.value)
            elif metric.metric_type == "security_vulnerabilities":
                scores["security"] = max(0, 100 - (metric.value * 10))
        
        overall_score = sum(scores.values()) / len(scores)
        
        return {
            "overall": overall_score,
            "components": scores
        }
    
    def _get_health_status(self, score: float) -> str:
        """Get health status based on score."""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        else:
            return "poor"
    
    def _extract_key_metrics(self, metrics: List[MetricData]) -> Dict[str, Any]:
        """Extract key metrics for summary."""
        key_metrics = {}
        
        for metric in metrics:
            if metric.metric_type in ["code_lines_total", "test_coverage_percent", 
                                    "security_vulnerabilities", "git_commits_30d"]:
                key_metrics[metric.metric_type] = {
                    "value": metric.value,
                    "unit": metric.unit
                }
        
        return key_metrics
    
    def _generate_recommendations(self, metrics: List[MetricData], 
                                automation_report: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Check test coverage
        for metric in metrics:
            if metric.metric_type == "test_coverage_percent" and metric.value < 80:
                recommendations.append(
                    f"Improve test coverage (currently {metric.value:.1f}%, target: 80%+)"
                )
        
        # Check automation health
        if automation_report["automation_status"]["disabled_tasks"] > 0:
            recommendations.append(
                "Enable disabled automation tasks to improve maintenance"
            )
        
        # Check security
        for metric in metrics:
            if metric.metric_type == "security_vulnerabilities" and metric.value > 0:
                recommendations.append(
                    f"Address {int(metric.value)} security vulnerabilities"
                )
        
        return recommendations
    
    def _generate_alerts(self, metrics: List[MetricData], 
                        automation_report: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate alerts for critical issues."""
        alerts = []
        
        # Critical security alerts
        for metric in metrics:
            if metric.metric_type == "security_vulnerabilities" and metric.value > 5:
                alerts.append({
                    "level": "critical",
                    "message": f"High number of security vulnerabilities: {int(metric.value)}",
                    "action": "Run security audit and update dependencies"
                })
        
        # Test coverage alerts
        for metric in metrics:
            if metric.metric_type == "test_coverage_percent" and metric.value < 50:
                alerts.append({
                    "level": "warning",
                    "message": f"Low test coverage: {metric.value:.1f}%",
                    "action": "Increase test coverage to improve code quality"
                })
        
        return alerts


def main():
    """Main entry point for automation system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DNA-Origami-AutoEncoder Automation System"
    )
    
    parser.add_argument("--collect-metrics", action="store_true", 
                       help="Collect repository metrics")
    parser.add_argument("--run-task", help="Run specific automation task")
    parser.add_argument("--run-scheduled", action="store_true", 
                       help="Run scheduled tasks")
    parser.add_argument("--health-report", action="store_true", 
                       help="Generate health report")
    parser.add_argument("--output", help="Output file for reports")
    
    args = parser.parse_args()
    
    if args.collect_metrics:
        collector = MetricsCollector()
        metrics = collector.collect_repository_metrics()
        filepath = collector.save_metrics(metrics)
        print(f"✅ Collected {len(metrics)} metrics, saved to {filepath}")
    
    elif args.run_task:
        engine = AutomationEngine()
        success = engine.run_task(args.run_task)
        sys.exit(0 if success else 1)
    
    elif args.run_scheduled:
        engine = AutomationEngine()
        engine.run_scheduled_tasks()
        print("✅ Scheduled tasks completed")
    
    elif args.health_report:
        monitor = RepositoryHealthMonitor()
        report = monitor.generate_health_report()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📊 Health report saved to {args.output}")
        else:
            print(json.dumps(report, indent=2))
    
    else:
        # Default: show automation status
        engine = AutomationEngine()
        report = engine.generate_automation_report()
        print("🤖 Automation Status:")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()