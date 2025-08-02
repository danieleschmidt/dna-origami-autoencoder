#!/usr/bin/env python3
"""
Metrics collection script for DNA-Origami-AutoEncoder project.
Collects various project metrics and updates the project-metrics.json file.
"""

import json
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import requests
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and aggregates project metrics from various sources."""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.metrics_file = repo_path / ".github" / "project-metrics.json"
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner = "danieleschmidt"
        self.repo_name = "dna-origami-autoencoder"
        
    def load_current_metrics(self) -> Dict[str, Any]:
        """Load current metrics from the metrics file."""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Metrics file not found, starting with empty metrics")
            return {"metrics": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metrics file: {e}")
            return {"metrics": {}}
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to the metrics file."""
        try:
            # Update timestamp
            metrics["last_updated"] = datetime.now().isoformat()
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info("Metrics saved successfully")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def get_git_metrics(self) -> Dict[str, Any]:
        """Collect Git-based metrics."""
        metrics = {}
        
        try:
            # Commit frequency (last 30 days)
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            result = subprocess.run([
                "git", "rev-list", "--count", "--since", thirty_days_ago, "HEAD"
            ], capture_output=True, text=True, check=True)
            
            commit_count = int(result.stdout.strip())
            metrics["commit_frequency"] = {
                "current": commit_count / 4.3,  # Convert to weekly average
                "unit": "commits_per_week",
                "last_updated": datetime.now().isoformat()
            }
            
            # Get contributor count
            result = subprocess.run([
                "git", "shortlog", "-sn", "--since", thirty_days_ago
            ], capture_output=True, text=True, check=True)
            
            contributors = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
            metrics["contributors"] = {
                "current": contributors,
                "unit": "count",
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info(f"Collected Git metrics: {commit_count} commits, {contributors} contributors")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to collect Git metrics: {e}")
        
        return metrics
    
    def get_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub API metrics."""
        metrics = {}
        
        if not self.github_token:
            logger.warning("GitHub token not provided, skipping GitHub metrics")
            return metrics
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            # Repository statistics
            repo_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
            response = requests.get(repo_url, headers=headers)
            response.raise_for_status()
            
            repo_data = response.json()
            
            metrics["github_stars"] = {
                "current": repo_data.get("stargazers_count", 0),
                "unit": "count",
                "last_updated": datetime.now().isoformat()
            }
            
            metrics["github_forks"] = {
                "current": repo_data.get("forks_count", 0),
                "unit": "count",
                "last_updated": datetime.now().isoformat()
            }
            
            metrics["open_issues"] = {
                "current": repo_data.get("open_issues_count", 0),
                "unit": "count",
                "last_updated": datetime.now().isoformat()
            }
            
            # Pull request metrics
            pr_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/pulls"
            response = requests.get(pr_url, headers=headers, params={"state": "closed"})
            response.raise_for_status()
            
            prs = response.json()
            if prs:
                # Calculate average PR cycle time
                cycle_times = []
                for pr in prs[:20]:  # Last 20 PRs
                    if pr.get("merged_at"):
                        created = datetime.fromisoformat(pr["created_at"].replace('Z', '+00:00'))
                        merged = datetime.fromisoformat(pr["merged_at"].replace('Z', '+00:00'))
                        cycle_times.append((merged - created).days)
                
                if cycle_times:
                    avg_cycle_time = sum(cycle_times) / len(cycle_times)
                    metrics["pull_request_cycle_time"] = {
                        "current": avg_cycle_time,
                        "unit": "days",
                        "last_updated": datetime.now().isoformat()
                    }
            
            logger.info(f"Collected GitHub metrics: {repo_data.get('stargazers_count', 0)} stars")
            
        except requests.RequestException as e:
            logger.error(f"Failed to collect GitHub metrics: {e}")
        
        return metrics
    
    def get_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        try:
            # Test coverage
            result = subprocess.run([
                "python", "-m", "pytest", "--cov=dna_origami_ae", 
                "--cov-report=json", "--cov-report=term-missing", "tests/"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            # Parse coverage report
            coverage_file = self.repo_path / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                metrics["test_coverage"] = {
                    "current": round(total_coverage, 2),
                    "unit": "percentage",
                    "last_updated": datetime.now().isoformat()
                }
            
            # Code complexity (using radon if available)
            try:
                result = subprocess.run([
                    "radon", "cc", "-a", "dna_origami_ae/"
                ], capture_output=True, text=True, check=True)
                
                # Parse radon output to get average complexity
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "Average complexity:" in line:
                        complexity = float(line.split(":")[1].strip().split()[0])
                        metrics["cyclomatic_complexity"] = {
                            "current": complexity,
                            "unit": "average",
                            "last_updated": datetime.now().isoformat()
                        }
                        break
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Radon not available for complexity analysis")
            
            logger.info("Collected code quality metrics")
            
        except Exception as e:
            logger.error(f"Failed to collect code quality metrics: {e}")
        
        return metrics
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        metrics = {}
        
        try:
            # CI build time (from GitHub Actions API if available)
            if self.github_token:
                headers = {
                    "Authorization": f"token {self.github_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
                
                runs_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
                response = requests.get(runs_url, headers=headers, params={"per_page": 10})
                
                if response.status_code == 200:
                    runs = response.json().get("workflow_runs", [])
                    build_times = []
                    
                    for run in runs:
                        if run.get("status") == "completed":
                            started = datetime.fromisoformat(run["created_at"].replace('Z', '+00:00'))
                            finished = datetime.fromisoformat(run["updated_at"].replace('Z', '+00:00'))
                            duration = (finished - started).total_seconds()
                            build_times.append(duration)
                    
                    if build_times:
                        avg_build_time = sum(build_times) / len(build_times)
                        metrics["build_time"] = {
                            "current": round(avg_build_time, 2),
                            "unit": "seconds",
                            "last_updated": datetime.now().isoformat()
                        }
            
            # Test execution time
            import time
            start_time = time.time()
            
            result = subprocess.run([
                "python", "-m", "pytest", "tests/unit/", "-v"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            test_duration = time.time() - start_time
            metrics["test_execution_time"] = {
                "current": round(test_duration, 2),
                "unit": "seconds",
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info(f"Collected performance metrics: {test_duration:.2f}s test time")
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
        
        return metrics
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics."""
        metrics = {}
        
        try:
            # Security vulnerabilities using safety
            result = subprocess.run([
                "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            vulnerability_count = 0
            if result.returncode != 0:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    vulnerability_count = len(vulnerabilities)
                except json.JSONDecodeError:
                    # If JSON parsing fails, assume there are vulnerabilities
                    vulnerability_count = 1
            
            metrics["security_vulnerabilities"] = {
                "current": vulnerability_count,
                "unit": "count",
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info(f"Collected security metrics: {vulnerability_count} vulnerabilities")
            
        except Exception as e:
            logger.error(f"Failed to collect security metrics: {e}")
        
        return metrics
    
    def get_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency metrics."""
        metrics = {}
        
        try:
            # Check for outdated dependencies
            result = subprocess.run([
                "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                outdated_packages = json.loads(result.stdout)
                outdated_count = len(outdated_packages)
                
                # Calculate average age (simplified)
                avg_age = min(outdated_count * 5, 90) if outdated_count > 0 else 0
                
                metrics["dependency_freshness"] = {
                    "current": avg_age,
                    "unit": "days",
                    "last_updated": datetime.now().isoformat()
                }
                
                metrics["outdated_dependencies"] = {
                    "current": outdated_count,
                    "unit": "count",
                    "last_updated": datetime.now().isoformat()
                }
            
            logger.info("Collected dependency metrics")
            
        except Exception as e:
            logger.error(f"Failed to collect dependency metrics: {e}")
        
        return metrics
    
    def calculate_trends(self, current_metrics: Dict[str, Any], new_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trend information for metrics."""
        
        def get_trend(current_value: float, new_value: float, higher_is_better: bool = True) -> str:
            """Determine trend direction."""
            if current_value == new_value:
                return "stable"
            elif new_value > current_value:
                return "improving" if higher_is_better else "declining"
            else:
                return "declining" if higher_is_better else "improving"
        
        # Define which metrics have "higher is better" semantics
        higher_is_better_metrics = {
            "test_coverage", "github_stars", "contributors", "commit_frequency",
            "deployment_success_rate", "model_accuracy", "encoding_efficiency",
            "folding_success_rate", "simulation_speed", "downloads", "citations"
        }
        
        # Update trends
        for category, metrics in new_metrics.items():
            if category in current_metrics.get("metrics", {}):
                current_category = current_metrics["metrics"][category]
                
                for metric_name, metric_data in metrics.items():
                    if metric_name in current_category:
                        current_value = current_category[metric_name].get("current", 0)
                        new_value = metric_data.get("current", 0)
                        
                        if isinstance(current_value, (int, float)) and isinstance(new_value, (int, float)):
                            trend = get_trend(
                                current_value, 
                                new_value, 
                                metric_name in higher_is_better_metrics
                            )
                            metric_data["trend"] = trend
        
        return new_metrics
    
    def run(self) -> bool:
        """Run the complete metrics collection process."""
        logger.info("Starting metrics collection")
        
        # Load current metrics
        current_metrics = self.load_current_metrics()
        
        # Collect new metrics
        new_metrics = {"metrics": {}}
        
        # Collect various metric categories
        git_metrics = self.get_git_metrics()
        if git_metrics:
            new_metrics["metrics"]["development_velocity"] = git_metrics
        
        github_metrics = self.get_github_metrics()
        if github_metrics:
            new_metrics["metrics"]["project_health"] = {
                **new_metrics["metrics"].get("project_health", {}),
                **github_metrics
            }
            new_metrics["metrics"]["community"] = {
                **new_metrics["metrics"].get("community", {}),
                **{k: v for k, v in github_metrics.items() if k.startswith("github_")}
            }
        
        code_quality_metrics = self.get_code_quality_metrics()
        if code_quality_metrics:
            new_metrics["metrics"]["code_quality"] = code_quality_metrics
        
        performance_metrics = self.get_performance_metrics()
        if performance_metrics:
            new_metrics["metrics"]["performance"] = performance_metrics
        
        security_metrics = self.get_security_metrics()
        if security_metrics:
            new_metrics["metrics"]["project_health"] = {
                **new_metrics["metrics"].get("project_health", {}),
                **security_metrics
            }
        
        dependency_metrics = self.get_dependency_metrics()
        if dependency_metrics:
            new_metrics["metrics"]["project_health"] = {
                **new_metrics["metrics"].get("project_health", {}),
                **dependency_metrics
            }
        
        # Calculate trends
        new_metrics = self.calculate_trends(current_metrics, new_metrics)
        
        # Merge with existing metrics structure
        final_metrics = current_metrics.copy()
        if "metrics" not in final_metrics:
            final_metrics["metrics"] = {}
        
        for category, metrics in new_metrics["metrics"].items():
            if category not in final_metrics["metrics"]:
                final_metrics["metrics"][category] = {}
            final_metrics["metrics"][category].update(metrics)
        
        # Save updated metrics
        self.save_metrics(final_metrics)
        
        logger.info("Metrics collection completed successfully")
        return True

def main():
    """Main entry point for the metrics collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--repo-path", type=Path, default=Path("."), help="Path to repository")
    
    args = parser.parse_args()
    
    collector = MetricsCollector(args.repo_path)
    success = collector.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()