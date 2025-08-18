#!/usr/bin/env python3
"""
Comprehensive test runner for DNA-Origami-AutoEncoder.

This script provides a centralized way to run different categories of tests
with proper configuration, reporting, and performance monitoring.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class TestResults:
    """Container for test execution results."""
    
    category: str
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    coverage_percentage: Optional[float] = None
    exit_code: int = 0
    output: str = ""
    command: str = ""

class TestRunner:
    """Comprehensive test runner with multiple test categories."""
    
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.results: List[TestResults] = []
        self.start_time = time.time()
    
    def run_unit_tests(self, 
                      verbose: bool = True,
                      coverage: bool = True,
                      parallel: bool = True) -> TestResults:
        """Run unit tests with coverage reporting."""
        cmd = ["python", "-m", "pytest", "tests/unit/", "-m", "unit"]
        
        if verbose:
            cmd.extend(["-v", "--tb=short"])
        
        if coverage:
            cmd.extend([
                "--cov=dna_origami_ae",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/unit",
                "--cov-report=xml:coverage-unit.xml"
            ])
        
        if parallel:
            cmd.extend(["-n", "auto"])
        
        cmd.extend([
            "--junitxml=junit-unit.xml",
            "--durations=10"
        ])
        
        return self._run_test_command(cmd, "unit")
    
    def run_integration_tests(self, 
                            verbose: bool = True,
                            timeout: int = 600) -> TestResults:
        """Run integration tests."""
        cmd = [
            "python", "-m", "pytest", 
            "tests/integration/", 
            "-m", "integration",
            f"--timeout={timeout}"
        ]
        
        if verbose:
            cmd.extend(["-v", "--tb=short"])
        
        cmd.extend([
            "--junitxml=junit-integration.xml",
            "--durations=10"
        ])
        
        return self._run_test_command(cmd, "integration")
    
    def run_performance_tests(self, 
                            verbose: bool = True,
                            benchmark: bool = True) -> TestResults:
        """Run performance and benchmark tests."""
        cmd = [
            "python", "-m", "pytest", 
            "tests/performance/", 
            "-m", "performance"
        ]
        
        if verbose:
            cmd.extend(["-v", "--tb=short"])
        
        if benchmark:
            cmd.extend([
                "--benchmark-only",
                "--benchmark-json=benchmark-results.json"
            ])
        
        cmd.extend([
            "--junitxml=junit-performance.xml"
        ])
        
        return self._run_test_command(cmd, "performance")
    
    def run_gpu_tests(self, 
                     verbose: bool = True,
                     min_memory_gb: float = 1.0) -> TestResults:
        """Run GPU-specific tests."""
        cmd = [
            "python", "-m", "pytest", 
            "tests/", 
            "-m", "gpu",
            f"--gpu-memory={min_memory_gb}"
        ]
        
        if verbose:
            cmd.extend(["-v", "--tb=short"])
        
        cmd.extend([
            "--junitxml=junit-gpu.xml"
        ])
        
        return self._run_test_command(cmd, "gpu")
    
    def run_slow_tests(self, verbose: bool = True) -> TestResults:
        """Run slow tests (typically molecular dynamics simulations)."""
        cmd = [
            "python", "-m", "pytest", 
            "tests/", 
            "-m", "slow",
            "--timeout=1800"  # 30 minutes
        ]
        
        if verbose:
            cmd.extend(["-v", "--tb=short"])
        
        cmd.extend([
            "--junitxml=junit-slow.xml"
        ])
        
        return self._run_test_command(cmd, "slow")
    
    def run_security_tests(self, verbose: bool = True) -> TestResults:
        """Run security-focused tests."""
        cmd = [
            "python", "-m", "pytest", 
            "tests/security/",
            "-m", "not experimental"
        ]
        
        if verbose:
            cmd.extend(["-v", "--tb=short"])
        
        cmd.extend([
            "--junitxml=junit-security.xml"
        ])
        
        return self._run_test_command(cmd, "security")
    
    def run_smoke_tests(self, verbose: bool = True) -> TestResults:
        """Run smoke tests for basic functionality verification."""
        cmd = [
            "python", "-m", "pytest", 
            "tests/", 
            "-m", "smoke",
            "--maxfail=1"  # Stop on first failure for smoke tests
        ]
        
        if verbose:
            cmd.extend(["-v", "--tb=short"])
        
        cmd.extend([
            "--junitxml=junit-smoke.xml"
        ])
        
        return self._run_test_command(cmd, "smoke")
    
    def run_code_quality_checks(self) -> Dict[str, TestResults]:
        """Run code quality and linting checks."""
        quality_results = {}
        
        # Black formatting check
        cmd = ["python", "-m", "black", "--check", "--diff", "dna_origami_ae/", "tests/"]
        quality_results["black"] = self._run_test_command(cmd, "black_formatting")
        
        # isort import sorting check
        cmd = ["python", "-m", "isort", "--check-only", "--diff", "dna_origami_ae/", "tests/"]
        quality_results["isort"] = self._run_test_command(cmd, "isort_imports")
        
        # flake8 linting
        cmd = ["python", "-m", "flake8", "dna_origami_ae/", "tests/"]
        quality_results["flake8"] = self._run_test_command(cmd, "flake8_linting")
        
        # mypy type checking
        cmd = ["python", "-m", "mypy", "dna_origami_ae/"]
        quality_results["mypy"] = self._run_test_command(cmd, "mypy_typing")
        
        # bandit security scanning
        cmd = ["python", "-m", "bandit", "-r", "dna_origami_ae/", "-f", "json", "-o", "bandit-report.json"]
        quality_results["bandit"] = self._run_test_command(cmd, "bandit_security")
        
        return quality_results
    
    def run_all_tests(self, 
                     include_slow: bool = False,
                     include_gpu: bool = True,
                     parallel: bool = True) -> Dict[str, TestResults]:
        """Run comprehensive test suite."""
        all_results = {}
        
        print("🧬 Running DNA-Origami-AutoEncoder Comprehensive Test Suite")
        print("=" * 60)
        
        # Core test categories
        print("\n📋 Running unit tests...")
        all_results["unit"] = self.run_unit_tests(parallel=parallel)
        
        print("\n🔗 Running integration tests...")
        all_results["integration"] = self.run_integration_tests()
        
        print("\n⚡ Running performance tests...")
        all_results["performance"] = self.run_performance_tests()
        
        print("\n🔍 Running smoke tests...")
        all_results["smoke"] = self.run_smoke_tests()
        
        print("\n🛡️ Running security tests...")
        all_results["security"] = self.run_security_tests()
        
        # Optional test categories
        if include_gpu and self._gpu_available():
            print("\n🖥️ Running GPU tests...")
            all_results["gpu"] = self.run_gpu_tests()
        
        if include_slow:
            print("\n🐌 Running slow tests...")
            all_results["slow"] = self.run_slow_tests()
        
        # Code quality checks
        print("\n✨ Running code quality checks...")
        quality_results = self.run_code_quality_checks()
        all_results.update(quality_results)
        
        return all_results
    
    def _run_test_command(self, cmd: List[str], category: str) -> TestResults:
        """Execute a test command and parse results."""
        start_time = time.time()
        cmd_str = " ".join(cmd)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=3600  # 1 hour max per test category
            )
            
            duration = time.time() - start_time
            output = result.stdout + result.stderr
            
            # Parse pytest output for test counts
            passed, failed, skipped = self._parse_pytest_output(output)
            
            # Extract coverage if available
            coverage = self._extract_coverage(output)
            
            test_result = TestResults(
                category=category,
                passed=passed,
                failed=failed,
                skipped=skipped,
                duration_seconds=duration,
                coverage_percentage=coverage,
                exit_code=result.returncode,
                output=output,
                command=cmd_str
            )
            
            self.results.append(test_result)
            return test_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            test_result = TestResults(
                category=category,
                passed=0,
                failed=1,
                skipped=0,
                duration_seconds=duration,
                exit_code=124,  # Timeout exit code
                output="Test timed out",
                command=cmd_str
            )
            
            self.results.append(test_result)
            return test_result
        
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResults(
                category=category,
                passed=0,
                failed=1,
                skipped=0,
                duration_seconds=duration,
                exit_code=1,
                output=f"Error running tests: {str(e)}",
                command=cmd_str
            )
            
            self.results.append(test_result)
            return test_result
    
    def _parse_pytest_output(self, output: str) -> tuple:
        """Parse pytest output to extract test counts."""
        passed = failed = skipped = 0
        
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line and 'failed' in line:
                # Look for summary line like "10 passed, 2 failed, 1 skipped"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        try:
                            passed = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part == 'failed' and i > 0:
                        try:
                            failed = int(parts[i-1])
                        except ValueError:
                            pass
                    elif part == 'skipped' and i > 0:
                        try:
                            skipped = int(parts[i-1])
                        except ValueError:
                            pass
        
        return passed, failed, skipped
    
    def _extract_coverage(self, output: str) -> Optional[float]:
        """Extract coverage percentage from output."""
        lines = output.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        try:
                            return float(part[:-1])
                        except ValueError:
                            pass
        return None
    
    def _gpu_available(self) -> bool:
        """Check if GPU is available for testing."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def generate_report(self, output_file: str = "test-report.json"):
        """Generate comprehensive test report."""
        total_duration = time.time() - self.start_time
        
        summary = {
            'total_duration_seconds': total_duration,
            'total_tests': sum(r.passed + r.failed + r.skipped for r in self.results),
            'total_passed': sum(r.passed for r in self.results),
            'total_failed': sum(r.failed for r in self.results),
            'total_skipped': sum(r.skipped for r in self.results),
            'success_rate': 0.0,
            'categories': len(self.results)
        }
        
        if summary['total_tests'] > 0:
            summary['success_rate'] = summary['total_passed'] / summary['total_tests'] * 100
        
        report = {
            'summary': summary,
            'results': [asdict(result) for result in self.results],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'gpu_available': self._gpu_available()
            }
        }
        
        # Write report to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self):
        """Print test execution summary."""
        print("\n" + "=" * 60)
        print("🧬 DNA-Origami-AutoEncoder Test Summary")
        print("=" * 60)
        
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_tests = total_passed + total_failed + total_skipped
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {total_passed}")
        print(f"❌ Failed: {total_failed}")
        print(f"⏭️ Skipped: {total_skipped}")
        
        if total_tests > 0:
            success_rate = total_passed / total_tests * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        print("\n📋 Category Results:")
        for result in self.results:
            status = "✅" if result.exit_code == 0 else "❌"
            print(f"  {status} {result.category}: {result.passed}P {result.failed}F {result.skipped}S "
                  f"({result.duration_seconds:.1f}s)")
            if result.coverage_percentage:
                print(f"    📊 Coverage: {result.coverage_percentage:.1f}%")
        
        print("\n🕒 Total Duration: {:.1f}s".format(time.time() - self.start_time))


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="DNA-Origami-AutoEncoder Comprehensive Test Runner"
    )
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--gpu", action="store_true", help="Run GPU tests only")
    parser.add_argument("--slow", action="store_true", help="Run slow tests only")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--security", action="store_true", help="Run security tests only")
    parser.add_argument("--quality", action="store_true", help="Run code quality checks only")
    
    parser.add_argument("--all", action="store_true", help="Run all test categories")
    parser.add_argument("--include-slow", action="store_true", help="Include slow tests in --all")
    parser.add_argument("--no-gpu", action="store_true", help="Skip GPU tests in --all")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel test execution")
    
    parser.add_argument("--report", default="test-report.json", help="Output report file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Run specific test categories
    if args.unit:
        runner.run_unit_tests(verbose=args.verbose, parallel=not args.no_parallel)
    elif args.integration:
        runner.run_integration_tests(verbose=args.verbose)
    elif args.performance:
        runner.run_performance_tests(verbose=args.verbose)
    elif args.gpu:
        runner.run_gpu_tests(verbose=args.verbose)
    elif args.slow:
        runner.run_slow_tests(verbose=args.verbose)
    elif args.smoke:
        runner.run_smoke_tests(verbose=args.verbose)
    elif args.security:
        runner.run_security_tests(verbose=args.verbose)
    elif args.quality:
        runner.run_code_quality_checks()
    elif args.all:
        runner.run_all_tests(
            include_slow=args.include_slow,
            include_gpu=not args.no_gpu,
            parallel=not args.no_parallel
        )
    else:
        # Default: run core tests
        runner.run_all_tests(
            include_slow=False,
            include_gpu=not args.no_gpu,
            parallel=not args.no_parallel
        )
    
    # Generate report and summary
    runner.generate_report(args.report)
    runner.print_summary()
    
    # Exit with failure if any tests failed
    total_failed = sum(r.failed for r in runner.results)
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()