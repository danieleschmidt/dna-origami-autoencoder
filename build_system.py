#!/usr/bin/env python3
"""
Comprehensive build system for DNA-Origami-AutoEncoder.

This script provides automated building, testing, and deployment
capabilities with support for multiple environments and architectures.
"""

import os
import sys
import subprocess
import argparse
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

@dataclass
class BuildResult:
    """Container for build results."""
    
    target: str
    success: bool
    duration_seconds: float
    image_size_mb: Optional[float] = None
    build_log: str = ""
    image_id: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class BuildSystem:
    """Comprehensive build system for DNA-Origami-AutoEncoder."""
    
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.build_results: List[BuildResult] = []
        self.start_time = time.time()
        
    def build_production_image(self, 
                             tag: str = "dna-origami-ae:latest",
                             platform: str = "linux/amd64",
                             push: bool = False) -> BuildResult:
        """Build production Docker image."""
        return self._build_docker_image(
            target="production",
            tag=tag,
            platform=platform,
            push=push,
            build_args={
                "BUILDKIT_INLINE_CACHE": "1"
            }
        )
    
    def build_development_image(self, 
                              tag: str = "dna-origami-ae:dev",
                              platform: str = "linux/amd64") -> BuildResult:
        """Build development Docker image."""
        return self._build_docker_image(
            target="development",
            tag=tag,
            platform=platform,
            build_args={
                "BUILDKIT_INLINE_CACHE": "1"
            }
        )
    
    def build_testing_image(self, 
                          tag: str = "dna-origami-ae:test",
                          platform: str = "linux/amd64") -> BuildResult:
        """Build testing Docker image."""
        return self._build_docker_image(
            target="testing",
            tag=tag,
            platform=platform
        )
    
    def build_gpu_image(self, 
                       tag: str = "dna-origami-ae:gpu",
                       platform: str = "linux/amd64") -> BuildResult:
        """Build GPU-optimized Docker image."""
        return self._build_docker_image(
            target="gpu",
            tag=tag,
            platform=platform,
            build_args={
                "CUDA_VERSION": "12.1"
            }
        )
    
    def build_multi_arch_images(self, 
                               tag_base: str = "dna-origami-ae",
                               platforms: List[str] = None,
                               push: bool = False) -> Dict[str, BuildResult]:
        """Build multi-architecture images."""
        if platforms is None:
            platforms = ["linux/amd64", "linux/arm64"]
        
        results = {}
        
        for platform in platforms:
            platform_tag = f"{tag_base}:{platform.replace('/', '-')}"
            
            result = self._build_docker_image(
                target="production",
                tag=platform_tag,
                platform=platform,
                push=push
            )
            
            results[platform] = result
        
        return results
    
    def build_all_targets(self, 
                         tag_base: str = "dna-origami-ae",
                         push: bool = False) -> Dict[str, BuildResult]:
        """Build all Docker targets."""
        targets = {
            "production": f"{tag_base}:latest",
            "development": f"{tag_base}:dev",
            "testing": f"{tag_base}:test",
            "gpu": f"{tag_base}:gpu"
        }
        
        results = {}
        
        for target, tag in targets.items():
            print(f"🔨 Building {target} image: {tag}")
            
            if target == "gpu":
                result = self.build_gpu_image(tag=tag)
            elif target == "development":
                result = self.build_development_image(tag=tag)
            elif target == "testing":
                result = self.build_testing_image(tag=tag)
            else:
                result = self.build_production_image(tag=tag, push=push)
            
            results[target] = result
            
            if result.success:
                print(f"✅ Successfully built {target}: {tag}")
            else:
                print(f"❌ Failed to build {target}: {tag}")
        
        return results
    
    def _build_docker_image(self, 
                           target: str,
                           tag: str,
                           platform: str = "linux/amd64",
                           push: bool = False,
                           build_args: Dict[str, str] = None) -> BuildResult:
        """Build a Docker image with specified configuration."""
        start_time = time.time()
        
        cmd = [
            "docker", "buildx", "build",
            "--platform", platform,
            "--target", target,
            "--tag", tag,
            "--file", "Dockerfile"
        ]
        
        # Add build arguments
        if build_args:
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])
        
        # Add push flag if requested
        if push:
            cmd.append("--push")
        else:
            cmd.append("--load")
        
        # Add context (current directory)
        cmd.append(".")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=3600  # 1 hour timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            build_log = result.stdout + result.stderr
            
            # Get image size if build was successful
            image_size_mb = None
            image_id = ""
            
            if success and not push:
                size_info = self._get_image_size(tag)
                image_size_mb = size_info.get("size_mb")
                image_id = size_info.get("image_id", "")
            
            build_result = BuildResult(
                target=target,
                success=success,
                duration_seconds=duration,
                image_size_mb=image_size_mb,
                build_log=build_log,
                image_id=image_id,
                tags=[tag]
            )
            
            self.build_results.append(build_result)
            return build_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            build_result = BuildResult(
                target=target,
                success=False,
                duration_seconds=duration,
                build_log="Build timed out after 1 hour",
                tags=[tag]
            )
            
            self.build_results.append(build_result)
            return build_result
        
        except Exception as e:
            duration = time.time() - start_time
            build_result = BuildResult(
                target=target,
                success=False,
                duration_seconds=duration,
                build_log=f"Build error: {str(e)}",
                tags=[tag]
            )
            
            self.build_results.append(build_result)
            return build_result
    
    def _get_image_size(self, tag: str) -> Dict[str, Any]:
        """Get Docker image size information."""
        try:
            # Get image ID
            id_result = subprocess.run(
                ["docker", "images", "--format", "{{.ID}}", tag],
                capture_output=True,
                text=True
            )
            
            if id_result.returncode != 0:
                return {}
            
            image_id = id_result.stdout.strip()
            
            # Get image size
            size_result = subprocess.run(
                ["docker", "images", "--format", "{{.Size}}", tag],
                capture_output=True,
                text=True
            )
            
            if size_result.returncode != 0:
                return {"image_id": image_id}
            
            size_str = size_result.stdout.strip()
            
            # Parse size (e.g., "1.2GB" -> 1200 MB)
            size_mb = self._parse_size_to_mb(size_str)
            
            return {
                "image_id": image_id,
                "size_mb": size_mb
            }
            
        except Exception:
            return {}
    
    def _parse_size_to_mb(self, size_str: str) -> Optional[float]:
        """Parse Docker size string to MB."""
        try:
            size_str = size_str.upper()
            
            if "GB" in size_str:
                return float(size_str.replace("GB", "")) * 1000
            elif "MB" in size_str:
                return float(size_str.replace("MB", ""))
            elif "KB" in size_str:
                return float(size_str.replace("KB", "")) / 1000
            elif "B" in size_str:
                return float(size_str.replace("B", "")) / 1000000
            
            return None
            
        except Exception:
            return None
    
    def run_build_tests(self, image_tag: str = "dna-origami-ae:test") -> bool:
        """Run tests inside Docker container."""
        print(f"🧪 Running tests in container: {image_tag}")
        
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.project_root}:/app",
            "-e", "PYTHONPATH=/app",
            image_tag,
            "python", "-m", "pytest", 
            "--cov=dna_origami_ae",
            "--cov-report=xml",
            "--cov-report=term-missing",
            "tests/"
        ]
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False
    
    def run_security_scan(self, image_tag: str) -> Dict[str, Any]:
        """Run security scan on Docker image."""
        print(f"🔒 Running security scan on: {image_tag}")
        
        scan_results = {}
        
        # Trivy vulnerability scan
        try:
            trivy_cmd = [
                "trivy", "image", 
                "--format", "json",
                "--exit-code", "0",
                image_tag
            ]
            
            result = subprocess.run(
                trivy_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )
            
            if result.returncode == 0 and result.stdout:
                scan_results["trivy"] = json.loads(result.stdout)
            else:
                scan_results["trivy"] = {"error": result.stderr}
                
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            scan_results["trivy"] = {"error": "Trivy scan failed or not available"}
        
        # Docker bench security (if available)
        try:
            bench_cmd = ["docker-bench-security"]
            result = subprocess.run(
                bench_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            scan_results["docker_bench"] = {
                "output": result.stdout,
                "exit_code": result.returncode
            }
        except (subprocess.TimeoutExpired, FileNotFoundError):
            scan_results["docker_bench"] = {"error": "Docker bench security not available"}
        
        return scan_results
    
    def cleanup_build_artifacts(self, keep_latest: bool = True):
        """Clean up build artifacts and unused images."""
        print("🧹 Cleaning up build artifacts...")
        
        # Remove dangling images
        subprocess.run(
            ["docker", "image", "prune", "-f"],
            capture_output=True
        )
        
        # Remove unused build cache
        subprocess.run(
            ["docker", "buildx", "prune", "-f"],
            capture_output=True
        )
        
        if not keep_latest:
            # Remove all dna-origami-ae images
            try:
                # Get all dna-origami-ae images
                result = subprocess.run(
                    ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "dna-origami-ae"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    images = result.stdout.strip().split('\n')
                    for image in images:
                        if image.strip():
                            subprocess.run(
                                ["docker", "rmi", image],
                                capture_output=True
                            )
            except Exception:
                pass
        
        print("✅ Cleanup completed")
    
    def generate_build_report(self, output_file: str = "build-report.json") -> Dict[str, Any]:
        """Generate comprehensive build report."""
        total_duration = time.time() - self.start_time
        
        summary = {
            "total_duration_seconds": total_duration,
            "total_builds": len(self.build_results),
            "successful_builds": sum(1 for r in self.build_results if r.success),
            "failed_builds": sum(1 for r in self.build_results if not r.success),
            "total_image_size_mb": sum(r.image_size_mb or 0 for r in self.build_results),
            "average_build_time": total_duration / len(self.build_results) if self.build_results else 0
        }
        
        report = {
            "summary": summary,
            "builds": [asdict(result) for result in self.build_results],
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "docker_version": self._get_docker_version()
            }
        }
        
        # Write report to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _get_docker_version(self) -> str:
        """Get Docker version information."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True
            )
            return result.stdout.strip() if result.returncode == 0 else "Unknown"
        except Exception:
            return "Unknown"
    
    def print_build_summary(self):
        """Print build execution summary."""
        print("\n" + "=" * 60)
        print("🔨 DNA-Origami-AutoEncoder Build Summary")
        print("=" * 60)
        
        successful = sum(1 for r in self.build_results if r.success)
        failed = sum(1 for r in self.build_results if not r.success)
        
        print(f"📊 Total Builds: {len(self.build_results)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        total_size = sum(r.image_size_mb or 0 for r in self.build_results)
        if total_size > 0:
            print(f"💾 Total Image Size: {total_size:.1f} MB")
        
        print("\n📋 Build Results:")
        for result in self.build_results:
            status = "✅" if result.success else "❌"
            size_info = f" ({result.image_size_mb:.1f}MB)" if result.image_size_mb else ""
            print(f"  {status} {result.target}: {result.duration_seconds:.1f}s{size_info}")
        
        print(f"\n🕒 Total Duration: {time.time() - self.start_time:.1f}s")


def main():
    """Main entry point for build system."""
    parser = argparse.ArgumentParser(
        description="DNA-Origami-AutoEncoder Build System"
    )
    
    parser.add_argument("--production", action="store_true", help="Build production image")
    parser.add_argument("--development", action="store_true", help="Build development image")
    parser.add_argument("--testing", action="store_true", help="Build testing image")
    parser.add_argument("--gpu", action="store_true", help="Build GPU image")
    parser.add_argument("--all", action="store_true", help="Build all targets")
    parser.add_argument("--multi-arch", action="store_true", help="Build multi-architecture images")
    
    parser.add_argument("--tag", default="dna-origami-ae", help="Base tag for images")
    parser.add_argument("--platform", default="linux/amd64", help="Target platform")
    parser.add_argument("--push", action="store_true", help="Push images to registry")
    
    parser.add_argument("--test", action="store_true", help="Run tests after building")
    parser.add_argument("--security-scan", action="store_true", help="Run security scan")
    parser.add_argument("--cleanup", action="store_true", help="Clean up after build")
    
    parser.add_argument("--report", default="build-report.json", help="Output report file")
    
    args = parser.parse_args()
    
    builder = BuildSystem()
    
    # Build specific targets
    if args.production:
        builder.build_production_image(
            tag=f"{args.tag}:latest",
            platform=args.platform,
            push=args.push
        )
    elif args.development:
        builder.build_development_image(
            tag=f"{args.tag}:dev",
            platform=args.platform
        )
    elif args.testing:
        builder.build_testing_image(
            tag=f"{args.tag}:test",
            platform=args.platform
        )
    elif args.gpu:
        builder.build_gpu_image(
            tag=f"{args.tag}:gpu",
            platform=args.platform
        )
    elif args.multi_arch:
        builder.build_multi_arch_images(
            tag_base=args.tag,
            push=args.push
        )
    elif args.all:
        builder.build_all_targets(
            tag_base=args.tag,
            push=args.push
        )
    else:
        # Default: build production image
        builder.build_production_image(
            tag=f"{args.tag}:latest",
            platform=args.platform,
            push=args.push
        )
    
    # Run tests if requested
    if args.test:
        test_success = builder.run_build_tests(f"{args.tag}:test")
        if not test_success:
            print("❌ Tests failed")
            sys.exit(1)
    
    # Run security scan if requested
    if args.security_scan:
        for result in builder.build_results:
            if result.success and result.tags:
                scan_results = builder.run_security_scan(result.tags[0])
                print(f"🔒 Security scan completed for {result.tags[0]}")
    
    # Cleanup if requested
    if args.cleanup:
        builder.cleanup_build_artifacts()
    
    # Generate report and summary
    builder.generate_build_report(args.report)
    builder.print_build_summary()
    
    # Exit with failure if any builds failed
    failed_builds = sum(1 for r in builder.build_results if not r.success)
    sys.exit(1 if failed_builds > 0 else 0)


if __name__ == "__main__":
    main()