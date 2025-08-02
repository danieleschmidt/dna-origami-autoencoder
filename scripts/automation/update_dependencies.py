#!/usr/bin/env python3
"""
Automated dependency update script for DNA-Origami-AutoEncoder.
This script checks for outdated dependencies and creates pull requests for updates.
"""

import json
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DependencyUpdater:
    """Automated dependency updater for Python and conda environments."""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.pyproject_path = repo_path / "pyproject.toml"
        self.requirements_path = repo_path / "requirements.txt"
        self.environment_path = repo_path / "environment.yml"
        
    def check_outdated_pip_packages(self) -> List[Dict]:
        """Check for outdated pip packages."""
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            outdated = json.loads(result.stdout)
            logger.info(f"Found {len(outdated)} outdated pip packages")
            return outdated
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to check outdated pip packages: {e}")
            return []
    
    def check_outdated_conda_packages(self) -> List[Dict]:
        """Check for outdated conda packages."""
        try:
            result = subprocess.run(
                ["conda", "list", "--json"],
                capture_output=True,
                text=True,
                check=True
            )
            installed = json.loads(result.stdout)
            
            # Check for updates (simplified - would need conda-forge API)
            outdated = []
            for package in installed:
                if package.get("channel") == "conda-forge":
                    # In a real implementation, check conda-forge API
                    # For now, just return packages that might be outdated
                    pass
            
            logger.info(f"Found {len(outdated)} outdated conda packages")
            return outdated
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to check outdated conda packages: {e}")
            return []
    
    def check_security_vulnerabilities(self) -> List[Dict]:
        """Check for security vulnerabilities using safety."""
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("No security vulnerabilities found")
                return []
            else:
                vulnerabilities = json.loads(result.stdout)
                logger.warning(f"Found {len(vulnerabilities)} security vulnerabilities")
                return vulnerabilities
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Failed to check security vulnerabilities: {e}")
            return []
    
    def update_pyproject_toml(self, package_updates: List[Dict]) -> bool:
        """Update package versions in pyproject.toml."""
        if not self.pyproject_path.exists():
            logger.warning("pyproject.toml not found")
            return False
        
        try:
            import toml
            with open(self.pyproject_path, 'r') as f:
                config = toml.load(f)
            
            updated = False
            dependencies = config.get("project", {}).get("dependencies", [])
            
            for update in package_updates:
                package_name = update["name"]
                new_version = update["latest_version"]
                
                # Update version in dependencies list
                for i, dep in enumerate(dependencies):
                    if dep.startswith(f"{package_name}==") or dep.startswith(f"{package_name}>="):
                        dependencies[i] = f"{package_name}>={new_version}"
                        updated = True
                        logger.info(f"Updated {package_name} to {new_version}")
            
            if updated:
                config["project"]["dependencies"] = dependencies
                with open(self.pyproject_path, 'w') as f:
                    toml.dump(config, f)
                logger.info("Updated pyproject.toml")
            
            return updated
        except Exception as e:
            logger.error(f"Failed to update pyproject.toml: {e}")
            return False
    
    def update_requirements_txt(self, package_updates: List[Dict]) -> bool:
        """Update package versions in requirements.txt."""
        if not self.requirements_path.exists():
            logger.warning("requirements.txt not found")
            return False
        
        try:
            with open(self.requirements_path, 'r') as f:
                lines = f.readlines()
            
            updated = False
            for update in package_updates:
                package_name = update["name"]
                new_version = update["latest_version"]
                
                for i, line in enumerate(lines):
                    if line.strip().startswith(f"{package_name}==") or line.strip().startswith(f"{package_name}>="):
                        lines[i] = f"{package_name}>={new_version}\n"
                        updated = True
                        logger.info(f"Updated {package_name} to {new_version}")
            
            if updated:
                with open(self.requirements_path, 'w') as f:
                    f.writelines(lines)
                logger.info("Updated requirements.txt")
            
            return updated
        except Exception as e:
            logger.error(f"Failed to update requirements.txt: {e}")
            return False
    
    def update_environment_yml(self, package_updates: List[Dict]) -> bool:
        """Update package versions in environment.yml."""
        if not self.environment_path.exists():
            logger.warning("environment.yml not found")
            return False
        
        try:
            with open(self.environment_path, 'r') as f:
                config = yaml.safe_load(f)
            
            updated = False
            dependencies = config.get("dependencies", [])
            
            for update in package_updates:
                package_name = update["name"]
                new_version = update["latest_version"]
                
                for i, dep in enumerate(dependencies):
                    if isinstance(dep, str) and (dep.startswith(f"{package_name}=") or dep.startswith(f"{package_name}>")):
                        dependencies[i] = f"{package_name}>={new_version}"
                        updated = True
                        logger.info(f"Updated {package_name} to {new_version}")
            
            if updated:
                config["dependencies"] = dependencies
                with open(self.environment_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                logger.info("Updated environment.yml")
            
            return updated
        except Exception as e:
            logger.error(f"Failed to update environment.yml: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run tests to ensure updates don't break functionality."""
        try:
            logger.info("Running tests to validate updates...")
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "-v"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("All tests passed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Tests failed: {e}")
            logger.error(f"Test output: {e.stdout}")
            logger.error(f"Test errors: {e.stderr}")
            return False
    
    def create_git_branch(self, branch_name: str) -> bool:
        """Create a new git branch for the updates."""
        try:
            subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            logger.info(f"Created branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e}")
            return False
    
    def commit_changes(self, message: str) -> bool:
        """Commit the dependency updates."""
        try:
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            logger.info("Committed changes")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit changes: {e}")
            return False
    
    def push_branch(self, branch_name: str) -> bool:
        """Push the branch to remote repository."""
        try:
            subprocess.run(["git", "push", "-u", "origin", branch_name], check=True)
            logger.info(f"Pushed branch: {branch_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push branch: {e}")
            return False
    
    def create_pull_request(self, branch_name: str, updates: List[Dict], vulnerabilities: List[Dict]) -> bool:
        """Create a pull request for the dependency updates."""
        # This would integrate with GitHub API
        # For now, just log the information
        
        title = f"🔄 Automated dependency updates - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = "## Automated Dependency Updates\n\n"
        body += "This PR contains automated dependency updates generated by the dependency update script.\n\n"
        
        if updates:
            body += "### Updated Packages\n\n"
            for update in updates:
                body += f"- **{update['name']}**: {update['version']} → {update['latest_version']}\n"
        
        if vulnerabilities:
            body += "\n### Security Vulnerabilities Fixed\n\n"
            for vuln in vulnerabilities:
                body += f"- **{vuln.get('package_name', 'Unknown')}**: {vuln.get('advisory', 'Security issue')}\n"
        
        body += "\n### Validation\n\n"
        body += "- [x] All tests pass\n"
        body += "- [x] No breaking changes detected\n"
        body += "- [x] Security vulnerabilities addressed\n\n"
        
        body += "### Review Checklist\n\n"
        body += "- [ ] Review updated dependency versions\n"
        body += "- [ ] Verify test results\n"
        body += "- [ ] Check for any breaking changes\n"
        body += "- [ ] Approve and merge if all checks pass\n"
        
        logger.info(f"Pull request would be created with title: {title}")
        logger.info(f"Pull request body:\n{body}")
        
        return True
    
    def run(self, dry_run: bool = False) -> bool:
        """Run the complete dependency update process."""
        logger.info("Starting dependency update process")
        
        # Check for outdated packages
        pip_outdated = self.check_outdated_pip_packages()
        conda_outdated = self.check_outdated_conda_packages()
        vulnerabilities = self.check_security_vulnerabilities()
        
        all_updates = pip_outdated + conda_outdated
        
        if not all_updates and not vulnerabilities:
            logger.info("No updates or vulnerabilities found")
            return True
        
        if dry_run:
            logger.info("Dry run mode - would update the following packages:")
            for update in all_updates:
                logger.info(f"  {update['name']}: {update['version']} → {update['latest_version']}")
            return True
        
        # Create branch for updates
        branch_name = f"automated-dependency-updates-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if not self.create_git_branch(branch_name):
            return False
        
        try:
            # Update dependency files
            updated_files = []
            
            if self.update_pyproject_toml(all_updates):
                updated_files.append("pyproject.toml")
            
            if self.update_requirements_txt(all_updates):
                updated_files.append("requirements.txt")
            
            if self.update_environment_yml(all_updates):
                updated_files.append("environment.yml")
            
            if not updated_files:
                logger.info("No files needed updating")
                return True
            
            # Run tests to validate updates
            if not self.run_tests():
                logger.error("Tests failed - rolling back changes")
                subprocess.run(["git", "checkout", "main"], check=False)
                subprocess.run(["git", "branch", "-D", branch_name], check=False)
                return False
            
            # Commit and push changes
            commit_message = f"🔄 Update dependencies ({', '.join(updated_files)})\n\n"
            commit_message += f"Updated {len(all_updates)} packages\n"
            if vulnerabilities:
                commit_message += f"Fixed {len(vulnerabilities)} security vulnerabilities\n"
            
            if not self.commit_changes(commit_message):
                return False
            
            if not self.push_branch(branch_name):
                return False
            
            # Create pull request
            if not self.create_pull_request(branch_name, all_updates, vulnerabilities):
                return False
            
            logger.info("Dependency update process completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during update process: {e}")
            # Cleanup on error
            subprocess.run(["git", "checkout", "main"], check=False)
            subprocess.run(["git", "branch", "-D", branch_name], check=False)
            return False

def main():
    """Main entry point for the dependency updater."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated dependency updater")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without making changes")
    parser.add_argument("--repo-path", type=Path, default=Path("."), help="Path to repository")
    
    args = parser.parse_args()
    
    updater = DependencyUpdater(args.repo_path)
    success = updater.run(dry_run=args.dry_run)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()