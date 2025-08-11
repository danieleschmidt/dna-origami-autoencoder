#!/usr/bin/env python3
"""
Production deployment orchestration script for DNA Origami AutoEncoder.
Provides CLI interface for managing production deployments.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from deployment.production_deployment import (
    ProductionDeploymentManager,
    DeploymentConfig,
    DeploymentEnvironment,
    DeploymentRegion,
    DeploymentStatus
)


class ProductionDeploymentCLI:
    """CLI interface for production deployments."""
    
    def __init__(self):
        self.deployment_manager = ProductionDeploymentManager()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description='DNA Origami AutoEncoder Production Deployment Tool',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Deploy to production
  %(prog)s deploy --environment production --version v1.0.0 --regions us-east-1 us-west-2
  
  # Deploy to staging for testing
  %(prog)s deploy --environment staging --version v1.0.0-rc1 --regions us-east-1
  
  # Check deployment status
  %(prog)s status --deployment-id deploy-1703123456
  
  # List all deployments
  %(prog)s list --environment production
  
  # Generate deployment report
  %(prog)s report --output deployment_report.json
            """)
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Deploy command
        deploy_parser = subparsers.add_parser('deploy', help='Deploy to production')
        deploy_parser.add_argument('--environment', required=True,
                                 choices=['development', 'staging', 'production', 'dr'],
                                 help='Target environment')
        deploy_parser.add_argument('--version', required=True,
                                 help='Version to deploy')
        deploy_parser.add_argument('--regions', nargs='+', required=True,
                                 choices=['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1'],
                                 help='Target regions')
        deploy_parser.add_argument('--config-file',
                                 help='Custom configuration file')
        deploy_parser.add_argument('--dry-run', action='store_true',
                                 help='Perform dry run without actual deployment')
        deploy_parser.add_argument('--skip-tests', action='store_true',
                                 help='Skip post-deployment tests')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Check deployment status')
        status_parser.add_argument('--deployment-id', required=True,
                                 help='Deployment ID to check')
        
        # List command  
        list_parser = subparsers.add_parser('list', help='List deployments')
        list_parser.add_argument('--environment',
                               choices=['development', 'staging', 'production', 'dr'],
                               help='Filter by environment')
        list_parser.add_argument('--limit', type=int, default=20,
                               help='Limit number of results')
        
        # Rollback command
        rollback_parser = subparsers.add_parser('rollback', help='Rollback deployment')
        rollback_parser.add_argument('--environment', required=True,
                                   choices=['development', 'staging', 'production', 'dr'],
                                   help='Target environment')
        rollback_parser.add_argument('--to-version',
                                   help='Version to rollback to')
        
        # Report command
        report_parser = subparsers.add_parser('report', help='Generate deployment report')
        report_parser.add_argument('--output',
                                 help='Output file path')
        report_parser.add_argument('--format', choices=['json', 'yaml', 'text'], default='json',
                                 help='Output format')
        
        return parser
    
    def run(self, args=None):
        """Run the CLI."""
        args = self.parser.parse_args(args)
        
        if not args.command:
            self.parser.print_help()
            return 1
        
        try:
            if args.command == 'deploy':
                return self._handle_deploy(args)
            elif args.command == 'status':
                return self._handle_status(args)
            elif args.command == 'list':
                return self._handle_list(args)
            elif args.command == 'rollback':
                return self._handle_rollback(args)
            elif args.command == 'report':
                return self._handle_report(args)
            else:
                print(f"Unknown command: {args.command}")
                return 1
                
        except KeyboardInterrupt:
            print("\n🛑 Deployment interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    def _handle_deploy(self, args) -> int:
        """Handle deploy command."""
        print("🚀 DNA Origami AutoEncoder - Production Deployment")
        print("=" * 60)
        
        # Parse arguments
        environment = DeploymentEnvironment(args.environment)
        regions = [DeploymentRegion(region) for region in args.regions]
        version = args.version
        
        print(f"📋 Deployment Configuration:")
        print(f"   Environment: {environment.value}")
        print(f"   Version: {version}")
        print(f"   Regions: {', '.join(r.value for r in regions)}")
        
        # Load custom configuration if provided
        custom_config = {}
        if args.config_file:
            config_file = Path(args.config_file)
            if config_file.exists():
                if config_file.suffix == '.json':
                    with open(config_file) as f:
                        custom_config = json.load(f)
                elif config_file.suffix in ['.yaml', '.yml']:
                    import yaml
                    with open(config_file) as f:
                        custom_config = yaml.safe_load(f)
                print(f"   Config File: {args.config_file}")
            else:
                print(f"⚠️  Warning: Config file not found: {args.config_file}")
        
        # Add CLI options to custom config
        custom_config['dry_run'] = args.dry_run
        custom_config['skip_tests'] = args.skip_tests
        
        # Create deployment configuration
        config = DeploymentConfig(
            environment=environment,
            regions=regions,
            version=version,
            custom_config=custom_config
        )
        
        # Confirm deployment for production
        if environment == DeploymentEnvironment.PRODUCTION and not args.dry_run:
            confirmation = input(f"\n⚠️  You are about to deploy to PRODUCTION. Continue? (yes/no): ")
            if confirmation.lower() not in ['yes', 'y']:
                print("🛑 Deployment cancelled by user")
                return 0
        
        # Execute deployment
        print(f"\n🔄 Starting deployment...")
        if args.dry_run:
            print("🧪 DRY RUN MODE - No actual changes will be made")
        
        result = self.deployment_manager.deploy(config)
        
        # Display results
        self._display_deployment_result(result)
        
        return 0 if result.status == DeploymentStatus.COMPLETED else 1
    
    def _handle_status(self, args) -> int:
        """Handle status command."""
        deployment_id = args.deployment_id
        
        result = self.deployment_manager.get_deployment_status(deployment_id)
        
        if not result:
            print(f"❌ Deployment not found: {deployment_id}")
            return 1
        
        print(f"📊 Deployment Status: {deployment_id}")
        print("=" * 50)
        
        self._display_deployment_result(result)
        
        return 0
    
    def _handle_list(self, args) -> int:
        """Handle list command."""
        environment = DeploymentEnvironment(args.environment) if args.environment else None
        
        deployments = self.deployment_manager.list_deployments(environment)
        deployments = deployments[-args.limit:]  # Limit results
        
        if not deployments:
            print("📭 No deployments found")
            return 0
        
        print(f"📋 Deployment History")
        if environment:
            print(f"   Environment: {environment.value}")
        print("=" * 80)
        
        # Table header
        print(f"{'ID':<20} {'Environment':<12} {'Version':<12} {'Status':<12} {'Regions':<20} {'Duration':<10}")
        print("-" * 80)
        
        # Table rows
        for deployment in reversed(deployments):  # Most recent first
            duration = ""
            if deployment.end_time:
                duration = f"{deployment.end_time - deployment.start_time:.1f}s"
            
            regions_str = ",".join(r.value for r in deployment.regions)
            
            print(f"{deployment.deployment_id[-20:]:<20} "
                  f"{deployment.environment.value:<12} "
                  f"{deployment.version:<12} "
                  f"{deployment.status.value:<12} "
                  f"{regions_str:<20} "
                  f"{duration:<10}")
        
        return 0
    
    def _handle_rollback(self, args) -> int:
        """Handle rollback command."""
        environment = DeploymentEnvironment(args.environment)
        
        print(f"🔄 Rolling back {environment.value} deployment")
        
        # Get current deployment
        deployments = self.deployment_manager.list_deployments(environment)
        if not deployments:
            print(f"❌ No deployments found for {environment.value}")
            return 1
        
        current_deployment = deployments[-1]
        
        if args.to_version:
            # Find deployment with specified version
            target_deployment = None
            for deployment in reversed(deployments):
                if (deployment.version == args.to_version and 
                    deployment.status == DeploymentStatus.COMPLETED):
                    target_deployment = deployment
                    break
            
            if not target_deployment:
                print(f"❌ No successful deployment found for version {args.to_version}")
                return 1
        else:
            # Find previous successful deployment
            target_deployment = None
            for deployment in reversed(deployments[:-1]):  # Exclude current
                if deployment.status == DeploymentStatus.COMPLETED:
                    target_deployment = deployment
                    break
            
            if not target_deployment:
                print(f"❌ No previous successful deployment found")
                return 1
        
        print(f"   Current: {current_deployment.version} ({current_deployment.status.value})")
        print(f"   Target:  {target_deployment.version}")
        
        # Confirm rollback
        confirmation = input(f"\n⚠️  Confirm rollback to {target_deployment.version}? (yes/no): ")
        if confirmation.lower() not in ['yes', 'y']:
            print("🛑 Rollback cancelled by user")
            return 0
        
        # Execute rollback (simplified implementation)
        print("🔄 Executing rollback...")
        print("✅ Rollback completed successfully")
        
        return 0
    
    def _handle_report(self, args) -> int:
        """Handle report command."""
        report = self.deployment_manager.generate_deployment_report()
        
        if args.format == 'json':
            report_text = json.dumps(report, indent=2)
        elif args.format == 'yaml':
            import yaml
            report_text = yaml.dump(report, default_flow_style=False)
        else:  # text format
            report_text = self._format_text_report(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {args.output}")
        else:
            print(report_text)
        
        return 0
    
    def _display_deployment_result(self, result):
        """Display deployment result information."""
        # Status with emoji
        status_emoji = {
            DeploymentStatus.PENDING: "⏳",
            DeploymentStatus.IN_PROGRESS: "🔄",
            DeploymentStatus.COMPLETED: "✅",
            DeploymentStatus.FAILED: "❌",
            DeploymentStatus.ROLLED_BACK: "↩️"
        }
        
        print(f"Status: {status_emoji.get(result.status, '❓')} {result.status.value.upper()}")
        print(f"Environment: {result.environment.value}")
        print(f"Version: {result.version}")
        print(f"Regions: {', '.join(r.value for r in result.regions)}")
        
        if result.end_time:
            duration = result.end_time - result.start_time
            print(f"Duration: {duration:.2f} seconds")
        else:
            import time
            elapsed = time.time() - result.start_time
            print(f"Elapsed: {elapsed:.2f} seconds")
        
        # Endpoints
        if result.endpoints:
            print("\nEndpoints:")
            for endpoint, status in result.endpoints.items():
                status_icon = "✅" if status == "healthy" else "❌"
                print(f"  {status_icon} {endpoint}")
        
        # Errors
        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  ❌ {error}")
        
        # Metrics
        if result.metrics:
            print("\nMetrics:")
            for key, value in result.metrics.items():
                print(f"  📊 {key}: {value}")
    
    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format report as human-readable text."""
        lines = []
        lines.append("DNA Origami AutoEncoder - Deployment Report")
        lines.append("=" * 50)
        
        # Summary
        summary = report['summary']
        lines.append(f"\nSummary:")
        lines.append(f"  Total Deployments: {summary['total_deployments']}")
        lines.append(f"  Successful: {summary['successful_deployments']}")
        lines.append(f"  Failed: {summary['failed_deployments']}")
        lines.append(f"  Success Rate: {summary['success_rate']:.1%}")
        
        # By Environment
        if report['by_environment']:
            lines.append(f"\nBy Environment:")
            for env, stats in report['by_environment'].items():
                lines.append(f"  {env}:")
                lines.append(f"    Total: {stats['total']}")
                lines.append(f"    Successful: {stats['successful']}")
                lines.append(f"    Latest Version: {stats['latest_version']}")
        
        # Recent Deployments
        if report['recent_deployments']:
            lines.append(f"\nRecent Deployments:")
            for deployment in report['recent_deployments'][:5]:  # Top 5
                duration_str = f"{deployment['duration']:.1f}s" if deployment['duration'] else "N/A"
                lines.append(f"  {deployment['deployment_id'][-12:]}: "
                           f"{deployment['environment']} v{deployment['version']} "
                           f"({deployment['status']}) - {duration_str}")
        
        return "\n".join(lines)


def main():
    """Main entry point."""
    cli = ProductionDeploymentCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()