# Manual Setup Requirements

Due to GitHub App permission limitations, the following setup steps must be performed manually by repository maintainers after the SDLC implementation is complete.

## Required GitHub Actions Workflows

The following workflow files must be manually created in `.github/workflows/` using the templates provided in `docs/workflows/examples/`:

### 1. Continuous Integration (`ci.yml`)
**Source**: `docs/workflows/examples/ci.yml`
**Destination**: `.github/workflows/ci.yml`

**Purpose**: Automated testing, code quality checks, and security scanning for all pull requests and pushes.

### 2. Continuous Deployment (`cd.yml`)
**Source**: `docs/workflows/examples/cd.yml` (to be created)
**Destination**: `.github/workflows/cd.yml`

**Purpose**: Automated deployment to staging and production environments.

### 3. Dependency Updates (`dependency-update.yml`)
**Source**: `docs/workflows/examples/dependency-update.yml` (to be created)
**Destination**: `.github/workflows/dependency-update.yml`

**Purpose**: Automated dependency updates and security patch management.

### 4. Security Scanning (`security-scan.yml`)
**Source**: `docs/workflows/examples/security-scan.yml` (to be created)
**Destination**: `.github/workflows/security-scan.yml`

**Purpose**: Comprehensive security scanning including SAST, dependency scanning, and container scanning.

## Required Repository Settings

### 1. Branch Protection Rules
Configure the following protection rules for the `main` branch:

```
Settings > Branches > Add rule

Branch name pattern: main

Protect matching branches:
☑ Require a pull request before merging
  ☑ Require approvals: 1
  ☑ Dismiss stale PR approvals when new commits are pushed
  ☑ Require review from code owners
☑ Require status checks to pass before merging
  ☑ Require branches to be up to date before merging
  Status checks (add when workflows are active):
    - CI / Code Quality & Security
    - CI / Test Python 3.11
    - CI / Integration Tests
    - CI / Docker Build & Test
☑ Require conversation resolution before merging
☑ Require signed commits
☑ Require linear history
☑ Include administrators
```

### 2. Repository Secrets
Configure the following secrets in `Settings > Secrets and variables > Actions`:

**Required Secrets:**
- `CODECOV_TOKEN` - For test coverage reporting
- `DOCKERHUB_USERNAME` - Docker Hub username
- `DOCKERHUB_TOKEN` - Docker Hub access token

**Optional Secrets (for enhanced functionality):**
- `SLACK_WEBHOOK_URL` - Slack notifications
- `SONAR_TOKEN` - SonarCloud integration
- `AWS_ACCESS_KEY_ID` - AWS deployment
- `AWS_SECRET_ACCESS_KEY` - AWS deployment
- `AZURE_CREDENTIALS` - Azure deployment
- `GCP_SA_KEY` - Google Cloud deployment

### 3. Repository Variables
Configure the following variables in `Settings > Secrets and variables > Actions`:

- `DOCKER_REGISTRY` - `ghcr.io` (or your preferred registry)
- `STAGING_URL` - Staging environment URL
- `PRODUCTION_URL` - Production environment URL

### 4. Repository Settings
Update the following repository settings:

**General Settings:**
- Description: "A groundbreaking wet-lab ML framework that encodes images into self-assembling DNA origami structures"
- Website: (your project website)
- Topics: `dna`, `origami`, `machine-learning`, `biocomputing`, `synthetic-biology`, `nanotechnology`, `information-storage`, `neural-networks`, `molecular-dynamics`, `bioinformatics`

**Features:**
- ☑ Wikis
- ☑ Issues
- ☑ Sponsorships
- ☑ Projects
- ☑ Preserve this repository

**Security:**
- ☑ Private vulnerability reporting
- ☑ Dependency graph
- ☑ Dependabot alerts
- ☑ Dependabot security updates
- ☑ Dependabot version updates

## Required GitHub Apps and Integrations

### 1. Codecov Integration
1. Visit https://codecov.io/
2. Sign in with GitHub account
3. Add the repository
4. Copy the upload token to `CODECOV_TOKEN` secret

### 2. Dependabot Configuration
Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    
  - package-ecosystem: "github-actions"
    directory: "/.github/workflows"
    schedule:
      interval: "weekly"
```

### 3. Security Scanning (Optional)
Consider integrating with:
- **SonarCloud**: Code quality and security analysis
- **Snyk**: Vulnerability scanning
- **CodeQL**: GitHub's built-in code analysis

## Manual Quality Assurance Setup

### 1. Issue Templates
The following issue templates have been created:
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`

### 2. Pull Request Template
A comprehensive PR template has been created:
- `.github/PULL_REQUEST_TEMPLATE.md`

### 3. Code Owners
The `CODEOWNERS` file has been configured to ensure proper code review.

## Development Environment Setup

### 1. Pre-commit Hooks
Developers should run:
```bash
pre-commit install
```

### 2. Development Dependencies
Install development dependencies:
```bash
pip install -e ".[dev]"
```

### 3. IDE Configuration
VS Code settings have been configured in `.vscode/settings.json`.
Dev container configuration is available in `.devcontainer/`.

## Monitoring and Observability

### 1. Metrics Dashboard
- Configure Grafana dashboard (if using)
- Set up Prometheus metrics collection
- Configure alerting rules

### 2. Log Aggregation
- Set up centralized logging (ELK stack, CloudWatch, etc.)
- Configure log retention policies

## Deployment Pipeline

### 1. Environment Configuration
Set up the following environments:
- **Development**: Local development environment
- **Staging**: Pre-production testing environment  
- **Production**: Live production environment

### 2. Infrastructure as Code
Consider using:
- **Terraform** for infrastructure provisioning
- **Ansible** for configuration management
- **Kubernetes** for container orchestration

## Performance Monitoring

### 1. Application Performance Monitoring (APM)
Consider integrating:
- **New Relic**
- **Datadog**
- **Application Insights**

### 2. Synthetic Monitoring
Set up synthetic tests for:
- API endpoints
- Critical user journeys
- Performance benchmarks

## Security Hardening

### 1. Secret Scanning
Enable GitHub's secret scanning and push protection.

### 2. Container Security
- Implement container image scanning
- Use minimal base images
- Regular security updates

### 3. Access Control
- Implement least privilege access
- Regular access reviews
- Multi-factor authentication

## Documentation Maintenance

### 1. API Documentation
- Auto-generate API docs from code
- Keep examples up to date
- Version documentation with releases

### 2. User Guides
- Create getting started guides
- Maintain troubleshooting documentation
- Update deployment guides

## Compliance and Governance

### 1. License Compliance
- Audit dependency licenses
- Maintain license compatibility
- Document license requirements

### 2. Data Governance
- Implement data classification
- Configure data retention policies
- Ensure GDPR/privacy compliance

## Checklist for Repository Maintainers

- [ ] Create all GitHub Actions workflows from templates
- [ ] Configure branch protection rules
- [ ] Set up repository secrets and variables
- [ ] Configure Dependabot
- [ ] Integrate with Codecov
- [ ] Set up monitoring and alerting
- [ ] Configure deployment environments
- [ ] Test all workflows and automation
- [ ] Document any environment-specific configurations
- [ ] Train team members on new processes

## Support and Troubleshooting

If you encounter issues during setup:

1. **Check the logs**: Review GitHub Actions logs for workflow issues
2. **Verify permissions**: Ensure repository settings allow the required operations
3. **Test incrementally**: Enable features one at a time to isolate issues
4. **Consult documentation**: Reference GitHub Actions and tool-specific documentation
5. **Contact support**: Reach out to the development team for assistance

## Maintenance Schedule

- **Weekly**: Review and merge dependency updates
- **Monthly**: Review security scanning results
- **Quarterly**: Update workflow templates and configurations
- **Annually**: Comprehensive security and compliance review