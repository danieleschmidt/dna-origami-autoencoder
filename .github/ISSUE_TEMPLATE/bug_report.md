---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''

---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Environment Information
**System Information:**
- OS: [e.g. Ubuntu 22.04, Windows 11, macOS 13.0]
- Python version: [e.g. 3.11.0]
- DNA-Origami-AutoEncoder version: [e.g. 0.1.0]
- GPU: [e.g. NVIDIA RTX 4090, None]
- CUDA version: [e.g. 12.1, N/A]

**Installation Method:**
- [ ] pip install
- [ ] conda install
- [ ] Development installation (pip install -e .)
- [ ] Docker
- [ ] Other: ___________

**Dependencies:**
```bash
# Output of: pip list | grep -E "(torch|numpy|scipy|biopython)"
# or paste your requirements.txt/environment.yml
```

## Error Logs and Stack Traces
```python
# Paste the complete error traceback here
```

## Screenshots
If applicable, add screenshots to help explain your problem.

## Minimal Reproducible Example
Please provide a minimal code example that reproduces the issue:

```python
# Minimal code that reproduces the bug
from dna_origami_ae import ...

# Your code here
```

## Additional Context
Add any other context about the problem here.

## Data Files (if applicable)
If your bug involves specific data files:
- [ ] I can share the data files publicly
- [ ] I need to share data files privately
- [ ] The bug occurs with synthetic/example data

## Impact Assessment
**Severity:**
- [ ] Critical - System unusable, data loss, security issue
- [ ] High - Major functionality broken, significant performance impact
- [ ] Medium - Minor functionality broken, workaround available
- [ ] Low - Cosmetic issue, minimal impact

**User Impact:**
- [ ] Affects all users
- [ ] Affects users with specific configuration
- [ ] Affects only development environment
- [ ] Affects specific use cases: ___________

## Checklist
- [ ] I have searched for similar issues
- [ ] I have tested with the latest version
- [ ] I have provided a minimal reproducible example
- [ ] I have included relevant error messages and logs
- [ ] I have filled out the environment information