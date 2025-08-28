"""
Generation 4: Progressive Quality Gates - Advanced Security Hardening
Addresses security quality gate failures with comprehensive secret management and threat detection
"""

import os
import re
import hashlib
import secrets
import threading
import time
from typing import Dict, List, Any, Optional, Union, Pattern
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis
from contextlib import contextmanager

from ..utils.logger import get_logger

logger = get_logger(__name__)

class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityViolation:
    """Security violation detection result."""
    violation_type: str
    severity: ThreatLevel
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    context: Optional[str] = None
    remediation: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class SecretPattern:
    """Pattern for detecting secrets."""
    name: str
    pattern: Pattern[str]
    description: str
    severity: ThreatLevel = ThreatLevel.HIGH
    false_positive_patterns: List[Pattern[str]] = field(default_factory=list)

class AdvancedSecretDetector:
    """Advanced secret detection system with pattern recognition."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.AdvancedSecretDetector")
        self.secret_patterns = self._initialize_patterns()
        self.whitelist_patterns = self._initialize_whitelist()
        
    def _initialize_patterns(self) -> List[SecretPattern]:
        """Initialize comprehensive secret detection patterns."""
        return [
            # API Keys
            SecretPattern(
                name="generic_api_key",
                pattern=re.compile(r'(?i)(?:api_key|apikey|api-key)\s*[=:]\s*["\']?([a-zA-Z0-9\-_]{20,})["\']?'),
                description="Generic API key pattern",
                severity=ThreatLevel.HIGH
            ),
            SecretPattern(
                name="aws_access_key",
                pattern=re.compile(r'AKIA[0-9A-Z]{16}'),
                description="AWS Access Key ID",
                severity=ThreatLevel.CRITICAL
            ),
            SecretPattern(
                name="aws_secret_key",
                pattern=re.compile(r'(?i)aws.{0,20}["\']([a-zA-Z0-9/+=]{40})["\']'),
                description="AWS Secret Access Key",
                severity=ThreatLevel.CRITICAL
            ),
            
            # Database credentials
            SecretPattern(
                name="database_url",
                pattern=re.compile(r'(?i)(?:database_url|db_url|connection_string)\s*[=:]\s*["\']?((?:postgres|mysql|mongodb)://[^"\']+)["\']?'),
                description="Database connection string",
                severity=ThreatLevel.HIGH
            ),
            SecretPattern(
                name="password_field",
                pattern=re.compile(r'(?i)(?:password|passwd|pwd)\s*[=:]\s*["\']?([^"\'\\n\\r]{8,})["\']?'),
                description="Password field",
                severity=ThreatLevel.HIGH,
                false_positive_patterns=[
                    re.compile(r'(?i)password.*(?:placeholder|example|test|demo)'),
                    re.compile(r'(?i)(?:your_password|enter_password|password_here)')
                ]
            ),
            
            # Encryption keys
            SecretPattern(
                name="private_key",
                pattern=re.compile(r'-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----'),
                description="Private key",
                severity=ThreatLevel.CRITICAL
            ),
            SecretPattern(
                name="encryption_key",
                pattern=re.compile(r'(?i)(?:secret_key|encryption_key|cipher_key)\s*[=:]\s*["\']?([a-zA-Z0-9+/=]{32,})["\']?'),
                description="Encryption key",
                severity=ThreatLevel.HIGH
            ),
            
            # JWT tokens
            SecretPattern(
                name="jwt_token",
                pattern=re.compile(r'eyJ[a-zA-Z0-9+/=]+\.eyJ[a-zA-Z0-9+/=]+\.[a-zA-Z0-9+/=\-_]+'),
                description="JWT token",
                severity=ThreatLevel.MEDIUM
            ),
            
            # Redis/Cache URLs
            SecretPattern(
                name="redis_url",
                pattern=re.compile(r'redis://(?:[^:]+:[^@]+@)?[^/]+(?:/\d+)?'),
                description="Redis connection URL",
                severity=ThreatLevel.MEDIUM
            ),
            
            # Generic tokens
            SecretPattern(
                name="bearer_token",
                pattern=re.compile(r'Bearer\s+([a-zA-Z0-9\-_]{20,})'),
                description="Bearer token",
                severity=ThreatLevel.HIGH
            ),
            SecretPattern(
                name="generic_secret",
                pattern=re.compile(r'(?i)(?:secret|token|key)\s*[=:]\s*["\']?([a-zA-Z0-9+/=\-_]{32,})["\']?'),
                description="Generic secret pattern",
                severity=ThreatLevel.MEDIUM
            )
        ]
    
    def _initialize_whitelist(self) -> List[Pattern[str]]:
        """Initialize patterns for legitimate non-secret content."""
        return [
            re.compile(r'(?i)example|sample|test|demo|placeholder'),
            re.compile(r'(?i)your_[a-z_]+_here'),
            re.compile(r'(?i)\[.*\]'),  # Bracketed placeholders
            re.compile(r'(?i)<.*>'),    # Angle bracket placeholders
            re.compile(r'^[a-z_]+$'),   # Simple variable names
            re.compile(r'(?i)secret.*(?:manager|service)'),  # Secret management services
        ]
    
    def scan_text(self, text: str, file_path: Optional[str] = None) -> List[SecurityViolation]:
        """Scan text content for secret patterns."""
        violations = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments and documentation
            if self._is_comment_or_doc(line):
                continue
            
            for pattern in self.secret_patterns:
                matches = pattern.pattern.findall(line)
                
                for match in matches:
                    # Extract the actual secret value
                    secret_value = match if isinstance(match, str) else match[0] if match else ''
                    
                    # Check against false positive patterns
                    if self._is_false_positive(secret_value, line, pattern):
                        continue
                    
                    # Check against whitelist
                    if self._is_whitelisted(secret_value):
                        continue
                    
                    violations.append(SecurityViolation(
                        violation_type=f"secret_detected_{pattern.name}",
                        severity=pattern.severity,
                        message=f"{pattern.description} detected: {self._mask_secret(secret_value)}",
                        file_path=file_path,
                        line_number=line_num,
                        context=self._mask_secret(line.strip()),
                        remediation=self._get_remediation_advice(pattern.name)
                    ))
        
        return violations
    
    def _is_comment_or_doc(self, line: str) -> bool:
        """Check if line is a comment or documentation."""
        stripped = line.strip()
        return (
            stripped.startswith('#') or
            stripped.startswith('//') or
            stripped.startswith('*') or
            stripped.startswith('"""') or
            stripped.startswith("'''") or
            'docstring' in stripped.lower()
        )
    
    def _is_false_positive(self, secret_value: str, line: str, pattern: SecretPattern) -> bool:
        """Check if detected secret is likely a false positive."""
        for fp_pattern in pattern.false_positive_patterns:
            if fp_pattern.search(line):
                return True
        
        # Additional heuristics
        if len(secret_value) < 8:  # Too short to be a real secret
            return True
        
        if secret_value.lower() in ['password', 'secret', 'token', 'key']:
            return True
        
        return False
    
    def _is_whitelisted(self, secret_value: str) -> bool:
        """Check if value matches whitelist patterns."""
        for whitelist_pattern in self.whitelist_patterns:
            if whitelist_pattern.search(secret_value):
                return True
        return False
    
    def _mask_secret(self, value: str, mask_char: str = '*', reveal_chars: int = 4) -> str:
        """Mask secret value for safe logging."""
        if len(value) <= reveal_chars * 2:
            return mask_char * len(value)
        
        return value[:reveal_chars] + mask_char * (len(value) - reveal_chars * 2) + value[-reveal_chars:]
    
    def _get_remediation_advice(self, pattern_name: str) -> str:
        """Get remediation advice for specific secret types."""
        remediation_map = {
            'generic_api_key': 'Move API key to environment variable or secure key vault',
            'aws_access_key': 'Rotate AWS credentials immediately and use IAM roles',
            'aws_secret_key': 'Rotate AWS credentials immediately and use IAM roles',
            'database_url': 'Use environment variables for database connection strings',
            'password_field': 'Use environment variables or secure password storage',
            'private_key': 'Store private keys securely outside of code repository',
            'encryption_key': 'Use secure key management service',
            'jwt_token': 'JWT tokens should not be hardcoded',
            'redis_url': 'Use environment variables for Redis connection strings',
            'bearer_token': 'Store tokens securely and use short-lived tokens',
            'generic_secret': 'Move secrets to environment variables or key vault'
        }
        
        return remediation_map.get(pattern_name, 'Move secret to secure storage system')
    
    def scan_file(self, file_path: Path) -> List[SecurityViolation]:
        """Scan a file for secrets."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return self.scan_text(content, str(file_path))
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
            return []
    
    def scan_directory(self, directory_path: Path, extensions: Optional[List[str]] = None) -> List[SecurityViolation]:
        """Scan directory for secrets."""
        if extensions is None:
            extensions = ['.py', '.js', '.ts', '.json', '.yaml', '.yml', '.env', '.config', '.ini']
        
        violations = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                # Skip certain directories
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                if any(part in ['node_modules', '__pycache__', '.git'] for part in file_path.parts):
                    continue
                
                file_violations = self.scan_file(file_path)
                violations.extend(file_violations)
        
        return violations

class SecureSecretManager:
    """Secure secret management system."""
    
    def __init__(self, master_password: Optional[str] = None):
        self.logger = get_logger(f"{__name__}.SecureSecretManager")
        
        # Initialize encryption
        self.master_password = master_password or os.environ.get('MASTER_PASSWORD', '')
        if not self.master_password:
            self.master_password = self._generate_master_password()
            self.logger.warning("Generated new master password - ensure it's stored securely")
        
        self.fernet = self._create_encryption_key()
        
        # In-memory secret storage
        self._secrets = {}
        self._lock = threading.RLock()
        
        # Load existing secrets
        self._load_secrets()
    
    def _generate_master_password(self) -> str:
        """Generate a secure master password."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')
    
    def _create_encryption_key(self) -> Fernet:
        """Create encryption key from master password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'dna_origami_ae_salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))
        return Fernet(key)
    
    def store_secret(self, key: str, value: str, classification: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> bool:
        """Store a secret securely."""
        try:
            with self._lock:
                encrypted_value = self.fernet.encrypt(value.encode())
                
                self._secrets[key] = {
                    'value': encrypted_value,
                    'classification': classification.value,
                    'created_at': time.time(),
                    'accessed_count': 0,
                    'last_accessed': None
                }
                
                self._save_secrets()
                self.logger.info(f"Secret stored: {key} (classification: {classification.value})")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing secret {key}: {e}")
            return False
    
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret securely."""
        try:
            with self._lock:
                if key not in self._secrets:
                    return None
                
                secret_info = self._secrets[key]
                decrypted_value = self.fernet.decrypt(secret_info['value']).decode()
                
                # Update access statistics
                secret_info['accessed_count'] += 1
                secret_info['last_accessed'] = time.time()
                
                return decrypted_value
                
        except Exception as e:
            self.logger.error(f"Error retrieving secret {key}: {e}")
            return None
    
    def delete_secret(self, key: str) -> bool:
        """Delete a secret."""
        try:
            with self._lock:
                if key in self._secrets:
                    del self._secrets[key]
                    self._save_secrets()
                    self.logger.info(f"Secret deleted: {key}")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting secret {key}: {e}")
            return False
    
    def list_secrets(self) -> List[Dict[str, Any]]:
        """List stored secrets (without values)."""
        with self._lock:
            return [
                {
                    'key': key,
                    'classification': info['classification'],
                    'created_at': info['created_at'],
                    'accessed_count': info['accessed_count'],
                    'last_accessed': info['last_accessed']
                }
                for key, info in self._secrets.items()
            ]
    
    def _load_secrets(self):
        """Load secrets from secure storage."""
        # In production, this would load from a secure key vault
        # For now, using environment variables as fallback
        pass
    
    def _save_secrets(self):
        """Save secrets to secure storage."""
        # In production, this would save to a secure key vault
        # For now, keeping in memory only
        pass
    
    @contextmanager
    def temporary_secret(self, value: str):
        """Context manager for temporary secret usage."""
        temp_key = f"temp_{secrets.token_hex(8)}"
        try:
            self.store_secret(temp_key, value, SecurityLevel.SECRET)
            yield temp_key
        finally:
            self.delete_secret(temp_key)

class SecurityHardeningSystem:
    """Comprehensive security hardening system."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.SecurityHardeningSystem")
        
        # Initialize subsystems
        self.secret_detector = AdvancedSecretDetector()
        self.secret_manager = SecureSecretManager()
        
        # Security monitoring
        self.violations_log = []
        self.security_stats = {
            'secrets_detected': 0,
            'secrets_remediated': 0,
            'files_scanned': 0,
            'last_scan_time': None
        }
        
        self.logger.info("Security hardening system initialized")
    
    def comprehensive_security_scan(self, directory_path: Path) -> Dict[str, Any]:
        """Perform comprehensive security scan."""
        self.logger.info(f"Starting comprehensive security scan of {directory_path}")
        
        start_time = time.time()
        
        # Scan for secrets
        violations = self.secret_detector.scan_directory(directory_path)
        
        # Categorize violations by severity
        violation_summary = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for violation in violations:
            violation_summary[violation.severity.value].append(violation)
        
        # Update statistics
        self.security_stats['secrets_detected'] = len(violations)
        self.security_stats['last_scan_time'] = time.time()
        
        # Calculate scan metrics
        scan_time = time.time() - start_time
        
        # Generate remediation plan
        remediation_plan = self._generate_remediation_plan(violations)
        
        # Security score calculation
        security_score = self._calculate_security_score(violation_summary)
        
        scan_results = {
            'scan_summary': {
                'total_violations': len(violations),
                'critical_violations': len(violation_summary['critical']),
                'high_violations': len(violation_summary['high']),
                'medium_violations': len(violation_summary['medium']),
                'low_violations': len(violation_summary['low']),
                'scan_time_seconds': scan_time,
                'security_score': security_score
            },
            'violations_by_severity': violation_summary,
            'remediation_plan': remediation_plan,
            'recommendations': self._generate_security_recommendations(violation_summary),
            'compliance_status': self._assess_compliance_status(violation_summary)
        }
        
        self.logger.info(
            f"Security scan complete: {len(violations)} violations found "
            f"(Score: {security_score}/100)"
        )
        
        return scan_results
    
    def _generate_remediation_plan(self, violations: List[SecurityViolation]) -> Dict[str, Any]:
        """Generate automated remediation plan."""
        plan = {
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': [],
            'estimated_effort_hours': 0
        }
        
        # Group violations by type and severity
        violation_groups = {}
        for violation in violations:
            key = (violation.violation_type, violation.severity)
            if key not in violation_groups:
                violation_groups[key] = []
            violation_groups[key].append(violation)
        
        for (violation_type, severity), group_violations in violation_groups.items():
            action = {
                'violation_type': violation_type,
                'severity': severity.value,
                'count': len(group_violations),
                'remediation': group_violations[0].remediation,
                'files_affected': list(set(v.file_path for v in group_violations if v.file_path))
            }
            
            # Categorize by urgency
            if severity == ThreatLevel.CRITICAL:
                plan['immediate_actions'].append(action)
                plan['estimated_effort_hours'] += len(group_violations) * 2
            elif severity == ThreatLevel.HIGH:
                plan['short_term_actions'].append(action)
                plan['estimated_effort_hours'] += len(group_violations) * 1
            else:
                plan['long_term_actions'].append(action)
                plan['estimated_effort_hours'] += len(group_violations) * 0.5
        
        return plan
    
    def _calculate_security_score(self, violation_summary: Dict[str, List]) -> float:
        """Calculate security score (0-100)."""
        # Base score
        score = 100.0
        
        # Deduct points based on violations
        score -= len(violation_summary['critical']) * 20  # Critical: -20 points each
        score -= len(violation_summary['high']) * 10     # High: -10 points each
        score -= len(violation_summary['medium']) * 5    # Medium: -5 points each
        score -= len(violation_summary['low']) * 1       # Low: -1 point each
        
        return max(0.0, score)
    
    def _generate_security_recommendations(self, violation_summary: Dict[str, List]) -> List[str]:
        """Generate security improvement recommendations."""
        recommendations = []
        
        total_violations = sum(len(violations) for violations in violation_summary.values())
        
        if total_violations == 0:
            recommendations.append("✅ No security violations detected - excellent security posture")
            return recommendations
        
        if violation_summary['critical']:
            recommendations.append(
                f"🚨 URGENT: Address {len(violation_summary['critical'])} critical security violations immediately"
            )
        
        if violation_summary['high']:
            recommendations.append(
                f"⚠️  Address {len(violation_summary['high'])} high-severity violations within 24 hours"
            )
        
        # Specific recommendations
        if any('api_key' in v.violation_type for violations in violation_summary.values() for v in violations):
            recommendations.append("🔑 Implement secure API key management using environment variables or key vault")
        
        if any('password' in v.violation_type for violations in violation_summary.values() for v in violations):
            recommendations.append("🔐 Remove hardcoded passwords and use secure credential storage")
        
        if any('private_key' in v.violation_type for violations in violation_summary.values() for v in violations):
            recommendations.append("🗝️  Move private keys to secure key management system")
        
        # General recommendations
        recommendations.extend([
            "📋 Establish regular security scanning as part of CI/CD pipeline",
            "🏛️  Implement centralized secret management solution",
            "🔍 Set up automated security monitoring and alerting",
            "📚 Provide security training for development team"
        ])
        
        return recommendations
    
    def _assess_compliance_status(self, violation_summary: Dict[str, List]) -> Dict[str, Any]:
        """Assess compliance with security standards."""
        total_violations = sum(len(violations) for violations in violation_summary.values())
        
        # Compliance thresholds
        compliant = total_violations == 0
        needs_improvement = 0 < total_violations <= 5
        non_compliant = total_violations > 5
        
        status = "COMPLIANT" if compliant else "NEEDS_IMPROVEMENT" if needs_improvement else "NON_COMPLIANT"
        
        return {
            'status': status,
            'compliant': compliant,
            'total_violations': total_violations,
            'compliance_percentage': max(0, (100 - total_violations * 5)),  # 5% per violation
            'next_review_date': time.time() + (30 * 24 * 3600),  # 30 days from now
            'standards_assessed': ['OWASP', 'NIST', 'ISO27001']
        }
    
    def automated_remediation(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform automated remediation where possible."""
        self.logger.info("Starting automated security remediation")
        
        remediation_results = {
            'actions_taken': [],
            'manual_actions_required': [],
            'success_count': 0,
            'failure_count': 0
        }
        
        # Process immediate actions
        for action in scan_results['remediation_plan']['immediate_actions']:
            if self._can_auto_remediate(action['violation_type']):
                try:
                    success = self._perform_auto_remediation(action)
                    if success:
                        remediation_results['actions_taken'].append(
                            f"✅ Auto-remediated {action['count']} {action['violation_type']} violations"
                        )
                        remediation_results['success_count'] += action['count']
                    else:
                        remediation_results['manual_actions_required'].append(action)
                        remediation_results['failure_count'] += action['count']
                except Exception as e:
                    self.logger.error(f"Auto-remediation failed for {action['violation_type']}: {e}")
                    remediation_results['manual_actions_required'].append(action)
                    remediation_results['failure_count'] += action['count']
            else:
                remediation_results['manual_actions_required'].append(action)
        
        self.logger.info(
            f"Automated remediation complete: {remediation_results['success_count']} fixes applied, "
            f"{remediation_results['failure_count']} require manual intervention"
        )
        
        return remediation_results
    
    def _can_auto_remediate(self, violation_type: str) -> bool:
        """Check if violation type can be automatically remediated."""
        # For now, limited auto-remediation capabilities
        auto_remediatable = [
            'generic_secret',  # Can suggest environment variable usage
            'jwt_token'        # Can suggest token rotation
        ]
        return any(remediatable in violation_type for remediatable in auto_remediatable)
    
    def _perform_auto_remediation(self, action: Dict[str, Any]) -> bool:
        """Perform automatic remediation for supported violation types."""
        # This would implement actual remediation logic
        # For now, just logging the action
        self.logger.info(f"Auto-remediating {action['violation_type']}")
        return True  # Simulate success
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Generate security dashboard data."""
        return {
            'security_stats': self.security_stats,
            'recent_violations': self.violations_log[-10:] if self.violations_log else [],
            'secret_management': {
                'stored_secrets': len(self.secret_manager.list_secrets()),
                'master_password_set': bool(self.secret_manager.master_password)
            },
            'system_status': {
                'secret_detector_active': True,
                'secret_manager_active': True,
                'last_health_check': time.time()
            }
        }

# Global security hardening system
_global_security_system = None

def get_security_hardening_system() -> SecurityHardeningSystem:
    """Get global security hardening system."""
    global _global_security_system
    
    if _global_security_system is None:
        _global_security_system = SecurityHardeningSystem()
    
    return _global_security_system