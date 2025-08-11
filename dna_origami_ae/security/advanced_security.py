"""Advanced security framework for DNA origami autoencoder system."""

import hashlib
import hmac
import secrets
import time
import json
import threading
import ipaddress
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import re
import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..utils.logger import get_logger


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ThreatLevel(Enum):
    """Threat levels for security events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source_ip: Optional[str]
    user_id: Optional[str]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class AccessAttempt:
    """Access attempt record for rate limiting."""
    ip_address: str
    user_id: Optional[str]
    timestamp: float
    success: bool
    endpoint: str
    user_agent: Optional[str] = None


class InputSanitizer:
    """Advanced input sanitization and validation."""
    
    def __init__(self):
        # Dangerous patterns that should never be allowed
        self.dangerous_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'vbscript:',   # VBScript URLs
            r'on\w+\s*=',   # Event handlers
            r'eval\s*\(',   # Eval calls
            r'expression\s*\(',  # CSS expressions
            r'\\\x[0-9a-fA-F]{2}',  # Hex escapes
            r'\\u[0-9a-fA-F]{4}',   # Unicode escapes
        ]
        
        # SQL injection patterns
        self.sql_patterns = [
            r'(\b(ALTER|CREATE|DELETE|DROP|EXEC(UTE)?|INSERT|SELECT|UNION|UPDATE)\b)',
            r'(\b(OR|AND)\s+\d+=\d+)',
            r'(\';\s*(DROP|DELETE|INSERT|UPDATE))',
            r'(\-\-|\#|\/\*)'
        ]
        
        # Command injection patterns
        self.command_patterns = [
            r'[;&|`\$\(\){}]',
            r'(wget|curl|nc|netcat|telnet|ssh)',
            r'(\.\./){2,}',  # Directory traversal
        ]
        
        self.logger = get_logger("input_sanitizer")
    
    def sanitize_string(self, input_str: str, max_length: int = 1000,
                       allow_html: bool = False, context: str = "") -> str:
        """Sanitize string input."""
        
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        
        # Length check
        if len(input_str) > max_length:
            self.logger.warning(f"Input too long: {len(input_str)} > {max_length}")
            input_str = input_str[:max_length]
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                self.logger.warning(f"Dangerous pattern detected in {context}: {pattern}")
                raise ValueError(f"Invalid input detected: potential security risk")
        
        # Check for SQL injection
        for pattern in self.sql_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                self.logger.warning(f"SQL injection attempt detected in {context}")
                raise ValueError("Invalid input: potential SQL injection")
        
        # Check for command injection
        for pattern in self.command_patterns:
            if re.search(pattern, input_str):
                self.logger.warning(f"Command injection attempt detected in {context}")
                raise ValueError("Invalid input: potential command injection")
        
        # Basic HTML encoding if not allowed
        if not allow_html:
            input_str = (input_str
                        .replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                        .replace('"', '&quot;')
                        .replace("'", '&#x27;'))
        
        return input_str
    
    def validate_dna_sequence(self, sequence: str) -> bool:
        """Validate DNA sequence input."""
        
        if not isinstance(sequence, str):
            return False
        
        # Check length (reasonable bounds)
        if len(sequence) > 100000:  # 100k bases max
            self.logger.warning(f"DNA sequence too long: {len(sequence)}")
            return False
        
        # Check valid DNA bases
        valid_bases = set('ATGC')
        sequence_upper = sequence.upper()
        
        if not all(base in valid_bases for base in sequence_upper):
            invalid_chars = set(sequence_upper) - valid_bases
            self.logger.warning(f"Invalid DNA bases detected: {invalid_chars}")
            return False
        
        return True
    
    def validate_image_data(self, image_data: Any) -> bool:
        """Validate image data input."""
        
        import numpy as np
        
        if not isinstance(image_data, np.ndarray):
            return False
        
        # Check dimensions
        if image_data.ndim not in [2, 3]:
            self.logger.warning(f"Invalid image dimensions: {image_data.ndim}")
            return False
        
        # Check size (prevent memory bombs)
        max_pixels = 10000000  # 10M pixels max
        if image_data.size > max_pixels:
            self.logger.warning(f"Image too large: {image_data.size} pixels")
            return False
        
        # Check data type and range
        if image_data.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            self.logger.warning(f"Invalid image data type: {image_data.dtype}")
            return False
        
        return True


class RateLimiter:
    """Advanced rate limiting with multiple strategies."""
    
    def __init__(self):
        self.attempts = defaultdict(deque)  # IP -> deque of timestamps
        self.user_attempts = defaultdict(deque)  # User -> deque of timestamps
        self.blocked_ips = {}  # IP -> block_until_timestamp
        self.blocked_users = {}  # User -> block_until_timestamp
        self._lock = threading.Lock()
        
        # Configuration
        self.limits = {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'login_attempts_per_hour': 10,
            'failed_attempts_threshold': 5
        }
        
        self.block_durations = {
            'temporary': 300,  # 5 minutes
            'extended': 3600,  # 1 hour
            'long': 86400     # 24 hours
        }
        
        self.logger = get_logger("rate_limiter")
    
    def check_rate_limit(self, ip_address: str, user_id: Optional[str] = None,
                        endpoint: str = "api", operation_type: str = "request") -> bool:
        """Check if request is within rate limits."""
        
        current_time = time.time()
        
        with self._lock:
            # Check if IP is blocked
            if ip_address in self.blocked_ips:
                if current_time < self.blocked_ips[ip_address]:
                    self.logger.warning(f"Blocked IP attempted access: {ip_address}")
                    return False
                else:
                    # Unblock expired blocks
                    del self.blocked_ips[ip_address]
            
            # Check if user is blocked
            if user_id and user_id in self.blocked_users:
                if current_time < self.blocked_users[user_id]:
                    self.logger.warning(f"Blocked user attempted access: {user_id}")
                    return False
                else:
                    del self.blocked_users[user_id]
            
            # Check IP rate limits
            ip_attempts = self.attempts[ip_address]
            
            # Clean old attempts (older than 1 hour)
            while ip_attempts and ip_attempts[0] < current_time - 3600:
                ip_attempts.popleft()
            
            # Check minute limit
            minute_attempts = sum(1 for t in ip_attempts if t > current_time - 60)
            if minute_attempts >= self.limits['requests_per_minute']:
                self.logger.warning(f"Rate limit exceeded (minute) for IP: {ip_address}")
                self._apply_temporary_block(ip_address, None)
                return False
            
            # Check hour limit
            if len(ip_attempts) >= self.limits['requests_per_hour']:
                self.logger.warning(f"Rate limit exceeded (hour) for IP: {ip_address}")
                self._apply_temporary_block(ip_address, None)
                return False
            
            # Check user-specific limits
            if user_id:
                user_attempts = self.user_attempts[user_id]
                
                # Clean old attempts
                while user_attempts and user_attempts[0] < current_time - 3600:
                    user_attempts.popleft()
                
                # Check user hour limit
                if len(user_attempts) >= self.limits['requests_per_hour']:
                    self.logger.warning(f"Rate limit exceeded for user: {user_id}")
                    self._apply_temporary_block(None, user_id)
                    return False
            
            return True
    
    def record_attempt(self, ip_address: str, user_id: Optional[str] = None,
                      success: bool = True, endpoint: str = "api") -> None:
        """Record access attempt."""
        
        current_time = time.time()
        
        with self._lock:
            self.attempts[ip_address].append(current_time)
            
            if user_id:
                self.user_attempts[user_id].append(current_time)
            
            # Track failed attempts for additional blocking
            if not success:
                failed_key = f"failed_{ip_address}"
                if failed_key not in self.attempts:
                    self.attempts[failed_key] = deque()
                
                self.attempts[failed_key].append(current_time)
                
                # Clean old failed attempts
                failed_attempts = self.attempts[failed_key]
                while failed_attempts and failed_attempts[0] < current_time - 3600:
                    failed_attempts.popleft()
                
                # Block after too many failures
                if len(failed_attempts) >= self.limits['failed_attempts_threshold']:
                    self.logger.warning(f"Too many failed attempts from IP: {ip_address}")
                    self._apply_extended_block(ip_address, user_id)
    
    def _apply_temporary_block(self, ip_address: Optional[str], user_id: Optional[str]):
        """Apply temporary block."""
        current_time = time.time()
        block_until = current_time + self.block_durations['temporary']
        
        if ip_address:
            self.blocked_ips[ip_address] = block_until
        
        if user_id:
            self.blocked_users[user_id] = block_until
    
    def _apply_extended_block(self, ip_address: Optional[str], user_id: Optional[str]):
        """Apply extended block."""
        current_time = time.time()
        block_until = current_time + self.block_durations['extended']
        
        if ip_address:
            self.blocked_ips[ip_address] = block_until
        
        if user_id:
            self.blocked_users[user_id] = block_until
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status."""
        current_time = time.time()
        
        with self._lock:
            active_blocks = {
                'blocked_ips': {ip: until - current_time 
                               for ip, until in self.blocked_ips.items() 
                               if until > current_time},
                'blocked_users': {user: until - current_time 
                                 for user, until in self.blocked_users.items() 
                                 if until > current_time}
            }
            
            return {
                'active_blocks': active_blocks,
                'tracked_ips': len(self.attempts),
                'tracked_users': len(self.user_attempts),
                'limits': self.limits
            }


class EncryptionManager:
    """Advanced encryption and key management."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.logger = get_logger("encryption_manager")
        
        # Generate or use provided master key
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = self._generate_master_key()
        
        # Initialize Fernet cipher
        self.cipher = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
        
        # RSA key pair for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # Key rotation
        self.key_rotation_interval = 86400  # 24 hours
        self.last_key_rotation = time.time()
        self.old_keys = []  # Store old keys for decryption
    
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key."""
        return secrets.token_bytes(32)
    
    def encrypt_data(self, data: Union[str, bytes], use_asymmetric: bool = False) -> bytes:
        """Encrypt data using symmetric or asymmetric encryption."""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if use_asymmetric:
            # Use RSA for small data
            if len(data) > 245:  # RSA 2048 limit
                raise ValueError("Data too large for RSA encryption")
            
            return self.public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        else:
            # Use Fernet for symmetric encryption
            return self.cipher.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes, use_asymmetric: bool = False) -> bytes:
        """Decrypt data."""
        
        if use_asymmetric:
            return self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        else:
            # Try current key first
            try:
                return self.cipher.decrypt(encrypted_data)
            except Exception:
                # Try old keys
                for old_cipher in self.old_keys:
                    try:
                        return old_cipher.decrypt(encrypted_data)
                    except Exception:
                        continue
                raise ValueError("Unable to decrypt data with available keys")
    
    def rotate_keys(self):
        """Rotate encryption keys."""
        
        # Store old cipher for backward compatibility
        self.old_keys.append(self.cipher)
        
        # Limit number of old keys stored
        if len(self.old_keys) > 5:
            self.old_keys.pop(0)
        
        # Generate new master key and cipher
        self.master_key = self._generate_master_key()
        self.cipher = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
        
        self.last_key_rotation = time.time()
        self.logger.info("Encryption keys rotated")
    
    def check_key_rotation(self):
        """Check if key rotation is needed."""
        if time.time() - self.last_key_rotation > self.key_rotation_interval:
            self.rotate_keys()
    
    def create_secure_hash(self, data: Union[str, bytes], salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Create secure hash with salt."""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 for secure hashing
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        hash_value = kdf.derive(data)
        return hash_value, salt
    
    def verify_hash(self, data: Union[str, bytes], hash_value: bytes, salt: bytes) -> bool:
        """Verify hash."""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        try:
            kdf.verify(data, hash_value)
            return True
        except Exception:
            return False


class SecurityAuditor:
    """Security auditing and monitoring system."""
    
    def __init__(self):
        self.events = deque(maxlen=10000)  # Store last 10k events
        self.threat_patterns = {}
        self.alert_callbacks = []
        self._lock = threading.Lock()
        
        self.logger = get_logger("security_auditor")
        
        # Initialize threat detection patterns
        self._setup_threat_patterns()
    
    def _setup_threat_patterns(self):
        """Setup threat detection patterns."""
        
        self.threat_patterns = {
            'brute_force': {
                'pattern': 'multiple_failed_logins',
                'threshold': 5,
                'timeframe': 300,  # 5 minutes
                'threat_level': ThreatLevel.HIGH
            },
            'sql_injection': {
                'pattern': 'sql_injection_attempt',
                'threshold': 1,
                'timeframe': 60,
                'threat_level': ThreatLevel.CRITICAL
            },
            'ddos': {
                'pattern': 'high_request_rate',
                'threshold': 100,
                'timeframe': 60,
                'threat_level': ThreatLevel.HIGH
            },
            'privilege_escalation': {
                'pattern': 'unauthorized_access_attempt',
                'threshold': 3,
                'timeframe': 600,
                'threat_level': ThreatLevel.CRITICAL
            }
        }
    
    def log_security_event(self, event_type: str, description: str,
                          threat_level: ThreatLevel = ThreatLevel.LOW,
                          source_ip: Optional[str] = None,
                          user_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None):
        """Log security event."""
        
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.events.append(event)
        
        self.logger.warning(f"Security event: {event_type} - {description}",
                           extra={'event_id': event.event_id, 'threat_level': threat_level.value})
        
        # Check for threat patterns
        self._check_threat_patterns(event)
    
    def _check_threat_patterns(self, event: SecurityEvent):
        """Check event against threat patterns."""
        
        current_time = time.time()
        
        for pattern_name, pattern_config in self.threat_patterns.items():
            if pattern_config['pattern'] in event.event_type:
                # Count matching events in timeframe
                timeframe_start = current_time - pattern_config['timeframe']
                
                matching_events = [
                    e for e in self.events
                    if (e.timestamp > timeframe_start and
                        pattern_config['pattern'] in e.event_type and
                        e.source_ip == event.source_ip)
                ]
                
                if len(matching_events) >= pattern_config['threshold']:
                    self._trigger_threat_alert(pattern_name, matching_events, pattern_config)
    
    def _trigger_threat_alert(self, pattern_name: str, events: List[SecurityEvent],
                             pattern_config: Dict[str, Any]):
        """Trigger threat alert."""
        
        alert_data = {
            'pattern_name': pattern_name,
            'threat_level': pattern_config['threat_level'].value,
            'event_count': len(events),
            'timeframe': pattern_config['timeframe'],
            'affected_ips': list(set(e.source_ip for e in events if e.source_ip)),
            'affected_users': list(set(e.user_id for e in events if e.user_id))
        }
        
        self.logger.critical(f"THREAT DETECTED: {pattern_name}",
                           extra=alert_data)
        
        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(pattern_name, alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback):
        """Add threat alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified timeframe."""
        
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.events if e.timestamp > cutoff_time]
        
        # Count by type and threat level
        event_types = defaultdict(int)
        threat_levels = defaultdict(int)
        source_ips = defaultdict(int)
        
        for event in recent_events:
            event_types[event.event_type] += 1
            threat_levels[event.threat_level.value] += 1
            if event.source_ip:
                source_ips[event.source_ip] += 1
        
        return {
            'timeframe_hours': hours,
            'total_events': len(recent_events),
            'event_types': dict(event_types),
            'threat_levels': dict(threat_levels),
            'top_source_ips': dict(sorted(source_ips.items(), key=lambda x: x[1], reverse=True)[:10]),
            'high_threat_events': len([e for e in recent_events if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]])
        }
    
    def export_security_log(self, format_type: str = 'json', hours: int = 24) -> str:
        """Export security log."""
        
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.events if e.timestamp > cutoff_time]
        
        if format_type.lower() == 'json':
            events_data = []
            for event in recent_events:
                events_data.append({
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'event_type': event.event_type,
                    'threat_level': event.threat_level.value,
                    'source_ip': event.source_ip,
                    'user_id': event.user_id,
                    'description': event.description,
                    'metadata': event.metadata
                })
            
            return json.dumps({
                'timeframe_hours': hours,
                'event_count': len(events_data),
                'events': events_data
            }, indent=2)
        
        else:
            return f"Unsupported format: {format_type}"


class AdvancedSecurityFramework:
    """Comprehensive security framework."""
    
    def __init__(self):
        self.input_sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter()
        self.encryption_manager = EncryptionManager()
        self.security_auditor = SecurityAuditor()
        
        self.logger = get_logger("security_framework")
        
        # Security policies
        self.security_policies = {
            'require_authentication': True,
            'require_encryption': True,
            'enable_rate_limiting': True,
            'enable_input_validation': True,
            'enable_audit_logging': True,
            'session_timeout': 3600,  # 1 hour
            'password_min_length': 8,
            'require_mfa': False  # Multi-factor authentication
        }
        
        # Active sessions
        self.active_sessions = {}
        self._sessions_lock = threading.Lock()
        
        # Setup automated security tasks
        self._start_security_tasks()
    
    def _start_security_tasks(self):
        """Start automated security maintenance tasks."""
        
        def security_maintenance():
            """Background security maintenance."""
            while True:
                try:
                    # Rotate encryption keys if needed
                    self.encryption_manager.check_key_rotation()
                    
                    # Clean expired sessions
                    self._clean_expired_sessions()
                    
                    # Sleep for 5 minutes
                    time.sleep(300)
                    
                except Exception as e:
                    self.logger.error(f"Error in security maintenance: {e}")
                    time.sleep(60)  # Wait 1 minute on error
        
        maintenance_thread = threading.Thread(target=security_maintenance, daemon=True)
        maintenance_thread.start()
    
    def validate_and_sanitize_input(self, data: Any, context: str = "",
                                   security_level: SecurityLevel = SecurityLevel.INTERNAL) -> Any:
        """Comprehensive input validation and sanitization."""
        
        if not self.security_policies['enable_input_validation']:
            return data
        
        try:
            if isinstance(data, str):
                max_length = 10000 if security_level == SecurityLevel.PUBLIC else 100000
                return self.input_sanitizer.sanitize_string(data, max_length, context=context)
            
            elif isinstance(data, dict):
                sanitized = {}
                for key, value in data.items():
                    sanitized_key = self.input_sanitizer.sanitize_string(str(key), context=f"{context}.key")
                    sanitized_value = self.validate_and_sanitize_input(value, f"{context}.{key}", security_level)
                    sanitized[sanitized_key] = sanitized_value
                return sanitized
            
            elif isinstance(data, list):
                return [self.validate_and_sanitize_input(item, f"{context}[{i}]", security_level)
                       for i, item in enumerate(data)]
            
            else:
                # For other types, perform basic validation
                if hasattr(data, '__len__') and len(str(data)) > 1000000:  # 1MB string limit
                    raise ValueError("Data too large")
                
                return data
                
        except Exception as e:
            self.security_auditor.log_security_event(
                'input_validation_failure',
                f'Input validation failed: {e}',
                ThreatLevel.MEDIUM,
                metadata={'context': context, 'data_type': type(data).__name__}
            )
            raise
    
    def check_access_permission(self, ip_address: str, user_id: Optional[str] = None,
                              endpoint: str = "api", operation_type: str = "read") -> bool:
        """Check access permissions with rate limiting."""
        
        # Check rate limits
        if self.security_policies['enable_rate_limiting']:
            if not self.rate_limiter.check_rate_limit(ip_address, user_id, endpoint, operation_type):
                self.security_auditor.log_security_event(
                    'rate_limit_exceeded',
                    f'Rate limit exceeded for {ip_address}',
                    ThreatLevel.MEDIUM,
                    source_ip=ip_address,
                    user_id=user_id
                )
                return False
        
        # Record successful access attempt
        self.rate_limiter.record_attempt(ip_address, user_id, success=True, endpoint=endpoint)
        
        return True
    
    def create_secure_session(self, user_id: str, ip_address: str,
                            additional_data: Optional[Dict[str, Any]] = None) -> str:
        """Create secure session."""
        
        session_id = secrets.token_urlsafe(32)
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'ip_address': ip_address,
            'created_at': time.time(),
            'last_activity': time.time(),
            'additional_data': additional_data or {}
        }
        
        with self._sessions_lock:
            self.active_sessions[session_id] = session_data
        
        self.security_auditor.log_security_event(
            'session_created',
            f'Session created for user {user_id}',
            ThreatLevel.LOW,
            source_ip=ip_address,
            user_id=user_id,
            metadata={'session_id': session_id}
        )
        
        return session_id
    
    def validate_session(self, session_id: str, ip_address: str) -> Optional[Dict[str, Any]]:
        """Validate session."""
        
        with self._sessions_lock:
            if session_id not in self.active_sessions:
                return None
            
            session_data = self.active_sessions[session_id]
            current_time = time.time()
            
            # Check session timeout
            if current_time - session_data['last_activity'] > self.security_policies['session_timeout']:
                del self.active_sessions[session_id]
                self.security_auditor.log_security_event(
                    'session_expired',
                    f'Session expired for user {session_data["user_id"]}',
                    ThreatLevel.LOW,
                    source_ip=ip_address,
                    user_id=session_data["user_id"]
                )
                return None
            
            # Check IP consistency (basic session hijacking protection)
            if session_data['ip_address'] != ip_address:
                self.security_auditor.log_security_event(
                    'session_ip_mismatch',
                    f'Session IP mismatch for user {session_data["user_id"]}',
                    ThreatLevel.HIGH,
                    source_ip=ip_address,
                    user_id=session_data["user_id"]
                )
                return None
            
            # Update last activity
            session_data['last_activity'] = current_time
            
            return session_data
    
    def invalidate_session(self, session_id: str):
        """Invalidate session."""
        
        with self._sessions_lock:
            if session_id in self.active_sessions:
                session_data = self.active_sessions[session_id]
                del self.active_sessions[session_id]
                
                self.security_auditor.log_security_event(
                    'session_invalidated',
                    f'Session invalidated for user {session_data["user_id"]}',
                    ThreatLevel.LOW,
                    user_id=session_data["user_id"]
                )
    
    def _clean_expired_sessions(self):
        """Clean expired sessions."""
        
        current_time = time.time()
        expired_sessions = []
        
        with self._sessions_lock:
            for session_id, session_data in self.active_sessions.items():
                if current_time - session_data['last_activity'] > self.security_policies['session_timeout']:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
        
        if expired_sessions:
            self.logger.info(f"Cleaned {len(expired_sessions)} expired sessions")
    
    def encrypt_sensitive_data(self, data: Union[str, bytes]) -> bytes:
        """Encrypt sensitive data."""
        
        if not self.security_policies['require_encryption']:
            return data if isinstance(data, bytes) else data.encode('utf-8')
        
        return self.encryption_manager.encrypt_data(data)
    
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data."""
        
        if not self.security_policies['require_encryption']:
            return encrypted_data
        
        return self.encryption_manager.decrypt_data(encrypted_data)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        
        return {
            'security_policies': self.security_policies,
            'active_sessions': len(self.active_sessions),
            'rate_limiter_status': self.rate_limiter.get_status(),
            'security_summary': self.security_auditor.get_security_summary(),
            'encryption_key_age': time.time() - self.encryption_manager.last_key_rotation
        }


# Global security framework instance
security_framework = AdvancedSecurityFramework()


def secure_endpoint(require_auth: bool = True, security_level: SecurityLevel = SecurityLevel.INTERNAL,
                   rate_limit: bool = True):
    """Decorator for securing API endpoints."""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract request context (this would be more sophisticated in a real web framework)
            ip_address = kwargs.get('ip_address', '127.0.0.1')
            session_id = kwargs.get('session_id')
            
            # Check access permissions
            if rate_limit and not security_framework.check_access_permission(
                ip_address, None, func.__name__, "api_call"
            ):
                raise PermissionError("Rate limit exceeded")
            
            # Validate session if authentication required
            if require_auth and session_id:
                session_data = security_framework.validate_session(session_id, ip_address)
                if not session_data:
                    raise PermissionError("Invalid or expired session")
                
                kwargs['session_data'] = session_data
            
            # Sanitize inputs based on security level
            sanitized_kwargs = {}
            for key, value in kwargs.items():
                if key not in ['ip_address', 'session_id', 'session_data']:
                    sanitized_kwargs[key] = security_framework.validate_and_sanitize_input(
                        value, f"param.{key}", security_level
                    )
                else:
                    sanitized_kwargs[key] = value
            
            # Log security event
            security_framework.security_auditor.log_security_event(
                'api_access',
                f'API endpoint {func.__name__} accessed',
                ThreatLevel.LOW,
                source_ip=ip_address,
                user_id=sanitized_kwargs.get('session_data', {}).get('user_id')
            )
            
            return func(*args, **sanitized_kwargs)
        
        return wrapper
    
    return decorator