"""Security utilities and input validation for DNA origami autoencoder."""

import hashlib
import hmac
import secrets
import re
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import time
from functools import wraps
from pathlib import Path
import json

from .logger import get_logger, dna_logger


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 10
    
    # Input validation
    max_file_size_mb: int = 100
    max_sequence_length: int = 50000
    max_image_dimensions: tuple = (2048, 2048)
    
    # Authentication
    session_timeout_minutes: int = 60
    max_sessions_per_user: int = 5
    
    # Content filtering
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.dna', '.fasta', '.json'
    ])
    
    # Security headers
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'"
    })


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize rate limiter."""
        self.config = config
        self.requests = {}  # client_id -> [(timestamp, count)]
        self.logger = get_logger('security')
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed under rate limits."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                (ts, count) for ts, count in self.requests[client_id]
                if ts > minute_ago
            ]
        else:
            self.requests[client_id] = []
        
        # Count requests in last minute
        total_requests = sum(count for _, count in self.requests[client_id])
        
        if total_requests >= self.config.rate_limit_requests_per_minute:
            dna_logger.log_security_event(
                'rate_limit_exceeded',
                {'client_id': client_id, 'requests': total_requests},
                'WARN'
            )
            return False
        
        # Add current request
        self.requests[client_id].append((current_time, 1))
        return True


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize input validator."""
        self.config = config
        self.logger = get_logger('security')
    
    def validate_dna_sequence(self, sequence: str) -> Dict[str, Any]:
        """Validate DNA sequence input."""
        result = {'valid': True, 'errors': [], 'sanitized': None}
        
        if not isinstance(sequence, str):
            result['valid'] = False
            result['errors'].append("Sequence must be a string")
            return result
        
        # Basic sanitization
        sanitized = sequence.upper().strip()
        
        # Length check
        if len(sanitized) > self.config.max_sequence_length:
            result['valid'] = False
            result['errors'].append(f"Sequence too long: {len(sanitized)} > {self.config.max_sequence_length}")
        
        # Valid bases check
        valid_bases = set('ATGCNRYWSMKBDHV-')  # Including ambiguous bases
        invalid_chars = set(sanitized) - valid_bases
        if invalid_chars:
            result['valid'] = False
            result['errors'].append(f"Invalid characters: {invalid_chars}")
        
        # Remove common injection patterns
        dangerous_patterns = ['<script', 'javascript:', 'data:', '<?php', '<%', '{{', '${']
        for pattern in dangerous_patterns:
            if pattern.lower() in sanitized.lower():
                result['valid'] = False
                result['errors'].append("Potentially malicious content detected")
                break
        
        if result['valid']:
            result['sanitized'] = sanitized
        
        return result
    
    def validate_file_upload(self, filename: str, file_size: int, file_content: bytes = None) -> Dict[str, Any]:
        """Validate file upload."""
        result = {'valid': True, 'errors': [], 'sanitized_filename': None}
        
        # Filename sanitization
        safe_filename = re.sub(r'[^\w\.\-_]', '_', filename)
        safe_filename = safe_filename[:100]  # Limit length
        
        # Extension validation
        file_ext = Path(safe_filename).suffix.lower()
        if file_ext not in self.config.allowed_file_extensions:
            result['valid'] = False
            result['errors'].append(f"File extension not allowed: {file_ext}")
        
        # Size validation
        if file_size > self.config.max_file_size_mb * 1024 * 1024:
            result['valid'] = False
            result['errors'].append(f"File too large: {file_size} bytes")
        
        # Content validation (basic)
        if file_content:
            # Check for potential malware signatures
            malware_signatures = [b'MZ', b'PK\x03\x04', b'#!/bin/sh']
            for sig in malware_signatures:
                if file_content.startswith(sig):
                    result['valid'] = False
                    result['errors'].append("Potentially malicious file content")
                    break
        
        if result['valid']:
            result['sanitized_filename'] = safe_filename
        
        return result
    
    def validate_image_parameters(self, width: int, height: int, channels: int = 3) -> Dict[str, Any]:
        """Validate image parameters."""
        result = {'valid': True, 'errors': []}
        
        max_width, max_height = self.config.max_image_dimensions
        
        if width <= 0 or width > max_width:
            result['valid'] = False
            result['errors'].append(f"Invalid width: {width}")
        
        if height <= 0 or height > max_height:
            result['valid'] = False
            result['errors'].append(f"Invalid height: {height}")
        
        if channels not in [1, 3, 4]:
            result['valid'] = False
            result['errors'].append(f"Invalid channels: {channels}")
        
        return result
    
    def sanitize_json_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize JSON input data."""
        def sanitize_value(value):
            if isinstance(value, str):
                # Remove potential XSS
                sanitized = re.sub(r'<[^>]*>', '', value)
                sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
                return sanitized
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            else:
                return value
        
        return sanitize_value(data)


class SessionManager:
    """Secure session management."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize session manager."""
        self.config = config
        self.sessions = {}  # session_id -> session_data
        self.user_sessions = {}  # user_id -> [session_ids]
        self.logger = get_logger('security')
        self.secret_key = secrets.token_hex(32)
    
    def create_session(self, user_id: str, additional_data: Dict[str, Any] = None) -> str:
        """Create new secure session."""
        # Check session limits
        user_sessions = self.user_sessions.get(user_id, [])
        if len(user_sessions) >= self.config.max_sessions_per_user:
            # Remove oldest session
            oldest_session = user_sessions[0]
            self.invalidate_session(oldest_session)
        
        # Generate secure session ID
        session_id = secrets.token_urlsafe(32)
        
        # Create session data
        session_data = {
            'user_id': user_id,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'ip_address': None,  # Would be set by calling code
            'user_agent': None,  # Would be set by calling code
            'additional_data': additional_data or {}
        }
        
        # Store session
        self.sessions[session_id] = session_data
        
        # Update user session tracking
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        dna_logger.log_security_event(
            'session_created',
            {'user_id': user_id, 'session_id': session_id}
        )
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and refresh session."""
        if session_id not in self.sessions:
            return None
        
        session_data = self.sessions[session_id]
        current_time = time.time()
        
        # Check timeout
        timeout_seconds = self.config.session_timeout_minutes * 60
        if current_time - session_data['last_accessed'] > timeout_seconds:
            self.invalidate_session(session_id)
            return None
        
        # Update last accessed
        session_data['last_accessed'] = current_time
        
        return session_data
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session."""
        if session_id not in self.sessions:
            return False
        
        session_data = self.sessions[session_id]
        user_id = session_data['user_id']
        
        # Remove from sessions
        del self.sessions[session_id]
        
        # Remove from user sessions
        if user_id in self.user_sessions:
            self.user_sessions[user_id] = [
                sid for sid in self.user_sessions[user_id] if sid != session_id
            ]
        
        dna_logger.log_security_event(
            'session_invalidated',
            {'session_id': session_id, 'user_id': user_id}
        )
        
        return True
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = time.time()
        timeout_seconds = self.config.session_timeout_minutes * 60
        
        expired_sessions = []
        for session_id, session_data in self.sessions.items():
            if current_time - session_data['last_accessed'] > timeout_seconds:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class SecurityHeaders:
    """Security headers middleware."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize security headers."""
        self.config = config
    
    def get_headers(self) -> Dict[str, str]:
        """Get security headers."""
        return self.config.security_headers.copy()


class AuditLogger:
    """Security audit logging."""
    
    def __init__(self):
        """Initialize audit logger."""
        self.logger = get_logger('audit')
    
    def log_authentication_attempt(self, user_id: str, success: bool, 
                                 ip_address: str = None, details: Dict[str, Any] = None):
        """Log authentication attempt."""
        dna_logger.log_security_event(
            'authentication_attempt',
            {
                'user_id': user_id,
                'success': success,
                'ip_address': ip_address,
                'details': details or {}
            },
            'INFO' if success else 'WARN'
        )
    
    def log_data_access(self, user_id: str, resource: str, action: str,
                       session_id: str = None, details: Dict[str, Any] = None):
        """Log data access events."""
        dna_logger.log_security_event(
            'data_access',
            {
                'user_id': user_id,
                'resource': resource,
                'action': action,
                'session_id': session_id,
                'details': details or {}
            }
        )
    
    def log_permission_check(self, user_id: str, permission: str, 
                           granted: bool, resource: str = None):
        """Log permission checks."""
        dna_logger.log_security_event(
            'permission_check',
            {
                'user_id': user_id,
                'permission': permission,
                'granted': granted,
                'resource': resource
            },
            'INFO' if granted else 'WARN'
        )


# Security decorators
def require_session(func: Callable) -> Callable:
    """Decorator to require valid session."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # This would be implemented with actual session checking
        # For now, just a placeholder
        return func(self, *args, **kwargs)
    return wrapper


def rate_limit(requests_per_minute: int = 60):
    """Decorator for rate limiting."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Placeholder for rate limiting logic
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input(validator_func: Callable):
    """Decorator for input validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply validation to arguments
            # This is a simplified version
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Global security instances
security_config = SecurityConfig()
rate_limiter = RateLimiter(security_config)
input_validator = InputValidator(security_config)
session_manager = SessionManager(security_config)
security_headers = SecurityHeaders(security_config)
audit_logger = AuditLogger()