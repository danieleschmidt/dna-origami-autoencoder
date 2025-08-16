"""
Advanced Threat Detection System - Generation 2 Enhancement
Real-time security monitoring with ML-based anomaly detection.
"""

import os
import re
import hashlib
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ThreatConfig:
    """Configuration for threat detection system."""
    enable_realtime_monitoring: bool = True
    anomaly_threshold: float = 0.7
    rate_limit_window: int = 60  # seconds
    max_requests_per_window: int = 100
    enable_ml_detection: bool = True
    log_retention_hours: int = 24
    alert_cooldown_seconds: int = 300

@dataclass 
class SecurityEvent:
    """Security event data structure."""
    timestamp: datetime
    event_type: str
    severity: str  # "low", "medium", "high", "critical"
    source_ip: str
    user_agent: str
    endpoint: str
    payload_hash: str
    anomaly_score: float = 0.0
    indicators: List[str] = field(default_factory=list)
    blocked: bool = False

class RequestPattern:
    """Pattern analysis for request behavior."""
    
    def __init__(self, max_history: int = 1000):
        self.request_history = deque(maxlen=max_history)
        self.ip_patterns = defaultdict(list)
        self.endpoint_patterns = defaultdict(int)
        self.user_agent_patterns = defaultdict(int)
        
    def add_request(self, ip: str, endpoint: str, user_agent: str, timestamp: datetime):
        """Add request to pattern analysis."""
        request = {
            'ip': ip,
            'endpoint': endpoint,
            'user_agent': user_agent,
            'timestamp': timestamp
        }
        
        self.request_history.append(request)
        self.ip_patterns[ip].append(timestamp)
        self.endpoint_patterns[endpoint] += 1
        self.user_agent_patterns[user_agent] += 1
        
        # Cleanup old IP patterns (keep last 100 requests per IP)
        if len(self.ip_patterns[ip]) > 100:
            self.ip_patterns[ip] = self.ip_patterns[ip][-100:]
    
    def get_ip_request_rate(self, ip: str, window_seconds: int = 60) -> float:
        """Get request rate for IP in given time window."""
        if ip not in self.ip_patterns:
            return 0.0
            
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_requests = [ts for ts in self.ip_patterns[ip] if ts > cutoff_time]
        
        return len(recent_requests) / window_seconds
    
    def detect_rapid_fire(self, ip: str, threshold_rps: float = 10.0) -> bool:
        """Detect rapid-fire requests from IP."""
        rate = self.get_ip_request_rate(ip, window_seconds=10)
        return rate > threshold_rps
    
    def detect_scanning_behavior(self, ip: str) -> Tuple[bool, List[str]]:
        """Detect port/endpoint scanning behavior."""
        if ip not in self.ip_patterns:
            return False, []
            
        # Get recent requests from this IP
        cutoff_time = datetime.now() - timedelta(minutes=10)
        recent_requests = [r for r in self.request_history 
                          if r['ip'] == ip and r['timestamp'] > cutoff_time]
        
        # Check for diverse endpoint access pattern
        endpoints_accessed = set(r['endpoint'] for r in recent_requests)
        
        indicators = []
        
        # Many different endpoints
        if len(endpoints_accessed) > 20:
            indicators.append(f"Accessed {len(endpoints_accessed)} different endpoints")
        
        # Check for systematic scanning patterns
        admin_endpoints = [ep for ep in endpoints_accessed if any(
            admin_term in ep.lower() for admin_term in ['admin', 'config', 'debug', 'test']
        )]
        
        if len(admin_endpoints) > 5:
            indicators.append(f"Attempted access to {len(admin_endpoints)} admin endpoints")
        
        # Check for error-generating requests
        error_patterns = [ep for ep in endpoints_accessed if any(
            pattern in ep for pattern in ['../', '.env', 'wp-admin', 'phpmyadmin']
        )]
        
        if len(error_patterns) > 3:
            indicators.append(f"Attempted known vulnerability patterns: {len(error_patterns)}")
        
        is_scanning = len(indicators) > 0
        return is_scanning, indicators

class PayloadAnalyzer:
    """Analyzes request payloads for malicious content."""
    
    def __init__(self):
        # Malicious pattern signatures
        self.sql_injection_patterns = [
            r"(\s|^)(select|insert|update|delete|drop|create|alter)\s",
            r"(union\s+select|1=1|'=')",
            r"(exec|execute|sp_)",
            r"(--|#|\*/)",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
            r"<img[^>]*src[^>]*>",
            r"eval\s*\(",
        ]
        
        self.command_injection_patterns = [
            r"(;|\||\&\&|\|\|)",
            r"(cat|ls|pwd|whoami|id)\s",
            r"(\.\./){2,}",
            r"(nc|netcat|wget|curl)\s",
        ]
        
        self.path_traversal_patterns = [
            r"(\.\./){2,}",
            r"(etc/passwd|boot\.ini|win\.ini)",
            r"(proc/version|proc/cpuinfo)",
        ]
        
    def analyze_payload(self, payload: str) -> Dict[str, Any]:
        """Analyze payload for malicious patterns."""
        if not payload:
            return {'malicious': False, 'threats': [], 'score': 0.0}
        
        payload_lower = payload.lower()
        threats = []
        total_score = 0.0
        
        # SQL Injection detection
        sql_matches = self._check_patterns(payload_lower, self.sql_injection_patterns)
        if sql_matches:
            threats.append('sql_injection')
            total_score += 0.8 * len(sql_matches)
        
        # XSS detection
        xss_matches = self._check_patterns(payload, self.xss_patterns)  # Case-sensitive for HTML
        if xss_matches:
            threats.append('xss')
            total_score += 0.7 * len(xss_matches)
        
        # Command injection detection
        cmd_matches = self._check_patterns(payload_lower, self.command_injection_patterns)
        if cmd_matches:
            threats.append('command_injection')
            total_score += 0.9 * len(cmd_matches)
        
        # Path traversal detection
        path_matches = self._check_patterns(payload_lower, self.path_traversal_patterns)
        if path_matches:
            threats.append('path_traversal')
            total_score += 0.6 * len(path_matches)
        
        # Anomalous character frequency
        anomaly_score = self._calculate_character_anomaly(payload)
        if anomaly_score > 0.7:
            threats.append('character_anomaly')
            total_score += 0.3 * anomaly_score
        
        # Length-based anomaly
        if len(payload) > 10000:  # Unusually long payload
            threats.append('oversized_payload')
            total_score += 0.4
        
        # Normalize score
        final_score = min(1.0, total_score)
        is_malicious = final_score > 0.5 or len(threats) > 1
        
        return {
            'malicious': is_malicious,
            'threats': threats,
            'score': final_score,
            'payload_length': len(payload),
            'anomaly_score': anomaly_score
        }
    
    def _check_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Check text against list of regex patterns."""
        matches = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        return matches
    
    def _calculate_character_anomaly(self, payload: str) -> float:
        """Calculate character frequency anomaly score."""
        if len(payload) < 10:
            return 0.0
        
        # Calculate character frequency
        char_counts = defaultdict(int)
        for char in payload:
            char_counts[char] += 1
        
        # Calculate entropy-like metric
        total_chars = len(payload)
        frequencies = [count / total_chars for count in char_counts.values()]
        
        # High entropy (uniform distribution) or very low entropy can be suspicious
        entropy = -sum(f * np.log2(f) for f in frequencies if f > 0)
        max_entropy = np.log2(len(char_counts))
        
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Anomaly score: high for very low or very high entropy
        if normalized_entropy < 0.3 or normalized_entropy > 0.9:
            return abs(normalized_entropy - 0.5) * 2
        
        return 0.0

class MLAnomalyDetector:
    """Machine learning-based anomaly detection."""
    
    def __init__(self):
        self.feature_history = deque(maxlen=1000)
        self.normal_profiles = {}
        self.anomaly_threshold = 0.7
        
    def extract_features(self, request_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from request data."""
        features = []
        
        # Timing features
        hour = datetime.now().hour
        features.extend([
            hour / 24.0,  # Normalized hour
            1.0 if 9 <= hour <= 17 else 0.0,  # Business hours
            1.0 if hour < 6 or hour > 22 else 0.0,  # Unusual hours
        ])
        
        # Request characteristics
        endpoint = request_data.get('endpoint', '')
        user_agent = request_data.get('user_agent', '')
        payload = request_data.get('payload', '')
        
        features.extend([
            len(endpoint) / 100.0,  # Normalized endpoint length
            len(user_agent) / 200.0,  # Normalized user agent length
            len(payload) / 1000.0,  # Normalized payload length
            payload.count('/') / max(len(payload), 1),  # Path separator density
            payload.count('=') / max(len(payload), 1),  # Parameter density
            payload.count('&') / max(len(payload), 1),  # Parameter separator density
        ])
        
        # IP-based features (simplified)
        ip = request_data.get('source_ip', '0.0.0.0')
        ip_parts = ip.split('.')
        if len(ip_parts) == 4:
            try:
                features.extend([
                    int(ip_parts[0]) / 255.0,  # First octet
                    int(ip_parts[3]) / 255.0,  # Last octet
                ])
            except ValueError:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        # Common attack indicators
        features.extend([
            1.0 if 'script' in payload.lower() else 0.0,
            1.0 if 'select' in payload.lower() else 0.0,
            1.0 if '../' in payload else 0.0,
            1.0 if len(user_agent) < 10 else 0.0,  # Suspicious short user agent
        ])
        
        return np.array(features)
    
    def update_normal_profile(self, features: np.ndarray, user_id: str = "global"):
        """Update normal behavior profile."""
        if user_id not in self.normal_profiles:
            self.normal_profiles[user_id] = {
                'feature_sum': np.zeros_like(features),
                'feature_sum_squared': np.zeros_like(features),
                'count': 0
            }
        
        profile = self.normal_profiles[user_id]
        profile['feature_sum'] += features
        profile['feature_sum_squared'] += features ** 2
        profile['count'] += 1
        
        # Limit profile size
        if profile['count'] > 1000:
            # Decay old data
            profile['feature_sum'] *= 0.9
            profile['feature_sum_squared'] *= 0.9
            profile['count'] = int(profile['count'] * 0.9)
    
    def calculate_anomaly_score(self, features: np.ndarray, user_id: str = "global") -> float:
        """Calculate anomaly score for given features."""
        if user_id not in self.normal_profiles or self.normal_profiles[user_id]['count'] < 10:
            return 0.5  # Neutral score for insufficient data
        
        profile = self.normal_profiles[user_id]
        count = profile['count']
        
        # Calculate mean and std
        mean = profile['feature_sum'] / count
        variance = (profile['feature_sum_squared'] / count) - (mean ** 2)
        std = np.sqrt(np.maximum(variance, 0.01))  # Prevent division by zero
        
        # Calculate z-scores
        z_scores = np.abs((features - mean) / std)
        
        # Anomaly score based on maximum z-score and average
        max_z_score = np.max(z_scores)
        avg_z_score = np.mean(z_scores)
        
        # Combine scores
        anomaly_score = min(1.0, (max_z_score + avg_z_score) / 6.0)  # Normalize to [0,1]
        
        return anomaly_score

class ThreatDetectionSystem:
    """
    Main threat detection system with real-time monitoring.
    """
    
    def __init__(self, config: ThreatConfig = None):
        self.config = config or ThreatConfig()
        self.logger = get_logger(f"{__name__}.ThreatDetectionSystem")
        
        # Components
        self.pattern_analyzer = RequestPattern()
        self.payload_analyzer = PayloadAnalyzer()
        self.ml_detector = MLAnomalyDetector()
        
        # State
        self.security_events = deque(maxlen=10000)
        self.blocked_ips = set()
        self.rate_limits = defaultdict(list)
        self.alert_cooldowns = defaultdict(float)
        
        # Monitoring
        self.active = False
        self.monitoring_thread = None
        
    def start_monitoring(self):
        """Start real-time threat monitoring."""
        if self.active:
            return
            
        self.active = True
        if self.config.enable_realtime_monitoring:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
        
        self.logger.info("Threat detection system started")
    
    def stop_monitoring(self):
        """Stop threat monitoring."""
        self.active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Threat detection system stopped")
    
    def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze incoming request for threats.
        
        Args:
            request_data: Dictionary containing:
                - source_ip: Client IP address
                - endpoint: Requested endpoint
                - user_agent: User agent string
                - payload: Request payload/parameters
                - method: HTTP method
        
        Returns:
            Analysis result with threat assessment
        """
        timestamp = datetime.now()
        source_ip = request_data.get('source_ip', 'unknown')
        endpoint = request_data.get('endpoint', '')
        user_agent = request_data.get('user_agent', '')
        payload = request_data.get('payload', '')
        
        # Create payload hash for tracking
        payload_hash = hashlib.md5(payload.encode()).hexdigest()
        
        # Initialize result
        result = {
            'threat_detected': False,
            'severity': 'low',
            'threats': [],
            'anomaly_score': 0.0,
            'blocked': False,
            'analysis_details': {}
        }
        
        # Check if IP is already blocked
        if source_ip in self.blocked_ips:
            result.update({
                'threat_detected': True,
                'severity': 'high',
                'threats': ['blocked_ip'],
                'blocked': True
            })
            return result
        
        # Rate limiting analysis
        rate_limit_violated = self._check_rate_limits(source_ip, timestamp)
        if rate_limit_violated:
            result['threats'].append('rate_limit_violation')
            result['severity'] = 'medium'
        
        # Pattern analysis
        self.pattern_analyzer.add_request(source_ip, endpoint, user_agent, timestamp)
        
        # Check for rapid-fire requests
        if self.pattern_analyzer.detect_rapid_fire(source_ip):
            result['threats'].append('rapid_fire_requests')
            result['severity'] = 'high'
        
        # Check for scanning behavior
        is_scanning, scan_indicators = self.pattern_analyzer.detect_scanning_behavior(source_ip)
        if is_scanning:
            result['threats'].append('scanning_behavior')
            result['analysis_details']['scan_indicators'] = scan_indicators
            result['severity'] = 'high'
        
        # Payload analysis
        if payload:
            payload_analysis = self.payload_analyzer.analyze_payload(payload)
            if payload_analysis['malicious']:
                result['threats'].extend(payload_analysis['threats'])
                result['severity'] = 'critical'
                result['analysis_details']['payload_analysis'] = payload_analysis
        
        # ML-based anomaly detection
        if self.config.enable_ml_detection:
            features = self.ml_detector.extract_features(request_data)
            anomaly_score = self.ml_detector.calculate_anomaly_score(features, source_ip)
            result['anomaly_score'] = anomaly_score
            
            if anomaly_score > self.config.anomaly_threshold:
                result['threats'].append('anomalous_behavior')
                if result['severity'] == 'low':
                    result['severity'] = 'medium'
            
            # Update normal profile for legitimate traffic
            if anomaly_score < 0.5 and not result['threats']:
                self.ml_detector.update_normal_profile(features, source_ip)
        
        # Determine final threat status
        result['threat_detected'] = len(result['threats']) > 0
        
        # Create security event
        if result['threat_detected']:
            event = SecurityEvent(
                timestamp=timestamp,
                event_type='threat_detected',
                severity=result['severity'],
                source_ip=source_ip,
                user_agent=user_agent,
                endpoint=endpoint,
                payload_hash=payload_hash,
                anomaly_score=result['anomaly_score'],
                indicators=result['threats']
            )
            
            self.security_events.append(event)
            
            # Auto-blocking for critical threats
            if result['severity'] == 'critical' or len(result['threats']) >= 3:
                self._block_ip(source_ip, "Multiple threat indicators")
                result['blocked'] = True
            
            # Alerting with cooldown
            self._generate_alert(event)
        
        return result
    
    def _check_rate_limits(self, ip: str, timestamp: datetime) -> bool:
        """Check if IP has violated rate limits."""
        # Clean old timestamps
        cutoff_time = timestamp - timedelta(seconds=self.config.rate_limit_window)
        self.rate_limits[ip] = [ts for ts in self.rate_limits[ip] if ts > cutoff_time]
        
        # Add current request
        self.rate_limits[ip].append(timestamp)
        
        # Check limit
        return len(self.rate_limits[ip]) > self.config.max_requests_per_window
    
    def _block_ip(self, ip: str, reason: str):
        """Block an IP address."""
        self.blocked_ips.add(ip)
        self.logger.warning(f"Blocked IP {ip}: {reason}")
        
        # Clean up rate limit tracking for blocked IP
        if ip in self.rate_limits:
            del self.rate_limits[ip]
    
    def _generate_alert(self, event: SecurityEvent):
        """Generate security alert with cooldown."""
        alert_key = f"{event.source_ip}_{event.event_type}"
        current_time = time.time()
        
        # Check cooldown
        if alert_key in self.alert_cooldowns:
            if current_time - self.alert_cooldowns[alert_key] < self.config.alert_cooldown_seconds:
                return  # Skip alert due to cooldown
        
        # Generate alert
        self.alert_cooldowns[alert_key] = current_time
        
        log_level = {
            'low': self.logger.info,
            'medium': self.logger.warning,
            'high': self.logger.error,
            'critical': self.logger.critical
        }.get(event.severity, self.logger.warning)
        
        log_level(
            f"SECURITY ALERT [{event.severity.upper()}] {event.source_ip}: "
            f"{', '.join(event.indicators)} on {event.endpoint}"
        )
    
    def _monitoring_loop(self):
        """Background monitoring loop for cleanup and analysis."""
        while self.active:
            try:
                current_time = datetime.now()
                
                # Clean old security events
                cutoff_time = current_time - timedelta(hours=self.config.log_retention_hours)
                original_count = len(self.security_events)
                self.security_events = deque(
                    (event for event in self.security_events if event.timestamp > cutoff_time),
                    maxlen=10000
                )
                
                if len(self.security_events) < original_count:
                    self.logger.debug(f"Cleaned {original_count - len(self.security_events)} old security events")
                
                # Clean old rate limit data
                for ip in list(self.rate_limits.keys()):
                    cutoff_time = current_time - timedelta(seconds=self.config.rate_limit_window * 2)
                    self.rate_limits[ip] = [ts for ts in self.rate_limits[ip] if ts > cutoff_time]
                    if not self.rate_limits[ip]:
                        del self.rate_limits[ip]
                
                # Clean old alert cooldowns
                current_timestamp = time.time()
                expired_cooldowns = [
                    key for key, timestamp in self.alert_cooldowns.items()
                    if current_timestamp - timestamp > self.config.alert_cooldown_seconds * 2
                ]
                for key in expired_cooldowns:
                    del self.alert_cooldowns[key]
                
                time.sleep(60)  # Run cleanup every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        recent_events = [
            event for event in self.security_events
            if (datetime.now() - event.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        status = {
            'monitoring_active': self.active,
            'blocked_ips_count': len(self.blocked_ips),
            'recent_events_count': len(recent_events),
            'critical_events_count': len([e for e in recent_events if e.severity == 'critical']),
            'high_events_count': len([e for e in recent_events if e.severity == 'high']),
            'top_threat_sources': self._get_top_threat_sources(recent_events),
            'top_threat_types': self._get_top_threat_types(recent_events),
            'ml_profiles_count': len(self.ml_detector.normal_profiles),
            'rate_limited_ips': len(self.rate_limits),
            'active_cooldowns': len(self.alert_cooldowns)
        }
        
        return status
    
    def _get_top_threat_sources(self, events: List[SecurityEvent], limit: int = 5) -> List[Dict[str, Any]]:
        """Get top threat source IPs."""
        ip_counts = defaultdict(int)
        for event in events:
            ip_counts[event.source_ip] += 1
        
        sorted_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'ip': ip, 'event_count': count} for ip, count in sorted_ips[:limit]]
    
    def _get_top_threat_types(self, events: List[SecurityEvent], limit: int = 5) -> List[Dict[str, Any]]:
        """Get top threat types."""
        threat_counts = defaultdict(int)
        for event in events:
            for indicator in event.indicators:
                threat_counts[indicator] += 1
        
        sorted_threats = sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'threat_type': threat, 'count': count} for threat, count in sorted_threats[:limit]]
    
    def unblock_ip(self, ip: str) -> bool:
        """Manually unblock an IP address."""
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            self.logger.info(f"Unblocked IP: {ip}")
            return True
        return False
    
    def get_blocked_ips(self) -> List[str]:
        """Get list of currently blocked IPs."""
        return list(self.blocked_ips)

# Integration function
def create_threat_detection_system(config: ThreatConfig = None) -> ThreatDetectionSystem:
    """
    Create and start threat detection system.
    """
    system = ThreatDetectionSystem(config)
    system.start_monitoring()
    
    logger.info("Threat detection system created and started")
    return system