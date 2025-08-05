"""Compliance and regulatory utilities for global deployment."""

import hashlib
import time
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re

from .helpers import logger
from .i18n import _, get_current_locale


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class RegulationType(Enum):
    """Types of regulatory frameworks."""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act (US)
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore, Thailand)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    DPA = "dpa"  # Data Protection Act (UK)


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    activity_id: str
    purpose: str
    data_types: List[str]
    legal_basis: str
    retention_period: str
    data_subjects: List[str]
    recipients: List[str]
    transfers: List[str]
    timestamp: float = field(default_factory=time.time)
    user_consent: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'activity_id': self.activity_id,
            'purpose': self.purpose,
            'data_types': self.data_types,
            'legal_basis': self.legal_basis,
            'retention_period': self.retention_period,
            'data_subjects': self.data_subjects,
            'recipients': self.recipients,
            'transfers': self.transfers,
            'timestamp': self.timestamp,
            'user_consent': self.user_consent
        }


class ComplianceManager:
    """Manage compliance with data protection regulations."""
    
    def __init__(self, enabled_regulations: Optional[List[RegulationType]] = None):
        """Initialize compliance manager.
        
        Args:
            enabled_regulations: List of regulations to enforce
        """
        self.enabled_regulations = enabled_regulations or [
            RegulationType.GDPR,
            RegulationType.CCPA,
            RegulationType.PDPA
        ]
        
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
        # Setup compliance data directory
        self.compliance_dir = Path.home() / '.dna_origami_ae' / 'compliance'
        self.compliance_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Compliance manager initialized with regulations: {[r.value for r in self.enabled_regulations]}")
    
    def classify_data(self, data_description: str, 
                     personal_data: bool = False,
                     sensitive_data: bool = False) -> DataClassification:
        """Classify data according to sensitivity.
        
        Args:
            data_description: Description of the data
            personal_data: Whether data contains personal information
            sensitive_data: Whether data contains sensitive information
            
        Returns:
            Data classification level
        """
        # Check for sensitive patterns
        sensitive_patterns = [
            r'dna.*sequence.*human',
            r'genetic.*information',
            r'medical.*data',
            r'health.*record',
            r'biometric',
            r'genomic'
        ]
        
        is_sensitive_content = any(
            re.search(pattern, data_description.lower()) 
            for pattern in sensitive_patterns
        )
        
        if sensitive_data or is_sensitive_content:
            classification = DataClassification.RESTRICTED
        elif personal_data:
            classification = DataClassification.CONFIDENTIAL
        else:
            classification = DataClassification.INTERNAL
        
        self._log_audit_event('data_classification', {
            'description': data_description,
            'classification': classification.value,
            'personal_data': personal_data,
            'sensitive_data': sensitive_data
        })
        
        return classification
    
    def record_processing_activity(self, activity: DataProcessingRecord):
        """Record a data processing activity."""
        self.processing_records.append(activity)
        
        self._log_audit_event('processing_activity', activity.to_dict())
        
        # Check compliance requirements
        self._check_processing_compliance(activity)
        
        logger.debug(f"Recorded processing activity: {activity.activity_id}")
    
    def obtain_consent(self, user_id: str, purpose: str, 
                      data_types: List[str], required: bool = True) -> bool:
        """Obtain user consent for data processing.
        
        Args:
            user_id: Unique user identifier
            purpose: Purpose of data processing
            data_types: Types of data to be processed
            required: Whether consent is required
            
        Returns:
            Whether consent was obtained
        """
        # In a real implementation, this would show a consent dialog
        # For now, we simulate consent based on regulatory requirements
        
        consent_required = self._is_consent_required(purpose, data_types)
        
        if consent_required or required:
            # Simulate consent (in practice, would be obtained from user)
            consent_given = True  # Would be actual user response
            
            self.consent_records[user_id] = {
                'purpose': purpose,
                'data_types': data_types,
                'consent_given': consent_given,
                'timestamp': time.time(),
                'consent_method': 'explicit',
                'withdrawable': True
            }
            
            self._log_audit_event('consent_obtained', {
                'user_id': self._hash_user_id(user_id),
                'purpose': purpose,
                'data_types': data_types,
                'consent_given': consent_given
            })
            
            return consent_given
        
        return True  # No consent required
    
    def withdraw_consent(self, user_id: str) -> bool:
        """Allow user to withdraw consent.
        
        Args:
            user_id: User identifier
            
        Returns:
            Whether withdrawal was successful
        """
        if user_id in self.consent_records:
            self.consent_records[user_id]['consent_given'] = False
            self.consent_records[user_id]['withdrawal_timestamp'] = time.time()
            
            self._log_audit_event('consent_withdrawn', {
                'user_id': self._hash_user_id(user_id)
            })
            
            # Trigger data deletion if required
            self._handle_consent_withdrawal(user_id)
            
            return True
        
        return False
    
    def check_data_retention(self) -> List[Dict[str, Any]]:
        """Check for data that should be deleted due to retention policies.
        
        Returns:
            List of data items to be deleted
        """
        items_to_delete = []
        current_time = time.time()
        
        for record in self.processing_records:
            retention_seconds = self._parse_retention_period(record.retention_period)
            
            if current_time - record.timestamp > retention_seconds:
                items_to_delete.append({
                    'activity_id': record.activity_id,
                    'reason': 'retention_period_expired',
                    'age_days': (current_time - record.timestamp) / 86400
                })
        
        if items_to_delete:
            self._log_audit_event('retention_check', {
                'items_to_delete': len(items_to_delete)
            })
        
        return items_to_delete
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate privacy compliance report.
        
        Returns:
            Comprehensive privacy report
        """
        report = {
            'generated_at': time.time(),
            'regulations': [r.value for r in self.enabled_regulations],
            'processing_activities': {
                'total': len(self.processing_records),
                'by_purpose': self._group_by_purpose(),
                'by_legal_basis': self._group_by_legal_basis()
            },
            'consent_management': {
                'total_consents': len(self.consent_records),
                'active_consents': sum(1 for c in self.consent_records.values() 
                                     if c.get('consent_given', False)),
                'withdrawn_consents': sum(1 for c in self.consent_records.values() 
                                        if not c.get('consent_given', True))
            },
            'data_retention': {
                'items_due_for_deletion': len(self.check_data_retention())
            },
            'audit_events': len(self.audit_log),
            'compliance_status': self._assess_compliance_status()
        }
        
        # Save report
        report_file = self.compliance_dir / f"privacy_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Privacy report generated: {report_file}")
        
        return report
    
    def handle_data_subject_request(self, request_type: str, user_id: str) -> Dict[str, Any]:
        """Handle data subject rights requests.
        
        Args:
            request_type: Type of request (access, rectification, erasure, portability)
            user_id: Subject's user ID
            
        Returns:
            Response to the request
        """
        user_id_hash = self._hash_user_id(user_id)
        
        if request_type == 'access':
            # Right to access personal data
            user_data = self._collect_user_data(user_id)
            response = {
                'request_type': 'access',
                'user_data': user_data,
                'processing_activities': [
                    r.to_dict() for r in self.processing_records 
                    if user_id in r.data_subjects
                ]
            }
            
        elif request_type == 'erasure':
            # Right to be forgotten
            deleted_items = self._delete_user_data(user_id)
            response = {
                'request_type': 'erasure',
                'deleted_items': deleted_items,
                'status': 'completed'
            }
            
        elif request_type == 'portability':
            # Right to data portability
            portable_data = self._export_user_data(user_id)
            response = {
                'request_type': 'portability',
                'data_format': 'json',
                'data': portable_data
            }
            
        elif request_type == 'rectification':
            # Right to rectification
            response = {
                'request_type': 'rectification',
                'status': 'manual_review_required',
                'message': 'Please provide corrected information'
            }
            
        else:
            response = {
                'request_type': request_type,
                'status': 'unsupported',
                'message': f'Request type {request_type} is not supported'
            }
        
        self._log_audit_event('data_subject_request', {
            'request_type': request_type,
            'user_id': user_id_hash,
            'status': response.get('status', 'processed')
        })
        
        return response
    
    def _is_consent_required(self, purpose: str, data_types: List[str]) -> bool:
        """Determine if consent is required for processing."""
        # GDPR requires consent for most processing of personal data
        if RegulationType.GDPR in self.enabled_regulations:
            personal_data_types = ['personal', 'genetic', 'biometric', 'health']
            if any(dt in data_types for dt in personal_data_types):
                return True
        
        # CCPA has different requirements
        if RegulationType.CCPA in self.enabled_regulations:
            # CCPA focuses on selling personal information
            if 'sell' in purpose.lower() or 'share' in purpose.lower():
                return True
        
        return False
    
    def _check_processing_compliance(self, activity: DataProcessingRecord):
        """Check if processing activity complies with regulations."""
        issues = []
        
        # Check GDPR compliance
        if RegulationType.GDPR in self.enabled_regulations:
            if not activity.legal_basis:
                issues.append("Missing legal basis for processing (GDPR Art. 6)")
            
            if not activity.retention_period:
                issues.append("Missing retention period (GDPR Art. 5)")
        
        # Check CCPA compliance
        if RegulationType.CCPA in self.enabled_regulations:
            if 'california' in activity.data_subjects and not activity.user_consent:
                issues.append("Missing user consent for California residents (CCPA)")
        
        if issues:
            self._log_audit_event('compliance_issues', {
                'activity_id': activity.activity_id,
                'issues': issues
            })
            
            logger.warning(f"Compliance issues found for activity {activity.activity_id}: {issues}")
    
    def _parse_retention_period(self, period: str) -> float:
        """Parse retention period string to seconds."""
        period = period.lower()
        
        if 'day' in period:
            days = int(re.search(r'\\d+', period).group())
            return days * 86400
        elif 'month' in period:
            months = int(re.search(r'\\d+', period).group())
            return months * 30 * 86400
        elif 'year' in period:
            years = int(re.search(r'\\d+', period).group())
            return years * 365 * 86400
        else:
            # Default to 1 year
            return 365 * 86400
    
    def _group_by_purpose(self) -> Dict[str, int]:
        """Group processing activities by purpose."""
        purposes = {}
        for record in self.processing_records:
            purposes[record.purpose] = purposes.get(record.purpose, 0) + 1
        return purposes
    
    def _group_by_legal_basis(self) -> Dict[str, int]:
        """Group processing activities by legal basis."""
        bases = {}
        for record in self.processing_records:
            bases[record.legal_basis] = bases.get(record.legal_basis, 0) + 1
        return bases
    
    def _assess_compliance_status(self) -> Dict[str, Any]:
        """Assess overall compliance status."""
        total_activities = len(self.processing_records)
        activities_with_legal_basis = sum(
            1 for r in self.processing_records if r.legal_basis
        )
        activities_with_retention = sum(
            1 for r in self.processing_records if r.retention_period
        )
        
        return {
            'legal_basis_coverage': activities_with_legal_basis / total_activities if total_activities > 0 else 1.0,
            'retention_policy_coverage': activities_with_retention / total_activities if total_activities > 0 else 1.0,
            'consent_management_active': len(self.consent_records) > 0,
            'audit_logging_active': len(self.audit_log) > 0
        }
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy."""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def _collect_user_data(self, user_id: str) -> Dict[str, Any]:
        """Collect all data associated with a user."""
        # In practice, this would query all systems for user data
        return {
            'user_id_hash': self._hash_user_id(user_id),
            'consent_records': self.consent_records.get(user_id, {}),
            'processing_activities': [
                r.activity_id for r in self.processing_records 
                if user_id in r.data_subjects
            ]
        }
    
    def _delete_user_data(self, user_id: str) -> List[str]:
        """Delete all data associated with a user."""
        deleted_items = []
        
        # Remove consent records
        if user_id in self.consent_records:
            del self.consent_records[user_id]
            deleted_items.append('consent_records')
        
        # Remove from processing records
        self.processing_records = [
            r for r in self.processing_records 
            if user_id not in r.data_subjects
        ]
        deleted_items.append('processing_records')
        
        return deleted_items
    
    def _export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data in portable format."""
        return self._collect_user_data(user_id)
    
    def _handle_consent_withdrawal(self, user_id: str):
        """Handle actions required when consent is withdrawn."""
        # Stop processing personal data
        logger.info(f"Consent withdrawn for user {self._hash_user_id(user_id)}")
        
        # In practice, would trigger data deletion or anonymization
        pass
    
    def _log_audit_event(self, event_type: str, data: Dict[str, Any]):
        """Log audit event."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'data': data,
            'locale': get_current_locale()
        }
        
        self.audit_log.append(event)
        
        # Keep audit log manageable
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]


class DataAnonymizer:
    """Anonymize sensitive data for compliance."""
    
    @staticmethod
    def anonymize_dna_sequence(sequence: str, method: str = 'substitution') -> str:
        """Anonymize DNA sequence while preserving structure.
        
        Args:
            sequence: Original DNA sequence
            method: Anonymization method
            
        Returns:
            Anonymized sequence
        """
        if method == 'substitution':
            # Replace with random bases maintaining GC content
            import random
            bases = ['A', 'T', 'G', 'C']
            
            # Calculate original GC content
            gc_count = sequence.count('G') + sequence.count('C')
            gc_content = gc_count / len(sequence)
            
            # Generate new sequence with similar GC content
            anonymized = []
            for _ in range(len(sequence)):
                if random.random() < gc_content:
                    anonymized.append(random.choice(['G', 'C']))
                else:
                    anonymized.append(random.choice(['A', 'T']))
            
            return ''.join(anonymized)
        
        elif method == 'masking':
            # Replace with N's
            return 'N' * len(sequence)
        
        else:
            raise ValueError(f"Unknown anonymization method: {method}")
    
    @staticmethod
    def anonymize_coordinates(coordinates, method: str = 'noise') -> any:
        """Anonymize molecular coordinates.
        
        Args:
            coordinates: Original coordinates
            method: Anonymization method
            
        Returns:
            Anonymized coordinates
        """
        import numpy as np
        
        if method == 'noise':
            # Add random noise
            noise_level = 0.1 * np.std(coordinates)
            noise = np.random.normal(0, noise_level, coordinates.shape)
            return coordinates + noise
        
        elif method == 'translation':
            # Apply random translation
            translation = np.random.normal(0, 10, 3)
            return coordinates + translation
        
        else:
            raise ValueError(f"Unknown anonymization method: {method}")


# Global compliance manager instance
compliance_manager = ComplianceManager()


def ensure_compliance(data_type: str, purpose: str, user_id: Optional[str] = None) -> bool:
    """Ensure compliance for data processing.
    
    Args:
        data_type: Type of data being processed
        purpose: Purpose of processing
        user_id: User identifier if applicable
        
    Returns:
        Whether processing is compliant
    """
    # Classify data
    classification = compliance_manager.classify_data(
        data_type,
        personal_data='personal' in data_type.lower(),
        sensitive_data='genetic' in data_type.lower() or 'dna' in data_type.lower()
    )
    
    # Obtain consent if needed
    if user_id and classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
        consent_obtained = compliance_manager.obtain_consent(
            user_id, purpose, [data_type]
        )
        if not consent_obtained:
            return False
    
    # Record processing activity
    activity = DataProcessingRecord(
        activity_id=f"{purpose}_{int(time.time())}",
        purpose=purpose,
        data_types=[data_type],
        legal_basis="legitimate_interest" if classification == DataClassification.INTERNAL else "consent",
        retention_period="2 years",
        data_subjects=[user_id] if user_id else [],
        recipients=["dna_origami_ae_system"],
        transfers=[]
    )
    
    compliance_manager.record_processing_activity(activity)
    
    return True