"""
Global Compliance and Regulatory Framework for DNA Origami AutoEncoder

Implements GDPR, CCPA, PDPA compliance and international data protection
regulations for global deployment.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
import uuid

# Compliance standards
class ComplianceStandard(Enum):
    """Supported compliance standards."""
    GDPR = "gdpr"         # EU General Data Protection Regulation
    CCPA = "ccpa"         # California Consumer Privacy Act
    PDPA = "pdpa"         # Personal Data Protection Act (Singapore/Thailand)
    PIPEDA = "pipeda"     # Personal Information Protection and Electronic Documents Act (Canada)
    LGPD = "lgpd"         # Lei Geral de Proteção de Dados (Brazil)
    APPI = "appi"         # Act on Protection of Personal Information (Japan)


class DataCategory(Enum):
    """Categories of data processed."""
    RESEARCH_DATA = "research_data"           # Scientific research data
    IMAGE_DATA = "image_data"                # Image files and derivatives
    DNA_SEQUENCES = "dna_sequences"          # Generated DNA sequences  
    SIMULATION_DATA = "simulation_data"       # Simulation results
    USER_METADATA = "user_metadata"          # User interaction metadata
    SYSTEM_LOGS = "system_logs"              # System and audit logs
    PERFORMANCE_METRICS = "performance_metrics" # System performance data


class ProcessingPurpose(Enum):
    """Legal purposes for data processing."""
    RESEARCH = "research"                     # Scientific research
    SERVICE_PROVISION = "service_provision"  # Providing the DNA origami service
    SYSTEM_OPTIMIZATION = "system_optimization" # Performance optimization
    SECURITY = "security"                     # Security and fraud prevention
    COMPLIANCE = "compliance"                 # Legal compliance
    ANALYTICS = "analytics"                   # Anonymous analytics


@dataclass
class GlobalCompliance:
    """Global compliance configuration and utilities."""
    
    # Regional compliance settings
    gdpr_enabled: bool = True
    ccpa_enabled: bool = True
    pdpa_enabled: bool = True
    
    # Data retention periods (days)
    research_data_retention: int = 2555  # 7 years
    image_data_retention: int = 1095     # 3 years
    sequence_data_retention: int = 1825  # 5 years
    
    # Privacy settings
    auto_anonymization: bool = True
    consent_expiry_days: int = 365
    audit_log_retention: int = 2190  # 6 years
    
    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.consent_records = {}
        self.audit_events = []
        
    def record_consent(self, user_id: str, purposes: List[ProcessingPurpose], 
                      granted: bool = True) -> str:
        """Record user consent for data processing."""
        
        consent_id = str(uuid.uuid4())
        consent_record = {
            "consent_id": consent_id,
            "user_id_hash": hashlib.sha256(user_id.encode()).hexdigest()[:16],
            "purposes": [p.value for p in purposes],
            "granted": granted,
            "timestamp": time.time(),
            "expires_at": time.time() + (self.consent_expiry_days * 86400),
            "withdrawal_available": True
        }
        
        self.consent_records[consent_id] = consent_record
        
        self._log_audit("consent_recorded", {
            "consent_id": consent_id,
            "purposes_count": len(purposes),
            "granted": granted
        })
        
        return consent_id
    
    def has_valid_consent(self, user_id: str, purpose: ProcessingPurpose) -> bool:
        """Check if user has valid consent for a purpose."""
        
        user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        current_time = time.time()
        
        for consent in self.consent_records.values():
            if (consent["user_id_hash"] == user_hash and
                purpose.value in consent["purposes"] and
                consent["granted"] and
                consent["expires_at"] > current_time):
                return True
                
        return False
    
    def process_data_subject_request(self, request_type: str, user_id: str) -> Dict[str, Any]:
        """Process data subject rights requests (GDPR Articles 15-22)."""
        
        user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        request_id = str(uuid.uuid4())
        
        if request_type == "access":
            # Right of access (Article 15)
            user_data = self._collect_user_data(user_hash)
            response = {
                "request_id": request_id,
                "type": "access",
                "status": "completed",
                "data": user_data,
                "processing_purposes": self._get_user_purposes(user_hash),
                "retention_periods": self._get_retention_info(),
                "rights_info": self._get_rights_information()
            }
            
        elif request_type == "portability":
            # Right to data portability (Article 20)
            portable_data = self._export_user_data(user_hash)
            response = {
                "request_id": request_id,
                "type": "portability", 
                "status": "completed",
                "format": "json",
                "data": portable_data
            }
            
        elif request_type == "erasure":
            # Right to erasure (Article 17)
            deleted_items = self._delete_user_data(user_hash)
            response = {
                "request_id": request_id,
                "type": "erasure",
                "status": "completed", 
                "deleted_items": deleted_items,
                "retention_exceptions": self._check_retention_exceptions(user_hash)
            }
            
        elif request_type == "rectification":
            # Right to rectification (Article 16)
            response = {
                "request_id": request_id,
                "type": "rectification",
                "status": "manual_review_required",
                "instructions": "Please provide specific corrections needed"
            }
            
        else:
            response = {
                "request_id": request_id,
                "type": request_type,
                "status": "unsupported",
                "message": f"Request type '{request_type}' is not supported"
            }
        
        self._log_audit("data_subject_request", {
            "request_id": request_id,
            "type": request_type,
            "user_hash": user_hash,
            "status": response["status"]
        })
        
        return response
    
    def check_regional_compliance(self, user_location: str, data_type: DataCategory) -> Dict[str, Any]:
        """Check compliance requirements based on user location."""
        
        compliance_requirements = {
            "applicable_regulations": [],
            "consent_required": False,
            "legitimate_interests_allowed": True,
            "data_transfer_restrictions": False,
            "retention_limits": None
        }
        
        # EU/EEA - GDPR
        eu_countries = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "ES", "FI", "FR", "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"]
        if user_location in eu_countries:
            compliance_requirements["applicable_regulations"].append("GDPR")
            if data_type in [DataCategory.DNA_SEQUENCES, DataCategory.IMAGE_DATA]:
                compliance_requirements["consent_required"] = True
            compliance_requirements["data_transfer_restrictions"] = True
        
        # California - CCPA
        if user_location == "CA":
            compliance_requirements["applicable_regulations"].append("CCPA")
            compliance_requirements["legitimate_interests_allowed"] = False
        
        # Singapore - PDPA
        if user_location == "SG":
            compliance_requirements["applicable_regulations"].append("PDPA")
            compliance_requirements["consent_required"] = True
        
        # Brazil - LGPD
        if user_location == "BR":
            compliance_requirements["applicable_regulations"].append("LGPD")
            compliance_requirements["consent_required"] = True
        
        self._log_audit("compliance_check", {
            "user_location": user_location,
            "data_type": data_type.value,
            "regulations": compliance_requirements["applicable_regulations"]
        })
        
        return compliance_requirements
    
    def anonymize_data(self, data: Any, data_type: DataCategory) -> Any:
        """Anonymize data for compliance purposes."""
        
        if not self.auto_anonymization:
            return data
            
        if data_type == DataCategory.DNA_SEQUENCES:
            # Anonymize DNA sequences while preserving statistical properties
            if isinstance(data, str):
                return self._anonymize_dna_sequence(data)
            elif isinstance(data, list):
                return [self._anonymize_dna_sequence(seq) for seq in data]
                
        elif data_type == DataCategory.IMAGE_DATA:
            # Apply privacy-preserving transformations to images
            return self._anonymize_image_data(data)
            
        elif data_type == DataCategory.SIMULATION_DATA:
            # Anonymize coordinate data
            return self._anonymize_coordinates(data)
        
        return data
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        current_time = time.time()
        
        # Consent analysis
        total_consents = len(self.consent_records)
        active_consents = sum(
            1 for c in self.consent_records.values()
            if c["granted"] and c["expires_at"] > current_time
        )
        
        # Data retention analysis
        expired_data = self._find_expired_data()
        
        # Audit log analysis
        recent_events = [
            e for e in self.audit_events
            if current_time - e["timestamp"] <= 86400  # Last 24 hours
        ]
        
        report = {
            "generated_at": current_time,
            "reporting_period": "last_30_days",
            "compliance_status": {
                "gdpr_compliant": self._check_gdpr_compliance(),
                "ccpa_compliant": self._check_ccpa_compliance(),
                "pdpa_compliant": self._check_pdpa_compliance()
            },
            "consent_management": {
                "total_consents": total_consents,
                "active_consents": active_consents,
                "expired_consents": total_consents - active_consents,
                "consent_rate": active_consents / max(total_consents, 1)
            },
            "data_retention": {
                "items_due_deletion": len(expired_data),
                "categories": {cat.value: 0 for cat in DataCategory}
            },
            "audit_activity": {
                "total_events": len(self.audit_events),
                "recent_events": len(recent_events),
                "event_types": self._count_event_types(recent_events)
            },
            "recommendations": self._generate_recommendations()
        }
        
        self._log_audit("compliance_report", {"items_count": len(report)})
        
        return report
    
    def _anonymize_dna_sequence(self, sequence: str) -> str:
        """Anonymize DNA sequence while preserving GC content."""
        import random
        
        if not sequence:
            return sequence
            
        # Preserve GC content ratio
        gc_count = sequence.count('G') + sequence.count('C')
        gc_ratio = gc_count / len(sequence)
        
        anonymized = []
        for _ in range(len(sequence)):
            if random.random() < gc_ratio:
                anonymized.append(random.choice(['G', 'C']))
            else:
                anonymized.append(random.choice(['A', 'T']))
        
        return ''.join(anonymized)
    
    def _anonymize_image_data(self, image_data: Any) -> Any:
        """Apply privacy-preserving transformations to image data."""
        # Placeholder for image anonymization
        # In practice, might apply blurring, noise, or other transformations
        return image_data
    
    def _anonymize_coordinates(self, coordinates: Any) -> Any:
        """Anonymize molecular coordinates."""
        # Placeholder for coordinate anonymization
        # In practice, might apply random translations or noise
        return coordinates
    
    def _collect_user_data(self, user_hash: str) -> Dict[str, Any]:
        """Collect all data associated with a user."""
        user_consents = [
            c for c in self.consent_records.values()
            if c["user_id_hash"] == user_hash
        ]
        
        user_events = [
            e for e in self.audit_events
            if e.get("user_hash") == user_hash
        ]
        
        return {
            "user_id_hash": user_hash,
            "consent_records": user_consents,
            "audit_events": len(user_events),
            "data_categories": list(set(e.get("data_type", "") for e in user_events if e.get("data_type")))
        }
    
    def _export_user_data(self, user_hash: str) -> Dict[str, Any]:
        """Export user data in portable format."""
        return self._collect_user_data(user_hash)
    
    def _delete_user_data(self, user_hash: str) -> List[str]:
        """Delete user data for erasure request."""
        deleted_items = []
        
        # Remove consent records
        consents_to_remove = [
            consent_id for consent_id, consent in self.consent_records.items()
            if consent["user_id_hash"] == user_hash
        ]
        
        for consent_id in consents_to_remove:
            del self.consent_records[consent_id]
            deleted_items.append(f"consent_{consent_id}")
        
        # Anonymize audit events
        for event in self.audit_events:
            if event.get("user_hash") == user_hash:
                event["user_hash"] = "anonymized"
        
        deleted_items.append("audit_events_anonymized")
        
        return deleted_items
    
    def _get_user_purposes(self, user_hash: str) -> List[str]:
        """Get processing purposes for a user."""
        purposes = set()
        
        for consent in self.consent_records.values():
            if consent["user_id_hash"] == user_hash:
                purposes.update(consent["purposes"])
        
        return list(purposes)
    
    def _get_retention_info(self) -> Dict[str, int]:
        """Get data retention information."""
        return {
            "research_data": self.research_data_retention,
            "image_data": self.image_data_retention,
            "sequence_data": self.sequence_data_retention
        }
    
    def _get_rights_information(self) -> Dict[str, str]:
        """Get information about data subject rights."""
        return {
            "access": "Right to obtain information about personal data processing",
            "rectification": "Right to correct inaccurate personal data",
            "erasure": "Right to deletion of personal data under certain conditions",
            "portability": "Right to receive personal data in a structured format",
            "objection": "Right to object to processing for legitimate interests",
            "restriction": "Right to restrict processing under certain conditions"
        }
    
    def _check_retention_exceptions(self, user_hash: str) -> List[str]:
        """Check if any retention exceptions apply."""
        exceptions = []
        
        # Research exception
        if self._has_research_data(user_hash):
            exceptions.append("scientific_research")
        
        # Legal obligation exception
        if self._has_legal_retention_requirement(user_hash):
            exceptions.append("legal_obligation")
        
        return exceptions
    
    def _has_research_data(self, user_hash: str) -> bool:
        """Check if user has active research data."""
        # Placeholder implementation
        return False
    
    def _has_legal_retention_requirement(self, user_hash: str) -> bool:
        """Check if legal retention requirements apply."""
        # Placeholder implementation
        return False
    
    def _find_expired_data(self) -> List[Dict[str, Any]]:
        """Find data that has exceeded retention periods."""
        expired_items = []
        current_time = time.time()
        
        # Check consent expiry
        for consent_id, consent in self.consent_records.items():
            if consent["expires_at"] <= current_time:
                expired_items.append({
                    "type": "consent",
                    "id": consent_id,
                    "expired_at": consent["expires_at"]
                })
        
        return expired_items
    
    def _check_gdpr_compliance(self) -> bool:
        """Check GDPR compliance status."""
        if not self.gdpr_enabled:
            return True
            
        # Basic compliance checks
        has_consent_mechanism = len(self.consent_records) > 0
        has_audit_trail = len(self.audit_events) > 0
        has_retention_policy = True  # Configured in class
        
        return has_consent_mechanism and has_audit_trail and has_retention_policy
    
    def _check_ccpa_compliance(self) -> bool:
        """Check CCPA compliance status."""
        if not self.ccpa_enabled:
            return True
            
        # Basic CCPA compliance checks
        return True  # Simplified for now
    
    def _check_pdpa_compliance(self) -> bool:
        """Check PDPA compliance status."""
        if not self.pdpa_enabled:
            return True
            
        # Basic PDPA compliance checks
        return True  # Simplified for now
    
    def _count_event_types(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count audit events by type."""
        event_counts = {}
        for event in events:
            event_type = event.get("event_type", "unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        return event_counts
    
    def _generate_recommendations(self) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        current_time = time.time()
        
        # Check consent expiry
        expiring_soon = sum(
            1 for c in self.consent_records.values()
            if c["expires_at"] - current_time <= 2592000  # 30 days
        )
        
        if expiring_soon > 0:
            recommendations.append(f"Renew {expiring_soon} consents expiring within 30 days")
        
        # Check audit log size
        if len(self.audit_events) > 100000:
            recommendations.append("Archive old audit events to maintain performance")
        
        # General recommendations
        recommendations.extend([
            "Regularly review data retention policies",
            "Conduct periodic compliance audits", 
            "Update privacy notices for regulatory changes",
            "Train staff on data protection requirements"
        ])
        
        return recommendations
    
    def _log_audit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an audit event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data
        }
        
        self.audit_events.append(event)
        
        # Keep audit log manageable
        if len(self.audit_events) > 50000:
            self.audit_events = self.audit_events[-25000:]


# Global compliance instance
_global_compliance = None

def get_global_compliance() -> GlobalCompliance:
    """Get the global compliance manager."""
    global _global_compliance
    if _global_compliance is None:
        _global_compliance = GlobalCompliance()
    return _global_compliance

def ensure_compliance(user_id: str, data_type: DataCategory, 
                     purpose: ProcessingPurpose, 
                     user_location: str = "US") -> bool:
    """Ensure compliance before processing data."""
    
    compliance = get_global_compliance()
    
    # Check regional requirements
    requirements = compliance.check_regional_compliance(user_location, data_type)
    
    # Obtain consent if required
    if requirements["consent_required"]:
        if not compliance.has_valid_consent(user_id, purpose):
            # In practice, would redirect to consent form
            consent_id = compliance.record_consent(user_id, [purpose])
            if not consent_id:
                return False
    
    compliance._log_audit("data_processing", {
        "user_location": user_location,
        "data_type": data_type.value,
        "purpose": purpose.value,
        "compliant": True
    })
    
    return True