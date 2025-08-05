"""Internationalization (i18n) support for DNA origami autoencoder."""

import os
import json
from typing import Dict, Optional, Any, List
from pathlib import Path
from functools import lru_cache
import locale
import gettext

from .helpers import logger


class TranslationManager:
    """Manage translations and localization."""
    
    def __init__(self, default_locale: str = 'en_US'):
        """Initialize translation manager.
        
        Args:
            default_locale: Default locale to use
        """
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations = {}
        self.fallback_translations = {}
        
        # Setup paths
        self.locale_dir = Path(__file__).parent.parent / 'locales'
        self.locale_dir.mkdir(exist_ok=True)
        
        # Load system locale
        try:
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                self.current_locale = system_locale
        except Exception:
            pass
        
        # Load translations
        self._load_translations()
    
    def _load_translations(self):
        """Load all available translations."""
        # Load default translations (English)
        self._load_locale_translations('en_US')
        self.fallback_translations = self.translations.get('en_US', {})
        
        # Load current locale if different from default
        if self.current_locale != 'en_US':
            self._load_locale_translations(self.current_locale)
    
    def _load_locale_translations(self, locale_code: str):
        """Load translations for specific locale."""
        locale_file = self.locale_dir / f'{locale_code}.json'
        
        if locale_file.exists():
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                self.translations[locale_code] = translations
                logger.debug(f"Loaded translations for {locale_code}")
            except Exception as e:
                logger.warning(f"Failed to load translations for {locale_code}: {e}")
        else:
            # Create default translation file if it doesn't exist
            if locale_code == 'en_US':
                self._create_default_translations()
    
    def _create_default_translations(self):
        """Create default English translations."""
        default_translations = {
            # General messages
            "welcome": "Welcome to DNA Origami AutoEncoder",
            "error": "Error",
            "warning": "Warning",
            "info": "Information",
            "success": "Success",
            "loading": "Loading...",
            "processing": "Processing...",
            "completed": "Completed",
            "failed": "Failed",
            "cancelled": "Cancelled",
            
            # DNA encoding
            "encoding_image": "Encoding image to DNA sequences",
            "encoding_complete": "DNA encoding completed successfully",
            "encoding_failed": "DNA encoding failed",
            "invalid_dna_sequence": "Invalid DNA sequence",
            "sequence_too_short": "DNA sequence is too short",
            "sequence_too_long": "DNA sequence is too long",
            "gc_content_violation": "GC content outside acceptable range",
            "homopolymer_violation": "Homopolymer run too long",
            
            # Image processing
            "loading_image": "Loading image",
            "image_loaded": "Image loaded successfully",
            "invalid_image_format": "Invalid image format",
            "image_too_large": "Image is too large",
            "preprocessing_image": "Preprocessing image",
            
            # Origami design
            "designing_structure": "Designing origami structure",
            "structure_created": "Origami structure created",
            "design_validation_failed": "Structure design validation failed",
            "optimizing_staples": "Optimizing staple design",
            "calculating_folding": "Calculating folding parameters",
            
            # Simulation
            "starting_simulation": "Starting molecular dynamics simulation",
            "simulation_progress": "Simulation progress: {progress}%",
            "simulation_completed": "Simulation completed successfully",
            "simulation_failed": "Simulation failed",
            "analyzing_trajectory": "Analyzing simulation trajectory",
            
            # Decoding
            "decoding_structure": "Decoding structure to image",
            "training_decoder": "Training transformer decoder",
            "decoder_training_complete": "Decoder training completed",
            "reconstruction_quality": "Reconstruction quality: {quality}",
            
            # File operations  
            "saving_file": "Saving file: {filename}",
            "file_saved": "File saved successfully",
            "loading_file": "Loading file: {filename}",
            "file_loaded": "File loaded successfully",
            "file_not_found": "File not found",
            "permission_denied": "Permission denied",
            
            # Performance
            "optimizing_performance": "Optimizing performance",
            "memory_usage": "Memory usage: {usage} MB",
            "cache_hit_rate": "Cache hit rate: {rate}%",
            "processing_time": "Processing time: {time}",
            
            # Validation
            "validating_input": "Validating input",
            "validation_passed": "Validation passed",
            "validation_failed": "Validation failed",
            "parameter_out_of_range": "Parameter out of valid range",
            
            # Units and measurements
            "nanometers": "nanometers",
            "angstroms": "angstroms", 
            "base_pairs": "base pairs",
            "kilobytes": "KB",
            "megabytes": "MB",
            "gigabytes": "GB",
            "seconds": "seconds",
            "minutes": "minutes",
            "hours": "hours",
            
            # Scientific terms
            "dna_sequence": "DNA sequence",
            "origami_structure": "origami structure",
            "molecular_dynamics": "molecular dynamics",
            "transformer_decoder": "transformer decoder",
            "autoencoder": "autoencoder",
            "biotechnology": "biotechnology",
            "nanotechnology": "nanotechnology"
        }
        
        en_file = self.locale_dir / 'en_US.json'
        with open(en_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, indent=2, ensure_ascii=False)
        
        self.translations['en_US'] = default_translations
        logger.info("Created default English translations")
    
    def set_locale(self, locale_code: str):
        """Set current locale."""
        if locale_code != self.current_locale:
            self.current_locale = locale_code
            self._load_locale_translations(locale_code)
            logger.info(f"Locale changed to {locale_code}")
    
    def get_available_locales(self) -> List[str]:
        """Get list of available locales."""
        locales = []
        for locale_file in self.locale_dir.glob('*.json'):
            locale_code = locale_file.stem
            locales.append(locale_code)
        return sorted(locales)
    
    @lru_cache(maxsize=1000)
    def translate(self, key: str, **kwargs) -> str:
        """Translate a key to current locale.
        
        Args:
            key: Translation key
            **kwargs: Format parameters
            
        Returns:
            Translated string
        """
        # Try current locale first
        current_translations = self.translations.get(self.current_locale, {})
        
        if key in current_translations:
            text = current_translations[key]
        elif key in self.fallback_translations:
            text = self.fallback_translations[key]
        else:
            # Return key as fallback
            text = key
            logger.warning(f"Missing translation for key: {key}")
        
        # Format with parameters
        try:
            return text.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing format parameter {e} for key: {key}")
            return text
    
    def get_locale_info(self, locale_code: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a locale."""
        locale_code = locale_code or self.current_locale
        
        # Basic locale information mapping
        locale_info = {
            'en_US': {
                'name': 'English (United States)',
                'native_name': 'English (United States)',
                'direction': 'ltr',
                'region': 'US'
            },
            'es_ES': {
                'name': 'Spanish (Spain)',
                'native_name': 'Español (España)',
                'direction': 'ltr',
                'region': 'ES'
            },
            'fr_FR': {
                'name': 'French (France)',
                'native_name': 'Français (France)',
                'direction': 'ltr',
                'region': 'FR'
            },
            'de_DE': {
                'name': 'German (Germany)',
                'native_name': 'Deutsch (Deutschland)',
                'direction': 'ltr',
                'region': 'DE'
            },
            'ja_JP': {
                'name': 'Japanese (Japan)',
                'native_name': '日本語 (日本)',
                'direction': 'ltr',
                'region': 'JP'
            },
            'zh_CN': {
                'name': 'Chinese (Simplified)',
                'native_name': '中文 (简体)',
                'direction': 'ltr',
                'region': 'CN'
            },
            'zh_TW': {
                'name': 'Chinese (Traditional)',
                'native_name': '中文 (繁體)',
                'direction': 'ltr',
                'region': 'TW'
            },
            'ko_KR': {
                'name': 'Korean (South Korea)',
                'native_name': '한국어 (대한민국)',
                'direction': 'ltr',
                'region': 'KR'
            },
            'ru_RU': {
                'name': 'Russian (Russia)',
                'native_name': 'Русский (Россия)',
                'direction': 'ltr',
                'region': 'RU'
            },
            'ar_SA': {
                'name': 'Arabic (Saudi Arabia)',
                'native_name': 'العربية (المملكة العربية السعودية)',
                'direction': 'rtl',
                'region': 'SA'
            }
        }
        
        return locale_info.get(locale_code, {
            'name': locale_code,
            'native_name': locale_code,
            'direction': 'ltr',
            'region': 'Unknown'
        })


# Global translation manager instance
_translation_manager = TranslationManager()


def _(key: str, **kwargs) -> str:
    """Shorthand translation function.
    
    Args:
        key: Translation key
        **kwargs: Format parameters
        
    Returns:
        Translated string
    """
    return _translation_manager.translate(key, **kwargs)


def set_locale(locale_code: str):
    """Set the current locale."""
    _translation_manager.set_locale(locale_code)


def get_current_locale() -> str:
    """Get the current locale."""
    return _translation_manager.current_locale


def get_available_locales() -> List[str]:
    """Get list of available locales."""
    return _translation_manager.get_available_locales()


def get_locale_info(locale_code: Optional[str] = None) -> Dict[str, Any]:
    """Get information about a locale."""
    return _translation_manager.get_locale_info(locale_code)


class LocalizedLogger:
    """Logger that supports localized messages."""
    
    def __init__(self, name: str):
        self.logger = logger.getChild(name)
    
    def info(self, key: str, **kwargs):
        """Log localized info message."""
        message = _(key, **kwargs)
        self.logger.info(message)
    
    def warning(self, key: str, **kwargs):
        """Log localized warning message."""
        message = _(key, **kwargs)
        self.logger.warning(message)
    
    def error(self, key: str, **kwargs):
        """Log localized error message."""
        message = _(key, **kwargs)
        self.logger.error(message)
    
    def debug(self, key: str, **kwargs):
        """Log localized debug message."""
        message = _(key, **kwargs)
        self.logger.debug(message)


def format_number(number: float, locale_code: Optional[str] = None) -> str:
    """Format number according to locale conventions."""
    locale_code = locale_code or get_current_locale()
    
    # Basic number formatting for different locales
    if locale_code.startswith('de'):
        # German uses comma for decimal, period for thousands
        return f"{number:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    elif locale_code.startswith('fr'):
        # French uses comma for decimal, space for thousands
        return f"{number:,.2f}".replace(',', ' ').replace('.', ',')
    else:
        # Default (English) formatting
        return f"{number:,.2f}"


def format_file_size(bytes_value: int, locale_code: Optional[str] = None) -> str:
    """Format file size with localized units."""
    locale_code = locale_code or get_current_locale()
    
    units = {
        'en_US': ['bytes', 'KB', 'MB', 'GB', 'TB'],
        'de_DE': ['Bytes', 'KB', 'MB', 'GB', 'TB'],
        'fr_FR': ['octets', 'Ko', 'Mo', 'Go', 'To'],
        'ja_JP': ['バイト', 'KB', 'MB', 'GB', 'TB'],
        'zh_CN': ['字节', 'KB', 'MB', 'GB', 'TB']
    }
    
    locale_units = units.get(locale_code, units['en_US'])
    
    size = float(bytes_value)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(locale_units) - 1:
        size /= 1024
        unit_index += 1
    
    formatted_size = format_number(size, locale_code)
    return f"{formatted_size} {locale_units[unit_index]}"


def localize_scientific_units(value: float, unit: str, 
                            locale_code: Optional[str] = None) -> str:
    """Localize scientific units and measurements."""
    locale_code = locale_code or get_current_locale()
    
    # Unit translations
    unit_translations = {
        'en_US': {
            'nm': 'nm',
            'angstrom': 'Å',
            'bp': 'bp',
            'kDa': 'kDa',
            'celsius': '°C',
            'kelvin': 'K'
        },
        'de_DE': {
            'nm': 'nm',
            'angstrom': 'Å',
            'bp': 'Bp',
            'kDa': 'kDa',
            'celsius': '°C',
            'kelvin': 'K'
        },
        'fr_FR': {
            'nm': 'nm',
            'angstrom': 'Å',
            'bp': 'pb',
            'kDa': 'kDa',
            'celsius': '°C',
            'kelvin': 'K'
        },
        'ja_JP': {
            'nm': 'nm',
            'angstrom': 'Å',
            'bp': 'bp',
            'kDa': 'kDa',
            'celsius': '°C',
            'kelvin': 'K'
        }
    }
    
    locale_units = unit_translations.get(locale_code, unit_translations['en_US'])
    localized_unit = locale_units.get(unit, unit)
    formatted_value = format_number(value, locale_code)
    
    return f"{formatted_value} {localized_unit}"


def create_translation_template() -> Dict[str, str]:
    """Create template for new translations."""
    return {
        "locale_info": {
            "locale_code": "",
            "locale_name": "",
            "native_name": "",
            "direction": "ltr",
            "region": ""
        },
        "metadata": {
            "version": "1.0",
            "translator": "",
            "last_updated": "",
            "completion_percentage": 0
        },
        "translations": _translation_manager.fallback_translations.copy()
    }