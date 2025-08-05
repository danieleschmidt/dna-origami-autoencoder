"""Cross-platform compatibility utilities."""

import os
import sys
import platform
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import shutil

from .helpers import logger


class PlatformInfo:
    """Information about the current platform."""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.processor = platform.processor()
        self.python_version = platform.python_version()
        self.python_implementation = platform.python_implementation()
        
    def is_windows(self) -> bool:
        """Check if running on Windows."""
        return self.system == 'Windows'
    
    def is_macos(self) -> bool:
        """Check if running on macOS."""
        return self.system == 'Darwin'
    
    def is_linux(self) -> bool:
        """Check if running on Linux."""
        return self.system == 'Linux'
    
    def is_unix_like(self) -> bool:
        """Check if running on Unix-like system."""
        return self.system in ['Linux', 'Darwin', 'FreeBSD', 'OpenBSD']
    
    def get_architecture(self) -> str:
        """Get system architecture."""
        arch_map = {
            'x86_64': 'x64',
            'AMD64': 'x64',
            'aarch64': 'arm64',
            'arm64': 'arm64',
            'i386': 'x86',
            'i686': 'x86'
        }
        return arch_map.get(self.machine, self.machine)
    
    def supports_multiprocessing(self) -> bool:
        """Check if platform supports multiprocessing."""
        try:
            import multiprocessing
            return multiprocessing.cpu_count() > 1
        except (ImportError, NotImplementedError):
            return False
    
    def get_cpu_count(self) -> int:
        """Get number of CPU cores."""
        try:
            return os.cpu_count() or 1
        except:
            return 1
    
    def get_memory_info(self) -> Dict[str, int]:
        """Get memory information."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            }
        except ImportError:
            logger.warning("psutil not available, cannot get memory info")
            return {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'system': self.system,
            'machine': self.machine,
            'processor': self.processor,
            'architecture': self.get_architecture(),
            'python_version': self.python_version,
            'python_implementation': self.python_implementation,
            'cpu_count': self.get_cpu_count(),
            'supports_multiprocessing': self.supports_multiprocessing(),
            'memory_info': self.get_memory_info()
        }


class PathManager:
    """Cross-platform path management."""
    
    def __init__(self):
        self.platform_info = PlatformInfo()
    
    def get_home_dir(self) -> Path:
        """Get user home directory."""
        return Path.home()
    
    def get_app_data_dir(self, app_name: str = "dna_origami_ae") -> Path:
        """Get application data directory."""
        if self.platform_info.is_windows():
            # Windows: %APPDATA%\\AppName
            base_dir = Path(os.environ.get('APPDATA', Path.home()))
        elif self.platform_info.is_macos():
            # macOS: ~/Library/Application Support/AppName
            base_dir = Path.home() / 'Library' / 'Application Support'
        else:
            # Linux/Unix: ~/.config/AppName
            base_dir = Path.home() / '.config'
        
        app_dir = base_dir / app_name
        app_dir.mkdir(parents=True, exist_ok=True)
        return app_dir
    
    def get_cache_dir(self, app_name: str = "dna_origami_ae") -> Path:
        """Get cache directory."""
        if self.platform_info.is_windows():
            # Windows: %LOCALAPPDATA%\\AppName\\Cache
            base_dir = Path(os.environ.get('LOCALAPPDATA', Path.home()))
            cache_dir = base_dir / app_name / 'Cache'
        elif self.platform_info.is_macos():
            # macOS: ~/Library/Caches/AppName
            cache_dir = Path.home() / 'Library' / 'Caches' / app_name
        else:
            # Linux/Unix: ~/.cache/AppName
            cache_dir = Path.home() / '.cache' / app_name
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def get_temp_dir(self) -> Path:
        """Get temporary directory."""
        import tempfile
        return Path(tempfile.gettempdir())
    
    def get_documents_dir(self) -> Path:
        """Get documents directory."""
        if self.platform_info.is_windows():
            # Try to get Documents folder
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                   r"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders")
                documents_path = winreg.QueryValueEx(key, "Personal")[0]
                winreg.CloseKey(key)
                return Path(documents_path)
            except:
                return Path.home() / 'Documents'
        else:
            return Path.home() / 'Documents'
    
    def normalize_path(self, path: Path) -> Path:
        """Normalize path for current platform."""
        return path.resolve()
    
    def is_path_safe(self, path: Path) -> bool:
        """Check if path is safe (no directory traversal)."""
        try:
            resolved = path.resolve()
            # Check for directory traversal attempts
            return not any(part == '..' for part in resolved.parts)
        except:
            return False


class ProcessManager:
    """Cross-platform process management."""
    
    def __init__(self):
        self.platform_info = PlatformInfo()
    
    def run_command(self, command: List[str], timeout: Optional[float] = None,
                   capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run command cross-platform."""
        try:
            if self.platform_info.is_windows():
                # On Windows, ensure proper shell handling
                return subprocess.run(
                    command,
                    timeout=timeout,
                    capture_output=capture_output,
                    text=True,
                    shell=False,  # Safer than shell=True
                    check=False
                )
            else:
                # Unix-like systems
                return subprocess.run(
                    command,
                    timeout=timeout,
                    capture_output=capture_output,
                    text=True,
                    check=False
                )
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(command)}")
            raise
        except FileNotFoundError:
            logger.error(f"Command not found: {command[0]}")
            raise
    
    def find_executable(self, name: str) -> Optional[Path]:
        """Find executable in PATH."""
        executable = shutil.which(name)
        return Path(executable) if executable else None
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables."""
        return dict(os.environ)
    
    def set_environment_variable(self, name: str, value: str):
        """Set environment variable."""
        os.environ[name] = value
    
    def get_python_executable(self) -> Path:
        """Get Python executable path."""
        return Path(sys.executable)


class ResourceManager:
    """Cross-platform resource management."""
    
    def __init__(self):
        self.platform_info = PlatformInfo()
        self.path_manager = PathManager()
    
    def get_available_space(self, path: Path) -> int:
        """Get available disk space in bytes."""
        try:
            if self.platform_info.is_windows():
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    str(path), ctypes.pointer(free_bytes), None, None
                )
                return free_bytes.value
            else:
                statvfs = os.statvfs(path)
                return statvfs.f_frsize * statvfs.f_bavail
        except:
            return 0
    
    def check_write_permissions(self, path: Path) -> bool:
        """Check if path is writable."""
        try:
            if path.is_file():
                return os.access(path, os.W_OK)
            elif path.is_dir():
                # Try to create a temporary file
                test_file = path / '.test_write_permission'
                try:
                    test_file.touch()
                    test_file.unlink()
                    return True
                except:
                    return False
            else:
                # Check parent directory
                return self.check_write_permissions(path.parent)
        except:
            return False
    
    def get_system_font_dirs(self) -> List[Path]:
        """Get system font directories."""
        font_dirs = []
        
        if self.platform_info.is_windows():
            font_dirs.extend([
                Path(os.environ.get('WINDIR', 'C:\\Windows')) / 'Fonts',
                Path.home() / 'AppData' / 'Local' / 'Microsoft' / 'Windows' / 'Fonts'
            ])
        elif self.platform_info.is_macos():
            font_dirs.extend([
                Path('/System/Library/Fonts'),
                Path('/Library/Fonts'),
                Path.home() / 'Library' / 'Fonts'
            ])
        else:  # Linux/Unix
            font_dirs.extend([
                Path('/usr/share/fonts'),
                Path('/usr/local/share/fonts'),
                Path.home() / '.fonts',
                Path.home() / '.local' / 'share' / 'fonts'
            ])
        
        return [d for d in font_dirs if d.exists()]


class DependencyChecker:
    """Check for required dependencies across platforms."""
    
    def __init__(self):
        self.platform_info = PlatformInfo()
        self.process_manager = ProcessManager()
    
    def check_python_packages(self, packages: List[str]) -> Dict[str, Dict[str, Any]]:
        """Check if Python packages are installed."""
        results = {}
        
        for package in packages:
            try:
                import importlib
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                results[package] = {
                    'installed': True,
                    'version': version,
                    'location': getattr(module, '__file__', 'unknown')
                }
            except ImportError:
                results[package] = {
                    'installed': False,
                    'version': None,
                    'location': None
                }
        
        return results
    
    def check_system_commands(self, commands: List[str]) -> Dict[str, Dict[str, Any]]:
        """Check if system commands are available."""
        results = {}
        
        for command in commands:
            executable_path = self.process_manager.find_executable(command)
            
            if executable_path:
                # Try to get version
                version = None
                try:
                    result = self.process_manager.run_command(
                        [command, '--version'], timeout=5
                    )
                    if result.returncode == 0:
                        version = result.stdout.strip()
                except:
                    try:
                        result = self.process_manager.run_command(
                            [command, '-v'], timeout=5
                        )
                        if result.returncode == 0:
                            version = result.stdout.strip()
                    except:
                        pass
                
                results[command] = {
                    'available': True,
                    'path': str(executable_path),
                    'version': version
                }
            else:
                results[command] = {
                    'available': False,
                    'path': None,
                    'version': None
                }
        
        return results
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive dependency report."""
        python_packages = [
            'numpy', 'scipy', 'matplotlib', 'pillow', 'psutil'
        ]
        
        system_commands = ['git', 'ffmpeg', 'convert']
        
        return {
            'platform_info': self.platform_info.to_dict(),
            'python_packages': self.check_python_packages(python_packages),
            'system_commands': self.check_system_commands(system_commands)
        }


class ConfigManager:
    """Cross-platform configuration management."""
    
    def __init__(self, app_name: str = "dna_origami_ae"):
        self.app_name = app_name
        self.path_manager = PathManager()
        self.platform_info = PlatformInfo()
        
        # Get platform-specific config file
        self.config_dir = self.path_manager.get_app_data_dir(app_name)
        self.config_file = self.config_dir / 'config.json'
        
        # Default configuration
        self.default_config = {
            'general': {
                'locale': 'en_US',
                'theme': 'light',
                'log_level': 'INFO'
            },
            'performance': {
                'max_workers': min(32, self.platform_info.get_cpu_count() + 4),
                'cache_size_mb': 500,
                'use_gpu': False
            },
            'paths': {
                'cache_dir': str(self.path_manager.get_cache_dir(app_name)),
                'temp_dir': str(self.path_manager.get_temp_dir()),
                'output_dir': str(self.path_manager.get_documents_dir() / app_name)
            },
            'compliance': {
                'enabled_regulations': ['gdpr', 'ccpa', 'pdpa'],
                'data_retention_days': 730,
                'anonymize_sensitive_data': True
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                import json
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Merge with defaults
                return self._merge_configs(self.default_config, config)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return self.default_config.copy()
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            import json
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with defaults."""
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result


# Global instances
platform_info = PlatformInfo()
path_manager = PathManager()
resource_manager = ResourceManager()
dependency_checker = DependencyChecker()
config_manager = ConfigManager()


def get_platform_summary() -> Dict[str, Any]:
    """Get comprehensive platform summary."""
    return {
        'platform': platform_info.to_dict(),
        'dependencies': dependency_checker.get_comprehensive_report(),
        'paths': {
            'home': str(path_manager.get_home_dir()),
            'app_data': str(path_manager.get_app_data_dir()),
            'cache': str(path_manager.get_cache_dir()),
            'temp': str(path_manager.get_temp_dir()),
            'documents': str(path_manager.get_documents_dir())
        },
        'resources': {
            'available_space_gb': resource_manager.get_available_space(Path.home()) / (1024**3),
            'font_directories': [str(d) for d in resource_manager.get_system_font_dirs()]
        }
    }


def ensure_cross_platform_compatibility():
    """Ensure the application works across platforms."""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8 or higher is required")
    
    # Check critical dependencies
    deps = dependency_checker.check_python_packages(['pathlib'])
    if not deps.get('pathlib', {}).get('installed', False):
        issues.append("pathlib module is required but not available")
    
    # Check file system permissions
    app_data_dir = path_manager.get_app_data_dir()
    if not resource_manager.check_write_permissions(app_data_dir):
        issues.append(f"No write permission for app data directory: {app_data_dir}")
    
    # Platform-specific checks
    if platform_info.is_windows():
        # Windows-specific checks
        pass
    elif platform_info.is_macos():
        # macOS-specific checks
        pass
    elif platform_info.is_linux():
        # Linux-specific checks
        pass
    
    if issues:
        logger.warning(f"Cross-platform compatibility issues: {issues}")
        return False
    
    logger.info("Cross-platform compatibility check passed")
    return True