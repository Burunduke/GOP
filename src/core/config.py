"""
Configuration module for managing project settings with proper dependency injection support
"""

import os
import yaml
import threading
from typing import Dict, Any, Optional, Union, List, TypeVar
from pathlib import Path

# Type aliases for better type safety
ConfigDict = Dict[str, Any]
ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any], None]
T = TypeVar("T")


class Config:
    """
    Thread-safe configuration management class with dependency injection support
    """

    _instance: Optional["Config"] = None
    _lock = threading.Lock()

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._get_default_config_path()
        self._config = self._load_config()
        self._lock = threading.Lock()

    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / "config.yaml"
        return str(config_file)

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file

        Returns:
            Dictionary with settings
        """
        if not os.path.exists(self.config_path):
            return self._get_default_config()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            # Use logging instead of print
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration

        Returns:
            Dictionary with default settings
        """
        return {
            "processing": {
                "max_image_size": 10000,
                "compression_ratio": 0.125,
                "batch_size": 32,
                "num_workers": 4,
            },
            "segmentation": {
                "model_path": "models/deeplabv3_resnet101.pth",
                "device": "auto",
                "confidence_threshold": 0.5,
            },
            "output": {
                "results_dir": "results",
                "save_intermediate": True,
                "output_format": "GeoTIFF",
            },
            "logging": {"level": "INFO", "file": "logs/gop.log"},
        }

    def get(self, key: str, default: T = None) -> Union[ConfigValue, T]:
        """
        Get configuration parameter value

        Args:
            key: Parameter key (supports nested keys with dots)
            default: Default value

        Returns:
            Parameter value
        """
        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: ConfigValue) -> None:
        """
        Set configuration parameter value

        Args:
            key: Parameter key (supports nested keys with dots)
            value: Parameter value
        """
        with self._lock:
            keys = key.split(".")
            config = self._config

            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to file

        Args:
            path: Path for saving (if None, uses current path)
        """
        save_path = path or self.config_path

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self._config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )
        except Exception as e:
            # Use logging instead of print
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error saving configuration: {e}")

    def update(self, config_dict: ConfigDict) -> None:
        """
        Update configuration from dictionary

        Args:
            config_dict: Dictionary with new settings
        """
        with self._lock:
            self._deep_update(self._config, config_dict)

    def _deep_update(self, base_dict: ConfigDict, update_dict: ConfigDict) -> None:
        """
        Recursive dictionary update

        Args:
            base_dict: Base dictionary
            update_dict: Update dictionary
        """
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    @property
    def config(self) -> ConfigDict:
        """Get full configuration dictionary"""
        with self._lock:
            return self._config.copy()

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> "Config":
        """
        Get singleton instance of Config (thread-safe)

        Args:
            config_path: Optional path to configuration file

        Returns:
            Singleton Config instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)"""
        with cls._lock:
            cls._instance = None


def create_config(config_path: Optional[str] = None) -> Config:
    """
    Factory function for creating configuration instance

    Args:
        config_path: Path to configuration file

    Returns:
        New Config instance
    """
    return Config(config_path)


def get_config(config_instance: Optional[Config] = None) -> Config:
    """
    Get configuration instance with dependency injection support

    Args:
        config_instance: Optional configuration instance for injection

    Returns:
        Configuration instance (injected or singleton)
    """
    if config_instance is not None:
        return config_instance
    return Config.get_instance()


# Remove global state - use dependency injection instead
# Applications should use get_config() or Config.get_instance()
