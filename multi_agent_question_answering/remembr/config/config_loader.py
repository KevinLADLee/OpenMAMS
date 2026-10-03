"""
Configuration loader for ReMEmbR

Supports loading configuration from:
1. YAML files
2. Environment variables
3. Command-line arguments

Priority: CLI args > Environment variables > Config file > Defaults
"""
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dotenv import load_dotenv

from .defaults import get_default_config


# Global configuration instance
_config_instance: Optional['Config'] = None


def _get_logger():
    """Lazy import to avoid circular dependency."""
    try:
        from remembr.utils.logging import config_logger
        return config_logger
    except ImportError:
        import logging
        return logging.getLogger(__name__)


class Config:
    """Configuration container with nested dictionary access"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., "llm.backend")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        Set a configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., "llm.backend")
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a dictionary"""
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self._config[key]

    def __setitem__(self, key: str, value: Any):
        """Allow dict-like setting"""
        self._config[key] = value


def _expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand environment variables in configuration.

    Supports ${VAR_NAME} and ${VAR_NAME:default} syntax.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with expanded environment variables
    """
    if isinstance(config, dict):
        return {k: _expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Match ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replace(match):
            var_name = match.group(1)
            default_value = match.group(2) or ''
            return os.getenv(var_name, default_value)

        return re.sub(pattern, replace, config)
    else:
        return config


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Configuration to merge on top

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def _load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        file_path: Path to YAML file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config or {}


def _load_env_overrides() -> Dict[str, Any]:
    """
    Load configuration overrides from environment variables.

    Supports REMEMBR_* environment variables with dot notation.
    Examples:
        REMEMBR_LLM_BACKEND=llama3 -> llm.backend
        REMEMBR_LLM_API_KEY=abc -> llm.api_key

    Returns:
        Configuration overrides from environment
    """
    overrides = {}

    for env_key, env_value in os.environ.items():
        if not env_key.startswith('REMEMBR_'):
            continue

        # Remove REMEMBR_ prefix and convert to lowercase
        config_key = env_key[len('REMEMBR_'):].lower()

        # Convert underscores to dot notation
        keys = config_key.split('_')

        # Build nested dictionary
        current = overrides
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        # Try to parse as number or boolean
        value = env_value
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():
            value = float(value)

        current[keys[-1]] = value

    return overrides


def load_config(
    config_path: Optional[str] = None,
    preset: Optional[str] = None,
    env_file: Optional[str] = None,
    validate: bool = True
) -> Config:
    """
    Load configuration from file, environment variables, and defaults.

    Priority: Environment variables > Config file > Defaults

    Args:
        config_path: Path to YAML config file (default: ./config.yaml or REMEMBR_CONFIG_PATH)
        preset: Preset name to use from config file (e.g., 'local', 'cloud')
        env_file: Path to .env file (default: ./.env)
        validate: Whether to validate configuration (default: True)

    Returns:
        Config instance

    Raises:
        ConfigValidationError: If validation fails and validate=True

    Example:
        >>> config = load_config()
        >>> llm_backend = config.get('llm.backend')
        >>> config = load_config(preset='cloud')  # Use cloud preset
    """
    global _config_instance
    logger = _get_logger()

    # Load .env file if exists
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()  # Load from .env in current directory

    # Start with defaults
    config_dict = get_default_config()

    # Determine config file path
    if config_path is None:
        config_path = os.getenv('REMEMBR_CONFIG_PATH', 'config.yaml')

    config_file = Path(config_path)

    # Load config file if it exists
    if config_file.exists():
        try:
            file_config = _load_yaml_file(config_file)
            logger.debug(f"Loaded config file: {config_file}")

            # Apply preset if specified
            if preset:
                if 'presets' in file_config and preset in file_config['presets']:
                    preset_config = file_config['presets'][preset]
                    config_dict = _merge_configs(config_dict, preset_config)
                    logger.info(f"Applied preset: {preset}")
                else:
                    logger.warning(f"Preset '{preset}' not found in config file")

            # Merge main config (excluding presets)
            main_config = {k: v for k, v in file_config.items() if k != 'presets'}
            config_dict = _merge_configs(config_dict, main_config)

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config file {config_file}: {e}")
            logger.warning("Using default configuration.")
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {e}")
            logger.warning("Using default configuration.")
    else:
        logger.debug(f"Config file not found: {config_file}, using defaults")

    # Expand environment variables in config
    config_dict = _expand_env_vars(config_dict)

    # Apply environment variable overrides
    env_overrides = _load_env_overrides()
    if env_overrides:
        config_dict = _merge_configs(config_dict, env_overrides)
        logger.debug(f"Applied environment variable overrides")

    # Validate configuration if requested
    if validate:
        try:
            from .validation import validate_config, ConfigValidationError
            is_valid, messages = validate_config(config_dict, strict=False)
            if not is_valid:
                error_msg = "Configuration validation failed:\n" + "\n".join(messages)
                logger.error(error_msg)
                raise ConfigValidationError(error_msg)
        except ImportError:
            # Validation module not available, skip validation
            logger.debug("Config validation module not available, skipping validation")

    # Create Config instance
    _config_instance = Config(config_dict)
    logger.debug("Configuration loaded successfully")
    return _config_instance


def get_config() -> Config:
    """
    Get the current global configuration instance.

    Returns:
        Config instance

    Raises:
        RuntimeError: If configuration hasn't been loaded yet
    """
    if _config_instance is None:
        # Auto-load with defaults if not explicitly loaded
        return load_config()

    return _config_instance
