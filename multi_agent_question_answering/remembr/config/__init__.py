"""
ReMEmbR Configuration Module

This module provides configuration management for ReMEmbR,
supporting YAML files, environment variables, and CLI arguments
with proper precedence handling.
"""

from .config_loader import load_config, get_config, Config
from .defaults import get_default_config

__all__ = ['load_config', 'get_config', 'Config', 'get_default_config']
