"""
Structured logging configuration for ReMEmbR.

Provides consistent logging across all components with helpful formatting
and configurable log levels.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes for console output
class LogColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.GRAY,
        logging.INFO: LogColors.BLUE,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BOLD + LogColors.RED,
    }

    def format(self, record):
        # Add color to level name
        if sys.stderr.isatty():  # Only use colors in terminal
            levelname = record.levelname
            color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
            record.levelname = f"{color}{levelname}{LogColors.RESET}"

        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for ReMEmbR.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        format_string: Custom format string (uses default if None)
    """
    # Default format: [TIME] LEVEL - Component: Message
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)-8s - %(name)s: %(message)s"

    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_formatter = ColoredFormatter(
        format_string,
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            format_string,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a component.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Example:
        >>> from remembr.utils.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Component initialized")
    """
    # Shorten remembr.* prefix for cleaner logs
    if name.startswith("remembr."):
        name = name[8:]  # Remove "remembr." prefix

    return logging.getLogger(name)


# Pre-configured loggers for common components
agent_logger = get_logger("agents")
memory_logger = get_logger("memory")
captioner_logger = get_logger("captioners")
config_logger = get_logger("config")
factory_logger = get_logger("factories")
