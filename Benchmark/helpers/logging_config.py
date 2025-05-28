import logging
import os
from pathlib import Path

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Map string log levels to logging module constants
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def configure_logging(log_level=None):
    """
    Configure the logging system for the benchmark
    
    Args:
        log_level: Logging level (default: from environment or INFO)
    """
    # Get log level from environment variable or use INFO as default
    if log_level is None:
        env_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_level = LOG_LEVELS.get(env_level.lower(), logging.INFO)
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicate logs
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler
    log_file = LOGS_DIR / "benchmark.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('azure.identity').setLevel(logging.ERROR)
    logging.getLogger('azure.core').setLevel(logging.ERROR)
    logging.getLogger('msrest').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Suppress LiteLLM INFO logs
    litellm_logger = logging.getLogger('litellm')
    litellm_logger.setLevel(logging.WARNING)
    # Ensure its handlers also respect this level
    for handler in litellm_logger.handlers:
        handler.setLevel(logging.WARNING)
    # To prevent litellm from adding its own handlers that might ignore the level:
    litellm_logger.propagate = False # Uncomment if logs are still too verbose, but be cautious
    
    return root_logger

# Get module-specific loggers
def get_logger(name):
    """Get a logger with the specified name"""
    # Ensure logging is configured
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
