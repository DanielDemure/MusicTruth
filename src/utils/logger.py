"""
Centralized logging configuration for MusicTruth.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler

def setup_logger(name: str = "MusicTruth", log_file: str = "musictruth.log", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file
        level: Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # prevent adding handlers multiple times if function is called repeatedly
    if logger.hasHandlers():
        return logger

    # Formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 1. File Handler (Rotating)
    # 5 MB per file, max 3 backups
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # 2. Console Handler (Rich)
    console_handler = RichHandler(rich_tracebacks=True, markup=True)
    console_handler.setLevel(level)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create default logger instance
logger = setup_logger()
