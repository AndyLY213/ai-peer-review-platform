"""
Centralized Logging Configuration System.

This module provides a unified logging interface for the entire application.
Replaces print statements with proper logging levels and file output.
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config.env")

class LoggerSetup:
    """
    Centralized logger setup and configuration.
    """
    
    def __init__(self):
        """Initialize logging configuration."""
        self.log_level = self._get_log_level()
        self.log_dir = self._get_log_directory()
        self.setup_logging()
    
    def _get_log_level(self) -> int:
        """
        Get log level from environment variable.
        
        Returns:
            Logging level constant
        """
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return level_map.get(level_str, logging.INFO)
    
    def _get_log_directory(self) -> str:
        """
        Get log directory from environment variable.
        
        Returns:
            Path to log directory
        """
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def setup_logging(self):
        """Set up logging configuration."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for all logs
        log_file = os.path.join(self.log_dir, f"peer_review_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB max, 5 backups
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler
        error_file = os.path.join(self.log_dir, f"errors_{datetime.now().strftime('%Y%m%d')}.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_file, maxBytes=5*1024*1024, backupCount=3  # 5MB max, 3 backups
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Simulation-specific file handler
        sim_file = os.path.join(self.log_dir, f"simulation_{datetime.now().strftime('%Y%m%d')}.log")
        sim_handler = logging.handlers.RotatingFileHandler(
            sim_file, maxBytes=10*1024*1024, backupCount=5
        )
        sim_handler.setLevel(logging.INFO)
        sim_handler.setFormatter(detailed_formatter)
        
        # Add simulation handler to simulation logger
        sim_logger = logging.getLogger('simulation')
        sim_logger.addHandler(sim_handler)
        sim_logger.setLevel(logging.INFO)
        sim_logger.propagate = True  # Also send to root logger

# Global logger setup instance
_logger_setup = None

def setup_logging():
    """Initialize the logging system."""
    global _logger_setup
    if _logger_setup is None:
        _logger_setup = LoggerSetup()

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (usually __name__)
        
    Returns:
        Logger instance
    """
    # Ensure logging is set up
    setup_logging()
    return logging.getLogger(name)

def get_simulation_logger() -> logging.Logger:
    """
    Get the simulation-specific logger.
    
    Returns:
        Simulation logger instance
    """
    setup_logging()
    return logging.getLogger('simulation')

def log_llm_config(provider_info: dict, logger: Optional[logging.Logger] = None):
    """
    Log LLM configuration information.
    
    Args:
        provider_info: Dictionary with provider information
        logger: Logger instance (if None, uses root logger)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"LLM Provider: {provider_info['provider']}")
    logger.info(f"Model: {provider_info['model']}")
    logger.info(f"Temperature: {provider_info['temperature']}")
    logger.info(f"Timeout: {provider_info['timeout']}")

def log_simulation_start(num_rounds: int, interactions_per_round: int, logger: Optional[logging.Logger] = None):
    """
    Log simulation start information.
    
    Args:
        num_rounds: Number of simulation rounds
        interactions_per_round: Number of interactions per round
        logger: Logger instance (if None, uses simulation logger)
    """
    if logger is None:
        logger = get_simulation_logger()
    
    logger.info("="*50)
    logger.info("SIMULATION STARTED")
    logger.info("="*50)
    logger.info(f"Rounds: {num_rounds}")
    logger.info(f"Interactions per round: {interactions_per_round}")

def log_simulation_end(results: dict, logger: Optional[logging.Logger] = None):
    """
    Log simulation end information.
    
    Args:
        results: Simulation results dictionary
        logger: Logger instance (if None, uses simulation logger)
    """
    if logger is None:
        logger = get_simulation_logger()
    
    logger.info("="*50)
    logger.info("SIMULATION COMPLETED")
    logger.info("="*50)
    logger.info(f"Total papers: {results.get('total_papers', 0)}")
    logger.info(f"Total reviews requested: {results.get('total_reviews_requested', 0)}")
    logger.info(f"Total reviews completed: {results.get('total_reviews_completed', 0)}")
    logger.info(f"Total tokens spent: {results.get('total_tokens_spent', 0)}")

def log_researcher_action(researcher_name: str, action: str, details: str, logger: Optional[logging.Logger] = None):
    """
    Log researcher actions during simulation.
    
    Args:
        researcher_name: Name of the researcher
        action: Type of action (e.g., "review_request", "review_completion")
        details: Additional details about the action
        logger: Logger instance (if None, uses simulation logger)
    """
    if logger is None:
        logger = get_simulation_logger()
    
    logger.info(f"[{researcher_name}] {action}: {details}")

def log_error_with_context(error: Exception, context: str, logger: Optional[logging.Logger] = None):
    """
    Log errors with additional context information.
    
    Args:
        error: Exception that occurred
        context: Context where the error occurred
        logger: Logger instance (if None, uses root logger)
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.error(f"Error in {context}: {type(error).__name__}: {str(error)}", exc_info=True)

# Convenience functions for common logging patterns
def log_info(message: str, logger_name: str = None):
    """Log info message."""
    logger = get_logger(logger_name or __name__)
    logger.info(message)

def log_warning(message: str, logger_name: str = None):
    """Log warning message."""
    logger = get_logger(logger_name or __name__)
    logger.warning(message)

def log_error(message: str, logger_name: str = None):
    """Log error message."""
    logger = get_logger(logger_name or __name__)
    logger.error(message)

def log_debug(message: str, logger_name: str = None):
    """Log debug message."""
    logger = get_logger(logger_name or __name__)
    logger.debug(message)