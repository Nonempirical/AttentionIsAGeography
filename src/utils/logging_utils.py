"""Logging utilities."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name, configured with basic settings."""
    logger = logging.getLogger(name)
    
    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler if root logger doesn't have one
        if not logging.root.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            
            # Set format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
            # Configure root logger
            logging.root.addHandler(handler)
            logging.root.setLevel(logging.INFO)
    
    return logger
