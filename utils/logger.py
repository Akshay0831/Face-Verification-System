"""Universal logger for all system components."""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


class FaceVerificationLogger:
    """Universal logger for system components"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(FaceVerificationLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logger if not done"""
        if not FaceVerificationLogger._initialized:
            self._setup_logger()
            FaceVerificationLogger._initialized = True
    
    def _setup_logger(self):
        """Set up universal logger"""
        self.logger = logging.getLogger('face_verification_system')
        self.logger.setLevel(logging.DEBUG)
        
        # Avoid adding multiple handlers
        if self.logger.handlers:
            return
        
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler with rotating logs
        log_file = os.path.join(log_dir, f'face_verification_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance for a specific module"""
        if name:
            return logging.getLogger(f'face_verification_system.{name}')
        return self.logger
    
    # Convenience methods
    def debug(self, message: str, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get the universal logger instance"""
    return FaceVerificationLogger().get_logger(name)