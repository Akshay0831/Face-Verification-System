"""Email notification plugin"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional
import os

from core.base import INotifier, NotificationResult, PluginMetadata, DeviceType
from utils import get_logger

logger = get_logger('email_notifier')


class EmailNotifier(INotifier):
    """Email notification implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.smtp_server = self.config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = self.config.get('smtp_port', 587)
        self.use_tls = self.config.get('use_tls', True)
        self.username = self.config.get('username', '')
        self.password = self.config.get('password', '')
        self.from_email = self.config.get('from_email', '')
        self.from_name = self.config.get('from_name', 'Face Verification System')
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.enabled = True  # Initialize as enabled, will be disabled if config missing
        
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='email_notifier',
            version='1.0.0',
            description='Email notification system',
            author='Face Verification System Team',
            dependencies=[],  # Standard library
            device_compatibility=[DeviceType.WINDOWS, DeviceType.LINUX, DeviceType.RASPBERRY_PI]
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the email notifier"""
        try:
            self.config.update(config)
            
            # Load credentials from environment if not provided
            if not self.username:
                self.username = os.getenv('EMAIL_USERNAME', '')
            if not self.password:
                self.password = os.getenv('EMAIL_PASSWORD', '')
            if not self.from_email:
                self.from_email = os.getenv('EMAIL_FROM', self.username)
            
            # Validate required configuration
            if not self.smtp_server:
                logger.warning("SMTP server not configured - email notifications disabled")
                self.enabled = False
                return True  # Return True to allow initialization to continue
            
            if not self.username or not self.password:
                logger.debug("Email credentials not provided - email notifications disabled")
                self.enabled = False
                return True  # Return True to allow initialization to continue
            
            logger.info(f"Email notifier initialized for {self.smtp_server}:{self.smtp_port}")
            self.enabled = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing email notifier: {e}")
            return False
    
    def send_notification(self, 
                         user_id: Optional[str] = None,
                         message: str = "",
                         **kwargs) -> NotificationResult:
        """Send an email notification"""
        # Check if email is enabled
        if not getattr(self, 'enabled', False):
            return NotificationResult(
                success=False, 
                message="Email notifications disabled - missing configuration"
            )
        
        try:
            to_email = kwargs.get('to_email', kwargs.get('recipient', ''))
            subject = kwargs.get('subject', 'Face Verification Notification')
            if not to_email:
                logger.error("No recipient email provided")
                return NotificationResult(success=False, message="No recipient email provided")
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add user ID to message if provided
            if user_id:
                message = f"User: {user_id}\n{message}"
            
            # Add body to email
            msg.attach(MIMEText(message, 'plain'))
            
            # Send email
            for attempt in range(self.retry_attempts):
                try:
                    with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                        if self.use_tls:
                            server.starttls()
                        
                        server.login(self.username, self.password)
                        server.send_message(msg)
                        break
                
                except Exception as e:
                    logger.error(f"Email send attempt {attempt + 1} failed: {e}")
                    if attempt == self.retry_attempts - 1:
                        raise
            
            logger.info(f"Email sent successfully to {to_email}")
            return NotificationResult(success=True, message="Email sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return NotificationResult(success=False, message=f"Email send failed: {e}")
    
    def send_verification_email(self, 
                              user_id: str,
                              email: str,
                              verification_code: str) -> NotificationResult:
        """Send a verification email"""
        message = f"""
Thank you for using the Face Verification System!

Your verification code is: {verification_code}

Please enter this code in the application to complete your verification.

If you did not request this code, please ignore this email.

Best regards,
The Face Verification System Team
        """
        
        return self.send_notification(
            user_id=user_id,
            to_email=email,
            subject=f"Face Verification - Code: {verification_code}",
            message=message
        )
    
    def send_alert_email(self,
                        user_id: str,
                        email: str,
                        alert_type: str = "Security Alert",
                        details: Optional[Dict[str, Any]] = None) -> NotificationResult:
        """Send an alert email"""
        details_str = ""
        if details:
            details_str = "\nDetails:\n"
            for key, value in details.items():
                details_str += f"  {key}: {value}\n"
        
        message = f"""
Security Alert Detected!

User ID: {user_id}
Alert Type: {alert_type}
{details_str}

Please take appropriate action.

Best regards,
The Face Verification System Team
        """
        
        return self.send_notification(
            user_id=user_id,
            to_email=email,
            subject=f"Security Alert: {alert_type}",
            message=message
        )
    
    def get_supported_modes(self) -> List[str]:
        """Return supported notification modes"""
        return ['email', 'verification', 'alert', 'bulk_email']
    
    def get_required_config(self) -> List[str]:
        """Return required configuration parameters"""
        return ['smtp_server', 'smtp_port', 'username', 'password', 'from_email']