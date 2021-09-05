"""Enhanced notification plugin implementation"""

from core.base import INotifier, NotificationResult, PluginMetadata
from .email_notifier import EmailNotifier
from utils import get_logger

logger = get_logger('enhanced_notifications')

__all__ = ['EmailNotifier']

class EnhancedNotificationPlugin(INotifier):
    """Enhanced notification plugin wrapper"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.email_notifier = EmailNotifier()
        self.email_notifier.initialize(self.config)
    
    def get_metadata(self):
        """Return plugin metadata"""
        return PluginMetadata(
            name='enhanced_notifications',
            version='1.0.0',
            description='Enhanced notification system with email support',
            author='Face Verification System Team',
            dependencies=['smtplib'],
            device_compatibility=['windows', 'linux', 'raspberry_pi']
        )
    
    def send_notification(self, user_id=None, message="", **kwargs):
        """Send notification using enhanced methods"""
        return self.email_notifier.send_notification(user_id=user_id, message=message, **kwargs)

__all__.append('EnhancedNotificationPlugin')