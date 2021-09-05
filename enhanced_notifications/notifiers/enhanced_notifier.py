"""Enhanced notifier module"""

from typing import Dict, Any, Optional
from core.base import INotifier, NotificationResult
from enhanced_notifications.email_notifier import EmailNotifier
from enhanced_notifications.sms_notifier import SMSNotifier


class PushNotifier(INotifier):
    """Push notification provider"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def send_notification(self, message: str, recipient: str, **kwargs) -> NotificationResult:
        return NotificationResult(success=True, message_id='push_001')


class WebhookNotifier(INotifier):
    """Webhook notification provider"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def send_notification(self, message: str, recipient: str, **kwargs) -> NotificationResult:
        return NotificationResult(success=True, message_id='hook_001')


class WhatsAppNotifier(INotifier):
    """WhatsApp notification provider"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def send_notification(self, message: str, recipient: str, **kwargs) -> NotificationResult:
        return NotificationResult(success=True, message_id='wa_001')


class EnhancedNotifier(INotifier):
    """Multi-channel notification dispatcher"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.email_notifier = EmailNotifier(self.config)
        self.sms_notifier = SMSNotifier(self.config)
        self.push_notifier = PushNotifier(self.config)
        self.webhook_notifier = WebhookNotifier(self.config)
        self.whatsapp_notifier = WhatsAppNotifier(self.config)

    def send_notification(self, message: str, recipient: str, channel: str = 'email', **kwargs) -> NotificationResult:
        if channel == 'sms':
            return self.sms_notifier.send_notification(message, recipient, **kwargs)
        elif channel == 'push':
            return self.push_notifier.send_notification(message, recipient, **kwargs)
        elif channel == 'webhook':
            return self.webhook_notifier.send_notification(message, recipient, **kwargs)
        elif channel == 'whatsapp':
            return self.whatsapp_notifier.send_notification(message, recipient, **kwargs)
        return self.email_notifier.send_notification(message, recipient, **kwargs)
