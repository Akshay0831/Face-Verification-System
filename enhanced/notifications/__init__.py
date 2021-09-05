"""Enhanced notifications alias package"""
from enhanced_notifications.email_notifier import EmailNotifier
from enhanced_notifications.sms_notifier import SMSNotifier
from enhanced_notifications.notifiers.enhanced_notifier import (
    PushNotifier, WebhookNotifier, WhatsAppNotifier, EnhancedNotifier
)
class WebhookNotificationChannel(WebhookNotifier):
    """WebhookNotificationChannel alias for test suite compatibility"""
    pass
