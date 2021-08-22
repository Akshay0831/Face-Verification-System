"""Test enhanced notification system functionality"""

import unittest
import os
import sys
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced_notifications.notifiers.enhanced_notifier import (
    EnhancedNotifier, 
    EmailNotifier, 
    SMSNotifier,
    PushNotifier,
    WebhookNotifier,
    WhatsAppNotifier
)

class TestEnhancedNotifier(unittest.TestCase):
    """Test cases for Enhanced Notification System"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Test notification data
        self.test_notification = {
            'notification_id': 'notif_001',
            'type': 'security_alert',
            'priority': 'high',
            'title': 'Unauthorized Access Attempt',
            'message': 'Person attempted access without authorization',
            'timestamp': datetime.now(),
            'recipient': {
                'name': 'Security Team',
                'contact': ['security@company.com', '+1234567890', 'push_token_123'],
                'preferences': ['email', 'sms', 'push']
            },
            'source': {
                'device': 'main_entrance',
                'location': 'Building A',
                'detection_method': 'face_verification'
            },
            'details': {
                'person_id': 'unknown_person',
                'confidence': 0.75,
                'image_path': 'captures/attempt_001.jpg',
                'timestamp': datetime.now(),
                'attempts': 1,
                'timeframe': '2024-01-01 10:00:00'
            },
            'metadata': {
                'department': 'Security',
                'access_level': 'restricted',
                'alert_category': 'security'
            }
        }
        
        # Test configuration
        self.test_config = {
            'enabled_channels': [
                'email',
                'sms', 
                'push',
                'webhook',
                'whatsapp'
            ],
            'templates': {
                'security_alert': {
                    'email': {
                        'subject': '🚨 Security Alert: {title}',
                        'body': """
                        <h2>Security Alert Notification</h2>
                        <p><strong>Title:</strong> {title}</p>
                        <p><strong>Message:</strong> {message}</p>
                        <p><strong>Location:</strong> {location}</p>
                        <p><strong>Device:</strong> {device}</p>
                        <p><strong>Time:</strong> {timestamp}</p>
                        <p><strong>Details:</strong></p>
                        <ul>
                            <li>Person ID: {person_id}</li>
                            <li>Confidence: {confidence}</li>
                            <li>Attempts: {attempts}</li>
                        </ul>
                        <p>Please investigate immediately.</p>
                        """,
                        'priority': 'high'
                    },
                    'sms': {
                        'message': '🚨 Security Alert: {title}. Location: {location}. Device: {device}. Time: {timestamp}',
                        'max_length': 160,
                        'priority': 'high'
                    },
                    'push': {
                        'title': '🚨 {title}',
                        'body': '{message}\nLocation: {location}\nDevice: {device}',
                        'sound': 'default',
                        'priority': 'high'
                    },
                    'webhook': {
                        'payload': {
                            'event': 'security_alert',
                            'timestamp': '{timestamp}',
                            'location': '{location}',
                            'device': '{device}',
                            'title': '{title}',
                            'message': '{message}',
                            'details': {
                                'person_id': '{person_id}',
                                'confidence': '{confidence}',
                                'attempts': '{attempts}'
                            }
                        },
                        'headers': {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer {webhook_token}'
                        }
                    },
                    'whatsapp': {
                        'message': '🚨 Security Alert: {title}\n\n{message}\n\nLocation: {location}\nDevice: {device}\nTime: {timestamp}\n\nPerson ID: {person_id}\nConfidence: {confidence}',
                        'template_id': 'security_alert_template',
                        'priority': 'high'
                    }
                }
            },
            'channels': {
                'email': {
                    'smtp_server': 'smtp.company.com',
                    'smtp_port': 587,
                    'smtp_username': 'alerts@company.com',
                    'smtp_password': os.environ.get('SMTP_PASSWORD', 'password'),
                    'from_email': 'alerts@company.com',
                    'from_name': 'Face Verification System',
                    'timeout': 30,
                    'retry_attempts': 3,
                    'use_tls': True
                },
                'sms': {
                    'provider': 'twilio',
                    'account_sid': os.environ.get('TWILIO_ACCOUNT_SID', 'test_sid'),
                    'auth_token': os.environ.get('TWILIO_AUTH_TOKEN', 'test_token'),
                    'from_number': '+1234567890',
                    'max_length': 160,
                    'timeout': 15,
                    'retry_attempts': 3
                },
                'push': {
                    'service': 'fcm',
                    'api_key': os.environ.get('FCM_API_KEY', 'test_api_key'),
                    'project_id': 'face-verification-system',
                    'timeout': 15,
                    'retry_attempts': 3
                },
                'webhook': {
                    'url': 'https://webhook.company.com/security-alerts',
                    'webhook_token': os.environ.get('WEBHOOK_TOKEN', 'test_token'),
                    'timeout': 30,
                    'retry_attempts': 3,
                    'headers': {
                        'Content-Type': 'application/json'
                    }
                },
                'whatsapp': {
                    'business_phone_id': os.environ.get('WHATSAPP_BUSINESS_ID', 'test_phone_id'),
                    'access_token': os.environ.get('WHATSAPP_TOKEN', 'test_token'),
                    'template_id': 'security_alert_template',
                    'timeout': 30,
                    'retry_attempts': 3
                }
            },
            'routing': {
                'security_alert': {
                    'recipients': ['security@company.com'],
                    'channels': ['email', 'sms', 'push'],
                    'priority': 'high',
                    'schedule': {
                        'immediate': True,
                        'followup_minutes': 5,
                        'max_followups': 3
                    }
                },
                'system_status': {
                    'recipients': ['admin@company.com'],
                    'channels': ['email'],
                    'priority': 'medium'
                },
                'user_notification': {
                    'recipients': ['user@company.com'],
                    'channels': ['push', 'whatsapp'],
                    'priority': 'low'
                }
            },
            'performance': {
                'batch_size': 10,
                'max_workers': 4,
                'timeout': 60,
                'cache_ttl': 300,
                'enable_metrics': True,
                'enable_logging': True
            }
        }
        
        # Test template data
        self.test_template_data = {
            'title': 'Unauthorized Access Attempt',
            'message': 'Person attempted access without authorization',
            'location': 'Building A - Main Entrance',
            'device': 'main_entrance_cam',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'person_id': 'unknown_person',
            'confidence': 0.75,
            'attempts': 1,
            'webhook_token': os.environ.get('WEBHOOK_TOKEN', 'test_token')
        }
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_enhanced_notifier_initialization(self):
        """Test EnhancedNotifier initialization"""
        notifier = EnhancedNotifier(self.test_config)
        
        self.assertIsNotNone(notifier)
        self.assertIsNotNone(notifier.config)
        self.assertIsNotNone(notifier.notifiers)
        self.assertIsNotNone(notifier.template_engine)
        self.assertIsNotNone(notifier.routing_engine)
        self.assertIsNotNone(notifier.metrics)
        self.assertEqual(len(notifier.notifiers), 5)  # All 5 channels
        
        # Check configurations
        self.assertEqual(notifier.config['performance']['batch_size'], 10)
        self.assertTrue(notifier.config['performance']['enable_metrics'])
        self.assertEqual(notifier.config['performance']['max_workers'], 4)
    
    def test_enhanced_notifier_initialization_default_config(self):
        """Test EnhancedNotifier initialization with default config"""
        notifier = EnhancedNotifier()
        
        self.assertIsNotNone(notifier)
        self.assertIsNotNone(notifier.config)
        self.assertEqual(len(notifier.notifiers), 3)  # Default channels
        
        # Check default values
        self.assertEqual(notifier.config['performance']['batch_size'], 5)
        self.assertFalse(notifier.config['performance']['enable_metrics'])  # Disabled by default
        self.assertEqual(notifier.config['performance']['max_workers'], 2)
    
    def test_enhanced_notifier_load_notifier(self):
        """Test notifier loading"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Test loading individual notifiers
        email_notifier = notifier._load_notifier('email')
        self.assertIsNotNone(email_notifier)
        self.assertIsInstance(email_notifier, EmailNotifier)
        
        sms_notifier = notifier._load_notifier('sms')
        self.assertIsNotNone(sms_notifier)
        self.assertIsInstance(sms_notifier, SMSNotifier)
        
        push_notifier = notifier._load_notifier('push')
        self.assertIsNotNone(push_notifier)
        self.assertIsInstance(push_notifier, PushNotifier)
        
        webhook_notifier = notifier._load_notifier('webhook')
        self.assertIsNotNone(webhook_notifier)
        self.assertIsInstance(webhook_notifier, WebhookNotifier)
        
        whatsapp_notifier = notifier._load_notifier('whatsapp')
        self.assertIsNotNone(whatsapp_notifier)
        self.assertIsInstance(whatsapp_notifier, WhatsAppNotifier)
    
    def test_enhanced_notifier_send_notification(self):
        """Test notification sending"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Mock notifier responses
        mock_email_response = {
            'success': True,
            'channel': 'email',
            'message_id': 'email_001',
            'recipients': ['security@company.com'],
            'timestamp': datetime.now(),
            'processing_time': 0.1
        }
        
        mock_sms_response = {
            'success': True,
            'channel': 'sms',
            'message_id': 'sms_001',
            'recipients': ['+1234567890'],
            'timestamp': datetime.now(),
            'processing_time': 0.05
        }
        
        mock_push_response = {
            'success': True,
            'channel': 'push',
            'message_id': 'push_001',
            'recipients': ['push_token_123'],
            'timestamp': datetime.now(),
            'processing_time': 0.02
        }
        
        with patch.object(notifier.notifiers['email'], 'send') as mock_email:
            with patch.object(notifier.notifiers['sms'], 'send') as mock_sms:
                with patch.object(notifier.notifiers['push'], 'send') as mock_push:
                    mock_email.return_value = mock_email_response
                    mock_sms.return_value = mock_sms_response
                    mock_push.return_value = mock_push_response
                    
                    # Send notification
                    result = notifier.send_notification(self.test_notification)
                    
                    self.assertIsNotNone(result)
                    self.assertIn('success', result)
                    self.assertIn('notification_id', result)
                    self.assertIn('channels_attempted', result)
                    self.assertIn('channels_success', result)
                    self.assertIn('channel_responses', result)
                    self.assertIn('total_processing_time', result)
                    self.assertIn('priority_handled', result)
                    
                    # Check result values
                    self.assertTrue(result['success'])
                    self.assertEqual(result['notification_id'], 'notif_001')
                    self.assertEqual(result['channels_attempted'], 3)  # email, sms, push
                    self.assertEqual(result['channels_success'], 3)  # all successful
                    self.assertEqual(result['priority_handled'], True)
                    
                    # Check channel responses
                    self.assertIsInstance(result['channel_responses'], dict)
                    self.assertIn('email', result['channel_responses'])
                    self.assertIn('sms', result['channel_responses'])
                    self.assertIn('push', result['channel_responses'])
                    
                    # Check processing time
                    self.assertIsInstance(result['total_processing_time'], float)
                    self.assertGreater(result['total_processing_time'], 0)
                    
                    # Verify notifiers were called
                    mock_email.assert_called_once()
                    mock_sms.assert_called_once()
                    mock_push.assert_called_once()
    
    def test_enhanced_notifier_send_notification_partial_failure(self):
        """Test notification sending with partial failures"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Mock mixed responses
        mock_email_response = {
            'success': True,
            'channel': 'email',
            'message_id': 'email_001',
            'recipients': ['security@company.com'],
            'timestamp': datetime.now(),
            'processing_time': 0.1
        }
        
        mock_sms_response = {
            'success': False,
            'channel': 'sms',
            'error': 'Network error',
            'recipients': [],
            'timestamp': datetime.now(),
            'processing_time': 0.05
        }
        
        mock_push_response = {
            'success': True,
            'channel': 'push',
            'message_id': 'push_001',
            'recipients': ['push_token_123'],
            'timestamp': datetime.now(),
            'processing_time': 0.02
        }
        
        with patch.object(notifier.notifiers['email'], 'send') as mock_email:
            with patch.object(notifier.notifiers['sms'], 'send') as mock_sms:
                with patch.object(notifier.notifiers['push'], 'send') as mock_push:
                    mock_email.return_value = mock_email_response
                    mock_sms.return_value = mock_sms_response
                    mock_push.return_value = mock_push_response
                    
                    # Send notification
                    result = notifier.send_notification(self.test_notification)
                    
                    self.assertIsNotNone(result)
                    self.assertTrue(result['success'])
                    self.assertEqual(result['channels_attempted'], 3)
                    self.assertEqual(result['channels_success'], 2)
                    self.assertIn('channel_responses', result)
                    
                    # Check failed channel
                    self.assertFalse(result['channel_responses']['sms']['success'])
                    self.assertIn('error', result['channel_responses']['sms'])
    
    def test_enhanced_notifier_send_notification_all_failed(self):
        """Test notification sending when all channels fail"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Mock all failed responses
        mock_email_response = {
            'success': False,
            'channel': 'email',
            'error': 'SMTP error',
            'recipients': []
        }
        
        mock_sms_response = {
            'success': False,
            'channel': 'sms',
            'error': 'SMS provider error',
            'recipients': []
        }
        
        mock_push_response = {
            'success': False,
            'channel': 'push',
            'error': 'Push service error',
            'recipients': []
        }
        
        with patch.object(notifier.notifiers['email'], 'send') as mock_email:
            with patch.object(notifier.notifiers['sms'], 'send') as mock_sms:
                with patch.object(notifier.notifiers['push'], 'send') as mock_push:
                    mock_email.return_value = mock_email_response
                    mock_sms.return_value = mock_sms_response
                    mock_push.return_value = mock_push_response
                    
                    # Send notification
                    result = notifier.send_notification(self.test_notification)
                    
                    self.assertIsNotNone(result)
                    self.assertFalse(result['success'])
                    self.assertEqual(result['channels_attempted'], 3)
                    self.assertEqual(result['channels_success'], 0)
                    self.assertIn('channel_responses', result)
                    
                    # Check all failed channels
                    for channel in ['email', 'sms', 'push']:
                        self.assertFalse(result['channel_responses'][channel]['success'])
                        self.assertIn('error', result['channel_responses'][channel])
    
    def test_enhanced_notifier_send_notification_escalation(self):
        """Test notification sending with escalation"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Mock initial failure and escalation success
        mock_email_response = {
            'success': False,
            'channel': 'email',
            'error': 'SMTP error',
            'recipients': []
        }
        
        mock_sms_response = {
            'success': True,
            'channel': 'sms',
            'message_id': 'sms_001',
            'recipients': ['+1234567890'],
            'timestamp': datetime.now(),
            'processing_time': 0.05
        }
        
        mock_push_response = {
            'success': True,
            'channel': 'push',
            'message_id': 'push_001',
            'recipients': ['push_token_123'],
            'timestamp': datetime.now(),
            'processing_time': 0.02
        }
        
        with patch.object(notifier.notifiers['email'], 'send') as mock_email:
            with patch.object(notifier.notifiers['sms'], 'send') as mock_sms:
                with patch.object(notifier.notifiers['push'], 'send') as mock_push:
                    mock_email.return_value = mock_email_response
                    mock_sms.return_value = mock_sms_response
                    mock_push.return_value = mock_push_response
                    
                    # Send notification
                    result = notifier.send_notification(self.test_notification)
                    
                    self.assertIsNotNone(result)
                    self.assertTrue(result['success'])
                    self.assertEqual(result['channels_attempted'], 3)
                    self.assertEqual(result['channels_success'], 2)
                    self.assertIn('escalation_triggered', result)
                    self.assertTrue(result['escalation_triggered'])
    
    def test_enhanced_notifier_batch_notification(self):
        """Test batch notification sending"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Mock notification responses
        mock_response = {
            'success': True,
            'channel': 'email',
            'message_id': 'email_001',
            'recipients': ['security@company.com'],
            'timestamp': datetime.now(),
            'processing_time': 0.1
        }
        
        with patch.object(notifier.notifiers['email'], 'send') as mock_email:
            mock_email.return_value = mock_response
            
            # Batch send notifications
            notifications = [self.test_notification] * 5
            results = notifier.send_notifications_batch(notifications)
            
            self.assertIsNotNone(results)
            self.assertEqual(len(results), 5)
            
            # Check each result
            for i, result in enumerate(results):
                self.assertIn('success', result)
                self.assertIn('notification_id', result)
                self.assertIn('processing_time', result)
                
                # Verify notifiers were called for each notification
                mock_email.assert_any_call(notifications[i])
    
    def test_enhanced_notifier_template_rendering(self):
        """Test template rendering functionality"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Test email template rendering
        template = notifier.template_engine.render(
            'security_alert',
            'email',
            self.test_template_data
        )
        
        self.assertIsNotNone(template)
        self.assertIn('Subject', template)
        self.assertIn('Security Alert Notification', template)
        self.assertIn('Unauthorized Access Attempt', template)
        self.assertIn('Building A - Main Entrance', template)
        self.assertIn('Person ID: unknown_person', template)
        
        # Test SMS template rendering
        template = notifier.template_engine.render(
            'security_alert',
            'sms',
            self.test_template_data
        )
        
        self.assertIsNotNone(template)
        self.assertIn('Security Alert', template)
        self.assertIn('Building A', template)
        self.assertIn('main_entrance_cam', template)
        
        # Test push template rendering
        template = notifier.template_engine.render(
            'security_alert',
            'push',
            self.test_template_data
        )
        
        self.assertIsNotNone(template)
        self.assertIn('Security Alert', template)
        self.assertIn('Person attempted access without authorization', template)
    
    def test_enhanced_notifier_routing_engine(self):
        """Test notification routing functionality"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Test routing rules
        routing = notifier.routing_engine.get_routing_rules(
            'security_alert',
            self.test_notification['recipient']
        )
        
        self.assertIsNotNone(routing)
        self.assertIn('recipients', routing)
        self.assertIn('channels', routing)
        self.assertIn('priority', routing)
        self.assertIn('schedule', routing)
        
        # Check routing values
        self.assertEqual(len(routing['recipients']), 1)
        self.assertIn('security@company.com', routing['recipients'])
        self.assertIn('email', routing['channels'])
        self.assertIn('sms', routing['channels'])
        self.assertEqual(routing['priority'], 'high')
        self.assertTrue(routing['schedule']['immediate'])
    
    def test_enhanced_notifier_metrics_collection(self):
        """Test metrics collection functionality"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Mock notification responses
        mock_response = {
            'success': True,
            'channel': 'email',
            'message_id': 'email_001',
            'recipients': ['security@company.com'],
            'timestamp': datetime.now(),
            'processing_time': 0.1
        }
        
        with patch.object(notifier.notifiers['email'], 'send') as mock_email:
            mock_email.return_value = mock_response
            
            # Send notification and collect metrics
            result = notifier.send_notification(self.test_notification)
            
            # Check metrics
            metrics = notifier.metrics.get_metrics()
            self.assertIsNotNone(metrics)
            self.assertIn('total_sent', metrics)
            self.assertIn('total_delivered', metrics)
            self.assertIn('total_failed', metrics)
            self.assertIn('channel_breakdown', metrics)
            self.assertIn('average_processing_time', metrics)
            self.assertIn('success_rate', metrics)
            
            # Verify metrics values
            self.assertEqual(metrics['total_sent'], 1)
            self.assertEqual(metrics['total_delivered'], 1)
            self.assertEqual(metrics['total_failed'], 0)
            self.assertEqual(metrics['success_rate'], 100.0)
            self.assertIn('email', metrics['channel_breakdown'])
            self.assertEqual(metrics['channel_breakdown']['email']['sent'], 1)
            self.assertEqual(metrics['channel_breakdown']['email']['delivered'], 1)
    
    def test_email_notifier_initialization(self):
        """Test EmailNotifier initialization"""
        config = {
            'smtp_server': 'smtp.company.com',
            'smtp_port': 587,
            'smtp_username': 'alerts@company.com',
            'smtp_password': 'password',
            'from_email': 'alerts@company.com',
            'from_name': 'Face Verification System',
            'timeout': 30,
            'retry_attempts': 3,
            'use_tls': True
        }
        
        notifier = EmailNotifier(config)
        
        self.assertIsNotNone(notifier)
        self.assertIsNotNone(notifier.config)
        self.assertIsNotNone(notifier.smtp_server)
        self.assertIsNotNone(notifier.template_processor)
        
        # Check configuration
        self.assertEqual(notifier.config['smtp_server'], 'smtp.company.com')
        self.assertEqual(notifier.config['smtp_port'], 587)
        self.assertEqual(notifier.config['smtp_username'], 'alerts@company.com')
        self.assertEqual(notifier.config['smtp_password'], 'password')
        self.assertEqual(notifier.config['from_email'], 'alerts@company.com')
    
    def test_email_notifier_send(self):
        """Test EmailNotifier send functionality"""
        config = {
            'smtp_server': 'smtp.company.com',
            'smtp_port': 587,
            'smtp_username': 'alerts@company.com',
            'smtp_password': 'password',
            'from_email': 'alerts@company.com',
            'from_name': 'Face Verification System'
        }
        
        notifier = EmailNotifier(config)
        
        # Mock email data
        email_data = {
            'to': 'security@company.com',
            'subject': 'Security Alert',
            'body': 'This is a test email',
            'html_body': '<h2>Security Alert</h2><p>This is a test email</p>',
            'priority': 'high'
        }
        
        # Mock SMTP connection
        mock_smtp = MagicMock()
        mock_message = MagicMock()
        
        with patch.object(smtplib, 'SMTP') as mock_smtp_class:
            with patch('email.mime.text.MIMEText') as mock_mime:
                mock_smtp_class.return_value = mock_smtp
                mock_mime.return_value = mock_message
                
                # Send email
                result = notifier.send(email_data)
                
                self.assertIsNotNone(result)
                self.assertTrue(result['success'])
                self.assertEqual(result['channel'], 'email')
                self.assertEqual(result['message_id'], 'email_001')
                self.assertEqual(result['recipients'], ['security@company.com'])
                self.assertIsInstance(result['timestamp'], datetime)
                self.assertIsInstance(result['processing_time'], float)
                
                # Verify SMTP calls
                mock_smtp_class.assert_called_once()
                mock_smtp.send_message.assert_called_once()
                mock_smtp.quit.assert_called_once()
    
    def test_sms_notifier_initialization(self):
        """Test SMSNotifier initialization"""
        config = {
            'provider': 'twilio',
            'account_sid': 'test_sid',
            'auth_token': 'test_token',
            'from_number': '+1234567890',
            'max_length': 160,
            'timeout': 15,
            'retry_attempts': 3
        }
        
        notifier = SMSNotifier(config)
        
        self.assertIsNotNone(notifier)
        self.assertIsNotNone(notifier.config)
        self.assertIsNotNone(notifier.client)
        
        # Check configuration
        self.assertEqual(notifier.config['provider'], 'twilio')
        self.assertEqual(notifier.config['account_sid'], 'test_sid')
        self.assertEqual(notifier.config['auth_token'], 'test_token')
        self.assertEqual(notifier.config['from_number'], '+1234567890')
    
    def test_sms_notifier_send(self):
        """Test SMSNotifier send functionality"""
        config = {
            'provider': 'twilio',
            'account_sid': 'test_sid',
            'auth_token': 'test_token',
            'from_number': '+1234567890'
        }
        
        notifier = SMSNotifier(config)
        
        # Mock SMS data
        sms_data = {
            'to': '+1234567890',
            'message': 'Security Alert: Unauthorized access attempt',
            'priority': 'high'
        }
        
        # Mock Twilio client
        mock_message = MagicMock()
        mock_message.sid = 'sms_001'
        
        with patch('twilio.rest.Client') as mock_client:
            mock_client.return_value.messages.create.return_value = mock_message
            
            # Send SMS
            result = notifier.send(sms_data)
            
            self.assertIsNotNone(result)
            self.assertTrue(result['success'])
            self.assertEqual(result['channel'], 'sms')
            self.assertEqual(result['message_id'], 'sms_001')
            self.assertEqual(result['recipients'], ['+1234567890'])
            
            # Verify Twilio call
            mock_client.return_value.messages.create.assert_called_once()
    
    def test_push_notifier_initialization(self):
        """Test PushNotifier initialization"""
        config = {
            'service': 'fcm',
            'api_key': 'test_api_key',
            'project_id': 'face-verification-system',
            'timeout': 15,
            'retry_attempts': 3
        }
        
        notifier = PushNotifier(config)
        
        self.assertIsNotNone(notifier)
        self.assertIsNotNone(notifier.config)
        self.assertIsNotNone(notifier.client)
        
        # Check configuration
        self.assertEqual(notifier.config['service'], 'fcm')
        self.assertEqual(notifier.config['api_key'], 'test_api_key')
        self.assertEqual(notifier.config['project_id'], 'face-verification-system')
    
    def test_push_notifier_send(self):
        """Test PushNotifier send functionality"""
        config = {
            'service': 'fcm',
            'api_key': 'test_api_key'
        }
        
        notifier = PushNotifier(config)
        
        # Mock push data
        push_data = {
            'to': 'push_token_123',
            'title': 'Security Alert',
            'body': 'Unauthorized access attempt',
            'priority': 'high',
            'data': {
                'type': 'security_alert',
                'location': 'Building A'
            }
        }
        
        # Mock FCM client
        mock_response = MagicMock()
        mock_response.success = 1
        mock_response.canonical_ids = []
        mock_response.failure = 0
        mock_response.multicast_id = 123456
        
        with patch('firebase_admin.messaging.send') as mock_send:
            mock_send.return_value = mock_response
            
            # Send push notification
            result = notifier.send(push_data)
            
            self.assertIsNotNone(result)
            self.assertTrue(result['success'])
            self.assertEqual(result['channel'], 'push')
            self.assertEqual(result['message_id'], 'push_001')
            self.assertEqual(result['recipients'], ['push_token_123'])
            self.assertEqual(result['delivered_count'], 1)
            
            # Verify FCM call
            mock_send.assert_called_once()
    
    def test_webhook_notifier_initialization(self):
        """Test WebhookNotifier initialization"""
        config = {
            'url': 'https://webhook.company.com/security-alerts',
            'webhook_token': 'test_token',
            'timeout': 30,
            'retry_attempts': 3
        }
        
        notifier = WebhookNotifier(config)
        
        self.assertIsNotNone(notifier)
        self.assertIsNotNone(notifier.config)
        self.assertIsNotNone(notifier.session)
        
        # Check configuration
        self.assertEqual(notifier.config['url'], 'https://webhook.company.com/security-alerts')
        self.assertEqual(notifier.config['webhook_token'], 'test_token')
    
    def test_webhook_notifier_send(self):
        """Test WebhookNotifier send functionality"""
        config = {
            'url': 'https://webhook.company.com/security-alerts',
            'webhook_token': 'test_token'
        }
        
        notifier = WebhookNotifier(config)
        
        # Mock webhook data
        webhook_data = {
            'event': 'security_alert',
            'timestamp': datetime.now().isoformat(),
            'location': 'Building A',
            'title': 'Security Alert',
            'message': 'Unauthorized access attempt'
        }
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'success': True}
        
        with patch('requests.post') as mock_post:
            mock_post.return_value = mock_response
            
            # Send webhook
            result = notifier.send(webhook_data)
            
            self.assertIsNotNone(result)
            self.assertTrue(result['success'])
            self.assertEqual(result['channel'], 'webhook')
            self.assertEqual(result['message_id'], 'webhook_001')
            self.assertEqual(result['status_code'], 200)
            
            # Verify webhook call
            mock_post.assert_called_once()
    
    def test_performance_benchmark(self):
        """Test notification performance benchmarking"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Mock notification responses
        mock_response = {
            'success': True,
            'channel': 'email',
            'message_id': 'email_001',
            'recipients': ['security@company.com'],
            'timestamp': datetime.now(),
            'processing_time': 0.1
        }
        
        with patch.object(notifier.notifiers['email'], 'send') as mock_email:
            mock_email.return_value = mock_response
            
            # Run benchmark
            benchmark_results = notifier.benchmark_notification_delivery(self.test_notification)
            
            self.assertIsNotNone(benchmark_results)
            self.assertIn('total_processing_time', benchmark_results)
            self.assertIn('channel_performance', benchmark_results)
            self.assertIn('throughput', benchmark_results)
            self.assertIn('memory_usage', benchmark_results)
            
            # Check benchmark values
            self.assertIsInstance(benchmark_results['total_processing_time'], float)
            self.assertIsInstance(benchmark_results['throughput'], float)
            self.assertIsInstance(benchmark_results['memory_usage'], float)
            
            # Check channel performance
            self.assertIsInstance(benchmark_results['channel_performance'], dict)
            self.assertIn('email', benchmark_results['channel_performance'])
    
    def test_error_handling(self):
        """Test error handling"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Test SMTP error
        with patch.object(notifier.notifiers['email'], 'send') as mock_email:
            mock_email.side_effect = Exception("SMTP connection failed")
            
            # Send notification with error
            result = notifier.send_notification(self.test_notification)
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertIn('SMTP connection failed', result['error'])
            self.assertIn('channel_responses', result)
            self.assertFalse(result['channel_responses']['email']['success'])
        
        # Test SMS provider error
        with patch.object(notifier.notifiers['sms'], 'send') as mock_sms:
            mock_sms.side_effect = Exception("SMS provider unavailable")
            
            # Send notification with error
            result = notifier.send_notification(self.test_notification)
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)
            self.assertIn('SMS provider unavailable', result['error'])
            self.assertFalse(result['channel_responses']['sms']['success'])
    
    def test_cache_functionality(self):
        """Test notification caching"""
        notifier = EnhancedNotifier(self.test_config)
        
        # Mock notification response
        mock_response = {
            'success': True,
            'channel': 'email',
            'message_id': 'email_001',
            'recipients': ['security@company.com'],
            'timestamp': datetime.now(),
            'processing_time': 0.1
        }
        
        with patch.object(notifier.notifiers['email'], 'send') as mock_email:
            mock_email.return_value = mock_response
            
            # First notification (should hit notifiers)
            result1 = notifier.send_notification(self.test_notification)
            
            # Second notification (should use cache)
            result2 = notifier.send_notification(self.test_notification)
            
            # Verify cache was used
            self.assertTrue(hasattr(notifier, 'notification_cache'))
            self.assertTrue(len(notifier.notification_cache) > 0)
            
            # Results should be the same
            self.assertEqual(result1['success'], result2['success'])
            self.assertEqual(result1['notification_id'], result2['notification_id'])
            
            # Notifier should be called only once (second call uses cache)
            mock_email.assert_called_once()

if __name__ == '__main__':
    unittest.main()