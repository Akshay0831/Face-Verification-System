"""Test enhanced notification functionality"""

import unittest
import os
import sys
import time
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced.notifications.notifiers import *
from enhanced.notifications.channels import *
from enhanced.notifications.templates import *

class TestEnhancedNotifications(unittest.TestCase):
    """Test cases for Enhanced Notification Systems"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test notification data
        self.test_notification = {
            'type': 'face_detected',
            'person_id': 'test_person',
            'name': 'Test Person',
            'timestamp': time.time(),
            'confidence': 0.95,
            'location': 'main_entrance'
        }
        
        # Create test events
        self.test_events = [
            {
                'type': 'face_detected',
                'person_id': 'person1',
                'name': 'Person 1',
                'timestamp': time.time(),
                'confidence': 0.95,
                'location': 'entrance'
            },
            {
                'type': 'face_detected',
                'person_id': 'person2',
                'name': 'Person 2',
                'timestamp': time.time(),
                'confidence': 0.88,
                'location': 'exit'
            }
        ]
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_email_notification_channel_initialization(self):
        """Test EmailNotificationChannel initialization"""
        channel = EmailNotificationChannel()
        
        self.assertIsNotNone(channel)
        self.assertEqual(channel.smtp_server, 'smtp.gmail.com')
        self.assertEqual(channel.smtp_port, 587)
        self.assertEqual(channel.use_tls, True)
        self.assertEqual(channel.sender_email, '')
        self.assertEqual(sender_email, '')
    
    def test_email_notification_channel_send(self):
        """Test email notification sending"""
        channel = EmailNotificationChannel()
        
        # Mock SMTP server
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value = mock_server
            
            # Send email
            success = channel.send(
                to='test@example.com',
                subject='Test Notification',
                body='This is a test notification',
                html_body='<html><body>Test</body></html>'
            )
            
            self.assertTrue(success)
            mock_smtp.assert_called_once_with('smtp.gmail.com', 587)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once()
            mock_server.send_message.assert_called_once()
            mock_server.quit.assert_called_once()
    
    def test_sms_notification_channel_initialization(self):
        """Test SMSNotificationChannel initialization"""
        channel = SMSNotificationChannel()
        
        self.assertIsNotNone(channel)
        self.assertEqual(channel.account_sid, '')
        self.assertEqual(channel.auth_token, '')
        self.assertEqual(channel.from_number, '')
    
    def test_sms_notification_channel_send(self):
        """Test SMS notification sending"""
        channel = SMSNotificationChannel()
        
        # Mock Twilio client
        with patch('twilio.rest.Client') as mock_client:
            mock_twilio = Mock()
            mock_client.return_value = mock_twilio
            
            # Send SMS
            success = channel.send(
                to='+1234567890',
                message='This is a test SMS notification'
            )
            
            self.assertTrue(success)
            mock_client.assert_called_once()
            mock_twilio.messages.create.assert_called_once()
    
    def test_push_notification_channel_initialization(self):
        """Test PushNotificationChannel initialization"""
        channel = PushNotificationChannel()
        
        self.assertIsNotNone(channel)
        self.assertEqual(channel.fcm_server_key, '')
        self.assertEqual(channel.apns_topic, '')
    
    def test_push_notification_channel_send(self):
        """Test push notification sending"""
        channel = PushNotificationChannel()
        
        # Mock FCM client
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'success': True}
            mock_post.return_value = mock_response
            
            # Send push notification
            success = channel.send(
                device_token='test_token',
                title='Test Notification',
                body='This is a test push notification',
                data={'key': 'value'}
            )
            
            self.assertTrue(success)
            mock_post.assert_called_once()
    
    def test_webhook_notification_channel_initialization(self):
        """Test WebhookNotificationChannel initialization"""
        channel = WebhookNotificationChannel()
        
        self.assertIsNotNone(channel)
        self.assertEqual(channel.webhook_url, '')
        self.assertEqual(channel.headers, {'Content-Type': 'application/json'})
    
    def test_webhook_notification_channel_send(self):
        """Test webhook notification sending"""
        channel = WebhookNotificationChannel()
        channel.webhook_url = 'https://example.com/webhook'
        
        # Mock requests
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'success': True}
            mock_post.return_value = mock_response
            
            # Send webhook
            success = channel.send(
                data=self.test_notification,
                headers={'Authorization': 'Bearer token'}
            )
            
            self.assertTrue(success)
            mock_post.assert_called_once()
    
    def test_console_notification_channel_initialization(self):
        """Test ConsoleNotificationChannel initialization"""
        channel = ConsoleNotificationChannel()
        
        self.assertIsNotNone(channel)
        self.assertEqual(channel.use_colors, True)
    
    def test_console_notification_channel_send(self):
        """Test console notification sending"""
        channel = ConsoleNotificationChannel()
        
        # Mock print function
        with patch('builtins.print') as mock_print:
            # Send console notification
            success = channel.send(
                title='Test Notification',
                body='This is a test console notification'
            )
            
            self.assertTrue(success)
            mock_print.assert_called()
    
    def test_file_notification_channel_initialization(self):
        """Test FileNotificationChannel initialization"""
        channel = FileNotificationChannel()
        
        self.assertIsNotNone(channel)
        self.assertEqual(channel.log_file, os.path.join(self.temp_dir, 'notifications.log'))
    
    def test_file_notification_channel_send(self):
        """Test file notification sending"""
        channel = FileNotificationChannel()
        channel.log_file = os.path.join(self.temp_dir, 'notifications.log')
        
        # Send file notification
        success = channel.send(
            title='Test Notification',
            body='This is a test file notification',
            data=self.test_notification
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(channel.log_file))
        
        # Check file content
        with open(channel.log_file, 'r') as f:
            content = f.read()
            self.assertIn('Test Notification', content)
            self.assertIn('test_person', content)
    
    def test_notification_manager_initialization(self):
        """Test NotificationManager initialization"""
        manager = NotificationManager()
        
        self.assertIsNotNone(manager)
        self.assertIsInstance(manager.channels, dict)
        self.assertEqual(len(manager.channels), 0)
        self.assertIsNotNone(manager.templates)
        self.assertEqual(manager.notification_history, [])
    
    def test_notification_manager_add_channel(self):
        """Test adding notification channels"""
        manager = NotificationManager()
        
        # Add email channel
        email_channel = EmailNotificationChannel()
        manager.add_channel('email', email_channel)
        
        self.assertIn('email', manager.channels)
        self.assertEqual(len(manager.channels), 1)
    
    def test_notification_manager_send_notification(self):
        """Test sending notification through manager"""
        manager = NotificationManager()
        
        # Add mock channel
        mock_channel = Mock()
        mock_channel.send.return_value = True
        manager.add_channel('test', mock_channel)
        
        # Send notification
        success = manager.send_notification(
            channel_name='test',
            title='Test Notification',
            body='This is a test notification',
            data=self.test_notification
        )
        
        self.assertTrue(success)
        mock_channel.send.assert_called_once()
    
    def test_notification_manager_send_multiple_channels(self):
        """Test sending notification through multiple channels"""
        manager = NotificationManager()
        
        # Add multiple mock channels
        mock_channel1 = Mock()
        mock_channel1.send.return_value = True
        
        mock_channel2 = Mock()
        mock_channel2.send.return_value = True
        
        manager.add_channel('channel1', mock_channel1)
        manager.add_channel('channel2', mock_channel2)
        
        # Send notification to both channels
        success = manager.send_notification(
            channel_names=['channel1', 'channel2'],
            title='Test Notification',
            body='This is a test notification'
        )
        
        self.assertTrue(success)
        mock_channel1.send.assert_called_once()
        mock_channel2.send.assert_called_once()
    
    def test_notification_manager_send_to_all_channels(self):
        """Test sending notification to all channels"""
        manager = NotificationManager()
        
        # Add multiple mock channels
        mock_channel1 = Mock()
        mock_channel1.send.return_value = True
        
        mock_channel2 = Mock()
        mock_channel2.send.return_value = True
        
        manager.add_channel('channel1', mock_channel1)
        manager.add_channel('channel2', mock_channel2)
        
        # Send notification to all channels
        success = manager.send_notification_to_all(
            title='Test Notification',
            body='This is a test notification'
        )
        
        self.assertTrue(success)
        mock_channel1.send.assert_called_once()
        mock_channel2.send.assert_called_once()
    
    def test_notification_manager_get_notification_history(self):
        """Test getting notification history"""
        manager = NotificationManager()
        
        # Add some notifications to history
        manager.notification_history.append({
            'timestamp': time.time(),
            'title': 'Test 1',
            'body': 'Body 1'
        })
        
        manager.notification_history.append({
            'timestamp': time.time(),
            'title': 'Test 2',
            'body': 'Body 2'
        })
        
        # Get history
        history = manager.get_notification_history()
        
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 2)
    
    def test_notification_manager_clear_history(self):
        """Test clearing notification history"""
        manager = NotificationManager()
        
        # Add some notifications to history
        manager.notification_history.append({
            'timestamp': time.time(),
            'title': 'Test',
            'body': 'Body'
        })
        
        # Clear history
        manager.clear_notification_history()
        
        self.assertEqual(len(manager.notification_history), 0)
    
    def test_notification_template_basic_initialization(self):
        """Test basic notification template initialization"""
        template = BasicNotificationTemplate()
        
        self.assertIsNotNone(template)
        self.assertEqual(template.title_template, '{{title}}')
        self.assertEqual(template.body_template, '{{body}}')
    
    def test_notification_template_basic_render(self):
        """Test basic notification template rendering"""
        template = BasicNotificationTemplate()
        
        # Render template
        result = template.render(
            title='Test Title',
            body='Test Body',
            data=self.test_notification
        )
        
        self.assertEqual(result['title'], 'Test Title')
        self.assertEqual(result['body'], 'Test Body')
    
    def test_notification_template_advanced_initialization(self):
        """Test advanced notification template initialization"""
        template = AdvancedNotificationTemplate()
        
        self.assertIsNotNone(template)
        self.assertIn('title', template.templates)
        self.assertIn('body', template.templates)
        self.assertIn('html', template.templates)
    
    def test_notification_template_advanced_render(self):
        """Test advanced notification template rendering"""
        template = AdvancedNotificationTemplate()
        
        # Render template
        result = template.render(
            title='Face Detection Alert',
            body='Person detected with high confidence',
            data=self.test_notification
        )
        
        self.assertIn('title', result)
        self.assertIn('body', result)
        self.assertIn('html', result)
        
        # Check content includes test data
        self.assertIn('Test Person', result['body'])
        self.assertIn('test_person', result['body'])
    
    def test_notification_manager_templates(self):
        """Test notification manager template functionality"""
        manager = NotificationManager()
        
        # Test default templates
        self.assertIn('basic', manager.templates)
        self.assertIn('advanced', manager.templates)
        
        # Test template rendering
        result = manager.render_template(
            template_name='basic',
            title='Test',
            body='Test Body'
        )
        
        self.assertEqual(result['title'], 'Test')
        self.assertEqual(result['body'], 'Test Body')
    
    def test_notification_manager_error_handling(self):
        """Test notification manager error handling"""
        manager = NotificationManager()
        
        # Test sending to non-existent channel
        success = manager.send_notification(
            channel_name='nonexistent',
            title='Test',
            body='Test'
        )
        
        self.assertFalse(success)
        
        # Test sending with invalid data
        success = manager.send_notification(
            channel_name='console',
            title=None,
            body=None
        )
        
        # Should still work with basic defaults
        self.assertTrue(success)
    
    def test_notification_channel_error_handling(self):
        """Test notification channel error handling"""
        # Test email channel with invalid credentials
        channel = EmailNotificationChannel()
        channel.sender_email = 'invalid@email.com'
        channel.sender_password = 'invalid'
        
        with patch('smtplib.SMTP') as mock_smtp:
            mock_smtp.side_effect = Exception("Connection failed")
            
            success = channel.send(
                to='test@example.com',
                subject='Test',
                body='Test body'
            )
            
            self.assertFalse(success)
    
    def test_notification_performance(self):
        """Test notification performance"""
        import time
        
        manager = NotificationManager()
        
        # Add mock channel
        mock_channel = Mock()
        mock_channel.send.return_value = True
        manager.add_channel('test', mock_channel)
        
        # Test performance
        start_time = time.time()
        
        for i in range(10):
            manager.send_notification(
                channel_name='test',
                title=f'Test {i}',
                body=f'Test body {i}'
            )
        
        end_time = time.time()
        
        # Should be fast (less than 5 seconds for 10 notifications)
        self.assertLess(end_time - start_time, 5.0)
    
    def test_notification_batching(self):
        """Test notification batching"""
        manager = NotificationManager()
        
        # Add mock channel
        mock_channel = Mock()
        mock_channel.send.return_value = True
        manager.add_channel('test', mock_channel)
        
        # Send batch notifications
        notifications = []
        for i in range(5):
            notifications.append({
                'title': f'Test {i}',
                'body': f'Test body {i}',
                'data': self.test_notification
            })
        
        success = manager.send_batch_notifications(
            channel_name='test',
            notifications=notifications
        )
        
        self.assertTrue(success)
        
        # Should have been called once with all notifications
        mock_channel.send.assert_called_once()
        call_args = mock_channel.send.call_args
        self.assertEqual(len(call_args[1]['data']), 5)
    
    def test_notification_filtering(self):
        """Test notification filtering"""
        manager = NotificationManager()
        
        # Add test notifications to history
        test_data = [
            {'type': 'face_detected', 'confidence': 0.95},
            {'type': 'face_detected', 'confidence': 0.85},
            {'type': 'unauthorized', 'confidence': 0.99}
        ]
        
        for i, data in enumerate(test_data):
            manager.notification_history.append({
                'timestamp': time.time() + i,
                'title': f'Test {i}',
                'body': f'Body {i}',
                'data': data
            })
        
        # Filter by type
        filtered = manager.filter_notifications(lambda n: n['data']['type'] == 'face_detected')
        self.assertEqual(len(filtered), 2)
        
        # Filter by confidence
        filtered = manager.filter_notifications(lambda n: n['data']['confidence'] > 0.9)
        self.assertEqual(len(filtered), 2)
    
    def test_notification_scheduler(self):
        """Test notification scheduling"""
        from enhanced.notifications.scheduler import NotificationScheduler
        
        scheduler = NotificationScheduler()
        manager = NotificationManager()
        
        # Add mock channel
        mock_channel = Mock()
        mock_channel.send.return_value = True
        manager.add_channel('test', mock_channel)
        
        # Schedule notification
        success = scheduler.schedule_notification(
            manager=manager,
            channel_name='test',
            title='Scheduled Test',
            body='This is a scheduled notification',
            delay=1  # 1 second delay
        )
        
        self.assertTrue(success)
        
        # Wait for notification to be sent
        import time
        time.sleep(2)
        
        # Check if notification was sent
        mock_channel.send.assert_called_once()

if __name__ == '__main__':
    unittest.main()