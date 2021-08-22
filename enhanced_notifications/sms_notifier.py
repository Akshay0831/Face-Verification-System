"""SMS notification plugin"""

import requests
from typing import List, Dict, Any, Optional
import json
import os

from core.base import INotifier, NotificationResult, PluginMetadata, DeviceType
from utils import get_logger

logger = get_logger('sms_notifier')


class SMSNotifier(INotifier):
    """SMS notification implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.service = self.config.get('service', 'twilio')  # twilio, vonage, aws_sns
        self.api_key = self.config.get('api_key', '')
        self.api_secret = self.config.get('api_secret', '')
        self.from_number = self.config.get('from_number', '')
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.api_timeout = self.config.get('api_timeout', 30)
        
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name='sms_notifier',
            version='1.0.0',
            description='SMS notification system',
            author='Face Verification System Team',
            dependencies=['requests'],
            device_compatibility=[DeviceType.WINDOWS, DeviceType.LINUX, DeviceType.RASPBERRY_PI]
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the SMS notifier"""
        try:
            self.config.update(config)
            
            # Load credentials from environment if not provided
            if not self.api_key:
                self.api_key = os.getenv('SMS_API_KEY', '')
            if not self.api_secret:
                self.api_secret = os.getenv('SMS_API_SECRET', '')
            if not self.from_number:
                self.from_number = os.getenv('SMS_FROM_NUMBER', '')
            
            # Validate required configuration
            if not self.service or not self.api_key or not self.from_number:
                logger.error("Missing required SMS configuration")
                return False
            
            logger.info(f"SMS notifier initialized for {self.service}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing SMS notifier: {e}")
            return False
    
    def send_notification(self, 
                         user_id: Optional[str] = None,
                         message: str = "",
                         **kwargs) -> NotificationResult:
        """Send an SMS notification"""
        try:
            # Get phone number from kwargs or config
            to_number = kwargs.get('to_number', self.config.get('default_to_number', ''))
            
            if not to_number:
                logger.error("No recipient phone number provided")
                return NotificationResult(success=False, message="No recipient phone number provided")
            
            # Clean phone number (remove non-numeric characters)
            to_number = ''.join(c for c in to_number if c.isdigit())
            
            # Limit message length (SMS standard)
            if len(message) > 160:
                message = message[:157] + "..."
            
            # Send SMS based on service
            if self.service == 'twilio':
                return self._send_twilio_sms(to_number, message)
            elif self.service == 'vonage':
                return self._send_vonage_sms(to_number, message)
            elif self.service == 'aws_sns':
                return self._send_aws_sns_sms(to_number, message)
            else:
                logger.error(f"Unsupported SMS service: {self.service}")
                return NotificationResult(success=False, message=f"Unsupported SMS service: {self.service}")
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return NotificationResult(success=False, message=f"SMS send failed: {e}")
    
    def _send_twilio_sms(self, to_number: str, message: str) -> NotificationResult:
        """Send SMS using Twilio API"""
        try:
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.api_key}/Messages.json"
            
            data = {
                'From': self.from_number,
                'To': to_number,
                'Body': message
            }
            
            # Send API request
            response = requests.post(
                url,
                data=data,
                auth=(self.api_key, self.api_secret),
                timeout=self.api_timeout
            )
            
            if response.status_code == 201:
                logger.info(f"SMS sent successfully to {to_number} via Twilio")
                return NotificationResult(success=True, message="SMS sent successfully")
            else:
                error_msg = response.json().get('message', 'Unknown error')
                logger.error(f"Twilio API error: {error_msg}")
                return NotificationResult(success=False, message=f"Twilio API error: {error_msg}")
                
        except Exception as e:
            logger.error(f"Twilio SMS send failed: {e}")
            return NotificationResult(success=False, message=f"Twilio send failed: {e}")
    
    def _send_vonage_sms(self, to_number: str, message: str) -> NotificationResult:
        """Send SMS using Vonage API"""
        try:
            url = "https://rest.nexmo.com/sms/json"
            
            data = {
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'from': self.from_number,
                'to': to_number,
                'text': message
            }
            
            # Send API request
            response = requests.post(url, data=data, timeout=self.api_timeout)
            
            response_data = response.json()
            
            if response_data.get('messages', [{}])[0].get('status') == '0':
                logger.info(f"SMS sent successfully to {to_number} via Vonage")
                return NotificationResult(success=True, message="SMS sent successfully")
            else:
                error_text = response_data.get('messages', [{}])[0].get('error-text', 'Unknown error')
                logger.error(f"Vonage API error: {error_text}")
                return NotificationResult(success=False, message=f"Vonage API error: {error_text}")
                
        except Exception as e:
            logger.error(f"Vonage SMS send failed: {e}")
            return NotificationResult(success=False, message=f"Vonage send failed: {e}")
    
    def _send_aws_sns_sms(self, to_number: str, message: str) -> NotificationResult:
        """Send SMS using AWS SNS"""
        try:
            # This is a simplified implementation
            # In production, you'd need AWS SDK and proper authentication
            url = "https://sns.us-east-1.amazonaws.com/"
            
            payload = {
                'Action': 'Publish',
                'TopicArn': self.config.get('topic_arn', ''),
                'Message': message,
                'MessageStructure': 'string'
            }
            
            # Add AWS credentials if available
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID', '')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY', '')
            
            if aws_access_key and aws_secret_key:
                headers = {
                    'X-Amz-Algorithm': 'AWS4-HMAC-SHA256',
                    'X-Amz-Credential': f"{aws_access_key}/20230101/us-east-1/sns/aws4_request",
                    'X-Amz-Date': '20230101T000000Z',
                    'X-Amz-SignedHeaders': 'host',
                    'Authorization': f'AWS4-HMAC-SHA256 Credential={aws_access_key}/20230101/us-east-1/sns/aws4_request, SignedHeaders=host, Signature=placeholder'
                }
                
                response = requests.post(url, data=payload, headers=headers, timeout=self.api_timeout)
            else:
                logger.error("AWS credentials not provided")
                return NotificationResult(success=False, message="AWS credentials not provided")
            
            if response.status_code == 200:
                logger.info(f"SMS sent successfully to {to_number} via AWS SNS")
                return NotificationResult(success=True, message="SMS sent successfully")
            else:
                error_msg = response.json().get('Message', 'Unknown error') if response.text else 'Unknown error'
                logger.error(f"AWS SNS error: {error_msg}")
                return NotificationResult(success=False, message=f"AWS SNS error: {error_msg}")
                
        except Exception as e:
            logger.error(f"AWS SNS SMS send failed: {e}")
            return NotificationResult(success=False, message=f"AWS SNS send failed: {e}")
    
    def send_verification_sms(self, 
                            user_id: str,
                            phone_number: str,
                            verification_code: str) -> NotificationResult:
        """Send a verification SMS"""
        message = f"""
Face Verification System

Your verification code is: {verification_code}

Please enter this code in the application to complete your verification.

If you did not request this code, please ignore this message.
        """
        
        return self.send_notification(
            user_id=user_id,
            to_number=phone_number,
            message=message
        )
    
    def send_alert_sms(self,
                      user_id: str,
                      phone_number: str,
                      alert_type: str = "Security Alert",
                      details: Optional[Dict[str, Any]] = None) -> NotificationResult:
        """Send an alert SMS"""
        details_str = ""
        if details:
            details_str = " Details: "
            for key, value in details.items():
                details_str += f"{key}={value}, "
            details_str = details_str.rstrip(", ")
        
        message = f"""
SECURITY ALERT - Face Verification System

User: {user_id}
Alert: {alert_type}
{details_str}

Take immediate action.
        """
        
        return self.send_notification(
            user_id=user_id,
            to_number=phone_number,
            message=message
        )
    
    def get_supported_modes(self) -> List[str]:
        """Return supported notification modes"""
        return ['sms', 'verification', 'alert', 'bulk_sms']
    
    def get_required_config(self) -> List[str]:
        """Return required configuration parameters"""
        return ['service', 'api_key', 'api_secret', 'from_number']