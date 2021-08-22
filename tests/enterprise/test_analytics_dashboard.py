"""Test analytics dashboard functionality"""

import unittest
import os
import sys
import json
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from enterprise.analytics_dashboard import AnalyticsDashboard, DataProcessor, ReportGenerator

class TestAnalyticsDashboard(unittest.TestCase):
    """Test cases for Analytics Dashboard"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_detection_data = [
            {
                'timestamp': datetime.now(),
                'person_id': 'person1',
                'name': 'Person 1',
                'confidence': 0.95,
                'location': 'entrance',
                'device_id': 'camera1',
                'detection_type': 'face',
                'processed': True
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=5),
                'person_id': 'person2',
                'name': 'Person 2',
                'confidence': 0.88,
                'location': 'exit',
                'device_id': 'camera2',
                'detection_type': 'face',
                'processed': True
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=10),
                'person_id': 'unauthorized',
                'name': 'Unknown',
                'confidence': 0.92,
                'location': 'server_room',
                'device_id': 'camera1',
                'detection_type': 'face',
                'processed': False,
                'alert_triggered': True
            }
        ]
        
        # Create test user data
        self.test_user_data = [
            {
                'user_id': 'user1',
                'username': 'admin',
                'role': 'administrator',
                'last_login': datetime.now(),
                'login_count': 150,
                'failed_logins': 0
            },
            {
                'user_id': 'user2',
                'username': 'operator',
                'role': 'operator',
                'last_login': datetime.now() - timedelta(hours=2),
                'login_count': 75,
                'failed_logins': 2
            }
        ]
        
        # Create test system metrics
        self.test_system_metrics = [
            {
                'timestamp': datetime.now(),
                'cpu_usage': 0.45,
                'memory_usage': 0.67,
                'disk_usage': 0.34,
                'network_usage': 0.12,
                'fps': 28.5,
                'active_devices': 3
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=5),
                'cpu_usage': 0.52,
                'memory_usage': 0.71,
                'disk_usage': 0.34,
                'network_usage': 0.15,
                'fps': 27.2,
                'active_devices': 3
            }
        ]
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_analytics_dashboard_initialization(self):
        """Test AnalyticsDashboard initialization"""
        dashboard = AnalyticsDashboard()
        
        self.assertIsNotNone(dashboard)
        self.assertIsNotNone(dashboard.data_processor)
        self.assertIsNotNone(dashboard.report_generator)
        self.assertIsNotNone(dashboard.config)
        self.assertIsInstance(dashboard.cached_data, dict)
        self.assertEqual(len(dashboard.cached_data), 0)
    
    def test_analytics_dashboard_custom_config(self):
        """Test AnalyticsDashboard with custom configuration"""
        config_file = os.path.join(self.temp_dir, 'analytics_config.json')
        
        # Save test config
        test_config = {
            'database': {
                'path': os.path.join(self.temp_dir, 'analytics.db'),
                'backup_enabled': True,
                'backup_interval': 3600
            },
            'caching': {
                'enabled': True,
                'max_size': 1000,
                'ttl': 300
            },
            'reporting': {
                'output_format': 'json',
                'schedule': 'daily',
                'email_notifications': True
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        
        dashboard = AnalyticsDashboard(config_file=config_file)
        
        self.assertEqual(dashboard.config['database']['backup_enabled'], True)
        self.assertEqual(dashboard.config['caching']['enabled'], True)
        self.assertEqual(dashboard.config['reporting']['schedule'], 'daily')
    
    def test_data_processor_initialization(self):
        """Test DataProcessor initialization"""
        processor = DataProcessor()
        
        self.assertIsNotNone(processor)
        self.assertIsNotNone(processor.data_cache)
        self.assertIsNotNone(processor.config)
        self.assertEqual(len(processor.data_cache), 0)
    
    def test_data_processor_load_detection_data(self):
        """Test detection data loading"""
        processor = DataProcessor()
        
        # Load test data
        data = processor.load_detection_data(self.test_detection_data)
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 3)
        
        # Check columns
        expected_columns = ['timestamp', 'person_id', 'name', 'confidence', 
                          'location', 'device_id', 'detection_type', 'processed', 'alert_triggered']
        for col in expected_columns:
            self.assertIn(col, data.columns)
    
    def test_data_processor_load_user_data(self):
        """Test user data loading"""
        processor = DataProcessor()
        
        # Load test data
        data = processor.load_user_data(self.test_user_data)
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 2)
        
        # Check columns
        expected_columns = ['user_id', 'username', 'role', 'last_login', 'login_count', 'failed_logins']
        for col in expected_columns:
            self.assertIn(col, data.columns)
    
    def test_data_processor_load_system_metrics(self):
        """Test system metrics loading"""
        processor = DataProcessor()
        
        # Load test data
        data = processor.load_system_metrics(self.test_system_metrics)
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 2)
        
        # Check columns
        expected_columns = ['timestamp', 'cpu_usage', 'memory_usage', 'disk_usage', 
                          'network_usage', 'fps', 'active_devices']
        for col in expected_columns:
            self.assertIn(col, data.columns)
    
    def test_data_processor_filter_data(self):
        """Test data filtering"""
        processor = DataProcessor()
        detection_data = processor.load_detection_data(self.test_detection_data)
        
        # Filter by location
        filtered = processor.filter_data(detection_data, location='entrance')
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]['location'], 'entrance')
        
        # Filter by confidence threshold
        filtered = processor.filter_data(detection_data, confidence_min=0.9)
        self.assertEqual(len(filtered), 3)  # All detections above 0.9
        
        # Filter by time range
        start_time = datetime.now() - timedelta(minutes=8)
        end_time = datetime.now() - timedelta(minutes=2)
        filtered = processor.filter_data(
            detection_data, 
            start_time=start_time, 
            end_time=end_time
        )
        self.assertEqual(len(filtered), 1)  # Only middle record
    
    def test_data_processor_aggregate_data(self):
        """Test data aggregation"""
        processor = DataProcessor()
        detection_data = processor.load_detection_data(self.test_detection_data)
        
        # Aggregate by location
        aggregated = processor.aggregate_data(detection_data, group_by='location')
        self.assertIsInstance(aggregated, pd.DataFrame)
        self.assertIn('count', aggregated.columns)
        
        # Aggregate by device
        aggregated = processor.aggregate_data(detection_data, group_by='device_id')
        self.assertIsInstance(aggregated, pd.DataFrame)
        self.assertIn('count', aggregated.columns)
        
        # Aggregate by time period
        aggregated = processor.aggregate_data(
            detection_data, 
            group_by='location', 
            agg_func={'confidence': 'mean'}
        )
        self.assertIsInstance(aggregated, pd.DataFrame)
        self.assertIn('confidence_mean', aggregated.columns)
    
    def test_data_processor_calculate_statistics(self):
        """Test statistics calculation"""
        processor = DataProcessor()
        detection_data = processor.load_detection_data(self.test_detection_data)
        
        # Calculate basic statistics
        stats = processor.calculate_statistics(detection_data, 'confidence')
        self.assertIn('mean', stats)
        self.assertIn('median', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        
        # Test numeric data
        self.assertIsInstance(stats['mean'], float)
        self.assertIsInstance(stats['median'], float)
        self.assertIsInstance(stats['std'], float)
        self.assertIsInstance(stats['min'], float)
        self.assertIsInstance(stats['max'], float)
    
    def test_data_processor_detect_anomalies(self):
        """Test anomaly detection"""
        processor = DataProcessor()
        
        # Create test data with anomaly
        normal_data = [10, 12, 11, 13, 14, 12, 11, 10, 12, 11]
        anomaly_data = normal_data + [100]  # Add anomaly
        
        # Detect anomalies
        anomalies = processor.detect_anomalies(anomaly_data, threshold=2.0)
        self.assertIsInstance(anomalies, list)
        self.assertGreater(len(anomalies), 0)
        
        # Check that anomaly was detected
        self.assertIn(100, anomalies)
    
    def test_data_processor_export_data(self):
        """Test data export"""
        processor = DataProcessor()
        detection_data = processor.load_detection_data(self.test_detection_data)
        
        # Export to CSV
        csv_file = os.path.join(self.temp_dir, 'export.csv')
        success = processor.export_data(detection_data, csv_file, format='csv')
        self.assertTrue(success)
        self.assertTrue(os.path.exists(csv_file))
        
        # Export to JSON
        json_file = os.path.join(self.temp_dir, 'export.json')
        success = processor.export_data(detection_data, json_file, format='json')
        self.assertTrue(success)
        self.assertTrue(os.path.exists(json_file))
        
        # Export to Excel
        excel_file = os.path.join(self.temp_dir, 'export.xlsx')
        success = processor.export_data(detection_data, excel_file, format='excel')
        self.assertTrue(success)
        self.assertTrue(os.path.exists(excel_file))
    
    def test_report_generator_initialization(self):
        """Test ReportGenerator initialization"""
        generator = ReportGenerator()
        
        self.assertIsNotNone(generator)
        self.assertIsNotNone(generator.config)
        self.assertIsNotNone(generator.templates)
        self.assertEqual(len(generator.templates), 0)
    
    def test_report_generator_create_report(self):
        """Test report creation"""
        generator = ReportGenerator()
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.load_detection_data.return_value = pd.DataFrame(self.test_detection_data)
        mock_processor.load_user_data.return_value = pd.DataFrame(self.test_user_data)
        mock_processor.load_system_metrics.return_value = pd.DataFrame(self.test_system_metrics)
        
        # Create report
        report_data = {
            'detection_data': self.test_detection_data,
            'user_data': self.test_user_data,
            'system_metrics': self.test_system_metrics,
            'time_range': {
                'start': datetime.now() - timedelta(hours=1),
                'end': datetime.now()
            }
        }
        
        # Generate PDF report
        pdf_file = os.path.join(self.temp_dir, 'report.pdf')
        success = generator.create_report(
            report_data=report_data,
            output_file=pdf_file,
            format='pdf'
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(pdf_file))
    
    def test_report_generator_create_html_report(self):
        """Test HTML report creation"""
        generator = ReportGenerator()
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.load_detection_data.return_value = pd.DataFrame(self.test_detection_data)
        mock_processor.load_user_data.return_value = pd.DataFrame(self.test_user_data)
        mock_processor.load_system_metrics.return_value = pd.DataFrame(self.test_system_metrics)
        
        # Create report
        report_data = {
            'detection_data': self.test_detection_data,
            'user_activity': self.test_user_data,
            'system_metrics': self.test_system_metrics
        }
        
        # Generate HTML report
        html_file = os.path.join(self.temp_dir, 'report.html')
        success = generator.create_report(
            report_data=report_data,
            output_file=html_file,
            format='html'
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(html_file))
        
        # Check HTML content
        with open(html_file, 'r') as f:
            content = f.read()
            self.assertIn('<html', content)
            self.assertIn('Face Verification Report', content)
    
    def test_report_generator_create_json_report(self):
        """Test JSON report creation"""
        generator = ReportGenerator()
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.load_detection_data.return_value = pd.DataFrame(self.test_detection_data)
        mock_processor.load_user_data.return_value = pd.DataFrame(self.test_user_data)
        mock_processor.load_system_metrics.return_value = pd.DataFrame(self.test_system_metrics)
        
        # Create report
        report_data = {
            'detection_data': self.test_detection_data,
            'user_activity': self.test_user_data,
            'system_metrics': self.test_system_metrics
        }
        
        # Generate JSON report
        json_file = os.path.join(self.temp_dir, 'report.json')
        success = generator.create_report(
            report_data=report_data,
            output_file=json_file,
            format='json'
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(json_file))
        
        # Check JSON content
        with open(json_file, 'r') as f:
            content = json.load(f)
            self.assertIn('report_metadata', content)
            self.assertIn('detection_data', content)
            self.assertIn('user_activity', content)
            self.assertIn('system_metrics', content)
    
    def test_analytics_dashboard_generate_detection_report(self):
        """Test detection report generation"""
        dashboard = AnalyticsDashboard()
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.load_detection_data.return_value = pd.DataFrame(self.test_detection_data)
        mock_processor.aggregate_data.return_value = pd.DataFrame({
            'location': ['entrance', 'exit', 'server_room'],
            'count': [1, 1, 1],
            'confidence_mean': [0.95, 0.88, 0.92]
        })
        dashboard.data_processor = mock_processor
        
        # Generate report
        report = dashboard.generate_detection_report(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        self.assertIsNotNone(report)
        self.assertIn('total_detections', report)
        self.assertIn('location_breakdown', report)
        self.assertIn('confidence_stats', report)
        self.assertIn('alerts_triggered', report)
    
    def test_analytics_dashboard_generate_user_report(self):
        """Test user activity report generation"""
        dashboard = AnalyticsDashboard()
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.load_user_data.return_value = pd.DataFrame(self.test_user_data)
        mock_processor.aggregate_data.return_value = pd.DataFrame({
            'role': ['administrator', 'operator'],
            'login_count': [150, 75],
            'failed_logins': [0, 2]
        })
        dashboard.data_processor = mock_processor
        
        # Generate report
        report = dashboard.generate_user_report(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        self.assertIsNotNone(report)
        self.assertIn('total_users', report)
        self.assertIn('role_breakdown', report)
        self.assertIn('login_stats', report)
        self.assertIn('security_events', report)
    
    def test_analytics_dashboard_generate_system_report(self):
        """Test system performance report generation"""
        dashboard = AnalyticsDashboard()
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.load_system_metrics.return_value = pd.DataFrame(self.test_system_metrics)
        mock_processor.aggregate_data.return_value = pd.DataFrame({
            'metric': ['cpu_usage', 'memory_usage', 'disk_usage', 'network_usage'],
            'mean': [0.485, 0.69, 0.34, 0.135],
            'max': [0.52, 0.71, 0.34, 0.15]
        })
        dashboard.data_processor = mock_processor
        
        # Generate report
        report = dashboard.generate_system_report(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        self.assertIsNotNone(report)
        self.assertIn('performance_metrics', report)
        self.assertIn('resource_usage', report)
        self.assertIn('device_status', report)
        self.assertIn('recommendations', report)
    
    def test_analytics_dashboard_generate_comprehensive_report(self):
        """Test comprehensive report generation"""
        dashboard = AnalyticsDashboard()
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.load_detection_data.return_value = pd.DataFrame(self.test_detection_data)
        mock_processor.load_user_data.return_value = pd.DataFrame(self.test_user_data)
        mock_processor.load_system_metrics.return_value = pd.DataFrame(self.test_system_metrics)
        dashboard.data_processor = mock_processor
        
        # Generate comprehensive report
        report = dashboard.generate_comprehensive_report(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        self.assertIsNotNone(report)
        self.assertIn('report_metadata', report)
        self.assertIn('executive_summary', report)
        self.assertIn('detection_analysis', report)
        self.assertIn('user_activity', report)
        self.assertIn('system_performance', report)
        self.assertIn('recommendations', report)
        self.assertIn('appendices', report)
    
    def test_analytics_dashboard_real_time_monitoring(self):
        """Test real-time monitoring"""
        dashboard = AnalyticsDashboard()
        
        # Test real-time data collection
        real_time_data = {
            'timestamp': datetime.now(),
            'current_detections': 2,
            'active_users': 5,
            'cpu_usage': 0.45,
            'memory_usage': 0.67,
            'network_usage': 0.12,
            'alert_count': 1
        }
        
        # Add real-time data
        dashboard.add_real_time_data(real_time_data)
        
        # Check metrics
        metrics = dashboard.get_current_metrics()
        self.assertIsNotNone(metrics)
        self.assertIn('current_detections', metrics)
        self.assertIn('active_users', metrics)
        self.assertIn('cpu_usage', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('network_usage', metrics)
        self.assertIn('alert_count', metrics)
    
    def test_analytics_dashboard_alert_system(self):
        """Test alert system"""
        dashboard = AnalyticsDashboard()
        
        # Test alert generation
        alert_data = {
            'type': 'high_confidence_detection',
            'severity': 'high',
            'message': 'High confidence detection of unauthorized person',
            'timestamp': datetime.now(),
            'details': {
                'person_id': 'unauthorized',
                'confidence': 0.95,
                'location': 'server_room'
            }
        }
        
        # Add alert
        dashboard.add_alert(alert_data)
        
        # Check alerts
        alerts = dashboard.get_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]['type'], 'high_confidence_detection')
        self.assertEqual(alerts[0]['severity'], 'high')
        
        # Test alert filtering
        high_severity_alerts = dashboard.filter_alerts(severity='high')
        self.assertEqual(len(high_severity_alerts), 1)
        
        time_filtered_alerts = dashboard.filter_alerts(
            start_time=datetime.now() - timedelta(minutes=5)
        )
        self.assertEqual(len(time_filtered_alerts), 1)
    
    def test_analytics_dashboard_data_retention(self):
        """Test data retention"""
        dashboard = AnalyticsDashboard()
        
        # Add test data
        for i in range(10):
            data = self.test_detection_data[0].copy()
            data['timestamp'] = datetime.now() - timedelta(days=i)
            dashboard.add_detection_data(data)
        
        # Check data
        self.assertEqual(len(dashboard.detection_data), 10)
        
        # Apply retention policy (keep last 7 days)
        dashboard.apply_retention_policy(days=7)
        
        # Check that old data was removed
        self.assertLessEqual(len(dashboard.detection_data), 7)
    
    def test_analytics_dashboard_export_dashboard(self):
        """Test dashboard export"""
        dashboard = AnalyticsDashboard()
        
        # Add some data
        for i in range(5):
            data = self.test_detection_data[0].copy()
            data['timestamp'] = datetime.now() - timedelta(minutes=i)
            dashboard.add_detection_data(data)
        
        # Export dashboard
        export_file = os.path.join(self.temp_dir, 'dashboard_export.json')
        success = dashboard.export_dashboard(export_file)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_file))
        
        # Check exported data
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
            self.assertIn('detection_data', exported_data)
            self.assertIn('alerts', exported_data)
            self.assertIn('metrics', exported_data)

if __name__ == '__main__':
    unittest.main()