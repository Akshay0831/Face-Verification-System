"""Performance analytics for enterprise features"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
from pathlib import Path
from .metrics_collector import RealTimeMetricsCollector

class PerformanceAnalytics:
    """Analyzes system performance trends and patterns"""
    
    def __init__(self, db_path: str = "enterprise/metrics.db"):
        self.db_path = db_path
        self.metrics_collector = RealTimeMetricsCollector(db_path)
        self.performance_thresholds = {
            'response_time_warning': 2.0,  # seconds
            'response_time_critical': 5.0,  # seconds
            'error_rate_warning': 5.0,     # percentage
            'error_rate_critical': 15.0,   # percentage
            'cpu_warning': 80.0,           # percentage
            'cpu_critical': 95.0,          # percentage
            'memory_warning': 80.0,        # percentage
            'memory_critical': 95.0       # percentage
        }
    
    def analyze_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze performance trends over specified days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query metrics for the specified period
        cursor.execute('''
            SELECT metric_name, value, timestamp 
            FROM metrics 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (start_date.isoformat(), end_date.isoformat()))
        
        metrics_data = cursor.fetchall()
        conn.close()
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(metrics_data, columns=['metric_name', 'value', 'timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        trends = {}
        
        # Analyze response time trends
        response_times = df[df['metric_name'] == 'face_verification_timing']
        if not response_times.empty:
            trends['response_time'] = {
                'average': response_times['value'].mean(),
                'median': response_times['value'].median(),
                'p95': response_times['value'].quantile(0.95),
                'p99': response_times['value'].quantile(0.99),
                'trend': self._calculate_trend(response_times['value']),
                'improvement_rate': self._calculate_improvement_rate(response_times['value'])
            }
        
        # Analyze error rate trends
        error_counts = df[df['metric_name'] == 'verification_errors']
        if not error_counts.empty:
            trends['error_rate'] = {
                'average_error_rate': error_counts['value'].mean(),
                'trend': self._calculate_trend(error_counts['value']),
                'peak_errors': error_counts['value'].max()
            }
        
        # Analyze throughput trends
        throughput_data = df[df['metric_name'] == 'requests_per_second']
        if not throughput_data.empty:
            trends['throughput'] = {
                'average': throughput_data['value'].mean(),
                'peak': throughput_data['value'].max(),
                'trend': self._calculate_trend(throughput_data['value'])
            }
        
        return trends
    
    def _calculate_trend(self, values: pd.Series) -> str:
        """Calculate trend direction from time series data"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression to determine trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_improvement_rate(self, values: pd.Series) -> float:
        """Calculate improvement rate over time"""
        if len(values) < 2:
            return 0.0
        
        first_half = values[:len(values)//2].mean()
        second_half = values[len(values)//2:].mean()
        
        if first_half == 0:
            return 0.0
        
        return ((first_half - second_half) / first_half) * 100
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'report_generated': datetime.now().isoformat(),
            'period': 'last_30_days',
            'trends': self.analyze_trends(30),
            'health_status': self.metrics_collector.get_health_status(),
            'thresholds': self.performance_thresholds,
            'recommendations': self._generate_performance_recommendations()
        }
        
        return report
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        health_status = self.metrics_collector.get_health_status()
        trends = self.analyze_trends(30)
        
        # Check response time recommendations
        if 'response_time' in trends:
            rt_trend = trends['response_time']['trend']
            if rt_trend == 'increasing':
                recommendations.append("Response times are increasing - consider optimizing detection algorithms or increasing server capacity")
            elif rt_trend == 'decreasing':
                recommendations.append("Response times are improving - current optimizations are working well")
        
        # Check error rate recommendations
        if 'error_rate' in trends:
            error_trend = trends['error_rate']['trend']
            if error_trend == 'increasing':
                recommendations.append("Error rates are increasing - investigate quality of input images and system health")
            elif error_trend == 'decreasing':
                recommendations.append("Error rates are decreasing - system reliability is improving")
        
        # Check resource utilization
        resources = health_status['resource_utilization']
        for resource, usage in resources.items():
            if usage > self.performance_thresholds[f'{resource}_critical']:
                recommendations.append(f"{resource.capitalize()} usage is critical ({usage:.1f}%) - consider scaling or optimization")
            elif usage > self.performance_thresholds[f'{resource}_warning']:
                recommendations.append(f"{resource.capitalize()} usage is high ({usage:.1f}%) - monitor closely")
        
        return recommendations
    
    def predict_future_performance(self, days_ahead: int = 7) -> Dict[str, Any]:
        """Predict future performance based on historical trends"""
        historical_trends = self.analyze_trends(30)
        
        predictions = {}
        
        # Simple prediction based on trend analysis
        if 'response_time' in historical_trends:
            rt_current = historical_trends['response_time']['average']
            rt_trend = historical_trends['response_time']['trend']
            
            if rt_trend == 'increasing':
                predicted_rt = rt_current * (1 + (0.05 * days_ahead / 7))
            elif rt_trend == 'decreasing':
                predicted_rt = rt_current * (1 - (0.05 * days_ahead / 7))
            else:
                predicted_rt = rt_current
            
            predictions['response_time'] = {
                'predicted_value': predicted_rt,
                'confidence': 'medium',
                'trend': rt_trend
            }
        
        if 'throughput' in historical_trends:
            tp_current = historical_trends['throughput']['average']
            tp_trend = historical_trends['throughput']['trend']
            
            if tp_trend == 'increasing':
                predicted_tp = tp_current * (1 + (0.1 * days_ahead / 7))
            elif tp_trend == 'decreasing':
                predicted_tp = tp_current * (1 - (0.1 * days_ahead / 7))
            else:
                predicted_tp = tp_current
            
            predictions['throughput'] = {
                'predicted_value': predicted_tp,
                'confidence': 'medium',
                'trend': tp_trend
            }
        
        return predictions
    
    def security_audit(self) -> Dict[str, Any]:
        """Perform security audit and generate security report"""
        audit_report = {
            'audit_timestamp': datetime.now().isoformat(),
            'security_checks': [],
            'findings': [],
            'recommendations': []
        }
        
        # Check for authentication failures
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Query authentication failures
        cursor.execute('''
            SELECT COUNT(*) as auth_failures 
            FROM metrics 
            WHERE metric_name = 'authentication_failures'
            AND timestamp > datetime('now', '-7 days')
        ''')
        auth_failures = cursor.fetchone()[0]
        
        if auth_failures > 10:
            audit_report['findings'].append({
                'type': 'authentication',
                'severity': 'high' if auth_failures > 50 else 'medium',
                'description': f'{auth_failures} authentication failures in last 7 days',
                'recommendation': 'Implement rate limiting and review authentication logs'
            })
        
        # Check for suspicious patterns
        cursor.execute('''
            SELECT metric_name, value, timestamp 
            FROM metrics 
            WHERE metric_name IN ('suspicious_requests', 'failed_verifications')
            AND timestamp > datetime('now', '-24 hours')
        ''')
        suspicious_data = cursor.fetchall()
        
        if len(suspicious_data) > 100:
            audit_report['findings'].append({
                'type': 'behavioral',
                'severity': 'medium',
                'description': f'{len(suspicious_data)} suspicious activities detected in last 24 hours',
                'recommendation': 'Implement behavioral analysis and alerting'
            })
        
        conn.close()
        
        # Add general security recommendations
        audit_report['recommendations'].extend([
            'Implement regular security updates for all components',
            'Enable detailed audit logging for all access',
            'Establish incident response procedures',
            'Regular security assessments and penetration testing'
        ])
        
        return audit_report