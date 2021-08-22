"""Enterprise analytics dashboard for face verification system"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import sqlite3
from pathlib import Path
from .analytics.metrics_collector import RealTimeMetricsCollector
from .analytics.performance_analytics import PerformanceAnalytics
from .analytics.user_behavior import UserBehaviorAnalytics

class AnalyticsDashboard:
    """Enterprise analytics dashboard with real-time metrics"""
    
    def __init__(self, db_path: str = "enterprise/analytics.db"):
        self.db_path = db_path
        self.metrics_collector = RealTimeMetricsCollector(db_path)
        self.performance_analytics = PerformanceAnalytics(db_path)
        self.user_behavior_analytics = UserBehaviorAnalytics(db_path)
        
        # Initialize tables if they don't exist
        self._initialize_dashboard_db()
    
    def _initialize_dashboard_db(self):
        """Initialize dashboard database for storing dashboard-specific data"""
        Path("enterprise").mkdir(exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create dashboard_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dashboard_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dashboard_name TEXT,
                data_type TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create user_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                user_id TEXT,
                start_time DATETIME,
                end_time DATETIME,
                duration INTEGER,
                verifications_attempted INTEGER,
                verifications_successful INTEGER,
                avg_response_time REAL,
                device_type TEXT,
                location TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_comprehensive_reports(self) -> Dict[str, Any]:
        """Generate comprehensive usage, performance, and security reports"""
        return {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'comprehensive',
                'period': 'last_30_days'
            },
            'usage_statistics': self._generate_usage_statistics(),
            'performance_analysis': self.performance_analytics.generate_performance_report(),
            'security_compliance': self.performance_analytics.security_audit(),
            'user_insights': self.user_behavior_analytics.get_insights(),
            'system_health': self.metrics_collector.get_health_status(),
            'recommendations': self._generate_dashboard_recommendations()
        }
    
    def _generate_usage_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive usage statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total verifications
        cursor.execute('SELECT SUM(verifications_attempted) FROM user_sessions')
        total_verifications = cursor.fetchone()[0] or 0
        
        # Get successful verifications
        cursor.execute('SELECT SUM(verifications_successful) FROM user_sessions')
        successful_verifications = cursor.fetchone()[0] or 0
        
        # Get average response time
        cursor.execute('SELECT AVG(avg_response_time) FROM user_sessions WHERE avg_response_time > 0')
        avg_response_time = cursor.fetchone()[0] or 0
        
        # Get user session statistics
        cursor.execute('SELECT COUNT(*) FROM user_sessions')
        total_sessions = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM user_sessions')
        unique_users = cursor.fetchone()[0] or 0
        
        # Get daily breakdown
        cursor.execute('''
            SELECT DATE(start_time) as date, COUNT(*) as sessions
            FROM user_sessions 
            WHERE start_time >= datetime('now', '-30 days')
            GROUP BY DATE(start_time)
            ORDER BY date
        ''')
        daily_sessions = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_verifications': total_verifications,
            'successful_verifications': successful_verifications,
            'success_rate': (successful_verifications / total_verifications * 100) if total_verifications > 0 else 0,
            'avg_response_time': avg_response_time,
            'total_sessions': total_sessions,
            'unique_users': unique_users,
            'avg_sessions_per_user': total_sessions / unique_users if unique_users > 0 else 0,
            'daily_sessions': daily_sessions,
            'conversion_rate': (unique_users / total_sessions * 100) if total_sessions > 0 else 0
        }
    
    def monitor_system_health(self) -> Dict[str, Any]:
        """Monitor system health with predictive analytics"""
        health_status = self.metrics_collector.get_health_status()
        
        # Get current metrics
        current_metrics = self.metrics_collector.get_current_metrics()
        
        # Get performance predictions
        performance_predictions = self.performance_analytics.predict_future_performance(7)
        
        # Calculate risk assessment
        risk_assessment = self._calculate_risk_assessment(health_status, current_metrics, performance_predictions)
        
        return {
            'current_status': health_status,
            'metrics_summary': current_metrics,
            'performance_predictions': performance_predictions,
            'risk_assessment': risk_assessment,
            'recommended_actions': self._get_health_recommendations(health_status, risk_assessment),
            'monitoring_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_risk_assessment(self, health_status: Dict, current_metrics: Dict, predictions: Dict) -> Dict[str, Any]:
        """Calculate overall system risk assessment"""
        risk_score = 0
        risk_factors = []
        
        # Health score factor
        health_score = health_status.get('health_score', 100)
        risk_score += max(0, (100 - health_score) * 0.3)
        if health_score < 70:
            risk_factors.append({
                'factor': 'low_health_score',
                'severity': 'high' if health_score < 50 else 'medium',
                'description': f'System health score is {health_score}'
            })
        
        # Resource utilization factors
        resources = health_status.get('resource_utilization', {})
        for resource, usage in resources.items():
            if usage > 95:
                risk_score += 20
                risk_factors.append({
                    'factor': f'{resource}_critical_usage',
                    'severity': 'high',
                    'description': f'{resource} usage at {usage:.1f}% (critical)'
                })
            elif usage > 80:
                risk_score += 10
                risk_factors.append({
                    'factor': f'{resource}_high_usage',
                    'severity': 'medium',
                    'description': f'{resource} usage at {usage:.1f}% (high)'
                })
        
        # Prediction-based risk
        if 'response_time' in predictions:
            predicted_rt = predictions['response_time'].get('predicted_value', 0)
            if predicted_rt > 5.0:  # Critical threshold
                risk_score += 15
                risk_factors.append({
                    'factor': 'predicted_response_time_increase',
                    'severity': 'high',
                    'description': f'Predicted response time: {predicted_rt:.2f}s'
                })
        
        # Error rate factor
        error_rate = current_metrics.get('counters', {}).get('error_rate', 0)
        if error_rate > 10:
            risk_score += error_rate * 2
            risk_factors.append({
                'factor': 'high_error_rate',
                'severity': 'high' if error_rate > 20 else 'medium',
                'description': f'Current error rate: {error_rate}%'
            })
        
        # Determine overall risk level
        if risk_score >= 50:
            risk_level = 'high'
        elif risk_score >= 25:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'overall_risk_score': min(100, risk_score),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'affected_components': self._identify_affected_components(risk_factors)
        }
    
    def _identify_affected_components(self, risk_factors: List[Dict]) -> List[str]:
        """Identify system components affected by risk"""
        components = set()
        
        for factor in risk_factors:
            if 'resource' in factor['factor']:
                components.add(factor['factor'].split('_')[0])
            elif 'response_time' in factor['factor']:
                components.add('performance')
            elif 'error' in factor['factor']:
                components.add('verification')
            elif 'health' in factor['factor']:
                components.add('system')
        
        return list(components)
    
    def _get_health_recommendations(self, health_status: Dict, risk_assessment: Dict) -> List[str]:
        """Generate health monitoring recommendations"""
        recommendations = []
        
        # Health-based recommendations
        health_score = health_status.get('health_score', 100)
        if health_score < 80:
            recommendations.append('System health is declining - investigate underlying issues')
        
        # Resource-based recommendations
        resources = health_status.get('resource_utilization', {})
        for resource, usage in resources.items():
            if usage > 95:
                recommendations.append(f'Immediate action needed: {resource} usage is critical ({usage:.1f}%)')
            elif usage > 80:
                recommendations.append(f'Monitor {resource} usage closely ({usage:.1f}%)')
        
        # Risk-based recommendations
        risk_level = risk_assessment.get('risk_level', 'low')
        if risk_level == 'high':
            recommendations.append('High risk detected - implement immediate remediation')
        elif risk_level == 'medium':
            recommendations.append('Medium risk detected - implement monitoring and mitigation')
        
        return recommendations
    
    def audit_security_compliance(self) -> Dict[str, Any]:
        """Audit security compliance and generate compliance reports"""
        security_audit = self.performance_analytics.security_audit()
        
        # Add compliance scoring
        compliance_score = self._calculate_compliance_score(security_audit)
        
        # Add compliance recommendations
        compliance_recommendations = self._generate_compliance_recommendations(security_audit, compliance_score)
        
        return {
            'security_audit': security_audit,
            'compliance_score': compliance_score,
            'compliance_grade': self._get_compliance_grade(compliance_score),
            'compliance_recommendations': compliance_recommendations,
            'compliance_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_compliance_score(self, security_audit: Dict) -> float:
        """Calculate overall compliance score"""
        if not security_audit.get('findings'):
            return 100.0
        
        deductions = 0
        for finding in security_audit['findings']:
            if finding['severity'] == 'high':
                deductions += 20
            elif finding['severity'] == 'medium':
                deductions += 10
            elif finding['severity'] == 'low':
                deductions += 5
        
        return max(0, 100 - deductions)
    
    def _get_compliance_grade(self, score: float) -> str:
        """Get compliance grade based on score"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_compliance_recommendations(self, security_audit: Dict, compliance_score: float) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Base recommendations based on score
        if compliance_score < 80:
            recommendations.append('Immediate security improvements required')
        elif compliance_score < 90:
            recommendations.append('Security enhancements recommended')
        
        # Add specific recommendations from audit findings
        for finding in security_audit.get('findings', []):
            recommendations.append(finding['recommendation'])
        
        # Add general compliance recommendations
        recommendations.extend([
            'Implement regular security audits',
            'Maintain incident response procedures',
            'Ensure regular security training for staff',
            'Keep all systems patched and updated'
        ])
        
        return recommendations
    
    def _generate_dashboard_recommendations(self) -> List[str]:
        """Generate dashboard-specific recommendations"""
        recommendations = []
        
        # Performance recommendations
        performance_report = self.performance_analytics.generate_performance_report()
        if 'recommendations' in performance_report:
            recommendations.extend(performance_report['recommendations'])
        
        # User experience recommendations
        user_insights = self.user_behavior_analytics.get_insights()
        if user_insights.get('overview', {}).get('overall_success_rate', 0) < 90:
            recommendations.append('User success rate is below target - investigate UX issues')
        
        # System health recommendations
        health_status = self.metrics_collector.get_health_status()
        if health_status.get('health_score', 100) < 90:
            recommendations.append('System health needs attention - review resource allocation')
        
        return recommendations
    
    def export_dashboard_data(self, export_format: str = 'json', days: int = 30) -> str:
        """Export dashboard data in specified format"""
        comprehensive_report = self.generate_comprehensive_reports()
        
        if export_format.lower() == 'json':
            return json.dumps(comprehensive_report, indent=2, default=str)
        elif export_format.lower() == 'csv':
            return self._export_to_csv(comprehensive_report, days)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def _export_to_csv(self, data: Dict, days: int) -> str:
        """Export dashboard data to CSV format"""
        import io
        
        output = io.StringIO()
        
        # Write header
        output.write("Metric,Value,Timestamp\n")
        
        # Write basic metrics
        timestamp = datetime.now().isoformat()
        
        # Usage statistics
        usage_stats = data.get('usage_statistics', {})
        output.write(f"Total Verifications,{usage_stats.get('total_verifications', 0)},{timestamp}\n")
        output.write(f"Success Rate,{usage_stats.get('success_rate', 0):.2f}%,{timestamp}\n")
        output.write(f"Average Response Time,{usage_stats.get('avg_response_time', 0):.2f}s,{timestamp}\n")
        output.write(f"Unique Users,{usage_stats.get('unique_users', 0)},{timestamp}\n")
        
        # Health score
        health = data.get('system_health', {})
        output.write(f"Health Score,{health.get('health_score', 0)},{timestamp}\n")
        
        # Performance predictions
        predictions = data.get('performance_predictions', {})
        if 'response_time' in predictions:
            rt_pred = predictions['response_time']
            output.write(f"Predicted Response Time,{rt_pred.get('predicted_value', 0):.2f}s,{timestamp}\n")
        
        output.seek(0)
        return output.read()