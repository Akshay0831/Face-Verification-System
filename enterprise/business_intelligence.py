"""Business intelligence for face verification system"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import sqlite3
from pathlib import Path
from .analytics.metrics_collector import RealTimeMetricsCollector
from .analytics.performance_analytics import PerformanceAnalytics

class DataPipeline:
    """Data pipeline for business intelligence"""
    
    def __init__(self, db_path: str = "enterprise/bi_data.db"):
        self.db_path = db_path
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize data pipeline"""
        Path("enterprise").mkdir(exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create business_metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                value REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Create business_insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS business_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_type TEXT,
                title TEXT,
                description TEXT,
                confidence_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                action_items TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_business_metrics(self, metrics: Dict[str, Any]):
        """Collect business intelligence metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            cursor.execute('''
                INSERT INTO business_metrics (metric_name, value, timestamp)
                VALUES (?, ?, ?)
            ''', (metric_name, value, timestamp))
        
        conn.commit()
        conn.close()
    
    def get_business_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get business trends over specified period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT metric_name, value, timestamp
            FROM business_metrics
            WHERE timestamp >= ?
            ORDER BY timestamp
        ''', (start_date.isoformat(),))
        
        metrics_data = cursor.fetchall()
        conn.close()
        
        if not metrics_data:
            return {'trends': {}, 'period': f'{days} days', 'generated_at': datetime.now().isoformat()}
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_data, columns=['metric_name', 'value', 'timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate trends for each metric
        trends = {}
        for metric_name in df['metric_name'].unique():
            metric_data = df[df['metric_name'] == metric_name]
            trend_direction = self._calculate_metric_trend(metric_data['value'])
            
            trends[metric_name] = {
                'current_value': metric_data['value'].iloc[-1],
                'average_value': metric_data['value'].mean(),
                'trend_direction': trend_direction,
                'volatility': metric_data['value'].std(),
                'peak_value': metric_data['value'].max(),
                'trough_value': metric_data['value'].min()
            }
        
        return {
            'trends': trends,
            'period': f'{days} days',
            'total_metrics_analyzed': len(trends),
            'generated_at': datetime.now().isoformat()
        }
    
    def _calculate_metric_trend(self, values: pd.Series) -> str:
        """Calculate trend direction for a metric"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

class MLModelPipeline:
    """Machine learning pipeline for business intelligence"""
    
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
    
    def train_behavior_prediction_model(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train user behavior prediction model"""
        # This would integrate with actual ML frameworks
        # For now, implement a simple rule-based predictor
        
        # Simple feature engineering
        features = self._extract_behavior_features(training_data)
        
        # Simple prediction logic
        predictions = self._predict_user_behavior_patterns(features)
        
        return {
            'model_type': 'behavior_prediction',
            'training_samples': len(training_data),
            'features_used': list(features.keys()),
            'predictions': predictions,
            'confidence_score': self._calculate_prediction_confidence(predictions)
        }
    
    def _extract_behavior_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract behavior features from user data"""
        features = {}
        
        # Time-based features
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            features['peak_usage_hour'] = data['hour'].mode()[0] if len(data['hour']) > 0 else 12
            features['usage_variance'] = data['hour'].var()
        
        # Activity features
        if 'action_count' in data.columns:
            features['avg_activity'] = data['action_count'].mean()
            features['activity_peak'] = data['action_count'].max()
            features['activity_trend'] = self._calculate_trend(data['action_count'])
        
        # Success rate features
        if 'success_rate' in data.columns:
            features['avg_success_rate'] = data['success_rate'].mean()
            features['success_consistency'] = 1 - (data['success_rate'].std() / 100)
        
        return features
    
    def _predict_user_behavior_patterns(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict user behavior patterns"""
        predictions = {}
        
        # Predict peak usage
        peak_hour = features.get('peak_usage_hour', 12)
        predictions['predicted_peak_usage'] = {
            'hour': peak_hour,
            'confidence': 0.8 if features.get('usage_variance', 0) < 4 else 0.6
        }
        
        # Predict activity levels
        avg_activity = features.get('avg_activity', 5)
        activity_trend = features.get('activity_trend', 'stable')
        
        if activity_trend == 'increasing':
            predictions['activity_forecast'] = 'increasing'
            predictions['recommended_action'] = 'increase_capacity'
        elif activity_trend == 'decreasing':
            predictions['activity_forecast'] = 'decreasing'
            predictions['recommended_action'] = 'optimize_resources'
        else:
            predictions['activity_forecast'] = 'stable'
            predictions['recommended_action'] = 'maintain_current'
        
        # Predict success patterns
        success_rate = features.get('avg_success_rate', 90)
        if success_rate > 95:
            predictions['success_prediction'] = 'high'
        elif success_rate > 85:
            predictions['success_prediction'] = 'medium'
        else:
            predictions['success_prediction'] = 'low'
        
        return predictions
    
    def _calculate_prediction_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall confidence score for predictions"""
        confidence_factors = []
        
        # Add confidence from each prediction
        for key, value in predictions.items():
            if isinstance(value, dict) and 'confidence' in value:
                confidence_factors.append(value['confidence'])
            else:
                confidence_factors.append(0.7)  # Default confidence
        
        return np.mean(confidence_factors) if confidence_factors else 0.5

class BusinessIntelligence:
    """Business intelligence for face verification system"""
    
    def __init__(self, db_path: str = "enterprise/bi_data.db"):
        self.data_pipeline = DataPipeline(db_path)
        self.ml_pipeline = MLModelPipeline()
        self.metrics_collector = RealTimeMetricsCollector(db_path)
        self.performance_analytics = PerformanceAnalytics(db_path)
    
    def predict_user_behavior(self) -> Dict[str, Any]:
        """Predict user behavior patterns"""
        # Collect recent user data
        recent_data = self._collect_recent_user_data()
        
        if recent_data.empty:
            return {
                'status': 'insufficient_data',
                'message': 'Not enough recent data for behavior prediction',
                'predictions': {}
            }
        
        # Train prediction model
        model_result = self.ml_pipeline.train_behavior_prediction_model(recent_data)
        
        # Generate business insights
        insights = self._generate_business_insights(recent_data, model_result)
        
        return {
            'predictions': model_result,
            'business_insights': insights,
            'recommendations': self._generate_bi_recommendations(insights),
            'timestamp': datetime.now().isoformat()
        }
    
    def _collect_recent_user_data(self) -> pd.DataFrame:
        """Collect recent user data for analysis"""
        # Mock data collection - in real implementation, this would query actual databases
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        data = {
            'date': dates,
            'user_count': np.random.randint(50, 200, size=30),
            'verification_count': np.random.randint(100, 500, size=30),
            'success_rate': np.random.uniform(85, 99, size=30),
            'avg_response_time': np.random.uniform(0.5, 2.5, size=30),
            'peak_hour': np.random.randint(9, 18, size=30)
        }
        
        return pd.DataFrame(data)
    
    def _generate_business_insights(self, data: pd.DataFrame, model_result: Dict) -> Dict[str, Any]:
        """Generate business insights from data and model predictions"""
        insights = {
            'user_growth': self._analyze_user_growth(data),
            'engagement_trends': self._analyze_engagement(data),
            'performance_insights': self._analyze_performance_insights(data),
            'market_opportunities': self._identify_market_opportunities(data, model_result)
        }
        
        return insights
    
    def _analyze_user_growth(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user growth patterns"""
        user_counts = data['user_count']
        
        # Calculate growth rate
        growth_rate = (user_counts.iloc[-1] - user_counts.iloc[0]) / user_counts.iloc[0] * 100
        
        # Determine growth pattern
        if growth_rate > 20:
            growth_pattern = 'rapid_growth'
        elif growth_rate > 5:
            growth_pattern = 'steady_growth'
        elif growth_rate > -5:
            growth_pattern = 'stable'
        else:
            growth_pattern = 'declining'
        
        return {
            'growth_rate': growth_rate,
            'growth_pattern': growth_pattern,
            'peak_users': user_counts.max(),
            'average_users': user_counts.mean(),
            'trend_direction': self._calculate_trend(user_counts)
        }
    
    def _analyze_engagement(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user engagement patterns"""
        verification_counts = data['verification_count']
        success_rates = data['success_rate']
        
        # Calculate engagement metrics
        avg_engagement = verification_counts.mean()
        engagement_trend = self._calculate_trend(verification_counts)
        
        # Calculate engagement quality
        quality_score = (success_rates.mean() / 100) * avg_engagement
        
        return {
            'average_engagement': avg_engagement,
            'engagement_trend': engagement_trend,
            'engagement_quality': quality_score,
            'peak_engagement': verification_counts.max(),
            'consistency': 1 - (verification_counts.std() / avg_engagement) if avg_engagement > 0 else 0
        }
    
    def _analyze_performance_insights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze system performance insights"""
        response_times = data['avg_response_time']
        success_rates = data['success_rate']
        
        # Performance metrics
        avg_response_time = response_times.mean()
        response_trend = self._calculate_trend(response_times)
        
        # Performance quality
        performance_score = ((success_rates.mean() / 100) * (1 - avg_response_time / 10)) * 100
        
        return {
            'average_response_time': avg_response_time,
            'response_time_trend': response_trend,
            'performance_score': max(0, performance_score),
            'reliability': success_rates.mean(),
            'consistency': 1 - (response_times.std() / avg_response_time) if avg_response_time > 0 else 0
        }
    
    def _identify_market_opportunities(self, data: pd.DataFrame, model_result: Dict) -> List[Dict[str, Any]]:
        """Identify business opportunities"""
        opportunities = []
        
        # Growth opportunities
        if data['user_count'].iloc[-1] > data['user_count'].mean() * 1.1:
            opportunities.append({
                'type': 'growth_opportunity',
                'description': 'User growth accelerating - consider capacity expansion',
                'priority': 'high',
                'estimated_impact': 'medium'
            })
        
        # Performance opportunities
        if data['success_rate'].mean() < 90:
            opportunities.append({
                'type': 'performance_improvement',
                'description': 'Success rate below target - UX optimization needed',
                'priority': 'high',
                'estimated_impact': 'high'
            })
        
        # Efficiency opportunities
        if data['avg_response_time'].mean() > 2.0:
            opportunities.append({
                'type': 'efficiency_gain',
                'description': 'Response times above optimal - performance optimization needed',
                'priority': 'medium',
                'estimated_impact': 'medium'
            })
        
        # Market expansion opportunities
        peak_hours = data['peak_hour'].value_counts()
        if peak_hours.iloc[0] > len(data) * 0.4:  # 40%+ usage in peak hour
            opportunities.append({
                'type': 'market_expansion',
                'description': 'Concentrated usage patterns - consider market expansion in off-peak hours',
                'priority': 'low',
                'estimated_impact': 'low'
            })
        
        return opportunities
    
    def _generate_bi_recommendations(self, insights: Dict) -> List[str]:
        """Generate business intelligence recommendations"""
        recommendations = []
        
        # User growth recommendations
        if insights['user_growth']['growth_pattern'] == 'rapid_growth':
            recommendations.append('Implement scaling infrastructure to support rapid user growth')
        elif insights['user_growth']['growth_pattern'] == 'declining':
            recommendations.append('Investigate user retention strategies and improve user experience')
        
        # Engagement recommendations
        engagement = insights['engagement_trends']
        if engagement['engagement_trend'] == 'decreasing':
            recommendations.append('User engagement is declining - implement engagement optimization strategies')
        elif engagement['engagement_quality'] < 50:
            recommendations.append('Low engagement quality - focus on improving success rates and response times')
        
        # Performance recommendations
        performance = insights['performance_insights']
        if performance['performance_score'] < 70:
            recommendations.append('System performance needs improvement - optimize algorithms and infrastructure')
        
        # Opportunity-based recommendations
        opportunities = insights['market_opportunities']
        for opportunity in opportunities:
            if opportunity['priority'] == 'high':
                recommendations.append(opportunity['description'])
        
        return recommendations
    
    def optimize_resources(self) -> Dict[str, Any]:
        """Optimize resource allocation based on usage patterns"""
        # Analyze current resource usage
        resource_analysis = self._analyze_resource_usage()
        
        # Generate optimization recommendations
        optimization_plan = self._generate_optimization_plan(resource_analysis)
        
        # Calculate potential cost savings
        cost_savings = self._calculate_potential_savings(optimization_plan)
        
        return {
            'current_usage': resource_analysis,
            'optimization_plan': optimization_plan,
            'potential_savings': cost_savings,
            'implementation_timeline': self._calculate_implementation_timeline(optimization_plan),
            'risk_assessment': self._assess_optimization_risks(optimization_plan),
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze current resource usage patterns"""
        # Mock resource analysis - in real implementation, this would query actual monitoring systems
        return {
            'compute_usage': {
                'average': 65,
                'peak': 85,
                'variance': 15,
                'efficiency': 0.7
            },
            'memory_usage': {
                'average': 45,
                'peak': 70,
                'variance': 10,
                'efficiency': 0.8
            },
            'network_usage': {
                'average': 30,
                'peak': 60,
                'variance': 20,
                'efficiency': 0.6
            },
            'storage_usage': {
                'average': 55,
                'peak': 75,
                'variance': 12,
                'efficiency': 0.75
            }
        }
    
    def _generate_optimization_plan(self, resource_analysis: Dict) -> Dict[str, Any]:
        """Generate resource optimization plan"""
        plan = {
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': [],
            'estimated_improvements': {}
        }
        
        resources = resource_analysis
        
        # Analyze each resource
        for resource, usage in resources.items():
            avg_usage = usage['average']
            peak_usage = usage['peak']
            efficiency = usage['efficiency']
            
            if peak_usage > 90:  # Critical usage
                plan['immediate_actions'].append(f'Scale up {resource} capacity immediately')
                plan['estimated_improvements'][resource] = {'performance': 20, 'cost': -15}
            elif avg_usage > 70:  # High usage
                plan['short_term_actions'].append(f'Optimize {resource} allocation for current load')
                plan['estimated_improvements'][resource] = {'performance': 15, 'cost': -10}
            elif efficiency < 0.7:  # Low efficiency
                plan['short_term_actions'].append(f'Restructure {resource} allocation for better efficiency')
                plan['estimated_improvements'][resource] = {'performance': 10, 'cost': -5}
        
        return plan
    
    def _calculate_potential_savings(self, optimization_plan: Dict) -> Dict[str, Any]:
        """Calculate potential cost savings from optimization"""
        total_savings = 0
        resource_savings = {}
        
        for resource, improvements in optimization_plan['estimated_improvements'].items():
            cost_saving = improvements.get('cost', 0)
            resource_savings[resource] = cost_saving
            total_savings += cost_saving
        
        return {
            'total_percentage_savings': max(0, total_savings),
            'resource_breakdown': resource_savings,
            'payback_period': '3-6 months' if total_savings > 0 else 'N/A',
            'roi_estimate': f'{total_savings * 2}%' if total_savings > 0 else 'N/A'
        }
    
    def _calculate_trend(self, values: pd.Series) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        slope, _ = np.polyfit(range(len(values)), values, 1)
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_implementation_timeline(self, optimization_plan: Dict) -> Dict[str, Any]:
        """Calculate implementation timeline for optimization plan"""
        timeline = {
            'immediate': 0,  # days
            'short_term': 7,  # days
            'long_term': 30   # days
        }
        
        return timeline
    
    def _assess_optimization_risks(self, optimization_plan: Dict) -> Dict[str, Any]:
        """Assess risks associated with optimization plan"""
        return {
            'overall_risk_level': 'medium',
            'risk_factors': [
                'Potential service disruption during scaling',
                'Need for careful testing of optimized configurations',
                'Possible need for staff training on new systems'
            ],
            'mitigation_strategies': [
                'Implement phased rollout with rollback capability',
                'Conduct thorough testing in staging environment',
                'Prepare comprehensive documentation and training materials'
            ]
        }