"""Test business intelligence functionality"""

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

from enterprise.business_intelligence import BusinessIntelligence, BIEngine, InsightsGenerator

class TestBusinessIntelligence(unittest.TestCase):
    """Test cases for Business Intelligence"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test business data
        self.test_business_data = [
            {
                'timestamp': datetime.now(),
                'event_type': 'face_detected',
                'person_id': 'employee_001',
                'name': 'John Doe',
                'location': 'main_entrance',
                'department': 'engineering',
                'access_level': 'employee',
                'confidence': 0.95,
                'business_impact': 'positive',
                'processing_time': 0.5
            },
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'event_type': 'face_detected',
                'person_id': 'visitor_001',
                'name': 'Jane Smith',
                'location': 'reception',
                'department': 'external',
                'access_level': 'visitor',
                'confidence': 0.88,
                'business_impact': 'neutral',
                'processing_time': 0.7
            },
            {
                'timestamp': datetime.now() - timedelta(hours=4),
                'event_type': 'unauthorized_access',
                'person_id': 'unknown_001',
                'name': 'Unknown',
                'location': 'server_room',
                'department': 'unknown',
                'access_level': 'none',
                'confidence': 0.92,
                'business_impact': 'negative',
                'processing_time': 0.4,
                'security_breach': True
            }
        ]
        
        # Create test performance data
        self.test_performance_data = [
            {
                'timestamp': datetime.now(),
                'department': 'engineering',
                'face_detections': 150,
                'processing_time_avg': 0.45,
                'accuracy_rate': 0.98,
                'compliance_score': 0.95,
                'user_satisfaction': 4.2,
                'cost_per_detection': 0.02
            },
            {
                'timestamp': datetime.now() - timedelta(days=1),
                'department': 'hr',
                'face_detections': 89,
                'processing_time_avg': 0.52,
                'accuracy_rate': 0.96,
                'compliance_score': 0.92,
                'user_satisfaction': 3.8,
                'cost_per_detection': 0.025
            },
            {
                'timestamp': datetime.now() - timedelta(days=2),
                'department': 'finance',
                'face_detections': 45,
                'processing_time_avg': 0.48,
                'accuracy_rate': 0.99,
                'compliance_score': 0.98,
                'user_satisfaction': 4.5,
                'cost_per_detection': 0.018
            }
        ]
        
        # Create test operational metrics
        self.test_operational_metrics = [
            {
                'timestamp': datetime.now(),
                'system_uptime': 99.9,
                'system_availability': 99.8,
                'incident_count': 2,
                'resolution_time_avg': 1.2,
                'downtime_minutes': 5,
                'maintenance_cost': 500,
                'roi': 120
            },
            {
                'timestamp': datetime.now() - timedelta(days=7),
                'system_uptime': 99.5,
                'system_availability': 99.2,
                'incident_count': 5,
                'resolution_time_avg': 2.1,
                'downtime_minutes': 15,
                'maintenance_cost': 750,
                'roi': 115
            }
        ]
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_business_intelligence_initialization(self):
        """Test BusinessIntelligence initialization"""
        bi = BusinessIntelligence()
        
        self.assertIsNotNone(bi)
        self.assertIsNotNone(bi.engine)
        self.assertIsNotNone(bi.insights_generator)
        self.assertIsNotNone(bi.config)
        self.assertIsInstance(bi.cached_insights, dict)
        self.assertEqual(len(bi.cached_insights), 0)
    
    def test_business_intelligence_custom_config(self):
        """Test BusinessIntelligence with custom configuration"""
        config_file = os.path.join(self.temp_dir, 'bi_config.json')
        
        # Save test config
        test_config = {
            'analysis': {
                'enable_predictive_analysis': True,
                'prediction_horizon': 30,  # days
                'confidence_threshold': 0.8,
                'sample_size': 1000
            },
            'insights': {
                'auto_generate': True,
                'schedule': 'daily',
                'priority_levels': ['high', 'medium', 'low'],
                'retention_days': 90
            },
            'reporting': {
                'format': 'pdf',
                'include_charts': True,
                'custom_templates': True
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        
        bi = BusinessIntelligence(config_file=config_file)
        
        self.assertTrue(bi.engine.enable_predictive_analysis)
        self.assertEqual(bi.engine.prediction_horizon, 30)
        self.assertEqual(bi.insights_generator.schedule, 'daily')
        self.assertTrue(bi.insights_generator.include_charts)
    
    def test_bi_engine_initialization(self):
        """Test BIEngine initialization"""
        engine = BIEngine()
        
        self.assertIsNotNone(engine)
        self.assertIsNotNone(engine.data_loader)
        self.assertIsNotNone(engine.analyzer)
        self.assertIsNotNone(engine.predictor)
        self.assertIsNotNone(engine.visualizer)
        self.assertEqual(len(engine.cache), 0)
    
    def test_bi_engine_load_business_data(self):
        """Test business data loading"""
        engine = BIEngine()
        
        # Load test data
        data = engine.load_business_data(self.test_business_data)
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 3)
        
        # Check columns
        expected_columns = ['timestamp', 'event_type', 'person_id', 'name', 
                          'location', 'department', 'access_level', 'confidence',
                          'business_impact', 'processing_time']
        for col in expected_columns:
            self.assertIn(col, data.columns)
    
    def test_bi_engine_load_performance_data(self):
        """Test performance data loading"""
        engine = BIEngine()
        
        # Load test data
        data = engine.load_performance_data(self.test_performance_data)
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 3)
        
        # Check columns
        expected_columns = ['timestamp', 'department', 'face_detections', 
                          'processing_time_avg', 'accuracy_rate', 'compliance_score',
                          'user_satisfaction', 'cost_per_detection']
        for col in expected_columns:
            self.assertIn(col, data.columns)
    
    def test_bi_engine_load_operational_metrics(self):
        """Test operational metrics loading"""
        engine = BIEngine()
        
        # Load test data
        data = engine.load_operational_metrics(self.test_operational_metrics)
        
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 2)
        
        # Check columns
        expected_columns = ['timestamp', 'system_uptime', 'system_availability',
                          'incident_count', 'resolution_time_avg', 'downtime_minutes',
                          'maintenance_cost', 'roi']
        for col in expected_columns:
            self.assertIn(col, data.columns)
    
    def test_bi_engine_analyze_business_impact(self):
        """Test business impact analysis"""
        engine = BIEngine()
        business_data = engine.load_business_data(self.test_business_data)
        
        # Analyze business impact
        impact_analysis = engine.analyze_business_impact(business_data)
        
        self.assertIsNotNone(impact_analysis)
        self.assertIsInstance(impact_analysis, dict)
        self.assertIn('total_events', impact_analysis)
        self.assertIn('impact_distribution', impact_analysis)
        self.assertIn('department_impact', impact_analysis)
        self.assertIn('location_impact', impact_analysis)
        
        # Check impact distribution
        self.assertIn('positive', impact_analysis['impact_distribution'])
        self.assertIn('neutral', impact_analysis['impact_distribution'])
        self.assertIn('negative', impact_analysis['impact_distribution'])
        
        # Check numeric values
        self.assertIsInstance(impact_analysis['total_events'], int)
        self.assertIsInstance(impact_analysis['impact_distribution']['positive'], int)
        self.assertIsInstance(impact_analysis['impact_distribution']['neutral'], int)
        self.assertIsInstance(impact_analysis['impact_distribution']['negative'], int)
    
    def test_bi_engine_analyze_department_performance(self):
        """Test department performance analysis"""
        engine = BIEngine()
        performance_data = engine.load_performance_data(self.test_performance_data)
        
        # Analyze department performance
        perf_analysis = engine.analyze_department_performance(performance_data)
        
        self.assertIsNotNone(perf_analysis)
        self.assertIsInstance(perf_analysis, dict)
        self.assertIn('department_ranking', perf_analysis)
        self.assertIn('performance_metrics', perf_analysis)
        self.assertIn('improvement_areas', perf_analysis)
        
        # Check department ranking
        self.assertIsInstance(perf_analysis['department_ranking'], dict)
        
        # Check performance metrics
        self.assertIn('accuracy_rate', perf_analysis['performance_metrics'])
        self.assertIn('processing_time', perf_analysis['performance_metrics'])
        self.assertIn('user_satisfaction', perf_analysis['performance_metrics'])
        self.assertIn('cost_efficiency', perf_analysis['performance_metrics'])
    
    def test_bi_engine_analyze_compliance(self):
        """Test compliance analysis"""
        engine = BIEngine()
        business_data = engine.load_business_data(self.test_business_data)
        performance_data = engine.load_performance_data(self.test_performance_data)
        
        # Analyze compliance
        compliance_analysis = engine.analyze_compliance(business_data, performance_data)
        
        self.assertIsNotNone(compliance_analysis)
        self.assertIsInstance(compliance_analysis, dict)
        self.assertIn('compliance_score', compliance_analysis)
        self.assertIn('compliance_breakdown', compliance_analysis)
        self.assertIn('violations', compliance_analysis)
        self.assertIn('recommendations', compliance_analysis)
        
        # Check compliance score
        self.assertIsInstance(compliance_analysis['compliance_score'], float)
        self.assertGreaterEqual(compliance_analysis['compliance_score'], 0)
        self.assertLessEqual(compliance_analysis['compliance_score'], 100)
        
        # Check compliance breakdown
        self.assertIsInstance(compliance_analysis['compliance_breakdown'], dict)
        
        # Check violations
        self.assertIsInstance(compliance_analysis['violations'], list)
    
    def test_bi_engine_predict_demand(self):
        """Test demand prediction"""
        engine = BIEngine()
        business_data = engine.load_business_data(self.test_business_data)
        
        # Predict demand
        prediction = engine.predict_demand(business_data, days=7)
        
        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, dict)
        self.assertIn('predictions', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('trend_analysis', prediction)
        
        # Check predictions
        self.assertIsInstance(prediction['predictions'], list)
        
        # Check confidence
        self.assertIsInstance(prediction['confidence'], float)
        self.assertGreaterEqual(prediction['confidence'], 0)
        self.assertLessEqual(prediction['confidence'], 1)
        
        # Check trend analysis
        self.assertIsInstance(prediction['trend_analysis'], dict)
        self.assertIn('direction', prediction['trend_analysis'])
        self.assertIn('strength', prediction['trend_analysis'])
    
    def test_bi_engine_predict_resource_needs(self):
        """Test resource needs prediction"""
        engine = BIEngine()
        business_data = engine.load_business_data(self.test_business_data)
        performance_data = engine.load_performance_data(self.test_performance_data)
        
        # Predict resource needs
        prediction = engine.predict_resource_needs(
            business_data, 
            performance_data, 
            days=7
        )
        
        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, dict)
        self.assertIn('computing_resources', prediction)
        self.assertIn('storage_needs', prediction)
        self.assertIn('bandwidth_requirements', prediction)
        self.assertIn('staffing_requirements', prediction)
        
        # Check computing resources
        self.assertIsInstance(prediction['computing_resources'], dict)
        self.assertIn('cpu', prediction['computing_resources'])
        self.assertIn('memory', prediction['computing_resources'])
        self.assertIn('gpu', prediction['computing_resources'])
        
        # Check storage needs
        self.assertIsInstance(prediction['storage_needs'], dict)
        self.assertIn('face_database', prediction['storage_needs'])
        self.assertIn('logs', prediction['storage_needs'])
        self.assertIn('backups', prediction['storage_needs'])
        
        # Check bandwidth requirements
        self.assertIsInstance(prediction['bandwidth_requirements'], dict)
        self.assertIn('upload', prediction['bandwidth_requirements'])
        self.assertIn('download', prediction['bandwidth_requirements'])
        
        # Check staffing requirements
        self.assertIsInstance(prediction['staffing_requirements'], dict)
        self.assertIn('administrators', prediction['staffing_requirements'])
        self.assertIn('analysts', prediction['staffing_requirements'])
        self.assertIn('support_staff', prediction['staffing_requirements'])
    
    def test_bi_engine_analyze_roi(self):
        """Test ROI analysis"""
        engine = BIEngine()
        business_data = engine.load_business_data(self.test_business_data)
        operational_metrics = engine.load_operational_metrics(self.test_operational_metrics)
        
        # Analyze ROI
        roi_analysis = engine.analyze_roi(business_data, operational_metrics)
        
        self.assertIsNotNone(roi_analysis)
        self.assertIsInstance(roi_analysis, dict)
        self.assertIn('current_roi', roi_analysis)
        self.assertIn('roi_breakdown', roi_analysis)
        self.assertIn('roi_projection', roi_analysis)
        self.assertIn('cost_analysis', roi_analysis)
        
        # Check current ROI
        self.assertIsInstance(roi_analysis['current_roi'], float)
        self.assertGreaterEqual(roi_analysis['current_roi'], 0)
        
        # Check ROI breakdown
        self.assertIsInstance(roi_analysis['roi_breakdown'], dict)
        
        # Check ROI projection
        self.assertIsInstance(roi_analysis['roi_projection'], dict)
        
        # Check cost analysis
        self.assertIsInstance(roi_analysis['cost_analysis'], dict)
        self.assertIn('initial_investment', roi_analysis['cost_analysis'])
        self.assertIn('operational_costs', roi_analysis['cost_analysis'])
        self.assertIn('maintenance_costs', roi_analysis['cost_analysis'])
    
    def test_insights_generator_initialization(self):
        """Test InsightsGenerator initialization"""
        generator = InsightsGenerator()
        
        self.assertIsNotNone(generator)
        self.assertIsNotNone(generator.config)
        self.assertIsNotNone(generator.templates)
        self.assertEqual(len(generator.templates), 0)
    
    def test_insights_generator_generate_insights(self):
        """Test insights generation"""
        generator = InsightsGenerator()
        
        # Mock analysis results
        mock_analysis_results = {
            'business_impact': {
                'total_events': 100,
                'impact_distribution': {'positive': 80, 'neutral': 15, 'negative': 5},
                'department_impact': {'engineering': 0.9, 'hr': 0.7, 'finance': 0.8}
            },
            'performance': {
                'department_ranking': {'finance': 1, 'engineering': 2, 'hr': 3},
                'performance_metrics': {
                    'accuracy_rate': 0.95,
                    'processing_time': 0.5,
                    'user_satisfaction': 4.2
                }
            },
            'compliance': {
                'compliance_score': 0.92,
                'compliance_breakdown': {'access_control': 0.95, 'data_protection': 0.89},
                'violations': []
            },
            'roi': {
                'current_roi': 125.5,
                'roi_breakdown': {'cost_savings': 50, 'revenue_increase': 75.5},
                'cost_analysis': {
                    'initial_investment': 10000,
                    'operational_costs': 2000,
                    'maintenance_costs': 500
                }
            }
        }
        
        # Generate insights
        insights = generator.generate_insights(mock_analysis_results)
        
        self.assertIsNotNone(insights)
        self.assertIsInstance(insights, dict)
        self.assertIn('executive_summary', insights)
        self.assertIn('key_findings', insights)
        self.assertIn('recommendations', insights)
        self.assertIn('action_items', insights)
        self.assertIn('risk_assessment', insights)
        self.assertIn('opportunity_analysis', insights)
        
        # Check executive summary
        self.assertIsInstance(insights['executive_summary'], str)
        self.assertGreater(len(insights['executive_summary']), 0)
        
        # Check key findings
        self.assertIsInstance(insights['key_findings'], list)
        self.assertGreater(len(insights['key_findings']), 0)
        
        # Check recommendations
        self.assertIsInstance(insights['recommendations'], list)
        self.assertGreater(len(insights['recommendations']), 0)
        
        # Check action items
        self.assertIsInstance(insights['action_items'], list)
        self.assertGreater(len(insights['action_items']), 0)
        
        # Check risk assessment
        self.assertIsInstance(insights['risk_assessment'], dict)
        
        # Check opportunity analysis
        self.assertIsInstance(insights['opportunity_analysis'], dict)
    
    def test_insights_generate_priority_insights(self):
        """Test priority insights generation"""
        generator = InsightsGenerator()
        
        # Mock analysis results
        mock_analysis_results = {
            'business_impact': {
                'total_events': 100,
                'impact_distribution': {'positive': 80, 'neutral': 15, 'negative': 5},
                'department_impact': {'engineering': 0.9, 'hr': 0.7, 'finance': 0.8}
            },
            'performance': {
                'department_ranking': {'finance': 1, 'engineering': 2, 'hr': 3},
                'performance_metrics': {
                    'accuracy_rate': 0.95,
                    'processing_time': 0.5,
                    'user_satisfaction': 4.2
                }
            },
            'compliance': {
                'compliance_score': 0.92,
                'compliance_breakdown': {'access_control': 0.95, 'data_protection': 0.89},
                'violations': []
            }
        }
        
        # Generate priority insights
        insights = generator.generate_priority_insights(mock_analysis_results)
        
        self.assertIsNotNone(insights)
        self.assertIsInstance(insights, dict)
        self.assertIn('priority_areas', insights)
        self.assertIn('urgent_actions', insights)
        self.assertIn('strategic_recommendations', insights)
        self.assertIn('resource_allocation', insights)
        
        # Check priority areas
        self.assertIsInstance(insights['priority_areas'], list)
        self.assertGreater(len(insights['priority_areas']), 0)
        
        # Check urgent actions
        self.assertIsInstance(insights['urgent_actions'], list)
        self.assertGreater(len(insights['urgent_actions']), 0)
        
        # Check strategic recommendations
        self.assertIsInstance(insights['strategic_recommendations'], list)
        self.assertGreater(len(insights['strategic_recommendations']), 0)
        
        # Check resource allocation
        self.assertIsInstance(insights['resource_allocation'], dict)
    
    def test_insights_generate_comparative_analysis(self):
        """Test comparative analysis"""
        generator = InsightsGenerator()
        
        # Mock period 1 results
        period1_results = {
            'performance': {
                'accuracy_rate': 0.90,
                'processing_time': 0.6,
                'user_satisfaction': 3.8
            },
            'business_impact': {
                'total_events': 80,
                'impact_distribution': {'positive': 65, 'neutral': 10, 'negative': 5}
            }
        }
        
        # Mock period 2 results
        period2_results = {
            'performance': {
                'accuracy_rate': 0.95,
                'processing_time': 0.5,
                'user_satisfaction': 4.2
            },
            'business_impact': {
                'total_events': 100,
                'impact_distribution': {'positive': 80, 'neutral': 15, 'negative': 5}
            }
        }
        
        # Generate comparative analysis
        comparison = generator.generate_comparative_analysis(period1_results, period2_results)
        
        self.assertIsNotNone(comparison)
        self.assertIsInstance(comparison, dict)
        self.assertIn('performance_improvement', comparison)
        self.assertIn('business_impact_change', comparison)
        self.assertIn('key_differences', comparison)
        self.assertIn('success_factors', comparison)
        self.assertIn('lessons_learned', comparison)
        
        # Check performance improvement
        self.assertIsInstance(comparison['performance_improvement'], dict)
        
        # Check business impact change
        self.assertIsInstance(comparison['business_impact_change'], dict)
        
        # Check key differences
        self.assertIsInstance(comparison['key_differences'], list)
        
        # Check success factors
        self.assertIsInstance(comparison['success_factors'], list)
        
        # Check lessons learned
        self.assertIsInstance(comparison['lessons_learned'], list)
    
    def test_business_intelligence_generate_business_report(self):
        """Test business report generation"""
        bi = BusinessIntelligence()
        
        # Mock engine
        mock_engine = Mock()
        mock_engine.load_business_data.return_value = pd.DataFrame(self.test_business_data)
        mock_engine.load_performance_data.return_value = pd.DataFrame(self.test_performance_data)
        mock_engine.load_operational_metrics.return_value = pd.DataFrame(self.test_operational_metrics)
        mock_engine.analyze_business_impact.return_value = {
            'total_events': 100,
            'impact_distribution': {'positive': 80, 'neutral': 15, 'negative': 5}
        }
        mock_engine.analyze_department_performance.return_value = {
            'department_ranking': {'finance': 1, 'engineering': 2, 'hr': 3}
        }
        mock_engine.analyze_compliance.return_value = {
            'compliance_score': 0.92,
            'compliance_breakdown': {'access_control': 0.95, 'data_protection': 0.89}
        }
        mock_engine.analyze_roi.return_value = {
            'current_roi': 125.5,
            'roi_breakdown': {'cost_savings': 50, 'revenue_increase': 75.5}
        }
        bi.engine = mock_engine
        
        # Generate business report
        report = bi.generate_business_report(
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now()
        )
        
        self.assertIsNotNone(report)
        self.assertIn('executive_summary', report)
        self.assertIn('business_overview', report)
        self.assertIn('performance_analysis', report)
        self.assertIn('compliance_status', report)
        self.assertIn('roi_analysis', report)
        self.assertIn('recommendations', report)
        self.assertIn('appendix', report)
    
    def test_business_intelligence_generate_strategy_report(self):
        """Test strategy report generation"""
        bi = BusinessIntelligence()
        
        # Mock engine and insights generator
        mock_engine = Mock()
        mock_insights = Mock()
        mock_insights.generate_insights.return_value = {
            'executive_summary': 'Test summary',
            'recommendations': ['Recommendation 1', 'Recommendation 2']
        }
        bi.engine = mock_engine
        bi.insights_generator = mock_insights
        
        # Generate strategy report
        report = bi.generate_strategy_report(
            start_time=datetime.now() - timedelta(days=90),
            end_time=datetime.now()
        )
        
        self.assertIsNotNone(report)
        self.assertIn('vision_and_mission', report)
        self.assertIn('strategic_objectives', report)
        self.assertIn('market_analysis', report)
        self.assertIn('competitive_position', report)
        self.assertIn('growth_opportunities', report)
        self.assertIn('implementation_plan', report)
        self.assertIn('risk_management', report)
        self.assertIn('resource_allocation', report)
        self.assertIn('success_metrics', report)
        self.assertIn('conclusion', report)
    
    def test_business_intelligence_generate_forecast_report(self):
        """Test forecast report generation"""
        bi = BusinessIntelligence()
        
        # Mock engine
        mock_engine = Mock()
        mock_engine.load_business_data.return_value = pd.DataFrame(self.test_business_data)
        mock_engine.load_performance_data.return_value = pd.DataFrame(self.test_performance_data)
        mock_engine.predict_demand.return_value = {
            'predictions': [120, 125, 130, 135, 140],
            'confidence': 0.85
        }
        mock_engine.predict_resource_needs.return_value = {
            'computing_resources': {'cpu': 'medium', 'memory': 'high', 'gpu': 'medium'},
            'staffing_requirements': {'administrators': 2, 'analysts': 3, 'support_staff': 1}
        }
        bi.engine = mock_engine
        
        # Generate forecast report
        report = bi.generate_forecast_report(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(days=90)
        )
        
        self.assertIsNotNone(report)
        self.assertIn('forecast_overview', report)
        self.assertIn('demand_forecast', report)
        self.assertIn('resource_planning', report)
        self.assertIn('financial_projections', report)
        self.assertIn('risk_assessment', report)
        self.assertIn('implementation_timeline', report)
        self.assertIn('monitoring_metrics', report)
        self.assertIn('contingency_planning', report)
    
    def test_business_intelligence_generate_competitive_analysis(self):
        """Test competitive analysis"""
        bi = BusinessIntelligence()
        
        # Mock engine
        mock_engine = Mock()
        mock_engine.load_business_data.return_value = pd.DataFrame(self.test_business_data)
        mock_engine.load_performance_data.return_value = pd.DataFrame(self.test_performance_data)
        bi.engine = mock_engine
        
        # Generate competitive analysis
        analysis = bi.generate_competitive_analysis(
            competitors=['competitor_a', 'competitor_b', 'competitor_c']
        )
        
        self.assertIsNotNone(analysis)
        self.assertIn('market_position', analysis)
        self.assertIn('competitive_advantages', analysis)
        self.assertIn('competitive_disadvantages', analysis)
        self.assertIn('benchmarking_results', analysis)
        self.assertIn('market_opportunities', analysis)
        self.assertIn('strategic_recommendations', analysis)
        self.assertIn('action_plan', analysis)
        
        # Check market position
        self.assertIsInstance(analysis['market_position'], dict)
        
        # Check competitive advantages
        self.assertIsInstance(analysis['competitive_advantages'], list)
        
        # Check competitive disadvantages
        self.assertIsInstance(analysis['competitive_disadvantages'], list)
        
        # Check benchmarking results
        self.assertIsInstance(analysis['benchmarking_results'], dict)
    
    def test_business_intelligence_generate_portfolio_analysis(self):
        """Test portfolio analysis"""
        bi = BusinessIntelligence()
        
        # Mock engine
        mock_engine = Mock()
        mock_engine.load_business_data.return_value = pd.DataFrame(self.test_business_data)
        mock_engine.load_performance_data.return_value = pd.DataFrame(self.test_performance_data)
        mock_engine.load_operational_metrics.return_value = pd.DataFrame(self.test_operational_metrics)
        bi.engine = mock_engine
        
        # Generate portfolio analysis
        analysis = bi.generate_portfolio_analysis()
        
        self.assertIsNotNone(analysis)
        self.assertIn('portfolio_overview', analysis)
        self.assertIn('product_matrix', analysis)
        self.assertIn('performance_by_segment', analysis)
        self.assertIn('resource_allocation', analysis)
        self.assertIn('optimization_opportunities', analysis)
        self.assertIn('strategic_recommendations', analysis)
        
        # Check portfolio overview
        self.assertIsInstance(analysis['portfolio_overview'], dict)
        
        # Check product matrix
        self.assertIsInstance(analysis['product_matrix'], dict)
        
        # Check performance by segment
        self.assertIsInstance(analysis['performance_by_segment'], dict)
        
        # Check resource allocation
        self.assertIsInstance(analysis['resource_allocation'], dict)
    
    def test_business_intelligence_generate_customer_insights(self):
        """Test customer insights"""
        bi = BusinessIntelligence()
        
        # Mock engine
        mock_engine = Mock()
        mock_engine.load_business_data.return_value = pd.DataFrame(self.test_business_data)
        mock_engine.load_performance_data.return_value = pd.DataFrame(self.test_performance_data)
        bi.engine = mock_engine
        
        # Generate customer insights
        insights = bi.generate_customer_insights(
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now()
        )
        
        self.assertIsNotNone(insights)
        self.assertIn('customer_demographics', insights)
        self.assertIn('behavior_patterns', insights)
        self.assertIn('satisfaction_analysis', insights)
        self.assertIn('loyalty_metrics', insights)
        self.assertIn('churn_risk', insights)
        self.assertIn('improvement_opportunities', insights)
        self.assertIn('retention_strategies', insights)
        
        # Check customer demographics
        self.assertIsInstance(insights['customer_demographics'], dict)
        
        # Check behavior patterns
        self.assertIsInstance(insights['behavior_patterns'], dict)
        
        # Check satisfaction analysis
        self.assertIsInstance(insights['satisfaction_analysis'], dict)
        
        # Check loyalty metrics
        self.assertIsInstance(insights['loyalty_metrics'], dict)
        
        # Check churn risk
        self.assertIsInstance(insights['churn_risk'], dict)
        
        # Check improvement opportunities
        self.assertIsInstance(insights['improvement_opportunities'], list)
        
        # Check retention strategies
        self.assertIsInstance(insights['retention_strategies'], list)

if __name__ == '__main__':
    unittest.main()