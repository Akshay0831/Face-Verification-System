"""User behavior analytics for enterprise features"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import sqlite3
from pathlib import Path

class UserBehaviorAnalytics:
    """Analyzes user behavior patterns and provides insights"""
    
    def __init__(self, db_path: str = "enterprise/user_behavior.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize database for user behavior tracking"""
        Path("enterprise").mkdir(exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create user_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                session_id TEXT,
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
        
        # Create user_actions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                action_type TEXT,
                timestamp DATETIME,
                details TEXT,
                duration INTEGER,
                success BOOLEAN
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def track_user_session(self, user_id: str, session_id: str, device_type: str = "unknown", location: str = "unknown"):
        """Start tracking a user session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_sessions (user_id, session_id, start_time, device_type, location)
            VALUES (?, ?, datetime('now'), ?, ?)
        ''', (user_id, session_id, device_type, location))
        
        conn.commit()
        conn.close()
    
    def end_user_session(self, session_id: str, verifications_attempted: int = 0, verifications_successful: int = 0, avg_response_time: float = 0.0):
        """End a user session and update statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE user_sessions 
            SET end_time = datetime('now'),
                duration = (julianday(datetime('now')) - julianday(start_time)) * 86400,
                verifications_attempted = ?,
                verifications_successful = ?,
                avg_response_time = ?
            WHERE session_id = ?
        ''', (verifications_attempted, verifications_successful, avg_response_time, session_id))
        
        conn.commit()
        conn.close()
    
    def track_user_action(self, user_id: str, action_type: str, details: str = "", duration: int = 0, success: bool = True):
        """Track a user action for behavior analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_actions (user_id, action_type, timestamp, details, duration, success)
            VALUES (?, ?, datetime('now'), ?, ?, ?)
        ''', (user_id, action_type, details, duration, success))
        
        conn.commit()
        conn.close()
    
    def get_user_insights(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get insights for a specific user"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user sessions
        cursor.execute('''
            SELECT * FROM user_sessions 
            WHERE user_id = ? AND start_time BETWEEN ? AND ?
            ORDER BY start_time DESC
        ''', (user_id, start_date.isoformat(), end_date.isoformat()))
        
        sessions = cursor.fetchall()
        
        # Get user actions
        cursor.execute('''
            SELECT * FROM user_actions 
            WHERE user_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        ''', (user_id, start_date.isoformat(), end_date.isoformat()))
        
        actions = cursor.fetchall()
        
        conn.close()
        
        # Analyze sessions
        session_analysis = self._analyze_sessions(sessions)
        
        # Analyze actions
        action_analysis = self._analyze_actions(actions)
        
        return {
            'user_id': user_id,
            'analysis_period': f'{days} days',
            'session_analysis': session_analysis,
            'action_analysis': action_analysis,
            'behavioral_patterns': self._detect_behavioral_patterns(sessions, actions),
            'recommendations': self._generate_user_recommendations(session_analysis, action_analysis)
        }
    
    def _analyze_sessions(self, sessions: List) -> Dict[str, Any]:
        """Analyze user sessions"""
        if not sessions:
            return {'total_sessions': 0, 'avg_session_duration': 0, 'success_rate': 0}
        
        total_sessions = len(sessions)
        total_duration = sum(s[4] for s in sessions if s[4])  # duration column
        successful_verifications = sum(s[6] for s in sessions)  # verifications_successful
        
        avg_duration = total_duration / total_sessions if total_sessions > 0 else 0
        total_attempts = sum(s[5] for s in sessions)  # verifications_attempted
        success_rate = (successful_verifications / total_attempts * 100) if total_attempts > 0 else 0
        
        # Find most active time periods
        session_hours = [datetime.fromisoformat(s[3]).hour for s in sessions]  # start_time
        hour_distribution = Counter(session_hours)
        
        return {
            'total_sessions': total_sessions,
            'avg_session_duration': avg_duration,
            'total_verifications': total_attempts,
            'successful_verifications': successful_verifications,
            'success_rate': success_rate,
            'peak_usage_hours': hour_distribution.most_common(3),
            'device_distribution': Counter(s[8] for s in sessions)  # device_type
        }
    
    def _analyze_actions(self, actions: List) -> Dict[str, Any]:
        """Analyze user actions"""
        if not actions:
            return {'total_actions': 0, 'success_rate': 0, 'action_distribution': {}}
        
        total_actions = len(actions)
        successful_actions = sum(1 for a in actions if a[6])  # success column
        
        action_types = Counter(a[2] for a in actions)  # action_type
        action_durations = [a[5] for a in actions if a[5]]  # duration
        
        avg_action_duration = sum(action_durations) / len(action_durations) if action_durations else 0
        
        return {
            'total_actions': total_actions,
            'successful_actions': successful_actions,
            'success_rate': (successful_actions / total_actions * 100) if total_actions > 0 else 0,
            'action_distribution': dict(action_types),
            'avg_action_duration': avg_action_duration
        }
    
    def _detect_behavioral_patterns(self, sessions: List, actions: List) -> Dict[str, Any]:
        """Detect behavioral patterns in user activity"""
        patterns = {
            'usage_patterns': [],
            'performance_patterns': [],
            'engagement_patterns': []
        }
        
        if sessions:
            # Detect peak usage times
            session_times = [datetime.fromisoformat(s[3]).hour for s in sessions]  # start_time
            if session_times:
                peak_hours = Counter(session_times).most_common(2)
                patterns['usage_patterns'].append({
                    'pattern_type': 'peak_usage_hours',
                    'hours': [h for h, _ in peak_hours],
                    'description': f'Peak usage at hours {peak_hours[0][0]} and {peak_hours[1][0] if len(peak_hours) > 1 else "same"}'
                })
            
            # Detect session duration patterns
            durations = [s[4] for s in sessions if s[4]]  # duration
            if durations:
                avg_duration = sum(durations) / len(durations)
                if avg_duration < 300:  # 5 minutes
                    patterns['engagement_patterns'].append({
                        'pattern_type': 'short_sessions',
                        'avg_duration': avg_duration,
                        'description': 'User tends to have short sessions, possibly low engagement'
                    })
        
        if actions:
            # Detect action sequence patterns
            action_sequences = []
            for i in range(len(actions) - 1):
                if actions[i][1] == actions[i+1][1]:  # same action_type
                    action_sequences.append(actions[i][2])  # action_type
            
            if len(action_sequences) > 5:
                patterns['performance_patterns'].append({
                    'pattern_type': 'repeated_actions',
                    'frequency': len(action_sequences),
                    'description': 'User frequently repeats actions, possible confusion or difficulty'
                })
        
        return patterns
    
    def _generate_user_recommendations(self, session_analysis: Dict, action_analysis: Dict) -> List[str]:
        """Generate personalized recommendations for users"""
        recommendations = []
        
        # Based on session analysis
        if session_analysis.get('success_rate', 0) < 80:
            recommendations.append('Consider providing additional guidance for verification success')
        
        if session_analysis.get('avg_session_duration', 0) > 1800:  # 30 minutes
            recommendations.append('Sessions are quite long - consider optimizing user flow')
        
        # Based on action analysis
        if action_analysis.get('success_rate', 0) < 90:
            recommendations.append('Some actions are failing - investigate user experience issues')
        
        if action_analysis.get('avg_action_duration', 0) > 10:  # 10 seconds
            recommendations.append('Actions are taking longer than expected - consider performance optimization')
        
        return recommendations
    
    def get_insights(self) -> Dict[str, Any]:
        """Get general insights for all users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get general statistics
        cursor.execute("SELECT COUNT(*) FROM user_sessions")
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_sessions")
        unique_users = cursor.fetchone()[0]
        
        # Get success rate across all sessions
        cursor.execute('''
            SELECT AVG(verifications_successful * 1.0 / verifications_attempted) * 100
            FROM user_sessions
            WHERE verifications_attempted > 0
        ''')
        overall_success_rate = cursor.fetchone()[0] or 0
        
        # Get device usage statistics
        cursor.execute('''
            SELECT device_type, COUNT(*) 
            FROM user_sessions 
            GROUP BY device_type
            ORDER BY COUNT(*) DESC
        ''')
        device_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            'overview': {
                'total_sessions': total_sessions,
                'unique_users': unique_users,
                'overall_success_rate': overall_success_rate,
                'avg_sessions_per_user': total_sessions / unique_users if unique_users > 0 else 0
            },
            'device_distribution': dict(device_stats),
            'period': 'all_time',
            'generated_at': datetime.now().isoformat()
        }