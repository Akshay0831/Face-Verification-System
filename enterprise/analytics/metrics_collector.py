"""Real-time metrics collection for enterprise analytics"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict, deque
import json
import sqlite3
from pathlib import Path

class RealTimeMetricsCollector:
    """Collects and manages real-time system metrics"""
    
    def __init__(self, db_path: str = "enterprise/metrics.db"):
        self.db_path = db_path
        self.metrics_cache = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
        self.counters = defaultdict(int)
        self.timers = {}
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        Path("enterprise").mkdir(exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                value REAL,
                metadata TEXT
            )
        ''')
        
        # Create daily_stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE,
                total_verifications INTEGER DEFAULT 0,
                successful_verifications INTEGER DEFAULT 0,
                failed_verifications INTEGER DEFAULT 0,
                avg_response_time REAL DEFAULT 0,
                peak_hourly_requests INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def increment_counter(self, metric_name: str, count: int = 1):
        """Increment a counter metric"""
        with self.lock:
            self.counters[metric_name] += count
            self.metrics_cache[metric_name].append({
                'timestamp': datetime.now(),
                'value': self.counters[metric_name],
                'type': 'counter'
            })
    
    def record_timing(self, metric_name: str, duration: float):
        """Record a timing metric"""
        with self.lock:
            if metric_name not in self.timers:
                self.timers[metric_name] = []
            self.timers[metric_name].append(duration)
            self.metrics_cache[metric_name].append({
                'timestamp': datetime.now(),
                'value': duration,
                'type': 'timing'
            })
    
    def record_gauge(self, metric_name: str, value: float):
        """Record a gauge metric (current value)"""
        with self.lock:
            self.metrics_cache[metric_name].append({
                'timestamp': datetime.now(),
                'value': value,
                'type': 'gauge'
            })
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        with self.lock:
            return {
                'counters': dict(self.counters),
                'timers_stats': self._calculate_timer_stats(),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_timer_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for timing metrics"""
        stats = {}
        for timer_name, times in self.timers.items():
            if times:
                stats[timer_name] = {
                    'count': len(times),
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'p95': sorted(times)[int(len(times) * 0.95)]
                }
        return stats
    
    def collect_daily_stats(self) -> Dict[str, Any]:
        """Collect daily statistics for reports"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get today's date
        today = datetime.now().date().isoformat()
        
        # Check if today's stats exist
        cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (today,))
        row = cursor.fetchone()
        
        if row:
            return {
                'date': today,
                'total_verifications': row[2],
                'successful_verifications': row[3],
                'failed_verifications': row[4],
                'avg_response_time': row[5],
                'peak_hourly_requests': row[6]
            }
        else:
            return {
                'date': today,
                'total_verifications': self.counters.get('total_verifications', 0),
                'successful_verifications': self.counters.get('successful_verifications', 0),
                'failed_verifications': self.counters.get('failed_verifications', 0),
                'avg_response_time': self._calculate_timer_stats().get('face_verification', {}).get('avg', 0),
                'peak_hourly_requests': self._get_peak_hourly_requests()
            }
    
    def _get_peak_hourly_requests(self) -> int:
        """Calculate peak hourly requests"""
        # This would analyze historical data or current hour data
        # For now, return current counter as approximation
        return self.counters.get('hourly_requests', 0)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        current_time = datetime.now()
        
        # Check system health indicators
        health_score = 100  # Start with perfect score
        
        # Check error rate
        total_verifications = self.counters.get('total_verifications', 0)
        failed_verifications = self.counters.get('failed_verifications', 0)
        
        if total_verifications > 0:
            error_rate = (failed_verifications / total_verifications) * 100
            health_score -= min(error_rate * 2, 50)  # Deduct up to 50 points for errors
        
        # Check response times
        timing_stats = self._calculate_timer_stats()
        if 'face_verification' in timing_stats:
            avg_response = timing_stats['face_verification']['avg']
            if avg_response > 2.0:  # 2 seconds threshold
                health_score -= min((avg_response - 2.0) * 10, 30)
        
        # Resource utilization (mock data for now)
        resource_utilization = {
            'cpu': 45,  # 45% CPU usage
            'memory': 60,  # 60% memory usage
            'disk': 30,   # 30% disk usage
            'network': 25 # 25% network usage
        }
        
        # Check for any threshold breaches
        thresholds_met = []
        for resource, usage in resource_utilization.items():
            if usage > 80:
                thresholds_met.append(f"{resource}_high_usage")
        
        return {
            'health_score': max(0, health_score),
            'resource_utilization': resource_utilization,
            'thresholds_met': thresholds_met,
            'last_updated': current_time.isoformat()
        }
    
    def save_metrics_to_db(self, metric_name: str, value: float, metadata: str = None):
        """Save metrics to database for persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metrics (metric_name, value, metadata)
            VALUES (?, ?, ?)
        ''', (metric_name, value, metadata))
        
        conn.commit()
        conn.close()