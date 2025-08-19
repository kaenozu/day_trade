#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Monitor - パフォーマンス監視システム
Issue #933 対応: パフォーマンス監視強化
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import json

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


class PerformanceMonitor:
    """リアルタイムパフォーマンス監視システム"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.analysis_times = deque(maxlen=max_history_size)
        self.api_response_times = defaultdict(lambda: deque(maxlen=100))
        self.memory_usage_history = deque(maxlen=100)
        self.error_counts = defaultdict(int)
        
        # 統計データ
        self.total_analyses = 0
        self.total_api_calls = 0
        self.start_time = datetime.now()
        
        # ロック
        self._lock = threading.Lock()
        
        # 自動監視スレッド
        self._monitoring_thread = None
        self._stop_monitoring = False
        self.start_monitoring()
    
    def track_analysis_time(self, symbol: str, duration: float, analysis_type: str = "unknown"):
        """分析処理時間を記録"""
        with self._lock:
            self.analysis_times.append({
                'symbol': symbol,
                'duration': duration,
                'analysis_type': analysis_type,
                'timestamp': datetime.now()
            })
            self.total_analyses += 1
    
    def track_api_response_time(self, endpoint: str, duration: float, status_code: int = 200):
        """API応答時間を記録"""
        with self._lock:
            self.api_response_times[endpoint].append({
                'duration': duration,
                'status_code': status_code,
                'timestamp': datetime.now()
            })
            self.total_api_calls += 1
    
    def track_error(self, error_type: str, details: str = ""):
        """エラー発生を記録"""
        with self._lock:
            self.error_counts[error_type] += 1
    
    def get_memory_usage(self) -> Optional[Dict[str, float]]:
        """現在のメモリ使用量を取得"""
        if not HAS_PSUTIL:
            return None
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # MB
                'vms_mb': memory_info.vms / 1024 / 1024,  # MB
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent()
            }
        except Exception:
            return None
    
    def start_monitoring(self):
        """バックグラウンドでの監視を開始"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring = False
            self._monitoring_thread = threading.Thread(target=self._monitor_system_resources)
            self._monitoring_thread.daemon = True
            self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """監視を停止"""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1)
    
    def _monitor_system_resources(self):
        """システムリソースを定期的に監視"""
        while not self._stop_monitoring:
            memory_usage = self.get_memory_usage()
            if memory_usage:
                with self._lock:
                    self.memory_usage_history.append({
                        'timestamp': datetime.now(),
                        **memory_usage
                    })
            
            time.sleep(30)  # 30秒間隔で監視
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリーを取得"""
        with self._lock:
            current_memory = self.get_memory_usage()
            uptime = datetime.now() - self.start_time
            
            # 分析時間統計
            analysis_durations = [a['duration'] for a in self.analysis_times]
            avg_analysis_time = sum(analysis_durations) / len(analysis_durations) if analysis_durations else 0
            
            # API応答時間統計
            all_api_times = []
            for endpoint_times in self.api_response_times.values():
                all_api_times.extend([r['duration'] for r in endpoint_times])
            
            avg_api_response = sum(all_api_times) / len(all_api_times) if all_api_times else 0
            
            return {
                'uptime_seconds': uptime.total_seconds(),
                'uptime_formatted': str(uptime),
                'total_analyses': self.total_analyses,
                'total_api_calls': self.total_api_calls,
                'avg_analysis_time_ms': avg_analysis_time * 1000,
                'avg_api_response_ms': avg_api_response * 1000,
                'current_memory': current_memory,
                'error_counts': dict(self.error_counts),
                'psutil_available': HAS_PSUTIL,
                'monitoring_active': not self._stop_monitoring
            }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """詳細なメトリクスを取得"""
        with self._lock:
            # API エンドポイント別統計
            endpoint_stats = {}
            for endpoint, times in self.api_response_times.items():
                durations = [r['duration'] for r in times]
                if durations:
                    endpoint_stats[endpoint] = {
                        'count': len(durations),
                        'avg_response_ms': (sum(durations) / len(durations)) * 1000,
                        'min_response_ms': min(durations) * 1000,
                        'max_response_ms': max(durations) * 1000,
                        'last_call': max(times, key=lambda x: x['timestamp'])['timestamp'].isoformat()
                    }
            
            # 分析タイプ別統計
            analysis_type_stats = defaultdict(list)
            for analysis in self.analysis_times:
                analysis_type_stats[analysis['analysis_type']].append(analysis['duration'])
            
            analysis_stats = {}
            for analysis_type, durations in analysis_type_stats.items():
                analysis_stats[analysis_type] = {
                    'count': len(durations),
                    'avg_duration_ms': (sum(durations) / len(durations)) * 1000,
                    'min_duration_ms': min(durations) * 1000,
                    'max_duration_ms': max(durations) * 1000
                }
            
            return {
                'endpoint_statistics': endpoint_stats,
                'analysis_statistics': analysis_stats,
                'memory_history': list(self.memory_usage_history)[-10:],  # 最新10件
                'recent_analyses': list(self.analysis_times)[-10:]  # 最新10件
            }
    
    def generate_performance_report(self) -> str:
        """パフォーマンスレポートを生成"""
        summary = self.get_performance_summary()
        detailed = self.get_detailed_metrics()
        
        report = [
            f"Day Trade Personal - Performance Report",
            f"Generated at: {datetime.now().isoformat()}",
            f"",
            f"=== System Overview ===",
            f"Uptime: {summary['uptime_formatted']}",
            f"Total Analyses: {summary['total_analyses']}",
            f"Total API Calls: {summary['total_api_calls']}",
            f"Average Analysis Time: {summary['avg_analysis_time_ms']:.2f}ms",
            f"Average API Response: {summary['avg_api_response_ms']:.2f}ms",
            f""
        ]
        
        if summary['current_memory']:
            mem = summary['current_memory']
            report.extend([
                f"=== Current Resource Usage ===",
                f"Memory (RSS): {mem['rss_mb']:.2f} MB",
                f"Memory (VMS): {mem['vms_mb']:.2f} MB", 
                f"CPU Usage: {mem['cpu_percent']:.1f}%",
                f"Memory Usage: {mem['memory_percent']:.1f}%",
                f""
            ])
        
        if detailed['endpoint_statistics']:
            report.append("=== API Endpoint Statistics ===")
            for endpoint, stats in detailed['endpoint_statistics'].items():
                report.append(f"{endpoint}:")
                report.append(f"  Calls: {stats['count']}")
                report.append(f"  Avg Response: {stats['avg_response_ms']:.2f}ms")
                report.append(f"  Min/Max: {stats['min_response_ms']:.2f}ms / {stats['max_response_ms']:.2f}ms")
            report.append("")
        
        if summary['error_counts']:
            report.append("=== Error Summary ===")
            for error_type, count in summary['error_counts'].items():
                report.append(f"{error_type}: {count}")
            report.append("")
        
        return "\n".join(report)


# グローバルインスタンス
performance_monitor = PerformanceMonitor()


def track_performance(func):
    """パフォーマンス追跡デコレーター"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            performance_monitor.track_analysis_time(
                symbol=kwargs.get('symbol', 'unknown'),
                duration=duration,
                analysis_type=func.__name__
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            performance_monitor.track_error(f"{func.__name__}_error", str(e))
            raise
    return wrapper


if __name__ == "__main__":
    # テスト実行
    monitor = PerformanceMonitor()
    
    # テストデータ
    monitor.track_analysis_time("7203", 0.15, "quick_analysis")
    monitor.track_api_response_time("/api/status", 0.05)
    monitor.track_api_response_time("/api/recommendations", 0.12)
    
    print(monitor.generate_performance_report())