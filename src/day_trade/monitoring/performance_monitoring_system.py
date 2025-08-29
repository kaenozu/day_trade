"""
パフォーマンス監視・自動調整システム

システム全体のパフォーマンスを監視し、自動的に最適化調整を行う
"""

import asyncio
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import threading
import queue
import logging
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import json
import sqlite3
from pathlib import Path
import statistics
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


@dataclass
class SystemMetrics:
    """システムメトリクス"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    process_count: int
    thread_count: int
    prediction_accuracy: Optional[float] = None
    prediction_latency_ms: Optional[float] = None
    throughput_records_per_sec: Optional[float] = None
    error_rate: Optional[float] = None


@dataclass
class PerformanceAlert:
    """パフォーマンスアラート"""
    timestamp: datetime
    severity: str  # 'info', 'warning', 'critical'
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    auto_action_taken: bool = False
    action_description: Optional[str] = None


@dataclass
class OptimizationAction:
    """最適化アクション"""
    name: str
    description: str
    trigger_condition: Callable[[SystemMetrics], bool]
    action_function: Callable[[], Dict[str, Any]]
    cooldown_minutes: int = 5
    max_executions_per_hour: int = 10
    priority: int = 1
    enabled: bool = True


class MetricsDatabase:
    """メトリクスデータベース"""
    
    def __init__(self, db_path: str = "performance_metrics.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_available_gb REAL,
                    disk_usage_percent REAL,
                    disk_io_read_mb REAL,
                    disk_io_write_mb REAL,
                    network_sent_mb REAL,
                    network_recv_mb REAL,
                    active_connections INTEGER,
                    process_count INTEGER,
                    thread_count INTEGER,
                    prediction_accuracy REAL,
                    prediction_latency_ms REAL,
                    throughput_records_per_sec REAL,
                    error_rate REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL,
                    threshold_value REAL,
                    message TEXT,
                    auto_action_taken BOOLEAN,
                    action_description TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON system_metrics(timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp 
                ON performance_alerts(timestamp)
            ''')
    
    def store_metrics(self, metrics: SystemMetrics):
        """メトリクス保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO system_metrics (
                    timestamp, cpu_percent, memory_percent, memory_available_gb,
                    disk_usage_percent, disk_io_read_mb, disk_io_write_mb,
                    network_sent_mb, network_recv_mb, active_connections,
                    process_count, thread_count, prediction_accuracy,
                    prediction_latency_ms, throughput_records_per_sec, error_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.memory_available_gb,
                metrics.disk_usage_percent,
                metrics.disk_io_read_mb,
                metrics.disk_io_write_mb,
                metrics.network_sent_mb,
                metrics.network_recv_mb,
                metrics.active_connections,
                metrics.process_count,
                metrics.thread_count,
                metrics.prediction_accuracy,
                metrics.prediction_latency_ms,
                metrics.throughput_records_per_sec,
                metrics.error_rate
            ))
    
    def store_alert(self, alert: PerformanceAlert):
        """アラート保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO performance_alerts (
                    timestamp, severity, metric_name, current_value,
                    threshold_value, message, auto_action_taken, action_description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp.isoformat(),
                alert.severity,
                alert.metric_name,
                alert.current_value,
                alert.threshold_value,
                alert.message,
                alert.auto_action_taken,
                alert.action_description
            ))
    
    def get_recent_metrics(self, hours: int = 24) -> List[SystemMetrics]:
        """最近のメトリクス取得"""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM system_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            ''', (since,))
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append(SystemMetrics(
                    timestamp=datetime.fromisoformat(row[1]),
                    cpu_percent=row[2],
                    memory_percent=row[3],
                    memory_available_gb=row[4],
                    disk_usage_percent=row[5],
                    disk_io_read_mb=row[6],
                    disk_io_write_mb=row[7],
                    network_sent_mb=row[8],
                    network_recv_mb=row[9],
                    active_connections=row[10],
                    process_count=row[11],
                    thread_count=row[12],
                    prediction_accuracy=row[13],
                    prediction_latency_ms=row[14],
                    throughput_records_per_sec=row[15],
                    error_rate=row[16]
                ))
            
            return metrics


class AdaptiveThresholdManager:
    """適応的閾値管理"""
    
    def __init__(self, history_size: int = 1440):  # 24時間分（1分間隔）
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.static_thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 85},
            'memory_percent': {'warning': 80, 'critical': 90},
            'disk_usage_percent': {'warning': 80, 'critical': 90},
            'error_rate': {'warning': 0.05, 'critical': 0.10},
            'prediction_latency_ms': {'warning': 100, 'critical': 500}
        }
    
    def update_history(self, metrics: SystemMetrics):
        """履歴更新"""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) >= 60:  # 1時間分のデータがあれば
            self._recalculate_dynamic_thresholds()
    
    def _recalculate_dynamic_thresholds(self):
        """動的閾値再計算"""
        if len(self.metrics_history) < 60:
            return
        
        # 各メトリクスの統計値計算
        recent_metrics = list(self.metrics_history)[-60:]  # 最近1時間
        
        metrics_data = {
            'cpu_percent': [m.cpu_percent for m in recent_metrics],
            'memory_percent': [m.memory_percent for m in recent_metrics],
            'prediction_latency_ms': [m.prediction_latency_ms for m in recent_metrics if m.prediction_latency_ms is not None],
            'throughput_records_per_sec': [m.throughput_records_per_sec for m in recent_metrics if m.throughput_records_per_sec is not None]
        }
        
        for metric_name, values in metrics_data.items():
            if len(values) < 10:
                continue
            
            # 統計値
            mean_val = np.mean(values)
            std_val = np.std(values)
            p95 = np.percentile(values, 95)
            
            # 動的閾値設定
            if metric_name in ['cpu_percent', 'memory_percent']:
                warning_threshold = min(mean_val + 1.5 * std_val, p95)
                critical_threshold = min(mean_val + 2.5 * std_val, p95 * 1.1)
            elif metric_name == 'prediction_latency_ms':
                warning_threshold = mean_val + 2 * std_val
                critical_threshold = mean_val + 3 * std_val
            else:  # throughput
                warning_threshold = max(mean_val - 2 * std_val, mean_val * 0.7)
                critical_threshold = max(mean_val - 3 * std_val, mean_val * 0.5)
            
            self.thresholds[metric_name] = {
                'warning': warning_threshold,
                'critical': critical_threshold
            }
    
    def get_threshold(self, metric_name: str, level: str) -> float:
        """閾値取得"""
        # 動的閾値が利用可能ならそれを、なければ静的閾値を使用
        if metric_name in self.thresholds:
            return self.thresholds[metric_name].get(level, 0.0)
        elif metric_name in self.static_thresholds:
            return self.static_thresholds[metric_name].get(level, 0.0)
        else:
            return 0.0
    
    def check_thresholds(self, metrics: SystemMetrics) -> List[PerformanceAlert]:
        """閾値チェック"""
        alerts = []
        
        metrics_to_check = {
            'cpu_percent': metrics.cpu_percent,
            'memory_percent': metrics.memory_percent,
            'disk_usage_percent': metrics.disk_usage_percent,
            'prediction_latency_ms': metrics.prediction_latency_ms,
            'error_rate': metrics.error_rate
        }
        
        for metric_name, value in metrics_to_check.items():
            if value is None:
                continue
            
            critical_threshold = self.get_threshold(metric_name, 'critical')
            warning_threshold = self.get_threshold(metric_name, 'warning')
            
            if critical_threshold > 0 and value >= critical_threshold:
                alerts.append(PerformanceAlert(
                    timestamp=metrics.timestamp,
                    severity='critical',
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=critical_threshold,
                    message=f'{metric_name} が危険レベル {value:.2f} (閾値: {critical_threshold:.2f})'
                ))
            elif warning_threshold > 0 and value >= warning_threshold:
                alerts.append(PerformanceAlert(
                    timestamp=metrics.timestamp,
                    severity='warning',
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=warning_threshold,
                    message=f'{metric_name} が警告レベル {value:.2f} (閾値: {warning_threshold:.2f})'
                ))
        
        return alerts


class PerformanceMonitoringSystem:
    """パフォーマンス監視システム"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # コンポーネント
        self.db = MetricsDatabase()
        self.threshold_manager = AdaptiveThresholdManager()
        
        # 監視制御
        self.monitoring_active = False
        self.monitoring_interval = self.config.get('monitoring_interval', 60)  # 秒
        
        # スレッドとキュー
        self.monitor_thread: Optional[threading.Thread] = None
        self.action_executor = ThreadPoolExecutor(max_workers=4)
        self.alert_queue = queue.Queue()
        
        # 最適化アクション
        self.optimization_actions: List[OptimizationAction] = []
        self.action_history: Dict[str, List[datetime]] = defaultdict(list)
        
        # 統計
        self.monitoring_stats = {
            'start_time': datetime.now(),
            'metrics_collected': 0,
            'alerts_generated': 0,
            'actions_executed': 0,
            'errors_encountered': 0
        }
        
        # デフォルトアクション登録
        self._register_default_actions()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _register_default_actions(self):
        """デフォルト最適化アクション登録"""
        
        # CPU最適化
        self.register_action(OptimizationAction(
            name='cpu_optimization',
            description='CPU使用率が高い場合の最適化',
            trigger_condition=lambda m: m.cpu_percent > 80,
            action_function=self._optimize_cpu_usage,
            cooldown_minutes=10
        ))
        
        # メモリ最適化
        self.register_action(OptimizationAction(
            name='memory_optimization',
            description='メモリ使用率が高い場合の最適化',
            trigger_condition=lambda m: m.memory_percent > 85,
            action_function=self._optimize_memory_usage,
            cooldown_minutes=5
        ))
        
        # レイテンシ最適化
        self.register_action(OptimizationAction(
            name='latency_optimization',
            description='予測レイテンシが高い場合の最適化',
            trigger_condition=lambda m: m.prediction_latency_ms is not None and m.prediction_latency_ms > 200,
            action_function=self._optimize_prediction_latency,
            cooldown_minutes=15
        ))
        
        # スループット最適化
        self.register_action(OptimizationAction(
            name='throughput_optimization',
            description='スループットが低い場合の最適化',
            trigger_condition=lambda m: m.throughput_records_per_sec is not None and m.throughput_records_per_sec < 1000,
            action_function=self._optimize_throughput,
            cooldown_minutes=10
        ))
    
    def register_action(self, action: OptimizationAction):
        """最適化アクション登録"""
        self.optimization_actions.append(action)
        self.optimization_actions.sort(key=lambda a: a.priority)
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """システムメトリクス収集"""
        try:
            # システムメトリクス
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # I/O統計
            try:
                disk_io = psutil.disk_io_counters()
                disk_io_read_mb = disk_io.read_bytes / 1024 / 1024 if disk_io else 0
                disk_io_write_mb = disk_io.write_bytes / 1024 / 1024 if disk_io else 0
            except:
                disk_io_read_mb = disk_io_write_mb = 0
            
            try:
                network_io = psutil.net_io_counters()
                network_sent_mb = network_io.bytes_sent / 1024 / 1024 if network_io else 0
                network_recv_mb = network_io.bytes_recv / 1024 / 1024 if network_io else 0
            except:
                network_sent_mb = network_recv_mb = 0
            
            # プロセス情報
            try:
                active_connections = len(psutil.net_connections())
            except:
                active_connections = 0
            
            process_count = len(psutil.pids())
            
            try:
                current_process = psutil.Process()
                thread_count = current_process.num_threads()
            except:
                thread_count = 0
            
            # アプリケーションメトリクス（模擬）
            prediction_accuracy = np.random.uniform(0.6, 0.8)  # 実際のシステムでは実際の値を使用
            prediction_latency_ms = np.random.uniform(50, 150)
            throughput_records_per_sec = np.random.uniform(800, 2000)
            error_rate = np.random.uniform(0.01, 0.05)
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / 1024 / 1024 / 1024,
                disk_usage_percent=disk.percent,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                active_connections=active_connections,
                process_count=process_count,
                thread_count=thread_count,
                prediction_accuracy=prediction_accuracy,
                prediction_latency_ms=prediction_latency_ms,
                throughput_records_per_sec=throughput_records_per_sec,
                error_rate=error_rate
            )
            
        except Exception as e:
            self.logger.error(f"メトリクス収集エラー: {e}")
            self.monitoring_stats['errors_encountered'] += 1
            raise
    
    def _monitoring_loop(self):
        """監視ループ"""
        self.logger.info("パフォーマンス監視開始")
        
        while self.monitoring_active:
            try:
                # メトリクス収集
                metrics = asyncio.run(self.collect_system_metrics())
                
                # データベース保存
                self.db.store_metrics(metrics)
                
                # 適応的閾値更新
                self.threshold_manager.update_history(metrics)
                
                # アラートチェック
                alerts = self.threshold_manager.check_thresholds(metrics)
                
                for alert in alerts:
                    self.db.store_alert(alert)
                    self.alert_queue.put(alert)
                    self.monitoring_stats['alerts_generated'] += 1
                    self.logger.warning(alert.message)
                
                # 最適化アクション実行
                self._execute_optimization_actions(metrics)
                
                self.monitoring_stats['metrics_collected'] += 1
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                self.monitoring_stats['errors_encountered'] += 1
                time.sleep(10)  # エラー時は長めに休憩
    
    def _execute_optimization_actions(self, metrics: SystemMetrics):
        """最適化アクション実行"""
        for action in self.optimization_actions:
            if not action.enabled:
                continue
            
            try:
                # トリガー条件チェック
                if not action.trigger_condition(metrics):
                    continue
                
                # クールダウンチェック
                if self._is_in_cooldown(action):
                    continue
                
                # 時間当たり実行数制限チェック
                if self._exceeds_hourly_limit(action):
                    continue
                
                # アクション実行
                self.logger.info(f"最適化アクション実行: {action.name}")
                
                future = self.action_executor.submit(action.action_function)
                result = future.result(timeout=30)  # 30秒タイムアウト
                
                # 実行履歴記録
                self.action_history[action.name].append(datetime.now())
                self.monitoring_stats['actions_executed'] += 1
                
                self.logger.info(f"最適化アクション完了: {action.name} - {result}")
                
            except Exception as e:
                self.logger.error(f"最適化アクション失敗: {action.name} - {e}")
                self.monitoring_stats['errors_encountered'] += 1
    
    def _is_in_cooldown(self, action: OptimizationAction) -> bool:
        """クールダウンチェック"""
        if action.name not in self.action_history:
            return False
        
        last_execution = max(self.action_history[action.name]) if self.action_history[action.name] else None
        if not last_execution:
            return False
        
        return (datetime.now() - last_execution).seconds < (action.cooldown_minutes * 60)
    
    def _exceeds_hourly_limit(self, action: OptimizationAction) -> bool:
        """時間当たり実行数制限チェック"""
        if action.name not in self.action_history:
            return False
        
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_executions = [
            dt for dt in self.action_history[action.name] 
            if dt > one_hour_ago
        ]
        
        return len(recent_executions) >= action.max_executions_per_hour
    
    # 最適化アクション実装
    def _optimize_cpu_usage(self) -> Dict[str, Any]:
        """CPU使用率最適化"""
        import gc
        gc.collect()
        
        # CPU集約タスクの調整（実際のシステムに応じて実装）
        return {'action': 'cpu_optimization', 'result': 'ガベージコレクション実行'}
    
    def _optimize_memory_usage(self) -> Dict[str, Any]:
        """メモリ使用量最適化"""
        import gc
        gc.collect()
        
        # キャッシュクリアなど（実際のシステムに応じて実装）
        return {'action': 'memory_optimization', 'result': 'メモリクリーンアップ実行'}
    
    def _optimize_prediction_latency(self) -> Dict[str, Any]:
        """予測レイテンシ最適化"""
        # バッチサイズ調整、並列度調整など
        return {'action': 'latency_optimization', 'result': '予測処理最適化'}
    
    def _optimize_throughput(self) -> Dict[str, Any]:
        """スループット最適化"""
        # 並列処理数増加など
        return {'action': 'throughput_optimization', 'result': 'スループット向上処理'}
    
    async def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_stats['start_time'] = datetime.now()
        
        # 監視スレッド開始
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("パフォーマンス監視システム開始")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        self.action_executor.shutdown(wait=True)
        
        self.logger.info("パフォーマンス監視システム停止")
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        uptime = (datetime.now() - self.monitoring_stats['start_time']).seconds
        
        # 最近のメトリクス統計
        recent_metrics = self.db.get_recent_metrics(hours=1)
        
        if recent_metrics:
            avg_cpu = statistics.mean([m.cpu_percent for m in recent_metrics])
            avg_memory = statistics.mean([m.memory_percent for m in recent_metrics])
            avg_latency = statistics.mean([
                m.prediction_latency_ms for m in recent_metrics 
                if m.prediction_latency_ms is not None
            ])
        else:
            avg_cpu = avg_memory = avg_latency = 0.0
        
        # アクション統計
        action_stats = {}
        for action_name, executions in self.action_history.items():
            recent_executions = [
                dt for dt in executions 
                if dt > (datetime.now() - timedelta(hours=24))
            ]
            action_stats[action_name] = len(recent_executions)
        
        return {
            'uptime_seconds': uptime,
            'monitoring_active': self.monitoring_active,
            'monitoring_stats': self.monitoring_stats,
            'recent_averages': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory,
                'prediction_latency_ms': avg_latency
            },
            'active_actions': len(self.optimization_actions),
            'action_executions_24h': action_stats,
            'total_metrics_stored': len(self.db.get_recent_metrics(hours=24))
        }
    
    async def generate_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """パフォーマンスレポート生成"""
        metrics = self.db.get_recent_metrics(hours)
        
        if not metrics:
            return {'error': 'データが不足しています'}
        
        # 統計計算
        cpu_values = [m.cpu_percent for m in metrics]
        memory_values = [m.memory_percent for m in metrics]
        latency_values = [m.prediction_latency_ms for m in metrics if m.prediction_latency_ms is not None]
        throughput_values = [m.throughput_records_per_sec for m in metrics if m.throughput_records_per_sec is not None]
        
        report = {
            'report_period_hours': hours,
            'total_data_points': len(metrics),
            'cpu_usage': {
                'average': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'std_dev': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory_usage': {
                'average': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'std_dev': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            },
            'prediction_performance': {
                'average_latency_ms': statistics.mean(latency_values) if latency_values else 0,
                'max_latency_ms': max(latency_values) if latency_values else 0,
                'average_throughput': statistics.mean(throughput_values) if throughput_values else 0,
                'min_throughput': min(throughput_values) if throughput_values else 0
            },
            'system_health_score': self._calculate_health_score(metrics),
            'recommendations': self._generate_recommendations(metrics)
        }
        
        return report
    
    def _calculate_health_score(self, metrics: List[SystemMetrics]) -> float:
        """システムヘルススコア計算"""
        if not metrics:
            return 0.0
        
        # 各メトリクスのスコア計算（0-100）
        cpu_score = max(0, 100 - statistics.mean([m.cpu_percent for m in metrics]))
        memory_score = max(0, 100 - statistics.mean([m.memory_percent for m in metrics]))
        
        latency_values = [m.prediction_latency_ms for m in metrics if m.prediction_latency_ms is not None]
        latency_score = max(0, 100 - statistics.mean(latency_values) / 2) if latency_values else 100
        
        throughput_values = [m.throughput_records_per_sec for m in metrics if m.throughput_records_per_sec is not None]
        throughput_score = min(100, statistics.mean(throughput_values) / 20) if throughput_values else 100
        
        # 重み付き平均
        health_score = (cpu_score * 0.3 + memory_score * 0.3 + 
                       latency_score * 0.2 + throughput_score * 0.2)
        
        return min(100, max(0, health_score))
    
    def _generate_recommendations(self, metrics: List[SystemMetrics]) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        if not metrics:
            return recommendations
        
        avg_cpu = statistics.mean([m.cpu_percent for m in metrics])
        avg_memory = statistics.mean([m.memory_percent for m in metrics])
        
        latency_values = [m.prediction_latency_ms for m in metrics if m.prediction_latency_ms is not None]
        avg_latency = statistics.mean(latency_values) if latency_values else 0
        
        if avg_cpu > 70:
            recommendations.append("CPU使用率が高いため、処理の最適化を推奨")
        
        if avg_memory > 80:
            recommendations.append("メモリ使用率が高いため、メモリ管理の改善を推奨")
        
        if avg_latency > 100:
            recommendations.append("予測レイテンシが高いため、アルゴリズムの最適化を推奨")
        
        if not recommendations:
            recommendations.append("システムは良好に動作しています")
        
        return recommendations


async def demo_performance_monitoring():
    """パフォーマンス監視デモ"""
    print("=== パフォーマンス監視・自動調整システム デモ ===")
    
    monitoring_system = PerformanceMonitoringSystem()
    
    try:
        # 監視開始
        await monitoring_system.start_monitoring()
        print("監視システム開始")
        
        # 30秒間の監視
        for i in range(30):
            if i % 10 == 0:
                status = monitoring_system.get_system_status()
                print(f"経過: {i}s - メトリクス: {status['monitoring_stats']['metrics_collected']}, "
                     f"アラート: {status['monitoring_stats']['alerts_generated']}, "
                     f"アクション: {status['monitoring_stats']['actions_executed']}")
            
            await asyncio.sleep(1)
        
        # レポート生成
        print("\nパフォーマンスレポート生成中...")
        report = await monitoring_system.generate_performance_report(hours=1)
        
        print("=== パフォーマンスレポート ===")
        print(f"データ点数: {report['total_data_points']}")
        print(f"CPU使用率: {report['cpu_usage']['average']:.1f}% (最大: {report['cpu_usage']['max']:.1f}%)")
        print(f"メモリ使用率: {report['memory_usage']['average']:.1f}% (最大: {report['memory_usage']['max']:.1f}%)")
        print(f"予測レイテンシ: {report['prediction_performance']['average_latency_ms']:.1f}ms")
        print(f"スループット: {report['prediction_performance']['average_throughput']:.1f} records/sec")
        print(f"システムヘルススコア: {report['system_health_score']:.1f}/100")
        
        print("\n=== 推奨事項 ===")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        # 最終状態
        final_status = monitoring_system.get_system_status()
        print(f"\n=== 最終統計 ===")
        print(f"収集メトリクス数: {final_status['monitoring_stats']['metrics_collected']}")
        print(f"生成アラート数: {final_status['monitoring_stats']['alerts_generated']}")
        print(f"実行アクション数: {final_status['monitoring_stats']['actions_executed']}")
        print(f"エラー数: {final_status['monitoring_stats']['errors_encountered']}")
        
    finally:
        monitoring_system.stop_monitoring()
        print("監視システム停止")


if __name__ == "__main__":
    asyncio.run(demo_performance_monitoring())