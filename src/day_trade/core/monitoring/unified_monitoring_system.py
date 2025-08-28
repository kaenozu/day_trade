"""
統合監視システム

包括的なシステム監視、メトリクス収集、アラート管理機能を提供。
"""

import asyncio
import time
import logging
import psutil
import json
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set, Tuple, Deque
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from threading import Lock, RLock
from concurrent.futures import ThreadPoolExecutor
import uuid

# 条件付きインポート
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp is not available. HTTP health checks will be disabled.")

from ..config.configuration_manager import ConfigurationManager
from ..security.security_manager import SecurityManager
from ..performance.performance_optimizer import PerformanceMonitor


class MetricType(Enum):
    """メトリクスタイプ定義"""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMER = "timer"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """アラートレベル定義"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringScope(Enum):
    """監視スコープ定義"""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class MetricValue:
    """メトリクス値"""
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """メトリクス定義"""
    name: str
    type: MetricType
    description: str
    unit: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    values: Deque[MetricValue] = field(default_factory=lambda: deque(maxlen=1000))
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Alert:
    """アラート定義"""
    alert_id: str
    name: str
    level: AlertLevel
    message: str
    source: str
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """ヘルスチェック定義"""
    name: str
    endpoint: str
    status: bool
    response_time: float
    last_check: datetime
    error_message: Optional[str] = None


class AlertRule(ABC):
    """アラートルール抽象基底クラス"""
    
    def __init__(self, name: str, level: AlertLevel):
        self.name = name
        self.level = level
        self.enabled = True
        
    @abstractmethod
    def evaluate(self, metric: Metric) -> Optional[Alert]:
        """アラート条件評価"""
        pass


class ThresholdAlertRule(AlertRule):
    """閾値ベースアラートルール"""
    
    def __init__(self, name: str, level: AlertLevel, metric_name: str,
                 threshold: float, comparison: str = "gt"):
        super().__init__(name, level)
        self.metric_name = metric_name
        self.threshold = threshold
        self.comparison = comparison  # gt, gte, lt, lte, eq, ne
        
    def evaluate(self, metric: Metric) -> Optional[Alert]:
        if metric.name != self.metric_name or not metric.values:
            return None
            
        current_value = metric.values[-1].value
        
        condition_met = False
        if self.comparison == "gt":
            condition_met = current_value > self.threshold
        elif self.comparison == "gte":
            condition_met = current_value >= self.threshold
        elif self.comparison == "lt":
            condition_met = current_value < self.threshold
        elif self.comparison == "lte":
            condition_met = current_value <= self.threshold
        elif self.comparison == "eq":
            condition_met = current_value == self.threshold
        elif self.comparison == "ne":
            condition_met = current_value != self.threshold
            
        if condition_met:
            return Alert(
                alert_id=str(uuid.uuid4()),
                name=self.name,
                level=self.level,
                message=f"{metric.name}が閾値{self.threshold}を超えました",
                source="threshold_rule",
                metric_name=metric.name,
                threshold_value=self.threshold,
                current_value=current_value
            )
        return None


class SystemMetricsCollector:
    """システムメトリクス収集器"""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval  # seconds
        self.running = False
        # プラットフォーム固有設定
        self.disk_path = 'C:\\' if platform.system() == 'Windows' else '/'
        
    async def collect_system_metrics(self) -> Dict[str, MetricValue]:
        """システムメトリクス収集"""
        metrics = {}
        now = datetime.now()
        
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics["system.cpu.usage"] = MetricValue(cpu_percent, now)
        
        # メモリ使用率
        memory = psutil.virtual_memory()
        metrics["system.memory.usage"] = MetricValue(memory.percent, now)
        metrics["system.memory.available"] = MetricValue(memory.available, now)
        
        # ディスク使用率
        try:
            disk = psutil.disk_usage(self.disk_path)
            metrics["system.disk.usage"] = MetricValue(
                (disk.used / disk.total) * 100, now
            )
        except (OSError, PermissionError) as e:
            logging.warning(f"ディスク使用率取得エラー: {e}")
            metrics["system.disk.usage"] = MetricValue(0.0, now)
        
        # ネットワーク統計
        net_io = psutil.net_io_counters()
        metrics["system.network.bytes_sent"] = MetricValue(net_io.bytes_sent, now)
        metrics["system.network.bytes_recv"] = MetricValue(net_io.bytes_recv, now)
        
        return metrics


class MetricsStorage:
    """メトリクスストレージ"""
    
    def __init__(self, retention_period: timedelta = timedelta(days=7), max_values: int = 1000):
        self.metrics: Dict[str, Metric] = {}
        self.retention_period = retention_period
        self.max_values = max_values
        self.lock = RLock()
        
    def create_metric(self, name: str, metric_type: MetricType, 
                     description: str, unit: str = "", 
                     labels: Optional[Dict[str, str]] = None) -> Metric:
        """メトリクス作成"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(
                    name=name,
                    type=metric_type,
                    description=description,
                    unit=unit,
                    labels=labels or {},
                    values=deque(maxlen=self.max_values)
                )
            return self.metrics[name]
            
    def record_metric(self, name: str, value: Union[int, float], 
                     labels: Dict[str, str] = None, 
                     metadata: Dict[str, Any] = None) -> bool:
        """メトリクス記録"""
        with self.lock:
            if name not in self.metrics:
                return False
                
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.now(),
                labels=labels or {},
                metadata=metadata or {}
            )
            
            self.metrics[name].values.append(metric_value)
            return True
            
    def get_metric(self, name: str) -> Optional[Metric]:
        """メトリクス取得"""
        with self.lock:
            return self.metrics.get(name)
            
    def get_all_metrics(self) -> Dict[str, Metric]:
        """全メトリクス取得"""
        with self.lock:
            return self.metrics.copy()
            
    def cleanup_old_metrics(self):
        """古いメトリクスクリーンアップ"""
        cutoff_time = datetime.now() - self.retention_period
        
        with self.lock:
            for metric in self.metrics.values():
                while (metric.values and 
                       metric.values[0].timestamp < cutoff_time):
                    metric.values.popleft()


class AlertManager:
    """アラート管理器"""
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: List[Alert] = []
        self.alert_rules: List[AlertRule] = []
        self.alert_handlers: Dict[AlertLevel, List[Callable[[Alert], None]]] = defaultdict(list)
        self.config_manager = config_manager
        self.lock = Lock()
        
    def add_alert_rule(self, rule: AlertRule):
        """アラートルール追加"""
        with self.lock:
            self.alert_rules.append(rule)
            
    def add_alert_handler(self, level: AlertLevel, 
                         handler: Callable[[Alert], None]):
        """アラートハンドラー追加"""
        self.alert_handlers[level].append(handler)
        
    def evaluate_alerts(self, metric: Metric):
        """アラート評価"""
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            alert = rule.evaluate(metric)
            if alert:
                self._trigger_alert(alert)
                
    def _trigger_alert(self, alert: Alert):
        """アラート発火"""
        with self.lock:
            self.active_alerts[alert.alert_id] = alert
            
        # アラートハンドラー実行
        handlers = self.alert_handlers.get(alert.level, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"アラートハンドラーエラー: {e}")
                
    def resolve_alert(self, alert_id: str) -> bool:
        """アラート解決"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts.pop(alert_id)
                alert.resolved_at = datetime.now()
                self.resolved_alerts.append(alert)
                return True
        return False
        
    def get_active_alerts(self) -> List[Alert]:
        """アクティブアラート取得"""
        with self.lock:
            return list(self.active_alerts.values())


class HealthCheckManager:
    """ヘルスチェック管理器"""
    
    def __init__(self, check_interval: int = 60, timeout: int = 10):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.check_interval = check_interval  # seconds
        self.timeout = timeout  # seconds
        self.running = False
        
    def register_health_check(self, name: str, endpoint: str):
        """ヘルスチェック登録"""
        self.health_checks[name] = HealthCheck(
            name=name,
            endpoint=endpoint,
            status=False,
            response_time=0.0,
            last_check=datetime.now()
        )
        
    async def perform_health_check(self, name: str) -> bool:
        """ヘルスチェック実行"""
        if name not in self.health_checks:
            return False
            
        health_check = self.health_checks[name]
        start_time = time.time()
        
        try:
            # HTTPチェック
            if not AIOHTTP_AVAILABLE:
                raise ImportError("aiohttp is not available")
                
            async with aiohttp.ClientSession() as session:
                async with session.get(health_check.endpoint, timeout=self.timeout) as response:
                    health_check.status = response.status == 200
                    health_check.response_time = time.time() - start_time
                    health_check.last_check = datetime.now()
                    health_check.error_message = None if health_check.status else f"HTTP {response.status}"
        except Exception as e:
            health_check.status = False
            health_check.response_time = time.time() - start_time
            health_check.last_check = datetime.now()
            health_check.error_message = str(e)
            
        return health_check.status


class DashboardMetrics:
    """ダッシュボード用メトリクス"""
    
    def __init__(self, storage: MetricsStorage):
        self.storage = storage
        
    def get_system_overview(self) -> Dict[str, Any]:
        """システム概要取得"""
        overview = {}
        
        # CPU使用率
        cpu_metric = self.storage.get_metric("system.cpu.usage")
        if cpu_metric and cpu_metric.values:
            overview["cpu_usage"] = cpu_metric.values[-1].value
            
        # メモリ使用率
        memory_metric = self.storage.get_metric("system.memory.usage")
        if memory_metric and memory_metric.values:
            overview["memory_usage"] = memory_metric.values[-1].value
            
        # ディスク使用率
        disk_metric = self.storage.get_metric("system.disk.usage")
        if disk_metric and disk_metric.values:
            overview["disk_usage"] = disk_metric.values[-1].value
            
        return overview
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンスメトリクス取得"""
        performance = {}
        
        # アプリケーション応答時間
        response_time_metric = self.storage.get_metric("app.response_time")
        if response_time_metric and response_time_metric.values:
            values = [v.value for v in response_time_metric.values[-100:]]
            performance["avg_response_time"] = sum(values) / len(values)
            performance["max_response_time"] = max(values)
            
        return performance


class UnifiedMonitoringSystem:
    """統合監視システム"""
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None,
                 security_manager: Optional[SecurityManager] = None,
                 performance_monitor: Optional[PerformanceMonitor] = None,
                 metrics_retention_days: int = 7,
                 collection_interval: int = 30,
                 health_check_interval: int = 60,
                 max_metric_values: int = 1000):
        self.config_manager = config_manager
        self.security_manager = security_manager
        self.performance_monitor = performance_monitor
        
        # コア コンポーネント
        self.metrics_storage = MetricsStorage(
            retention_period=timedelta(days=metrics_retention_days),
            max_values=max_metric_values
        )
        self.alert_manager = AlertManager(config_manager)
        self.health_check_manager = HealthCheckManager(
            check_interval=health_check_interval
        )
        self.system_metrics_collector = SystemMetricsCollector(
            collection_interval=collection_interval
        )
        self.dashboard_metrics = DashboardMetrics(self.metrics_storage)
        
        # 実行制御
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 統計情報
        self.start_time = datetime.now()
        self.metrics_collected = 0
        self.alerts_triggered = 0
        
        # 初期化
        self._initialize_default_metrics()
        self._initialize_default_alert_rules()
        self._initialize_alert_handlers()
        
    def _initialize_default_metrics(self):
        """デフォルトメトリクス初期化"""
        # システムメトリクス
        self.metrics_storage.create_metric(
            "system.cpu.usage", MetricType.GAUGE, "CPU使用率", "%"
        )
        self.metrics_storage.create_metric(
            "system.memory.usage", MetricType.GAUGE, "メモリ使用率", "%"
        )
        self.metrics_storage.create_metric(
            "system.disk.usage", MetricType.GAUGE, "ディスク使用率", "%"
        )
        
        # アプリケーションメトリクス
        self.metrics_storage.create_metric(
            "app.requests.total", MetricType.COUNTER, "リクエスト総数"
        )
        self.metrics_storage.create_metric(
            "app.response_time", MetricType.HISTOGRAM, "応答時間", "ms"
        )
        self.metrics_storage.create_metric(
            "app.errors.total", MetricType.COUNTER, "エラー総数"
        )
        
        # ビジネスメトリクス
        self.metrics_storage.create_metric(
            "business.trades.total", MetricType.COUNTER, "取引総数"
        )
        self.metrics_storage.create_metric(
            "business.profit_loss", MetricType.GAUGE, "損益", "円"
        )
        
    def _initialize_default_alert_rules(self):
        """デフォルトアラートルール初期化"""
        # CPU使用率アラート
        self.alert_manager.add_alert_rule(
            ThresholdAlertRule("CPU使用率高", AlertLevel.WARNING, 
                             "system.cpu.usage", 80.0, "gt")
        )
        self.alert_manager.add_alert_rule(
            ThresholdAlertRule("CPU使用率危険", AlertLevel.CRITICAL,
                             "system.cpu.usage", 95.0, "gt")
        )
        
        # メモリ使用率アラート
        self.alert_manager.add_alert_rule(
            ThresholdAlertRule("メモリ使用率高", AlertLevel.WARNING,
                             "system.memory.usage", 85.0, "gt")
        )
        
    def _initialize_alert_handlers(self):
        """アラートハンドラー初期化"""
        def log_alert(alert: Alert):
            logging.log(
                logging.ERROR if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else logging.WARNING,
                f"アラート発生: {alert.name} - {alert.message}"
            )
            
        # 全レベルのアラートをログに記録
        for level in AlertLevel:
            self.alert_manager.add_alert_handler(level, log_alert)
            
    async def start(self):
        """監視システム開始"""
        if self.running:
            return
            
        self.running = True
        logging.info("統合監視システムを開始します")
        
        # バックグラウンドタスク開始
        self.tasks = [
            asyncio.create_task(self._system_metrics_collection_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
    async def stop(self):
        """監視システム停止"""
        if not self.running:
            return
            
        self.running = False
        logging.info("統合監視システムを停止します")
        
        # タスク停止
        for task in self.tasks:
            task.cancel()
            
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        
        # リソースクリーンアップ
        self.executor.shutdown(wait=True)
        
    async def _system_metrics_collection_loop(self):
        """システムメトリクス収集ループ"""
        while self.running:
            try:
                metrics = await self.system_metrics_collector.collect_system_metrics()
                
                for name, metric_value in metrics.items():
                    self.metrics_storage.record_metric(
                        name, metric_value.value, 
                        metric_value.labels, metric_value.metadata
                    )
                    self.metrics_collected += 1
                    
                    # アラート評価
                    metric = self.metrics_storage.get_metric(name)
                    if metric:
                        self.alert_manager.evaluate_alerts(metric)
                        
            except Exception as e:
                logging.error(f"システムメトリクス収集エラー: {e}")
                
            await asyncio.sleep(self.system_metrics_collector.collection_interval)
            
    async def _health_check_loop(self):
        """ヘルスチェックループ"""
        while self.running:
            try:
                for name in self.health_check_manager.health_checks.keys():
                    await self.health_check_manager.perform_health_check(name)
            except Exception as e:
                logging.error(f"ヘルスチェックエラー: {e}")
                
            await asyncio.sleep(self.health_check_manager.check_interval)
            
    async def _cleanup_loop(self):
        """クリーンアップループ"""
        while self.running:
            try:
                self.metrics_storage.cleanup_old_metrics()
            except Exception as e:
                logging.error(f"クリーンアップエラー: {e}")
                
            await asyncio.sleep(3600)  # 1時間毎
            
    def record_business_metric(self, name: str, value: Union[int, float],
                              labels: Dict[str, str] = None):
        """ビジネスメトリクス記録"""
        success = self.metrics_storage.record_metric(name, value, labels)
        if success:
            metric = self.metrics_storage.get_metric(name)
            if metric:
                self.alert_manager.evaluate_alerts(metric)
                
    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視システム状態取得"""
        return {
            "running": self.running,
            "uptime": str(datetime.now() - self.start_time),
            "metrics_collected": self.metrics_collected,
            "alerts_triggered": len(self.alert_manager.active_alerts),
            "active_alerts": len(self.alert_manager.active_alerts),
            "health_checks": len(self.health_check_manager.health_checks),
            "system_overview": self.dashboard_metrics.get_system_overview()
        }
        
    def export_metrics(self, format: str = "json") -> str:
        """メトリクスエクスポート"""
        metrics_data = {}
        
        for name, metric in self.metrics_storage.get_all_metrics().items():
            metrics_data[name] = {
                "type": metric.type.value,
                "description": metric.description,
                "unit": metric.unit,
                "values": [
                    {
                        "value": v.value,
                        "timestamp": v.timestamp.isoformat(),
                        "labels": v.labels
                    }
                    for v in list(metric.values)[-100:]  # 最新100件
                ]
            }
            
        if format == "json":
            return json.dumps(metrics_data, ensure_ascii=False, indent=2)
        else:
            return str(metrics_data)


# グローバル監視システムインスタンス
_global_monitoring_system: Optional[UnifiedMonitoringSystem] = None


def get_global_monitoring_system() -> UnifiedMonitoringSystem:
    """グローバル監視システム取得"""
    global _global_monitoring_system
    if _global_monitoring_system is None:
        _global_monitoring_system = UnifiedMonitoringSystem()
    return _global_monitoring_system


def monitor_performance(func_name: str = ""):
    """パフォーマンス監視デコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            monitoring = get_global_monitoring_system()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # ms
                
                # パフォーマンスメトリクス記録
                metric_name = f"function.{func_name or func.__name__}.duration"
                monitoring.record_business_metric(metric_name, execution_time)
                
                return result
            except Exception as e:
                # エラーメトリクス記録
                error_metric = f"function.{func_name or func.__name__}.errors"
                monitoring.record_business_metric(error_metric, 1)
                raise
                
        return wrapper
    return decorator