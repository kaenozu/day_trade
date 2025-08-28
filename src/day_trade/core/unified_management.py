#!/usr/bin/env python3
"""
統合管理システム

設定管理、メトリクス収集、パフォーマンス監視を統合提供します。
"""

import asyncio
import json
import logging
import sqlite3
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import threading
import psutil

from .base import BaseComponent, BaseConfig, HealthStatus, SystemStatus
from .common_infrastructure import (
    BaseStorage, InMemoryStorage, SystemMetrics, Priority
)
from .unified_system_error import UnifiedSystemError, ErrorSeverity

logger = logging.getLogger(__name__)


@dataclass
class ConfigSchema:
    """設定スキーマ"""
    name: str
    data_type: str  # str, int, float, bool, list, dict
    default_value: Any
    required: bool = False
    validation_rule: Optional[str] = None
    description: str = ""
    category: str = "general"
    sensitive: bool = False  # 機密情報フラグ


@dataclass
class ConfigEntry:
    """設定エントリ"""
    key: str
    value: Any
    schema: ConfigSchema
    last_updated: float = field(default_factory=time.time)
    updated_by: str = "system"
    version: int = 1


@dataclass
class MetricDefinition:
    """メトリクス定義"""
    name: str
    metric_type: str  # counter, gauge, histogram, summary
    description: str = ""
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    aggregation_window: float = 60.0  # 秒
    retention_hours: float = 24.0


@dataclass
class MetricPoint:
    """メトリクスポイント"""
    name: str
    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """アラートルール"""
    name: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: Union[int, float]
    duration_seconds: float = 60.0
    severity: ErrorSeverity = ErrorSeverity.WARNING
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    description: str = ""


class UnifiedConfigManager(BaseComponent):
    """統合設定管理"""
    
    def __init__(self, name: str, config: BaseConfig):
        super().__init__(name, config)
        self.config_storage = InMemoryStorage(f"{name}_configs", {
            'max_size': 10000,
            'cleanup_interval': 3600
        })
        self.schema_registry: Dict[str, ConfigSchema] = {}
        self.config_entries: Dict[str, ConfigEntry] = {}
        self.config_watchers: Dict[str, List[Callable]] = defaultdict(list)
        self._file_watchers: Dict[str, float] = {}  # ファイル監視用
        self._watch_task = None
        self._lock = threading.RLock()
        
        # デフォルトスキーマ登録
        self._register_default_schemas()
    
    async def start(self) -> bool:
        """設定管理開始"""
        try:
            await self.config_storage.initialize()
            self._watch_task = asyncio.create_task(self._watch_config_files())
            
            # 設定ファイル読み込み
            await self._load_configurations()
            
            self.status = SystemStatus.RUNNING
            logger.info("UnifiedConfigManager started successfully")
            return True
            
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="config_manager_start",
                component=self.name,
                severity=ErrorSeverity.HIGH
            ))
            return False
    
    async def stop(self) -> bool:
        """設定管理停止"""
        try:
            self.status = SystemStatus.STOPPING
            
            if self._watch_task:
                self._watch_task.cancel()
            
            # 設定保存
            await self._save_configurations()
            
            self.status = SystemStatus.STOPPED
            return True
            
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="config_manager_stop",
                component=self.name,
                severity=ErrorSeverity.HIGH
            ))
            return False
    
    def register_schema(self, schema: ConfigSchema):
        """設定スキーマ登録"""
        with self._lock:
            self.schema_registry[schema.name] = schema
            
            # デフォルト値設定
            if schema.name not in self.config_entries:
                self.config_entries[schema.name] = ConfigEntry(
                    key=schema.name,
                    value=schema.default_value,
                    schema=schema
                )
            
            logger.info(f"Configuration schema registered: {schema.name}")
    
    def set_config(self, key: str, value: Any, updated_by: str = "system") -> bool:
        """設定値設定"""
        try:
            with self._lock:
                if key not in self.schema_registry:
                    logger.warning(f"No schema found for config key: {key}")
                    return False
                
                schema = self.schema_registry[key]
                
                # バリデーション
                if not self._validate_config_value(schema, value):
                    logger.error(f"Invalid config value for {key}: {value}")
                    return False
                
                # 既存エントリ更新または新規作成
                if key in self.config_entries:
                    entry = self.config_entries[key]
                    old_value = entry.value
                    entry.value = value
                    entry.last_updated = time.time()
                    entry.updated_by = updated_by
                    entry.version += 1
                else:
                    entry = ConfigEntry(
                        key=key,
                        value=value,
                        schema=schema,
                        updated_by=updated_by
                    )
                    self.config_entries[key] = entry
                
                # ストレージ保存
                asyncio.create_task(self.config_storage.set(key, entry))
                
                # 変更通知
                self._notify_config_watchers(key, value, entry.value if 'old_value' in locals() else None)
                
                logger.info(f"Configuration updated: {key} = {value if not schema.sensitive else '[HIDDEN]'}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            return False
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """設定値取得"""
        with self._lock:
            if key in self.config_entries:
                return self.config_entries[key].value
            elif key in self.schema_registry:
                return self.schema_registry[key].default_value
            else:
                return default
    
    def get_config_entry(self, key: str) -> Optional[ConfigEntry]:
        """設定エントリ取得"""
        with self._lock:
            return self.config_entries.get(key)
    
    def watch_config(self, key: str, callback: Callable[[str, Any, Any], None]):
        """設定監視"""
        self.config_watchers[key].append(callback)
        logger.info(f"Config watcher registered for: {key}")
    
    def unwatch_config(self, key: str, callback: Callable):
        """設定監視解除"""
        if key in self.config_watchers:
            try:
                self.config_watchers[key].remove(callback)
                logger.info(f"Config watcher unregistered for: {key}")
            except ValueError:
                pass
    
    def _register_default_schemas(self):
        """デフォルトスキーマ登録"""
        default_schemas = [
            ConfigSchema("log_level", "str", "INFO", description="ログレベル"),
            ConfigSchema("max_workers", "int", 4, description="最大ワーカー数"),
            ConfigSchema("timeout_seconds", "float", 30.0, description="タイムアウト"),
            ConfigSchema("enable_metrics", "bool", True, description="メトリクス収集有効"),
            ConfigSchema("database_url", "str", "", sensitive=True, description="データベースURL"),
            ConfigSchema("api_key", "str", "", sensitive=True, description="APIキー"),
        ]
        
        for schema in default_schemas:
            self.register_schema(schema)
    
    def _validate_config_value(self, schema: ConfigSchema, value: Any) -> bool:
        """設定値バリデーション"""
        try:
            # 型チェック
            if schema.data_type == "str" and not isinstance(value, str):
                return False
            elif schema.data_type == "int" and not isinstance(value, int):
                return False
            elif schema.data_type == "float" and not isinstance(value, (int, float)):
                return False
            elif schema.data_type == "bool" and not isinstance(value, bool):
                return False
            elif schema.data_type == "list" and not isinstance(value, list):
                return False
            elif schema.data_type == "dict" and not isinstance(value, dict):
                return False
            
            # 追加バリデーション（実際の実装では正規表現等を使用）
            if schema.validation_rule:
                # バリデーションルール適用
                pass
            
            return True
        except Exception:
            return False
    
    def _notify_config_watchers(self, key: str, new_value: Any, old_value: Any):
        """設定変更通知"""
        for callback in self.config_watchers.get(key, []):
            try:
                callback(key, new_value, old_value)
            except Exception as e:
                logger.error(f"Config watcher callback failed for {key}: {e}")
    
    async def _load_configurations(self):
        """設定ファイル読み込み"""
        try:
            config_file = Path("config.json")
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                for key, value in config_data.items():
                    if key in self.schema_registry:
                        self.set_config(key, value, "file")
                
                logger.info("Configuration loaded from file")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    async def _save_configurations(self):
        """設定ファイル保存"""
        try:
            config_data = {}
            for key, entry in self.config_entries.items():
                if not entry.schema.sensitive:  # 機密情報は除外
                    config_data[key] = entry.value
            
            config_file = Path("config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Configuration saved to file")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    async def _watch_config_files(self):
        """設定ファイル監視"""
        while self.status == SystemStatus.RUNNING:
            try:
                config_file = Path("config.json")
                if config_file.exists():
                    mtime = config_file.stat().st_mtime
                    
                    if str(config_file) not in self._file_watchers:
                        self._file_watchers[str(config_file)] = mtime
                    elif self._file_watchers[str(config_file)] != mtime:
                        logger.info("Configuration file changed, reloading...")
                        await self._load_configurations()
                        self._file_watchers[str(config_file)] = mtime
                
                await asyncio.sleep(5)  # 5秒間隔でチェック
                
            except Exception as e:
                logger.error(f"Config file watching error: {e}")
                await asyncio.sleep(5)
    
    async def health_check(self) -> HealthStatus:
        """健全性チェック"""
        try:
            if self.status != SystemStatus.RUNNING:
                return HealthStatus.UNHEALTHY
            
            # ストレージ健全性チェック
            storage_health = await self.config_storage.health_check()
            return storage_health
            
        except Exception:
            return HealthStatus.UNHEALTHY


class UnifiedMetricsCollector(BaseComponent):
    """統合メトリクス収集"""
    
    def __init__(self, name: str, config: BaseConfig):
        super().__init__(name, config)
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.metric_storage = InMemoryStorage(f"{name}_metrics", {
            'max_size': 50000,
            'cleanup_interval': 300
        })
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_states: Dict[str, Dict[str, Any]] = {}
        self._collection_tasks: Dict[str, asyncio.Task] = {}
        self._aggregation_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()
    
    async def start(self) -> bool:
        """メトリクス収集開始"""
        try:
            await self.metric_storage.initialize()
            
            # システムメトリクス収集開始
            self._collection_tasks['system'] = asyncio.create_task(
                self._collect_system_metrics()
            )
            
            # アラート監視開始
            self._collection_tasks['alerts'] = asyncio.create_task(
                self._monitor_alerts()
            )
            
            self.status = SystemStatus.RUNNING
            logger.info("UnifiedMetricsCollector started successfully")
            return True
            
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="metrics_collector_start",
                component=self.name,
                severity=ErrorSeverity.HIGH
            ))
            return False
    
    async def stop(self) -> bool:
        """メトリクス収集停止"""
        try:
            self.status = SystemStatus.STOPPING
            
            # 収集タスク停止
            for task in self._collection_tasks.values():
                task.cancel()
            
            if self._collection_tasks:
                await asyncio.gather(*self._collection_tasks.values(), return_exceptions=True)
            
            self.status = SystemStatus.STOPPED
            return True
            
        except Exception as e:
            await self._handle_error(e, ErrorContext(
                operation="metrics_collector_stop",
                component=self.name,
                severity=ErrorSeverity.HIGH
            ))
            return False
    
    def register_metric(self, metric_def: MetricDefinition):
        """メトリクス定義登録"""
        with self._lock:
            self.metric_definitions[metric_def.name] = metric_def
            logger.info(f"Metric definition registered: {metric_def.name}")
    
    def register_alert_rule(self, alert_rule: AlertRule):
        """アラートルール登録"""
        with self._lock:
            self.alert_rules[alert_rule.name] = alert_rule
            self.alert_states[alert_rule.name] = {
                'triggered': False,
                'trigger_count': 0,
                'first_trigger': None,
                'last_check': time.time()
            }
            logger.info(f"Alert rule registered: {alert_rule.name}")
    
    async def record_metric(self, metric_point: MetricPoint):
        """メトリクス記録"""
        try:
            with self._lock:
                if metric_point.name not in self.metric_definitions:
                    logger.warning(f"Unknown metric: {metric_point.name}")
                    return
                
                # ストレージ保存
                metric_key = f"{metric_point.name}_{int(metric_point.timestamp)}"
                await self.metric_storage.set(metric_key, metric_point, ttl=86400)
                
                # 集約バッファ追加
                self._aggregation_buffers[metric_point.name].append(metric_point)
                
        except Exception as e:
            logger.error(f"Failed to record metric {metric_point.name}: {e}")
    
    def record_counter(self, name: str, value: Union[int, float] = 1, 
                      tags: Dict[str, str] = None):
        """カウンターメトリクス記録"""
        metric_point = MetricPoint(
            name=name,
            value=value,
            tags=tags or {}
        )
        asyncio.create_task(self.record_metric(metric_point))
    
    def record_gauge(self, name: str, value: Union[int, float], 
                    tags: Dict[str, str] = None):
        """ゲージメトリクス記録"""
        metric_point = MetricPoint(
            name=name,
            value=value,
            tags=tags or {}
        )
        asyncio.create_task(self.record_metric(metric_point))
    
    async def get_metric_values(self, metric_name: str, 
                              start_time: float, end_time: float,
                              aggregation: str = "avg") -> List[Tuple[float, float]]:
        """メトリクス値取得"""
        try:
            values = []
            
            # バッファから取得（簡略実装）
            buffer = self._aggregation_buffers.get(metric_name, deque())
            
            for point in buffer:
                if start_time <= point.timestamp <= end_time:
                    values.append((point.timestamp, point.value))
            
            if not values:
                return []
            
            # 集約処理
            if aggregation == "avg":
                avg_value = statistics.mean([v[1] for v in values])
                return [(start_time, avg_value)]
            elif aggregation == "sum":
                sum_value = sum([v[1] for v in values])
                return [(start_time, sum_value)]
            elif aggregation == "max":
                max_value = max([v[1] for v in values])
                return [(start_time, max_value)]
            elif aggregation == "min":
                min_value = min([v[1] for v in values])
                return [(start_time, min_value)]
            else:
                return values
                
        except Exception as e:
            logger.error(f"Failed to get metric values for {metric_name}: {e}")
            return []
    
    async def _collect_system_metrics(self):
        """システムメトリクス収集"""
        while self.status == SystemStatus.RUNNING:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent()
                await self.record_metric(MetricPoint(
                    name="system_cpu_percent",
                    value=cpu_percent,
                    tags={"type": "system"}
                ))
                
                # メモリ使用量
                memory = psutil.virtual_memory()
                await self.record_metric(MetricPoint(
                    name="system_memory_percent",
                    value=memory.percent,
                    tags={"type": "system"}
                ))
                
                # ディスク使用量
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                await self.record_metric(MetricPoint(
                    name="system_disk_percent",
                    value=disk_percent,
                    tags={"type": "system"}
                ))
                
                # ネットワーク統計
                network = psutil.net_io_counters()
                await self.record_metric(MetricPoint(
                    name="system_network_bytes_sent",
                    value=network.bytes_sent,
                    tags={"type": "system"}
                ))
                await self.record_metric(MetricPoint(
                    name="system_network_bytes_recv",
                    value=network.bytes_recv,
                    tags={"type": "system"}
                ))
                
                await asyncio.sleep(10)  # 10秒間隔
                
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_alerts(self):
        """アラート監視"""
        while self.status == SystemStatus.RUNNING:
            try:
                current_time = time.time()
                
                for rule_name, rule in self.alert_rules.items():
                    if not rule.enabled:
                        continue
                    
                    alert_state = self.alert_states[rule_name]
                    
                    # メトリクス値取得
                    end_time = current_time
                    start_time = current_time - rule.duration_seconds
                    
                    metric_values = await self.get_metric_values(
                        rule.metric_name, start_time, end_time, "avg"
                    )
                    
                    if metric_values:
                        current_value = metric_values[0][1]
                        
                        # 条件チェック
                        triggered = self._check_alert_condition(
                            rule.condition, current_value, rule.threshold
                        )
                        
                        if triggered and not alert_state['triggered']:
                            # アラート発火
                            alert_state['triggered'] = True
                            alert_state['trigger_count'] += 1
                            alert_state['first_trigger'] = current_time
                            
                            await self._fire_alert(rule, current_value)
                            
                        elif not triggered and alert_state['triggered']:
                            # アラート解除
                            alert_state['triggered'] = False
                            await self._resolve_alert(rule, current_value)
                    
                    alert_state['last_check'] = current_time
                
                await asyncio.sleep(30)  # 30秒間隔
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(30)
    
    def _check_alert_condition(self, condition: str, value: float, threshold: float) -> bool:
        """アラート条件チェック"""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        else:
            return False
    
    async def _fire_alert(self, rule: AlertRule, current_value: float):
        """アラート発火"""
        message = (f"Alert triggered: {rule.name}\n"
                  f"Metric: {rule.metric_name}\n"
                  f"Current Value: {current_value}\n"
                  f"Threshold: {rule.threshold}\n"
                  f"Condition: {rule.condition}")
        
        logger.warning(message)
        
        # 実際の実装では通知チャンネルに送信
        await self._emit_event('alert_fired', {
            'rule_name': rule.name,
            'metric_name': rule.metric_name,
            'current_value': current_value,
            'threshold': rule.threshold,
            'severity': rule.severity.value
        })
    
    async def _resolve_alert(self, rule: AlertRule, current_value: float):
        """アラート解除"""
        message = (f"Alert resolved: {rule.name}\n"
                  f"Metric: {rule.metric_name}\n"
                  f"Current Value: {current_value}")
        
        logger.info(message)
        
        await self._emit_event('alert_resolved', {
            'rule_name': rule.name,
            'metric_name': rule.metric_name,
            'current_value': current_value
        })
    
    async def health_check(self) -> HealthStatus:
        """健全性チェック"""
        try:
            if self.status != SystemStatus.RUNNING:
                return HealthStatus.UNHEALTHY
            
            # ストレージ健全性チェック
            storage_health = await self.metric_storage.health_check()
            
            # 収集タスク状態チェック
            failed_tasks = 0
            for task in self._collection_tasks.values():
                if task.done() and task.exception():
                    failed_tasks += 1
            
            if failed_tasks > 0:
                return HealthStatus.DEGRADED
            
            return storage_health
            
        except Exception:
            return HealthStatus.UNHEALTHY


# ファクトリ関数
def create_config_manager(config: BaseConfig) -> UnifiedConfigManager:
    """統合設定管理ファクトリ"""
    return UnifiedConfigManager("unified_config", config)


def create_metrics_collector(config: BaseConfig) -> UnifiedMetricsCollector:
    """統合メトリクス収集ファクトリ"""
    return UnifiedMetricsCollector("unified_metrics", config)