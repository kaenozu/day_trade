#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合監視システム

システム全体の監視・アラート・メトリクス収集を統合管理
"""

import asyncio
import time
import threading
import logging
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
from pathlib import Path

from ..utils.memory_optimizer import get_memory_optimizer, MemoryStats
from ..utils.error_handler import get_error_handler, ErrorSeverity
from ..utils.enhanced_logging import get_enhanced_logger


class AlertSeverity(Enum):
    """アラート重要度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringTarget(Enum):
    """監視対象"""
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    BUSINESS = "business"


@dataclass
class MetricData:
    """メトリクスデータ"""
    timestamp: datetime
    name: str
    value: float
    unit: str
    tags: Dict[str, str]
    target: MonitoringTarget


@dataclass
class Alert:
    """アラート情報"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    target: MonitoringTarget
    title: str
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MetricCollector:
    """メトリクス収集器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[MetricData] = []
        self.collection_interval = 30  # seconds
        self.max_history_size = 10000
        self.lock = threading.RLock()
    
    def collect_system_metrics(self) -> List[MetricData]:
        """システムメトリクス収集"""
        timestamp = datetime.now()
        metrics = []
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricData(
                timestamp=timestamp,
                name="cpu_usage_percent",
                value=cpu_percent,
                unit="percent",
                tags={"type": "system"},
                target=MonitoringTarget.SYSTEM
            ))
            
            # メモリ使用率
            memory = psutil.virtual_memory()
            metrics.append(MetricData(
                timestamp=timestamp,
                name="memory_usage_percent",
                value=memory.percent,
                unit="percent",
                tags={"type": "system"},
                target=MonitoringTarget.SYSTEM
            ))
            
            # ディスク使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(MetricData(
                timestamp=timestamp,
                name="disk_usage_percent",
                value=disk_percent,
                unit="percent",
                tags={"type": "system"},
                target=MonitoringTarget.SYSTEM
            ))
            
            # ネットワークI/O
            network = psutil.net_io_counters()
            if network:
                metrics.extend([
                    MetricData(
                        timestamp=timestamp,
                        name="network_bytes_sent",
                        value=network.bytes_sent,
                        unit="bytes",
                        tags={"type": "network", "direction": "sent"},
                        target=MonitoringTarget.NETWORK
                    ),
                    MetricData(
                        timestamp=timestamp,
                        name="network_bytes_recv",
                        value=network.bytes_recv,
                        unit="bytes",
                        tags={"type": "network", "direction": "received"},
                        target=MonitoringTarget.NETWORK
                    )
                ])
            
        except Exception as e:
            self.logger.error(f"システムメトリクス収集エラー: {e}")
        
        return metrics
    
    def collect_application_metrics(self) -> List[MetricData]:
        """アプリケーションメトリクス収集"""
        timestamp = datetime.now()
        metrics = []
        
        try:
            # メモリ最適化システムから統計取得
            memory_optimizer = get_memory_optimizer()
            memory_stats = memory_optimizer.get_memory_stats()
            
            metrics.extend([
                MetricData(
                    timestamp=timestamp,
                    name="app_memory_usage_mb",
                    value=memory_stats.process_mb,
                    unit="megabytes",
                    tags={"type": "application", "component": "process"},
                    target=MonitoringTarget.APPLICATION
                ),
                MetricData(
                    timestamp=timestamp,
                    name="app_memory_usage_percent",
                    value=memory_stats.usage_percent,
                    unit="percent",
                    tags={"type": "application", "component": "system"},
                    target=MonitoringTarget.APPLICATION
                )
            ])
            
            # エラーハンドラーから統計取得
            error_handler = get_error_handler()
            error_stats = error_handler.get_error_stats()
            
            metrics.append(MetricData(
                timestamp=timestamp,
                name="error_count_total",
                value=error_stats["total_errors"],
                unit="count",
                tags={"type": "application", "component": "errors"},
                target=MonitoringTarget.APPLICATION
            ))
            
            # ログシステムから統計取得
            enhanced_logger = get_enhanced_logger()
            log_stats = enhanced_logger.get_performance_stats()
            
            if log_stats.get("total_logs_1h") is not None:
                metrics.append(MetricData(
                    timestamp=timestamp,
                    name="log_count_1h",
                    value=log_stats["total_logs_1h"],
                    unit="count",
                    tags={"type": "application", "component": "logging"},
                    target=MonitoringTarget.APPLICATION
                ))
            
        except Exception as e:
            self.logger.error(f"アプリケーションメトリクス収集エラー: {e}")
        
        return metrics
    
    def store_metrics(self, metrics: List[MetricData]):
        """メトリクスを保存"""
        with self.lock:
            self.metrics_history.extend(metrics)
            
            # 履歴サイズ制限
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size//2:]
    
    def get_metrics(self, 
                   target: Optional[MonitoringTarget] = None,
                   hours_back: int = 1) -> List[MetricData]:
        """メトリクス取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self.lock:
            filtered_metrics = [
                m for m in self.metrics_history
                if m.timestamp >= cutoff_time
            ]
            
            if target:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if m.target == target
                ]
            
            return filtered_metrics


class AlertManager:
    """アラート管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.thresholds = {
            "cpu_usage_percent": 80.0,
            "memory_usage_percent": 85.0,
            "disk_usage_percent": 90.0,
            "error_rate_per_hour": 10.0,
            "response_time_ms": 5000.0
        }
        self.lock = threading.RLock()
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """アラートハンドラーを追加"""
        self.alert_handlers.append(handler)
    
    def create_alert(self, 
                    severity: AlertSeverity,
                    target: MonitoringTarget,
                    title: str,
                    message: str,
                    details: Optional[Dict[str, Any]] = None) -> str:
        """アラートを作成"""
        alert_id = f"{target.value}_{int(time.time())}_{hash(message) % 10000}"
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            target=target,
            title=title,
            message=message,
            details=details or {}
        )
        
        with self.lock:
            self.alerts[alert_id] = alert
        
        # アラートハンドラーに通知
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"アラートハンドラーエラー: {e}")
        
        self.logger.log(
            self._severity_to_log_level(severity),
            f"アラート作成: [{alert_id}] {title} - {message}",
            extra={"alert_id": alert_id, "severity": severity.value}
        )
        
        return alert_id
    
    def resolve_alert(self, alert_id: str):
        """アラートを解決"""
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolution_time = datetime.now()
                self.logger.info(f"アラート解決: [{alert_id}]")
    
    def check_metric_thresholds(self, metrics: List[MetricData]):
        """メトリクスのしきい値チェック"""
        for metric in metrics:
            threshold = self.thresholds.get(metric.name)
            if threshold and metric.value > threshold:
                self.create_alert(
                    severity=AlertSeverity.WARNING,
                    target=metric.target,
                    title=f"{metric.name} 閾値超過",
                    message=f"{metric.name}が閾値({threshold})を超過: {metric.value:.2f}{metric.unit}",
                    details={
                        "metric_name": metric.name,
                        "current_value": metric.value,
                        "threshold": threshold,
                        "unit": metric.unit,
                        "tags": metric.tags
                    }
                )
    
    def get_active_alerts(self, 
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """アクティブなアラートを取得"""
        with self.lock:
            alerts = [alert for alert in self.alerts.values() if not alert.resolved]
            
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def _severity_to_log_level(self, severity: AlertSeverity) -> int:
        """アラート重要度をログレベルに変換"""
        mapping = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }
        return mapping.get(severity, logging.INFO)


class IntegratedMonitoringSystem:
    """統合監視システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # デフォルトアラートハンドラーを追加
        self.alert_manager.add_alert_handler(self._default_alert_handler)
        
        self.logger.info("統合監視システム初期化完了")
    
    def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            self.logger.warning("監視は既に開始されています")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="IntegratedMonitoring"
        )
        self.monitor_thread.start()
        
        self.logger.info("統合監視開始")
    
    def stop_monitoring(self):
        """監視停止"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("統合監視停止")
    
    def _monitoring_loop(self):
        """監視ループ"""
        while self.monitoring_active:
            try:
                # システムメトリクス収集
                system_metrics = self.metric_collector.collect_system_metrics()
                
                # アプリケーションメトリクス収集
                app_metrics = self.metric_collector.collect_application_metrics()
                
                # 全メトリクスを統合
                all_metrics = system_metrics + app_metrics
                
                # メトリクス保存
                self.metric_collector.store_metrics(all_metrics)
                
                # しきい値チェック
                self.alert_manager.check_metric_thresholds(all_metrics)
                
                # 収集間隔待機
                time.sleep(self.metric_collector.collection_interval)
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(30)  # エラー時は30秒待機
    
    def _default_alert_handler(self, alert: Alert):
        """デフォルトアラートハンドラー"""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }[alert.severity]
        
        self.logger.log(
            log_level,
            f"アラート: {alert.title} - {alert.message}",
            extra={
                "alert_id": alert.id,
                "alert_severity": alert.severity.value,
                "alert_target": alert.target.value,
                "alert_details": alert.details
            }
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        # 最新のメトリクス取得
        latest_metrics = self.metric_collector.get_metrics(hours_back=0.1)  # 6分以内
        
        # アクティブアラート取得
        active_alerts = self.alert_manager.get_active_alerts()
        
        # メトリクスを種類別に分類
        metric_summary = {}
        for metric in latest_metrics:
            if metric.name not in metric_summary:
                metric_summary[metric.name] = []
            metric_summary[metric.name].append({
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat(),
                "tags": metric.tags
            })
        
        return {
            "monitoring_active": self.monitoring_active,
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": metric_summary,
            "active_alerts": {
                "total": len(active_alerts),
                "by_severity": {
                    severity.value: len([a for a in active_alerts if a.severity == severity])
                    for severity in AlertSeverity
                }
            },
            "system_health": self._calculate_system_health(latest_metrics, active_alerts)
        }
    
    def _calculate_system_health(self, 
                               metrics: List[MetricData], 
                               alerts: List[Alert]) -> str:
        """システムヘルス計算"""
        # 重要なアラートがある場合
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            return "CRITICAL"
        
        error_alerts = [a for a in alerts if a.severity == AlertSeverity.ERROR]
        if error_alerts:
            return "DEGRADED"
        
        warning_alerts = [a for a in alerts if a.severity == AlertSeverity.WARNING]
        if len(warning_alerts) > 3:
            return "DEGRADED"
        
        # メトリクス基準
        recent_metrics = {m.name: m.value for m in metrics}
        
        cpu_usage = recent_metrics.get("cpu_usage_percent", 0)
        memory_usage = recent_metrics.get("memory_usage_percent", 0)
        
        if cpu_usage > 90 or memory_usage > 90:
            return "DEGRADED"
        elif cpu_usage > 70 or memory_usage > 70:
            return "WARNING"
        
        return "HEALTHY"
    
    def export_metrics(self, format_type: str = "json") -> str:
        """メトリクスをエクスポート"""
        metrics = self.metric_collector.get_metrics(hours_back=24)
        
        if format_type == "json":
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "metrics_count": len(metrics),
                "metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "name": m.name,
                        "value": m.value,
                        "unit": m.unit,
                        "tags": m.tags,
                        "target": m.target.value
                    }
                    for m in metrics
                ]
            }
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        
        return f"Unsupported format: {format_type}"


# グローバル監視システムインスタンス
_monitoring_system: Optional[IntegratedMonitoringSystem] = None


def get_monitoring_system() -> IntegratedMonitoringSystem:
    """グローバル監視システムインスタンスを取得"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = IntegratedMonitoringSystem()
    return _monitoring_system


def start_system_monitoring():
    """システム監視を開始"""
    system = get_monitoring_system()
    system.start_monitoring()


def stop_system_monitoring():
    """システム監視を停止"""
    system = get_monitoring_system()
    system.stop_monitoring()


def get_system_status() -> Dict[str, Any]:
    """システム状態を取得"""
    system = get_monitoring_system()
    return system.get_system_status()


def create_custom_alert(severity: AlertSeverity, 
                       target: MonitoringTarget,
                       title: str, 
                       message: str,
                       details: Optional[Dict[str, Any]] = None) -> str:
    """カスタムアラートを作成"""
    system = get_monitoring_system()
    return system.alert_manager.create_alert(
        severity, target, title, message, details
    )