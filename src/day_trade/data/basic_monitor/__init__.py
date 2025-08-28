#!/usr/bin/env python3
"""
Basic Monitor Package
基本監視システムパッケージ

後方互換性を保持するための公開API定義
"""

# 基本モデルとEnum
from .models import (
    AlertSeverity,
    AlertType,
    DataSourceHealth,
    MonitorAlert,
    MonitorRule,
    MonitorStatus,
    RecoveryAction,
    SLAMetrics,
)

# チェック機能
from .base_checks import MonitorCheck
from .consistency_checker import ConsistencyCheck
from .freshness_checker import FreshnessCheck

# コア監視システム
from .monitor_core import DataFreshnessMonitor

# ファクトリー関数
from .factory import create_data_freshness_monitor, test_data_freshness_monitor

# アラートハンドラー
from .alert_handler import AlertHandler

__all__ = [
    # Enums
    "MonitorStatus",
    "AlertSeverity",
    "AlertType", 
    "RecoveryAction",
    
    # Data classes
    "MonitorRule",
    "MonitorAlert",
    "DataSourceHealth",
    "SLAMetrics",
    
    # Check classes
    "MonitorCheck",
    "FreshnessCheck",
    "ConsistencyCheck",
    
    # Core system
    "DataFreshnessMonitor",
    "AlertHandler",
    
    # Factory functions
    "create_data_freshness_monitor",
    "test_data_freshness_monitor",
]

# 後方互換性のためのバージョン情報
__version__ = "1.0.0"