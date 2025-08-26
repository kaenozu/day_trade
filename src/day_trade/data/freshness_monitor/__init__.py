#!/usr/bin/env python3
"""
データ鮮度監視システム - バックワード互換性レイヤー
元のadvanced_data_freshness_monitor.pyとの互換性を保つためのインポート
"""

# 主要なクラスと関数をインポート
from .advanced_monitor import AdvancedDataFreshnessMonitor, create_advanced_freshness_monitor

# バックワード互換性のためのエイリアス
AdvancedFreshnessMonitor = AdvancedDataFreshnessMonitor

# エンティティ・データモデル
from .enums import (
    AlertSeverity,
    DataSourceType,
    FreshnessStatus,
    MonitoringLevel,
    RecoveryAction
)
from .models import (
    DataAlert,
    DataSourceConfig,
    DashboardData,
    FreshnessCheck,
    IntegrityCheck,
    MonitoringStats,
    SLAMetrics
)

# 個別コンポーネント（高度な使用のため）
from .alert_manager import AlertManager
from .dashboard import DashboardManager
from .database_operations import DatabaseOperations
from .freshness_checker import FreshnessChecker
from .integrity_checker import IntegrityChecker
from .recovery_manager import RecoveryManager
from .sla_metrics import SLAMetricsCalculator

# バックワード互換性のためのエイリアス（重複削除）
# AdvancedDataFreshnessMonitor = AdvancedFreshnessMonitor  # 不要

# パッケージメタデータ
__version__ = "1.0.0"
__author__ = "Day Trade System"
__description__ = "高度データ鮮度・整合性監視システム"

# 公開API
__all__ = [
    # メインクラス
    "AdvancedFreshnessMonitor",
    "AdvancedDataFreshnessMonitor",  # バックワード互換性
    "create_advanced_freshness_monitor",
    
    # エンティティ・データモデル
    "DataSourceConfig",
    "FreshnessCheck", 
    "IntegrityCheck",
    "SLAMetrics",
    "DataAlert",
    "MonitoringStats",
    "DashboardData",
    
    # 列挙型
    "DataSourceType",
    "MonitoringLevel",
    "AlertSeverity",
    "RecoveryAction",
    "FreshnessStatus",
    
    # 個別コンポーネント
    "DatabaseOperations",
    "FreshnessChecker",
    "IntegrityChecker",
    "AlertManager",
    "RecoveryManager",
    "SLAMetricsCalculator",
    "DashboardManager",
]


def get_system_info():
    """システム情報取得
    
    Returns:
        システム情報の辞書
    """
    return {
        "name": "Advanced Data Freshness Monitor",
        "version": __version__,
        "description": __description__,
        "components": [
            "FreshnessChecker - データ鮮度チェック",
            "IntegrityChecker - データ整合性チェック", 
            "AlertManager - アラート管理",
            "RecoveryManager - 自動回復処理",
            "SLAMetricsCalculator - SLA メトリクス計算",
            "DashboardManager - ダッシュボード・レポート生成",
            "DatabaseOperations - データベース操作",
        ],
        "features": [
            "リアルタイム鮮度監視",
            "多層整合性チェック",
            "インテリジェントアラート管理", 
            "自動回復アクション",
            "SLA追跡・レポート",
            "包括的ダッシュボード",
            "統計的異常検出",
            "トレンド分析",
        ],
    }