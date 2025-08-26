#!/usr/bin/env python3
"""
高度データ鮮度・整合性監視システム
分割されたモジュールの統合インターフェースと後方互換性保証

このパッケージは、advanced_data_freshness_monitor.py を機能別にモジュール分割したものです。
元のAPIとの完全な後方互換性を提供します。
"""

# メインクラスと主要コンポーネント
from .monitor_core import AdvancedDataFreshnessMonitor, create_advanced_freshness_monitor

# データモデル
from .models import (
    AlertSeverity,
    DataAlert,
    DataSourceConfig,
    DataSourceState,
    DataSourceType,
    FreshnessCheck,
    FreshnessStatus,
    IntegrityCheck,
    MonitoringLevel,
    RecoveryAction,
    SLAMetrics,
)

# 管理クラス（高度な使用向け）
from .alert_manager import AlertManager
from .dashboard import DashboardManager
from .database import DatabaseManager
from .freshness_checker import FreshnessCheckManager
from .integrity_checker import IntegrityCheckManager
from .recovery_manager import RecoveryManager
from .sla_metrics import SLAMetricsManager

# 後方互換性のために元のクラス名でエクスポート
__all__ = [
    # メインクラス
    "AdvancedDataFreshnessMonitor",
    "create_advanced_freshness_monitor",
    
    # データモデル
    "DataSourceType",
    "MonitoringLevel", 
    "AlertSeverity",
    "RecoveryAction",
    "FreshnessStatus",
    "DataSourceConfig",
    "FreshnessCheck", 
    "IntegrityCheck",
    "SLAMetrics",
    "DataAlert",
    "DataSourceState",
    
    # 管理クラス（オプション）
    "DatabaseManager",
    "FreshnessCheckManager",
    "IntegrityCheckManager", 
    "AlertManager",
    "RecoveryManager",
    "SLAMetricsManager",
    "DashboardManager",
]

# バージョン情報
__version__ = "2.0.0"
__author__ = "Day Trade System"
__description__ = "高度データ鮮度・整合性監視システム（モジュール分割版）"

# モジュール情報
MODULES = {
    "monitor_core": "メインモニタリングシステムとデータソース管理",
    "models": "データモデル、列挙型、データクラス定義", 
    "database": "データベース操作と永続化",
    "freshness_checker": "データ鮮度チェックと品質評価",
    "integrity_checker": "整合性チェックと異常検出",
    "alert_manager": "アラート評価、生成、通知管理",
    "recovery_manager": "回復アクション評価と実行",
    "sla_metrics": "SLA計算、追跡、メトリクス管理",
    "dashboard": "監視ダッシュボードとレポート生成",
}

def get_module_info():
    """モジュール情報取得"""
    return {
        "version": __version__,
        "description": __description__, 
        "modules": MODULES,
        "total_modules": len(MODULES),
    }

def validate_installation():
    """インストール検証"""
    try:
        # 主要クラスのインポートテスト
        monitor = create_advanced_freshness_monitor()
        
        # 基本機能テスト
        test_config = DataSourceConfig(
            source_id="test",
            source_type=DataSourceType.API,
        )
        
        monitor.add_data_source(test_config)
        health = monitor.get_source_health("test")
        
        return {
            "status": "success",
            "message": "すべてのモジュールが正常にロードされました",
            "modules_loaded": list(MODULES.keys()),
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"インストール検証エラー: {e}",
            "modules_loaded": [],
        }

# 使用例とドキュメント
USAGE_EXAMPLES = {
    "basic_usage": """
# 基本的な使用方法
from day_trade.data.freshness_monitoring import (
    create_advanced_freshness_monitor,
    DataSourceConfig,
    DataSourceType,
)

# モニター作成
monitor = create_advanced_freshness_monitor()

# データソース追加
config = DataSourceConfig(
    source_id="stock_api",
    source_type=DataSourceType.API,
    endpoint_url="https://api.example.com/stocks",
    expected_frequency=30,
    freshness_threshold=120,
)
monitor.add_data_source(config)

# 監視開始
monitor.start_monitoring()
""",
    
    "advanced_usage": """  
# 高度な使用方法
import asyncio
from day_trade.data.freshness_monitoring import *

async def main():
    monitor = create_advanced_freshness_monitor()
    
    # アラートコールバック登録
    async def alert_handler(alert):
        print(f"アラート: {alert.message}")
    
    monitor.add_alert_callback(alert_handler)
    
    # ダッシュボードデータ取得
    dashboard = await monitor.get_monitoring_dashboard()
    print("システム概要:", dashboard["overview"])
    
asyncio.run(main())
""",
}

def get_usage_examples():
    """使用例取得"""
    return USAGE_EXAMPLES