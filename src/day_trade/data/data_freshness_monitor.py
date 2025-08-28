#!/usr/bin/env python3
"""
データ鮮度・整合性監視システム (Refactored)
Issue #420: データ管理とデータ品質保証メカニズムの強化

このファイルは分割されたモジュールからの統合インポートを提供します。
実装は basic_monitor/ パッケージに移動されました。

リアルタイムデータ品質監視とアラート機能:
- データ鮮度監視
- 整合性チェック
- リアルタイムアラート
- SLA追跡
- ヘルスメトリクス
- 自動回復機能
- 監視ダッシュボード
"""

import asyncio
import warnings

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 分割されたモジュールからのインポート
from .basic_monitor import (
    AlertHandler,
    AlertSeverity,
    AlertType,
    ConsistencyCheck,
    DataFreshnessMonitor,
    DataSourceHealth,
    FreshnessCheck,
    MonitorAlert,
    MonitorCheck,
    MonitorRule,
    MonitorStatus,
    RecoveryAction,
    SLAMetrics,
    create_data_freshness_monitor,
    test_data_freshness_monitor,
)

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)

# 後方互換性のためのエクスポート
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


if __name__ == "__main__":
    # テスト実行（後方互換性のため）
    logger.info("基本監視システムテスト実行開始")
    asyncio.run(test_data_freshness_monitor())