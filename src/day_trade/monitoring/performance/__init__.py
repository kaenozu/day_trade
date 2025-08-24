#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Monitoring Package - 性能監視システムパッケージ

MLモデル性能監視システムの統合パッケージ
元のmodel_performance_monitor.pyからの分割版で、バックワード互換性を提供
"""

# 主要クラスのインポート
from .monitor import ModelPerformanceMonitor, create_enhanced_performance_monitor
from .config import EnhancedPerformanceConfigManager
from .symbol_manager import DynamicSymbolManager
from .retraining_manager import GranularRetrainingManager
from .metrics_evaluator import MetricsEvaluator
from .alert_manager import AlertManager
from .database_manager import DatabaseManager
from .retraining_executor import RetrainingExecutor, MonitoringController

# 型定義のインポート
from .types import (
    PerformanceStatus,
    AlertLevel,
    RetrainingScope,
    PerformanceMetrics,
    PerformanceAlert,
    RetrainingTrigger,
    RetrainingResult,
)

# バックワード互換性のためのエイリアス
ModelPerformanceMonitor = ModelPerformanceMonitor
create_enhanced_performance_monitor = create_enhanced_performance_monitor

# レガシー関数（後方互換性）
async def test_model_performance_monitor():
    """モデル性能監視システムテスト
    
    元のmodel_performance_monitor.pyのtest関数と互換性を保つテスト関数
    """
    import asyncio
    import json
    
    print("=== ModelPerformanceMonitor テスト開始 ===")

    # 監視システム初期化
    monitor = ModelPerformanceMonitor()

    # 監視ステータス確認
    status = monitor.get_monitoring_status()
    print(f"監視ステータス: {json.dumps(status, indent=2, ensure_ascii=False)}")

    # 監視対象銘柄確認
    symbols = monitor.get_monitoring_symbols()
    print(f"監視対象銘柄: {symbols}")

    # 性能チェック実行
    print("\n性能チェック実行中...")
    await monitor.run_performance_check()

    # 短時間の監視実行
    print("\n短時間監視開始...")
    monitoring_task = asyncio.create_task(monitor.start_monitoring())

    # 10秒後に停止
    await asyncio.sleep(10)
    await monitor.stop_monitoring()

    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass

    print("\n=== テスト完了 ===")


# パッケージメタデータ
__version__ = "2.0.0"
__author__ = "Day Trade System Team"
__description__ = "MLモデル性能監視システム（モジュール分割版）"

# 公開API
__all__ = [
    # メインクラス
    'ModelPerformanceMonitor',
    'create_enhanced_performance_monitor',
    
    # 管理クラス
    'EnhancedPerformanceConfigManager',
    'DynamicSymbolManager',
    'GranularRetrainingManager',
    'MetricsEvaluator',
    'AlertManager',
    'DatabaseManager',
    'RetrainingExecutor',
    'MonitoringController',
    
    # データ型
    'PerformanceStatus',
    'AlertLevel',
    'RetrainingScope',
    'PerformanceMetrics',
    'PerformanceAlert',
    'RetrainingTrigger',
    'RetrainingResult',
    
    # ユーティリティ関数
    'test_model_performance_monitor',
]


def get_package_info():
    """パッケージ情報を取得
    
    Returns:
        パッケージの基本情報を含む辞書
    """
    return {
        'name': 'performance_monitoring',
        'version': __version__,
        'description': __description__,
        'modules': [
            'monitor',
            'config', 
            'symbol_manager',
            'retraining_manager',
            'metrics_evaluator',
            'alert_manager',
            'database_manager',
            'retraining_executor',
            'types'
        ],
        'migration_info': {
            'original_file': 'model_performance_monitor.py',
            'split_date': '2024-08-24',
            'backward_compatible': True,
            'key_changes': [
                '機能別モジュール分割',
                '300行以内の制限準拠',
                '循環依存の回避',
                'PEP8準拠のコード整理'
            ]
        }
    }


# 動的インポート時のメッセージ
import logging
logger = logging.getLogger(__name__)
logger.info(f"Performance Monitoring Package v{__version__} loaded - modularized from model_performance_monitor.py")