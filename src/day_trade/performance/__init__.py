#!/usr/bin/env python3
"""
パフォーマンス統合モジュール

全ての最適化機能を統合するメインモジュール
"""

import asyncio
import atexit
from typing import Dict, Any

from .lazy_imports import optimized_imports
from .optimized_cache import cache_manager
from .database_optimizer import get_db_manager
from .memory_optimizer import start_memory_monitoring, stop_memory_monitoring, get_memory_stats
from .async_optimizer import task_manager, hybrid_executor


class PerformanceManager:
    """パフォーマンス管理統合クラス"""

    def __init__(self):
        self.initialized = False
        self.db_managers = {}

    def initialize(self, config: Dict[str, Any] = None):
        """パフォーマンス最適化初期化"""
        if self.initialized:
            return

        print("🚀 パフォーマンス最適化初期化中...")

        # デフォルト設定
        if config is None:
            config = {
                'memory_monitoring': True,
                'cache_enabled': True,
                'async_optimization': True,
                'db_optimization': True
            }

        # メモリ監視開始
        if config.get('memory_monitoring', True):
            start_memory_monitoring()
            print("  ✅ メモリ監視開始")

        # キャッシュ初期化
        if config.get('cache_enabled', True):
            cache_manager.clear_all()  # 初期化時にクリア
            print("  ✅ キャッシュシステム初期化")

        # クリーンアップ登録
        atexit.register(self.cleanup)

        self.initialized = True
        print("🎯 パフォーマンス最適化完了")

    def get_db_manager(self, db_path: str):
        """データベースマネージャー取得"""
        if db_path not in self.db_managers:
            self.db_managers[db_path] = get_db_manager(db_path)
        return self.db_managers[db_path]

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        stats = {
            'timestamp': asyncio.get_event_loop().time(),
            'memory': get_memory_stats(),
            'cache': cache_manager.get_global_stats(),
            'async_tasks': task_manager.get_stats(),
        }

        # データベース統計
        db_stats = {}
        for db_path, manager in self.db_managers.items():
            db_stats[db_path] = manager.get_performance_stats()
        stats['databases'] = db_stats

        return stats

    def cleanup(self):
        """リソースクリーンアップ"""
        if not self.initialized:
            return

        print("パフォーマンス最適化クリーンアップ中...")

        # メモリ監視停止
        stop_memory_monitoring()

        # 非同期リソースクリーンアップ
        hybrid_executor.cleanup()

        # キャッシュクリア
        cache_manager.clear_all()

        print("✅ クリーンアップ完了")


# グローバルマネージャー
performance_manager = PerformanceManager()

# 便利な関数
def initialize_performance(config: Dict[str, Any] = None):
    """パフォーマンス最適化初期化"""
    performance_manager.initialize(config)

def get_performance_stats() -> Dict[str, Any]:
    """パフォーマンス統計取得"""
    return performance_manager.get_performance_stats()

def get_optimized_db(db_path: str):
    """最適化されたDB取得"""
    return performance_manager.get_db_manager(db_path)

# 自動初期化（インポート時）
def auto_initialize():
    """自動初期化"""
    import os
    if os.environ.get('DAY_TRADE_AUTO_OPTIMIZE', '1') == '1':
        initialize_performance()

# モジュールインポート時に自動実行
auto_initialize()
