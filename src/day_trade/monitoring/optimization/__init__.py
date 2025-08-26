#!/usr/bin/env python3
"""
パフォーマンス最適化モジュール

分割されたパフォーマンス最適化コンポーネントの統合インターフェース
後方互換性を保ちつつ、モジュール化された機能を提供
"""

# 主要コンポーネントのインポート
from .performance_manager import (
    PerformanceManager,
    PerformanceOptimizer,
    performance_manager,
    initialize_performance,
    get_performance_stats,
    get_optimized_db,
    run_performance_optimization,
    cleanup_performance
)

# 各最適化モジュールの主要機能をエクスポート
from .lazy_imports import optimized_imports, import_manager
from .cache_manager import cache_manager, cached, get_cache_stats, clear_all_caches
from .database_optimizer import get_db_manager, cleanup_all_managers
from .memory_optimizer import (
    start_memory_monitoring,
    stop_memory_monitoring,
    get_memory_stats,
    force_cleanup,
    auto_cleanup,
    memory_profile
)
from .async_executor import (
    task_manager,
    hybrid_executor,
    async_cache,
    async_retry,
    async_timeout,
    run_parallel,
    async_map
)

# 後方互換性のためのエイリアス
# 元のPerformanceOptimizerクラスとして使用可能
PerformanceOptimizerLegacy = PerformanceOptimizer


def main():
    """メイン実行関数（元のファイルとの互換性）"""
    print("パフォーマンス最適化実行")
    print("=" * 50)
    
    # パフォーマンス最適化実行
    run_performance_optimization()
    
    # レポート生成
    report = performance_manager.get_optimization_report()
    print("\n" + report)
    
    print("\n" + "=" * 50)
    print("✅ パフォーマンス最適化完了")
    print("=" * 50)


# 自動初期化
initialize_performance()

# パッケージ情報
__version__ = "1.0.0"
__author__ = "Day Trade System"
__description__ = "Modular performance optimization system"

# パブリックAPI
__all__ = [
    # メイン管理クラス
    'PerformanceManager',
    'PerformanceOptimizer',
    'PerformanceOptimizerLegacy',
    'performance_manager',
    
    # 主要機能
    'initialize_performance',
    'get_performance_stats',
    'get_optimized_db',
    'run_performance_optimization',
    'cleanup_performance',
    
    # 遅延インポート
    'optimized_imports',
    'import_manager',
    
    # キャッシュ管理
    'cache_manager',
    'cached',
    'get_cache_stats',
    'clear_all_caches',
    
    # データベース最適化
    'get_db_manager',
    'cleanup_all_managers',
    
    # メモリ最適化
    'start_memory_monitoring',
    'stop_memory_monitoring',
    'get_memory_stats',
    'force_cleanup',
    'auto_cleanup',
    'memory_profile',
    
    # 非同期実行
    'task_manager',
    'hybrid_executor',
    'async_cache',
    'async_retry',
    'async_timeout',
    'run_parallel',
    'async_map',
    
    # ユーティリティ
    'main'
]