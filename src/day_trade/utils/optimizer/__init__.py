#!/usr/bin/env python3
"""
Performance Optimizer Package

パフォーマンス最適化のためのユーティリティパッケージです。
元のperformance_optimizer.pyファイルから分割されたモジュール群を提供し、
後方互換性を維持します。

モジュール構成:
- metrics: パフォーマンスメトリクス定義
- profiler: パフォーマンスプロファイラー
- data_optimizer: データ取得最適化
- database_optimizer: データベース操作最適化
- calculation_optimizer: 計算処理最適化
- monitor: パフォーマンスモニター
- utils: ユーティリティ関数

使用例:
    from day_trade.utils.optimizer import (
        PerformanceProfiler,
        DataFetchOptimizer,
        performance_monitor
    )

    profiler = PerformanceProfiler()
    optimizer = DataFetchOptimizer()

    with performance_monitor("処理名"):
        # 処理内容
        pass
"""

# 主要クラスとデータクラスのインポート
from .calculation_optimizer import CalculationOptimizer
from .data_optimizer import DataFetchOptimizer
from .database_optimizer import DatabaseOptimizer
from .metrics import PerformanceMetrics
from .monitor import performance_monitor
from .profiler import PerformanceProfiler
from .utils import create_sample_data

# 後方互換性のためのエクスポート
__all__ = [
    "PerformanceMetrics",
    "PerformanceProfiler",
    "DataFetchOptimizer",
    "DatabaseOptimizer",
    "CalculationOptimizer",
    "performance_monitor",
    "create_sample_data",
]
