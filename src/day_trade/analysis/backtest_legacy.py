"""
バックテスト機能（レガシー版）

後方互換性のため、元のbacktest.pyの内容を保持
新しい分割された実装への移行が完了するまでの暫定ファイル

⚠️ 非推奨: 新しいコードでは analysis.backtest パッケージを使用してください
"""

# 後方互換性のため、新しいパッケージから必要なクラスを再エクスポート
import warnings

from .backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestMode,
    BacktestResult,
    MonteCarloResult,
    OptimizationObjective,
    OptimizationResult,
    Position,
    Trade,
    WalkForwardResult,
)

# Deprecation警告を表示
warnings.warn(
    "backtest_legacy.py は非推奨です。analysis.backtest パッケージを使用してください。",
    DeprecationWarning,
    stacklevel=2,
)

# すべてのクラスを再エクスポート（後方互換性のため）
__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestMode",
    "BacktestResult",
    "MonteCarloResult",
    "OptimizationObjective",
    "OptimizationResult",
    "Position",
    "Trade",
    "WalkForwardResult",
]
