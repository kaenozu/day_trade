"""
高度なバックテストエンジン - パッケージ初期化

分割されたモジュールからの主要クラスのエクスポートと
後方互換性の維持。
"""

# メインエンジン
from .engine import AdvancedBacktestEngine

# データ構造
from .data_structures import (
    TradingCosts,
    Position,
    PerformanceMetrics,
    PerformanceAnalyzer,
)

# 各種管理システム
from .order_management import OrderManager
from .position_management import PositionManager
from .event_handlers import EventHandler
from .risk_management import RiskManager
from .performance_calculator import PerformanceCalculator

# ウォークフォワード最適化
from .walk_forward_optimizer import WalkForwardOptimizer

# 後方互換性のため、元の名前でもアクセス可能にする
__all__ = [
    # メイン
    "AdvancedBacktestEngine",
    
    # データ構造
    "TradingCosts",
    "Position", 
    "PerformanceMetrics",
    "PerformanceAnalyzer",
    
    # 管理システム
    "OrderManager",
    "PositionManager", 
    "EventHandler",
    "RiskManager",
    "PerformanceCalculator",
    
    # 最適化
    "WalkForwardOptimizer",
]

# バージョン情報
__version__ = "2.0.0"
__author__ = "Day Trade Analysis System"
__description__ = "Advanced backtesting engine with modular architecture"