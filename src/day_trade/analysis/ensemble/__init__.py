"""
アンサンブル戦略パッケージ

元のensemble.pyファイルを機能別にモジュール分割しています。
バックワード互換性のために、すべての主要クラスをここから公開します。
"""

# 基本型とEnum定義
from .types import (
    EnsembleSignal,
    EnsembleStrategy, 
    EnsembleVotingType,
)

# 戦略パフォーマンス管理
from .performance import (
    PerformanceManager,
    StrategyPerformance,
)

# 投票システム
from .voting import EnsembleVotingSystem
from .voting_base import VotingBase
from .voting_standard import StandardVotingMethods
from .voting_advanced import AdvancedVotingMethods

# メタ特徴量計算
from .meta_features import MetaFeatureCalculator

# 市場レジーム検出
from .market_regime import MarketRegimeDetector

# 機械学習統合
from .ml_integration import MLIntegrationManager
from .ml_models import MLModelManager
from .ml_prediction import MLPredictionGenerator
from .ml_training import MLTrainingManager

# メインアンサンブル戦略
from .strategy import EnsembleTradingStrategy

# バックワード互換性のために元のクラス名もエクスポート
__all__ = [
    # 主要クラス（新しい実装）
    "EnsembleTradingStrategy",
    
    # 型定義
    "EnsembleSignal",
    "EnsembleStrategy", 
    "EnsembleVotingType",
    "StrategyPerformance",
    
    # コンポーネントクラス
    "EnsembleVotingSystem",
    "VotingBase",
    "StandardVotingMethods", 
    "AdvancedVotingMethods",
    "MetaFeatureCalculator", 
    "MarketRegimeDetector",
    "MLIntegrationManager",
    "MLModelManager",
    "MLPredictionGenerator",
    "MLTrainingManager",
    "PerformanceManager",
]

# バージョン情報
__version__ = "2.0.0"  # モジュール分割版