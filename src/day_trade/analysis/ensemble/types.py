"""
アンサンブル戦略の基本型とEnum定義
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..signals import TradingSignal


class EnsembleVotingType(Enum):
    """アンサンブル投票タイプ"""

    SOFT_VOTING = "soft"  # 信頼度による重み付け投票
    HARD_VOTING = "hard"  # 多数決投票
    WEIGHTED_AVERAGE = "weighted"  # 重み付け平均
    ML_ENSEMBLE = "ml_ensemble"  # 機械学習アンサンブル
    STACKING = "stacking"  # スタッキング手法
    DYNAMIC_ENSEMBLE = "dynamic"  # 動的アンサンブル


class EnsembleStrategy(Enum):
    """アンサンブル戦略タイプ"""

    CONSERVATIVE = "conservative"  # 保守的（合意重視）
    AGGRESSIVE = "aggressive"  # 積極的（機会重視）
    BALANCED = "balanced"  # バランス型
    ADAPTIVE = "adaptive"  # 適応型（パフォーマンス最適化）
    ML_OPTIMIZED = "ml_optimized"  # 機械学習最適化
    REGIME_ADAPTIVE = "regime_adaptive"  # 市場レジーム適応型


@dataclass
class EnsembleSignal:
    """アンサンブルシグナル"""

    ensemble_signal: TradingSignal
    strategy_signals: List[Tuple[str, TradingSignal]]  # (strategy_name, signal)
    voting_scores: Dict[str, float]  # 各戦略の投票スコア
    ensemble_confidence: float
    strategy_weights: Dict[str, float]  # 各戦略の重み
    voting_type: EnsembleVotingType
    meta_features: Dict[str, Any]  # メタ特徴量
    ml_predictions: Optional[Dict[str, float]] = None  # 機械学習予測結果
    feature_importance: Optional[Dict[str, float]] = None  # 特徴量重要度
    market_regime: Optional[str] = None  # 市場レジーム判定
    ensemble_uncertainty: Optional[float] = None  # アンサンブル不確実性