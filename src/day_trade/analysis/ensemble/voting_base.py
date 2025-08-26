"""
アンサンブル投票システムの基盤クラス
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.logging_config import get_context_logger
from ..signals import SignalStrength, SignalType, TradingSignal
from .performance import PerformanceManager
from .types import EnsembleStrategy, EnsembleVotingType

logger = get_context_logger(__name__, component="ensemble_voting_base")


class VotingBase:
    """アンサンブル投票システムの基盤クラス"""

    def __init__(
        self,
        voting_type: EnsembleVotingType,
        ensemble_strategy: EnsembleStrategy,
        performance_manager: PerformanceManager,
        strategy_weights: Dict[str, float],
    ):
        """
        Args:
            voting_type: 投票方式
            ensemble_strategy: アンサンブル戦略
            performance_manager: パフォーマンス管理
            strategy_weights: 戦略重み
        """
        self.voting_type = voting_type
        self.ensemble_strategy = ensemble_strategy
        self.performance_manager = performance_manager
        self.strategy_weights = strategy_weights

    def get_confidence_threshold(self) -> float:
        """信頼度閾値を取得"""
        if self.ensemble_strategy == EnsembleStrategy.CONSERVATIVE:
            return 60.0
        elif self.ensemble_strategy == EnsembleStrategy.AGGRESSIVE:
            return 30.0
        elif self.ensemble_strategy == EnsembleStrategy.BALANCED:
            return 45.0
        elif self.ensemble_strategy == EnsembleStrategy.ML_OPTIMIZED:
            return 35.0  # ML使用時は閾値を低く
        elif self.ensemble_strategy == EnsembleStrategy.REGIME_ADAPTIVE:
            # 市場レジームに基づく動的閾値
            if hasattr(self, 'current_market_regime'):
                if self.current_market_regime == "high_volatility":
                    return 65.0
                elif self.current_market_regime in ["uptrend", "downtrend"]:
                    return 40.0
                else:
                    return 50.0
            return 50.0
        else:  # ADAPTIVE
            # 過去のパフォーマンスに基づいて動的調整
            perf_summary = self.performance_manager.get_performance_summary()
            avg_success_rate = perf_summary.get('avg_success_rate', 0.0)
            return 30.0 + (70.0 - 30.0) * (1 - avg_success_rate)

    def create_ensemble_signal(
        self,
        signal_type: str,
        confidence: float,
        reasons: List[str],
        latest_signal: TradingSignal,
    ) -> TradingSignal:
        """アンサンブルシグナルを作成"""
        # 強度を決定
        if confidence >= 70:
            strength = SignalStrength.STRONG
        elif confidence >= 40:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK

        return TradingSignal(
            signal_type=SignalType(signal_type),
            strength=strength,
            confidence=confidence,
            reasons=reasons,
            conditions_met={},
            timestamp=latest_signal.timestamp,
            price=latest_signal.price,
            symbol=getattr(latest_signal, "symbol", None),
        )

    def calculate_performance_weight(self, strategy_name: str, base_weight: float) -> float:
        """パフォーマンスに基づく重み調整を計算"""
        if strategy_name in self.performance_manager.strategy_performance:
            perf = self.performance_manager.strategy_performance[strategy_name]
            performance_multiplier = 0.5 + perf.success_rate  # 0.5-1.5の範囲
            return base_weight * performance_multiplier
        return base_weight

    def normalize_voting_scores(
        self, voting_scores: Dict[str, float], total_weight: float
    ) -> Dict[str, float]:
        """投票スコアを正規化"""
        if total_weight == 0:
            return voting_scores
        
        for signal_type in voting_scores:
            voting_scores[signal_type] /= total_weight
        return voting_scores

    def get_best_signal_type(
        self, voting_scores: Dict[str, float]
    ) -> Tuple[str, float]:
        """最高スコアのシグナルタイプを取得"""
        best_signal_type = max(voting_scores, key=voting_scores.get)
        best_score = voting_scores[best_signal_type]
        
        # 閾値チェック
        confidence_threshold = self.get_confidence_threshold()
        if best_score < confidence_threshold:
            best_signal_type = "hold"
            best_score = 0.0
            
        return best_signal_type, best_score

    def collect_reasons(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        best_signal_type: str,
        prefix: str = ""
    ) -> List[str]:
        """理由を収集"""
        reasons = []
        for strategy_name, signal in strategy_signals:
            if signal.signal_type.value == best_signal_type:
                reasons.extend(
                    [f"{strategy_name}: {reason}" for reason in signal.reasons]
                )

        if not reasons and prefix:
            reasons = [f"{prefix}: {best_signal_type}"]
        elif not reasons:
            reasons = [f"アンサンブル投票結果: {best_signal_type}"]
            
        return reasons

    def calculate_ensemble_uncertainty(
        self, voting_scores: Dict[str, float]
    ) -> float:
        """アンサンブル不確実性を計算"""
        score_values = list(voting_scores.values())
        return np.std(score_values) if len(score_values) > 1 else 0.0

    def get_average_confidence(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        signal_type: str
    ) -> float:
        """指定されたシグナルタイプの平均信頼度を取得"""
        confidences = [
            signal.confidence
            for strategy_name, signal in strategy_signals
            if signal.signal_type.value == signal_type
        ]
        return np.mean(confidences) if confidences else 0.0