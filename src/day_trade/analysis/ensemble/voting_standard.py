"""
標準的なアンサンブル投票手法
ソフト投票、ハード投票、重み付け平均投票を実装
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.logging_config import get_context_logger
from ..signals import TradingSignal
from .performance import PerformanceManager
from .types import EnsembleStrategy, EnsembleVotingType
from .voting_base import VotingBase

logger = get_context_logger(__name__, component="ensemble_voting_standard")


class StandardVotingMethods(VotingBase):
    """標準的な投票手法を実装するクラス"""

    def __init__(
        self,
        voting_type: EnsembleVotingType,
        ensemble_strategy: EnsembleStrategy,
        performance_manager: PerformanceManager,
        strategy_weights: Dict[str, float],
    ):
        super().__init__(voting_type, ensemble_strategy, performance_manager, strategy_weights)

    def soft_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """ソフト投票（信頼度による重み付け投票）"""
        try:
            voting_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
            total_weight = 0.0
            strategy_contributions = {}

            # 各戦略の信頼度を重み付けして投票
            for strategy_name, signal in strategy_signals:
                base_weight = self.strategy_weights.get(strategy_name, 0.2)
                strategy_weight = self.calculate_performance_weight(
                    strategy_name, base_weight
                )

                weighted_confidence = signal.confidence * strategy_weight
                voting_scores[signal.signal_type.value] += weighted_confidence
                total_weight += strategy_weight
                strategy_contributions[strategy_name] = weighted_confidence

            if total_weight == 0:
                return None

            # 投票スコアを正規化
            voting_scores = self.normalize_voting_scores(voting_scores, total_weight)

            # 最高スコアのシグナルタイプを決定
            best_signal_type, best_score = self.get_best_signal_type(voting_scores)

            # 理由をまとめる
            reasons = self.collect_reasons(strategy_signals, best_signal_type)

            # アンサンブルシグナルを作成
            latest_signal = strategy_signals[0][1]
            ensemble_signal = self.create_ensemble_signal(
                best_signal_type, best_score, reasons, latest_signal
            )

            # アンサンブル不確実性を計算
            ensemble_uncertainty = self.calculate_ensemble_uncertainty(voting_scores)

            return ensemble_signal, strategy_contributions, best_score, ensemble_uncertainty

        except Exception as e:
            logger.error(f"ソフト投票エラー: {e}")
            return None

    def hard_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """ハード投票（多数決投票）"""
        try:
            vote_counts = {"buy": 0, "sell": 0, "hold": 0}
            strategy_contributions = {}

            # 各戦略の投票を集計（パフォーマンスの低い戦略は除外）
            for strategy_name, signal in strategy_signals:
                if strategy_name in self.performance_manager.strategy_performance:
                    perf = self.performance_manager.strategy_performance[strategy_name]
                    if perf.success_rate < 0.3:  # 成功率が低い戦略は投票権を減らす
                        continue

                vote_counts[signal.signal_type.value] += 1
                strategy_contributions[strategy_name] = 1.0

            if sum(vote_counts.values()) == 0:
                return None

            # 最多得票のシグナルタイプを決定
            best_signal_type = max(vote_counts, key=vote_counts.get)
            vote_count = vote_counts[best_signal_type]

            # 過半数を取得した場合のみ有効
            total_votes = sum(vote_counts.values())
            if vote_count / total_votes < 0.5:
                best_signal_type = "hold"

            # 信頼度は参加戦略の平均信頼度
            ensemble_confidence = self.get_average_confidence(
                strategy_signals, best_signal_type
            )

            # 理由をまとめる
            reasons = [f"多数決投票: {vote_count}/{total_votes} 票獲得"]

            # アンサンブルシグナルを作成
            latest_signal = strategy_signals[0][1]
            ensemble_signal = self.create_ensemble_signal(
                best_signal_type, ensemble_confidence, reasons, latest_signal
            )

            # アンサンブル不確実性を計算
            vote_percentages = [count / total_votes for count in vote_counts.values()]
            ensemble_uncertainty = 1.0 - max(vote_percentages)  # 最大得票率の逆数

            return (
                ensemble_signal,
                strategy_contributions,
                ensemble_confidence,
                ensemble_uncertainty,
            )

        except Exception as e:
            logger.error(f"ハード投票エラー: {e}")
            return None

    def weighted_average_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """重み付け平均投票（ソフト投票の変種として実装）"""
        return self.soft_voting(
            strategy_signals, meta_features, ml_predictions, market_regime
        )