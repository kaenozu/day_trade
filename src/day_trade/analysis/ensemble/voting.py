"""
アンサンブル投票システム
分割されたモジュールを統合するメインクラス
"""

from typing import Any, Dict, List, Optional, Tuple

from ...utils.logging_config import get_context_logger
from ..signals import TradingSignal
from .performance import PerformanceManager
from .types import EnsembleStrategy, EnsembleVotingType
from .voting_advanced import AdvancedVotingMethods
from .voting_standard import StandardVotingMethods

logger = get_context_logger(__name__, component="ensemble_voting")


class EnsembleVotingSystem:
    """アンサンブル投票システムのメインクラス"""

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

        # 投票手法の実装クラス
        self.standard_voting = StandardVotingMethods(
            voting_type, ensemble_strategy, performance_manager, strategy_weights
        )
        self.advanced_voting = AdvancedVotingMethods(
            voting_type, ensemble_strategy, performance_manager, strategy_weights
        )

    def perform_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """アンサンブル投票を実行"""
        try:
            if self.voting_type == EnsembleVotingType.SOFT_VOTING:
                return self.standard_voting.soft_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif self.voting_type == EnsembleVotingType.HARD_VOTING:
                return self.standard_voting.hard_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif self.voting_type == EnsembleVotingType.ML_ENSEMBLE:
                return self.advanced_voting.ml_ensemble_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif self.voting_type == EnsembleVotingType.STACKING:
                return self.advanced_voting.stacking_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif self.voting_type == EnsembleVotingType.DYNAMIC_ENSEMBLE:
                return self.advanced_voting.dynamic_ensemble_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            else:  # WEIGHTED_AVERAGE
                return self.standard_voting.weighted_average_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )

        except Exception as e:
            logger.error(f"アンサンブル投票エラー: {e}")
            return None

    # 後方互換性のためのメソッド
    def _get_confidence_threshold(self) -> float:
        """信頼度閾値を取得（後方互換性）"""
        return self.standard_voting.get_confidence_threshold()

    def _create_ensemble_signal(self, signal_type, confidence, reasons, latest_signal):
        """アンサンブルシグナルを作成（後方互換性）"""
        return self.standard_voting.create_ensemble_signal(
            signal_type, confidence, reasons, latest_signal
        )

    def _soft_voting(self, strategy_signals, meta_features, ml_predictions=None, market_regime=None):
        """ソフト投票（後方互換性）"""
        return self.standard_voting.soft_voting(
            strategy_signals, meta_features, ml_predictions, market_regime
        )

    def _hard_voting(self, strategy_signals, meta_features, ml_predictions=None, market_regime=None):
        """ハード投票（後方互換性）"""
        return self.standard_voting.hard_voting(
            strategy_signals, meta_features, ml_predictions, market_regime
        )

    def _weighted_average_voting(self, strategy_signals, meta_features, ml_predictions=None, market_regime=None):
        """重み付け平均投票（後方互換性）"""
        return self.standard_voting.weighted_average_voting(
            strategy_signals, meta_features, ml_predictions, market_regime
        )

    def _ml_ensemble_voting(self, strategy_signals, meta_features, ml_predictions=None, market_regime=None):
        """機械学習投票（後方互換性）"""
        return self.advanced_voting.ml_ensemble_voting(
            strategy_signals, meta_features, ml_predictions, market_regime
        )

    def _stacking_voting(self, strategy_signals, meta_features, ml_predictions=None, market_regime=None):
        """スタッキング投票（後方互換性）"""
        return self.advanced_voting.stacking_voting(
            strategy_signals, meta_features, ml_predictions, market_regime
        )

    def _dynamic_ensemble_voting(self, strategy_signals, meta_features, ml_predictions=None, market_regime=None):
        """動的投票（後方互換性）"""
        return self.advanced_voting.dynamic_ensemble_voting(
            strategy_signals, meta_features, ml_predictions, market_regime
        )