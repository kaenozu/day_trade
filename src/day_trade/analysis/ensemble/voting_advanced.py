"""
高度なアンサンブル投票手法
機械学習投票、スタッキング、動的投票を実装
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.logging_config import get_context_logger
from ..signals import TradingSignal
from .performance import PerformanceManager
from .types import EnsembleStrategy, EnsembleVotingType
from .voting_base import VotingBase
from .voting_standard import StandardVotingMethods

logger = get_context_logger(__name__, component="ensemble_voting_advanced")


class AdvancedVotingMethods(VotingBase):
    """高度な投票手法を実装するクラス"""

    def __init__(
        self,
        voting_type: EnsembleVotingType,
        ensemble_strategy: EnsembleStrategy,
        performance_manager: PerformanceManager,
        strategy_weights: Dict[str, float],
    ):
        super().__init__(voting_type, ensemble_strategy, performance_manager, strategy_weights)
        # 標準投票手法への参照
        self.standard_voting = StandardVotingMethods(
            voting_type, ensemble_strategy, performance_manager, strategy_weights
        )

    def ml_ensemble_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """機械学習アンサンブル投票"""
        try:
            if not ml_predictions:
                # MLがない場合はソフト投票にフォールバック
                return self.standard_voting.soft_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )

            # 戦略シグナルとML予測を統合
            combined_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
            total_weight = 0.0
            strategy_contributions = {}

            # 1. 戦略シグナルの処理（重み0.4）
            strategy_weight_factor = 0.4
            for strategy_name, signal in strategy_signals:
                base_weight = (
                    self.strategy_weights.get(strategy_name, 0.2)
                    * strategy_weight_factor
                )
                weighted_confidence = signal.confidence * base_weight
                combined_scores[signal.signal_type.value] += weighted_confidence
                total_weight += base_weight
                strategy_contributions[strategy_name] = weighted_confidence

            # 2. ML予測の処理（重み0.6）
            ml_weight_factor = 0.6
            ml_contribution = {}

            # リターン予測の処理
            if "return_predictor" in ml_predictions:
                return_pred = ml_predictions["return_predictor"]
                weight = ml_weight_factor * 0.4
                
                if return_pred > 0.01:  # 1%以上の上昇予測
                    combined_scores["buy"] += abs(return_pred) * 100 * weight
                elif return_pred < -0.01:  # 1%以上の下落予測
                    combined_scores["sell"] += abs(return_pred) * 100 * weight
                else:
                    combined_scores["hold"] += 20 * weight

                ml_contribution["return_predictor"] = abs(return_pred) * 100 * weight
                total_weight += weight

            # 方向性予測の処理
            if "direction_predictor" in ml_predictions:
                direction_pred = ml_predictions["direction_predictor"]
                weight = ml_weight_factor * 0.4
                
                if direction_pred > 0.6:  # 上昇確率が高い
                    combined_scores["buy"] += direction_pred * 100 * weight
                elif direction_pred < 0.4:  # 下落確率が高い
                    combined_scores["sell"] += (1 - direction_pred) * 100 * weight
                else:
                    combined_scores["hold"] += 20 * weight

                ml_contribution["direction_predictor"] = direction_pred * 100 * weight
                total_weight += weight

            # ボラティリティ予測の処理（リスク調整）
            if "volatility_predictor" in ml_predictions:
                vol_pred = ml_predictions["volatility_predictor"]
                weight = ml_weight_factor * 0.2
                
                if vol_pred > 0.3:  # 高ボラティリティ予測時はリスク回避
                    risk_adjustment = 0.7
                    for signal_type in combined_scores:
                        if signal_type != "hold":
                            combined_scores[signal_type] *= risk_adjustment
                            combined_scores["hold"] += combined_scores[signal_type] * 0.3

                ml_contribution["volatility_predictor"] = vol_pred * weight
                total_weight += weight

            if total_weight == 0:
                return None

            # 正規化
            combined_scores = self.normalize_voting_scores(combined_scores, total_weight)

            # 最高スコアの決定（ML使用時は閾値を下げる）
            best_signal_type = max(combined_scores, key=combined_scores.get)
            best_score = combined_scores[best_signal_type]
            
            confidence_threshold = self.get_confidence_threshold() * 0.8
            if best_score < confidence_threshold:
                best_signal_type = "hold"
                best_score = 0.0

            # 理由の統合
            reasons = [f"機械学習アンサンブル投票: {best_signal_type}"]
            strategy_reasons = self.collect_reasons(strategy_signals, best_signal_type)
            reasons.extend(strategy_reasons[:3])  # 最大3つの戦略理由

            # ML予測の理由を追加
            if ml_predictions:
                for model_name, pred in ml_predictions.items():
                    reasons.append(f"{model_name}: {pred:.3f}")

            # シグナル作成
            latest_signal = strategy_signals[0][1]
            ensemble_signal = self.create_ensemble_signal(
                best_signal_type, best_score, reasons, latest_signal
            )

            # 不確実性計算（ML予測の分散を考慮）
            base_uncertainty = self.calculate_ensemble_uncertainty(combined_scores)
            
            # ML予測の一致度を考慮
            ml_agreement = 0.0
            if len(ml_predictions) > 1:
                ml_values = list(ml_predictions.values())
                ml_agreement = 1.0 - np.std(ml_values) / (
                    np.mean(np.abs(ml_values)) + 1e-8
                )

            ensemble_uncertainty = base_uncertainty * (1.0 - ml_agreement * 0.5)

            # 貢献度を統合
            all_contributions = {**strategy_contributions, **ml_contribution}

            return ensemble_signal, all_contributions, best_score, ensemble_uncertainty

        except Exception as e:
            logger.error(f"機械学習アンサンブル投票エラー: {e}")
            return self.standard_voting.soft_voting(
                strategy_signals, meta_features, ml_predictions, market_regime
            )

    def stacking_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """スタッキング手法"""
        try:
            # まず基本的なアンサンブル予測を取得
            base_ensemble = self.standard_voting.soft_voting(
                strategy_signals, meta_features, ml_predictions, market_regime
            )
            if not base_ensemble:
                return None

            # メタ特徴量を構築
            meta_features_array = []

            # 各戦略の信頼度を特徴量として使用
            for _strategy_name, signal in strategy_signals:
                meta_features_array.append(signal.confidence)

            # ML予測を特徴量として追加
            if ml_predictions:
                for pred in ml_predictions.values():
                    meta_features_array.append(pred * 100)  # スケール調整

            # メタ特徴量を追加
            if meta_features:
                for key in [
                    "volatility",
                    "trend_strength", 
                    "rsi_level",
                    "volume_ratio",
                ]:
                    meta_features_array.append(meta_features.get(key, 0))

            # 十分な特徴量がある場合は調整を実行
            if len(meta_features_array) >= 5:
                # 単純な重み付け調整（実際のメタラーナーの代替）
                feature_variance = np.std(meta_features_array)
                confidence_adjustment = 0.5 + min(0.5, feature_variance / 10.0)
                
                adjusted_confidence = base_ensemble[2] * confidence_adjustment
                return (
                    base_ensemble[0],
                    base_ensemble[1],
                    adjusted_confidence,
                    base_ensemble[3],
                )

            return base_ensemble

        except Exception as e:
            logger.error(f"スタッキング投票エラー: {e}")
            return self.standard_voting.soft_voting(
                strategy_signals, meta_features, ml_predictions, market_regime
            )

    def dynamic_ensemble_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """動的アンサンブル投票"""
        try:
            # 市場状況に基づいて投票手法を動的に選択
            if market_regime in [
                "high_volatility",
                "high_vol_overbought", 
                "high_vol_oversold",
            ]:
                # 高ボラティリティ時は保守的にハード投票
                return self.standard_voting.hard_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif market_regime in ["uptrend", "downtrend"] and ml_predictions:
                # トレンド相場でML予測がある場合はML重視
                return self.ml_ensemble_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif market_regime == "sideways":
                # レンジ相場ではスタッキング手法
                return self.stacking_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            else:
                # その他の場合はソフト投票
                return self.standard_voting.soft_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )

        except Exception as e:
            logger.error(f"動的アンサンブル投票エラー: {e}")
            return self.standard_voting.soft_voting(
                strategy_signals, meta_features, ml_predictions, market_regime
            )