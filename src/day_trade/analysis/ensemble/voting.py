"""
アンサンブル投票システム
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.logging_config import get_context_logger
from ..signals import SignalStrength, SignalType, TradingSignal
from .performance import PerformanceManager
from .types import EnsembleStrategy, EnsembleVotingType

logger = get_context_logger(__name__, component="ensemble_voting")


class EnsembleVotingSystem:
    """アンサンブル投票システム"""

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
                return self._soft_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif self.voting_type == EnsembleVotingType.HARD_VOTING:
                return self._hard_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif self.voting_type == EnsembleVotingType.ML_ENSEMBLE:
                return self._ml_ensemble_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif self.voting_type == EnsembleVotingType.STACKING:
                return self._stacking_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif self.voting_type == EnsembleVotingType.DYNAMIC_ENSEMBLE:
                return self._dynamic_ensemble_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            else:  # WEIGHTED_AVERAGE
                return self._weighted_average_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )

        except Exception as e:
            logger.error(f"アンサンブル投票エラー: {e}")
            return None

    def _get_confidence_threshold(self) -> float:
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

    def _create_ensemble_signal(
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

    def _soft_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """ソフト投票（信頼度による重み付け投票）"""
        voting_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        total_weight = 0.0
        strategy_contributions = {}

        for strategy_name, signal in strategy_signals:
            strategy_weight = self.strategy_weights.get(strategy_name, 0.2)

            # パフォーマンスによる重み調整
            if strategy_name in self.performance_manager.strategy_performance:
                perf = self.performance_manager.strategy_performance[strategy_name]
                performance_multiplier = 0.5 + perf.success_rate  # 0.5-1.5の範囲
                strategy_weight *= performance_multiplier

            weighted_confidence = signal.confidence * strategy_weight
            voting_scores[signal.signal_type.value] += weighted_confidence
            total_weight += strategy_weight

            strategy_contributions[strategy_name] = weighted_confidence

        if total_weight == 0:
            return None

        # 正規化
        for signal_type in voting_scores:
            voting_scores[signal_type] /= total_weight

        # 最高スコアのシグナルタイプを決定
        best_signal_type = max(voting_scores, key=voting_scores.get)
        best_score = voting_scores[best_signal_type]

        # 閾値チェック
        confidence_threshold = self._get_confidence_threshold()
        if best_score < confidence_threshold:
            best_signal_type = "hold"
            best_score = 0.0

        # 理由をまとめる
        reasons = []
        for strategy_name, signal in strategy_signals:
            if signal.signal_type.value == best_signal_type:
                reasons.extend(
                    [f"{strategy_name}: {reason}" for reason in signal.reasons]
                )

        if not reasons:
            reasons = [f"アンサンブル投票結果: {best_signal_type}"]

        # 最新の価格とタイムスタンプ
        latest_signal = strategy_signals[0][1]

        ensemble_signal = self._create_ensemble_signal(
            best_signal_type, best_score, reasons, latest_signal
        )

        # アンサンブル不確実性を計算（投票スコアの分散）
        score_values = list(voting_scores.values())
        ensemble_uncertainty = np.std(score_values) if len(score_values) > 1 else 0.0

        return ensemble_signal, strategy_contributions, best_score, ensemble_uncertainty

    def _hard_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """ハード投票（多数決投票）"""
        vote_counts = {"buy": 0, "sell": 0, "hold": 0}
        strategy_contributions = {}

        for strategy_name, signal in strategy_signals:
            # パフォーマンスによる重み調整
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
        confidences = [
            signal.confidence
            for strategy_name, signal in strategy_signals
            if signal.signal_type.value == best_signal_type
        ]
        ensemble_confidence = np.mean(confidences) if confidences else 0.0

        # 理由をまとめる
        reasons = [f"多数決投票: {vote_count}/{total_votes} 票獲得"]

        # 最新の価格とタイムスタンプ
        latest_signal = strategy_signals[0][1]

        ensemble_signal = self._create_ensemble_signal(
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

    def _weighted_average_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """重み付け平均投票"""
        # ソフト投票の変種として実装
        return self._soft_voting(
            strategy_signals, meta_features, ml_predictions, market_regime
        )

    def _ml_ensemble_voting(
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
                return self._soft_voting(
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
                if return_pred > 0.01:  # 1%以上の上昇予測
                    combined_scores["buy"] += (
                        abs(return_pred) * 100 * ml_weight_factor * 0.4
                    )
                elif return_pred < -0.01:  # 1%以上の下落予測
                    combined_scores["sell"] += (
                        abs(return_pred) * 100 * ml_weight_factor * 0.4
                    )
                else:
                    combined_scores["hold"] += 20 * ml_weight_factor * 0.4

                ml_contribution["return_predictor"] = (
                    abs(return_pred) * 100 * ml_weight_factor * 0.4
                )
                total_weight += ml_weight_factor * 0.4

            # 方向性予測の処理
            if "direction_predictor" in ml_predictions:
                direction_pred = ml_predictions["direction_predictor"]
                if direction_pred > 0.6:  # 上昇確率が高い
                    combined_scores["buy"] += (
                        direction_pred * 100 * ml_weight_factor * 0.4
                    )
                elif direction_pred < 0.4:  # 下落確率が高い
                    combined_scores["sell"] += (
                        (1 - direction_pred) * 100 * ml_weight_factor * 0.4
                    )
                else:
                    combined_scores["hold"] += 20 * ml_weight_factor * 0.4

                ml_contribution["direction_predictor"] = (
                    direction_pred * 100 * ml_weight_factor * 0.4
                )
                total_weight += ml_weight_factor * 0.4

            # ボラティリティ予測の処理（リスク調整）
            if "volatility_predictor" in ml_predictions:
                vol_pred = ml_predictions["volatility_predictor"]
                if vol_pred > 0.3:  # 高ボラティリティ予測時はリスク回避
                    risk_adjustment = 0.7  # 信頼度を30%減らす
                    for signal_type in combined_scores:
                        if signal_type != "hold":
                            combined_scores[signal_type] *= risk_adjustment
                            combined_scores["hold"] += (
                                combined_scores[signal_type] * 0.3
                            )

                ml_contribution["volatility_predictor"] = (
                    vol_pred * ml_weight_factor * 0.2
                )
                total_weight += ml_weight_factor * 0.2

            if total_weight == 0:
                return None

            # 正規化
            for signal_type in combined_scores:
                combined_scores[signal_type] /= total_weight

            # 最高スコアの決定
            best_signal_type = max(combined_scores, key=combined_scores.get)
            best_score = combined_scores[best_signal_type]

            # 閾値チェック
            confidence_threshold = (
                self._get_confidence_threshold() * 0.8
            )  # ML使用時は閾値を下げる
            if best_score < confidence_threshold:
                best_signal_type = "hold"
                best_score = 0.0

            # 理由の統合
            reasons = [f"機械学習アンサンブル投票: {best_signal_type}"]
            for strategy_name, signal in strategy_signals:
                if signal.signal_type.value == best_signal_type:
                    reasons.extend(
                        [f"{strategy_name}: {reason}" for reason in signal.reasons[:2]]
                    )

            # ML予測の理由を追加
            if ml_predictions:
                for model_name, pred in ml_predictions.items():
                    reasons.append(f"{model_name}: {pred:.3f}")

            # シグナル作成
            latest_signal = strategy_signals[0][1]
            ensemble_signal = self._create_ensemble_signal(
                best_signal_type, best_score, reasons, latest_signal
            )

            # 不確実性計算（ML予測の分散を考慮）
            score_values = list(combined_scores.values())
            base_uncertainty = np.std(score_values) if len(score_values) > 1 else 0.0

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
            return self._soft_voting(
                strategy_signals, meta_features, ml_predictions, market_regime
            )

    def _stacking_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """スタッキング手法"""
        try:
            # まず基本的なアンサンブル予測を取得
            base_ensemble = self._soft_voting(
                strategy_signals, meta_features, ml_predictions, market_regime
            )
            if not base_ensemble:
                return None

            # メタラーナーが利用可能な場合の特徴量構築
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

            # 基本アンサンブルを返す
            return base_ensemble

        except Exception as e:
            logger.error(f"スタッキング投票エラー: {e}")
            return self._soft_voting(
                strategy_signals, meta_features, ml_predictions, market_regime
            )

    def _dynamic_ensemble_voting(
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
                return self._hard_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif market_regime in ["uptrend", "downtrend"] and ml_predictions:
                # トレンド相場でML予測がある場合はML重視
                return self._ml_ensemble_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            elif market_regime == "sideways":
                # レンジ相場ではスタッキング手法
                return self._stacking_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )
            else:
                # その他の場合はソフト投票
                return self._soft_voting(
                    strategy_signals, meta_features, ml_predictions, market_regime
                )

        except Exception as e:
            logger.error(f"動的アンサンブル投票エラー: {e}")
            return self._soft_voting(
                strategy_signals, meta_features, ml_predictions, market_regime
            )