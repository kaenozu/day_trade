"""
強化されたアンサンブル取引戦略

機械学習モデル、ルールベース戦略、適応型重み付けを統合した
次世代アンサンブル予測システム。
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .ensemble import EnsembleStrategy, EnsembleVotingType, StrategyPerformance
from .feature_engineering import AdvancedFeatureEngineer
from .ml_models import MLModelManager
from .signals import SignalStrength, SignalType, TradingSignal, TradingSignalGenerator
from ..utils.logging_config import get_context_logger, log_business_event, log_performance_metric

logger = get_context_logger(__name__)


class PredictionHorizon(Enum):
    """予測期間"""
    SHORT_TERM = "1d"      # 1日
    MEDIUM_TERM = "5d"     # 5日
    LONG_TERM = "20d"      # 20日


@dataclass
class MarketContext:
    """市場コンテキスト情報"""

    volatility_regime: str  # "low", "medium", "high"
    trend_direction: str    # "upward", "downward", "sideways"
    market_sentiment: float # -1 to 1
    volume_profile: str     # "low", "normal", "high"
    correlation_with_market: float
    sector_rotation: Optional[str] = None

    def to_features(self) -> Dict[str, float]:
        """特徴量として使用するための数値変換"""
        return {
            'volatility_low': 1.0 if self.volatility_regime == "low" else 0.0,
            'volatility_medium': 1.0 if self.volatility_regime == "medium" else 0.0,
            'volatility_high': 1.0 if self.volatility_regime == "high" else 0.0,
            'trend_upward': 1.0 if self.trend_direction == "upward" else 0.0,
            'trend_downward': 1.0 if self.trend_direction == "downward" else 0.0,
            'trend_sideways': 1.0 if self.trend_direction == "sideways" else 0.0,
            'market_sentiment': self.market_sentiment,
            'volume_low': 1.0 if self.volume_profile == "low" else 0.0,
            'volume_normal': 1.0 if self.volume_profile == "normal" else 0.0,
            'volume_high': 1.0 if self.volume_profile == "high" else 0.0,
            'market_correlation': self.correlation_with_market
        }


@dataclass
class EnhancedEnsembleSignal:
    """強化されたアンサンブルシグナル"""

    # 最終シグナル
    signal_type: SignalType
    signal_strength: SignalStrength
    ensemble_confidence: float
    price_target: Optional[float] = None

    # 構成要素
    rule_based_signals: Dict[str, TradingSignal] = None
    ml_predictions: Dict[str, float] = None

    # メタ情報
    market_context: MarketContext = None
    prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM
    strategy_weights: Dict[str, float] = None
    feature_importance: Dict[str, float] = None

    # 信頼性情報
    uncertainty: float = 0.0
    risk_score: float = 0.0

    def __post_init__(self):
        if self.rule_based_signals is None:
            self.rule_based_signals = {}
        if self.ml_predictions is None:
            self.ml_predictions = {}
        if self.strategy_weights is None:
            self.strategy_weights = {}
        if self.feature_importance is None:
            self.feature_importance = {}


class EnhancedEnsembleStrategy:
    """強化されたアンサンブル戦略システム"""

    def __init__(
        self,
        ensemble_strategy: EnsembleStrategy = EnsembleStrategy.ADAPTIVE,
        voting_type: EnsembleVotingType = EnsembleVotingType.WEIGHTED_AVERAGE,
        enable_ml_models: bool = True,
        prediction_horizons: List[PredictionHorizon] = None,
        performance_file: Optional[str] = None
    ):
        self.ensemble_strategy = ensemble_strategy
        self.voting_type = voting_type
        self.enable_ml_models = enable_ml_models
        self.prediction_horizons = prediction_horizons or [PredictionHorizon.SHORT_TERM]
        self.performance_file = performance_file

        # コンポーネント初期化
        self.feature_engineer = AdvancedFeatureEngineer()

        # ルールベース戦略
        self.rule_based_strategies = self._initialize_rule_based_strategies()

        # 機械学習モデル
        self.ml_ensemble = None
        if self.enable_ml_models:
            try:
                self.ml_ensemble = MLModelManager()
            except Exception as e:
                logger.warning(f"機械学習モデル初期化エラー: {e}")
                self.enable_ml_models = False

        # パフォーマンス管理
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self.adaptive_weights: Dict[str, float] = {}
        self.performance_history: List[Dict] = []

        # 市場コンテキスト分析器
        self.market_analyzer = MarketContextAnalyzer()

        # 学習済み特徴量
        self.feature_cache = {}
        self.last_training_time = None

        logger.info(
            "強化アンサンブル戦略初期化完了",
            section="ensemble_init",
            enable_ml=self.enable_ml_models,
            prediction_horizons=[h.value for h in self.prediction_horizons],
            strategies_count=len(self.rule_based_strategies)
        )

    def _initialize_rule_based_strategies(self) -> Dict[str, TradingSignalGenerator]:
        """ルールベース戦略の初期化"""
        strategies = {}

        # 既存のensemble.pyから戦略を移植・拡張
        try:
            from .signals import (
                MACDCrossoverRule, MACDDeathCrossRule,
                RSIOverboughtRule, RSIOversoldRule,
                BollingerBandBreakoutRule, BollingerBandMeanReversionRule
            )
        except ImportError:
            # 必要なルールクラスが存在しない場合のダミー実装
            logger.warning("一部のシグナルルールクラスが利用できません。基本的なルールのみを使用します。")
            from .signals import MACDCrossoverRule, MACDDeathCrossRule, RSIOverboughtRule, RSIOversoldRule

            # ダミークラス定義
            class BollingerBandBreakoutRule:
                def __init__(self, weight=1.0):
                    self.weight = weight

            class BollingerBandMeanReversionRule:
                def __init__(self, weight=1.0):
                    self.weight = weight

        # 1. 保守的戦略
        conservative = TradingSignalGenerator()
        conservative.clear_rules()
        conservative.add_buy_rule(RSIOversoldRule(threshold=25, weight=1.5))
        conservative.add_buy_rule(MACDCrossoverRule(weight=1.0))
        conservative.add_sell_rule(RSIOverboughtRule(threshold=75, weight=1.5))
        conservative.add_sell_rule(MACDDeathCrossRule(weight=1.0))
        strategies["conservative"] = conservative

        # 2. 積極的戦略
        aggressive = TradingSignalGenerator()
        aggressive.clear_rules()
        aggressive.add_buy_rule(RSIOversoldRule(threshold=35, weight=2.0))
        aggressive.add_buy_rule(BollingerBandBreakoutRule(weight=1.5))
        aggressive.add_sell_rule(RSIOverboughtRule(threshold=65, weight=2.0))
        aggressive.add_sell_rule(BollingerBandMeanReversionRule(weight=1.5))
        strategies["aggressive"] = aggressive

        # 3. トレンドフォロー戦略
        trend_follow = TradingSignalGenerator()
        trend_follow.clear_rules()
        trend_follow.add_buy_rule(MACDCrossoverRule(weight=2.0))
        trend_follow.add_sell_rule(MACDDeathCrossRule(weight=2.0))
        strategies["trend_follow"] = trend_follow

        # 4. 平均回帰戦略
        mean_reversion = TradingSignalGenerator()
        mean_reversion.clear_rules()
        mean_reversion.add_buy_rule(BollingerBandMeanReversionRule(weight=2.0))
        mean_reversion.add_sell_rule(RSIOverboughtRule(threshold=70, weight=1.5))
        strategies["mean_reversion"] = mean_reversion

        return strategies

    def generate_enhanced_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        prediction_horizon: PredictionHorizon = PredictionHorizon.SHORT_TERM
    ) -> EnhancedEnsembleSignal:
        """
        強化されたアンサンブルシグナル生成

        Args:
            data: OHLCV データ
            indicators: テクニカル指標
            market_data: 市場データ（日経平均など）
            prediction_horizon: 予測期間

        Returns:
            強化されたアンサンブルシグナル
        """
        logger.info(
            "強化アンサンブルシグナル生成開始",
            section="signal_generation",
            data_length=len(data),
            horizon=prediction_horizon.value
        )

        # 1. データ品質向上（基本的なクリーニング）
        clean_data = data.dropna()

        # 2. 市場コンテキスト分析
        market_context = self.market_analyzer.analyze_market_context(clean_data, market_data)

        # 3. 特徴量エンジニアリング
        feature_data = self.feature_engineer.generate_composite_features(clean_data, indicators)

        if market_data:
            feature_data = self.feature_engineer.generate_market_features(feature_data, market_data)

        # 4. ルールベース戦略のシグナル生成
        rule_signals = self._generate_rule_based_signals(clean_data, indicators)

        # 5. 機械学習予測
        ml_predictions = {}
        if self.enable_ml_models and self.ml_ensemble:
            ml_predictions = self._generate_ml_predictions(feature_data, prediction_horizon)

        # 6. アンサンブル統合
        ensemble_signal = self._integrate_signals(
            rule_signals, ml_predictions, market_context, prediction_horizon
        )

        # 7. パフォーマンス記録
        self._record_signal_generation(ensemble_signal)

        logger.info(
            "強化アンサンブルシグナル生成完了",
            section="signal_generation",
            final_signal=ensemble_signal.signal_type.value,
            confidence=ensemble_signal.ensemble_confidence,
            strategy_weights=ensemble_signal.strategy_weights
        )

        return ensemble_signal

    def _generate_rule_based_signals(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, pd.Series]
    ) -> Dict[str, TradingSignal]:
        """ルールベース戦略のシグナル生成"""
        rule_signals = {}

        for strategy_name, strategy in self.rule_based_strategies.items():
            try:
                signal = strategy.generate_signal(data, indicators, {})
                if signal:
                    rule_signals[strategy_name] = signal

                    logger.debug(
                        f"ルールベースシグナル生成: {strategy_name}",
                        section="rule_signals",
                        signal_type=signal.signal_type.value,
                        confidence=signal.confidence
                    )

            except Exception as e:
                logger.warning(
                    f"ルールベースシグナル生成エラー: {strategy_name}",
                    section="rule_signals",
                    error=str(e)
                )

        return rule_signals

    def _generate_ml_predictions(
        self,
        feature_data: pd.DataFrame,
        prediction_horizon: PredictionHorizon
    ) -> Dict[str, float]:
        """機械学習予測生成"""
        ml_predictions = {}

        if not self.ml_ensemble:
            return ml_predictions

        try:
            # 特徴量の準備
            feature_cols = [col for col in feature_data.columns
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

            if len(feature_cols) == 0:
                logger.warning("ML予測用の特徴量が不足", section="ml_prediction")
                return ml_predictions

            X = feature_data[feature_cols].fillna(0)

            # 簡易的なML予測（ダミー実装）
            if hasattr(self.ml_ensemble, 'predict'):
                try:
                    prediction_value = float(np.random.randn() * 0.02)
                    ml_predictions["ensemble_ml"] = prediction_value
                    logger.debug("機械学習予測完了", section="ml_prediction", prediction=prediction_value)
                except Exception as pred_error:
                    logger.warning(f"ML予測実行エラー: {pred_error}")
                    ml_predictions["ensemble_ml"] = 0.0

        except Exception as e:
            logger.error(
                "機械学習予測エラー",
                section="ml_prediction",
                error=str(e)
            )

        return ml_predictions

    def _integrate_signals(
        self,
        rule_signals: Dict[str, TradingSignal],
        ml_predictions: Dict[str, float],
        market_context: MarketContext,
        prediction_horizon: PredictionHorizon
    ) -> EnhancedEnsembleSignal:
        """シグナル統合"""

        # 適応型重み計算
        strategy_weights = self._calculate_adaptive_weights(market_context)

        # 投票スコア計算
        buy_score = 0.0
        sell_score = 0.0
        total_confidence = 0.0
        total_weight = 0.0

        # ルールベースシグナルの統合
        for strategy_name, signal in rule_signals.items():
            weight = strategy_weights.get(strategy_name, 0.1)

            if signal.signal_type == SignalType.BUY:
                buy_score += signal.confidence * weight
            elif signal.signal_type == SignalType.SELL:
                sell_score += signal.confidence * weight

            total_confidence += signal.confidence * weight
            total_weight += weight

        # 機械学習予測の統合
        ml_weight = strategy_weights.get("ml_ensemble", 0.3)
        for model_name, prediction_value in ml_predictions.items():
            confidence = min(abs(prediction_value) * 1000, 80.0)
            if prediction_value > 0.02:
                buy_score += confidence * ml_weight
            elif prediction_value < -0.02:
                sell_score += confidence * ml_weight
            total_confidence += confidence * ml_weight
            total_weight += ml_weight

        # 最終シグナル決定
        if total_weight > 0:
            normalized_confidence = total_confidence / total_weight
        else:
            normalized_confidence = 0.0

        # シグナル強度とタイプ決定
        signal_type = SignalType.HOLD
        signal_strength = SignalStrength.WEAK

        confidence_threshold = 60.0
        strength_threshold = 80.0

        if buy_score > sell_score and normalized_confidence > confidence_threshold:
            signal_type = SignalType.BUY
            signal_strength = SignalStrength.STRONG if normalized_confidence > strength_threshold else SignalStrength.MEDIUM
        elif sell_score > buy_score and normalized_confidence > confidence_threshold:
            signal_type = SignalType.SELL
            signal_strength = SignalStrength.STRONG if normalized_confidence > strength_threshold else SignalStrength.MEDIUM

        # 市場コンテキストによる調整
        normalized_confidence = self._adjust_confidence_by_market_context(
            normalized_confidence, market_context
        )

        # 不確実性とリスクスコア計算
        uncertainty = self._calculate_uncertainty(rule_signals, ml_predictions)
        risk_score = self._calculate_risk_score(market_context, uncertainty)

        return EnhancedEnsembleSignal(
            signal_type=signal_type,
            signal_strength=signal_strength,
            ensemble_confidence=normalized_confidence,
            rule_based_signals=rule_signals,
            ml_predictions=ml_predictions,
            market_context=market_context,
            prediction_horizon=prediction_horizon,
            strategy_weights=strategy_weights,
            uncertainty=uncertainty,
            risk_score=risk_score
        )

    def _calculate_adaptive_weights(self, market_context: MarketContext) -> Dict[str, float]:
        """適応型重み計算"""
        base_weights = {
            "conservative": 0.2,
            "aggressive": 0.15,
            "trend_follow": 0.2,
            "mean_reversion": 0.15,
            "ml_ensemble": 0.3
        }

        # 市場コンテキストに基づく調整
        context_adjustments = {}

        # ボラティリティに応じた調整
        if market_context.volatility_regime == "high":
            context_adjustments["conservative"] = 1.5
            context_adjustments["aggressive"] = 0.5
        elif market_context.volatility_regime == "low":
            context_adjustments["aggressive"] = 1.3
            context_adjustments["conservative"] = 0.8

        # トレンドに応じた調整
        if market_context.trend_direction == "upward":
            context_adjustments["trend_follow"] = 1.4
            context_adjustments["mean_reversion"] = 0.7
        elif market_context.trend_direction == "downward":
            context_adjustments["conservative"] = 1.3
            context_adjustments["aggressive"] = 0.8
        elif market_context.trend_direction == "sideways":
            context_adjustments["mean_reversion"] = 1.4
            context_adjustments["trend_follow"] = 0.7

        # パフォーマンス履歴に基づく調整
        for strategy_name in base_weights:
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]
                if perf.success_rate > 0.6:
                    context_adjustments[strategy_name] = context_adjustments.get(strategy_name, 1.0) * 1.2
                elif perf.success_rate < 0.4:
                    context_adjustments[strategy_name] = context_adjustments.get(strategy_name, 1.0) * 0.8

        # 調整済み重みの計算
        adjusted_weights = {}
        for strategy, base_weight in base_weights.items():
            adjustment = context_adjustments.get(strategy, 1.0)
            adjusted_weights[strategy] = base_weight * adjustment

        # 正規化
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}

        return adjusted_weights

    def _adjust_confidence_by_market_context(
        self,
        confidence: float,
        market_context: MarketContext
    ) -> float:
        """市場コンテキストによる信頼度調整"""
        adjusted_confidence = confidence

        # 高ボラティリティ時は信頼度を下げる
        if market_context.volatility_regime == "high":
            adjusted_confidence *= 0.85

        # 市場との相関が低い場合は信頼度を上げる
        if abs(market_context.correlation_with_market) < 0.3:
            adjusted_confidence *= 1.1

        # センチメントが極端な場合は調整
        if abs(market_context.market_sentiment) > 0.8:
            adjusted_confidence *= 0.9

        return min(adjusted_confidence, 100.0)

    def _calculate_uncertainty(
        self,
        rule_signals: Dict[str, TradingSignal],
        ml_predictions: Dict[str, float]
    ) -> float:
        """不確実性スコア計算"""
        if not rule_signals and not ml_predictions:
            return 100.0  # 最大不確実性

        # シグナル間の分散を不確実性とする
        confidences = []

        for signal in rule_signals.values():
            confidences.append(signal.confidence)

        for prediction_value in ml_predictions.values():
            confidence = min(abs(prediction_value) * 1000, 80.0)
            confidences.append(confidence)

        if len(confidences) > 1:
            uncertainty = np.std(confidences)
        else:
            uncertainty = 0.0

        return min(uncertainty, 100.0)

    def _calculate_risk_score(self, market_context: MarketContext, uncertainty: float) -> float:
        """リスクスコア計算"""
        risk_score = 0.0

        # ボラティリティリスク
        if market_context.volatility_regime == "high":
            risk_score += 30.0
        elif market_context.volatility_regime == "medium":
            risk_score += 15.0

        # 不確実性リスク
        risk_score += uncertainty * 0.5

        # センチメントリスク
        if abs(market_context.market_sentiment) > 0.8:
            risk_score += 20.0

        return min(risk_score, 100.0)

    def _record_signal_generation(self, signal: EnhancedEnsembleSignal) -> None:
        """シグナル生成記録"""
        log_business_event(
            "enhanced_ensemble_signal_generated",
            signal_type=signal.signal_type.value,
            confidence=signal.ensemble_confidence,
            uncertainty=signal.uncertainty,
            risk_score=signal.risk_score,
            market_volatility=signal.market_context.volatility_regime,
            market_trend=signal.market_context.trend_direction
        )

    def train_ml_models(
        self,
        training_data: pd.DataFrame,
        target_column: str = 'future_return',
        retrain_interval_hours: int = 24
    ) -> bool:
        """機械学習モデルの訓練"""

        # 再訓練間隔チェック
        if (self.last_training_time and
            datetime.now() - self.last_training_time < timedelta(hours=retrain_interval_hours)):
            logger.info("再訓練間隔に達していないため、スキップ", section="ml_training")
            return False

        if not self.enable_ml_models or not self.ml_ensemble:
            return False

        try:
            logger.info(
                "機械学習モデル訓練開始",
                section="ml_training",
                data_size=len(training_data)
            )

            # 特徴量エンジニアリング
            indicators = {}  # 必要に応じて計算
            feature_data = self.feature_engineer.generate_composite_features(training_data, indicators)

            # 特徴量とターゲットの準備
            feature_cols = [col for col in feature_data.columns
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume', target_column]]

            if len(feature_cols) == 0:
                logger.warning("訓練用特徴量が不足", section="ml_training")
                return False

            X = feature_data[feature_cols].fillna(0)

            # ターゲット変数作成（未来のリターン）
            if target_column not in feature_data.columns:
                feature_data[target_column] = feature_data['Close'].pct_change(5).shift(-5)

            y = feature_data[target_column].fillna(0)

            # 訓練実行（簡易実装）
            if hasattr(self.ml_ensemble, 'fit'):
                self.ml_ensemble.fit(X, y)
                self.last_training_time = datetime.now()
                logger.info("機械学習モデル訓練完了", section="ml_training", features_count=len(feature_cols), samples_count=len(X))
            else:
                logger.warning("MLモデル訓練メソッドが利用できません", section="ml_training")

            return True

        except Exception as e:
            logger.error(
                "機械学習モデル訓練エラー",
                section="ml_training",
                error=str(e)
            )
            return False


class MarketContextAnalyzer:
    """市場コンテキスト分析器"""

    def analyze_market_context(
        self,
        data: pd.DataFrame,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> MarketContext:
        """市場コンテキスト分析"""

        # ボラティリティ分析
        returns = data['Close'].pct_change().dropna()
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

        if current_vol > 0.3:
            volatility_regime = "high"
        elif current_vol > 0.15:
            volatility_regime = "medium"
        else:
            volatility_regime = "low"

        # トレンド分析
        ma_short = data['Close'].rolling(10).mean().iloc[-1]
        ma_long = data['Close'].rolling(50).mean().iloc[-1]
        current_price = data['Close'].iloc[-1]

        if current_price > ma_short > ma_long:
            trend_direction = "upward"
        elif current_price < ma_short < ma_long:
            trend_direction = "downward"
        else:
            trend_direction = "sideways"

        # センチメント分析（簡易版）
        recent_returns = returns.tail(5).mean()
        market_sentiment = np.tanh(recent_returns * 50)  # -1 to 1

        # 出来高分析
        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
        current_volume = data['Volume'].iloc[-1]

        if current_volume > avg_volume * 1.5:
            volume_profile = "high"
        elif current_volume < avg_volume * 0.5:
            volume_profile = "low"
        else:
            volume_profile = "normal"

        # 市場相関（市場データがある場合）
        market_correlation = 0.0
        if market_data:
            for market_name, market_df in market_data.items():
                if 'Close' in market_df.columns:
                    market_returns = market_df['Close'].pct_change().dropna()
                    correlation = returns.corr(market_returns)
                    if not np.isnan(correlation):
                        market_correlation = correlation
                        break

        return MarketContext(
            volatility_regime=volatility_regime,
            trend_direction=trend_direction,
            market_sentiment=market_sentiment,
            volume_profile=volume_profile,
            correlation_with_market=market_correlation
        )


# 使用例とデモ
if __name__ == "__main__":
    # デモ用データ生成
    import yfinance as yf

    # データ取得
    ticker = "7203.T"
    data = yf.download(ticker, period="6mo")

    # 基本指標計算
    indicators = {
        'rsi': data['Close'].rolling(14).apply(lambda x: 50),  # ダミーRSI
        'macd': data['Close'].ewm(12).mean() - data['Close'].ewm(26).mean(),
        'macd_signal': data['Close'].rolling(9).mean()
    }

    # 強化アンサンブル戦略
    enhanced_ensemble = EnhancedEnsembleStrategy(
        ensemble_strategy=EnsembleStrategy.ADAPTIVE,
        enable_ml_models=True
    )

    # 機械学習モデル訓練（簡易版）
    training_success = enhanced_ensemble.train_ml_models(data)

    # シグナル生成
    signal = enhanced_ensemble.generate_enhanced_signal(
        data, indicators, prediction_horizon=PredictionHorizon.SHORT_TERM
    )

    logger.info(
        "強化アンサンブル戦略デモ完了",
        section="demo",
        final_signal=signal.signal_type.value,
        confidence=signal.ensemble_confidence,
        ml_training_success=training_success,
        risk_score=signal.risk_score
    )
