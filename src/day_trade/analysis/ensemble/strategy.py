"""
メインアンサンブル戦略クラス
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger
from ..signals import TradingSignalGenerator
from .market_regime import MarketRegimeDetector
from .meta_features import MetaFeatureCalculator
from .ml_integration import MLIntegrationManager
from .performance import PerformanceManager
from .types import EnsembleSignal, EnsembleStrategy, EnsembleVotingType
from .voting import EnsembleVotingSystem

logger = get_context_logger(__name__, component="ensemble_strategy")


class EnsembleTradingStrategy:
    """アンサンブル取引戦略"""

    def __init__(
        self,
        ensemble_strategy: EnsembleStrategy = EnsembleStrategy.BALANCED,
        voting_type: EnsembleVotingType = EnsembleVotingType.SOFT_VOTING,
        performance_file: Optional[str] = None,
        enable_ml_models: bool = True,
        models_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            ensemble_strategy: アンサンブル戦略タイプ
            voting_type: 投票方式
            performance_file: パフォーマンス記録ファイル
            enable_ml_models: 機械学習モデルを有効にするか
            models_dir: 機械学習モデル保存ディレクトリ
        """
        self.ensemble_strategy = ensemble_strategy
        self.voting_type = voting_type
        self.performance_file = performance_file
        self.enable_ml_models = enable_ml_models

        # 個別戦略の初期化
        self.strategies = self._initialize_strategies()

        # 動的重み
        self.strategy_weights = self._initialize_weights()

        # コンポーネントの初期化
        self._initialize_components()

    def _initialize_components(self) -> None:
        """各コンポーネントを初期化"""
        # パフォーマンス管理
        self.performance_manager = PerformanceManager(self.performance_file)
        
        # メタ特徴量計算
        self.meta_feature_calculator = MetaFeatureCalculator()
        
        # 市場レジーム検出
        self.regime_detector = MarketRegimeDetector()
        
        # 機械学習統合
        self.ml_integration = MLIntegrationManager(self.enable_ml_models)
        
        # 投票システム
        self.voting_system = EnsembleVotingSystem(
            self.voting_type,
            self.ensemble_strategy,
            self.performance_manager,
            self.strategy_weights
        )

    def _initialize_strategies(self) -> Dict[str, TradingSignalGenerator]:
        """個別戦略を初期化"""
        strategies = {}

        # 1. 保守的RSI戦略
        conservative_strategy = TradingSignalGenerator()
        conservative_strategy.clear_rules()
        from ..signals import (
            MACDCrossoverRule,
            MACDDeathCrossRule,
            RSIOverboughtRule,
            RSIOversoldRule,
        )

        conservative_strategy.add_buy_rule(RSIOversoldRule(threshold=30, weight=1.5))
        conservative_strategy.add_buy_rule(MACDCrossoverRule(weight=2.0))
        conservative_strategy.add_sell_rule(RSIOverboughtRule(threshold=70, weight=1.5))
        conservative_strategy.add_sell_rule(MACDDeathCrossRule(weight=2.0))
        strategies["conservative_rsi"] = conservative_strategy

        # 2. 積極的モメンタム戦略
        momentum_strategy = TradingSignalGenerator()
        momentum_strategy.clear_rules()
        from ..signals import BollingerBandRule, PatternBreakoutRule, VolumeSpikeBuyRule

        momentum_strategy.add_buy_rule(BollingerBandRule(position="lower", weight=2.0))
        momentum_strategy.add_buy_rule(
            PatternBreakoutRule(direction="upward", weight=2.0)
        )
        momentum_strategy.add_buy_rule(VolumeSpikeBuyRule(weight=1.5))
        momentum_strategy.add_sell_rule(BollingerBandRule(position="upper", weight=2.0))
        momentum_strategy.add_sell_rule(
            PatternBreakoutRule(direction="downward", weight=2.0)
        )
        strategies["aggressive_momentum"] = momentum_strategy

        # 3. トレンドフォロー戦略
        trend_strategy = TradingSignalGenerator()
        trend_strategy.clear_rules()
        from ..signals import DeadCrossRule, GoldenCrossRule

        trend_strategy.add_buy_rule(GoldenCrossRule(weight=2.5))
        trend_strategy.add_buy_rule(MACDCrossoverRule(weight=1.5))
        trend_strategy.add_sell_rule(DeadCrossRule(weight=2.5))
        trend_strategy.add_sell_rule(MACDDeathCrossRule(weight=1.5))
        strategies["trend_following"] = trend_strategy

        # 4. 平均回帰戦略
        mean_reversion_strategy = TradingSignalGenerator()
        mean_reversion_strategy.clear_rules()
        mean_reversion_strategy.add_buy_rule(RSIOversoldRule(threshold=35, weight=2.5))
        mean_reversion_strategy.add_buy_rule(
            BollingerBandRule(position="lower", weight=2.0)
        )
        mean_reversion_strategy.add_sell_rule(
            RSIOverboughtRule(threshold=65, weight=2.5)
        )
        mean_reversion_strategy.add_sell_rule(
            BollingerBandRule(position="upper", weight=2.0)
        )
        strategies["mean_reversion"] = mean_reversion_strategy

        # 5. デフォルト統合戦略
        default_strategy = TradingSignalGenerator()  # 既存のデフォルトルール
        strategies["default_integrated"] = default_strategy

        return strategies

    def _initialize_weights(self) -> Dict[str, float]:
        """戦略の初期重みを設定"""
        if self.ensemble_strategy == EnsembleStrategy.CONSERVATIVE:
            return {
                "conservative_rsi": 0.3,
                "aggressive_momentum": 0.1,
                "trend_following": 0.2,
                "mean_reversion": 0.3,
                "default_integrated": 0.1,
            }
        elif self.ensemble_strategy == EnsembleStrategy.AGGRESSIVE:
            return {
                "conservative_rsi": 0.1,
                "aggressive_momentum": 0.35,
                "trend_following": 0.3,
                "mean_reversion": 0.15,
                "default_integrated": 0.1,
            }
        elif self.ensemble_strategy == EnsembleStrategy.BALANCED:
            return {
                "conservative_rsi": 0.2,
                "aggressive_momentum": 0.25,
                "trend_following": 0.25,
                "mean_reversion": 0.2,
                "default_integrated": 0.1,
            }
        elif self.ensemble_strategy == EnsembleStrategy.ML_OPTIMIZED:
            # 機械学習に重きを置いた設定
            return {
                "conservative_rsi": 0.15,
                "aggressive_momentum": 0.2,
                "trend_following": 0.2,
                "mean_reversion": 0.15,
                "default_integrated": 0.3,
            }
        elif self.ensemble_strategy == EnsembleStrategy.REGIME_ADAPTIVE:
            # 市場レジームに基づく動的重み
            return {name: 0.2 for name in self.strategies}
        else:  # ADAPTIVE
            # 初期は均等、パフォーマンスに基づいて動的調整
            return {
                "conservative_rsi": 0.2,
                "aggressive_momentum": 0.25,
                "trend_following": 0.25,
                "mean_reversion": 0.2,
                "default_integrated": 0.1,
            }

    def generate_ensemble_signal(
        self,
        df: pd.DataFrame,
        indicators: Optional[pd.DataFrame] = None,
        patterns: Optional[Dict] = None,
    ) -> Optional[EnsembleSignal]:
        """
        アンサンブルシグナルを生成

        Args:
            df: 価格データのDataFrame
            indicators: テクニカル指標のDataFrame
            patterns: チャートパターン認識結果

        Returns:
            EnsembleSignal or None
        """
        try:
            if isinstance(indicators, dict):
                indicators = pd.DataFrame(indicators)

            # 各戦略からシグナルを取得
            strategy_signals = []
            for strategy_name, strategy in self.strategies.items():
                try:
                    signal = strategy.generate_signal(df, indicators, patterns)
                    if signal:
                        strategy_signals.append((strategy_name, signal))
                except Exception as e:
                    logger.warning(f"戦略 {strategy_name} でエラー: {e}")

            if not strategy_signals:
                return None

            # メタ特徴量を計算
            meta_features = self.meta_feature_calculator.calculate_meta_features(
                df, indicators, patterns
            )

            # 機械学習予測の実行
            ml_predictions = None
            feature_importance = None
            if self.ml_integration.is_available():
                ml_predictions, feature_importance = (
                    self.ml_integration.generate_ml_predictions(df, indicators)
                )

            # 市場レジーム検出
            market_regime = self.regime_detector.detect_market_regime(
                df, indicators, meta_features
            )

            # 動的重み調整
            self._update_weights(market_regime)

            # 投票システムの重みを更新
            self.voting_system.strategy_weights = self.strategy_weights

            # アンサンブル投票を実行
            ensemble_result = self.voting_system.perform_voting(
                strategy_signals, meta_features, ml_predictions, market_regime
            )

            if ensemble_result:
                (
                    ensemble_signal,
                    voting_scores,
                    ensemble_confidence,
                    ensemble_uncertainty,
                ) = ensemble_result

                return EnsembleSignal(
                    ensemble_signal=ensemble_signal,
                    strategy_signals=strategy_signals,
                    voting_scores=voting_scores,
                    ensemble_confidence=ensemble_confidence,
                    strategy_weights=self.strategy_weights.copy(),
                    voting_type=self.voting_type,
                    meta_features=meta_features,
                    ml_predictions=ml_predictions,
                    feature_importance=feature_importance,
                    market_regime=market_regime,
                    ensemble_uncertainty=ensemble_uncertainty,
                )

            return None

        except Exception as e:
            logger.error(f"アンサンブルシグナル生成エラー: {e}")
            return None

    def _update_weights(self, market_regime: str) -> None:
        """重みを更新"""
        if self.ensemble_strategy == EnsembleStrategy.ADAPTIVE:
            self.strategy_weights = (
                self.performance_manager.update_adaptive_weights(self.strategies)
            )
        elif self.ensemble_strategy == EnsembleStrategy.REGIME_ADAPTIVE:
            self.strategy_weights = self.regime_detector.get_regime_weights(market_regime)

    def update_strategy_performance(
        self,
        strategy_name: str,
        success: bool,
        confidence: float,
        return_rate: float = 0.0,
    ) -> None:
        """戦略パフォーマンスを更新"""
        self.performance_manager.update_strategy_performance(
            strategy_name, success, confidence, return_rate
        )

    def train_ml_models(
        self, historical_data: pd.DataFrame, retrain: bool = False
    ) -> Dict[str, Any]:
        """機械学習モデルを訓練"""
        return self.ml_integration.train_ml_models(historical_data, retrain)

    def get_ml_model_info(self) -> Dict[str, Any]:
        """機械学習モデルの情報を取得"""
        return self.ml_integration.get_ml_model_info()

    def get_strategy_summary(self) -> Dict[str, Any]:
        """戦略サマリーを取得"""
        summary = {
            "ensemble_strategy": self.ensemble_strategy.value,
            "voting_type": self.voting_type.value,
            "strategy_weights": self.strategy_weights,
            "strategy_count": len(self.strategies),
            "performance_records": len(self.performance_manager.strategy_performance),
            "ml_enabled": self.ml_integration.is_available(),
            "current_market_regime": self.regime_detector.current_market_regime,
            "regime_history": self.regime_detector.regime_history[-5:] 
                              if self.regime_detector.regime_history else [],
        }

        # パフォーマンスサマリーを追加
        perf_summary = self.performance_manager.get_performance_summary()
        summary.update(perf_summary)

        # ML情報を追加
        if self.ml_integration.is_available():
            fitted_models = self.ml_integration.get_fitted_models()
            summary["ml_models"] = {
                "total_models": len(self.ml_integration.ml_manager.list_models())
                               if self.ml_integration.ml_manager else 0,
                "fitted_models": len(fitted_models),
                "fitted_model_names": fitted_models,
            }

        # レジーム情報を追加
        regime_summary = self.regime_detector.get_regime_summary()
        summary["regime_info"] = regime_summary

        return summary

    def reset_performance(self, strategy_name: str = None) -> None:
        """パフォーマンスをリセット"""
        self.performance_manager.reset_performance(strategy_name)

    def reset_regime_history(self) -> None:
        """レジーム履歴をリセット"""
        self.regime_detector.reset_history()