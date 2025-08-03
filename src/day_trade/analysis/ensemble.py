"""
アンサンブル取引戦略
複数の戦略と機械学習モデルを組み合わせて最適化されたシグナルを生成する
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .signals import SignalStrength, SignalType, TradingSignal, TradingSignalGenerator
from .ml_models import MLModelManager, ModelConfig, create_ensemble_predictions
from .feature_engineering import AdvancedFeatureEngineer, create_target_variables
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="ensemble")


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
class StrategyPerformance:
    """戦略パフォーマンス記録"""

    strategy_name: str
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    success_rate: float = 0.0
    average_confidence: float = 0.0
    average_return: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = None

    def update_performance(
        self, success: bool, confidence: float, return_rate: float = 0.0
    ):
        """パフォーマンスを更新"""
        self.total_signals += 1
        if success:
            self.successful_signals += 1
        else:
            self.failed_signals += 1

        self.success_rate = (
            self.successful_signals / self.total_signals
            if self.total_signals > 0
            else 0.0
        )

        # 移動平均で信頼度と収益率を更新
        alpha = 0.1  # 学習率
        if self.total_signals == 1:
            # 初回は直接設定
            self.average_confidence = confidence
        else:
            self.average_confidence = (
                1 - alpha
            ) * self.average_confidence + alpha * confidence
        self.average_return = (1 - alpha) * self.average_return + alpha * return_rate

        self.last_updated = datetime.now()


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


class EnsembleTradingStrategy:
    """アンサンブル取引戦略"""

    def __init__(
        self,
        ensemble_strategy: EnsembleStrategy = EnsembleStrategy.BALANCED,
        voting_type: EnsembleVotingType = EnsembleVotingType.SOFT_VOTING,
        performance_file: Optional[str] = None,
        enable_ml_models: bool = True,
        models_dir: Optional[str] = None,
    ):
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

        # パフォーマンス履歴
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self._load_performance_history()

        # 動的重み
        self.strategy_weights = self._initialize_weights()

        # メタ学習のための特徴量
        self.meta_features = {}

        # 機械学習コンポーネント
        self.ml_manager = None
        self.feature_engineer = None
        self.ml_predictions_history = []

        if self.enable_ml_models:
            try:
                self.ml_manager = MLModelManager(models_dir)
                self.feature_engineer = AdvancedFeatureEngineer()
                self._initialize_ml_models()
            except ImportError as e:
                logger.warning(f"機械学習モジュールが利用できません: {e}")
                self.enable_ml_models = False

        # 市場レジーム検出
        self.current_market_regime = "unknown"
        self.regime_history = []

    def _initialize_strategies(self) -> Dict[str, TradingSignalGenerator]:
        """個別戦略を初期化"""
        strategies = {}

        # 1. 保守的RSI戦略
        conservative_strategy = TradingSignalGenerator()
        conservative_strategy.clear_rules()
        from .signals import (
            MACDCrossoverRule,
            MACDDeathCrossRule,
            RSIOverboughtRule,
            RSIOversoldRule,
        )

        conservative_strategy.add_buy_rule(RSIOversoldRule(threshold=20, weight=2.0))
        conservative_strategy.add_buy_rule(MACDCrossoverRule(weight=1.5))
        conservative_strategy.add_sell_rule(RSIOverboughtRule(threshold=80, weight=2.0))
        conservative_strategy.add_sell_rule(MACDDeathCrossRule(weight=1.5))
        strategies["conservative_rsi"] = conservative_strategy

        # 2. 積極的モメンタム戦略
        momentum_strategy = TradingSignalGenerator()
        momentum_strategy.clear_rules()
        from .signals import BollingerBandRule, PatternBreakoutRule, VolumeSpikeBuyRule

        momentum_strategy.add_buy_rule(BollingerBandRule(position="lower", weight=1.5))
        momentum_strategy.add_buy_rule(
            PatternBreakoutRule(direction="upward", weight=2.5)
        )
        momentum_strategy.add_buy_rule(VolumeSpikeBuyRule(weight=2.0))
        momentum_strategy.add_sell_rule(BollingerBandRule(position="upper", weight=1.5))
        momentum_strategy.add_sell_rule(
            PatternBreakoutRule(direction="downward", weight=2.5)
        )
        strategies["aggressive_momentum"] = momentum_strategy

        # 3. トレンドフォロー戦略
        trend_strategy = TradingSignalGenerator()
        trend_strategy.clear_rules()
        from .signals import DeadCrossRule, GoldenCrossRule

        trend_strategy.add_buy_rule(GoldenCrossRule(weight=3.0))
        trend_strategy.add_buy_rule(MACDCrossoverRule(weight=2.0))
        trend_strategy.add_sell_rule(DeadCrossRule(weight=3.0))
        trend_strategy.add_sell_rule(MACDDeathCrossRule(weight=2.0))
        strategies["trend_following"] = trend_strategy

        # 4. 平均回帰戦略
        mean_reversion_strategy = TradingSignalGenerator()
        mean_reversion_strategy.clear_rules()
        mean_reversion_strategy.add_buy_rule(RSIOversoldRule(threshold=30, weight=2.0))
        mean_reversion_strategy.add_buy_rule(
            BollingerBandRule(position="lower", weight=2.5)
        )
        mean_reversion_strategy.add_sell_rule(
            RSIOverboughtRule(threshold=70, weight=2.0)
        )
        mean_reversion_strategy.add_sell_rule(
            BollingerBandRule(position="upper", weight=2.5)
        )
        strategies["mean_reversion"] = mean_reversion_strategy

        # 5. デフォルト統合戦略
        default_strategy = TradingSignalGenerator()  # 既存のデフォルトルール
        strategies["default_integrated"] = default_strategy

        return strategies

    def _initialize_ml_models(self):
        """機械学習モデルを初期化"""
        if not self.ml_manager:
            return

        try:
            # 1. 回帰モデル（リターン予測）
            return_model_config = ModelConfig(
                model_type="random_forest",
                task_type="regression",
                cv_folds=5,
                model_params={"n_estimators": 100, "max_depth": 10}
            )
            self.ml_manager.create_model("return_predictor", return_model_config)

            # 2. 分類モデル（方向性予測）
            direction_model_config = ModelConfig(
                model_type="gradient_boosting",
                task_type="classification",
                cv_folds=5,
                model_params={"n_estimators": 100, "learning_rate": 0.1}
            )
            self.ml_manager.create_model("direction_predictor", direction_model_config)

            # 3. ボラティリティ予測モデル
            volatility_model_config = ModelConfig(
                model_type="xgboost",
                task_type="regression",
                cv_folds=3,
                model_params={"n_estimators": 50, "max_depth": 6}
            )
            self.ml_manager.create_model("volatility_predictor", volatility_model_config)

            # 4. メタラーナー（アンサンブル最適化）
            meta_model_config = ModelConfig(
                model_type="linear",
                task_type="regression",
                cv_folds=3
            )
            self.ml_manager.create_model("meta_learner", meta_model_config)

            logger.info("機械学習モデルを初期化しました")

        except Exception as e:
            logger.error(f"機械学習モデル初期化エラー: {e}")
            self.enable_ml_models = False

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
            return {name: 0.2 for name in self.strategies}

    def _load_performance_history(self):
        """パフォーマンス履歴をロード"""
        if not self.performance_file:
            return

        try:
            performance_path = Path(self.performance_file)
            if performance_path.exists():
                with open(performance_path, encoding="utf-8") as f:
                    data = json.load(f)

                for strategy_name, perf_data in data.items():
                    self.strategy_performance[strategy_name] = StrategyPerformance(
                        strategy_name=strategy_name,
                        total_signals=perf_data.get("total_signals", 0),
                        successful_signals=perf_data.get("successful_signals", 0),
                        failed_signals=perf_data.get("failed_signals", 0),
                        success_rate=perf_data.get("success_rate", 0.0),
                        average_confidence=perf_data.get("average_confidence", 0.0),
                        average_return=perf_data.get("average_return", 0.0),
                        sharpe_ratio=perf_data.get("sharpe_ratio", 0.0),
                        last_updated=(
                            datetime.fromisoformat(perf_data["last_updated"])
                            if perf_data.get("last_updated")
                            else None
                        ),
                    )

                logger.info(
                    f"パフォーマンス履歴をロード: {len(self.strategy_performance)} 戦略"
                )
        except Exception as e:
            logger.warning(f"パフォーマンス履歴ロードエラー: {e}")

    def _save_performance_history(self):
        """パフォーマンス履歴を保存"""
        if not self.performance_file:
            return

        try:
            data = {}
            for strategy_name, perf in self.strategy_performance.items():
                data[strategy_name] = {
                    "total_signals": perf.total_signals,
                    "successful_signals": perf.successful_signals,
                    "failed_signals": perf.failed_signals,
                    "success_rate": perf.success_rate,
                    "average_confidence": perf.average_confidence,
                    "average_return": perf.average_return,
                    "sharpe_ratio": perf.sharpe_ratio,
                    "last_updated": (
                        perf.last_updated.isoformat() if perf.last_updated else None
                    ),
                }

            performance_path = Path(self.performance_file)
            performance_path.parent.mkdir(parents=True, exist_ok=True)

            with open(performance_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug("パフォーマンス履歴を保存")
        except Exception as e:
            logger.error(f"パフォーマンス履歴保存エラー: {e}")

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
            meta_features = self._calculate_meta_features(df, indicators, patterns)

            # 機械学習予測の実行
            ml_predictions = None
            feature_importance = None
            if self.enable_ml_models and self.ml_manager:
                ml_predictions, feature_importance = self._generate_ml_predictions(df, indicators)

            # 市場レジーム検出
            market_regime = self._detect_market_regime(df, indicators, meta_features)

            # 動的重み調整
            if self.ensemble_strategy == EnsembleStrategy.ADAPTIVE:
                self._update_adaptive_weights()
            elif self.ensemble_strategy == EnsembleStrategy.REGIME_ADAPTIVE:
                self._update_regime_adaptive_weights(market_regime)

            # アンサンブル投票を実行
            ensemble_result = self._perform_ensemble_voting(
                strategy_signals, meta_features, ml_predictions, market_regime
            )

            if ensemble_result:
                ensemble_signal, voting_scores, ensemble_confidence, ensemble_uncertainty = ensemble_result

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

    def _calculate_meta_features(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Dict[str, Any]:
        """メタ特徴量を計算"""
        try:
            meta_features = {}

            # 市場状況の特徴量
            if len(df) >= 20:
                # ボラティリティ
                returns = df["Close"].pct_change().dropna()
                if len(returns) > 0:
                    meta_features["volatility"] = returns.std() * np.sqrt(
                        252
                    )  # 年率ボラティリティ
                    meta_features["mean_return"] = returns.mean()

                # トレンド強度
                if len(df) >= 50:
                    sma_20 = df["Close"].rolling(20).mean()
                    sma_50 = df["Close"].rolling(50).mean()
                    if not sma_20.empty and not sma_50.empty:
                        meta_features["trend_strength"] = (
                            sma_20.iloc[-1] / sma_50.iloc[-1] - 1
                        ) * 100

                # 価格位置（過去のレンジ内での位置）
                high_20 = df["High"].rolling(20).max().iloc[-1]
                low_20 = df["Low"].rolling(20).min().iloc[-1]
                current_price = df["Close"].iloc[-1]
                if high_20 != low_20:
                    meta_features["price_position"] = (current_price - low_20) / (
                        high_20 - low_20
                    )

            # テクニカル指標の状況
            if indicators is not None and not indicators.empty:
                if "RSI" in indicators.columns:
                    meta_features["rsi_level"] = indicators["RSI"].iloc[-1]

                if "MACD" in indicators.columns and "MACD_Signal" in indicators.columns:
                    macd_diff = (
                        indicators["MACD"].iloc[-1] - indicators["MACD_Signal"].iloc[-1]
                    )
                    meta_features["macd_divergence"] = macd_diff

            # 出来高の特徴量
            if "Volume" in df.columns and len(df) >= 10:
                avg_volume = df["Volume"].rolling(10).mean().iloc[-1]
                current_volume = df["Volume"].iloc[-1]
                meta_features["volume_ratio"] = (
                    current_volume / avg_volume if avg_volume > 0 else 1.0
                )

            return meta_features

        except Exception as e:
            logger.error(f"メタ特徴量計算エラー: {e}")
            return {}

    def _generate_ml_predictions(
        self, df: pd.DataFrame, indicators: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """機械学習予測を生成"""
        try:
            if not self.feature_engineer or len(df) < 50:
                return None, None

            # 高度な特徴量を生成
            volume_data = df["Volume"] if "Volume" in df.columns else None
            features = self.feature_engineer.generate_all_features(
                price_data=df, volume_data=volume_data
            )

            if features.empty:
                return None, None

            # 最新の特徴量を取得
            latest_features = features.tail(1)

            predictions = {}
            feature_importance = {}

            # 各モデルで予測を実行
            for model_name in self.ml_manager.list_models():
                try:
                    model = self.ml_manager.models[model_name]
                    if model.is_fitted:
                        pred = self.ml_manager.predict(model_name, latest_features)
                        predictions[model_name] = float(pred[0]) if len(pred) > 0 else 0.0

                        # 特徴量重要度を取得
                        importance = model._get_feature_importance()
                        if importance:
                            feature_importance[model_name] = importance

                except Exception as e:
                    logger.warning(f"モデル {model_name} の予測でエラー: {e}")

            return predictions, feature_importance

        except Exception as e:
            logger.error(f"機械学習予測エラー: {e}")
            return None, None

    def _detect_market_regime(
        self,
        df: pd.DataFrame,
        indicators: Optional[pd.DataFrame] = None,
        meta_features: Dict[str, Any] = None
    ) -> str:
        """市場レジームを検出"""
        try:
            if len(df) < 50:
                return "insufficient_data"

            # ボラティリティベースのレジーム検出
            returns = df["Close"].pct_change().dropna()
            current_vol = returns.tail(20).std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)

            # トレンドベースのレジーム検出
            sma_20 = df["Close"].rolling(20).mean()
            sma_50 = df["Close"].rolling(50).mean()

            if len(sma_20) > 0 and len(sma_50) > 0:
                trend_ratio = sma_20.iloc[-1] / sma_50.iloc[-1]
            else:
                trend_ratio = 1.0

            # RSIベースのレジーム検出
            rsi_level = meta_features.get("rsi_level", 50) if meta_features else 50

            # レジーム判定ロジック
            if current_vol > historical_vol * 1.5:
                if rsi_level > 70:
                    regime = "high_vol_overbought"
                elif rsi_level < 30:
                    regime = "high_vol_oversold"
                else:
                    regime = "high_volatility"
            elif current_vol < historical_vol * 0.7:
                regime = "low_volatility"
            elif trend_ratio > 1.05:
                regime = "uptrend"
            elif trend_ratio < 0.95:
                regime = "downtrend"
            else:
                regime = "sideways"

            # レジーム履歴を更新
            if len(self.regime_history) >= 10:
                self.regime_history.pop(0)
            self.regime_history.append(regime)
            self.current_market_regime = regime

            return regime

        except Exception as e:
            logger.error(f"市場レジーム検出エラー: {e}")
            return "unknown"

    def _update_regime_adaptive_weights(self, market_regime: str):
        """市場レジームに基づく重み調整"""
        try:
            if market_regime == "high_volatility":
                # 高ボラティリティ時は保守的戦略を重視
                self.strategy_weights.update({
                    "conservative_rsi": 0.4,
                    "aggressive_momentum": 0.1,
                    "trend_following": 0.2,
                    "mean_reversion": 0.2,
                    "default_integrated": 0.1,
                })
            elif market_regime in ["uptrend", "downtrend"]:
                # トレンド相場ではトレンドフォロー戦略を重視
                self.strategy_weights.update({
                    "conservative_rsi": 0.1,
                    "aggressive_momentum": 0.3,
                    "trend_following": 0.4,
                    "mean_reversion": 0.1,
                    "default_integrated": 0.1,
                })
            elif market_regime == "sideways":
                # レンジ相場では平均回帰戦略を重視
                self.strategy_weights.update({
                    "conservative_rsi": 0.2,
                    "aggressive_momentum": 0.1,
                    "trend_following": 0.1,
                    "mean_reversion": 0.5,
                    "default_integrated": 0.1,
                })
            else:  # low_volatility, unknown など
                # デフォルトのバランス型
                self.strategy_weights.update({
                    "conservative_rsi": 0.2,
                    "aggressive_momentum": 0.25,
                    "trend_following": 0.25,
                    "mean_reversion": 0.2,
                    "default_integrated": 0.1,
                })

            logger.debug(f"レジーム適応重み更新: {market_regime} -> {self.strategy_weights}")

        except Exception as e:
            logger.error(f"レジーム適応重み更新エラー: {e}")

    def _perform_ensemble_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """アンサンブル投票を実行"""
        try:
            if self.voting_type == EnsembleVotingType.SOFT_VOTING:
                return self._soft_voting(strategy_signals, meta_features, ml_predictions, market_regime)
            elif self.voting_type == EnsembleVotingType.HARD_VOTING:
                return self._hard_voting(strategy_signals, meta_features, ml_predictions, market_regime)
            elif self.voting_type == EnsembleVotingType.ML_ENSEMBLE:
                return self._ml_ensemble_voting(strategy_signals, meta_features, ml_predictions, market_regime)
            elif self.voting_type == EnsembleVotingType.STACKING:
                return self._stacking_voting(strategy_signals, meta_features, ml_predictions, market_regime)
            elif self.voting_type == EnsembleVotingType.DYNAMIC_ENSEMBLE:
                return self._dynamic_ensemble_voting(strategy_signals, meta_features, ml_predictions, market_regime)
            else:  # WEIGHTED_AVERAGE
                return self._weighted_average_voting(strategy_signals, meta_features, ml_predictions, market_regime)

        except Exception as e:
            logger.error(f"アンサンブル投票エラー: {e}")
            return None

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
            _strategy_weight = self.strategy_weights.get(
                strategy_name, 0.2
            )  # Renamed variable

            # パフォーマンスによる重み調整
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]
                performance_multiplier = 0.5 + perf.success_rate  # 0.5-1.5の範囲
                _strategy_weight *= performance_multiplier  # Renamed variable

            weighted_confidence = (
                signal.confidence * _strategy_weight
            )  # Renamed variable
            voting_scores[signal.signal_type.value] += weighted_confidence
            total_weight += _strategy_weight  # Renamed variable

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

        # 強度を決定
        if best_score >= 70:
            strength = SignalStrength.STRONG
        elif best_score >= 40:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK

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
        latest_signal = strategy_signals[0][1]  # 最初のシグナルから取得

        ensemble_signal = TradingSignal(
            signal_type=SignalType(best_signal_type),
            strength=strength,
            confidence=best_score,
            reasons=reasons,
            conditions_met={},
            timestamp=latest_signal.timestamp,
            price=latest_signal.price,
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
            _strategy_weight = self.strategy_weights.get(
                strategy_name, 0.2
            )  # Renamed variable

            # パフォーマンスによる重み調整
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]
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

        # 強度を決定（投票数に基づく）
        if vote_count >= len(strategy_signals) * 0.8:
            strength = SignalStrength.STRONG
        elif vote_count >= len(strategy_signals) * 0.6:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK

        # 理由をまとめる
        reasons = [f"多数決投票: {vote_count}/{total_votes} 票獲得"]

        # 最新の価格とタイムスタンプ
        latest_signal = strategy_signals[0][1]

        ensemble_signal = TradingSignal(
            signal_type=SignalType(best_signal_type),
            strength=strength,
            confidence=ensemble_confidence,
            reasons=reasons,
            conditions_met={},
            timestamp=latest_signal.timestamp,
            price=latest_signal.price,
        )

        # アンサンブル不確実性を計算
        vote_percentages = [count / total_votes for count in vote_counts.values()]
        ensemble_uncertainty = 1.0 - max(vote_percentages)  # 最大得票率の逆数

        return ensemble_signal, strategy_contributions, ensemble_confidence, ensemble_uncertainty

    def _weighted_average_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
        ml_predictions: Optional[Dict[str, float]] = None,
        market_regime: Optional[str] = None,
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float, float]]:
        """重み付け平均投票"""
        # ソフト投票の変種として実装
        return self._soft_voting(strategy_signals, meta_features, ml_predictions, market_regime)

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
                return self._soft_voting(strategy_signals, meta_features, ml_predictions, market_regime)

            # 戦略シグナルとML予測を統合
            combined_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
            total_weight = 0.0
            strategy_contributions = {}

            # 1. 戦略シグナルの処理（重み0.4）
            strategy_weight_factor = 0.4
            for strategy_name, signal in strategy_signals:
                base_weight = self.strategy_weights.get(strategy_name, 0.2) * strategy_weight_factor
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
                    combined_scores["buy"] += abs(return_pred) * 100 * ml_weight_factor * 0.4
                elif return_pred < -0.01:  # 1%以上の下落予測
                    combined_scores["sell"] += abs(return_pred) * 100 * ml_weight_factor * 0.4
                else:
                    combined_scores["hold"] += 20 * ml_weight_factor * 0.4

                ml_contribution["return_predictor"] = abs(return_pred) * 100 * ml_weight_factor * 0.4
                total_weight += ml_weight_factor * 0.4

            # 方向性予測の処理
            if "direction_predictor" in ml_predictions:
                direction_pred = ml_predictions["direction_predictor"]
                if direction_pred > 0.6:  # 上昇確率が高い
                    combined_scores["buy"] += direction_pred * 100 * ml_weight_factor * 0.4
                elif direction_pred < 0.4:  # 下落確率が高い
                    combined_scores["sell"] += (1 - direction_pred) * 100 * ml_weight_factor * 0.4
                else:
                    combined_scores["hold"] += 20 * ml_weight_factor * 0.4

                ml_contribution["direction_predictor"] = direction_pred * 100 * ml_weight_factor * 0.4
                total_weight += ml_weight_factor * 0.4

            # ボラティリティ予測の処理（リスク調整）
            if "volatility_predictor" in ml_predictions:
                vol_pred = ml_predictions["volatility_predictor"]
                if vol_pred > 0.3:  # 高ボラティリティ予測時はリスク回避
                    risk_adjustment = 0.7  # 信頼度を30%減らす
                    for signal_type in combined_scores:
                        if signal_type != "hold":
                            combined_scores[signal_type] *= risk_adjustment
                            combined_scores["hold"] += combined_scores[signal_type] * 0.3

                ml_contribution["volatility_predictor"] = vol_pred * ml_weight_factor * 0.2
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
            confidence_threshold = self._get_confidence_threshold() * 0.8  # ML使用時は閾値を下げる
            if best_score < confidence_threshold:
                best_signal_type = "hold"
                best_score = 0.0

            # 強度決定
            if best_score >= 75:
                strength = SignalStrength.STRONG
            elif best_score >= 45:
                strength = SignalStrength.MEDIUM
            else:
                strength = SignalStrength.WEAK

            # 理由の統合
            reasons = [f"機械学習アンサンブル投票: {best_signal_type}"]
            for strategy_name, signal in strategy_signals:
                if signal.signal_type.value == best_signal_type:
                    reasons.extend([f"{strategy_name}: {reason}" for reason in signal.reasons[:2]])

            # ML予測の理由を追加
            if ml_predictions:
                for model_name, pred in ml_predictions.items():
                    reasons.append(f"{model_name}: {pred:.3f}")

            # シグナル作成
            latest_signal = strategy_signals[0][1]
            ensemble_signal = TradingSignal(
                signal_type=SignalType(best_signal_type),
                strength=strength,
                confidence=best_score,
                reasons=reasons,
                conditions_met={},
                timestamp=latest_signal.timestamp,
                price=latest_signal.price,
            )

            # 不確実性計算（ML予測の分散を考慮）
            score_values = list(combined_scores.values())
            base_uncertainty = np.std(score_values) if len(score_values) > 1 else 0.0

            # ML予測の一致度を考慮
            ml_agreement = 0.0
            if len(ml_predictions) > 1:
                ml_values = list(ml_predictions.values())
                ml_agreement = 1.0 - np.std(ml_values) / (np.mean(np.abs(ml_values)) + 1e-8)

            ensemble_uncertainty = base_uncertainty * (1.0 - ml_agreement * 0.5)

            # 貢献度を統合
            all_contributions = {**strategy_contributions, **ml_contribution}

            return ensemble_signal, all_contributions, best_score, ensemble_uncertainty

        except Exception as e:
            logger.error(f"機械学習アンサンブル投票エラー: {e}")
            return self._soft_voting(strategy_signals, meta_features, ml_predictions, market_regime)

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
            base_ensemble = self._soft_voting(strategy_signals, meta_features, ml_predictions, market_regime)
            if not base_ensemble:
                return None

            # メタラーナーの特徴量を構築
            meta_features_array = []

            # 各戦略の信頼度を特徴量として使用
            for strategy_name, signal in strategy_signals:
                meta_features_array.append(signal.confidence)

            # ML予測を特徴量として追加
            if ml_predictions:
                for pred in ml_predictions.values():
                    meta_features_array.append(pred * 100)  # スケール調整

            # メタ特徴量を追加
            if meta_features:
                for key in ["volatility", "trend_strength", "rsi_level", "volume_ratio"]:
                    meta_features_array.append(meta_features.get(key, 0))

            # メタラーナーが利用可能で十分な特徴量がある場合
            if (self.ml_manager and "meta_learner" in self.ml_manager.models and
                len(meta_features_array) >= 5):

                try:
                    # 特徴量をDataFrameに変換
                    feature_df = pd.DataFrame([meta_features_array])

                    # メタラーナーで最終予測
                    meta_pred = self.ml_manager.predict("meta_learner", feature_df)
                    if len(meta_pred) > 0:
                        # メタ予測を使って信頼度を調整
                        adjusted_confidence = base_ensemble[2] * (0.5 + abs(meta_pred[0]) * 0.5)
                        return (base_ensemble[0], base_ensemble[1], adjusted_confidence, base_ensemble[3])

                except Exception as e:
                    logger.warning(f"メタラーナー予測エラー: {e}")

            # メタラーナーが使えない場合は基本アンサンブルを返す
            return base_ensemble

        except Exception as e:
            logger.error(f"スタッキング投票エラー: {e}")
            return self._soft_voting(strategy_signals, meta_features, ml_predictions, market_regime)

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
            if market_regime in ["high_volatility", "high_vol_overbought", "high_vol_oversold"]:
                # 高ボラティリティ時は保守的にハード投票
                return self._hard_voting(strategy_signals, meta_features, ml_predictions, market_regime)
            elif market_regime in ["uptrend", "downtrend"] and ml_predictions:
                # トレンド相場でML予測がある場合はML重視
                return self._ml_ensemble_voting(strategy_signals, meta_features, ml_predictions, market_regime)
            elif market_regime == "sideways":
                # レンジ相場ではスタッキング手法
                return self._stacking_voting(strategy_signals, meta_features, ml_predictions, market_regime)
            else:
                # その他の場合はソフト投票
                return self._soft_voting(strategy_signals, meta_features, ml_predictions, market_regime)

        except Exception as e:
            logger.error(f"動的アンサンブル投票エラー: {e}")
            return self._soft_voting(strategy_signals, meta_features, ml_predictions, market_regime)

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
            if self.current_market_regime == "high_volatility":
                return 65.0
            elif self.current_market_regime in ["uptrend", "downtrend"]:
                return 40.0
            else:
                return 50.0
        else:  # ADAPTIVE
            # 過去のパフォーマンスに基づいて動的調整
            avg_success_rate = (
                np.mean(
                    [perf.success_rate for perf in self.strategy_performance.values()]
                )
                if self.strategy_performance
                else 0.0
            )
            return 30.0 + (70.0 - 30.0) * (1 - avg_success_rate)

    def _update_adaptive_weights(self):
        """適応型戦略の重みを更新"""
        if not self.strategy_performance:
            return

        # パフォーマンスベースの重み計算
        total_score = 0.0
        strategy_scores = {}

        for strategy_name in self.strategies:
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]

                # 複合スコア計算（成功率 + シャープレシオ + 最新性）
                recency_factor = 1.0
                if perf.last_updated:
                    days_old = (datetime.now() - perf.last_updated).days
                    recency_factor = max(
                        0.1, 1.0 - days_old / 365.0
                    )  # 1年で0.1まで減衰

                score = (
                    perf.success_rate * 0.4
                    + max(0, perf.sharpe_ratio) * 0.3
                    + max(0, perf.average_return) * 0.2
                    + recency_factor * 0.1
                )

                strategy_scores[strategy_name] = max(0.01, score)  # 最小重み保証
                total_score += strategy_scores[strategy_name]
            else:
                strategy_scores[strategy_name] = 0.2  # デフォルト重み
                total_score += 0.2

        # 正規化
        if total_score > 0:
            for strategy_name in strategy_scores:
                self.strategy_weights[strategy_name] = (
                    strategy_scores[strategy_name] / total_score
                )

        logger.debug(f"適応型重み更新: {self.strategy_weights}")

    def update_strategy_performance(
        self,
        strategy_name: str,
        success: bool,
        confidence: float,
        return_rate: float = 0.0,
    ):
        """戦略パフォーマンスを更新"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = StrategyPerformance(
                strategy_name
            )

        self.strategy_performance[strategy_name].update_performance(
            success, confidence, return_rate
        )
        self._save_performance_history()

    def train_ml_models(self, historical_data: pd.DataFrame, retrain: bool = False) -> Dict[str, Any]:
        """機械学習モデルを訓練"""
        try:
            if not self.enable_ml_models or not self.ml_manager or not self.feature_engineer:
                return {"error": "機械学習機能が無効です"}

            if len(historical_data) < 200:
                return {"error": "訓練に十分なデータがありません（最低200日必要）"}

            logger.info("機械学習モデルの訓練を開始")

            # 特徴量生成
            volume_data = historical_data["Volume"] if "Volume" in historical_data.columns else None
            features = self.feature_engineer.generate_all_features(
                price_data=historical_data, volume_data=volume_data
            )

            if features.empty:
                return {"error": "特徴量生成に失敗しました"}

            # ターゲット変数生成
            targets = create_target_variables(historical_data, prediction_horizon=5)

            training_results = {}

            # 各モデルの訓練
            model_configs = [
                ("return_predictor", "future_returns"),
                ("direction_predictor", "future_direction"),
                ("volatility_predictor", "future_high_volatility"),
            ]

            for model_name, target_name in model_configs:
                try:
                    if target_name not in targets:
                        logger.warning(f"ターゲット変数 {target_name} が見つかりません")
                        continue

                    # データの整合性チェック
                    common_index = features.index.intersection(targets[target_name].index)
                    if len(common_index) < 100:
                        logger.warning(f"モデル {model_name} に十分なデータがありません")
                        continue

                    X_train = features.loc[common_index]
                    y_train = targets[target_name].loc[common_index]

                    # モデルが存在しない場合は作成
                    if model_name not in self.ml_manager.list_models():
                        logger.warning(f"モデル {model_name} が存在しません。スキップします。")
                        continue

                    # 既に訓練済みでretrainがFalseの場合はスキップ
                    if not retrain and self.ml_manager.models[model_name].is_fitted:
                        logger.info(f"モデル {model_name} は既に訓練済みです")
                        continue

                    # モデル訓練
                    logger.info(f"モデル {model_name} を訓練中...")
                    result = self.ml_manager.train_model(model_name, X_train, y_train)
                    training_results[model_name] = result

                    # モデル保存
                    try:
                        self.ml_manager.save_model(model_name)
                        logger.info(f"モデル {model_name} を保存しました")
                    except Exception as e:
                        logger.warning(f"モデル {model_name} の保存に失敗: {e}")

                except Exception as e:
                    logger.error(f"モデル {model_name} の訓練エラー: {e}")
                    training_results[model_name] = {"error": str(e)}

            # メタラーナーの訓練
            try:
                if len(training_results) >= 2:  # 少なくとも2つのモデルが訓練された場合
                    meta_features_list = []
                    meta_targets_list = []

                    # 各データポイントでメタ特徴量を生成
                    for i in range(50, len(historical_data) - 10):  # 十分な履歴とフォワードルッキングを確保
                        try:
                            subset_data = historical_data.iloc[:i+1]
                            future_return = historical_data["Close"].iloc[i+5] / historical_data["Close"].iloc[i] - 1

                            # 各基本モデルの予測を特徴量として使用
                            model_predictions = []
                            for model_name in ["return_predictor", "direction_predictor", "volatility_predictor"]:
                                if model_name in self.ml_manager.list_models():
                                    try:
                                        subset_features = self.feature_engineer.generate_all_features(
                                            price_data=subset_data, volume_data=volume_data.iloc[:i+1] if volume_data is not None else None
                                        )
                                        if not subset_features.empty:
                                            pred = self.ml_manager.predict(model_name, subset_features.tail(1))
                                            model_predictions.append(pred[0] if len(pred) > 0 else 0.0)
                                        else:
                                            model_predictions.append(0.0)
                                    except:
                                        model_predictions.append(0.0)

                            if len(model_predictions) >= 2:
                                meta_features_list.append(model_predictions)
                                meta_targets_list.append(future_return)

                        except Exception as e:
                            continue

                    if len(meta_features_list) >= 50:
                        meta_X = pd.DataFrame(meta_features_list)
                        meta_y = pd.Series(meta_targets_list)

                        meta_result = self.ml_manager.train_model("meta_learner", meta_X, meta_y)
                        training_results["meta_learner"] = meta_result

                        try:
                            self.ml_manager.save_model("meta_learner")
                            logger.info("メタラーナーを保存しました")
                        except Exception as e:
                            logger.warning(f"メタラーナーの保存に失敗: {e}")

            except Exception as e:
                logger.error(f"メタラーナー訓練エラー: {e}")
                training_results["meta_learner"] = {"error": str(e)}

            logger.info(f"機械学習モデル訓練完了: {len(training_results)}個のモデル")
            return {
                "success": True,
                "models_trained": len(training_results),
                "training_results": training_results,
                "feature_count": len(features.columns),
                "data_points": len(historical_data)
            }

        except Exception as e:
            logger.error(f"機械学習モデル訓練エラー: {e}")
            return {"error": str(e)}

    def get_ml_model_info(self) -> Dict[str, Any]:
        """機械学習モデルの情報を取得"""
        if not self.enable_ml_models or not self.ml_manager:
            return {"ml_enabled": False}

        info = {
            "ml_enabled": True,
            "models": {},
            "feature_engineer_available": self.feature_engineer is not None
        }

        for model_name in self.ml_manager.list_models():
            try:
                model_info = self.ml_manager.get_model_info(model_name)
                info["models"][model_name] = {
                    "model_type": model_info["model_type"],
                    "task_type": model_info["task_type"],
                    "is_fitted": model_info["is_fitted"],
                    "feature_count": model_info["feature_count"]
                }
                if model_info["is_fitted"] and "feature_importance" in model_info:
                    # 上位5個の重要特徴量のみ表示
                    importance = model_info["feature_importance"]
                    if importance:
                        top_features = dict(list(importance.items())[:5])
                        info["models"][model_name]["top_features"] = top_features
            except Exception as e:
                info["models"][model_name] = {"error": str(e)}

        return info

    def get_strategy_summary(self) -> Dict[str, Any]:
        """戦略サマリーを取得"""
        summary = {
            "ensemble_strategy": self.ensemble_strategy.value,
            "voting_type": self.voting_type.value,
            "strategy_weights": self.strategy_weights,
            "strategy_count": len(self.strategies),
            "performance_records": len(self.strategy_performance),
            "avg_success_rate": (
                np.mean(
                    [perf.success_rate for perf in self.strategy_performance.values()]
                )
                if self.strategy_performance
                else 0.0
            ),
            "ml_enabled": self.enable_ml_models,
            "current_market_regime": self.current_market_regime,
            "regime_history": self.regime_history[-5:] if self.regime_history else [],
        }

        # ML情報を追加
        if self.enable_ml_models and self.ml_manager:
            summary["ml_models"] = {
                "total_models": len(self.ml_manager.list_models()),
                "fitted_models": sum(
                    1 for model_name in self.ml_manager.list_models()
                    if self.ml_manager.models[model_name].is_fitted
                )
            }

        return summary


# 使用例
if __name__ == "__main__":
    from datetime import datetime

    import numpy as np

    # サンプルデータ作成
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    np.random.seed(42)

    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 2
    close_prices = trend + noise

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close_prices + np.random.randn(100) * 0.5,
            "High": close_prices + np.abs(np.random.randn(100)) * 2,
            "Low": close_prices - np.abs(np.random.randn(100)) * 2,
            "Close": close_prices,
            "Volume": np.random.randint(1000000, 5000000, 100),
        }
    )
    df.set_index("Date", inplace=True)

    # アンサンブル戦略テスト
    ensemble = EnsembleTradingStrategy(
        ensemble_strategy=EnsembleStrategy.BALANCED,
        voting_type=EnsembleVotingType.SOFT_VOTING,
    )

    # シグナル生成
    ensemble_signal = ensemble.generate_ensemble_signal(df)

    if ensemble_signal:
        signal = ensemble_signal.ensemble_signal
        print(f"アンサンブルシグナル: {signal.signal_type.value.upper()}")
        print(f"強度: {signal.strength.value}")
        print(f"信頼度: {signal.confidence:.1f}%")
        print(f"価格: {signal.price:.2f}")

        print("\n戦略別貢献度:")
        for strategy_name, score in ensemble_signal.voting_scores.items():
            print(f"  {strategy_name}: {score:.2f}")

        print("\n戦略重み:")
        for strategy_name, weight in ensemble_signal.strategy_weights.items():
            print(f"  {strategy_name}: {weight:.2f}")

        print("\nメタ特徴量:")
        for feature, value in ensemble_signal.meta_features.items():
            print(f"  {feature}: {value}")
    else:
        print("アンサンブルシグナルなし")

    # 戦略サマリー
    print("\n戦略サマリー:")
    summary = ensemble.get_strategy_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
