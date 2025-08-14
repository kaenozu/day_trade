#!/usr/bin/env python3
"""
動的重み付けアルゴリズム
Adaptive Weighting Algorithms

Issue #762: 高度なアンサンブル予測システムの強化 - Phase 1
"""

import numpy as np
import pandas as pd
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque
import warnings

# 数値計算・機械学習
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.linalg import inv
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# フィルタリング
from filterpy.kalman import KalmanFilter
from hmmlearn import hmm

# ログ設定
logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """市場レジーム情報"""
    regime_id: int
    name: str
    volatility: float
    trend: float
    volume_profile: float
    confidence: float
    timestamp: float
    characteristics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelPerformance:
    """モデルパフォーマンス情報"""
    model_id: str
    accuracy: float
    mse: float
    mae: float
    sharpe_ratio: float
    max_drawdown: float
    predictions_count: int
    last_updated: float
    regime_specific_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)

@dataclass
class WeightUpdate:
    """重み更新情報"""
    model_id: str
    old_weight: float
    new_weight: float
    reason: str
    confidence: float
    timestamp: float

class MarketRegimeDetector:
    """市場レジーム検出器"""

    def __init__(self, lookback_window: int = 252, n_regimes: int = 3):
        self.lookback_window = lookback_window
        self.n_regimes = n_regimes

        # Hidden Markov Model
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            random_state=42
        )

        # Gaussian Mixture Model
        self.gmm_model = GaussianMixture(
            n_components=n_regimes,
            random_state=42
        )

        # データ蓄積
        self.market_data_buffer = deque(maxlen=lookback_window * 2)
        self.feature_scaler = StandardScaler()

        # 学習済みフラグ
        self.is_fitted = False

        # レジーム定義
        self.regime_names = {
            0: "Low Volatility Trend",
            1: "High Volatility Sideways",
            2: "Crisis/Extreme Volatility"
        }

        logger.info(f"MarketRegimeDetector initialized with {n_regimes} regimes")

    def _extract_market_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """市場特徴量抽出"""
        try:
            features = []

            if len(market_data) < 20:
                # データが不足している場合はデフォルト値
                return np.array([[0.1, 0.0, 1.0]])

            # 1. ボラティリティ特徴量
            returns = market_data['Close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1]
            if np.isnan(volatility):
                volatility = 0.1

            # 2. トレンド特徴量
            price_sma_20 = market_data['Close'].rolling(window=20).mean().iloc[-1]
            price_sma_50 = market_data['Close'].rolling(window=min(50, len(market_data))).mean().iloc[-1]
            trend = (price_sma_20 - price_sma_50) / price_sma_50 if price_sma_50 > 0 else 0.0

            # 3. ボリューム特徴量
            if 'Volume' in market_data.columns:
                volume_sma = market_data['Volume'].rolling(window=20).mean().iloc[-1]
                current_volume = market_data['Volume'].iloc[-1]
                volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1.0
            else:
                volume_ratio = 1.0

            # 4. 追加特徴量
            # RSI
            rsi = self._calculate_rsi(market_data['Close'])

            # Bollinger Bands位置
            bb_position = self._calculate_bb_position(market_data['Close'])

            # VIX的指標（実現ボラティリティ）
            realized_vol = returns.rolling(window=10).std().iloc[-1] * np.sqrt(252)
            if np.isnan(realized_vol):
                realized_vol = volatility * np.sqrt(252)

            features = [
                volatility,
                trend,
                volume_ratio,
                rsi,
                bb_position,
                realized_vol
            ]

            # NaN処理
            features = [f if not np.isnan(f) else 0.0 for f in features]

            return np.array([features])

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            # フォールバック値
            return np.array([[0.1, 0.0, 1.0, 50.0, 0.5, 0.2]])

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """RSI計算"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50.0
        except:
            return 50.0

    def _calculate_bb_position(self, prices: pd.Series, window: int = 20) -> float:
        """Bollinger Bands内の位置"""
        try:
            sma = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)

            current_price = prices.iloc[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]

            if current_upper == current_lower:
                return 0.5

            position = (current_price - current_lower) / (current_upper - current_lower)
            return np.clip(position, 0.0, 1.0)
        except:
            return 0.5

    async def fit_regime_models(self, historical_data: pd.DataFrame) -> None:
        """レジームモデル学習"""
        try:
            if len(historical_data) < self.lookback_window:
                logger.warning(f"Insufficient data for training: {len(historical_data)} < {self.lookback_window}")
                return

            # 特徴量抽出
            features_list = []
            for i in range(len(historical_data) - 20):
                window_data = historical_data.iloc[i:i+20]
                features = self._extract_market_features(window_data)
                features_list.append(features[0])

            if len(features_list) < 10:
                logger.warning("Not enough feature windows for training")
                return

            X = np.array(features_list)

            # 正規化
            X_scaled = self.feature_scaler.fit_transform(X)

            # HMM学習
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.hmm_model.fit(X_scaled)

                # GMM学習（フォールバック用）
                self.gmm_model.fit(X_scaled)

            self.is_fitted = True
            logger.info("Regime models fitted successfully")

        except Exception as e:
            logger.error(f"Error fitting regime models: {e}")
            self.is_fitted = False

    async def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """市場レジーム検出"""
        try:
            # データバッファ更新
            current_data = {
                'timestamp': time.time(),
                'close': market_data['Close'].iloc[-1] if len(market_data) > 0 else 100.0,
                'volume': market_data.get('Volume', pd.Series([1000000])).iloc[-1]
            }
            self.market_data_buffer.append(current_data)

            # 特徴量抽出
            features = self._extract_market_features(market_data)

            if not self.is_fitted:
                # モデル未学習の場合はデフォルトレジーム
                return MarketRegime(
                    regime_id=0,
                    name=self.regime_names[0],
                    volatility=features[0][0],
                    trend=features[0][1],
                    volume_profile=features[0][2],
                    confidence=0.5,
                    timestamp=time.time(),
                    characteristics={
                        'rsi': features[0][3] if len(features[0]) > 3 else 50.0,
                        'bb_position': features[0][4] if len(features[0]) > 4 else 0.5,
                        'realized_vol': features[0][5] if len(features[0]) > 5 else 0.2
                    }
                )

            # 正規化
            features_scaled = self.feature_scaler.transform(features)

            # HMM予測
            try:
                regime_id = self.hmm_model.predict(features_scaled)[0]
                state_probs = self.hmm_model.predict_proba(features_scaled)[0]
                confidence = np.max(state_probs)
            except:
                # HMM失敗時はGMMフォールバック
                regime_id = self.gmm_model.predict(features_scaled)[0]
                confidence = 0.6

            # レジーム情報作成
            regime = MarketRegime(
                regime_id=int(regime_id),
                name=self.regime_names.get(regime_id, f"Regime_{regime_id}"),
                volatility=features[0][0],
                trend=features[0][1],
                volume_profile=features[0][2],
                confidence=float(confidence),
                timestamp=time.time(),
                characteristics={
                    'rsi': features[0][3] if len(features[0]) > 3 else 50.0,
                    'bb_position': features[0][4] if len(features[0]) > 4 else 0.5,
                    'realized_vol': features[0][5] if len(features[0]) > 5 else 0.2
                }
            )

            logger.debug(f"Detected regime: {regime.name} (confidence: {confidence:.3f})")
            return regime

        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            # エラー時はデフォルトレジーム
            return MarketRegime(
                regime_id=0,
                name="Default",
                volatility=0.1,
                trend=0.0,
                volume_profile=1.0,
                confidence=0.3,
                timestamp=time.time()
            )

class PerformanceTracker:
    """モデルパフォーマンス追跡"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.model_performances: Dict[str, ModelPerformance] = {}
        self.prediction_history: Dict[str, deque] = {}
        self.target_history = deque(maxlen=window_size)

        logger.info(f"PerformanceTracker initialized with window size {window_size}")

    def update_performance(
        self,
        model_id: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        regime_id: Optional[int] = None
    ) -> ModelPerformance:
        """パフォーマンス更新"""
        try:
            # 予測履歴更新
            if model_id not in self.prediction_history:
                self.prediction_history[model_id] = deque(maxlen=self.window_size)

            # データ追加
            for pred, target in zip(predictions.flatten(), targets.flatten()):
                self.prediction_history[model_id].append({
                    'prediction': pred,
                    'target': target,
                    'timestamp': time.time(),
                    'regime_id': regime_id
                })

            # ターゲット履歴更新
            for target in targets.flatten():
                self.target_history.append(target)

            # パフォーマンス計算
            performance = self._calculate_performance(model_id, regime_id)
            self.model_performances[model_id] = performance

            return performance

        except Exception as e:
            logger.error(f"Error updating performance for {model_id}: {e}")
            return self._get_default_performance(model_id)

    def _calculate_performance(self, model_id: str, regime_id: Optional[int] = None) -> ModelPerformance:
        """パフォーマンス計算"""
        try:
            if model_id not in self.prediction_history or len(self.prediction_history[model_id]) == 0:
                return self._get_default_performance(model_id)

            history = list(self.prediction_history[model_id])

            # 予測と実績分離
            predictions = np.array([h['prediction'] for h in history])
            targets = np.array([h['target'] for h in history])

            if len(predictions) == 0:
                return self._get_default_performance(model_id)

            # 基本メトリクス
            mse = mean_squared_error(targets, predictions)
            mae = mean_absolute_error(targets, predictions)

            # 相関係数（精度代理）
            correlation = np.corrcoef(predictions, targets)[0, 1] if len(predictions) > 1 else 0.0
            accuracy = max(0.0, correlation) if not np.isnan(correlation) else 0.0

            # Sharpe ratio（リターン予測の場合）
            if len(predictions) > 10:
                pred_returns = np.diff(predictions)
                sharpe_ratio = np.mean(pred_returns) / (np.std(pred_returns) + 1e-8) if np.std(pred_returns) > 0 else 0.0
            else:
                sharpe_ratio = 0.0

            # Max Drawdown（予測値ベース）
            cumulative = np.cumsum(predictions - np.mean(predictions))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0

            # レジーム別パフォーマンス
            regime_performance = {}
            if regime_id is not None:
                regime_data = [h for h in history if h.get('regime_id') == regime_id]
                if regime_data:
                    regime_preds = np.array([h['prediction'] for h in regime_data])
                    regime_targets = np.array([h['target'] for h in regime_data])
                    regime_correlation = np.corrcoef(regime_preds, regime_targets)[0, 1] if len(regime_preds) > 1 else 0.0
                    regime_performance[regime_id] = {
                        'accuracy': max(0.0, regime_correlation) if not np.isnan(regime_correlation) else 0.0,
                        'mse': mean_squared_error(regime_targets, regime_preds),
                        'count': len(regime_data)
                    }

            performance = ModelPerformance(
                model_id=model_id,
                accuracy=accuracy,
                mse=mse,
                mae=mae,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                predictions_count=len(predictions),
                last_updated=time.time(),
                regime_specific_performance=regime_performance
            )

            logger.debug(f"Performance calculated for {model_id}: accuracy={accuracy:.3f}, mse={mse:.6f}")
            return performance

        except Exception as e:
            logger.error(f"Error calculating performance for {model_id}: {e}")
            return self._get_default_performance(model_id)

    def _get_default_performance(self, model_id: str) -> ModelPerformance:
        """デフォルトパフォーマンス"""
        return ModelPerformance(
            model_id=model_id,
            accuracy=0.5,
            mse=1.0,
            mae=0.8,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            predictions_count=0,
            last_updated=time.time()
        )

    def get_performance(self, model_id: str) -> Optional[ModelPerformance]:
        """パフォーマンス取得"""
        return self.model_performances.get(model_id)

    def get_all_performances(self) -> Dict[str, ModelPerformance]:
        """全パフォーマンス取得"""
        return self.model_performances.copy()

class WeightOptimizer:
    """重み最適化アルゴリズム"""

    def __init__(self,
                 momentum: float = 0.9,
                 min_weight: float = 0.01,
                 max_weight: float = 0.5,
                 regularization: float = 0.01):
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.regularization = regularization

        # Kalman Filter設定
        self.kalman_filters: Dict[str, KalmanFilter] = {}

        # 重み履歴
        self.weight_history: Dict[str, deque] = {}

        logger.info("WeightOptimizer initialized")

    def _initialize_kalman_filter(self, model_id: str) -> KalmanFilter:
        """Kalman Filter初期化"""
        kf = KalmanFilter(dim_x=2, dim_z=1)

        # 状態変数: [重み, 重み変化率]
        kf.x = np.array([[0.5], [0.0]])  # 初期重み=0.5, 変化率=0

        # 状態遷移モデル
        dt = 1.0
        kf.F = np.array([[1.0, dt],
                         [0.0, 1.0]])

        # 観測モデル
        kf.H = np.array([[1.0, 0.0]])

        # ノイズ
        kf.Q = np.array([[0.01, 0.0],
                         [0.0, 0.001]])  # プロセスノイズ
        kf.R = np.array([[0.1]])  # 観測ノイズ

        # 初期共分散
        kf.P = np.eye(2) * 0.1

        return kf

    async def optimize_weights(self,
                             performances: Dict[str, ModelPerformance],
                             regime: MarketRegime,
                             current_weights: Dict[str, float]) -> Dict[str, float]:
        """重み最適化"""
        try:
            if not performances:
                return current_weights

            model_ids = list(performances.keys())
            n_models = len(model_ids)

            if n_models == 0:
                return {}

            # パフォーマンススコア計算
            scores = self._calculate_performance_scores(performances, regime)

            # 制約付き最適化
            optimized_weights = await self._constrained_optimization(
                model_ids, scores, current_weights, regime
            )

            # Kalman Filter更新
            filtered_weights = self._apply_kalman_filtering(
                model_ids, optimized_weights, current_weights
            )

            # 制約適用
            final_weights = self._apply_constraints(filtered_weights)

            # 重み履歴更新
            self._update_weight_history(final_weights)

            return final_weights

        except Exception as e:
            logger.error(f"Error in weight optimization: {e}")
            return current_weights

    def _calculate_performance_scores(self,
                                    performances: Dict[str, ModelPerformance],
                                    regime: MarketRegime) -> Dict[str, float]:
        """パフォーマンススコア計算"""
        scores = {}

        for model_id, performance in performances.items():
            # ベーススコア
            base_score = performance.accuracy

            # レジーム別調整
            regime_bonus = 0.0
            if regime.regime_id in performance.regime_specific_performance:
                regime_perf = performance.regime_specific_performance[regime.regime_id]
                regime_bonus = (regime_perf['accuracy'] - base_score) * regime.confidence

            # 安定性ボーナス
            stability_bonus = max(0, -performance.max_drawdown) * 0.1

            # 予測数ペナルティ（データ不足）
            data_penalty = max(0, 0.2 - performance.predictions_count / 100.0)

            # 総合スコア
            total_score = base_score + regime_bonus + stability_bonus - data_penalty
            scores[model_id] = max(0.01, total_score)  # 最小値保証

        return scores

    async def _constrained_optimization(self,
                                      model_ids: List[str],
                                      scores: Dict[str, float],
                                      current_weights: Dict[str, float],
                                      regime: MarketRegime) -> Dict[str, float]:
        """制約付き最適化"""
        try:
            n_models = len(model_ids)

            # 初期重み
            x0 = np.array([current_weights.get(mid, 1.0/n_models) for mid in model_ids])

            # 目的関数（負のスコア和を最小化）
            def objective(weights):
                total_score = sum(weights[i] * scores[model_ids[i]] for i in range(n_models))
                # 正則化項（重みの分散ペナルティ）
                diversity_penalty = self.regularization * np.var(weights)
                return -(total_score - diversity_penalty)

            # 制約
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # 重みの和=1
            ]

            # 境界
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_models)]

            # 最適化実行
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            if result.success:
                optimized_weights = dict(zip(model_ids, result.x))
            else:
                logger.warning("Optimization failed, using uniform weights")
                uniform_weight = 1.0 / n_models
                optimized_weights = {mid: uniform_weight for mid in model_ids}

            return optimized_weights

        except Exception as e:
            logger.error(f"Error in constrained optimization: {e}")
            # フォールバック: 均等重み
            uniform_weight = 1.0 / len(model_ids)
            return {mid: uniform_weight for mid in model_ids}

    def _apply_kalman_filtering(self,
                              model_ids: List[str],
                              optimized_weights: Dict[str, float],
                              current_weights: Dict[str, float]) -> Dict[str, float]:
        """Kalman Filter適用"""
        filtered_weights = {}

        for model_id in model_ids:
            # Kalman Filter初期化（未作成の場合）
            if model_id not in self.kalman_filters:
                self.kalman_filters[model_id] = self._initialize_kalman_filter(model_id)
                # 現在の重みで初期化
                current_weight = current_weights.get(model_id, 1.0/len(model_ids))
                self.kalman_filters[model_id].x[0] = current_weight

            kf = self.kalman_filters[model_id]

            # 予測ステップ
            kf.predict()

            # 更新ステップ
            optimized_weight = optimized_weights.get(model_id, kf.x[0])
            kf.update(np.array([[optimized_weight]]))

            # フィルタ済み重み
            filtered_weights[model_id] = float(kf.x[0])

        return filtered_weights

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """制約適用"""
        if not weights:
            return weights

        # 最小・最大制約
        constrained = {}
        for model_id, weight in weights.items():
            constrained[model_id] = np.clip(weight, self.min_weight, self.max_weight)

        # 正規化（合計=1）
        total = sum(constrained.values())
        if total > 0:
            for model_id in constrained:
                constrained[model_id] /= total
        else:
            # 全て0の場合は均等重み
            uniform_weight = 1.0 / len(constrained)
            constrained = {mid: uniform_weight for mid in constrained}

        return constrained

    def _update_weight_history(self, weights: Dict[str, float]) -> None:
        """重み履歴更新"""
        for model_id, weight in weights.items():
            if model_id not in self.weight_history:
                self.weight_history[model_id] = deque(maxlen=100)

            self.weight_history[model_id].append({
                'weight': weight,
                'timestamp': time.time()
            })

class AdaptiveWeightingEngine:
    """動的重み付けエンジン"""

    def __init__(self,
                 models: List[Any],
                 initial_weights: Optional[Dict[str, float]] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.models = models
        self.model_ids = [f"model_{i}" for i in range(len(models))]

        # 初期重み設定
        if initial_weights:
            self.weights = initial_weights
        else:
            uniform_weight = 1.0 / len(models) if models else 1.0
            self.weights = {mid: uniform_weight for mid in self.model_ids}

        # 設定
        self.config = config or {}

        # コンポーネント初期化
        self.regime_detector = MarketRegimeDetector(
            lookback_window=self.config.get('lookback_window', 252),
            n_regimes=self.config.get('n_regimes', 3)
        )

        self.performance_tracker = PerformanceTracker(
            window_size=self.config.get('performance_window', 100)
        )

        self.weight_optimizer = WeightOptimizer(
            momentum=self.config.get('weight_momentum', 0.9),
            min_weight=self.config.get('min_weight', 0.01),
            max_weight=self.config.get('max_weight', 0.5),
            regularization=self.config.get('regularization', 0.01)
        )

        # 状態管理
        self.current_regime: Optional[MarketRegime] = None
        self.weight_updates: List[WeightUpdate] = []
        self.is_initialized = False

        logger.info(f"AdaptiveWeightingEngine initialized with {len(models)} models")

    async def initialize(self, historical_data: pd.DataFrame) -> None:
        """システム初期化"""
        try:
            # レジーム検出器学習
            await self.regime_detector.fit_regime_models(historical_data)

            self.is_initialized = True
            logger.info("AdaptiveWeightingEngine initialization completed")

        except Exception as e:
            logger.error(f"Error initializing AdaptiveWeightingEngine: {e}")
            self.is_initialized = False

    async def update_weights(self,
                           market_data: pd.DataFrame,
                           predictions: Dict[str, np.ndarray],
                           targets: np.ndarray) -> Dict[str, float]:
        """重み更新"""
        try:
            # レジーム検出
            regime = await self.regime_detector.detect_regime(market_data)
            self.current_regime = regime

            # パフォーマンス更新
            performances = {}
            for model_id in self.model_ids:
                if model_id in predictions:
                    performance = self.performance_tracker.update_performance(
                        model_id, predictions[model_id], targets, regime.regime_id
                    )
                    performances[model_id] = performance

            # 重み最適化
            new_weights = await self.weight_optimizer.optimize_weights(
                performances, regime, self.weights
            )

            # 重み更新記録
            self._record_weight_updates(new_weights, regime)

            # 重み更新
            self.weights = new_weights

            logger.debug(f"Weights updated for regime {regime.name}: {new_weights}")
            return new_weights

        except Exception as e:
            logger.error(f"Error updating weights: {e}")
            return self.weights

    def _record_weight_updates(self, new_weights: Dict[str, float], regime: MarketRegime) -> None:
        """重み更新記録"""
        current_time = time.time()

        for model_id, new_weight in new_weights.items():
            old_weight = self.weights.get(model_id, 0.0)

            if abs(new_weight - old_weight) > 0.01:  # 意味のある変更のみ記録
                update = WeightUpdate(
                    model_id=model_id,
                    old_weight=old_weight,
                    new_weight=new_weight,
                    reason=f"Regime: {regime.name}, Confidence: {regime.confidence:.3f}",
                    confidence=regime.confidence,
                    timestamp=current_time
                )
                self.weight_updates.append(update)

                # 履歴上限管理
                if len(self.weight_updates) > 1000:
                    self.weight_updates = self.weight_updates[-500:]

    def get_current_weights(self) -> Dict[str, float]:
        """現在の重み取得"""
        return self.weights.copy()

    def get_current_regime(self) -> Optional[MarketRegime]:
        """現在のレジーム取得"""
        return self.current_regime

    def get_weight_history(self) -> List[WeightUpdate]:
        """重み更新履歴取得"""
        return self.weight_updates.copy()

    def get_model_performances(self) -> Dict[str, ModelPerformance]:
        """モデルパフォーマンス取得"""
        return self.performance_tracker.get_all_performances()

    async def predict_ensemble(self, input_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """アンサンブル予測"""
        try:
            predictions = {}

            # 各モデルで予測
            for i, model in enumerate(self.models):
                model_id = self.model_ids[i]
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(input_data)
                    else:
                        # フォールバック予測
                        pred = np.random.randn(input_data.shape[0], 1) * 0.1

                    predictions[model_id] = pred
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_id}: {e}")
                    predictions[model_id] = np.zeros((input_data.shape[0], 1))

            # 重み付き予測
            weighted_predictions = []
            for i in range(input_data.shape[0]):
                weighted_pred = 0.0
                total_weight = 0.0

                for model_id, pred in predictions.items():
                    weight = self.weights.get(model_id, 0.0)
                    if len(pred) > i:
                        weighted_pred += weight * pred[i]
                        total_weight += weight

                if total_weight > 0:
                    weighted_pred /= total_weight

                weighted_predictions.append(weighted_pred)

            ensemble_prediction = np.array(weighted_predictions).reshape(-1, 1)

            # メタデータ
            metadata = {
                'weights': self.weights.copy(),
                'regime': self.current_regime.__dict__ if self.current_regime else None,
                'individual_predictions': {k: v.tolist() for k, v in predictions.items()},
                'prediction_timestamp': time.time()
            }

            return ensemble_prediction, metadata

        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            # フォールバック予測
            fallback_pred = np.zeros((input_data.shape[0], 1))
            return fallback_pred, {'error': str(e)}

# 便利関数
def create_adaptive_weighting_engine(
    models: List[Any],
    config: Optional[Dict[str, Any]] = None
) -> AdaptiveWeightingEngine:
    """動的重み付けエンジン作成"""
    return AdaptiveWeightingEngine(models=models, config=config)

async def demo_adaptive_weighting():
    """動的重み付けデモ"""
    # サンプルデータ作成
    np.random.seed(42)
    n_samples = 1000

    # 市場データ
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)
    volumes = np.random.lognormal(15, 0.5, n_samples).astype(int)

    market_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': volumes
    })

    # ダミーモデル
    class DummyModel:
        def __init__(self, noise_level=0.1):
            self.noise_level = noise_level

        def predict(self, X):
            return np.random.randn(X.shape[0], 1) * self.noise_level

    models = [DummyModel(0.05), DummyModel(0.1), DummyModel(0.15)]

    # エンジン作成
    engine = create_adaptive_weighting_engine(models)

    # 初期化
    await engine.initialize(market_data[:500])

    # 予測・重み更新
    for i in range(500, 600, 10):
        window_data = market_data.iloc[i-50:i]
        input_data = np.random.randn(5, 10)
        targets = np.random.randn(5, 1)

        # 個別予測
        predictions = {}
        for j, model in enumerate(models):
            model_id = f"model_{j}"
            predictions[model_id] = model.predict(input_data)

        # 重み更新
        new_weights = await engine.update_weights(window_data, predictions, targets)

        # アンサンブル予測
        ensemble_pred, metadata = await engine.predict_ensemble(input_data)

        print(f"Step {i}: Weights = {new_weights}")
        print(f"Regime = {engine.get_current_regime().name if engine.get_current_regime() else 'None'}")
        print(f"Ensemble prediction shape: {ensemble_pred.shape}")
        print("---")

if __name__ == "__main__":
    asyncio.run(demo_adaptive_weighting())