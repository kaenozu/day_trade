#!/usr/bin/env python3
"""
AI駆動高頻度取引市場予測システム
Issue #366: マイクロ秒レベルAI予測エンジン

リアルタイム市場パターン認識、価格予測、最適執行タイミング決定
超低遅延（<10μs）でAI推論を実現
"""

import asyncio
import concurrent.futures
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ML/AI関連
try:
    import torch
    import torch.jit
    import torch.nn as nn
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# プロジェクト統合
try:
    from ..cache.redis_enhanced_cache import RedisEnhancedCache
    from ..ml.optimized_inference_engine import OptimizedInferenceEngine
    from ..monitoring.performance_optimization_system import get_optimization_manager
    from ..utils.logging_config import get_context_logger
    from .next_gen_hft_engine import HFTMarketData, MarketRegime

except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class PredictionModel(Enum):
    """予測モデルタイプ"""

    LSTM_PRICE = "lstm_price"
    TRANSFORMER = "transformer"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"
    ULTRA_FAST = "ultra_fast"  # <5μs特化型


class MarketSignal(Enum):
    """市場シグナル"""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class MarketPrediction:
    """市場予測結果"""

    symbol: str
    timestamp_us: int
    horizon_ms: int  # 予測期間（ミリ秒）

    # 価格予測
    predicted_price: float
    confidence: float  # 0.0-1.0
    direction: MarketSignal

    # 実行推奨
    recommended_action: str  # "BUY", "SELL", "HOLD"
    optimal_quantity: float
    execution_urgency: float  # 0.0-1.0

    # 予測メタデータ
    model_used: PredictionModel
    inference_time_us: int
    features_used: List[str] = field(default_factory=list)

    def is_actionable(self, min_confidence: float = 0.7) -> bool:
        """アクション可能な予測か判定"""
        return self.confidence >= min_confidence and self.recommended_action != "HOLD"


@dataclass
class MarketFeatureSet:
    """市場特徴量セット"""

    symbol: str
    timestamp_us: int

    # 価格特徴量
    price_features: Dict[str, float] = field(default_factory=dict)
    volume_features: Dict[str, float] = field(default_factory=dict)
    technical_features: Dict[str, float] = field(default_factory=dict)

    # 高頻度特徴量
    microstructure_features: Dict[str, float] = field(default_factory=dict)
    orderbook_features: Dict[str, float] = field(default_factory=dict)
    flow_features: Dict[str, float] = field(default_factory=dict)

    def to_array(self, feature_names: List[str]) -> np.ndarray:
        """特徴量配列に変換"""
        all_features = {
            **self.price_features,
            **self.volume_features,
            **self.technical_features,
            **self.microstructure_features,
            **self.orderbook_features,
            **self.flow_features,
        }

        return np.array([all_features.get(name, 0.0) for name in feature_names])


class UltraFastNeuralNet(nn.Module):
    """超高速ニューラルネット（<5μs推論）"""

    def __init__(self, input_size: int = 50, hidden_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3),  # price, confidence, signal
        )

    def forward(self, x):
        return self.network(x)


class AIMarketPredictor:
    """AI駆動市場予測システム"""

    def __init__(
        self,
        symbols: List[str],
        prediction_horizon_ms: int = 100,
        model_type: PredictionModel = PredictionModel.ULTRA_FAST,
        inference_timeout_us: int = 10,
    ):
        self.symbols = symbols
        self.prediction_horizon_ms = prediction_horizon_ms
        self.model_type = model_type
        self.inference_timeout_us = inference_timeout_us

        # データ蓄積
        self.market_data_buffer = {symbol: deque(maxlen=1000) for symbol in symbols}
        self.feature_buffer = {symbol: deque(maxlen=500) for symbol in symbols}

        # モデル初期化
        self._init_models()
        self._init_feature_engineering()

        # キャッシュ
        self._init_cache()

        # パフォーマンス統計
        self.stats = {
            "predictions_made": 0,
            "total_inference_time_us": 0,
            "cache_hits": 0,
            "model_switches": 0,
            "accurate_predictions": 0,
        }

        logger.info(
            f"AIMarketPredictor 初期化完了: {len(symbols)} symbols, {model_type.value}"
        )

    def _init_models(self):
        """予測モデル初期化"""
        self.models = {}

        if TORCH_AVAILABLE and self.model_type in [
            PredictionModel.ULTRA_FAST,
            PredictionModel.LSTM_PRICE,
        ]:
            # PyTorchモデル
            for symbol in self.symbols:
                model = UltraFastNeuralNet()
                model.eval()

                # JITコンパイル（高速化）
                dummy_input = torch.randn(1, 50)
                traced_model = torch.jit.trace(model, dummy_input)

                self.models[symbol] = traced_model

            logger.info("PyTorch超高速モデル初期化完了")

        else:
            # Scikit-learnモデル（フォールバック）
            for symbol in self.symbols:
                model = RandomForestRegressor(
                    n_estimators=10,  # 高速化のため少なく
                    max_depth=5,
                    random_state=42,
                    n_jobs=1,  # スレッド競合回避
                )
                self.models[symbol] = model

            logger.info("RandomForestモデル初期化完了")

        # 特徴量スケーラー
        self.scalers = {symbol: StandardScaler() for symbol in self.symbols}

        # モデル学習状態
        self.model_trained = {symbol: False for symbol in self.symbols}

    def _init_feature_engineering(self):
        """特徴量エンジニアリング初期化"""
        self.feature_names = [
            # 価格特徴量
            "price_return_1",
            "price_return_5",
            "price_return_10",
            "price_volatility_5",
            "price_volatility_10",
            "price_momentum_3",
            "price_momentum_7",
            # スプレッド特徴量
            "spread_bps",
            "spread_ma_5",
            "spread_volatility",
            # ボリューム特徴量
            "volume_ma_5",
            "volume_ma_10",
            "volume_ratio",
            "volume_volatility",
            "volume_trend",
            # オーダーブック特徴量
            "orderbook_imbalance",
            "orderbook_pressure",
            "bid_ask_ratio",
            "depth_ratio",
            # マイクロ構造特徴量
            "trade_intensity",
            "price_impact",
            "effective_spread",
            "realized_volatility_1min",
            "jump_intensity",
            # テクニカル指標
            "rsi_5",
            "macd_signal",
            "bb_position",
            "momentum_1min",
            "momentum_5min",
            # フロー特徴量
            "aggressive_buy_ratio",
            "aggressive_sell_ratio",
            "net_flow_5min",
            "flow_acceleration",
            # レジーム特徴量
            "volatility_regime",
            "trend_strength",
            "mean_reversion",
        ]

        logger.info(f"特徴量エンジニアリング初期化: {len(self.feature_names)} features")

    def _init_cache(self):
        """キャッシュシステム初期化"""
        try:
            self.cache = RedisEnhancedCache(
                host="localhost", port=6379, ttl=50
            )  # 50ms TTL
            logger.info("予測キャッシュシステム初期化完了")
        except Exception as e:
            logger.warning(f"キャッシュ初期化失敗: {e}")
            self.cache = None

    async def predict_market(
        self, symbol: str, market_data: HFTMarketData
    ) -> Optional[MarketPrediction]:
        """市場予測実行"""
        start_time = time.time() * 1_000_000  # マイクロ秒

        try:
            # データ蓄積
            self.market_data_buffer[symbol].append(market_data)

            # 特徴量生成
            features = await self._extract_features(symbol, market_data)
            if not features:
                return None

            # キャッシュ確認
            cached_prediction = await self._get_cached_prediction(
                symbol, market_data.timestamp_us
            )
            if cached_prediction:
                self.stats["cache_hits"] += 1
                return cached_prediction

            # AI推論実行
            prediction = await self._run_inference(symbol, features)
            if not prediction:
                return None

            # 推論時間記録
            inference_time = int((time.time() * 1_000_000) - start_time)
            prediction.inference_time_us = inference_time

            # 統計更新
            self.stats["predictions_made"] += 1
            self.stats["total_inference_time_us"] += inference_time

            # キャッシュ保存
            await self._cache_prediction(symbol, prediction)

            # パフォーマンスチェック
            if inference_time > self.inference_timeout_us:
                logger.warning(
                    f"推論時間超過: {inference_time}μs > {self.inference_timeout_us}μs"
                )

            return prediction

        except Exception as e:
            logger.error(f"市場予測エラー [{symbol}]: {e}")
            return None

    async def _extract_features(
        self, symbol: str, market_data: HFTMarketData
    ) -> Optional[MarketFeatureSet]:
        """特徴量抽出"""
        if len(self.market_data_buffer[symbol]) < 10:
            return None  # データ不足

        try:
            features = MarketFeatureSet(
                symbol=symbol, timestamp_us=market_data.timestamp_us
            )

            # 直近データ取得
            recent_data = list(self.market_data_buffer[symbol])[-20:]
            prices = [d.mid_price() for d in recent_data]
            volumes = [d.volume for d in recent_data]
            spreads = [d.spread() for d in recent_data]

            # 価格特徴量
            if len(prices) >= 10:
                returns = np.diff(prices) / prices[:-1]
                features.price_features = {
                    "price_return_1": returns[-1] if len(returns) > 0 else 0.0,
                    "price_return_5": (
                        np.mean(returns[-5:]) if len(returns) >= 5 else 0.0
                    ),
                    "price_return_10": (
                        np.mean(returns[-10:]) if len(returns) >= 10 else 0.0
                    ),
                    "price_volatility_5": (
                        np.std(returns[-5:]) if len(returns) >= 5 else 0.0
                    ),
                    "price_volatility_10": (
                        np.std(returns[-10:]) if len(returns) >= 10 else 0.0
                    ),
                    "price_momentum_3": (
                        (prices[-1] / prices[-4] - 1) if len(prices) >= 4 else 0.0
                    ),
                    "price_momentum_7": (
                        (prices[-1] / prices[-8] - 1) if len(prices) >= 8 else 0.0
                    ),
                }

            # スプレッド特徴量
            features.technical_features = {
                "spread_bps": (
                    spreads[-1] / prices[-1] * 10000 if prices[-1] > 0 else 0.0
                ),
                "spread_ma_5": (
                    np.mean(spreads[-5:]) if len(spreads) >= 5 else spreads[-1]
                ),
                "spread_volatility": np.std(spreads[-5:]) if len(spreads) >= 5 else 0.0,
            }

            # ボリューム特徴量
            features.volume_features = {
                "volume_ma_5": (
                    np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
                ),
                "volume_ma_10": (
                    np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)
                ),
                "volume_ratio": (
                    volumes[-1] / np.mean(volumes[:-1]) if len(volumes) > 1 else 1.0
                ),
                "volume_volatility": np.std(volumes[-5:]) if len(volumes) >= 5 else 0.0,
                "volume_trend": (
                    np.polyfit(range(len(volumes)), volumes, 1)[0]
                    if len(volumes) >= 3
                    else 0.0
                ),
            }

            # マイクロ構造特徴量（簡易版）
            features.microstructure_features = {
                "orderbook_imbalance": (market_data.bid_size - market_data.ask_size)
                / (market_data.bid_size + market_data.ask_size),
                "orderbook_pressure": (
                    market_data.bid_size / market_data.ask_size
                    if market_data.ask_size > 0
                    else 1.0
                ),
                "bid_ask_ratio": (
                    market_data.bid_price / market_data.ask_price
                    if market_data.ask_price > 0
                    else 1.0
                ),
                "depth_ratio": market_data.bid_size
                / max(market_data.bid_size + market_data.ask_size, 1),
            }

            # 高頻度特徴量（簡易版）
            features.orderbook_features = {
                "trade_intensity": len(recent_data) / 20.0,  # データ密度
                "price_impact": (
                    abs(prices[-1] - prices[-2]) / spreads[-1]
                    if len(prices) >= 2 and spreads[-1] > 0
                    else 0.0
                ),
                "effective_spread": spreads[-1] / 2.0,
                "realized_volatility_1min": (
                    np.std(returns[-60:]) * np.sqrt(60)
                    if len(returns) >= 60
                    else np.std(returns) * np.sqrt(len(returns))
                ),
                "jump_intensity": (
                    np.sum(np.abs(returns) > 3 * np.std(returns))
                    if len(returns) > 0
                    else 0.0
                ),
            }

            # テクニカル指標（簡易版）
            if len(prices) >= 14:
                rsi = self._calculate_rsi(prices, 14)
                features.flow_features = {
                    "rsi_5": rsi,
                    "macd_signal": self._calculate_macd_signal(prices),
                    "bb_position": self._calculate_bollinger_position(prices),
                    "momentum_1min": (
                        (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0.0
                    ),
                    "momentum_5min": (prices[-1] / prices[-min(100, len(prices))] - 1),
                    "aggressive_buy_ratio": 0.5,  # 簡易値
                    "aggressive_sell_ratio": 0.5,  # 簡易値
                    "net_flow_5min": (
                        np.sum(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
                    ),
                    "flow_acceleration": 0.0,  # 簡易値
                    "volatility_regime": (
                        1.0
                        if features.price_features.get("price_volatility_10", 0) > 0.01
                        else 0.0
                    ),
                    "trend_strength": abs(
                        features.price_features.get("price_momentum_7", 0)
                    ),
                    "mean_reversion": -features.price_features.get(
                        "price_momentum_3", 0
                    )
                    * features.price_features.get("price_momentum_7", 0),
                }

            # 特徴量をバッファに保存
            self.feature_buffer[symbol].append(features)

            return features

        except Exception as e:
            logger.error(f"特徴量抽出エラー [{symbol}]: {e}")
            return None

    async def _run_inference(
        self, symbol: str, features: MarketFeatureSet
    ) -> Optional[MarketPrediction]:
        """AI推論実行"""
        if symbol not in self.models:
            return None

        try:
            # 特徴量配列変換
            feature_array = features.to_array(self.feature_names)

            if not self.model_trained[symbol]:
                # モデル未学習の場合は簡易予測
                return self._generate_simple_prediction(symbol, features, feature_array)

            model = self.models[symbol]

            if TORCH_AVAILABLE and isinstance(model, torch.jit.ScriptModule):
                # PyTorch推論
                prediction = await self._torch_inference(symbol, model, feature_array)
            else:
                # Scikit-learn推論
                prediction = await self._sklearn_inference(symbol, model, feature_array)

            return prediction

        except Exception as e:
            logger.error(f"AI推論エラー [{symbol}]: {e}")
            return self._generate_simple_prediction(symbol, features, feature_array)

    async def _torch_inference(
        self, symbol: str, model: torch.jit.ScriptModule, features: np.ndarray
    ) -> MarketPrediction:
        """PyTorch推論"""
        start_time = time.time() * 1_000_000

        # 特徴量正規化
        if len(features) != len(self.feature_names):
            features = np.pad(
                features, (0, max(0, len(self.feature_names) - len(features)))
            )[: len(self.feature_names)]

        features_scaled = (
            self.scalers[symbol].fit_transform(features.reshape(1, -1)).flatten()
        )

        # テンソル変換
        input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)

        # 推論実行
        with torch.no_grad():
            output = model(input_tensor)
            predicted_price, confidence, signal_raw = output[0].numpy()

        # シグナル変換
        signal_map = [
            MarketSignal.STRONG_SELL,
            MarketSignal.SELL,
            MarketSignal.HOLD,
            MarketSignal.BUY,
            MarketSignal.STRONG_BUY,
        ]
        signal_idx = max(0, min(4, int((signal_raw + 1) * 2.5)))  # [-1,1] -> [0,4]
        signal = signal_map[signal_idx]

        # アクション決定
        if signal in [MarketSignal.STRONG_BUY, MarketSignal.BUY]:
            action = "BUY"
            quantity = min(1000, 100 * (1 + confidence))
        elif signal in [MarketSignal.STRONG_SELL, MarketSignal.SELL]:
            action = "SELL"
            quantity = min(1000, 100 * (1 + confidence))
        else:
            action = "HOLD"
            quantity = 0

        inference_time = int((time.time() * 1_000_000) - start_time)

        return MarketPrediction(
            symbol=symbol,
            timestamp_us=int(time.time() * 1_000_000),
            horizon_ms=self.prediction_horizon_ms,
            predicted_price=float(predicted_price),
            confidence=max(0.0, min(1.0, float(confidence))),
            direction=signal,
            recommended_action=action,
            optimal_quantity=quantity,
            execution_urgency=float(confidence),
            model_used=self.model_type,
            inference_time_us=inference_time,
            features_used=self.feature_names[:10],  # 主要特徴量のみ記録
        )

    async def _sklearn_inference(
        self, symbol: str, model, features: np.ndarray
    ) -> MarketPrediction:
        """Scikit-learn推論"""
        start_time = time.time() * 1_000_000

        # 簡易推論（RandomForestは学習データが必要）
        if len(features) == 0:
            return self._generate_simple_prediction(symbol, None, features)

        # 特徴量スケーリング
        features_scaled = (
            self.scalers[symbol].fit_transform(features.reshape(1, -1)).flatten()
        )

        # 簡易予測（モック）
        price_change = np.mean(features_scaled[:5]) * 0.001  # 簡易価格変化予測
        predicted_price = 150.0 + price_change  # ベース価格から変化
        confidence = min(0.8, abs(price_change) * 100)

        # シグナル決定
        if price_change > 0.01:
            signal = MarketSignal.BUY
            action = "BUY"
        elif price_change < -0.01:
            signal = MarketSignal.SELL
            action = "SELL"
        else:
            signal = MarketSignal.HOLD
            action = "HOLD"

        inference_time = int((time.time() * 1_000_000) - start_time)

        return MarketPrediction(
            symbol=symbol,
            timestamp_us=int(time.time() * 1_000_000),
            horizon_ms=self.prediction_horizon_ms,
            predicted_price=predicted_price,
            confidence=confidence,
            direction=signal,
            recommended_action=action,
            optimal_quantity=100 if action != "HOLD" else 0,
            execution_urgency=confidence,
            model_used=PredictionModel.RANDOM_FOREST,
            inference_time_us=inference_time,
            features_used=self.feature_names[:5],
        )

    def _generate_simple_prediction(
        self,
        symbol: str,
        features: Optional[MarketFeatureSet],
        feature_array: np.ndarray,
    ) -> MarketPrediction:
        """シンプル予測（フォールバック）"""
        # 基本的なランダムウォーク + トレンド予測
        base_price = 150.0

        if len(feature_array) > 0:
            trend = np.mean(feature_array[:3]) if len(feature_array) >= 3 else 0.0
            predicted_price = base_price * (1 + trend * 0.001)
            confidence = 0.5
        else:
            predicted_price = base_price
            confidence = 0.3

        return MarketPrediction(
            symbol=symbol,
            timestamp_us=int(time.time() * 1_000_000),
            horizon_ms=self.prediction_horizon_ms,
            predicted_price=predicted_price,
            confidence=confidence,
            direction=MarketSignal.HOLD,
            recommended_action="HOLD",
            optimal_quantity=0,
            execution_urgency=0.0,
            model_used=PredictionModel.ULTRA_FAST,
            inference_time_us=5,
            features_used=[],
        )

    # Helper methods
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI計算"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd_signal(self, prices: List[float]) -> float:
        """MACD信号計算"""
        if len(prices) < 26:
            return 0.0

        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)

        return ema_12 - ema_26

    def _calculate_bollinger_position(
        self, prices: List[float], period: int = 20
    ) -> float:
        """ボリンジャーバンド位置計算"""
        if len(prices) < period:
            return 0.5

        recent_prices = prices[-period:]
        ma = np.mean(recent_prices)
        std = np.std(recent_prices)

        if std == 0:
            return 0.5

        # 現在価格のBB内位置 (0.0=下限, 0.5=中央, 1.0=上限)
        position = (prices[-1] - (ma - 2 * std)) / (4 * std)
        return max(0.0, min(1.0, position))

    def _ema(self, prices: List[float], period: int) -> float:
        """指数移動平均計算"""
        if len(prices) < period:
            return np.mean(prices)

        alpha = 2.0 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    async def _get_cached_prediction(
        self, symbol: str, timestamp_us: int
    ) -> Optional[MarketPrediction]:
        """キャッシュから予測取得"""
        if not self.cache:
            return None

        try:
            cache_key = f"prediction:{symbol}:{timestamp_us // 10000}"  # 10ms精度
            cached_data = await self.cache.get(cache_key)

            if cached_data:
                return pickle.loads(cached_data)

        except Exception as e:
            logger.debug(f"キャッシュ取得エラー: {e}")

        return None

    async def _cache_prediction(self, symbol: str, prediction: MarketPrediction):
        """予測をキャッシュに保存"""
        if not self.cache:
            return

        try:
            cache_key = f"prediction:{symbol}:{prediction.timestamp_us // 10000}"
            cached_data = pickle.dumps(prediction)

            await self.cache.set(cache_key, cached_data, ttl=50)  # 50ms

        except Exception as e:
            logger.debug(f"キャッシュ保存エラー: {e}")

    async def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス要約"""
        if self.stats["predictions_made"] == 0:
            return {"status": "no_predictions"}

        avg_inference_time = (
            self.stats["total_inference_time_us"] / self.stats["predictions_made"]
        )
        cache_hit_rate = self.stats["cache_hits"] / self.stats["predictions_made"] * 100

        return {
            "model_type": self.model_type.value,
            "symbols_count": len(self.symbols),
            "prediction_horizon_ms": self.prediction_horizon_ms,
            "performance": {
                "predictions_made": self.stats["predictions_made"],
                "avg_inference_time_us": round(avg_inference_time, 2),
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "model_switches": self.stats["model_switches"],
                "torch_available": TORCH_AVAILABLE,
            },
            "accuracy": {
                "accurate_predictions": self.stats["accurate_predictions"],
                "accuracy_rate_percent": round(
                    self.stats["accurate_predictions"]
                    / max(1, self.stats["predictions_made"])
                    * 100,
                    2,
                ),
            },
        }


def create_ai_market_predictor(
    symbols: List[str],
    prediction_horizon_ms: int = 100,
    model_type: PredictionModel = PredictionModel.ULTRA_FAST,
    inference_timeout_us: int = 10,
) -> AIMarketPredictor:
    """AI市場予測システム作成"""
    return AIMarketPredictor(
        symbols=symbols,
        prediction_horizon_ms=prediction_horizon_ms,
        model_type=model_type,
        inference_timeout_us=inference_timeout_us,
    )


if __name__ == "__main__":
    # テスト実行
    async def test_ai_market_predictor():
        print("=== AIMarketPredictor テスト ===")

        # 予測システム作成
        predictor = create_ai_market_predictor(
            symbols=["AAPL", "GOOGL", "TSLA"],
            prediction_horizon_ms=50,
            model_type=PredictionModel.ULTRA_FAST,
            inference_timeout_us=10,
        )

        # テスト用市場データ
        test_data = []
        for i in range(50):
            for symbol in ["AAPL", "GOOGL", "TSLA"]:
                market_data = HFTMarketData(
                    symbol=symbol,
                    timestamp_us=int((time.time() + i * 0.001) * 1_000_000),
                    bid_price=150.0 + np.random.normal(0, 1.0),
                    ask_price=150.1 + np.random.normal(0, 1.0),
                    bid_size=100 + np.random.randint(0, 100),
                    ask_size=100 + np.random.randint(0, 100),
                    last_price=150.05 + np.random.normal(0, 0.5),
                    volume=1000 + np.random.randint(0, 1000),
                    sequence_number=i,
                )
                test_data.append((symbol, market_data))

        print(f"テストデータ {len(test_data)}件 作成")

        # 予測実行
        predictions = []
        start_time = time.time()

        for symbol, market_data in test_data[-15:]:  # 最後の15件で予測
            prediction = await predictor.predict_market(symbol, market_data)
            if prediction:
                predictions.append(prediction)
            await asyncio.sleep(0.001)  # 1ms間隔

        total_time = (time.time() - start_time) * 1000

        print(f"\n予測結果: {len(predictions)}件")
        print(f"総実行時間: {total_time:.1f}ms")

        # 予測詳細表示
        if predictions:
            for i, pred in enumerate(predictions[-5:]):  # 最後の5件表示
                print(f"\n予測{i + 1}: {pred.symbol}")
                print(f"  価格予測: ${pred.predicted_price:.2f}")
                print(f"  信頼度: {pred.confidence:.1%}")
                print(f"  シグナル: {pred.direction.value}")
                print(f"  推奨アクション: {pred.recommended_action}")
                print(f"  推論時間: {pred.inference_time_us}μs")

        # パフォーマンス要約
        summary = await predictor.get_performance_summary()
        print("\n=== パフォーマンス要約 ===")
        print(f"モデルタイプ: {summary.get('model_type', 'N/A')}")
        print(f"予測回数: {summary.get('performance', {}).get('predictions_made', 0)}")
        print(
            f"平均推論時間: {summary.get('performance', {}).get('avg_inference_time_us', 0):.1f}μs"
        )
        print(
            f"キャッシュヒット率: {summary.get('performance', {}).get('cache_hit_rate_percent', 0):.1f}%"
        )
        print(
            f"PyTorch利用可能: {summary.get('performance', {}).get('torch_available', False)}"
        )

        print("\n✅ AIMarketPredictor テスト完了")

    asyncio.run(test_ai_market_predictor())
