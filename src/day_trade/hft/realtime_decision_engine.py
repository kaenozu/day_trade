#!/usr/bin/env python3
"""
リアルタイム取引決定エンジン
Issue #366: 高頻度取引最適化エンジン - AI決定システム

<1ms意思決定、リアルタイムパターン認識、
マイクロ秒精度リスク評価による超高速取引決定
"""

import asyncio
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ONNX Runtime (optional)
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    warnings.warn(
        "ONNXRuntimeが利用できません。AI推論機能は制限されます。",
        UserWarning,
        stacklevel=2,
    )

# プロジェクトモジュール
try:
    from ..distributed.distributed_computing_manager import (
        DistributedComputingManager,
        DistributedTask,
        TaskType,
    )
    from ..utils.logging_config import get_context_logger, log_performance_metric
    from .market_data_processor import (
        MarketUpdate,
        OrderBook,
        UltraFastMarketDataProcessor,
    )
    from .ultra_fast_executor import OrderEntry, OrderSide, OrderType
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    def log_performance_metric(*args, **kwargs):
        pass

    # モッククラス
    class MarketUpdate:
        def __init__(self, **kwargs):
            self.symbol_id = kwargs.get("symbol_id", 0)
            self.price = kwargs.get("price", 0)
            self.size = kwargs.get("size", 0)

    class OrderBook:
        def get_mid_price(self):
            return 10000

        def get_spread(self):
            return 10

    class OrderEntry:
        def __init__(self, **kwargs):
            pass

    class OrderSide:
        BUY = 1
        SELL = -1

    class OrderType:
        MARKET = 1
        LIMIT = 2

    class DistributedComputingManager:
        def __init__(self):
            pass

        async def execute_distributed_task(self, task):
            return type(
                "MockResult", (), {"success": True, "result": np.array([0.5])}
            )()


logger = get_context_logger(__name__)


class TradingAction(IntEnum):
    """取引アクション"""

    HOLD = 0
    BUY = 1
    SELL = -1
    STRONG_BUY = 2
    STRONG_SELL = -2


class SignalStrength(IntEnum):
    """シグナル強度"""

    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


class MarketRegime(IntEnum):
    """市場局面"""

    TRENDING_UP = 1
    TRENDING_DOWN = -1
    SIDEWAYS = 0
    VOLATILE = 2
    QUIET = -2


@dataclass
class MarketFeatures:
    """
    市場特徴量（高速計算最適化）

    リアルタイム特徴量抽出：<200μs target
    """

    # Price features
    mid_price: float = 0.0
    spread_bps: float = 0.0
    price_change_bps: float = 0.0

    # Volume features
    volume_ratio: float = 1.0
    volume_weighted_price: float = 0.0

    # Order book features
    bid_ask_imbalance: float = 0.0
    book_pressure: float = 0.0
    top_of_book_ratio: float = 1.0

    # Technical indicators (pre-computed)
    rsi_14: float = 50.0
    ema_ratio: float = 1.0
    bollinger_position: float = 0.5

    # Microstructure features
    effective_spread_bps: float = 0.0
    realized_volatility: float = 0.0
    tick_direction: int = 0

    # Timing features
    time_since_last_trade_ms: float = 0.0
    market_hours_normalized: float = 0.5

    # Additional context
    market_regime: MarketRegime = MarketRegime.SIDEWAYS
    volatility_percentile: float = 50.0

    def to_numpy_array(self) -> np.ndarray:
        """NumPy配列変換（ML推論用）"""
        return np.array(
            [
                self.mid_price,
                self.spread_bps,
                self.price_change_bps,
                self.volume_ratio,
                self.volume_weighted_price,
                self.bid_ask_imbalance,
                self.book_pressure,
                self.top_of_book_ratio,
                self.rsi_14,
                self.ema_ratio,
                self.bollinger_position,
                self.effective_spread_bps,
                self.realized_volatility,
                self.tick_direction,
                self.time_since_last_trade_ms,
                self.market_hours_normalized,
                float(self.market_regime),
                self.volatility_percentile,
            ],
            dtype=np.float32,
        )


@dataclass
class TradingSignal:
    """取引シグナル"""

    symbol_id: int
    action: TradingAction
    strength: SignalStrength
    confidence: float  # 0.0-1.0
    target_price: float
    suggested_quantity: int
    max_position_size: int
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    # Signal metadata
    signal_source: str = "ai_model"
    generation_time_ns: int = field(default_factory=time.perf_counter_ns)
    features_used: Optional[MarketFeatures] = None

    # Execution hints
    urgency_level: int = 1  # 1-5
    execution_time_limit_ms: int = 1000

    def is_actionable(
        self,
        min_confidence: float = 0.6,
        min_strength: SignalStrength = SignalStrength.MODERATE,
    ) -> bool:
        """実行可能シグナル判定"""
        return (
            self.confidence >= min_confidence
            and self.strength >= min_strength
            and self.action != TradingAction.HOLD
        )


@dataclass
class DecisionResult:
    """決定結果"""

    symbol_id: int
    decision: TradingAction
    confidence: float
    suggested_orders: List[OrderEntry] = field(default_factory=list)
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    processing_time_us: float = 0.0

    # Decision context
    market_features: Optional[MarketFeatures] = None
    ai_prediction: Optional[np.ndarray] = None
    rule_based_score: float = 0.0
    ensemble_score: float = 0.0


class FastFeatureExtractor:
    """高速特徴量抽出器（<200μs target）"""

    def __init__(self, lookback_window: int = 100):
        self.lookback_window = lookback_window

        # Price history ring buffers (per symbol)
        self.price_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=lookback_window)
        )
        self.volume_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=lookback_window)
        )
        self.timestamp_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=lookback_window)
        )

        # Pre-computed indicators cache
        self.indicator_cache: Dict[int, Dict[str, float]] = defaultdict(dict)

        # Performance counters
        self.extraction_count = 0
        self.total_extraction_time_ns = 0

    def extract_features(
        self, market_update: MarketUpdate, order_book: Optional[OrderBook] = None
    ) -> MarketFeatures:
        """
        特徴量抽出 (<200μs target)

        Args:
            market_update: 市場データ更新
            order_book: Order Book (optional)

        Returns:
            MarketFeatures: 抽出された特徴量
        """
        start_time = time.perf_counter_ns()

        try:
            symbol_id = market_update.symbol_id
            current_price = market_update.price / 10000.0  # Convert from fixed-point

            # Update history buffers
            self._update_history_buffers(symbol_id, market_update)

            # Extract basic price features
            features = MarketFeatures()
            features.mid_price = current_price

            # Price change calculation
            if len(self.price_history[symbol_id]) > 1:
                prev_price = list(self.price_history[symbol_id])[-2]
                features.price_change_bps = (
                    (current_price - prev_price) / prev_price
                ) * 10000

            # Order book features (if available)
            if order_book:
                features = self._extract_orderbook_features(features, order_book)

            # Technical indicators (cached/fast computation)
            features = self._extract_technical_indicators(features, symbol_id)

            # Microstructure features
            features = self._extract_microstructure_features(
                features, symbol_id, market_update
            )

            # Timing features
            features.market_hours_normalized = self._get_market_hours_normalized()

            self.extraction_count += 1
            return features

        finally:
            end_time = time.perf_counter_ns()
            extraction_time_ns = end_time - start_time
            self.total_extraction_time_ns += extraction_time_ns

            # Alert if too slow
            extraction_time_us = extraction_time_ns / 1000
            if extraction_time_us > 500:  # >500μs is concerning
                logger.warning(f"Slow feature extraction: {extraction_time_us:.1f}μs")

    def _update_history_buffers(self, symbol_id: int, market_update: MarketUpdate):
        """履歴バッファ更新"""
        price = market_update.price / 10000.0
        size = market_update.size / 10000.0

        self.price_history[symbol_id].append(price)
        self.volume_history[symbol_id].append(size)
        self.timestamp_history[symbol_id].append(market_update.receive_timestamp_ns)

    def _extract_orderbook_features(
        self, features: MarketFeatures, order_book: OrderBook
    ) -> MarketFeatures:
        """Order Book特徴量抽出"""
        try:
            # Mid price from order book
            mid_price = order_book.get_mid_price()
            if mid_price:
                features.mid_price = mid_price / 10000.0

            # Spread calculation
            spread = order_book.get_spread()
            if spread and features.mid_price > 0:
                features.spread_bps = (spread / 10000.0) / features.mid_price * 10000

            # Order book imbalance (simplified)
            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()

            if best_bid and best_ask:
                bid_size = best_bid.get_size_float()
                ask_size = best_ask.get_size_float()
                total_size = bid_size + ask_size

                if total_size > 0:
                    features.bid_ask_imbalance = (bid_size - ask_size) / total_size
                    features.top_of_book_ratio = (
                        bid_size / ask_size if ask_size > 0 else 1.0
                    )

        except Exception as e:
            logger.debug(f"Order book特徴量抽出エラー: {e}")

        return features

    def _extract_technical_indicators(
        self, features: MarketFeatures, symbol_id: int
    ) -> MarketFeatures:
        """テクニカル指標抽出（高速版）"""
        prices = list(self.price_history[symbol_id])

        if len(prices) < 14:
            return features

        try:
            # RSI (simplified calculation)
            price_changes = np.diff(prices[-15:])  # Last 14 changes
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)

            if len(gains) > 0:
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)

                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    features.rsi_14 = 100 - (100 / (1 + rs))

            # EMA ratio (current price vs EMA)
            if len(prices) >= 10:
                ema_10 = np.mean(prices[-10:])  # Simplified EMA
                if ema_10 > 0:
                    features.ema_ratio = prices[-1] / ema_10

            # Bollinger position (simplified)
            if len(prices) >= 20:
                recent_prices = np.array(prices[-20:])
                mean_price = np.mean(recent_prices)
                std_price = np.std(recent_prices)

                if std_price > 0:
                    features.bollinger_position = (prices[-1] - mean_price) / (
                        2 * std_price
                    ) + 0.5
                    features.bollinger_position = np.clip(
                        features.bollinger_position, 0, 1
                    )

        except Exception as e:
            logger.debug(f"テクニカル指標計算エラー: {e}")

        return features

    def _extract_microstructure_features(
        self, features: MarketFeatures, symbol_id: int, market_update: MarketUpdate
    ) -> MarketFeatures:
        """マイクロストラクチャー特徴量抽出"""
        try:
            # Tick direction
            prices = list(self.price_history[symbol_id])
            if len(prices) >= 2:
                if prices[-1] > prices[-2]:
                    features.tick_direction = 1
                elif prices[-1] < prices[-2]:
                    features.tick_direction = -1
                else:
                    features.tick_direction = 0

            # Time since last trade
            timestamps = list(self.timestamp_history[symbol_id])
            if len(timestamps) >= 2:
                time_diff_ns = timestamps[-1] - timestamps[-2]
                features.time_since_last_trade_ms = time_diff_ns / 1_000_000

            # Realized volatility (simplified)
            if len(prices) >= 10:
                recent_returns = np.diff(np.log(prices[-10:]))
                features.realized_volatility = np.std(recent_returns) * np.sqrt(
                    252 * 24 * 60
                )  # Annualized

        except Exception as e:
            logger.debug(f"マイクロストラクチャー特徴量エラー: {e}")

        return features

    def _get_market_hours_normalized(self) -> float:
        """市場時間正規化 (0.0-1.0)"""
        # Simplified: based on time of day
        import datetime

        now = datetime.datetime.now()

        # Assuming 9:30 AM - 4:00 PM market hours (6.5 hours)
        market_start_minutes = 9 * 60 + 30  # 9:30 AM
        market_end_minutes = 16 * 60  # 4:00 PM
        current_minutes = now.hour * 60 + now.minute

        if market_start_minutes <= current_minutes <= market_end_minutes:
            market_progress = (current_minutes - market_start_minutes) / (
                market_end_minutes - market_start_minutes
            )
            return np.clip(market_progress, 0, 1)

        return 0.5  # Outside market hours

    def get_stats(self) -> Dict[str, Any]:
        """特徴量抽出統計"""
        avg_time_us = 0
        if self.extraction_count > 0:
            avg_time_us = (self.total_extraction_time_ns / self.extraction_count) / 1000

        return {
            "extractions_performed": self.extraction_count,
            "avg_extraction_time_us": avg_time_us,
            "symbols_tracked": len(self.price_history),
        }


class UltraFastAIPredictor:
    """超高速AI予測器（<300μs推論）"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.onnx_session = None
        self.model_available = False

        # Initialize ONNX model if available
        if ONNX_AVAILABLE and model_path:
            self._initialize_onnx_model()

        # Fallback rule-based predictor
        self.rule_based_predictor = RuleBasedPredictor()

        # Performance counters
        self.predictions_made = 0
        self.total_prediction_time_ns = 0

    def _initialize_onnx_model(self):
        """ONNX モデル初期化"""
        try:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.onnx_session = ort.InferenceSession(
                self.model_path, providers=providers
            )

            # Verify input shape
            input_shape = self.onnx_session.get_inputs()[0].shape
            logger.info(
                f"ONNX model loaded: {self.model_path}, input_shape: {input_shape}"
            )
            self.model_available = True

        except Exception as e:
            logger.warning(f"ONNX model initialization failed: {e}")
            self.model_available = False

    def predict(self, features: MarketFeatures) -> Tuple[np.ndarray, float]:
        """
        AI予測実行 (<300μs target)

        Args:
            features: 市場特徴量

        Returns:
            (prediction_array, confidence_score)
        """
        start_time = time.perf_counter_ns()

        try:
            if self.model_available and self.onnx_session:
                return self._predict_with_onnx(features)
            else:
                return self._predict_with_rules(features)

        finally:
            end_time = time.perf_counter_ns()
            prediction_time_ns = end_time - start_time
            self.total_prediction_time_ns += prediction_time_ns
            self.predictions_made += 1

            # Alert if too slow
            prediction_time_us = prediction_time_ns / 1000
            if prediction_time_us > 500:  # >500μs is concerning
                logger.warning(f"Slow AI prediction: {prediction_time_us:.1f}μs")

    def _predict_with_onnx(self, features: MarketFeatures) -> Tuple[np.ndarray, float]:
        """ONNX推論"""
        try:
            # Convert features to numpy array
            input_array = features.to_numpy_array().reshape(1, -1)

            # Run inference
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: input_array})

            prediction = outputs[0][0]  # First output, first batch
            confidence = float(np.max(np.abs(prediction)))  # Simple confidence metric

            return prediction, confidence

        except Exception as e:
            logger.debug(f"ONNX推論エラー: {e}")
            return self._predict_with_rules(features)

    def _predict_with_rules(self, features: MarketFeatures) -> Tuple[np.ndarray, float]:
        """ルールベース推論（フォールバック）"""
        return self.rule_based_predictor.predict(features)

    def get_stats(self) -> Dict[str, Any]:
        """予測統計"""
        avg_time_us = 0
        if self.predictions_made > 0:
            avg_time_us = (self.total_prediction_time_ns / self.predictions_made) / 1000

        return {
            "predictions_made": self.predictions_made,
            "avg_prediction_time_us": avg_time_us,
            "model_available": self.model_available,
            "model_path": self.model_path,
        }


class RuleBasedPredictor:
    """ルールベース予測器（AI推論フォールバック）"""

    def predict(self, features: MarketFeatures) -> Tuple[np.ndarray, float]:
        """
        ルールベース予測

        Returns:
            (prediction_array, confidence_score)
        """
        # Simplified momentum + mean reversion strategy
        score = 0.0
        confidence = 0.5

        # Momentum signals
        if features.price_change_bps > 10:  # Strong up move
            score += 0.3
        elif features.price_change_bps < -10:  # Strong down move
            score -= 0.3

        # RSI signals
        if features.rsi_14 < 30:  # Oversold
            score += 0.2
        elif features.rsi_14 > 70:  # Overbought
            score -= 0.2

        # Order book imbalance
        if abs(features.bid_ask_imbalance) > 0.2:
            score += 0.1 * np.sign(features.bid_ask_imbalance)
            confidence += 0.1

        # Spread consideration
        if features.spread_bps < 5:  # Tight spread = higher confidence
            confidence += 0.2
        elif features.spread_bps > 20:  # Wide spread = lower confidence
            confidence -= 0.2

        # Volatility adjustment
        if features.realized_volatility > 0.3:  # High volatility
            confidence *= 0.8

        confidence = np.clip(confidence, 0.1, 1.0)

        # Convert to action probability distribution
        # [STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY]
        if score > 0.5:
            prediction = np.array([0.0, 0.1, 0.2, 0.4, 0.3])  # Buy bias
        elif score < -0.5:
            prediction = np.array([0.3, 0.4, 0.2, 0.1, 0.0])  # Sell bias
        else:
            prediction = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # Hold bias

        return prediction, confidence


class RealtimeDecisionEngine:
    """
    リアルタイム取引決定エンジン

    目標: <1ms 意思決定レイテンシー
    機能: 特徴量抽出 → AI推論 → リスク評価 → 注文生成
    """

    def __init__(
        self,
        distributed_manager: Optional[DistributedComputingManager] = None,
        ai_model_path: Optional[str] = None,
        feature_lookback: int = 100,
        enable_ai_prediction: bool = True,
    ):
        """
        初期化

        Args:
            distributed_manager: 分散処理マネージャー
            ai_model_path: AIモデルファイルパス
            feature_lookback: 特徴量計算lookback期間
            enable_ai_prediction: AI予測有効化
        """
        self.distributed_manager = distributed_manager or DistributedComputingManager()
        self.enable_ai_prediction = enable_ai_prediction

        # Core components
        self.feature_extractor = FastFeatureExtractor(feature_lookback)
        self.ai_predictor = (
            UltraFastAIPredictor(ai_model_path) if enable_ai_prediction else None
        )
        self.rule_predictor = RuleBasedPredictor()

        # Decision parameters
        self.min_confidence_threshold = 0.6
        self.min_signal_strength = SignalStrength.MODERATE
        self.max_position_size = 1000
        self.default_stop_loss_pct = 0.02  # 2%

        # Performance statistics
        self.stats = {
            "decisions_made": 0,
            "actionable_signals": 0,
            "total_decision_time_ns": 0,
            "ai_predictions": 0,
            "rule_predictions": 0,
        }

        logger.info("RealtimeDecisionEngine初期化完了")

    async def make_decision(
        self,
        market_update: MarketUpdate,
        order_book: Optional[OrderBook] = None,
        current_position: int = 0,
    ) -> DecisionResult:
        """
        リアルタイム取引決定 (<1ms target)

        Args:
            market_update: 市場データ更新
            order_book: Order Book
            current_position: 現在ポジション

        Returns:
            DecisionResult: 決定結果
        """
        start_time = time.perf_counter_ns()

        try:
            # Stage 1: Feature extraction (200μs)
            features = self.feature_extractor.extract_features(
                market_update, order_book
            )

            # Stage 2: AI prediction (300μs)
            if self.ai_predictor and self.enable_ai_prediction:
                ai_prediction, ai_confidence = self.ai_predictor.predict(features)
                self.stats["ai_predictions"] += 1
            else:
                ai_prediction, ai_confidence = self.rule_predictor.predict(features)
                self.stats["rule_predictions"] += 1

            # Stage 3: Decision synthesis (200μs)
            trading_action = self._synthesize_decision(
                ai_prediction, ai_confidence, features
            )

            # Stage 4: Order generation (200μs)
            suggested_orders = []
            if trading_action != TradingAction.HOLD:
                suggested_orders = self._generate_orders(
                    market_update.symbol_id, trading_action, features, current_position
                )

            # Stage 5: Risk assessment (100μs)
            risk_assessment = self._assess_risk(
                features, trading_action, current_position
            )

            # Construct result
            decision_result = DecisionResult(
                symbol_id=market_update.symbol_id,
                decision=trading_action,
                confidence=ai_confidence,
                suggested_orders=suggested_orders,
                risk_assessment=risk_assessment,
                market_features=features,
                ai_prediction=ai_prediction,
            )

            # Update statistics
            self.stats["decisions_made"] += 1
            if trading_action != TradingAction.HOLD:
                self.stats["actionable_signals"] += 1

            return decision_result

        except Exception as e:
            logger.error(f"決定エンジンエラー {market_update.symbol_id}: {e}")

            return DecisionResult(
                symbol_id=market_update.symbol_id,
                decision=TradingAction.HOLD,
                confidence=0.0,
                risk_assessment={"error": 1.0},
            )

        finally:
            end_time = time.perf_counter_ns()
            decision_time_ns = end_time - start_time
            self.stats["total_decision_time_ns"] += decision_time_ns

            decision_time_us = decision_time_ns / 1000

            # Alert if too slow
            if decision_time_us > 2000:  # >2ms is concerning for HFT
                logger.warning(f"Slow decision making: {decision_time_us:.1f}μs")

            # Update result with timing
            if "decision_result" in locals():
                decision_result.processing_time_us = decision_time_us

    def _synthesize_decision(
        self, prediction: np.ndarray, confidence: float, features: MarketFeatures
    ) -> TradingAction:
        """決定合成"""
        try:
            # Prediction array: [STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY]
            if len(prediction) != 5:
                return TradingAction.HOLD

            # Find strongest signal
            max_prob_index = np.argmax(prediction)
            max_prob = prediction[max_prob_index]

            # Map index to action
            action_map = {
                0: TradingAction.STRONG_SELL,
                1: TradingAction.SELL,
                2: TradingAction.HOLD,
                3: TradingAction.BUY,
                4: TradingAction.STRONG_BUY,
            }

            suggested_action = action_map[max_prob_index]

            # Confidence and probability thresholds
            if confidence < self.min_confidence_threshold or max_prob < 0.4:
                return TradingAction.HOLD

            # Additional filters based on market conditions
            if features.spread_bps > 50:  # Very wide spread
                return TradingAction.HOLD

            if features.realized_volatility > 1.0:  # Extreme volatility
                return TradingAction.HOLD

            return suggested_action

        except Exception as e:
            logger.debug(f"決定合成エラー: {e}")
            return TradingAction.HOLD

    def _generate_orders(
        self,
        symbol_id: int,
        action: TradingAction,
        features: MarketFeatures,
        current_position: int,
    ) -> List[OrderEntry]:
        """注文生成"""
        try:
            orders = []

            # Determine order side and quantity
            if action in [TradingAction.BUY, TradingAction.STRONG_BUY]:
                side = OrderSide.BUY
                base_quantity = 100 if action == TradingAction.BUY else 200
            elif action in [TradingAction.SELL, TradingAction.STRONG_SELL]:
                side = OrderSide.SELL
                base_quantity = 100 if action == TradingAction.SELL else 200
            else:
                return orders

            # Adjust quantity based on position and risk
            target_position = current_position + (base_quantity * side)
            if abs(target_position) > self.max_position_size:
                # Position size risk management
                max_additional = self.max_position_size - abs(current_position)
                if max_additional <= 0:
                    return orders
                base_quantity = min(base_quantity, max_additional)

            # Create primary order
            primary_order = OrderEntry(
                order_id=int(time.perf_counter_ns() % 1000000),  # Simple ID
                symbol_id=symbol_id,
                side=side,
                order_type=OrderType.MARKET,  # For speed
                quantity=base_quantity,
                price=int(features.mid_price * 10000),  # Convert to fixed-point
                target_latency_us=50,
                priority=5 if "STRONG" in action.name else 3,
            )

            orders.append(primary_order)

            return orders

        except Exception as e:
            logger.debug(f"注文生成エラー: {e}")
            return []

    def _assess_risk(
        self, features: MarketFeatures, action: TradingAction, current_position: int
    ) -> Dict[str, float]:
        """リスク評価"""
        risk_assessment = {
            "market_risk": 0.0,
            "position_risk": 0.0,
            "liquidity_risk": 0.0,
            "timing_risk": 0.0,
            "overall_risk": 0.0,
        }

        try:
            # Market risk (volatility based)
            risk_assessment["market_risk"] = min(
                features.realized_volatility / 0.5, 1.0
            )

            # Position risk (concentration)
            position_risk = abs(current_position) / self.max_position_size
            risk_assessment["position_risk"] = position_risk

            # Liquidity risk (spread based)
            liquidity_risk = min(features.spread_bps / 20, 1.0)  # 20bps = high risk
            risk_assessment["liquidity_risk"] = liquidity_risk

            # Timing risk (based on time since last trade)
            timing_risk = 0.0
            if features.time_since_last_trade_ms > 5000:  # >5s since last trade
                timing_risk = 0.3
            risk_assessment["timing_risk"] = timing_risk

            # Overall risk (weighted combination)
            weights = [0.4, 0.3, 0.2, 0.1]  # market, position, liquidity, timing
            risks = [
                risk_assessment["market_risk"],
                risk_assessment["position_risk"],
                risk_assessment["liquidity_risk"],
                risk_assessment["timing_risk"],
            ]

            risk_assessment["overall_risk"] = sum(w * r for w, r in zip(weights, risks))

        except Exception as e:
            logger.debug(f"リスク評価エラー: {e}")
            risk_assessment["overall_risk"] = 1.0  # Conservative default

        return risk_assessment

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """包括的統計情報"""
        avg_decision_time_us = 0
        if self.stats["decisions_made"] > 0:
            avg_decision_time_us = (
                self.stats["total_decision_time_ns"] / self.stats["decisions_made"]
            ) / 1000

        stats = self.stats.copy()
        stats.update(
            {
                "avg_decision_time_us": avg_decision_time_us,
                "actionable_signal_rate": (
                    self.stats["actionable_signals"]
                    / max(self.stats["decisions_made"], 1)
                ),
                "feature_extractor_stats": self.feature_extractor.get_stats(),
            }
        )

        if self.ai_predictor:
            stats["ai_predictor_stats"] = self.ai_predictor.get_stats()

        return stats

    async def cleanup(self):
        """クリーンアップ"""
        logger.info("RealtimeDecisionEngine クリーンアップ開始")
        # No persistent resources to clean up
        logger.info("クリーンアップ完了")


# Factory function
def create_decision_engine(
    distributed_manager: Optional[DistributedComputingManager] = None, **config
) -> RealtimeDecisionEngine:
    """RealtimeDecisionEngineファクトリ関数"""
    return RealtimeDecisionEngine(distributed_manager=distributed_manager, **config)


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #366 リアルタイム決定エンジンテスト ===")

        engine = None
        try:
            # 決定エンジン初期化
            engine = create_decision_engine(
                enable_ai_prediction=True, feature_lookback=50
            )

            # テスト市場データ
            test_updates = []
            for i in range(10):
                update = MarketUpdate(
                    symbol_id=1001,
                    price=int((100.0 + i * 0.1) * 10000),  # Gradually increasing price
                    size=100 * 10000,
                    receive_timestamp_ns=time.perf_counter_ns(),
                )
                test_updates.append(update)

            print("\n1. 単一決定テスト")
            decision = await engine.make_decision(
                test_updates[0], order_book=None, current_position=0
            )

            print(
                f"決定結果: action={decision.decision.name}, confidence={decision.confidence:.2f}"
            )
            print(f"処理時間: {decision.processing_time_us:.1f}μs")
            print(f"提案注文数: {len(decision.suggested_orders)}")

            if decision.risk_assessment:
                print("リスク評価:")
                for risk_type, risk_value in decision.risk_assessment.items():
                    print(f"  {risk_type}: {risk_value:.3f}")

            print("\n2. バッチ決定テスト")
            decisions = []
            start_time = time.perf_counter()

            for update in test_updates:
                decision = await engine.make_decision(update)
                decisions.append(decision)

            batch_time_ms = (time.perf_counter() - start_time) * 1000
            print(f"バッチ処理: {len(decisions)} decisions in {batch_time_ms:.3f}ms")
            print(
                f"平均決定時間: {batch_time_ms * 1000 / len(decisions):.1f}μs per decision"
            )

            # Action distribution
            actions = [d.decision.name for d in decisions]
            action_counts = {action: actions.count(action) for action in set(actions)}
            print(f"Action分布: {action_counts}")

            print("\n3. パフォーマンス統計")
            stats = engine.get_comprehensive_stats()
            for key, value in stats.items():
                if key == "feature_extractor_stats":
                    print("  特徴量抽出統計:")
                    for fk, fv in value.items():
                        print(f"    {fk}: {fv}")
                elif key == "ai_predictor_stats":
                    print("  AI予測統計:")
                    for ak, av in value.items():
                        print(f"    {ak}: {av}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

            print("\n4. レイテンシー目標確認")
            if stats["avg_decision_time_us"] < 1000:
                print(
                    f"✅ レイテンシー目標達成: {stats['avg_decision_time_us']:.1f}μs < 1000μs"
                )
            else:
                print(
                    f"⚠️  レイテンシー目標未達: {stats['avg_decision_time_us']:.1f}μs > 1000μs"
                )

        except Exception as e:
            print(f"テスト実行エラー: {e}")

        finally:
            if engine:
                await engine.cleanup()

        print("\n=== リアルタイム決定エンジンテスト完了 ===")

    asyncio.run(main())
