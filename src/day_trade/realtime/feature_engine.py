#!/usr/bin/env python3
"""
リアルタイム特徴量生成エンジン
Real-Time Feature Generation Engine

Issue #763: リアルタイム特徴量生成と予測パイプラインの構築
"""

import asyncio
import logging
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# ログ設定
logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """市場データポイント"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'price': self.price,
            'volume': self.volume,
            'bid': self.bid,
            'ask': self.ask
        }


@dataclass
class FeatureValue:
    """特徴量値"""
    name: str
    value: float
    timestamp: datetime
    symbol: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'metadata': self.metadata
        }


class IncrementalIndicator(ABC):
    """インクリメンタル計算可能な指標の基底クラス"""

    def __init__(self, symbol: str, period: int):
        self.symbol = symbol
        self.period = period
        self.is_ready = False

    @abstractmethod
    def update(self, data_point: MarketDataPoint) -> Optional[FeatureValue]:
        """新しいデータポイントで指標を更新"""
        pass

    @abstractmethod
    def get_current_value(self) -> Optional[float]:
        """現在の指標値を取得"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """指標をリセット"""
        pass


class IncrementalSMA(IncrementalIndicator):
    """インクリメンタル単純移動平均"""

    def __init__(self, symbol: str, period: int):
        super().__init__(symbol, period)
        self.prices = deque(maxlen=period)
        self.sum_value = 0.0

    def update(self, data_point: MarketDataPoint) -> Optional[FeatureValue]:
        """新しい価格でSMAを更新"""
        if len(self.prices) == self.period:
            # 古い値を削除
            old_price = self.prices[0]
            self.sum_value -= old_price

        # 新しい値を追加
        self.prices.append(data_point.price)
        self.sum_value += data_point.price

        if len(self.prices) >= self.period:
            self.is_ready = True
            sma_value = self.sum_value / len(self.prices)

            return FeatureValue(
                name=f"sma_{self.period}",
                value=sma_value,
                timestamp=data_point.timestamp,
                symbol=self.symbol,
                metadata={'period': self.period, 'data_points': len(self.prices)}
            )

        return None

    def get_current_value(self) -> Optional[float]:
        if self.is_ready:
            return self.sum_value / len(self.prices)
        return None

    def reset(self) -> None:
        self.prices.clear()
        self.sum_value = 0.0
        self.is_ready = False


class IncrementalEMA(IncrementalIndicator):
    """インクリメンタル指数移動平均"""

    def __init__(self, symbol: str, period: int):
        super().__init__(symbol, period)
        self.alpha = 2.0 / (period + 1)
        self.ema_value = None
        self.initialized = False

    def update(self, data_point: MarketDataPoint) -> Optional[FeatureValue]:
        """新しい価格でEMAを更新"""
        if not self.initialized:
            self.ema_value = data_point.price
            self.initialized = True
            self.is_ready = True
        else:
            self.ema_value = self.alpha * data_point.price + (1 - self.alpha) * self.ema_value

        return FeatureValue(
            name=f"ema_{self.period}",
            value=self.ema_value,
            timestamp=data_point.timestamp,
            symbol=self.symbol,
            metadata={'period': self.period, 'alpha': self.alpha}
        )

    def get_current_value(self) -> Optional[float]:
        return self.ema_value if self.is_ready else None

    def reset(self) -> None:
        self.ema_value = None
        self.initialized = False
        self.is_ready = False


class IncrementalRSI(IncrementalIndicator):
    """インクリメンタルRSI"""

    def __init__(self, symbol: str, period: int = 14):
        super().__init__(symbol, period)
        self.gains = deque(maxlen=period)
        self.losses = deque(maxlen=period)
        self.prev_price = None
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.alpha = 1.0 / period

    def update(self, data_point: MarketDataPoint) -> Optional[FeatureValue]:
        """新しい価格でRSIを更新"""
        if self.prev_price is None:
            self.prev_price = data_point.price
            return None

        change = data_point.price - self.prev_price
        gain = max(change, 0)
        loss = max(-change, 0)

        if len(self.gains) < self.period:
            # 初期期間中は単純平均
            self.gains.append(gain)
            self.losses.append(loss)

            if len(self.gains) == self.period:
                self.avg_gain = sum(self.gains) / self.period
                self.avg_loss = sum(self.losses) / self.period
                self.is_ready = True
        else:
            # 指数移動平均で更新
            self.avg_gain = (1 - self.alpha) * self.avg_gain + self.alpha * gain
            self.avg_loss = (1 - self.alpha) * self.avg_loss + self.alpha * loss

        self.prev_price = data_point.price

        if self.is_ready and self.avg_loss != 0:
            rs = self.avg_gain / self.avg_loss
            rsi_value = 100 - (100 / (1 + rs))

            return FeatureValue(
                name=f"rsi_{self.period}",
                value=rsi_value,
                timestamp=data_point.timestamp,
                symbol=self.symbol,
                metadata={
                    'period': self.period,
                    'avg_gain': self.avg_gain,
                    'avg_loss': self.avg_loss
                }
            )

        return None

    def get_current_value(self) -> Optional[float]:
        if self.is_ready and self.avg_loss != 0:
            rs = self.avg_gain / self.avg_loss
            return 100 - (100 / (1 + rs))
        return None

    def reset(self) -> None:
        self.gains.clear()
        self.losses.clear()
        self.prev_price = None
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.is_ready = False


class IncrementalMACD(IncrementalIndicator):
    """インクリメンタルMACD"""

    def __init__(self, symbol: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(symbol, max(fast_period, slow_period, signal_period))
        self.fast_ema = IncrementalEMA(symbol, fast_period)
        self.slow_ema = IncrementalEMA(symbol, slow_period)
        self.signal_ema = IncrementalEMA(symbol, signal_period)
        self.macd_values = []

    def update(self, data_point: MarketDataPoint) -> Optional[FeatureValue]:
        """新しい価格でMACDを更新"""
        fast_result = self.fast_ema.update(data_point)
        slow_result = self.slow_ema.update(data_point)

        if fast_result and slow_result:
            macd_line = fast_result.value - slow_result.value

            # シグナルライン計算用の疑似データポイント
            macd_data_point = MarketDataPoint(
                symbol=self.symbol,
                timestamp=data_point.timestamp,
                price=macd_line,
                volume=0
            )

            signal_result = self.signal_ema.update(macd_data_point)
            signal_line = signal_result.value if signal_result else 0.0

            histogram = macd_line - signal_line

            self.is_ready = True

            return FeatureValue(
                name="macd",
                value=macd_line,
                timestamp=data_point.timestamp,
                symbol=self.symbol,
                metadata={
                    'macd_line': macd_line,
                    'signal_line': signal_line,
                    'histogram': histogram
                }
            )

        return None

    def get_current_value(self) -> Optional[float]:
        fast_val = self.fast_ema.get_current_value()
        slow_val = self.slow_ema.get_current_value()

        if fast_val is not None and slow_val is not None:
            return fast_val - slow_val
        return None

    def reset(self) -> None:
        self.fast_ema.reset()
        self.slow_ema.reset()
        self.signal_ema.reset()
        self.macd_values.clear()
        self.is_ready = False


class IncrementalBollingerBands(IncrementalIndicator):
    """インクリメンタルボリンジャーバンド"""

    def __init__(self, symbol: str, period: int = 20, std_dev: float = 2.0):
        super().__init__(symbol, period)
        self.sma = IncrementalSMA(symbol, period)
        self.prices = deque(maxlen=period)
        self.std_dev = std_dev

    def update(self, data_point: MarketDataPoint) -> Optional[FeatureValue]:
        """新しい価格でボリンジャーバンドを更新"""
        self.prices.append(data_point.price)
        sma_result = self.sma.update(data_point)

        if sma_result and len(self.prices) >= self.period:
            prices_array = np.array(list(self.prices))
            std = np.std(prices_array)

            upper_band = sma_result.value + (self.std_dev * std)
            lower_band = sma_result.value - (self.std_dev * std)

            # バンド位置の計算 (0-1の値)
            band_position = (data_point.price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5

            self.is_ready = True

            return FeatureValue(
                name=f"bollinger_{self.period}",
                value=band_position,
                timestamp=data_point.timestamp,
                symbol=self.symbol,
                metadata={
                    'middle_band': sma_result.value,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'band_width': upper_band - lower_band,
                    'std': std
                }
            )

        return None

    def get_current_value(self) -> Optional[float]:
        if self.is_ready and len(self.prices) >= self.period:
            sma_val = self.sma.get_current_value()
            if sma_val:
                prices_array = np.array(list(self.prices))
                std = np.std(prices_array)
                upper_band = sma_val + (self.std_dev * std)
                lower_band = sma_val - (self.std_dev * std)
                current_price = self.prices[-1]
                return (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        return None

    def reset(self) -> None:
        self.sma.reset()
        self.prices.clear()
        self.is_ready = False


class RealTimeFeatureEngine:
    """リアルタイム特徴量生成エンジン"""

    def __init__(self, buffer_size: int = 1000):
        self.indicators: Dict[str, Dict[str, IncrementalIndicator]] = defaultdict(dict)
        self.feature_cache: Dict[str, Dict[str, FeatureValue]] = defaultdict(dict)
        self.buffer_size = buffer_size
        self.data_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))

        # パフォーマンス監視
        self.processing_times: deque = deque(maxlen=1000)
        self.feature_count = 0

        # デフォルト指標の設定
        self._setup_default_indicators()

        logger.info("RealTimeFeatureEngine initialized")

    def _setup_default_indicators(self) -> None:
        """デフォルト指標の設定"""
        self.default_indicators = [
            ('sma_5', lambda symbol: IncrementalSMA(symbol, 5)),
            ('sma_20', lambda symbol: IncrementalSMA(symbol, 20)),
            ('sma_50', lambda symbol: IncrementalSMA(symbol, 50)),
            ('ema_12', lambda symbol: IncrementalEMA(symbol, 12)),
            ('ema_26', lambda symbol: IncrementalEMA(symbol, 26)),
            ('rsi_14', lambda symbol: IncrementalRSI(symbol, 14)),
            ('macd', lambda symbol: IncrementalMACD(symbol)),
            ('bollinger_20', lambda symbol: IncrementalBollingerBands(symbol, 20)),
        ]

    def add_symbol(self, symbol: str) -> None:
        """新しい銘柄を追加"""
        if symbol not in self.indicators:
            for indicator_name, indicator_factory in self.default_indicators:
                self.indicators[symbol][indicator_name] = indicator_factory(symbol)
            logger.info(f"Added indicators for symbol: {symbol}")

    def add_custom_indicator(self, symbol: str, name: str, indicator: IncrementalIndicator) -> None:
        """カスタム指標を追加"""
        if symbol not in self.indicators:
            self.add_symbol(symbol)

        self.indicators[symbol][name] = indicator
        logger.info(f"Added custom indicator {name} for symbol: {symbol}")

    async def process_data_point(self, data_point: MarketDataPoint) -> List[FeatureValue]:
        """データポイントを処理して特徴量を生成"""
        start_time = time.time()

        try:
            # バッファに追加
            self.data_buffer[data_point.symbol].append(data_point)

            # 銘柄の指標が存在しない場合は追加
            if data_point.symbol not in self.indicators:
                self.add_symbol(data_point.symbol)

            generated_features = []

            # 各指標を更新
            for indicator_name, indicator in self.indicators[data_point.symbol].items():
                try:
                    feature_value = indicator.update(data_point)
                    if feature_value:
                        # キャッシュに保存
                        self.feature_cache[data_point.symbol][indicator_name] = feature_value
                        generated_features.append(feature_value)
                        self.feature_count += 1

                except Exception as e:
                    logger.error(f"Error updating indicator {indicator_name} for {data_point.symbol}: {e}")

            # パフォーマンス監視
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            if len(generated_features) > 0:
                logger.debug(f"Generated {len(generated_features)} features for {data_point.symbol} in {processing_time*1000:.2f}ms")

            return generated_features

        except Exception as e:
            logger.error(f"Error processing data point for {data_point.symbol}: {e}")
            return []

    def get_latest_features(self, symbol: str, feature_names: Optional[List[str]] = None) -> Dict[str, FeatureValue]:
        """最新の特徴量を取得"""
        if symbol not in self.feature_cache:
            return {}

        if feature_names is None:
            return self.feature_cache[symbol].copy()

        return {
            name: feature for name, feature in self.feature_cache[symbol].items()
            if name in feature_names
        }

    def get_feature_vector(self, symbol: str, feature_names: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """特徴量ベクトルを取得"""
        features = self.get_latest_features(symbol, feature_names)

        if not features:
            return None

        if feature_names is None:
            feature_names = sorted(features.keys())

        vector = []
        for name in feature_names:
            if name in features:
                vector.append(features[name].value)
            else:
                vector.append(np.nan)

        return np.array(vector) if vector else None

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        if not self.processing_times:
            return {
                'avg_processing_time_ms': 0,
                'max_processing_time_ms': 0,
                'min_processing_time_ms': 0,
                'total_features_generated': self.feature_count,
                'active_symbols': len(self.indicators)
            }

        times_ms = [t * 1000 for t in self.processing_times]

        return {
            'avg_processing_time_ms': np.mean(times_ms),
            'max_processing_time_ms': np.max(times_ms),
            'min_processing_time_ms': np.min(times_ms),
            'p95_processing_time_ms': np.percentile(times_ms, 95),
            'total_features_generated': self.feature_count,
            'active_symbols': len(self.indicators),
            'throughput_features_per_second': len(self.processing_times) / sum(self.processing_times) if sum(self.processing_times) > 0 else 0
        }

    def reset_symbol(self, symbol: str) -> None:
        """指定銘柄の指標をリセット"""
        if symbol in self.indicators:
            for indicator in self.indicators[symbol].values():
                indicator.reset()
            self.feature_cache[symbol].clear()
            self.data_buffer[symbol].clear()
            logger.info(f"Reset indicators for symbol: {symbol}")

    def reset_all(self) -> None:
        """全ての指標をリセット"""
        for symbol in list(self.indicators.keys()):
            self.reset_symbol(symbol)

        self.processing_times.clear()
        self.feature_count = 0
        logger.info("Reset all indicators")


# 使用例とテスト
async def test_feature_engine():
    """特徴量エンジンのテスト"""
    engine = RealTimeFeatureEngine()

    # テストデータ生成
    symbol = "7203"  # トヨタ
    base_price = 2000.0

    print(f"Testing RealTimeFeatureEngine with symbol: {symbol}")

    # データポイントを逐次処理
    for i in range(100):
        # 価格変動をシミュレート
        price_change = np.random.normal(0, 10)
        current_price = base_price + price_change

        data_point = MarketDataPoint(
            symbol=symbol,
            timestamp=datetime.now() + timedelta(seconds=i),
            price=current_price,
            volume=1000 + int(np.random.normal(0, 100))
        )

        # 特徴量生成
        features = await engine.process_data_point(data_point)

        if features and i % 10 == 0:  # 10件ごとに出力
            print(f"\nStep {i+1}: Generated {len(features)} features")
            for feature in features:
                print(f"  {feature.name}: {feature.value:.4f}")

        base_price = current_price

    # パフォーマンス統計表示
    stats = engine.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"  Average processing time: {stats['avg_processing_time_ms']:.2f}ms")
    print(f"  Total features generated: {stats['total_features_generated']}")
    print(f"  Throughput: {stats['throughput_features_per_second']:.2f} features/sec")

    # 最新特徴量取得
    latest_features = engine.get_latest_features(symbol)
    print(f"\nLatest features for {symbol}:")
    for name, feature in latest_features.items():
        print(f"  {name}: {feature.value:.4f}")


if __name__ == "__main__":
    asyncio.run(test_feature_engine())