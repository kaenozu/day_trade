#!/usr/bin/env python3
"""
ストリーミングデータプロセッサ
Streaming Data Processor for Real-Time Market Data

Issue #763: リアルタイム特徴量生成と予測パイプライン Phase 2
"""

import asyncio
import json
import logging
import time
import websockets
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Union
import aiohttp
import numpy as np
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from .feature_engine import MarketDataPoint, RealTimeFeatureEngine

# ログ設定
logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """ストリーム設定"""
    url: str
    symbols: List[str]
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 10
    heartbeat_interval: float = 30.0
    buffer_size: int = 1000
    rate_limit: int = 1000  # messages per second


@dataclass
class StreamMetrics:
    """ストリーム監視メトリクス"""
    messages_received: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0
    connection_count: int = 0
    last_message_time: Optional[datetime] = None
    avg_latency_ms: float = 0.0
    error_count: int = 0


class DataFilter(ABC):
    """データフィルターの基底クラス"""

    @abstractmethod
    async def filter(self, data: Dict[str, Any]) -> bool:
        """データをフィルタリング"""
        pass


class SymbolFilter(DataFilter):
    """銘柄フィルター"""

    def __init__(self, allowed_symbols: List[str]):
        self.allowed_symbols = set(allowed_symbols)

    async def filter(self, data: Dict[str, Any]) -> bool:
        symbol = data.get('symbol', '')
        return symbol in self.allowed_symbols


class PriceRangeFilter(DataFilter):
    """価格範囲フィルター"""

    def __init__(self, min_price: float = 0.01, max_price: float = 1000000.0):
        self.min_price = min_price
        self.max_price = max_price

    async def filter(self, data: Dict[str, Any]) -> bool:
        price = data.get('price', 0.0)
        return self.min_price <= price <= self.max_price


class VolumeFilter(DataFilter):
    """出来高フィルター"""

    def __init__(self, min_volume: int = 1):
        self.min_volume = min_volume

    async def filter(self, data: Dict[str, Any]) -> bool:
        volume = data.get('volume', 0)
        return volume >= self.min_volume


class TimeRangeFilter(DataFilter):
    """時間範囲フィルター（取引時間のみ）"""

    def __init__(self, start_hour: int = 9, end_hour: int = 15):
        self.start_hour = start_hour
        self.end_hour = end_hour

    async def filter(self, data: Dict[str, Any]) -> bool:
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()

        hour = timestamp.hour
        return self.start_hour <= hour <= self.end_hour


class DataTransformer:
    """データ変換器"""

    @staticmethod
    def normalize_yahoo_finance(data: Dict[str, Any]) -> Optional[MarketDataPoint]:
        """Yahoo Finance形式のデータを正規化"""
        try:
            return MarketDataPoint(
                symbol=data.get('symbol', ''),
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                price=float(data.get('price', 0.0)),
                volume=int(data.get('volume', 0)),
                bid=data.get('bid'),
                ask=data.get('ask')
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error transforming Yahoo Finance data: {e}")
            return None

    @staticmethod
    def normalize_polygon_io(data: Dict[str, Any]) -> Optional[MarketDataPoint]:
        """Polygon.io形式のデータを正規化"""
        try:
            # Polygon.ioのタイムスタンプはUnixタイムスタンプ（ミリ秒）
            timestamp = datetime.fromtimestamp(data.get('t', 0) / 1000)

            return MarketDataPoint(
                symbol=data.get('sym', ''),
                timestamp=timestamp,
                price=float(data.get('c', 0.0)),  # close price
                volume=int(data.get('v', 0)),
                bid=data.get('b'),
                ask=data.get('a')
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error transforming Polygon.io data: {e}")
            return None

    @staticmethod
    def normalize_alpaca(data: Dict[str, Any]) -> Optional[MarketDataPoint]:
        """Alpaca形式のデータを正規化"""
        try:
            return MarketDataPoint(
                symbol=data.get('S', ''),  # Symbol
                timestamp=datetime.fromisoformat(data.get('t', datetime.now().isoformat())),
                price=float(data.get('p', 0.0)),  # Price
                volume=int(data.get('s', 0)),  # Size
                bid=data.get('bp'),  # Bid price
                ask=data.get('ap')   # Ask price
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error transforming Alpaca data: {e}")
            return None


class StreamingDataProcessor:
    """ストリーミングデータプロセッサ"""

    def __init__(self,
                 feature_engine: RealTimeFeatureEngine,
                 stream_config: StreamConfig,
                 data_transformer: Callable[[Dict[str, Any]], Optional[MarketDataPoint]] = DataTransformer.normalize_yahoo_finance):

        self.feature_engine = feature_engine
        self.config = stream_config
        self.data_transformer = data_transformer

        # フィルター設定
        self.filters: List[DataFilter] = [
            SymbolFilter(stream_config.symbols),
            PriceRangeFilter(),
            VolumeFilter(),
            TimeRangeFilter()
        ]

        # 内部状態
        self.is_running = False
        self.websocket = None
        self.session = None
        self.reconnect_count = 0

        # バッファ
        self.message_buffer: deque = deque(maxlen=stream_config.buffer_size)
        self.processed_data: deque = deque(maxlen=stream_config.buffer_size)

        # メトリクス
        self.metrics = StreamMetrics()

        # レート制限
        self.rate_limiter = asyncio.Semaphore(stream_config.rate_limit)
        self.last_process_time = time.time()

        logger.info("StreamingDataProcessor initialized")

    def add_filter(self, data_filter: DataFilter) -> None:
        """データフィルターを追加"""
        self.filters.append(data_filter)
        logger.info(f"Added filter: {type(data_filter).__name__}")

    async def _apply_filters(self, data: Dict[str, Any]) -> bool:
        """全フィルターを適用"""
        for data_filter in self.filters:
            try:
                if not await data_filter.filter(data):
                    return False
            except Exception as e:
                logger.error(f"Error applying filter {type(data_filter).__name__}: {e}")
                return False
        return True

    async def _rate_limit_check(self) -> bool:
        """レート制限チェック"""
        try:
            await asyncio.wait_for(self.rate_limiter.acquire(), timeout=0.001)
            return True
        except asyncio.TimeoutError:
            return False

    async def start_websocket_stream(self) -> None:
        """WebSocketストリーミング開始"""
        self.is_running = True

        while self.is_running and self.reconnect_count < self.config.max_reconnect_attempts:
            try:
                logger.info(f"Connecting to WebSocket: {self.config.url}")

                async with websockets.connect(
                    self.config.url,
                    ping_interval=self.config.heartbeat_interval,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:

                    self.websocket = websocket
                    self.metrics.connection_count += 1
                    self.reconnect_count = 0

                    logger.info("WebSocket connected successfully")

                    # 購読メッセージ送信（必要に応じて）
                    subscribe_message = {
                        "action": "subscribe",
                        "symbols": self.config.symbols
                    }
                    await websocket.send(json.dumps(subscribe_message))

                    # メッセージ受信ループ
                    async for message in websocket:
                        if not self.is_running:
                            break

                        await self._handle_websocket_message(message)

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                self.metrics.error_count += 1

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.metrics.error_count += 1

            finally:
                self.websocket = None

                if self.is_running:
                    self.reconnect_count += 1
                    wait_time = min(self.config.reconnect_interval * self.reconnect_count, 60)
                    logger.info(f"Reconnecting in {wait_time} seconds (attempt {self.reconnect_count})")
                    await asyncio.sleep(wait_time)

    async def _handle_websocket_message(self, message: str) -> None:
        """WebSocketメッセージ処理"""
        try:
            # レート制限チェック
            if not await self._rate_limit_check():
                self.metrics.messages_dropped += 1
                return

            # メッセージをパース
            data = json.loads(message)
            self.metrics.messages_received += 1
            self.metrics.last_message_time = datetime.now()

            # バッファに追加
            self.message_buffer.append(data)

            # フィルタリング
            if not await self._apply_filters(data):
                return

            # データ変換
            market_data = self.data_transformer(data)
            if not market_data:
                return

            # 特徴量生成エンジンに送信
            await self._process_market_data(market_data)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
            self.metrics.error_count += 1

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            self.metrics.error_count += 1

        finally:
            # レート制限セマフォ解放
            try:
                self.rate_limiter.release()
            except ValueError:
                pass  # 既に解放済み

    async def _process_market_data(self, market_data: MarketDataPoint) -> None:
        """市場データ処理"""
        start_time = time.time()

        try:
            # 特徴量生成
            features = await self.feature_engine.process_data_point(market_data)

            if features:
                # 処理済みデータとしてバッファに保存
                processed_entry = {
                    'market_data': asdict(market_data),
                    'features': [asdict(feature) for feature in features],
                    'processing_time_ms': (time.time() - start_time) * 1000
                }
                self.processed_data.append(processed_entry)

                logger.debug(f"Processed {len(features)} features for {market_data.symbol}")

            self.metrics.messages_processed += 1

            # レイテンシ計算
            latency = (time.time() - start_time) * 1000
            if self.metrics.avg_latency_ms == 0:
                self.metrics.avg_latency_ms = latency
            else:
                self.metrics.avg_latency_ms = (self.metrics.avg_latency_ms * 0.9) + (latency * 0.1)

        except Exception as e:
            logger.error(f"Error processing market data for {market_data.symbol}: {e}")
            self.metrics.error_count += 1

    async def start_http_polling(self, interval: float = 1.0) -> None:
        """HTTPポーリング開始（WebSocketの代替）"""
        self.is_running = True

        async with aiohttp.ClientSession() as session:
            self.session = session

            while self.is_running:
                try:
                    for symbol in self.config.symbols:
                        await self._poll_symbol_data(symbol)

                    await asyncio.sleep(interval)

                except Exception as e:
                    logger.error(f"Error in HTTP polling: {e}")
                    self.metrics.error_count += 1
                    await asyncio.sleep(interval * 2)

    async def _poll_symbol_data(self, symbol: str) -> None:
        """個別銘柄データのポーリング"""
        try:
            # Yahoo Finance APIのサンプル（実際のURLは環境に応じて変更）
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # データ変換とフィルタリング
                    if await self._apply_filters(data):
                        market_data = self.data_transformer(data)
                        if market_data:
                            await self._process_market_data(market_data)

        except Exception as e:
            logger.error(f"Error polling data for {symbol}: {e}")
            self.metrics.error_count += 1

    async def start_simulation_stream(self, data_generator: AsyncGenerator[MarketDataPoint, None]) -> None:
        """シミュレーションストリーム開始（テスト用）"""
        self.is_running = True

        try:
            async for market_data in data_generator:
                if not self.is_running:
                    break

                # フィルタリング
                data_dict = asdict(market_data)
                if await self._apply_filters(data_dict):
                    await self._process_market_data(market_data)

                # 適度な遅延
                await asyncio.sleep(0.001)  # 1ms

        except Exception as e:
            logger.error(f"Error in simulation stream: {e}")
            self.metrics.error_count += 1

    def stop(self) -> None:
        """ストリーミング停止"""
        self.is_running = False
        logger.info("StreamingDataProcessor stopped")

    def get_metrics(self) -> StreamMetrics:
        """メトリクス取得"""
        return self.metrics

    def get_recent_data(self, count: int = 10) -> List[Dict[str, Any]]:
        """最近処理されたデータを取得"""
        return list(self.processed_data)[-count:]

    def clear_buffers(self) -> None:
        """バッファクリア"""
        self.message_buffer.clear()
        self.processed_data.clear()
        logger.info("Buffers cleared")


# テスト用データ生成器
async def generate_test_data(symbols: List[str], duration_seconds: int = 60) -> AsyncGenerator[MarketDataPoint, None]:
    """テスト用のマーケットデータ生成"""
    end_time = time.time() + duration_seconds
    base_prices = {symbol: np.random.uniform(1000, 5000) for symbol in symbols}

    while time.time() < end_time:
        for symbol in symbols:
            # 価格変動シミュレート
            price_change = np.random.normal(0, base_prices[symbol] * 0.001)
            current_price = base_prices[symbol] + price_change
            base_prices[symbol] = current_price

            market_data = MarketDataPoint(
                symbol=symbol,
                timestamp=datetime.now(),
                price=current_price,
                volume=np.random.randint(100, 10000),
                bid=current_price - np.random.uniform(0.1, 1.0),
                ask=current_price + np.random.uniform(0.1, 1.0)
            )

            yield market_data
            await asyncio.sleep(0.1)  # 100ms間隔


# 使用例とテスト
async def test_streaming_processor():
    """ストリーミングプロセッサのテスト"""

    # 特徴量エンジン初期化
    feature_engine = RealTimeFeatureEngine()

    # ストリーム設定
    config = StreamConfig(
        url="wss://example.com/market-data",  # ダミーURL
        symbols=["7203", "8306", "9984"],
        buffer_size=1000,
        rate_limit=100
    )

    # プロセッサ初期化
    processor = StreamingDataProcessor(
        feature_engine=feature_engine,
        stream_config=config
    )

    print("Starting simulation stream test...")

    # シミュレーションストリーム開始
    data_generator = generate_test_data(config.symbols, duration_seconds=30)

    # バックグラウンドでストリーム処理開始
    stream_task = asyncio.create_task(
        processor.start_simulation_stream(data_generator)
    )

    # 定期的にメトリクス表示
    for i in range(6):  # 30秒間、5秒ごと
        await asyncio.sleep(5)

        metrics = processor.get_metrics()
        feature_stats = feature_engine.get_performance_stats()

        print(f"\n--- Metrics at {i*5+5}s ---")
        print(f"Messages processed: {metrics.messages_processed}")
        print(f"Messages dropped: {metrics.messages_dropped}")
        print(f"Average latency: {metrics.avg_latency_ms:.2f}ms")
        print(f"Error count: {metrics.error_count}")
        print(f"Features generated: {feature_stats['total_features_generated']}")
        print(f"Feature throughput: {feature_stats['throughput_features_per_second']:.2f}/sec")

    # ストリーム停止
    processor.stop()
    await stream_task

    # 最終結果表示
    print(f"\n--- Final Results ---")
    recent_data = processor.get_recent_data(5)
    print(f"Recent processed data entries: {len(recent_data)}")

    if recent_data:
        latest_entry = recent_data[-1]
        print(f"Latest processing time: {latest_entry['processing_time_ms']:.2f}ms")
        print(f"Latest features count: {len(latest_entry['features'])}")


if __name__ == "__main__":
    asyncio.run(test_streaming_processor())