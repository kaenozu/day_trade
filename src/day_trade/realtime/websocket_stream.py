#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine - WebSocketリアルタイムストリーミング
高性能WebSocketデータストリーム・統合管理システム

市場データ、ニュース、ソーシャルメディアの統合リアルタイムストリーミング
"""

import asyncio
import time
import json
import warnings
import websockets
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# WebSocket・HTTP通信
import aiohttp
import websockets

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger
from ..data.batch_data_fetcher import DataRequest

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class StreamConfig:
    """ストリーム設定"""
    # WebSocket設定
    max_connections: int = 10
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    heartbeat_interval: float = 30.0

    # データ設定
    buffer_size: int = 1000  # データポイント保持数
    data_quality_threshold: float = 0.95  # 品質閾値

    # パフォーマンス設定
    batch_size: int = 100
    processing_interval: float = 1.0  # 処理間隔（秒）

    # API Keys (環境変数から取得推奨)
    alpha_vantage_key: Optional[str] = None
    news_api_key: Optional[str] = None

    # 監視銘柄
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"])

@dataclass
class MarketTick:
    """市場ティックデータ"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float = 0.0
    ask: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0

    # メタデータ
    source: str = "unknown"
    quality_score: float = 1.0
    latency_ms: float = 0.0

@dataclass
class NewsItem:
    """ニュース項目"""
    title: str
    content: str
    timestamp: datetime
    source: str
    url: Optional[str] = None
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    symbols: List[str] = field(default_factory=list)

@dataclass
class SocialPost:
    """ソーシャル投稿"""
    platform: str
    content: str
    timestamp: datetime
    author: str
    engagement: int = 0
    sentiment_score: float = 0.0
    symbols: List[str] = field(default_factory=list)

class WebSocketDataStream:
    """WebSocketデータストリーム基底クラス"""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.is_connected = False
        self.websocket = None
        self.reconnect_count = 0
        self.last_heartbeat = time.time()

        # データバッファ
        self.data_buffer: List[Any] = []
        self.buffer_lock = asyncio.Lock()

        # 統計
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'connection_errors': 0,
            'data_quality_issues': 0,
            'average_latency': 0.0
        }

    async def connect(self, uri: str, headers: Optional[Dict] = None):
        """WebSocket接続"""
        try:
            self.websocket = await websockets.connect(
                uri,
                extra_headers=headers or {},
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=10,
                close_timeout=10
            )

            self.is_connected = True
            self.reconnect_count = 0
            logger.info(f"WebSocket connected: {uri}")

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.is_connected = False
            raise

    async def disconnect(self):
        """WebSocket切断"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("WebSocket disconnected")

    async def send_message(self, message: Dict):
        """メッセージ送信"""
        if not self.is_connected or not self.websocket:
            raise ConnectionError("WebSocket not connected")

        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise

    async def listen(self, message_handler: Callable):
        """メッセージリスニング"""
        if not self.is_connected:
            raise ConnectionError("WebSocket not connected")

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await message_handler(data)
                    self.stats['messages_received'] += 1

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message[:100]}...")
                except Exception as e:
                    logger.error(f"Message handling error: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"WebSocket listening error: {e}")
            self.is_connected = False

    async def auto_reconnect(self, connect_func: Callable):
        """自動再接続"""
        while self.reconnect_count < self.config.max_reconnect_attempts:
            try:
                if not self.is_connected:
                    logger.info(f"Attempting reconnection #{self.reconnect_count + 1}")
                    await connect_func()

                    if self.is_connected:
                        logger.info("Reconnection successful")
                        return True

            except Exception as e:
                logger.error(f"Reconnection failed: {e}")

            self.reconnect_count += 1
            await asyncio.sleep(self.config.reconnect_delay)

        logger.error("Max reconnection attempts exceeded")
        return False

class YahooFinanceStream(WebSocketDataStream):
    """Yahoo Finance WebSocketストリーム"""

    def __init__(self, config: StreamConfig):
        super().__init__(config)
        self.base_url = "wss://streamer.finance.yahoo.com"

    async def start_stream(self, symbols: List[str]):
        """ストリーミング開始"""

        # Yahoo Finance WebSocketは実際には異なる実装が必要
        # ここでは模擬的な実装
        logger.info(f"Starting Yahoo Finance stream for: {symbols}")

        # 模擬WebSocketサーバーに接続（実際のYahoo APIを使用する場合は適宜変更）
        mock_uri = "ws://localhost:8765/yahoo"  # 模擬サーバー

        try:
            # 接続試行
            await self.connect(mock_uri)

            # 購読メッセージ送信
            subscribe_message = {
                "subscribe": symbols
            }
            await self.send_message(subscribe_message)

            # リスニング開始
            await self.listen(self._handle_yahoo_message)

        except Exception as e:
            logger.error(f"Yahoo Finance stream error: {e}")
            # 自動再接続試行
            await self.auto_reconnect(lambda: self.start_stream(symbols))

    async def _handle_yahoo_message(self, data: Dict):
        """Yahoo Financeメッセージ処理"""
        try:
            if 'id' in data and 'price' in data:
                tick = MarketTick(
                    symbol=data.get('id', 'UNKNOWN'),
                    timestamp=datetime.fromtimestamp(data.get('time', time.time())),
                    price=float(data.get('price', 0)),
                    volume=int(data.get('volume', 0)),
                    bid=float(data.get('bid', 0)),
                    ask=float(data.get('ask', 0)),
                    change=float(data.get('change', 0)),
                    change_percent=float(data.get('changePercent', 0)),
                    source="yahoo_finance",
                    latency_ms=data.get('latency', 0)
                )

                # データバッファに追加
                async with self.buffer_lock:
                    self.data_buffer.append(tick)

                    # バッファサイズ制限
                    if len(self.data_buffer) > self.config.buffer_size:
                        self.data_buffer.pop(0)

                self.stats['messages_processed'] += 1

        except Exception as e:
            logger.error(f"Yahoo message processing error: {e}")

class NewsAPIStream:
    """NewsAPI ストリーム（HTTPポーリング）"""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.api_key = config.news_api_key
        self.base_url = "https://newsapi.org/v2"
        self.last_fetch = datetime.now() - timedelta(hours=1)

        # HTTPセッション
        self.session: Optional[aiohttp.ClientSession] = None

    async def start_stream(self, keywords: List[str] = None):
        """ニュースストリーミング開始"""

        keywords = keywords or ["stock market", "trading", "finance"]

        logger.info(f"Starting NewsAPI stream for: {keywords}")

        self.session = aiohttp.ClientSession()

        try:
            while True:
                try:
                    news_items = await self._fetch_news(keywords)

                    for item in news_items:
                        logger.debug(f"News: {item.title[:50]}...")

                    # ポーリング間隔
                    await asyncio.sleep(300)  # 5分間隔

                except Exception as e:
                    logger.error(f"News fetching error: {e}")
                    await asyncio.sleep(60)  # エラー時は1分待機

        finally:
            if self.session:
                await self.session.close()

    async def _fetch_news(self, keywords: List[str]) -> List[NewsItem]:
        """ニュース取得"""

        if not self.api_key:
            # 模擬ニュース生成
            return self._generate_mock_news(keywords)

        news_items = []

        try:
            query = " OR ".join(keywords)
            params = {
                'q': query,
                'sortBy': 'publishedAt',
                'from': self.last_fetch.isoformat(),
                'apiKey': self.api_key,
                'language': 'en'
            }

            url = f"{self.base_url}/everything"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    for article in data.get('articles', []):
                        news_item = NewsItem(
                            title=article.get('title', ''),
                            content=article.get('description', ''),
                            timestamp=datetime.fromisoformat(
                                article.get('publishedAt', '').replace('Z', '+00:00')
                            ),
                            source=article.get('source', {}).get('name', ''),
                            url=article.get('url'),
                            symbols=self._extract_symbols(article.get('title', '') +
                                                        article.get('description', ''))
                        )
                        news_items.append(news_item)

                    self.last_fetch = datetime.now()

                else:
                    logger.error(f"NewsAPI error: {response.status}")

        except Exception as e:
            logger.error(f"News API request error: {e}")

        return news_items

    def _generate_mock_news(self, keywords: List[str]) -> List[NewsItem]:
        """模擬ニュース生成"""

        mock_headlines = [
            "Stock Market Shows Strong Performance Amid Economic Recovery",
            "Tech Giants Report Record Quarterly Earnings",
            "Federal Reserve Maintains Interest Rates",
            "Cryptocurrency Market Experiences High Volatility",
            "Investment Flows Continue Into ESG Funds"
        ]

        news_items = []

        for i, headline in enumerate(mock_headlines):
            if i >= 3:  # 3件まで
                break

            news_item = NewsItem(
                title=headline,
                content=f"Mock news content for: {headline}",
                timestamp=datetime.now() - timedelta(minutes=i*30),
                source="Mock News",
                sentiment_score=np.random.uniform(-0.5, 0.5),
                relevance_score=np.random.uniform(0.6, 0.9),
                symbols=["AAPL", "MSFT"] if "Tech" in headline else ["SPY"]
            )
            news_items.append(news_item)

        return news_items

    def _extract_symbols(self, text: str) -> List[str]:
        """テキストからシンボル抽出（簡易）"""

        symbols = []
        symbol_patterns = ["$AAPL", "$MSFT", "$GOOGL", "$TSLA", "$NVDA"]

        for pattern in symbol_patterns:
            if pattern in text.upper():
                symbols.append(pattern[1:])  # $を除去

        return symbols

class RealTimeStreamManager:
    """リアルタイムストリーム管理システム"""

    def __init__(self, config: StreamConfig):
        self.config = config

        # ストリーム一覧
        self.market_stream = YahooFinanceStream(config)
        self.news_stream = NewsAPIStream(config)

        # 統合データバッファ
        self.market_data: List[MarketTick] = []
        self.news_data: List[NewsItem] = []
        self.social_data: List[SocialPost] = []

        # コールバック
        self.data_callbacks: List[Callable] = []

        # タスク管理
        self.running_tasks: List[asyncio.Task] = []

        logger.info("Real-Time Stream Manager initialized")

    def add_data_callback(self, callback: Callable):
        """データコールバック追加"""
        self.data_callbacks.append(callback)

    async def start_all_streams(self):
        """全ストリーム開始"""

        logger.info("Starting all real-time streams...")

        try:
            # 市場データストリーム
            market_task = asyncio.create_task(
                self.market_stream.start_stream(self.config.symbols)
            )
            self.running_tasks.append(market_task)

            # ニュースストリーム
            news_task = asyncio.create_task(
                self.news_stream.start_stream()
            )
            self.running_tasks.append(news_task)

            # データ処理タスク
            processing_task = asyncio.create_task(
                self._data_processing_loop()
            )
            self.running_tasks.append(processing_task)

            logger.info(f"Started {len(self.running_tasks)} streaming tasks")

            # タスク監視
            await self._monitor_tasks()

        except Exception as e:
            logger.error(f"Stream startup error: {e}")
            await self.stop_all_streams()

    async def stop_all_streams(self):
        """全ストリーム停止"""

        logger.info("Stopping all streams...")

        # WebSocket切断
        await self.market_stream.disconnect()

        # HTTPセッション終了
        if hasattr(self.news_stream, 'session') and self.news_stream.session:
            await self.news_stream.session.close()

        # タスクキャンセル
        for task in self.running_tasks:
            if not task.done():
                task.cancel()

        # タスク完了待機
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks, return_exceptions=True)

        self.running_tasks.clear()
        logger.info("All streams stopped")

    async def _data_processing_loop(self):
        """データ処理ループ"""

        logger.info("Data processing loop started")

        while True:
            try:
                # 市場データ処理
                await self._process_market_data()

                # ニュースデータ処理
                await self._process_news_data()

                # コールバック実行
                await self._execute_callbacks()

                # 統計更新
                self._update_statistics()

                # 処理間隔
                await asyncio.sleep(self.config.processing_interval)

            except Exception as e:
                logger.error(f"Data processing error: {e}")
                await asyncio.sleep(1)

    async def _process_market_data(self):
        """市場データ処理"""

        # バッファからデータ取得
        async with self.market_stream.buffer_lock:
            if self.market_stream.data_buffer:
                new_data = self.market_stream.data_buffer.copy()
                self.market_stream.data_buffer.clear()

                self.market_data.extend(new_data)

                # 古いデータ削除
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.market_data = [
                    tick for tick in self.market_data
                    if tick.timestamp > cutoff_time
                ]

    async def _process_news_data(self):
        """ニュースデータ処理"""
        # ニュースデータは直接処理（バッファなし）
        pass

    async def _execute_callbacks(self):
        """データコールバック実行"""

        if not self.data_callbacks:
            return

        # 最新データ準備
        latest_data = {
            'market_ticks': self.market_data[-10:] if self.market_data else [],
            'news_items': self.news_data[-5:] if self.news_data else [],
            'social_posts': self.social_data[-5:] if self.social_data else [],
            'timestamp': datetime.now()
        }

        # 全コールバック実行
        for callback in self.data_callbacks:
            try:
                await callback(latest_data)
            except Exception as e:
                logger.error(f"Callback execution error: {e}")

    def _update_statistics(self):
        """統計情報更新"""

        stats = {
            'market_stream_stats': self.market_stream.stats,
            'total_market_data': len(self.market_data),
            'total_news_data': len(self.news_data),
            'active_tasks': len([t for t in self.running_tasks if not t.done()])
        }

        # ログ出力（5分間隔）
        if int(time.time()) % 300 == 0:
            logger.info(f"Stream statistics: {stats}")

    async def _monitor_tasks(self):
        """タスク監視"""

        while self.running_tasks:
            # 完了したタスクをチェック
            done_tasks = [task for task in self.running_tasks if task.done()]

            for task in done_tasks:
                try:
                    result = await task
                    logger.info(f"Task completed: {task}")
                except Exception as e:
                    logger.error(f"Task failed: {e}")

                self.running_tasks.remove(task)

            # 生きているタスクがあれば継続監視
            if self.running_tasks:
                await asyncio.sleep(10)
            else:
                break

    def get_latest_data(self, symbol: str = None, limit: int = 10) -> Dict:
        """最新データ取得"""

        market_data = self.market_data
        if symbol:
            market_data = [tick for tick in market_data if tick.symbol == symbol]

        return {
            'market_ticks': market_data[-limit:],
            'news_items': self.news_data[-limit:],
            'social_posts': self.social_data[-limit:],
            'timestamp': datetime.now(),
            'data_quality': self._calculate_data_quality()
        }

    def _calculate_data_quality(self) -> float:
        """データ品質計算"""

        if not self.market_data:
            return 0.0

        # 最近1分間のデータ数
        recent_cutoff = datetime.now() - timedelta(minutes=1)
        recent_data = [tick for tick in self.market_data if tick.timestamp > recent_cutoff]

        # 期待データ数との比較
        expected_ticks = len(self.config.symbols) * 60  # 1分間で1銘柄60ティック想定
        actual_ticks = len(recent_data)

        quality_score = min(actual_ticks / expected_ticks, 1.0) if expected_ticks > 0 else 0.0

        return quality_score

# 便利関数
async def create_realtime_stream_manager(symbols: List[str] = None) -> RealTimeStreamManager:
    """リアルタイムストリーム管理システム作成"""

    config = StreamConfig(
        symbols=symbols or ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
        buffer_size=1000,
        processing_interval=1.0
    )

    manager = RealTimeStreamManager(config)
    return manager

if __name__ == "__main__":
    # WebSocketストリーミングテスト
    async def test_websocket_streaming():
        print("=== WebSocket Streaming System Test ===")

        # テスト用コールバック
        async def data_callback(data):
            market_count = len(data['market_ticks'])
            news_count = len(data['news_items'])
            print(f"Data received: {market_count} market ticks, {news_count} news items")

        try:
            # ストリーム管理システム作成
            manager = await create_realtime_stream_manager(["AAPL", "MSFT"])

            # データコールバック追加
            manager.add_data_callback(data_callback)

            print("Starting real-time streams...")

            # ストリーミング開始（テスト用に短時間）
            start_task = asyncio.create_task(manager.start_all_streams())

            # 10秒後に停止
            await asyncio.sleep(10)

            print("Stopping streams...")
            await manager.stop_all_streams()

            # 最新データ取得
            latest_data = manager.get_latest_data()
            print(f"Final data count: {len(latest_data['market_ticks'])} market ticks")

            print("WebSocket streaming test completed")

        except Exception as e:
            print(f"Test error: {e}")
            import traceback
            traceback.print_exc()

    # テスト実行
    asyncio.run(test_websocket_streaming())
