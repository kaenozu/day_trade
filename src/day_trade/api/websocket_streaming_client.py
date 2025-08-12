#!/usr/bin/env python3
"""
WebSocketリアルタイムデータストリーミングシステム
Issue #331: API・外部統合システム - Phase 2

WebSocketリアルタイム市場データストリーミング・自動再接続・データ配信
- リアルタイム価格ストリーミング
- 自動接続回復システム
- マルチチャンネル対応
- バックプレッシャー制御
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosedError, WebSocketException

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class StreamProvider(Enum):
    """ストリーミングプロバイダー"""

    FINNHUB = "finnhub"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"
    TWELVEDATA = "twelvedata"
    BINANCE = "binance"
    MOCK_STREAM = "mock_stream"  # 開発・テスト用


class StreamType(Enum):
    """ストリームタイプ"""

    REAL_TIME_QUOTES = "real_time_quotes"
    TRADES = "trades"
    ORDER_BOOK = "order_book"
    MARKET_NEWS = "market_news"
    ECONOMIC_DATA = "economic_data"
    INDEX_DATA = "index_data"
    CRYPTO_PRICES = "crypto_prices"


class MessageType(Enum):
    """メッセージタイプ"""

    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    DATA = "data"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    STATUS = "status"


@dataclass
class StreamSubscription:
    """ストリーム購読設定"""

    provider: StreamProvider
    stream_type: StreamType
    symbols: List[str]
    callback: Callable[[Dict[str, Any]], None]

    # 接続設定
    websocket_url: str
    auth_required: bool = False
    api_key: Optional[str] = None

    # 購読設定
    subscription_id: Optional[str] = None
    active: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    # パフォーマンス設定
    buffer_size: int = 1000
    max_reconnect_attempts: int = 5
    heartbeat_interval_seconds: int = 30


@dataclass
class StreamMessage:
    """ストリームメッセージ"""

    subscription_id: str
    provider: StreamProvider
    stream_type: StreamType
    symbol: str
    data: Dict[str, Any]

    # メタデータ
    timestamp: datetime
    sequence_number: Optional[int] = None
    message_type: MessageType = MessageType.DATA
    raw_message: Optional[str] = None


@dataclass
class ConnectionState:
    """接続状態"""

    provider: StreamProvider
    websocket_url: str
    connected: bool = False
    last_connected_at: Optional[datetime] = None
    last_disconnected_at: Optional[datetime] = None

    # 再接続管理
    reconnect_attempts: int = 0
    max_reconnect_attempts: int = 5
    reconnect_delay_seconds: float = 1.0

    # 統計情報
    messages_received: int = 0
    messages_processed: int = 0
    connection_errors: int = 0
    last_message_at: Optional[datetime] = None


@dataclass
class StreamConfig:
    """ストリーミング設定"""

    # 接続設定
    connection_timeout_seconds: int = 10
    read_timeout_seconds: int = 30
    ping_interval_seconds: int = 20

    # バッファリング設定
    message_buffer_size: int = 10000
    enable_message_buffering: bool = True
    buffer_overflow_strategy: str = "drop_oldest"  # drop_oldest, drop_newest, block

    # 再接続設定
    enable_auto_reconnect: bool = True
    max_reconnect_attempts: int = 10
    initial_reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    exponential_backoff: bool = True

    # パフォーマンス設定
    max_concurrent_connections: int = 5
    message_processing_batch_size: int = 100
    enable_compression: bool = True

    # 認証設定
    api_keys: Dict[str, str] = field(default_factory=dict)


class WebSocketStreamingClient:
    """WebSocketストリーミングクライアント"""

    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.connections: Dict[StreamProvider, ConnectionState] = {}
        self.websockets: Dict[StreamProvider, websockets.WebSocketServerProtocol] = {}

        # メッセージ処理
        self.message_buffer: Dict[str, List[StreamMessage]] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}

        # 制御フラグ
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # 統計情報
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_subscriptions": 0,
            "messages_received": 0,
            "messages_processed": 0,
            "connection_errors": 0,
            "reconnections": 0,
        }

        # デフォルトストリーム設定
        self._setup_default_streams()

    def _setup_default_streams(self) -> None:
        """デフォルトストリーム設定"""
        # Mock Stream（開発・テスト用）
        self.register_stream_provider(
            provider=StreamProvider.MOCK_STREAM,
            websocket_url="ws://localhost:8765/stream",
            supports_auth=False,
        )

        # Finnhub（要APIキー）
        self.register_stream_provider(
            provider=StreamProvider.FINNHUB,
            websocket_url="wss://ws.finnhub.io",
            supports_auth=True,
        )

        # Alpha Vantage（要APIキー）
        self.register_stream_provider(
            provider=StreamProvider.ALPHA_VANTAGE,
            websocket_url="wss://www.alphavantage.co/stream",
            supports_auth=True,
        )

    def register_stream_provider(
        self, provider: StreamProvider, websocket_url: str, supports_auth: bool = False
    ) -> None:
        """ストリームプロバイダー登録"""
        self.connections[provider] = ConnectionState(
            provider=provider,
            websocket_url=websocket_url,
            max_reconnect_attempts=self.config.max_reconnect_attempts,
        )

        logger.info(f"ストリームプロバイダー登録: {provider.value} ({websocket_url})")

    async def start_streaming(self) -> None:
        """ストリーミング開始"""
        if self._running:
            logger.warning("ストリーミングは既に実行中です")
            return

        self._running = True

        # メッセージ処理タスク開始
        message_task = asyncio.create_task(self._message_processing_loop())
        self._tasks.append(message_task)

        logger.info("WebSocketストリーミング開始")

    async def stop_streaming(self) -> None:
        """ストリーミング停止"""
        self._running = False

        # 全接続を閉じる
        for provider, websocket in self.websockets.items():
            if websocket and not websocket.closed:
                await websocket.close()

        self.websockets.clear()

        # 全タスクを停止
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._tasks.clear()

        logger.info("WebSocketストリーミング停止")

    async def subscribe(
        self,
        provider: StreamProvider,
        stream_type: StreamType,
        symbols: List[str],
        callback: Callable[[StreamMessage], None],
    ) -> str:
        """ストリーム購読"""

        if provider not in self.connections:
            raise ValueError(f"未登録のプロバイダー: {provider.value}")

        connection_state = self.connections[provider]
        subscription_id = f"{provider.value}_{stream_type.value}_{int(time.time())}"

        # API キー取得
        api_key = self.config.api_keys.get(provider.value)

        subscription = StreamSubscription(
            provider=provider,
            stream_type=stream_type,
            symbols=symbols,
            callback=callback,
            websocket_url=connection_state.websocket_url,
            auth_required=bool(api_key),
            api_key=api_key,
            subscription_id=subscription_id,
        )

        self.subscriptions[subscription_id] = subscription

        # メッセージハンドラー登録
        if subscription_id not in self.message_handlers:
            self.message_handlers[subscription_id] = []
        self.message_handlers[subscription_id].append(callback)

        # 接続・購読開始
        await self._establish_connection_and_subscribe(subscription)

        self.stats["total_subscriptions"] += 1

        logger.info(f"ストリーム購読開始: {subscription_id} ({len(symbols)} 銘柄)")

        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """ストリーム購読解除"""
        if subscription_id not in self.subscriptions:
            logger.warning(f"存在しない購読ID: {subscription_id}")
            return False

        subscription = self.subscriptions[subscription_id]

        # WebSocket経由で購読解除
        await self._send_unsubscribe_message(subscription)

        # 購読情報削除
        subscription.active = False
        del self.subscriptions[subscription_id]

        # メッセージハンドラー削除
        if subscription_id in self.message_handlers:
            del self.message_handlers[subscription_id]

        logger.info(f"ストリーム購読解除: {subscription_id}")

        return True

    async def _establish_connection_and_subscribe(
        self, subscription: StreamSubscription
    ) -> None:
        """接続確立・購読開始"""
        provider = subscription.provider

        # 既存接続チェック
        if provider in self.websockets and not self.websockets[provider].closed:
            # 既存接続に追加購読
            await self._send_subscribe_message(subscription)
            subscription.active = True
            return

        # 新規接続作成
        connection_task = asyncio.create_task(self._connect_websocket(subscription))
        self._tasks.append(connection_task)

    async def _connect_websocket(self, subscription: StreamSubscription) -> None:
        """WebSocket接続"""
        provider = subscription.provider
        connection_state = self.connections[provider]

        retry_count = 0

        while self._running and retry_count < connection_state.max_reconnect_attempts:
            try:
                logger.info(f"WebSocket接続試行: {provider.value}")

                # WebSocket接続
                websocket = await websockets.connect(
                    subscription.websocket_url,
                    timeout=self.config.connection_timeout_seconds,
                    compression="deflate" if self.config.enable_compression else None,
                    ping_interval=self.config.ping_interval_seconds,
                )

                self.websockets[provider] = websocket
                connection_state.connected = True
                connection_state.last_connected_at = datetime.now()
                connection_state.reconnect_attempts = 0

                self.stats["total_connections"] += 1
                self.stats["active_connections"] += 1

                logger.info(f"WebSocket接続成功: {provider.value}")

                # 認証
                if subscription.auth_required:
                    await self._authenticate(websocket, subscription)

                # 購読開始
                await self._send_subscribe_message(subscription)
                subscription.active = True

                # メッセージ受信ループ
                await self._message_receive_loop(websocket, provider)

            except asyncio.CancelledError:
                break
            except Exception as e:
                retry_count += 1
                connection_state.connection_errors += 1
                self.stats["connection_errors"] += 1

                logger.error(
                    f"WebSocket接続エラー {provider.value} (試行 {retry_count}): {e}"
                )

                if retry_count < connection_state.max_reconnect_attempts:
                    delay = self._calculate_reconnect_delay(retry_count)
                    logger.info(f"再接続まで {delay:.1f} 秒待機")
                    await asyncio.sleep(delay)

                    if self.config.enable_auto_reconnect:
                        self.stats["reconnections"] += 1

        # 接続失敗・切断処理
        connection_state.connected = False
        connection_state.last_disconnected_at = datetime.now()

        if provider in self.websockets:
            del self.websockets[provider]
            self.stats["active_connections"] = max(
                0, self.stats["active_connections"] - 1
            )

    async def _authenticate(self, websocket, subscription: StreamSubscription) -> None:
        """WebSocket認証"""
        if not subscription.api_key:
            raise ValueError(
                f"APIキーが設定されていません: {subscription.provider.value}"
            )

        # プロバイダー別認証メッセージ
        if subscription.provider == StreamProvider.FINNHUB:
            auth_message = {"type": "auth", "token": subscription.api_key}
        elif subscription.provider == StreamProvider.ALPHA_VANTAGE:
            auth_message = {"type": "auth", "apikey": subscription.api_key}
        else:
            # 汎用認証
            auth_message = {"type": "auth", "api_key": subscription.api_key}

        await websocket.send(json.dumps(auth_message))
        logger.info(f"認証メッセージ送信: {subscription.provider.value}")

    async def _send_subscribe_message(self, subscription: StreamSubscription) -> None:
        """購読メッセージ送信"""
        provider = subscription.provider

        if provider not in self.websockets:
            logger.error(f"WebSocket接続なし: {provider.value}")
            return

        websocket = self.websockets[provider]

        # プロバイダー別購読メッセージ
        if provider == StreamProvider.FINNHUB:
            for symbol in subscription.symbols:
                subscribe_message = {"type": "subscribe", "symbol": symbol}
                await websocket.send(json.dumps(subscribe_message))

        elif provider == StreamProvider.MOCK_STREAM:
            subscribe_message = {
                "action": "subscribe",
                "stream": subscription.stream_type.value,
                "symbols": subscription.symbols,
            }
            await websocket.send(json.dumps(subscribe_message))

        else:
            # 汎用購読メッセージ
            subscribe_message = {
                "type": "subscribe",
                "stream": subscription.stream_type.value,
                "symbols": subscription.symbols,
            }
            await websocket.send(json.dumps(subscribe_message))

        logger.info(
            f"購読メッセージ送信: {provider.value} ({len(subscription.symbols)} 銘柄)"
        )

    async def _send_unsubscribe_message(self, subscription: StreamSubscription) -> None:
        """購読解除メッセージ送信"""
        provider = subscription.provider

        if provider not in self.websockets:
            return

        websocket = self.websockets[provider]

        # プロバイダー別購読解除メッセージ
        if provider == StreamProvider.FINNHUB:
            for symbol in subscription.symbols:
                unsubscribe_message = {"type": "unsubscribe", "symbol": symbol}
                await websocket.send(json.dumps(unsubscribe_message))

        elif provider == StreamProvider.MOCK_STREAM:
            unsubscribe_message = {
                "action": "unsubscribe",
                "stream": subscription.stream_type.value,
                "symbols": subscription.symbols,
            }
            await websocket.send(json.dumps(unsubscribe_message))

        logger.info(f"購読解除メッセージ送信: {provider.value}")

    async def _message_receive_loop(self, websocket, provider: StreamProvider) -> None:
        """メッセージ受信ループ"""
        connection_state = self.connections[provider]

        try:
            async for raw_message in websocket:
                if not self._running:
                    break

                try:
                    # メッセージ解析
                    message_data = (
                        json.loads(raw_message)
                        if isinstance(raw_message, str)
                        else raw_message
                    )

                    # ストリームメッセージ作成
                    stream_messages = await self._parse_stream_message(
                        provider, message_data, raw_message
                    )

                    # メッセージバッファに追加
                    for stream_message in stream_messages:
                        await self._buffer_message(stream_message)

                    connection_state.messages_received += len(stream_messages)
                    connection_state.last_message_at = datetime.now()

                    self.stats["messages_received"] += len(stream_messages)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析エラー: {e}")
                except Exception as e:
                    logger.error(f"メッセージ処理エラー: {e}")

        except ConnectionClosedError:
            logger.warning(f"WebSocket接続切断: {provider.value}")
        except WebSocketException as e:
            logger.error(f"WebSocketエラー: {e}")
        except Exception as e:
            logger.error(f"メッセージ受信エラー: {e}")

    async def _parse_stream_message(
        self, provider: StreamProvider, message_data: Dict[str, Any], raw_message: str
    ) -> List[StreamMessage]:
        """ストリームメッセージ解析"""

        messages = []

        try:
            if provider == StreamProvider.MOCK_STREAM:
                messages = await self._parse_mock_message(message_data, raw_message)
            elif provider == StreamProvider.FINNHUB:
                messages = await self._parse_finnhub_message(message_data, raw_message)
            elif provider == StreamProvider.ALPHA_VANTAGE:
                messages = await self._parse_alpha_vantage_message(
                    message_data, raw_message
                )
            else:
                messages = await self._parse_generic_message(
                    provider, message_data, raw_message
                )

        except Exception as e:
            logger.error(f"メッセージ解析エラー {provider.value}: {e}")

        return messages

    async def _parse_mock_message(
        self, message_data: Dict[str, Any], raw_message: str
    ) -> List[StreamMessage]:
        """モックメッセージ解析"""
        messages = []

        if message_data.get("type") == "quote":
            for quote in message_data.get("data", []):
                message = StreamMessage(
                    subscription_id=f"mock_{quote.get('symbol', 'UNKNOWN')}",
                    provider=StreamProvider.MOCK_STREAM,
                    stream_type=StreamType.REAL_TIME_QUOTES,
                    symbol=quote.get("symbol", "UNKNOWN"),
                    data={
                        "price": quote.get("price", 0),
                        "volume": quote.get("volume", 0),
                        "bid": quote.get("bid", 0),
                        "ask": quote.get("ask", 0),
                        "timestamp": quote.get("timestamp", datetime.now().isoformat()),
                    },
                    timestamp=datetime.now(),
                    raw_message=raw_message,
                )
                messages.append(message)

        return messages

    async def _parse_finnhub_message(
        self, message_data: Dict[str, Any], raw_message: str
    ) -> List[StreamMessage]:
        """Finnhubメッセージ解析"""
        messages = []

        if message_data.get("type") == "trade":
            for trade in message_data.get("data", []):
                message = StreamMessage(
                    subscription_id=f"finnhub_{trade.get('s', 'UNKNOWN')}",
                    provider=StreamProvider.FINNHUB,
                    stream_type=StreamType.TRADES,
                    symbol=trade.get("s", "UNKNOWN"),
                    data={
                        "price": trade.get("p", 0),
                        "volume": trade.get("v", 0),
                        "timestamp": trade.get("t", 0),
                        "conditions": trade.get("c", []),
                    },
                    timestamp=datetime.now(),
                    raw_message=raw_message,
                )
                messages.append(message)

        return messages

    async def _parse_alpha_vantage_message(
        self, message_data: Dict[str, Any], raw_message: str
    ) -> List[StreamMessage]:
        """Alpha Vantageメッセージ解析"""
        messages = []

        if "symbol" in message_data and "price" in message_data:
            message = StreamMessage(
                subscription_id=f"alphavantage_{message_data['symbol']}",
                provider=StreamProvider.ALPHA_VANTAGE,
                stream_type=StreamType.REAL_TIME_QUOTES,
                symbol=message_data["symbol"],
                data={
                    "price": message_data.get("price", 0),
                    "volume": message_data.get("volume", 0),
                    "change": message_data.get("change", 0),
                    "change_percent": message_data.get("change_percent", 0),
                },
                timestamp=datetime.now(),
                raw_message=raw_message,
            )
            messages.append(message)

        return messages

    async def _parse_generic_message(
        self, provider: StreamProvider, message_data: Dict[str, Any], raw_message: str
    ) -> List[StreamMessage]:
        """汎用メッセージ解析"""
        messages = []

        # 基本的な構造を仮定
        symbol = message_data.get("symbol", message_data.get("s", "UNKNOWN"))

        message = StreamMessage(
            subscription_id=f"generic_{provider.value}_{symbol}",
            provider=provider,
            stream_type=StreamType.REAL_TIME_QUOTES,
            symbol=symbol,
            data=message_data,
            timestamp=datetime.now(),
            raw_message=raw_message,
        )
        messages.append(message)

        return messages

    async def _buffer_message(self, message: StreamMessage) -> None:
        """メッセージバッファ追加"""
        subscription_id = message.subscription_id

        if not self.config.enable_message_buffering:
            # バッファリング無効の場合は即座に処理
            await self._process_message(message)
            return

        # バッファ初期化
        if subscription_id not in self.message_buffer:
            self.message_buffer[subscription_id] = []

        buffer = self.message_buffer[subscription_id]

        # バッファサイズ制限
        if len(buffer) >= self.config.message_buffer_size:
            if self.config.buffer_overflow_strategy == "drop_oldest":
                buffer.pop(0)
            elif self.config.buffer_overflow_strategy == "drop_newest":
                return  # 新しいメッセージを破棄
            # "block"の場合は何もせず（バッファが満杯の間は待機）

        buffer.append(message)

    async def _message_processing_loop(self) -> None:
        """メッセージ処理ループ"""
        while self._running:
            try:
                # 全購読のメッセージをバッチ処理
                for subscription_id, message_buffer in self.message_buffer.items():
                    if message_buffer:
                        # バッチサイズ分取得
                        batch_size = min(
                            len(message_buffer),
                            self.config.message_processing_batch_size,
                        )
                        messages_to_process = message_buffer[:batch_size]
                        del message_buffer[:batch_size]

                        # バッチ処理
                        for message in messages_to_process:
                            await self._process_message(message)
                            self.stats["messages_processed"] += 1

                # 処理間隔
                await asyncio.sleep(0.01)  # 10ms

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"メッセージ処理ループエラー: {e}")
                await asyncio.sleep(1.0)

    async def _process_message(self, message: StreamMessage) -> None:
        """メッセージ処理"""
        # 関連する購読のコールバック実行
        handlers = self.message_handlers.get(message.subscription_id, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"メッセージハンドラーエラー: {e}")

    def _calculate_reconnect_delay(self, retry_count: int) -> float:
        """再接続遅延時間計算"""
        if not self.config.exponential_backoff:
            return self.config.initial_reconnect_delay

        delay = self.config.initial_reconnect_delay * (2 ** (retry_count - 1))
        return min(delay, self.config.max_reconnect_delay)

    def get_subscription_status(self) -> Dict[str, Any]:
        """購読状況取得"""
        active_subscriptions = {
            sub_id: {
                "provider": sub.provider.value,
                "stream_type": sub.stream_type.value,
                "symbols": sub.symbols,
                "active": sub.active,
                "created_at": sub.created_at.isoformat(),
            }
            for sub_id, sub in self.subscriptions.items()
        }

        return {
            "total_subscriptions": len(self.subscriptions),
            "active_subscriptions": len(
                [s for s in self.subscriptions.values() if s.active]
            ),
            "subscriptions": active_subscriptions,
        }

    def get_connection_status(self) -> Dict[str, Any]:
        """接続状況取得"""
        connection_info = {}

        for provider, state in self.connections.items():
            connection_info[provider.value] = {
                "connected": state.connected,
                "messages_received": state.messages_received,
                "connection_errors": state.connection_errors,
                "reconnect_attempts": state.reconnect_attempts,
                "last_connected": (
                    state.last_connected_at.isoformat()
                    if state.last_connected_at
                    else None
                ),
                "last_message": (
                    state.last_message_at.isoformat() if state.last_message_at else None
                ),
            }

        return connection_info

    def get_streaming_statistics(self) -> Dict[str, Any]:
        """ストリーミング統計取得"""
        buffer_sizes = {
            sub_id: len(buffer) for sub_id, buffer in self.message_buffer.items()
        }

        return {
            **self.stats,
            "message_buffer_sizes": buffer_sizes,
            "total_buffer_messages": sum(buffer_sizes.values()),
            "active_subscriptions": len(
                [s for s in self.subscriptions.values() if s.active]
            ),
            "is_running": self._running,
        }

    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        return {
            "status": "healthy" if self._running else "stopped",
            "active_connections": self.stats["active_connections"],
            "active_subscriptions": len(
                [s for s in self.subscriptions.values() if s.active]
            ),
            "messages_processed_rate": self.stats["messages_processed"],
            "error_rate": self.stats["connection_errors"],
            "timestamp": datetime.now().isoformat(),
        }


# 使用例・テスト関数


async def setup_streaming_client() -> WebSocketStreamingClient:
    """ストリーミングクライアントセットアップ"""
    config = StreamConfig(
        max_concurrent_connections=3,
        enable_auto_reconnect=True,
        message_buffer_size=1000,
    )

    client = WebSocketStreamingClient(config)
    await client.start_streaming()

    return client


async def test_stock_streaming():
    """株価ストリーミングテスト"""
    client = await setup_streaming_client()

    # メッセージハンドラー
    def price_handler(message: StreamMessage):
        print(
            f"📈 {message.symbol}: ¥{message.data.get('price', 0):.2f} "
            f"[{message.timestamp.strftime('%H:%M:%S')}]"
        )

    try:
        # 株価ストリーミング購読
        subscription_id = await client.subscribe(
            provider=StreamProvider.MOCK_STREAM,
            stream_type=StreamType.REAL_TIME_QUOTES,
            symbols=["7203", "8306", "9984"],  # トヨタ、三菱UFJ、SBG
            callback=price_handler,
        )

        print(f"ストリーミング開始: {subscription_id}")

        # 30秒間ストリーミング
        await asyncio.sleep(30)

        # 統計情報表示
        stats = client.get_streaming_statistics()
        print("\n📊 ストリーミング統計:")
        print(f"  受信メッセージ数: {stats['messages_received']}")
        print(f"  処理メッセージ数: {stats['messages_processed']}")
        print(f"  アクティブ接続数: {stats['active_connections']}")

        # 購読解除
        await client.unsubscribe(subscription_id)

    finally:
        await client.stop_streaming()


if __name__ == "__main__":
    asyncio.run(test_stock_streaming())
