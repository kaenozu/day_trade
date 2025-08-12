"""
WebSocketベースリアルタイムデータフィード

Phase 3a-1: WebSocketリアルタイムデータフィード実装
Issue #271対応
"""

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from ..utils.enhanced_error_handler import get_default_error_handler
from ..utils.logging_config import get_context_logger, log_performance_metric
from ..utils.security_helpers import SecurityHelpers

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()


class ConnectionStatus(Enum):
    """WebSocket接続状態"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class DataSource(Enum):
    """データソース種別"""

    MOCK = "mock"  # テスト用モックデータ
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"


@dataclass
class MarketData:
    """市場データ"""

    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    source: str = "unknown"


@dataclass
class WebSocketConfig:
    """WebSocket設定"""

    url: str
    symbols: List[str]
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    heartbeat_interval: float = 30.0
    message_timeout: float = 60.0
    enable_compression: bool = True


class WebSocketClient:
    """WebSocket接続管理クライアント"""

    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.status = ConnectionStatus.DISCONNECTED
        self.reconnect_count = 0
        self.last_heartbeat = time.time()
        self.message_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self._running = False

    async def connect(self) -> bool:
        """WebSocket接続確立"""
        try:
            self.status = ConnectionStatus.CONNECTING
            logger.info(f"WebSocket接続開始: {self.config.url}")

            self.websocket = await websockets.connect(
                self.config.url,
                compression="deflate" if self.config.enable_compression else None,
                ping_interval=self.config.heartbeat_interval,
                ping_timeout=self.config.message_timeout,
                close_timeout=10,
            )

            self.status = ConnectionStatus.CONNECTED
            self.reconnect_count = 0
            self.last_heartbeat = time.time()

            logger.info("WebSocket接続成功")
            await self._send_subscription()
            return True

        except Exception as e:
            self.status = ConnectionStatus.FAILED
            logger.error(f"WebSocket接続失敗: {e}")
            return False

    async def disconnect(self) -> None:
        """WebSocket接続切断"""
        self._running = False
        self.status = ConnectionStatus.DISCONNECTED

        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("WebSocket接続を正常に切断")
            except Exception as e:
                logger.error(f"WebSocket切断エラー: {e}")
            finally:
                self.websocket = None

    async def _send_subscription(self) -> None:
        """購読リクエスト送信"""
        if not self.websocket or self.status != ConnectionStatus.CONNECTED:
            return

        try:
            subscription_message = {
                "action": "subscribe",
                "symbols": self.config.symbols,
                "timestamp": datetime.now().isoformat(),
            }

            # メッセージをログ用にサニタイズ（セキュリティ対策）
            SecurityHelpers.sanitize_log_message(json.dumps(subscription_message))

            await self.websocket.send(json.dumps(subscription_message))
            logger.info(f"購読リクエスト送信: {len(self.config.symbols)}銘柄")

        except Exception as e:
            logger.error(f"購読リクエスト送信エラー: {e}")

    async def receive_messages(self) -> None:
        """メッセージ受信処理"""
        self._running = True

        while self._running and self.websocket:
            try:
                # タイムアウト付きメッセージ受信
                message = await asyncio.wait_for(
                    self.websocket.recv(), timeout=self.config.message_timeout
                )

                self.last_heartbeat = time.time()
                await self._process_message(message)

            except asyncio.TimeoutError:
                logger.warning("WebSocketメッセージ受信タイムアウト")
                if not await self._check_connection_health():
                    await self._handle_reconnect()

            except ConnectionClosed:
                logger.warning("WebSocket接続が閉じられました")
                await self._handle_reconnect()

            except WebSocketException as e:
                logger.error(f"WebSocketエラー: {e}")
                await self._handle_reconnect()

            except Exception as e:
                error_handler.handle_error(e, {"component": "websocket_client"})
                await asyncio.sleep(1)  # エラー後の短い待機

    async def _process_message(self, message: str) -> None:
        """受信メッセージ処理"""
        try:
            # パフォーマンス監視開始
            start_time = time.time()

            data = json.loads(message)

            # 基本的な検証
            if not isinstance(data, dict):
                logger.warning("無効なメッセージ形式を受信")
                return

            # メッセージハンドラーに通知
            for handler in self.message_handlers:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"メッセージハンドラーエラー: {e}")

            # パフォーマンス監視記録
            processing_time = time.time() - start_time
            log_performance_metric(
                "websocket_message_processing",
                processing_time,
                {"message_size": len(message)},
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析エラー: {e}")
        except Exception as e:
            logger.error(f"メッセージ処理エラー: {e}")

    async def _check_connection_health(self) -> bool:
        """接続ヘルス確認"""
        if not self.websocket:
            return False

        try:
            pong = await self.websocket.ping()
            await asyncio.wait_for(pong, timeout=5.0)
            return True
        except Exception:
            return False

    async def _handle_reconnect(self) -> None:
        """再接続処理"""
        if self.reconnect_count >= self.config.max_reconnect_attempts:
            logger.error("最大再接続試行回数に達しました")
            self.status = ConnectionStatus.FAILED
            return

        self.status = ConnectionStatus.RECONNECTING
        self.reconnect_count += 1

        logger.info(f"再接続試行 {self.reconnect_count}/{self.config.max_reconnect_attempts}")

        # 既存接続をクリーンアップ
        if self.websocket:
            with contextlib.suppress(Exception):
                await self.websocket.close()
            self.websocket = None

        # 再接続待機
        await asyncio.sleep(self.config.reconnect_delay)

        # 再接続試行
        if await self.connect():
            logger.info("再接続成功")
        else:
            logger.error("再接続失敗")

    def add_message_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """メッセージハンドラー追加"""
        self.message_handlers.append(handler)

    def remove_message_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """メッセージハンドラー削除"""
        if handler in self.message_handlers:
            self.message_handlers.remove(handler)


class DataNormalizer:
    """データ正規化処理"""

    @staticmethod
    def normalize_market_data(raw_data: Dict[str, Any], source: DataSource) -> Optional[MarketData]:
        """市場データ正規化"""
        try:
            if source == DataSource.MOCK:
                return DataNormalizer._normalize_mock_data(raw_data)
            elif source == DataSource.YAHOO_FINANCE:
                return DataNormalizer._normalize_yahoo_data(raw_data)
            else:
                logger.warning(f"未対応のデータソース: {source}")
                return None

        except Exception as e:
            logger.error(f"データ正規化エラー: {e}")
            return None

    @staticmethod
    def _normalize_mock_data(data: Dict[str, Any]) -> MarketData:
        """モックデータ正規化"""
        return MarketData(
            symbol=data.get("symbol", "UNKNOWN"),
            price=float(data.get("price", 0.0)),
            volume=int(data.get("volume", 0)),
            timestamp=datetime.now(),
            bid=data.get("bid"),
            ask=data.get("ask"),
            high=data.get("high"),
            low=data.get("low"),
            source="mock",
        )

    @staticmethod
    def _normalize_yahoo_data(data: Dict[str, Any]) -> MarketData:
        """Yahoo Finance データ正規化"""
        return MarketData(
            symbol=data.get("symbol", "UNKNOWN"),
            price=float(data.get("regularMarketPrice", 0.0)),
            volume=int(data.get("regularMarketVolume", 0)),
            timestamp=datetime.fromtimestamp(data.get("regularMarketTime", time.time())),
            bid=data.get("bid"),
            ask=data.get("ask"),
            high=data.get("regularMarketDayHigh"),
            low=data.get("regularMarketDayLow"),
            source="yahoo_finance",
        )


class RealtimeDataFeed:
    """リアルタイムデータフィード管理"""

    def __init__(self, data_source: DataSource = DataSource.MOCK):
        self.data_source = data_source
        self.websocket_client: Optional[WebSocketClient] = None
        self.subscribers: Set[Callable[[MarketData], None]] = set()
        self.is_streaming = False
        self.data_buffer: List[MarketData] = []
        self.max_buffer_size = 1000

    async def start_streaming(
        self, symbols: List[str], websocket_url: Optional[str] = None
    ) -> bool:
        """ストリーミング開始"""
        try:
            # WebSocket設定
            if not websocket_url:
                websocket_url = self._get_default_websocket_url()

            config = WebSocketConfig(
                url=websocket_url,
                symbols=symbols,
                reconnect_delay=5.0,
                max_reconnect_attempts=10,
            )

            # WebSocketクライアント初期化
            self.websocket_client = WebSocketClient(config)
            self.websocket_client.add_message_handler(self._handle_market_data)

            # 接続確立
            if await self.websocket_client.connect():
                self.is_streaming = True

                # メッセージ受信を開始（バックグラウンドタスク）
                asyncio.create_task(self.websocket_client.receive_messages())

                logger.info(f"リアルタイムストリーミング開始: {len(symbols)}銘柄")
                return True
            else:
                logger.error("WebSocket接続失敗")
                return False

        except Exception as e:
            logger.error(f"ストリーミング開始エラー: {e}")
            return False

    async def stop_streaming(self) -> None:
        """ストリーミング停止"""
        self.is_streaming = False

        if self.websocket_client:
            await self.websocket_client.disconnect()
            self.websocket_client = None

        logger.info("リアルタイムストリーミング停止")

    def subscribe(self, callback: Callable[[MarketData], None]) -> None:
        """データ更新の購読"""
        self.subscribers.add(callback)
        logger.info(f"新しい購読者追加（総数: {len(self.subscribers)}）")

    def unsubscribe(self, callback: Callable[[MarketData], None]) -> None:
        """データ更新の購読解除"""
        self.subscribers.discard(callback)
        logger.info(f"購読者削除（総数: {len(self.subscribers)}）")

    def _handle_market_data(self, raw_data: Dict[str, Any]) -> None:
        """市場データ処理"""
        try:
            # データ正規化
            market_data = DataNormalizer.normalize_market_data(raw_data, self.data_source)

            if market_data:
                # バッファに追加
                self._add_to_buffer(market_data)

                # 購読者に通知
                self._notify_subscribers(market_data)

        except Exception as e:
            logger.error(f"市場データ処理エラー: {e}")

    def _add_to_buffer(self, data: MarketData) -> None:
        """データバッファ追加"""
        self.data_buffer.append(data)

        # バッファサイズ制限
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size :]

    def _notify_subscribers(self, data: MarketData) -> None:
        """購読者通知"""
        for callback in self.subscribers.copy():  # コピーして安全に反復
            try:
                callback(data)
            except Exception as e:
                logger.error(f"購読者通知エラー: {e}")

    def _get_default_websocket_url(self) -> str:
        """デフォルトWebSocket URL取得"""
        if self.data_source == DataSource.MOCK:
            return "ws://localhost:8765/mock"  # テスト用モックサーバー
        elif self.data_source == DataSource.YAHOO_FINANCE:
            return "wss://streamer.finance.yahoo.com/"
        else:
            raise ValueError(f"未対応のデータソース: {self.data_source}")

    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """最新データ取得"""
        # バッファから指定銘柄の最新データを検索
        for data in reversed(self.data_buffer):
            if data.symbol == symbol:
                return data
        return None

    def get_connection_status(self) -> ConnectionStatus:
        """接続状態取得"""
        if self.websocket_client:
            return self.websocket_client.status
        return ConnectionStatus.DISCONNECTED

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        return {
            "is_streaming": self.is_streaming,
            "connection_status": self.get_connection_status().value,
            "subscribers_count": len(self.subscribers),
            "buffer_size": len(self.data_buffer),
            "data_source": self.data_source.value,
            "reconnect_count": (
                self.websocket_client.reconnect_count if self.websocket_client else 0
            ),
        }
