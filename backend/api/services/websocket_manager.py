# WebSocket Manager Service
# Day Trade ML System - Issue #803

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

# Models
from ..database.models import User

# Services
from .notification_service import NotificationService
from .market_service import MarketService
from .trading_service import TradingService
from .ml_service import MLService

# Utils
from ..utils.metrics import track_websocket_event


@dataclass
class WebSocketConnection:
    """WebSocket接続情報"""
    websocket: WebSocket
    user_id: str
    connection_id: str
    connected_at: datetime
    last_ping: datetime
    subscriptions: Set[str]
    connection_type: str  # "dashboard", "trading", "monitoring"

    def __post_init__(self):
        self.subscriptions = set()


class WebSocketManager:
    """WebSocket接続管理クラス"""

    def __init__(self):
        # アクティブ接続
        self.active_connections: Dict[str, WebSocketConnection] = {}

        # ユーザー別接続マップ
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)

        # 購読グループ
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)

        # ハートビート設定
        self.heartbeat_interval = 30  # 30秒
        self.heartbeat_timeout = 90   # 90秒

        # メッセージキュー
        self.message_queue: Dict[str, List[Dict]] = defaultdict(list)

        # 統計情報
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
        }

        # サービス依存関係
        self.notification_service: Optional[NotificationService] = None
        self.market_service: Optional[MarketService] = None
        self.trading_service: Optional[TradingService] = None
        self.ml_service: Optional[MLService] = None

        logging.info("WebSocket Manager initialized")

    def set_services(
        self,
        notification_service: NotificationService,
        market_service: MarketService,
        trading_service: TradingService,
        ml_service: MLService
    ):
        """サービス依存関係を設定"""
        self.notification_service = notification_service
        self.market_service = market_service
        self.trading_service = trading_service
        self.ml_service = ml_service

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        connection_type: str = "dashboard"
    ) -> str:
        """新しいWebSocket接続を追加"""
        connection_id = str(uuid.uuid4())

        try:
            # 接続情報作成
            connection = WebSocketConnection(
                websocket=websocket,
                user_id=user_id,
                connection_id=connection_id,
                connected_at=datetime.utcnow(),
                last_ping=datetime.utcnow(),
                subscriptions=set(),
                connection_type=connection_type
            )

            # 接続を追加
            self.active_connections[connection_id] = connection
            self.user_connections[user_id].add(connection_id)

            # 統計更新
            self.stats["total_connections"] += 1
            self.stats["active_connections"] = len(self.active_connections)

            # メトリクス記録
            await track_websocket_event("connection_established", {
                "user_id": user_id,
                "connection_type": connection_type,
                "connection_id": connection_id
            })

            logging.info(f"WebSocket connected: {connection_id} (user: {user_id}, type: {connection_type})")

            # 遅延メッセージを送信
            await self._send_queued_messages(connection_id)

            return connection_id

        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
            raise

    async def disconnect(self, websocket: WebSocket) -> None:
        """WebSocket接続を切断"""
        connection_id = None

        # 該当する接続を検索
        for conn_id, conn in self.active_connections.items():
            if conn.websocket == websocket:
                connection_id = conn_id
                break

        if connection_id:
            await self._remove_connection(connection_id)

    async def _remove_connection(self, connection_id: str) -> None:
        """接続を削除"""
        if connection_id not in self.active_connections:
            return

        connection = self.active_connections[connection_id]
        user_id = connection.user_id

        try:
            # WebSocket接続をクローズ
            if connection.websocket.client_state == WebSocketState.CONNECTED:
                await connection.websocket.close()

        except Exception as e:
            logging.warning(f"Error closing WebSocket: {e}")

        # 購読を削除
        for subscription in connection.subscriptions:
            self.subscriptions[subscription].discard(connection_id)
            if not self.subscriptions[subscription]:
                del self.subscriptions[subscription]

        # 接続マップから削除
        self.user_connections[user_id].discard(connection_id)
        if not self.user_connections[user_id]:
            del self.user_connections[user_id]

        # アクティブ接続から削除
        del self.active_connections[connection_id]

        # 統計更新
        self.stats["active_connections"] = len(self.active_connections)

        # メトリクス記録
        await track_websocket_event("connection_closed", {
            "user_id": user_id,
            "connection_id": connection_id,
            "duration": (datetime.utcnow() - connection.connected_at).total_seconds()
        })

        logging.info(f"WebSocket disconnected: {connection_id} (user: {user_id})")

    async def send_personal_message(
        self,
        user_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """特定ユーザーにメッセージを送信"""
        if user_id not in self.user_connections:
            # オフラインユーザーのメッセージをキューに保存
            self.message_queue[user_id].append({
                **message,
                "queued_at": datetime.utcnow().isoformat()
            })
            return False

        success_count = 0
        connection_ids = list(self.user_connections[user_id])

        for connection_id in connection_ids:
            if await self._send_to_connection(connection_id, message):
                success_count += 1

        return success_count > 0

    async def send_to_connection(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """特定の接続にメッセージを送信"""
        return await self._send_to_connection(connection_id, message)

    async def _send_to_connection(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """接続にメッセージを送信（内部使用）"""
        if connection_id not in self.active_connections:
            return False

        connection = self.active_connections[connection_id]

        try:
            # メッセージにタイムスタンプを追加
            enhanced_message = {
                **message,
                "timestamp": datetime.utcnow().isoformat(),
                "connection_id": connection_id
            }

            await connection.websocket.send_json(enhanced_message)

            # 統計更新
            self.stats["messages_sent"] += 1

            # メトリクス記録
            await track_websocket_event("message_sent", {
                "user_id": connection.user_id,
                "connection_id": connection_id,
                "message_type": message.get("type", "unknown")
            })

            return True

        except WebSocketDisconnect:
            await self._remove_connection(connection_id)
            return False
        except Exception as e:
            logging.error(f"Error sending message to {connection_id}: {e}")
            self.stats["errors"] += 1
            return False

    async def broadcast_to_subscriptions(
        self,
        subscription: str,
        message: Dict[str, Any]
    ) -> int:
        """購読者全員にメッセージをブロードキャスト"""
        if subscription not in self.subscriptions:
            return 0

        connection_ids = list(self.subscriptions[subscription])
        success_count = 0

        for connection_id in connection_ids:
            if await self._send_to_connection(connection_id, message):
                success_count += 1

        return success_count

    async def subscribe(self, websocket: WebSocket, subscription: str) -> bool:
        """購読を追加"""
        connection_id = None

        # 接続IDを検索
        for conn_id, conn in self.active_connections.items():
            if conn.websocket == websocket:
                connection_id = conn_id
                break

        if not connection_id:
            return False

        connection = self.active_connections[connection_id]
        connection.subscriptions.add(subscription)
        self.subscriptions[subscription].add(connection_id)

        # メトリクス記録
        await track_websocket_event("subscription_added", {
            "user_id": connection.user_id,
            "connection_id": connection_id,
            "subscription": subscription
        })

        logging.info(f"Subscription added: {connection_id} -> {subscription}")
        return True

    async def unsubscribe(self, websocket: WebSocket, subscription: str) -> bool:
        """購読を削除"""
        connection_id = None

        # 接続IDを検索
        for conn_id, conn in self.active_connections.items():
            if conn.websocket == websocket:
                connection_id = conn_id
                break

        if not connection_id:
            return False

        connection = self.active_connections[connection_id]
        connection.subscriptions.discard(subscription)
        self.subscriptions[subscription].discard(connection_id)

        if not self.subscriptions[subscription]:
            del self.subscriptions[subscription]

        logging.info(f"Subscription removed: {connection_id} -> {subscription}")
        return True

    async def subscribe_market_data(
        self,
        websocket: WebSocket,
        symbols: List[str]
    ) -> bool:
        """市場データ購読"""
        success = True
        for symbol in symbols:
            subscription = f"market_data_{symbol}"
            if not await self.subscribe(websocket, subscription):
                success = False

        # 市場データサービスに購読を通知
        if self.market_service and success:
            await self.market_service.subscribe_symbols(symbols)

        return success

    async def ping_connection(self, connection_id: str) -> bool:
        """接続にpingを送信"""
        if connection_id not in self.active_connections:
            return False

        return await self._send_to_connection(connection_id, {
            "type": "ping",
            "timestamp": datetime.utcnow().isoformat()
        })

    async def handle_pong(self, websocket: WebSocket) -> bool:
        """pongメッセージを処理"""
        connection_id = None

        for conn_id, conn in self.active_connections.items():
            if conn.websocket == websocket:
                connection_id = conn_id
                break

        if connection_id:
            self.active_connections[connection_id].last_ping = datetime.utcnow()
            return True

        return False

    async def start_heartbeat(self) -> None:
        """ハートビート開始"""
        while True:
            try:
                await self._heartbeat_check()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logging.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)  # エラー時は短い間隔で再試行

    async def _heartbeat_check(self) -> None:
        """ハートビートチェック"""
        current_time = datetime.utcnow()
        timeout_threshold = current_time - timedelta(seconds=self.heartbeat_timeout)

        dead_connections = []

        for connection_id, connection in self.active_connections.items():
            if connection.last_ping < timeout_threshold:
                dead_connections.append(connection_id)
            else:
                # 生きている接続にpingを送信
                await self.ping_connection(connection_id)

        # タイムアウトした接続を削除
        for connection_id in dead_connections:
            logging.warning(f"Connection timeout: {connection_id}")
            await self._remove_connection(connection_id)

    async def _send_queued_messages(self, connection_id: str) -> None:
        """キューに溜まったメッセージを送信"""
        if connection_id not in self.active_connections:
            return

        connection = self.active_connections[connection_id]
        user_id = connection.user_id

        if user_id in self.message_queue:
            messages = self.message_queue[user_id]

            for message in messages:
                await self._send_to_connection(connection_id, message)

            # キューをクリア
            del self.message_queue[user_id]

            logging.info(f"Sent {len(messages)} queued messages to {connection_id}")

    async def disconnect_all(self) -> None:
        """全ての接続を切断"""
        connection_ids = list(self.active_connections.keys())

        for connection_id in connection_ids:
            await self._remove_connection(connection_id)

        logging.info("All WebSocket connections disconnected")

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return {
            **self.stats,
            "subscriptions": {
                subscription: len(connections)
                for subscription, connections in self.subscriptions.items()
            },
            "users_online": len(self.user_connections),
            "message_queue_size": sum(len(msgs) for msgs in self.message_queue.values())
        }

    def get_user_connections(self, user_id: str) -> List[str]:
        """ユーザーの接続IDリストを取得"""
        return list(self.user_connections.get(user_id, set()))

    def is_user_online(self, user_id: str) -> bool:
        """ユーザーがオンラインかチェック"""
        return user_id in self.user_connections

    async def send_trade_update(self, trade_data: Dict[str, Any]) -> None:
        """取引更新を送信"""
        message = {
            "type": "trade_update",
            "data": trade_data
        }

        # 取引データ購読者に送信
        await self.broadcast_to_subscriptions("trade_updates", message)

        # 関連ユーザーに個別送信
        if "user_id" in trade_data:
            await self.send_personal_message(trade_data["user_id"], message)

    async def send_market_update(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """市場データ更新を送信"""
        message = {
            "type": "market_update",
            "symbol": symbol,
            "data": market_data
        }

        subscription = f"market_data_{symbol}"
        await self.broadcast_to_subscriptions(subscription, message)

    async def send_alert(self, user_id: str, alert_data: Dict[str, Any]) -> None:
        """アラートを送信"""
        message = {
            "type": "alert",
            "data": alert_data
        }

        await self.send_personal_message(user_id, message)

    async def send_ml_update(self, model_data: Dict[str, Any]) -> None:
        """ML更新を送信"""
        message = {
            "type": "ml_update",
            "data": model_data
        }

        await self.broadcast_to_subscriptions("ml_updates", message)