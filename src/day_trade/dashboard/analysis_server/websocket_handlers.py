"""
WebSocket接続管理とリアルタイム通信

分析結果の配信、リアルタイムログ、システム状態通知を行う
"""

import json
from datetime import datetime
from typing import Dict, List

from fastapi import WebSocket, WebSocketDisconnect

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# グローバル接続管理
connected_clients: Dict[str, WebSocket] = {}


class ConnectionManager:
    """WebSocket接続管理クラス"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """新しいWebSocket接続を受け入れる"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"新しいクライアント接続: {len(self.active_connections)}台接続中")

    def disconnect(self, websocket: WebSocket):
        """WebSocket接続を切断する"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"クライアント切断: {len(self.active_connections)}台接続中")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """特定のクライアントにメッセージを送信"""
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"メッセージ送信エラー: {e}")
            # 接続が切れている場合は削除
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """全てのクライアントにメッセージをブロードキャスト"""
        if not self.active_connections:
            return
            
        # 切断された接続を収集
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                logger.warning(f"ブロードキャスト送信エラー: {e}")
                disconnected.append(connection)

        # 切断された接続を削除
        for connection in disconnected:
            if connection in self.active_connections:
                self.active_connections.remove(connection)

    async def broadcast_analysis_update(self, analysis_data: dict):
        """分析結果の更新をブロードキャスト"""
        message = {
            "type": "analysis_update",
            "data": analysis_data,
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(message)

    async def broadcast_market_data(self, market_data: dict):
        """市場データ更新をブロードキャスト"""
        message = {
            "type": "market_data",
            "data": market_data,
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(message)

    async def broadcast_system_status(self, status: dict):
        """システム状態変更をブロードキャスト"""
        message = {
            "type": "system_status",
            "data": status,
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(message)

    async def broadcast_log_message(self, log_level: str, message: str):
        """ログメッセージをブロードキャスト"""
        log_message = {
            "type": "log",
            "level": log_level,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(log_message)

    def get_connection_count(self) -> int:
        """現在の接続数を取得"""
        return len(self.active_connections)

    def get_connection_info(self) -> dict:
        """接続情報を取得"""
        return {
            "total_connections": len(self.active_connections),
            "active_since": datetime.now().isoformat(),
        }


# グローバルマネージャーインスタンス
manager = ConnectionManager()


async def handle_websocket_connection(websocket: WebSocket):
    """WebSocket接続のメインハンドラー"""
    await manager.connect(websocket)

    try:
        # 接続時の初期メッセージ
        await manager.send_personal_message(
            {
                "type": "connection",
                "message": "分析専用ダッシュボードに接続しました",
                "safe_mode": True,
                "trading_disabled": True,
                "connection_time": datetime.now().isoformat(),
            },
            websocket,
        )

        # メッセージ受信ループ
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            await process_websocket_message(message_data, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket処理エラー: {e}")
        manager.disconnect(websocket)


async def process_websocket_message(message_data: dict, websocket: WebSocket):
    """WebSocketメッセージの処理"""
    message_type = message_data.get("type")

    if message_type == "ping":
        # Ping/Pongメッセージ処理
        await manager.send_personal_message(
            {
                "type": "pong",
                "message": "分析システム稼働中（セーフモード）",
                "timestamp": datetime.now().isoformat(),
            },
            websocket,
        )
    
    elif message_type == "subscribe":
        # 特定データの購読要求
        subscription_type = message_data.get("subscription")
        await handle_subscription_request(subscription_type, websocket)
    
    elif message_type == "request_status":
        # システム状態要求
        from .app_config import get_system_status
        status = get_system_status()
        status["server_time"] = datetime.now().isoformat()
        
        await manager.send_personal_message(
            {
                "type": "status_response",
                "data": status,
                "timestamp": datetime.now().isoformat(),
            },
            websocket,
        )
    
    else:
        logger.warning(f"未知のメッセージタイプ: {message_type}")


async def handle_subscription_request(subscription_type: str, websocket: WebSocket):
    """データ購読要求の処理"""
    valid_subscriptions = ["analysis", "market_data", "logs", "system_status"]
    
    if subscription_type in valid_subscriptions:
        await manager.send_personal_message(
            {
                "type": "subscription_confirmed",
                "subscription": subscription_type,
                "message": f"{subscription_type}データの配信を開始します",
                "timestamp": datetime.now().isoformat(),
            },
            websocket,
        )
    else:
        await manager.send_personal_message(
            {
                "type": "subscription_error",
                "message": f"無効な購読タイプ: {subscription_type}",
                "valid_types": valid_subscriptions,
                "timestamp": datetime.now().isoformat(),
            },
            websocket,
        )


def get_connection_manager() -> ConnectionManager:
    """ConnectionManagerインスタンスを取得"""
    return manager