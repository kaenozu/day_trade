#!/usr/bin/env python3
"""
Webダッシュボード WebSocketハンドラーモジュール

WebSocketイベント処理
"""

from flask import request
from flask_socketio import emit

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


def setup_websocket_events(socketio, dashboard_core):
    """WebSocketイベント設定"""

    @socketio.on("connect")
    def handle_connect():
        """クライアント接続"""
        logger.info(f"クライアント接続: {request.sid}")
        emit("status", {"message": "ダッシュボードに接続されました"})

    @socketio.on("disconnect")
    def handle_disconnect():
        """クライアント切断"""
        logger.info(f"クライアント切断: {request.sid}")

    @socketio.on("request_update")
    def handle_request_update():
        """手動更新要求"""
        try:
            status = dashboard_core.get_current_status()
            emit("dashboard_update", status)
        except Exception as e:
            logger.error(f"更新要求処理エラー: {e}")
            emit("error", {"message": str(e)})

    @socketio.on("subscribe_updates")
    def handle_subscribe_updates(data):
        """更新通知購読"""
        try:
            update_types = data.get("types", ["all"])
            logger.info(f"更新通知購読: {request.sid} -> {update_types}")
            
            # 購読確認
            emit("subscription_confirmed", {
                "types": update_types,
                "message": "更新通知を購読しました"
            })
            
        except Exception as e:
            logger.error(f"購読処理エラー: {e}")
            emit("error", {"message": "購読処理でエラーが発生しました"})

    @socketio.on("unsubscribe_updates") 
    def handle_unsubscribe_updates():
        """更新通知購読解除"""
        try:
            logger.info(f"更新通知購読解除: {request.sid}")
            emit("unsubscription_confirmed", {
                "message": "更新通知の購読を解除しました"
            })
        except Exception as e:
            logger.error(f"購読解除処理エラー: {e}")
            emit("error", {"message": "購読解除処理でエラーが発生しました"})

    @socketio.on("get_client_info")
    def handle_get_client_info():
        """クライアント情報取得"""
        try:
            client_info = {
                "session_id": request.sid,
                "remote_addr": request.environ.get("REMOTE_ADDR"),
                "user_agent": request.headers.get("User-Agent"),
                "connected_at": request.environ.get("wsgi.websocket_start_time")
            }
            emit("client_info", client_info)
        except Exception as e:
            logger.error(f"クライアント情報取得エラー: {e}")
            emit("error", {"message": "クライアント情報の取得に失敗しました"})

    return {
        "connect": handle_connect,
        "disconnect": handle_disconnect,
        "request_update": handle_request_update,
        "subscribe_updates": handle_subscribe_updates,
        "unsubscribe_updates": handle_unsubscribe_updates,
        "get_client_info": handle_get_client_info,
    }