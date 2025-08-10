"""
リアルタイムデータストリーミングシステム

WebSocket経由でリアルタイムデータをクライアントに配信
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect

from ...utils.logging_config import get_context_logger
from .metrics_collector import MetricsCollector
from .feature_store_monitor import FeatureStoreMonitor

logger = get_context_logger(__name__)


class RealtimeStream:
    """リアルタイムデータストリーミングクラス"""

    def __init__(self, broadcast_interval: float = 2.0):
        """
        リアルタイムストリーム初期化

        Args:
            broadcast_interval: データブロードキャスト間隔（秒）
        """
        self.broadcast_interval = broadcast_interval
        self.active_connections: Set[WebSocket] = set()
        self.metrics_collector: Optional[MetricsCollector] = None
        self.feature_store_monitor: Optional[FeatureStoreMonitor] = None
        self.is_streaming = False
        self._streaming_task: Optional[asyncio.Task] = None

    def set_metrics_collector(self, collector: MetricsCollector) -> None:
        """メトリクス収集器を設定"""
        self.metrics_collector = collector
        logger.info("メトリクス収集器を設定しました")

    def set_feature_store_monitor(self, monitor: FeatureStoreMonitor) -> None:
        """Feature Store監視器を設定"""
        self.feature_store_monitor = monitor
        logger.info("Feature Store監視器を設定しました")

    async def connect(self, websocket: WebSocket) -> None:
        """新しいWebSocket接続を受け入れ"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"新しいクライアント接続: {len(self.active_connections)}台接続中")

        # 接続時の初期データ送信
        await self._send_initial_data(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """WebSocket接続を切断"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"クライアント切断: {len(self.active_connections)}台接続中")

    async def _send_initial_data(self, websocket: WebSocket) -> None:
        """接続時の初期データ送信"""
        try:
            # システム情報送信
            if self.metrics_collector:
                system_info = self.metrics_collector.get_system_info()
                await self._send_to_connection(websocket, {
                    'type': 'system_info',
                    'data': system_info,
                    'timestamp': datetime.now().isoformat()
                })

            # Feature Store情報送信（利用可能な場合）
            if self.feature_store_monitor:
                fs_status = self.feature_store_monitor.get_health_status()
                await self._send_to_connection(websocket, {
                    'type': 'feature_store_status',
                    'data': fs_status,
                    'timestamp': datetime.now().isoformat()
                })

        except Exception as e:
            logger.error(f"初期データ送信エラー: {e}")

    async def start_streaming(self) -> None:
        """リアルタイムストリーミング開始"""
        if self.is_streaming:
            logger.warning("ストリーミングは既に開始されています")
            return

        self.is_streaming = True
        self._streaming_task = asyncio.create_task(self._streaming_loop())
        logger.info(f"リアルタイムストリーミングを開始しました (間隔: {self.broadcast_interval}秒)")

    async def stop_streaming(self) -> None:
        """リアルタイムストリーミング停止"""
        if not self.is_streaming:
            return

        self.is_streaming = False
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass

        # 全接続に停止通知
        await self.broadcast({
            'type': 'streaming_stopped',
            'message': 'リアルタイムストリーミングを停止しました',
            'timestamp': datetime.now().isoformat()
        })

        logger.info("リアルタイムストリーミングを停止しました")

    async def _streaming_loop(self) -> None:
        """ストリーミングメインループ"""
        while self.is_streaming and self.active_connections:
            try:
                # データ収集と配信
                stream_data = await self._collect_streaming_data()
                if stream_data:
                    await self.broadcast(stream_data)

                await asyncio.sleep(self.broadcast_interval)

            except Exception as e:
                logger.error(f"ストリーミングループでエラーが発生: {e}")
                await asyncio.sleep(self.broadcast_interval)

    async def _collect_streaming_data(self) -> Optional[Dict]:
        """ストリーミング用データ収集"""
        data = {
            'type': 'realtime_update',
            'timestamp': datetime.now().isoformat(),
            'data': {}
        }

        # システムメトリクス
        if self.metrics_collector:
            try:
                current_metrics = self.metrics_collector.get_current_metrics()
                aggregated_metrics = self.metrics_collector.get_aggregated_metrics(1)  # 直近1分

                data['data']['system_metrics'] = {
                    'current': current_metrics,
                    'aggregated': aggregated_metrics
                }
            except Exception as e:
                logger.error(f"システムメトリクス収集エラー: {e}")

        # Feature Store メトリクス
        if self.feature_store_monitor:
            try:
                fs_metrics = self.feature_store_monitor.get_current_metrics()
                fs_summary = self.feature_store_monitor.get_performance_summary()

                data['data']['feature_store'] = {
                    'metrics': fs_metrics,
                    'performance_summary': fs_summary
                }
            except Exception as e:
                logger.error(f"Feature Store メトリクス収集エラー: {e}")

        # データが空でない場合のみ返す
        if data['data']:
            return data
        return None

    async def broadcast(self, message: Dict) -> None:
        """全クライアントにメッセージブロードキャスト"""
        if not self.active_connections:
            return

        disconnected = set()

        for connection in self.active_connections.copy():
            try:
                await self._send_to_connection(connection, message)
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception as e:
                logger.warning(f"ブロードキャストエラー: {e}")
                disconnected.add(connection)

        # 切断された接続を削除
        self.active_connections -= disconnected

    async def _send_to_connection(self, websocket: WebSocket, message: Dict) -> None:
        """個別接続にメッセージ送信"""
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"メッセージ送信失敗: {e}")
            raise

    async def send_alert(self, alert_type: str, message: str, data: Optional[Dict] = None) -> None:
        """アラートメッセージ送信"""
        alert_message = {
            'type': 'alert',
            'alert_type': alert_type,  # 'info', 'warning', 'error', 'critical'
            'message': message,
            'data': data or {},
            'timestamp': datetime.now().isoformat()
        }

        await self.broadcast(alert_message)
        logger.info(f"アラート送信: {alert_type} - {message}")

    async def send_notification(self, title: str, message: str, category: str = 'system') -> None:
        """通知メッセージ送信"""
        notification = {
            'type': 'notification',
            'title': title,
            'message': message,
            'category': category,  # 'system', 'performance', 'feature_store', 'user'
            'timestamp': datetime.now().isoformat()
        }

        await self.broadcast(notification)
        logger.info(f"通知送信: {title} - {message}")

    async def handle_client_message(self, websocket: WebSocket, message: Dict) -> None:
        """クライアントからのメッセージ処理"""
        try:
            message_type = message.get('type')

            if message_type == 'ping':
                # Ping-Pong応答
                await self._send_to_connection(websocket, {
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })

            elif message_type == 'request_data':
                # データリクエスト処理
                data_type = message.get('data_type')
                await self._handle_data_request(websocket, data_type)

            elif message_type == 'subscribe':
                # 購読設定
                await self._handle_subscription(websocket, message)

            elif message_type == 'unsubscribe':
                # 購読解除
                await self._handle_unsubscription(websocket, message)

            else:
                logger.warning(f"未知のメッセージタイプ: {message_type}")

        except Exception as e:
            logger.error(f"クライアントメッセージ処理エラー: {e}")
            await self._send_to_connection(websocket, {
                'type': 'error',
                'message': 'メッセージ処理中にエラーが発生しました',
                'timestamp': datetime.now().isoformat()
            })

    async def _handle_data_request(self, websocket: WebSocket, data_type: str) -> None:
        """データリクエスト処理"""
        try:
            if data_type == 'system_health':
                if self.metrics_collector:
                    health_report = self.metrics_collector.generate_health_report()
                    await self._send_to_connection(websocket, {
                        'type': 'data_response',
                        'data_type': 'system_health',
                        'data': health_report,
                        'timestamp': datetime.now().isoformat()
                    })

            elif data_type == 'feature_store_report':
                if self.feature_store_monitor:
                    fs_report = self.feature_store_monitor.generate_report()
                    await self._send_to_connection(websocket, {
                        'type': 'data_response',
                        'data_type': 'feature_store_report',
                        'data': fs_report,
                        'timestamp': datetime.now().isoformat()
                    })

            elif data_type == 'historical_metrics':
                if self.metrics_collector:
                    history = self.metrics_collector.get_metrics_history(30)  # 30分
                    await self._send_to_connection(websocket, {
                        'type': 'data_response',
                        'data_type': 'historical_metrics',
                        'data': history,
                        'timestamp': datetime.now().isoformat()
                    })

        except Exception as e:
            logger.error(f"データリクエスト処理エラー: {e}")

    async def _handle_subscription(self, websocket: WebSocket, message: Dict) -> None:
        """購読設定処理"""
        # 将来の拡張用: クライアント別購読設定
        channels = message.get('channels', [])
        logger.info(f"購読設定: {channels}")

        await self._send_to_connection(websocket, {
            'type': 'subscription_confirmed',
            'channels': channels,
            'timestamp': datetime.now().isoformat()
        })

    async def _handle_unsubscription(self, websocket: WebSocket, message: Dict) -> None:
        """購読解除処理"""
        # 将来の拡張用: クライアント別購読解除
        channels = message.get('channels', [])
        logger.info(f"購読解除: {channels}")

        await self._send_to_connection(websocket, {
            'type': 'unsubscription_confirmed',
            'channels': channels,
            'timestamp': datetime.now().isoformat()
        })

    def get_connection_stats(self) -> Dict:
        """接続統計取得"""
        return {
            'active_connections': len(self.active_connections),
            'streaming_active': self.is_streaming,
            'broadcast_interval': self.broadcast_interval,
            'metrics_collector_active': self.metrics_collector is not None,
            'feature_store_monitor_active': self.feature_store_monitor is not None
        }
