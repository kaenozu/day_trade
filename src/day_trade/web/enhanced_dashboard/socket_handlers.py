#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Socket.IO Event Handlers - SocketIOイベントハンドラー

WebSocketイベントの処理とリアルタイム通信の管理
"""

import asyncio
import logging
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask import request
from typing import Any, Dict


class SocketHandlers:
    """SocketIOイベントハンドラー管理"""
    
    def __init__(self, socketio: SocketIO, dashboard_instance):
        self.socketio = socketio
        self.dashboard = dashboard_instance
        self.logger = logging.getLogger(__name__)
        self._setup_socket_events()
    
    def _setup_socket_events(self):
        """SocketIOイベントの設定"""

        @self.socketio.on('connect')
        def handle_connect():
            """クライアント接続"""
            self.logger.info(f"クライアント接続: {request.sid}")
            emit('connected', {
                'status': 'success',
                'message': 'ダッシュボードに接続しました',
                'timestamp': self._get_current_timestamp()
            })

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """クライアント切断"""
            self.logger.info(f"クライアント切断: {request.sid}")

        @self.socketio.on('subscribe')
        async def handle_subscribe(data):
            """リアルタイムデータ購読"""
            symbol = data.get('symbol')
            if symbol:
                try:
                    await self.dashboard.real_time_manager.subscribe_symbol(symbol, self.socketio)
                    join_room(f'symbol_{symbol}')
                    emit('subscribed', {
                        'symbol': symbol,
                        'status': 'success',
                        'message': f'{symbol}の購読を開始しました'
                    })
                    self.logger.info(f"銘柄購読開始: {symbol} (session: {request.sid})")
                except Exception as e:
                    emit('subscribed', {
                        'symbol': symbol,
                        'status': 'error',
                        'message': f'購読エラー: {str(e)}'
                    })
                    self.logger.error(f"購読エラー {symbol}: {e}")

        @self.socketio.on('unsubscribe')
        async def handle_unsubscribe(data):
            """リアルタイムデータ購読停止"""
            symbol = data.get('symbol')
            if symbol:
                try:
                    await self.dashboard.real_time_manager.unsubscribe_symbol(symbol)
                    leave_room(f'symbol_{symbol}')
                    emit('unsubscribed', {
                        'symbol': symbol,
                        'status': 'success',
                        'message': f'{symbol}の購読を停止しました'
                    })
                    self.logger.info(f"銘柄購読停止: {symbol} (session: {request.sid})")
                except Exception as e:
                    emit('unsubscribed', {
                        'symbol': symbol,
                        'status': 'error',
                        'message': f'購読停止エラー: {str(e)}'
                    })
                    self.logger.error(f"購読停止エラー {symbol}: {e}")

        @self.socketio.on('request_analysis')
        async def handle_request_analysis(data):
            """分析リクエスト"""
            symbol = data.get('symbol')
            analysis_type = data.get('type', 'technical')

            if not symbol:
                emit('analysis_error', {'message': '銘柄が指定されていません'})
                return

            try:
                self.logger.info(f"分析リクエスト: {symbol} ({analysis_type})")
                analysis_result = await self._perform_analysis(symbol, analysis_type)
                emit('analysis_result', {
                    'symbol': symbol,
                    'type': analysis_type,
                    'result': analysis_result,
                    'timestamp': self._get_current_timestamp()
                })
            except Exception as e:
                emit('analysis_error', {
                    'symbol': symbol,
                    'type': analysis_type,
                    'error': str(e)
                })
                self.logger.error(f"分析エラー {symbol}: {e}")

        @self.socketio.on('request_chart')
        async def handle_request_chart(data):
            """チャートリクエスト"""
            symbol = data.get('symbol')
            chart_type = data.get('chart_type', 'candlestick')
            indicators = data.get('indicators', [])

            if not symbol:
                emit('chart_error', {'message': '銘柄が指定されていません'})
                return

            try:
                chart_data = self.dashboard._get_enhanced_chart_data(symbol, chart_type, indicators)
                emit('chart_data', {
                    'symbol': symbol,
                    'chart_type': chart_type,
                    'data': chart_data
                })
            except Exception as e:
                emit('chart_error', {
                    'symbol': symbol,
                    'error': str(e)
                })
                self.logger.error(f"チャート取得エラー {symbol}: {e}")

        @self.socketio.on('alert_test')
        def handle_alert_test(data):
            """アラートテスト"""
            alert_id = data.get('alert_id')
            if alert_id:
                emit('alert_triggered', {
                    'alert_id': alert_id,
                    'symbol': 'TEST',
                    'type': 'test',
                    'message': 'テストアラートです',
                    'severity': 'info',
                    'timestamp': self._get_current_timestamp()
                })

        @self.socketio.on('join_room')
        def handle_join_room(data):
            """ルーム参加"""
            room = data.get('room')
            if room:
                join_room(room)
                emit('room_joined', {'room': room, 'status': 'success'})
                self.logger.info(f"ルーム参加: {room} (session: {request.sid})")

        @self.socketio.on('leave_room')
        def handle_leave_room(data):
            """ルーム退出"""
            room = data.get('room')
            if room:
                leave_room(room)
                emit('room_left', {'room': room, 'status': 'success'})
                self.logger.info(f"ルーム退出: {room} (session: {request.sid})")

        @self.socketio.on('get_status')
        def handle_get_status():
            """ステータス取得"""
            status = {
                'active_connections': len(self.dashboard.real_time_manager.active_subscriptions),
                'server_time': self._get_current_timestamp(),
                'config': {
                    'theme': self.dashboard.config.theme.value,
                    'update_frequency': self.dashboard.config.update_frequency.value
                }
            }
            emit('status_update', status)

        @self.socketio.on('update_preferences')
        def handle_update_preferences(data):
            """設定更新"""
            user_id = data.get('user_id', 'default')
            preferences = data.get('preferences', {})
            
            try:
                result = self.dashboard._save_user_preferences(user_id, preferences)
                emit('preferences_updated', {
                    'user_id': user_id,
                    'status': 'success' if result.get('success') else 'error',
                    'message': result.get('message', '設定を更新しました')
                })
            except Exception as e:
                emit('preferences_updated', {
                    'user_id': user_id,
                    'status': 'error',
                    'message': f'設定更新エラー: {str(e)}'
                })

        @self.socketio.on('bulk_subscribe')
        async def handle_bulk_subscribe(data):
            """複数銘柄一括購読"""
            symbols = data.get('symbols', [])
            successful = []
            failed = []

            for symbol in symbols:
                try:
                    await self.dashboard.real_time_manager.subscribe_symbol(symbol, self.socketio)
                    join_room(f'symbol_{symbol}')
                    successful.append(symbol)
                except Exception as e:
                    failed.append({'symbol': symbol, 'error': str(e)})

            emit('bulk_subscribe_result', {
                'successful': successful,
                'failed': failed,
                'total': len(symbols)
            })

        self.logger.info("SocketIOイベントハンドラーの設定が完了しました")

    async def _perform_analysis(self, symbol: str, analysis_type: str) -> Dict[str, Any]:
        """分析実行（非同期）"""
        # この部分は実際のダッシュボードクラスの分析メソッドを呼び出す
        if hasattr(self.dashboard, '_perform_analysis'):
            return await self.dashboard._perform_analysis(symbol, analysis_type)
        else:
            # フォールバック
            return {
                'symbol': symbol,
                'type': analysis_type,
                'status': 'completed',
                'result': f'{symbol}の{analysis_type}分析結果',
                'timestamp': self._get_current_timestamp()
            }

    def _get_current_timestamp(self) -> str:
        """現在のタイムスタンプ取得"""
        from datetime import datetime
        return datetime.now().isoformat()

    def broadcast_price_update(self, data: Dict[str, Any]):
        """価格更新のブロードキャスト"""
        symbol = data.get('symbol')
        if symbol:
            self.socketio.emit('price_update', data, room=f'symbol_{symbol}')

    def broadcast_alert(self, alert: Dict[str, Any]):
        """アラートのブロードキャスト"""
        self.socketio.emit('alert_triggered', alert)

    def broadcast_system_message(self, message: str, message_type: str = 'info'):
        """システムメッセージのブロードキャスト"""
        self.socketio.emit('system_message', {
            'message': message,
            'type': message_type,
            'timestamp': self._get_current_timestamp()
        })

    def send_to_user(self, user_id: str, event: str, data: Dict[str, Any]):
        """特定ユーザーへのメッセージ送信"""
        # ユーザーIDとセッションIDのマッピングが必要
        # この実装は簡略化されています
        self.socketio.emit(event, data, room=f'user_{user_id}')

    def get_active_connections(self) -> int:
        """アクティブな接続数取得"""
        # 実際の実装では、接続数を追跡する必要があります
        return len(self.dashboard.real_time_manager.active_subscriptions)

    def disconnect_all(self):
        """全クライアント切断"""
        self.socketio.emit('server_shutdown', {
            'message': 'サーバーが停止されます',
            'timestamp': self._get_current_timestamp()
        })
        self.logger.info("全クライアントに切断通知を送信しました")