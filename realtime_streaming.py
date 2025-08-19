#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Streaming Engine - リアルタイムデータ配信システム
Issue #935対応: WebSocket + SSE + リアルタイム分析統合
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import logging

# WebSocket サポート
try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    websockets = None

# Flask-SSE サポート
try:
    from flask import Flask, Response, request, jsonify
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# 統合モジュール
try:
    from advanced_ai_engine import advanced_ai_engine, MarketSignal
    HAS_AI_ENGINE = True
except ImportError:
    HAS_AI_ENGINE = False

try:
    from performance_monitor import performance_monitor
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False

try:
    from data_persistence import data_persistence
    HAS_DATA_PERSISTENCE = True
except ImportError:
    HAS_DATA_PERSISTENCE = False


@dataclass
class StreamingMessage:
    """ストリーミングメッセージ"""
    message_type: str
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime
    client_id: Optional[str] = None


@dataclass
class ClientSubscription:
    """クライアント購読情報"""
    client_id: str
    symbols: Set[str]
    message_types: Set[str]
    last_seen: datetime
    websocket: Optional[Any] = None


class MarketDataSimulator:
    """市場データシミュレーター"""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.symbols = ['7203', '8306', '9984', '6758', '4689', '9434', '8001', '7267']
        self.prices = {}
        self.volumes = {}
        self.running = False
        self._task = None

        # 初期価格設定
        for symbol in self.symbols:
            base_price = 1500 + hash(symbol) % 1500
            self.prices[symbol] = base_price
            self.volumes[symbol] = abs(hash(symbol + 'vol')) % 2000000 + 500000

    async def start(self):
        """シミュレーター開始"""
        if self.running:
            return

        self.running = True
        self._task = asyncio.create_task(self._simulation_loop())

    async def stop(self):
        """シミュレーター停止"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _simulation_loop(self):
        """シミュレーションループ"""
        import random

        while self.running:
            try:
                # 市場時間チェック（9:00-15:00の間は活発、それ以外は低活動）
                current_hour = datetime.now().hour
                is_market_hours = 9 <= current_hour <= 15
                volatility_multiplier = 1.0 if is_market_hours else 0.3

                updated_symbols = []

                for symbol in self.symbols:
                    # 価格更新（ランダムウォーク）
                    change_rate = random.normalvariate(0, 0.005 * volatility_multiplier)
                    self.prices[symbol] *= (1 + change_rate)

                    # 出来高更新
                    volume_change = random.normalvariate(1.0, 0.2 * volatility_multiplier)
                    base_volume = abs(hash(symbol + 'vol')) % 1000000 + 500000
                    self.volumes[symbol] = int(base_volume * volume_change)

                    # 10%の確率でこの銘柄を更新通知対象に
                    if random.random() < 0.1 or is_market_hours:
                        updated_symbols.append(symbol)

                # 更新されたシンボルの通知
                if updated_symbols:
                    await self._notify_updates(updated_symbols)

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Market simulation error: {e}")
                await asyncio.sleep(1.0)

    async def _notify_updates(self, symbols: List[str]):
        """更新通知"""
        # ここで外部の通知システムに連携
        pass

    def get_current_data(self, symbol: str) -> Dict[str, Any]:
        """現在のデータ取得"""
        return {
            'symbol': symbol,
            'price': self.prices.get(symbol, 0.0),
            'volume': self.volumes.get(symbol, 0),
            'timestamp': datetime.now().isoformat(),
            'change_percent': round(random.normalvariate(0, 1.5), 2)  # 模擬変動率
        }


class WebSocketStreaming:
    """WebSocketストリーミングサーバー"""

    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, ClientSubscription] = {}
        self.message_queue = asyncio.Queue()
        self.server = None
        self.running = False

        # レート制限
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
        self.max_messages_per_minute = 60

    async def start(self):
        """WebSocketサーバー開始"""
        if not HAS_WEBSOCKETS:
            raise RuntimeError("websockets library not available")

        self.running = True
        self.server = await serve(self.handle_client, self.host, self.port)
        logging.info(f"WebSocket server started on {self.host}:{self.port}")

        # メッセージ配信タスク開始
        asyncio.create_task(self.message_distributor())

    async def stop(self):
        """WebSocketサーバー停止"""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """クライアント接続処理"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}:{int(time.time())}"

        try:
            # 初期購読設定
            subscription = ClientSubscription(
                client_id=client_id,
                symbols=set(),
                message_types={'market_data', 'analysis'},
                last_seen=datetime.now(),
                websocket=websocket
            )

            self.clients[client_id] = subscription

            # 歓迎メッセージ
            await websocket.send(json.dumps({
                'type': 'connection_established',
                'client_id': client_id,
                'server_time': datetime.now().isoformat(),
                'supported_message_types': ['market_data', 'analysis', 'subscribe', 'unsubscribe'],
                'available_symbols': ['7203', '8306', '9984', '6758', '4689', '9434', '8001', '7267']
            }))

            # メッセージ受信ループ
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_client_message(client_id, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': f'Message processing error: {str(e)}'
                    }))

        except websockets.exceptions.ConnectionClosed:
            logging.info(f"Client {client_id} disconnected")
        except Exception as e:
            logging.error(f"WebSocket error for {client_id}: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]

    async def handle_client_message(self, client_id: str, data: Dict[str, Any]):
        """クライアントメッセージ処理"""
        client = self.clients.get(client_id)
        if not client:
            return

        message_type = data.get('type', '')

        # レート制限チェック
        if not self._check_rate_limit(client_id):
            await client.websocket.send(json.dumps({
                'type': 'rate_limit_exceeded',
                'message': f'Rate limit exceeded: {self.max_messages_per_minute} messages per minute'
            }))
            return

        client.last_seen = datetime.now()

        if message_type == 'subscribe':
            symbols = set(data.get('symbols', []))
            message_types = set(data.get('message_types', ['market_data']))

            client.symbols.update(symbols)
            client.message_types.update(message_types)

            await client.websocket.send(json.dumps({
                'type': 'subscription_updated',
                'subscribed_symbols': list(client.symbols),
                'subscribed_message_types': list(client.message_types)
            }))

        elif message_type == 'unsubscribe':
            symbols = set(data.get('symbols', []))
            client.symbols -= symbols

            await client.websocket.send(json.dumps({
                'type': 'subscription_updated',
                'subscribed_symbols': list(client.symbols),
                'subscribed_message_types': list(client.message_types)
            }))

        elif message_type == 'get_current_data':
            symbol = data.get('symbol', '')
            if symbol:
                # 現在のデータ送信
                current_data = self._get_current_market_data(symbol)
                await client.websocket.send(json.dumps({
                    'type': 'current_data_response',
                    'symbol': symbol,
                    'data': current_data
                }))

        elif message_type == 'ping':
            await client.websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': datetime.now().isoformat()
            }))

    def _check_rate_limit(self, client_id: str) -> bool:
        """レート制限チェック"""
        now = datetime.now()
        client_rates = self.rate_limits[client_id]

        # 1分以内のメッセージをカウント
        recent_messages = [
            timestamp for timestamp in client_rates
            if now - timestamp < timedelta(minutes=1)
        ]

        if len(recent_messages) >= self.max_messages_per_minute:
            return False

        # 現在のタイムスタンプを記録
        client_rates.append(now)
        return True

    def _get_current_market_data(self, symbol: str) -> Dict[str, Any]:
        """現在の市場データ取得"""
        # 実際の実装では市場データプロバイダーから取得
        # ここではダミーデータを返す
        import random
        base_price = 1500 + hash(symbol) % 1500

        return {
            'price': round(base_price * (1 + random.normalvariate(0, 0.02)), 2),
            'volume': random.randint(100000, 5000000),
            'change_percent': round(random.normalvariate(0, 1.5), 2),
            'bid': round(base_price * 0.999, 2),
            'ask': round(base_price * 1.001, 2),
            'timestamp': datetime.now().isoformat()
        }

    async def broadcast_message(self, message: StreamingMessage):
        """メッセージブロードキャスト"""
        await self.message_queue.put(message)

    async def message_distributor(self):
        """メッセージ配信処理"""
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)

                # 対象クライアントを選択
                target_clients = [
                    client for client in self.clients.values()
                    if (message.symbol in client.symbols or not message.symbol) and
                       message.message_type in client.message_types
                ]

                # 並行配信
                if target_clients:
                    await asyncio.gather(*[
                        self._send_to_client(client, message)
                        for client in target_clients
                    ], return_exceptions=True)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Message distribution error: {e}")

    async def _send_to_client(self, client: ClientSubscription, message: StreamingMessage):
        """個別クライアントへの送信"""
        try:
            data = {
                'type': message.message_type,
                'symbol': message.symbol,
                'data': message.data,
                'timestamp': message.timestamp.isoformat()
            }

            await client.websocket.send(json.dumps(data))

        except websockets.exceptions.ConnectionClosed:
            # 切断されたクライアントをクリーンアップ
            if client.client_id in self.clients:
                del self.clients[client.client_id]
        except Exception as e:
            logging.error(f"Failed to send message to client {client.client_id}: {e}")


class SSEStreaming:
    """Server-Sent Events ストリーミング"""

    def __init__(self):
        self.subscribers: Dict[str, Any] = {}
        self.message_queue = asyncio.Queue()

        if HAS_FLASK:
            self.app = Flask(__name__)
            CORS(self.app)
            self._setup_routes()

    def _setup_routes(self):
        """Flask ルート設定"""

        @self.app.route('/stream')
        def stream():
            """SSEストリーム"""
            def event_stream():
                client_id = f"sse_{int(time.time())}_{id(event_stream)}"

                try:
                    # 初期メッセージ
                    yield f"data: {json.dumps({'type': 'connection', 'client_id': client_id})}\n\n"

                    while True:
                        # メッセージチェック（ブロッキング回避のため短いタイムアウト）
                        try:
                            message = self._get_next_message(client_id, timeout=1.0)
                            if message:
                                yield f"data: {json.dumps(message)}\n\n"
                        except:
                            # キープアライブ
                            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"

                        time.sleep(0.1)  # 短い間隔でポーリング

                except GeneratorExit:
                    if client_id in self.subscribers:
                        del self.subscribers[client_id]

            return Response(event_stream(), mimetype='text/event-stream')

        @self.app.route('/api/stream/subscribe', methods=['POST'])
        def subscribe():
            """購読設定"""
            data = request.get_json()
            client_id = data.get('client_id', '')
            symbols = data.get('symbols', [])

            if client_id:
                self.subscribers[client_id] = {
                    'symbols': set(symbols),
                    'last_seen': datetime.now()
                }

                return jsonify({
                    'status': 'success',
                    'subscribed_symbols': symbols
                })

            return jsonify({'status': 'error', 'message': 'Invalid client_id'}), 400

    def _get_next_message(self, client_id: str, timeout: float = 1.0) -> Optional[Dict]:
        """次のメッセージを取得（ノンブロッキング）"""
        # 実際の実装では、メッセージキューから適切なメッセージを取得
        # ここでは簡易実装
        return None

    def publish_message(self, message: StreamingMessage):
        """メッセージ配信"""
        # SSE用のメッセージ配信実装
        pass


class RealTimeStreamingEngine:
    """リアルタイムストリーミング統合エンジン"""

    def __init__(self, websocket_port: int = 8765, sse_port: int = 8766):
        self.websocket_streaming = WebSocketStreaming(port=websocket_port) if HAS_WEBSOCKETS else None
        self.sse_streaming = SSEStreaming() if HAS_FLASK else None
        self.market_simulator = MarketDataSimulator()

        self.running = False
        self.analysis_interval = 5.0  # 5秒間隔で分析
        self.data_update_interval = 1.0  # 1秒間隔でデータ更新

        # 統計
        self.messages_sent = 0
        self.active_connections = 0
        self.error_count = 0

    async def start(self):
        """ストリーミングエンジン開始"""
        self.running = True

        tasks = []

        # WebSocketサーバー開始
        if self.websocket_streaming:
            tasks.append(self.websocket_streaming.start())

        # 市場シミュレーター開始
        tasks.append(self.market_simulator.start())

        # 分析タスク開始
        tasks.append(self._analysis_loop())
        tasks.append(self._data_update_loop())

        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        """ストリーミングエンジン停止"""
        self.running = False

        if self.websocket_streaming:
            await self.websocket_streaming.stop()

        await self.market_simulator.stop()

    async def _analysis_loop(self):
        """分析ループ"""
        while self.running:
            try:
                if HAS_AI_ENGINE:
                    # 全銘柄の分析を実行
                    symbols = ['7203', '8306', '9984', '6758', '4689']

                    for symbol in symbols:
                        # 市場データ更新
                        market_data = self.market_simulator.get_current_data(symbol)
                        advanced_ai_engine.update_market_data(
                            symbol=symbol,
                            price=market_data['price'],
                            volume=market_data['volume']
                        )

                        # AI分析実行
                        signal = advanced_ai_engine.analyze_symbol(symbol)

                        # WebSocket配信
                        if self.websocket_streaming:
                            message = StreamingMessage(
                                message_type='analysis',
                                symbol=symbol,
                                data={
                                    'signal_type': signal.signal_type,
                                    'confidence': signal.confidence,
                                    'strength': signal.strength,
                                    'risk_level': signal.risk_level,
                                    'reasons': signal.reasons,
                                    'indicators': signal.indicators
                                },
                                timestamp=datetime.now()
                            )

                            await self.websocket_streaming.broadcast_message(message)
                            self.messages_sent += 1

                await asyncio.sleep(self.analysis_interval)

            except Exception as e:
                logging.error(f"Analysis loop error: {e}")
                self.error_count += 1
                await asyncio.sleep(1.0)

    async def _data_update_loop(self):
        """データ更新ループ"""
        while self.running:
            try:
                # 市場データ配信
                symbols = ['7203', '8306', '9984', '6758', '4689']

                for symbol in symbols:
                    market_data = self.market_simulator.get_current_data(symbol)

                    # WebSocket配信
                    if self.websocket_streaming:
                        message = StreamingMessage(
                            message_type='market_data',
                            symbol=symbol,
                            data=market_data,
                            timestamp=datetime.now()
                        )

                        await self.websocket_streaming.broadcast_message(message)
                        self.messages_sent += 1

                await asyncio.sleep(self.data_update_interval)

            except Exception as e:
                logging.error(f"Data update loop error: {e}")
                self.error_count += 1
                await asyncio.sleep(1.0)

    def get_streaming_statistics(self) -> Dict[str, Any]:
        """ストリーミング統計取得"""
        active_connections = len(self.websocket_streaming.clients) if self.websocket_streaming else 0

        return {
            'total_messages_sent': self.messages_sent,
            'active_websocket_connections': active_connections,
            'error_count': self.error_count,
            'websocket_available': HAS_WEBSOCKETS and self.websocket_streaming is not None,
            'sse_available': HAS_FLASK and self.sse_streaming is not None,
            'ai_engine_available': HAS_AI_ENGINE,
            'performance_monitoring': HAS_PERFORMANCE_MONITOR,
            'data_persistence': HAS_DATA_PERSISTENCE,
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
        }


# グローバルインスタンス
streaming_engine = RealTimeStreamingEngine()


async def main():
    """メイン実行関数"""
    if not HAS_WEBSOCKETS:
        print("Warning: websockets library not available. WebSocket streaming disabled.")

    if not HAS_FLASK:
        print("Warning: Flask not available. SSE streaming disabled.")

    print("Starting Real-time Streaming Engine...")
    print("WebSocket Server: localhost:8765")
    print("Press Ctrl+C to stop")

    try:
        await streaming_engine.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        await streaming_engine.stop()


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 実行
    asyncio.run(main())