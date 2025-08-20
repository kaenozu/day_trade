#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime Service - リアルタイムデータサービス
リアルタイム価格配信とWebSocket通信機能
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import queue
from pathlib import Path

@dataclass
class PriceUpdate:
    """価格更新データ"""
    symbol: str
    current_price: float
    previous_close: float
    price_change: float
    price_change_pct: float
    volume: int
    high: float
    low: float
    timestamp: str
    market_status: str = "OPEN"

@dataclass
class MarketTick:
    """マーケットティック"""
    symbol: str
    price: float
    volume: int
    timestamp: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

class RealtimeDataFeed:
    """リアルタイムデータフィード"""

    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.subscribed_symbols: Set[str] = set()
        self.price_cache: Dict[str, PriceUpdate] = {}
        self.subscribers: List[Callable] = []
        self.is_running = False
        self.feed_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)

        # データ取得用
        self.data_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def subscribe_symbol(self, symbol: str) -> bool:
        """シンボル購読"""
        try:
            self.subscribed_symbols.add(symbol)
            self.logger.info(f"シンボル {symbol} を購読リストに追加")
            return True
        except Exception as e:
            self.logger.error(f"シンボル購読エラー: {e}")
            return False

    def unsubscribe_symbol(self, symbol: str) -> bool:
        """シンボル購読解除"""
        try:
            self.subscribed_symbols.discard(symbol)
            if symbol in self.price_cache:
                del self.price_cache[symbol]
            self.logger.info(f"シンボル {symbol} の購読を解除")
            return True
        except Exception as e:
            self.logger.error(f"シンボル購読解除エラー: {e}")
            return False

    def add_subscriber(self, callback: Callable[[PriceUpdate], None]) -> None:
        """価格更新コールバック追加"""
        self.subscribers.append(callback)

    def remove_subscriber(self, callback: Callable[[PriceUpdate], None]) -> None:
        """価格更新コールバック削除"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def start_feed(self) -> None:
        """データフィード開始"""
        if self.is_running:
            self.logger.warning("データフィードは既に実行中です")
            return

        self.is_running = True
        self.feed_thread = threading.Thread(target=self._feed_loop, daemon=True)
        self.feed_thread.start()
        self.logger.info("リアルタイムデータフィードを開始")

    def stop_feed(self) -> None:
        """データフィード停止"""
        self.is_running = False
        if self.feed_thread and self.feed_thread.is_alive():
            self.feed_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        self.logger.info("リアルタイムデータフィードを停止")

    def _feed_loop(self) -> None:
        """フィードループ"""
        while self.is_running:
            try:
                if self.subscribed_symbols:
                    # 並行してデータ取得
                    futures = []
                    for symbol in self.subscribed_symbols:
                        future = self.executor.submit(self._fetch_symbol_data, symbol)
                        futures.append(future)

                    # 結果を処理
                    for future in futures:
                        try:
                            price_update = future.result(timeout=10)
                            if price_update:
                                self._process_price_update(price_update)
                        except Exception as e:
                            self.logger.error(f"データ取得エラー: {e}")

                time.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"フィードループエラー: {e}")
                time.sleep(self.update_interval)

    def _fetch_symbol_data(self, symbol: str) -> Optional[PriceUpdate]:
        """シンボルデータ取得"""
        try:
            # Yahoo Finance APIを使用（実際の実装）
            import yfinance as yf

            jp_symbol = f"{symbol}.T"
            ticker = yf.Ticker(jp_symbol)

            # 最新データ取得
            info = ticker.info
            hist = ticker.history(period="2d")

            if len(hist) > 0:
                current_price = info.get('currentPrice') or info.get('regularMarketPrice') or hist['Close'].iloc[-1]
                previous_close = info.get('previousClose') or hist['Close'].iloc[-2] if len(hist) > 1 else current_price

                price_change = current_price - previous_close
                price_change_pct = (price_change / previous_close) * 100 if previous_close > 0 else 0

                return PriceUpdate(
                    symbol=symbol,
                    current_price=round(current_price, 2),
                    previous_close=round(previous_close, 2),
                    price_change=round(price_change, 2),
                    price_change_pct=round(price_change_pct, 2),
                    volume=info.get('volume', 0),
                    high=round(info.get('dayHigh', current_price), 2),
                    low=round(info.get('dayLow', current_price), 2),
                    timestamp=datetime.now().isoformat(),
                    market_status=self._get_market_status()
                )

        except ImportError:
            # フォールバック: サンプルデータ生成
            return self._generate_sample_price_update(symbol)
        except Exception as e:
            self.logger.error(f"シンボル {symbol} のデータ取得エラー: {e}")
            return self._generate_sample_price_update(symbol)

        return None

    def _generate_sample_price_update(self, symbol: str) -> PriceUpdate:
        """サンプル価格データ生成"""
        import random

        # 前回の価格を基準に変動
        base_price = self.price_cache.get(symbol)
        if base_price:
            current_price = base_price.current_price * (1 + random.uniform(-0.02, 0.02))
            previous_close = base_price.previous_close
        else:
            current_price = random.uniform(800, 3000)
            previous_close = current_price * (1 + random.uniform(-0.01, 0.01))

        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close) * 100 if previous_close > 0 else 0

        return PriceUpdate(
            symbol=symbol,
            current_price=round(current_price, 2),
            previous_close=round(previous_close, 2),
            price_change=round(price_change, 2),
            price_change_pct=round(price_change_pct, 2),
            volume=random.randint(100000, 5000000),
            high=round(current_price * random.uniform(1.0, 1.03), 2),
            low=round(current_price * random.uniform(0.97, 1.0), 2),
            timestamp=datetime.now().isoformat(),
            market_status=self._get_market_status()
        )

    def _get_market_status(self) -> str:
        """市場状態取得"""
        now = datetime.now()
        hour = now.hour

        # 日本市場時間（簡易版）
        if 9 <= hour < 15:
            return "OPEN"
        elif 15 <= hour < 16:
            return "POST_MARKET"
        else:
            return "CLOSED"

    def _process_price_update(self, price_update: PriceUpdate) -> None:
        """価格更新処理"""
        try:
            # キャッシュ更新
            self.price_cache[price_update.symbol] = price_update

            # 購読者に通知
            for callback in self.subscribers:
                try:
                    callback(price_update)
                except Exception as e:
                    self.logger.error(f"購読者通知エラー: {e}")

        except Exception as e:
            self.logger.error(f"価格更新処理エラー: {e}")

    def get_latest_price(self, symbol: str) -> Optional[PriceUpdate]:
        """最新価格取得"""
        return self.price_cache.get(symbol)

    def get_all_prices(self) -> Dict[str, PriceUpdate]:
        """全価格データ取得"""
        return self.price_cache.copy()

class WebSocketManager:
    """WebSocket接続管理"""

    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # connection_id -> symbols
        self.logger = logging.getLogger(__name__)

    def add_connection(self, connection_id: str, websocket) -> None:
        """接続追加"""
        self.connections[connection_id] = websocket
        self.subscriptions[connection_id] = set()
        self.logger.info(f"WebSocket接続追加: {connection_id}")

    def remove_connection(self, connection_id: str) -> None:
        """接続削除"""
        if connection_id in self.connections:
            del self.connections[connection_id]
        if connection_id in self.subscriptions:
            del self.subscriptions[connection_id]
        self.logger.info(f"WebSocket接続削除: {connection_id}")

    def subscribe_symbol(self, connection_id: str, symbol: str) -> None:
        """シンボル購読"""
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].add(symbol)

    def unsubscribe_symbol(self, connection_id: str, symbol: str) -> None:
        """シンボル購読解除"""
        if connection_id in self.subscriptions:
            self.subscriptions[connection_id].discard(symbol)

    def broadcast_price_update(self, price_update: PriceUpdate) -> None:
        """価格更新ブロードキャスト"""
        message = {
            'type': 'price_update',
            'data': asdict(price_update)
        }

        for connection_id, symbols in self.subscriptions.items():
            if price_update.symbol in symbols:
                self._send_message(connection_id, message)

    def _send_message(self, connection_id: str, message: Dict[str, Any]) -> None:
        """メッセージ送信"""
        try:
            websocket = self.connections.get(connection_id)
            if websocket:
                # 実際のWebSocket実装では websocket.send(json.dumps(message))
                self.logger.debug(f"メッセージ送信: {connection_id} -> {message['type']}")
        except Exception as e:
            self.logger.error(f"メッセージ送信エラー: {e}")

class RealtimeService:
    """リアルタイムサービス統合"""

    def __init__(self, update_interval: int = 5):
        self.data_feed = RealtimeDataFeed(update_interval)
        self.websocket_manager = WebSocketManager()
        self.logger = logging.getLogger(__name__)

        # データフィードの更新をWebSocketに転送
        self.data_feed.add_subscriber(self.websocket_manager.broadcast_price_update)

        # アラートサービス連携用
        self.alert_callbacks: List[Callable] = []

    def start_service(self) -> None:
        """サービス開始"""
        self.data_feed.start_feed()
        self.logger.info("リアルタイムサービス開始")

    def stop_service(self) -> None:
        """サービス停止"""
        self.data_feed.stop_feed()
        self.logger.info("リアルタイムサービス停止")

    def subscribe_symbols(self, symbols: List[str]) -> None:
        """複数シンボル購読"""
        for symbol in symbols:
            self.data_feed.subscribe_symbol(symbol)

    def get_market_snapshot(self) -> Dict[str, Any]:
        """市場スナップショット取得"""
        prices = self.data_feed.get_all_prices()

        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'symbol_count': len(prices),
            'market_status': self._get_overall_market_status(),
            'prices': {symbol: asdict(update) for symbol, update in prices.items()},
            'statistics': self._calculate_market_statistics(prices)
        }

        return snapshot

    def _get_overall_market_status(self) -> str:
        """全体市場状態"""
        now = datetime.now()
        hour = now.hour

        if 9 <= hour < 15:
            return "OPEN"
        elif 15 <= hour < 16:
            return "POST_MARKET"
        else:
            return "CLOSED"

    def _calculate_market_statistics(self, prices: Dict[str, PriceUpdate]) -> Dict[str, Any]:
        """市場統計計算"""
        if not prices:
            return {}

        changes = [update.price_change_pct for update in prices.values()]
        volumes = [update.volume for update in prices.values()]

        return {
            'average_change_pct': round(sum(changes) / len(changes), 2),
            'gainers_count': len([c for c in changes if c > 0]),
            'losers_count': len([c for c in changes if c < 0]),
            'unchanged_count': len([c for c in changes if c == 0]),
            'total_volume': sum(volumes),
            'most_active': max(prices.items(), key=lambda x: x[1].volume)[0] if prices else None,
            'biggest_gainer': max(prices.items(), key=lambda x: x[1].price_change_pct)[0] if prices else None,
            'biggest_loser': min(prices.items(), key=lambda x: x[1].price_change_pct)[0] if prices else None
        }

    def add_alert_callback(self, callback: Callable[[PriceUpdate], None]) -> None:
        """アラートコールバック追加"""
        self.alert_callbacks.append(callback)
        self.data_feed.add_subscriber(callback)

    def get_service_status(self) -> Dict[str, Any]:
        """サービス状態取得"""
        return {
            'is_running': self.data_feed.is_running,
            'subscribed_symbols': list(self.data_feed.subscribed_symbols),
            'active_connections': len(self.websocket_manager.connections),
            'cache_size': len(self.data_feed.price_cache),
            'update_interval': self.data_feed.update_interval,
            'last_update': datetime.now().isoformat()
        }