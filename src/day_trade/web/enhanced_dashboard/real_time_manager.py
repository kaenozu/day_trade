#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Data Manager - リアルタイムデータ管理

リアルタイムデータの購読、配信、更新を管理
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Set
from flask_socketio import SocketIO

from .config import DashboardConfig, UpdateFrequency

# 既存システムとの統合
try:
    from real_data_provider_v2 import MultiSourceDataProvider
    DATA_PROVIDER_AVAILABLE = True
except ImportError:
    DATA_PROVIDER_AVAILABLE = False


class RealTimeDataManager:
    """リアルタイムデータ管理"""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_subscriptions: Set[str] = set()
        self.data_cache: Dict[str, Any] = {}
        self.last_update: Dict[str, datetime] = {}

        # データプロバイダー初期化
        if DATA_PROVIDER_AVAILABLE:
            self.data_provider = MultiSourceDataProvider()
        else:
            self.data_provider = None

    async def subscribe_symbol(self, symbol: str, socketio: SocketIO):
        """銘柄データの購読開始"""
        self.active_subscriptions.add(symbol)
        self.logger.info(f"リアルタイム購読開始: {symbol}")

        # 初回データ送信
        await self._send_initial_data(symbol, socketio)

    async def unsubscribe_symbol(self, symbol: str):
        """銘柄データの購読停止"""
        self.active_subscriptions.discard(symbol)
        self.logger.info(f"リアルタイム購読停止: {symbol}")

    async def _send_initial_data(self, symbol: str, socketio: SocketIO):
        """初回データの送信"""
        try:
            if self.data_provider:
                data = await self.data_provider.get_stock_data(symbol, "1d")
                if data is not None and not data.empty:
                    latest_data = {
                        'symbol': symbol,
                        'price': float(data['Close'].iloc[-1]),
                        'change': float(data['Close'].iloc[-1] - data['Close'].iloc[-2]) if len(data) > 1 else 0.0,
                        'volume': int(data['Volume'].iloc[-1]),
                        'timestamp': datetime.now().isoformat()
                    }
                    socketio.emit('price_update', latest_data)

        except Exception as e:
            self.logger.error(f"初回データ送信エラー {symbol}: {e}")

    async def update_data_loop(self, socketio: SocketIO):
        """データ更新ループ"""
        while True:
            try:
                for symbol in self.active_subscriptions.copy():
                    await self._update_symbol_data(symbol, socketio)

                # 更新頻度に応じた待機
                interval = self._get_update_interval()
                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"データ更新ループエラー: {e}")
                await asyncio.sleep(5)

    async def _update_symbol_data(self, symbol: str, socketio: SocketIO):
        """個別銘柄データの更新"""
        try:
            # レート制限チェック
            last_update = self.last_update.get(symbol, datetime.min)
            min_interval = timedelta(seconds=self._get_update_interval())

            if datetime.now() - last_update < min_interval:
                return

            if self.data_provider:
                data = await self.data_provider.get_stock_data(symbol, "1d")
                if data is not None and not data.empty:
                    latest_data = {
                        'symbol': symbol,
                        'price': float(data['Close'].iloc[-1]),
                        'change': float(data['Close'].iloc[-1] - data['Close'].iloc[-2]) if len(data) > 1 else 0.0,
                        'change_percent': float(((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)) if len(data) > 1 else 0.0,
                        'volume': int(data['Volume'].iloc[-1]),
                        'high': float(data['High'].iloc[-1]),
                        'low': float(data['Low'].iloc[-1]),
                        'timestamp': datetime.now().isoformat()
                    }

                    # キャッシュ更新
                    self.data_cache[symbol] = latest_data
                    self.last_update[symbol] = datetime.now()

                    # クライアントに送信
                    socketio.emit('price_update', latest_data)

        except Exception as e:
            self.logger.error(f"銘柄データ更新エラー {symbol}: {e}")

    def _get_update_interval(self) -> float:
        """更新間隔の取得"""
        intervals = {
            UpdateFrequency.REAL_TIME: 1.0,
            UpdateFrequency.HIGH: 5.0,
            UpdateFrequency.MEDIUM: 30.0,
            UpdateFrequency.LOW: 300.0,
            UpdateFrequency.MANUAL: 3600.0
        }
        return intervals.get(self.config.update_frequency, 30.0)

    def get_cached_data(self, symbol: str) -> Dict[str, Any]:
        """キャッシュされたデータの取得"""
        return self.data_cache.get(symbol, {})

    def clear_cache(self):
        """キャッシュのクリア"""
        self.data_cache.clear()
        self.logger.info("データキャッシュをクリアしました")