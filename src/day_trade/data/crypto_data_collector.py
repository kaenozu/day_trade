#!/usr/bin/env python3
"""
Global Trading Engine - Cryptocurrency Data Collector
暗号通貨市場のリアルタイムデータ収集システム

主要取引所（Binance, Coinbase）からの高頻度データ取得
"""

import asyncio
import aiohttp
import websockets
import json
import time
import hmac
import hashlib
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger
from ..models.database import get_session, CryptoPrice, GlobalMarketData

logger = get_context_logger(__name__)

@dataclass
class CryptoTick:
    """暗号通貨取引データ"""
    symbol: str
    timestamp: datetime
    price: float
    volume_24h: float
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    spread: Optional[float] = None
    market_cap: Optional[float] = None
    price_change_24h: Optional[float] = None
    exchange: str = "binance"

@dataclass
class CryptoOrderBook:
    """オーダーブック情報"""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [price, quantity]
    asks: List[Tuple[float, float]]
    exchange: str

class CryptoDataCollector:
    """暗号通貨データ収集器"""

    def __init__(self, binance_api_key: Optional[str] = None,
                 binance_secret: Optional[str] = None):
        self.binance_api_key = binance_api_key
        self.binance_secret = binance_secret

        # 主要暗号通貨
        self.major_cryptos = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT",
            "DOTUSDT", "XRPUSDT", "LTCUSDT", "LINKUSDT",
            "BCHUSDT", "XLMUSDT", "UNIUSDT", "SOLUSDT"
        ]

        # DeFiトークン
        self.defi_tokens = [
            "AAVEUSDT", "COMPUSDT", "MKRUSDT", "SUSHIUSDT",
            "YFIUSDT", "CRVUSDT", "1INCHUSDT", "BALUSDT"
        ]

        # 全監視対象
        self.all_symbols = self.major_cryptos + self.defi_tokens

        # データ保存用
        self.latest_ticks: Dict[str, CryptoTick] = {}
        self.order_books: Dict[str, CryptoOrderBook] = {}
        self.websocket_connections: Dict[str, Any] = {}

        # API設定
        self.binance_base_url = "https://api.binance.com"
        self.binance_ws_url = "wss://stream.binance.com:9443"
        self.coinbase_base_url = "https://api.pro.coinbase.com"
        self.coinbase_ws_url = "wss://ws-feed.pro.coinbase.com"

        # レート制限
        self.request_weights = {"minute": 0}
        self.last_reset = time.time()
        self.weight_limit = 1200  # Binance制限

        logger.info(f"Crypto Data Collector initialized for {len(self.all_symbols)} symbols")

    async def start_collection(self):
        """データ収集開始"""
        logger.info("Starting crypto data collection...")

        # WebSocket接続開始
        ws_tasks = [
            self._connect_binance_websocket(),
            self._start_rest_api_updates()
        ]

        await asyncio.gather(*ws_tasks, return_exceptions=True)

    async def _connect_binance_websocket(self):
        """Binance WebSocket接続"""
        try:
            # マルチストリーム設定
            streams = []
            for symbol in self.major_cryptos[:8]:  # 最初の8銘柄
                streams.append(f"{symbol.lower()}@ticker")
                streams.append(f"{symbol.lower()}@depth5")

            stream_url = f"{self.binance_ws_url}/stream?streams=" + "/".join(streams)

            async with websockets.connect(stream_url) as websocket:
                logger.info(f"Connected to Binance WebSocket: {len(streams)} streams")

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._process_binance_message(data)

                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON received from Binance")
                    except Exception as e:
                        logger.error(f"WebSocket message processing error: {e}")

        except Exception as e:
            logger.error(f"Binance WebSocket connection error: {e}")
            await asyncio.sleep(5)
            # 再接続試行
            await self._connect_binance_websocket()

    async def _process_binance_message(self, data: dict):
        """Binance WebSocketメッセージ処理"""
        try:
            if 'stream' not in data:
                return

            stream = data['stream']
            message_data = data['data']

            # ティッカーデータ処理
            if '@ticker' in stream:
                await self._process_ticker_data(message_data)

            # 板データ処理
            elif '@depth' in stream:
                await self._process_depth_data(message_data)

        except Exception as e:
            logger.error(f"Binance message processing error: {e}")

    async def _process_ticker_data(self, data: dict):
        """ティッカーデータ処理"""
        try:
            symbol = data.get('s')
            if not symbol or symbol not in self.all_symbols:
                return

            # ティック作成
            tick = CryptoTick(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(
                    data.get('E', 0) / 1000, tz=timezone.utc
                ),
                price=float(data.get('c', 0)),
                volume_24h=float(data.get('v', 0)),
                bid_price=float(data.get('b', 0)),
                ask_price=float(data.get('a', 0)),
                price_change_24h=float(data.get('P', 0)),
                exchange="binance"
            )

            # スプレッド計算
            if tick.bid_price and tick.ask_price:
                tick.spread = tick.ask_price - tick.bid_price

            self.latest_ticks[symbol] = tick

            # データベース保存
            await self._save_crypto_tick(tick)

            logger.debug(f"Updated {symbol}: ${tick.price:.4f} "
                        f"(24h: {tick.price_change_24h:+.2f}%)")

        except Exception as e:
            logger.error(f"Ticker data processing error: {e}")

    async def _process_depth_data(self, data: dict):
        """板データ処理"""
        try:
            symbol = data.get('s')
            if not symbol:
                return

            # Bid/Ask変換
            bids = [(float(bid[0]), float(bid[1])) for bid in data.get('bids', [])]
            asks = [(float(ask[0]), float(ask[1])) for ask in data.get('asks', [])]

            order_book = CryptoOrderBook(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bids=bids,
                asks=asks,
                exchange="binance"
            )

            self.order_books[symbol] = order_book

        except Exception as e:
            logger.error(f"Depth data processing error: {e}")

    async def _start_rest_api_updates(self):
        """REST API定期更新"""
        while True:
            try:
                # 市場概況取得
                await self._fetch_market_overview()

                # DeFiトークン更新（WebSocketで取得していない銘柄）
                for symbol in self.defi_tokens:
                    if symbol not in self.latest_ticks:
                        await self._fetch_symbol_data(symbol)

                await asyncio.sleep(10)  # 10秒間隔

            except Exception as e:
                logger.error(f"REST API update error: {e}")
                await asyncio.sleep(30)

    async def _fetch_market_overview(self):
        """市場概況取得"""
        try:
            url = f"{self.binance_base_url}/api/v3/ticker/24hr"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()

                        # 監視対象のみ処理
                        for ticker in data:
                            symbol = ticker.get('symbol')
                            if symbol in self.all_symbols:
                                await self._process_ticker_data(ticker)

        except Exception as e:
            logger.error(f"Market overview fetch error: {e}")

    async def _fetch_symbol_data(self, symbol: str):
        """個別銘柄データ取得"""
        try:
            url = f"{self.binance_base_url}/api/v3/ticker/24hr"
            params = {'symbol': symbol}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        await self._process_ticker_data(data)

        except Exception as e:
            logger.debug(f"Symbol {symbol} data fetch error: {e}")

    async def _save_crypto_tick(self, tick: CryptoTick):
        """暗号通貨ティックデータをDB保存"""
        try:
            session = get_session()

            crypto_price = CryptoPrice(
                symbol=tick.symbol,
                timestamp=tick.timestamp,
                price=Decimal(str(tick.price)).quantize(
                    Decimal('0.00000001'), rounding=ROUND_HALF_UP
                ),
                volume_24h=Decimal(str(tick.volume_24h)) if tick.volume_24h else None,
                bid_price=Decimal(str(tick.bid_price)) if tick.bid_price else None,
                ask_price=Decimal(str(tick.ask_price)) if tick.ask_price else None,
                spread=Decimal(str(tick.spread)) if tick.spread else None,
                price_change_24h=tick.price_change_24h,
                exchange=tick.exchange
            )

            session.add(crypto_price)
            session.commit()
            session.close()

        except Exception as e:
            logger.error(f"Failed to save crypto tick to DB: {e}")

    async def get_market_data(self, symbol: str) -> Optional[CryptoTick]:
        """特定銘柄の最新データ取得"""
        return self.latest_ticks.get(symbol)

    def get_all_market_data(self) -> Dict[str, CryptoTick]:
        """全銘柄の最新データ取得"""
        return self.latest_ticks.copy()

    def get_order_book(self, symbol: str) -> Optional[CryptoOrderBook]:
        """オーダーブック取得"""
        return self.order_books.get(symbol)

    def calculate_market_metrics(self) -> Dict[str, Any]:
        """市場指標計算"""
        try:
            if not self.latest_ticks:
                return {}

            prices = [tick.price for tick in self.latest_ticks.values()]
            changes = [tick.price_change_24h for tick in self.latest_ticks.values()
                      if tick.price_change_24h is not None]
            volumes = [tick.volume_24h for tick in self.latest_ticks.values()
                      if tick.volume_24h is not None]

            metrics = {
                'total_symbols': len(self.latest_ticks),
                'average_price': np.mean(prices) if prices else 0,
                'total_volume_24h': sum(volumes) if volumes else 0,
                'average_change_24h': np.mean(changes) if changes else 0,
                'market_volatility': np.std(changes) if changes else 0,
                'bullish_symbols': len([c for c in changes if c > 0]),
                'bearish_symbols': len([c for c in changes if c < 0]),
                'timestamp': datetime.now(timezone.utc)
            }

            return metrics

        except Exception as e:
            logger.error(f"Market metrics calculation error: {e}")
            return {}

    async def get_historical_data(self, symbol: str, interval: str = "1h",
                                limit: int = 100) -> List[dict]:
        """過去データ取得"""
        try:
            url = f"{self.binance_base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()

                        # OHLCV形式に変換
                        ohlcv_data = []
                        for candle in data:
                            ohlcv_data.append({
                                'timestamp': datetime.fromtimestamp(
                                    candle[0] / 1000, tz=timezone.utc
                                ),
                                'open': float(candle[1]),
                                'high': float(candle[2]),
                                'low': float(candle[3]),
                                'close': float(candle[4]),
                                'volume': float(candle[5])
                            })

                        return ohlcv_data

            return []

        except Exception as e:
            logger.error(f"Historical data error for {symbol}: {e}")
            return []

    def get_top_gainers(self, limit: int = 10) -> List[CryptoTick]:
        """上昇率上位取得"""
        ticks_with_change = [
            tick for tick in self.latest_ticks.values()
            if tick.price_change_24h is not None
        ]

        return sorted(ticks_with_change,
                     key=lambda x: x.price_change_24h, reverse=True)[:limit]

    def get_top_losers(self, limit: int = 10) -> List[CryptoTick]:
        """下落率上位取得"""
        ticks_with_change = [
            tick for tick in self.latest_ticks.values()
            if tick.price_change_24h is not None
        ]

        return sorted(ticks_with_change,
                     key=lambda x: x.price_change_24h)[:limit]

    async def cleanup(self):
        """リソースクリーンアップ"""
        # WebSocket接続終了
        for connection in self.websocket_connections.values():
            try:
                await connection.close()
            except:
                pass

        logger.info("Crypto Data Collector cleanup completed")

def create_crypto_collector(binance_api_key: Optional[str] = None,
                          binance_secret: Optional[str] = None) -> CryptoDataCollector:
    """Crypto Data Collector ファクトリー関数"""
    return CryptoDataCollector(
        binance_api_key=binance_api_key,
        binance_secret=binance_secret
    )

async def main():
    """テスト実行"""
    collector = create_crypto_collector()

    print("Crypto Data Collector Test")
    print("=" * 40)

    # 短期間のデータ収集テスト
    collection_task = asyncio.create_task(collector.start_collection())

    # 15秒後に結果確認
    await asyncio.sleep(15)

    # 結果表示
    market_data = collector.get_all_market_data()
    print(f"\nCollected data for {len(market_data)} crypto symbols:")

    for symbol, tick in list(market_data.items())[:10]:  # 最初の10件
        print(f"{symbol}: ${tick.price:.4f} "
              f"(24h: {tick.price_change_24h:+.2f}% | Vol: ${tick.volume_24h:,.0f})")

    # 市場指標表示
    metrics = collector.calculate_market_metrics()
    if metrics:
        print(f"\nMarket Metrics:")
        print(f"  Total Volume 24h: ${metrics['total_volume_24h']:,.0f}")
        print(f"  Average Change 24h: {metrics['average_change_24h']:+.2f}%")
        print(f"  Market Volatility: {metrics['market_volatility']:.2f}")
        print(f"  Bullish/Bearish: {metrics['bullish_symbols']}/{metrics['bearish_symbols']}")

    # トップゲイナー・ルーザー
    gainers = collector.get_top_gainers(5)
    losers = collector.get_top_losers(5)

    if gainers:
        print(f"\nTop 5 Gainers:")
        for tick in gainers:
            print(f"  {tick.symbol}: {tick.price_change_24h:+.2f}%")

    if losers:
        print(f"\nTop 5 Losers:")
        for tick in losers:
            print(f"  {tick.symbol}: {tick.price_change_24h:+.2f}%")

    collection_task.cancel()
    await collector.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
