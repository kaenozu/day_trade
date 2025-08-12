#!/usr/bin/env python3
"""
Global Trading Engine - Forex Market Data Collector
外国為替市場のリアルタイムデータ収集システム

主要通貨ペアの高頻度データ取得・処理
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional

import aiohttp
import numpy as np

from ..models.database import ForexPrice, get_session

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class ForexTick:
    """Forex取引データ"""

    pair: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    spread: float
    volume: Optional[float] = None
    source: str = "forex_api"


@dataclass
class ForexOHLC:
    """Forex OHLC データ"""

    pair: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str  # 1m, 5m, 1h, etc.


class ForexDataCollector:
    """外国為替データ収集器"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "demo_key"  # デモ用キー

        # 主要通貨ペア
        self.major_pairs = [
            "EUR/USD",
            "GBP/USD",
            "USD/JPY",
            "USD/CHF",
            "AUD/USD",
            "USD/CAD",
            "NZD/USD",
        ]

        # マイナー・エキゾチック通貨ペア
        self.minor_pairs = [
            "EUR/GBP",
            "EUR/JPY",
            "GBP/JPY",
            "AUD/JPY",
            "EUR/AUD",
            "GBP/AUD",
            "CHF/JPY",
            "CAD/JPY",
        ]

        # 全監視対象
        self.all_pairs = self.major_pairs + self.minor_pairs

        # データ保存用
        self.latest_ticks: Dict[str, ForexTick] = {}
        self.historical_data: Dict[str, List[ForexOHLC]] = {}

        # API設定
        self.base_urls = {
            "alpha_vantage": "https://www.alphavantage.co/query",
            "fixer": "http://data.fixer.io/api",
            "forex_api": "https://api.forex.com/v1",
            "yahoo": "https://query1.finance.yahoo.com/v8/finance/chart",
        }

        # レート制限管理
        self.request_counts = {"minute": 0, "daily": 0}
        self.last_reset = {"minute": time.time(), "daily": time.time()}
        self.rate_limits = {"minute": 5, "daily": 500}  # デモ制限

        logger.info(f"Forex Data Collector initialized for {len(self.all_pairs)} pairs")

    async def start_collection(self, update_interval: float = 1.0):
        """データ収集開始"""
        logger.info(f"Starting forex data collection (interval: {update_interval}s)")

        while True:
            try:
                # 並列データ収集
                tasks = []

                # メジャーペアを優先的に収集
                for pair in self.major_pairs:
                    tasks.append(self._collect_pair_data(pair))

                # タスク実行
                await asyncio.gather(*tasks, return_exceptions=True)

                # 統計更新
                await self._update_market_statistics()

                # 次の更新まで待機
                await asyncio.sleep(update_interval)

            except Exception as e:
                logger.error(f"Data collection cycle error: {e}")
                await asyncio.sleep(5)  # エラー時は少し長く待機

    async def _collect_pair_data(self, pair: str):
        """特定通貨ペアのデータ収集"""
        try:
            # レート制限チェック
            if not self._check_rate_limit():
                logger.warning("Rate limit reached, skipping collection")
                return

            # Yahoo Finance APIを使用（無料・制限緩い）
            yahoo_symbol = self._convert_to_yahoo_symbol(pair)
            tick_data = await self._fetch_yahoo_forex(yahoo_symbol, pair)

            if tick_data:
                self.latest_ticks[pair] = tick_data

                # データベースに保存
                await self._save_tick_to_db(tick_data)

                logger.debug(f"Updated {pair}: {tick_data.bid_price:.5f}/{tick_data.ask_price:.5f}")

        except Exception as e:
            logger.warning(f"Failed to collect {pair} data: {e}")

    async def _fetch_yahoo_forex(self, symbol: str, pair: str) -> Optional[ForexTick]:
        """Yahoo Finance からForexデータ取得"""
        try:
            url = f"{self.base_urls['yahoo']}/{symbol}=X"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()

                        # レスポンス解析
                        result = data.get("chart", {}).get("result", [])
                        if not result:
                            return None

                        chart_data = result[0]
                        meta = chart_data.get("meta", {})

                        # 最新価格取得
                        current_price = meta.get("regularMarketPrice", 0)
                        if current_price == 0:
                            return None

                        # スプレッド推定（通常は0.1-0.3%程度）
                        estimated_spread = current_price * 0.0015  # 0.15%

                        # Bid/Ask価格計算
                        bid_price = current_price - (estimated_spread / 2)
                        ask_price = current_price + (estimated_spread / 2)

                        return ForexTick(
                            pair=pair,
                            timestamp=datetime.now(timezone.utc),
                            bid_price=bid_price,
                            ask_price=ask_price,
                            spread=estimated_spread,
                            volume=meta.get("regularMarketVolume"),
                            source="yahoo_finance",
                        )

        except Exception as e:
            logger.debug(f"Yahoo forex fetch error for {symbol}: {e}")
            return None

    def _convert_to_yahoo_symbol(self, pair: str) -> str:
        """通貨ペアをYahoo Finance形式に変換"""
        # EUR/USD -> EURUSD
        return pair.replace("/", "")

    def _check_rate_limit(self) -> bool:
        """レート制限チェック"""
        current_time = time.time()

        # 分単位リセット
        if current_time - self.last_reset["minute"] >= 60:
            self.request_counts["minute"] = 0
            self.last_reset["minute"] = current_time

        # 日単位リセット
        if current_time - self.last_reset["daily"] >= 86400:
            self.request_counts["daily"] = 0
            self.last_reset["daily"] = current_time

        # 制限チェック
        if (
            self.request_counts["minute"] >= self.rate_limits["minute"]
            or self.request_counts["daily"] >= self.rate_limits["daily"]
        ):
            return False

        # カウント増加
        self.request_counts["minute"] += 1
        self.request_counts["daily"] += 1

        return True

    async def _save_tick_to_db(self, tick: ForexTick):
        """ティックデータをデータベースに保存"""
        try:
            session = get_session()

            forex_price = ForexPrice(
                pair=tick.pair,
                timestamp=tick.timestamp,
                bid_price=Decimal(str(tick.bid_price)).quantize(
                    Decimal("0.00001"), rounding=ROUND_HALF_UP
                ),
                ask_price=Decimal(str(tick.ask_price)).quantize(
                    Decimal("0.00001"), rounding=ROUND_HALF_UP
                ),
                spread=Decimal(str(tick.spread)).quantize(
                    Decimal("0.00001"), rounding=ROUND_HALF_UP
                ),
                volume=tick.volume,
                source=tick.source,
            )

            session.add(forex_price)
            session.commit()
            session.close()

        except Exception as e:
            logger.error(f"Failed to save forex tick to DB: {e}")

    async def _update_market_statistics(self):
        """市場統計更新"""
        try:
            if not self.latest_ticks:
                return

            # 全体統計計算
            total_volume = 0
            price_changes = []
            active_pairs = len(self.latest_ticks)

            for pair, tick in self.latest_ticks.items():
                if tick.volume:
                    total_volume += tick.volume

                # 価格変動率計算（仮）
                price_changes.append(0.001)  # プレースホルダー

            # 統計データ
            market_stats = {
                "active_pairs": active_pairs,
                "total_volume": total_volume,
                "average_spread": np.mean([t.spread for t in self.latest_ticks.values()]),
                "market_volatility": np.std(price_changes) if price_changes else 0,
                "update_timestamp": datetime.now(timezone.utc),
            }

            logger.debug(
                f"Market stats: {active_pairs} pairs, avg spread: {market_stats['average_spread']:.5f}"
            )

        except Exception as e:
            logger.error(f"Market statistics update error: {e}")

    async def get_historical_data(
        self, pair: str, timeframe: str = "1h", periods: int = 100
    ) -> List[ForexOHLC]:
        """過去データ取得"""
        try:
            # Yahoo Finance履歴データ
            symbol = self._convert_to_yahoo_symbol(pair)

            # 期間計算
            end_date = datetime.now()
            if timeframe == "1m":
                start_date = end_date - timedelta(minutes=periods)
            elif timeframe == "5m":
                start_date = end_date - timedelta(minutes=periods * 5)
            elif timeframe == "1h":
                start_date = end_date - timedelta(hours=periods)
            elif timeframe == "1d":
                start_date = end_date - timedelta(days=periods)
            else:
                start_date = end_date - timedelta(hours=periods)

            # API呼び出し
            url = f"{self.base_urls['yahoo']}/{symbol}=X"
            params = {
                "period1": int(start_date.timestamp()),
                "period2": int(end_date.timestamp()),
                "interval": "1h",
                "includePrePost": "false",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_yahoo_ohlc(data, pair, timeframe)

            return []

        except Exception as e:
            logger.error(f"Historical data fetch error for {pair}: {e}")
            return []

    def _parse_yahoo_ohlc(self, data: dict, pair: str, timeframe: str) -> List[ForexOHLC]:
        """Yahoo FinanceデータをOHLC形式に変換"""
        try:
            result = data.get("chart", {}).get("result", [])
            if not result:
                return []

            chart_data = result[0]
            timestamps = chart_data.get("timestamp", [])
            indicators = chart_data.get("indicators", {})
            quote = indicators.get("quote", [{}])[0]

            ohlc_data = []
            for i, ts in enumerate(timestamps):
                try:
                    ohlc = ForexOHLC(
                        pair=pair,
                        timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
                        open=quote.get("open", [])[i] or 0,
                        high=quote.get("high", [])[i] or 0,
                        low=quote.get("low", [])[i] or 0,
                        close=quote.get("close", [])[i] or 0,
                        volume=quote.get("volume", [])[i] or 0,
                        timeframe=timeframe,
                    )

                    # データ有効性チェック
                    if all([ohlc.open, ohlc.high, ohlc.low, ohlc.close]):
                        ohlc_data.append(ohlc)

                except (IndexError, TypeError):
                    continue

            return ohlc_data[-100:]  # 最新100件

        except Exception as e:
            logger.error(f"OHLC parsing error: {e}")
            return []

    def get_latest_tick(self, pair: str) -> Optional[ForexTick]:
        """最新ティックデータ取得"""
        return self.latest_ticks.get(pair)

    def get_all_latest_ticks(self) -> Dict[str, ForexTick]:
        """全通貨ペアの最新データ取得"""
        return self.latest_ticks.copy()

    def calculate_cross_rates(self) -> Dict[str, float]:
        """クロスレート計算"""
        cross_rates = {}

        try:
            # USD/JPYベースでクロスレート計算
            usd_jpy = self.latest_ticks.get("USD/JPY")
            eur_usd = self.latest_ticks.get("EUR/USD")
            gbp_usd = self.latest_ticks.get("GBP/USD")

            if usd_jpy and eur_usd:
                # EUR/JPY = EUR/USD × USD/JPY
                eur_jpy_rate = eur_usd.bid_price * usd_jpy.bid_price
                cross_rates["EUR/JPY_calculated"] = eur_jpy_rate

            if usd_jpy and gbp_usd:
                # GBP/JPY = GBP/USD × USD/JPY
                gbp_jpy_rate = gbp_usd.bid_price * usd_jpy.bid_price
                cross_rates["GBP/JPY_calculated"] = gbp_jpy_rate

            return cross_rates

        except Exception as e:
            logger.error(f"Cross rate calculation error: {e}")
            return {}

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("Forex Data Collector cleanup completed")


def create_forex_collector(api_key: Optional[str] = None) -> ForexDataCollector:
    """Forex Data Collector ファクトリー関数"""
    return ForexDataCollector(api_key=api_key)


async def main():
    """テスト実行"""
    collector = create_forex_collector()

    print("Forex Data Collector Test")
    print("=" * 40)

    # 短期間のデータ収集テスト
    collection_task = asyncio.create_task(collector.start_collection(2.0))

    # 10秒後に停止
    await asyncio.sleep(10)
    collection_task.cancel()

    # 結果表示
    latest_data = collector.get_all_latest_ticks()
    print(f"\nCollected data for {len(latest_data)} pairs:")

    for pair, tick in latest_data.items():
        print(f"{pair}: {tick.bid_price:.5f}/{tick.ask_price:.5f} (spread: {tick.spread:.5f})")

    # クロスレート表示
    cross_rates = collector.calculate_cross_rates()
    if cross_rates:
        print("\nCalculated cross rates:")
        for pair, rate in cross_rates.items():
            print(f"{pair}: {rate:.5f}")

    await collector.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
