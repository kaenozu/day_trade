#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider V2 - 実用レベルデータプロバイダー

Yahoo Finance修正版 + 複数ソース冗長化による実データ取得システム
Phase5-A #901実装：実際のデイトレード運用向けデータ基盤
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from collections import defaultdict
import aiohttp
import sqlite3

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

class DataSource(Enum):
    """データソース種別"""
    YAHOO_FINANCE = "Yahoo Finance"
    STOOQ = "Stooq"
    ALPHA_VANTAGE = "Alpha Vantage"
    MATSUI_SECURITIES = "松井証券"
    GMO_CLICK = "GMOクリック"

class DataQuality(Enum):
    """データ品質"""
    EXCELLENT = "優秀"     # 遅延5分以内、欠損率1%以下
    GOOD = "良好"         # 遅延20分以内、欠損率5%以下
    FAIR = "普通"         # 遅延60分以内、欠損率10%以下
    POOR = "低品質"       # それ以下

@dataclass
class DataSourceInfo:
    """データソース情報"""
    source: DataSource
    is_available: bool
    delay_minutes: int          # 遅延時間（分）
    daily_limit: int           # 日次リクエスト上限
    cost_per_request: float    # リクエスト単価
    quality: DataQuality
    last_success: Optional[datetime] = None
    success_rate: float = 0.0
    avg_response_time: float = 0.0

@dataclass
class StockDataPoint:
    """株価データポイント"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    source: DataSource
    delay_minutes: int
    quality_score: float       # 0-100の品質スコア

@dataclass
class MarketData:
    """市場データ"""
    symbol: str
    name: str
    current_price: float
    price_change: float
    price_change_pct: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    sector: Optional[str]
    data_source: DataSource
    last_updated: datetime
    quality_score: float

class YahooFinanceProviderV2:
    """Yahoo Finance改良版プロバイダー"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
        self.daily_limit = 1000
        self.last_request_time = 0
        self.min_request_interval = 0.6  # 1分間100リクエスト制限対応

    async def get_stock_data(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """株価データ取得（改良版）"""

        if not YFINANCE_AVAILABLE:
            return None

        # レート制限対応
        await self._wait_for_rate_limit()

        try:
            # 複数の銘柄コード形式を試行
            symbol_variations = self._generate_symbol_variations(symbol)

            for ticker_symbol in symbol_variations:
                try:
                    self.logger.debug(f"Trying symbol: {ticker_symbol}")

                    ticker = yf.Ticker(ticker_symbol)
                    data = ticker.history(period=period)

                    if not data.empty and len(data) > 0:
                        # データ品質チェック
                        if self._validate_data_quality(data):
                            self.request_count += 1
                            self.last_request_time = time.time()

                            self.logger.info(f"Successfully fetched data for {symbol} as {ticker_symbol}")
                            return data

                except Exception as e:
                    self.logger.debug(f"Failed to fetch {ticker_symbol}: {e}")
                    continue

            self.logger.warning(f"All symbol variations failed for {symbol}")
            return None

        except Exception as e:
            self.logger.error(f"Yahoo Finance fetch error for {symbol}: {e}")
            return None

    def _generate_symbol_variations(self, symbol: str) -> List[str]:
        """日本株銘柄コードのバリエーション生成"""
        variations = []

        # 基本銘柄コード（数字のみ）
        if symbol.isdigit():
            variations.extend([
                f"{symbol}.T",      # 東京証券取引所
                f"{symbol}.JP",     # 日本
                symbol,             # そのまま
                f"{symbol}.TO",     # 東京
                f"{symbol}.TYO"     # Tokyo
            ])
        else:
            # すでにサフィックス付き
            variations.append(symbol)

            # サフィックス除去版も試行
            if '.' in symbol:
                base_symbol = symbol.split('.')[0]
                variations.extend([
                    f"{base_symbol}.T",
                    f"{base_symbol}.JP",
                    base_symbol
                ])

        return variations

    def _validate_data_quality(self, data: pd.DataFrame) -> bool:
        """データ品質チェック"""

        if data.empty:
            return False

        # 最低データ量チェック
        if len(data) < 5:
            return False

        # 価格データ整合性チェック
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return False

        # 異常値チェック（価格が0以下、異常に高い値等）
        for col in ['Open', 'High', 'Low', 'Close']:
            if (data[col] <= 0).any():
                return False
            if (data[col] > 1000000).any():  # 100万円超は異常値として除外
                return False

        # OHLC関係の整合性チェック
        if ((data['High'] < data['Low']) |
            (data['High'] < data['Open']) |
            (data['High'] < data['Close']) |
            (data['Low'] > data['Open']) |
            (data['Low'] > data['Close'])).any():
            return False

        return True

    async def _wait_for_rate_limit(self):
        """レート制限対応の待機"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            await asyncio.sleep(wait_time)

class StooqProvider:
    """Stooqデータプロバイダー"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://stooq.com/q/d/l/"

    async def get_stock_data(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """Stooqから株価データ取得"""

        try:
            # Stooq用の銘柄コード変換
            stooq_symbol = self._convert_to_stooq_symbol(symbol)

            # 期間変換
            days = self._period_to_days(period)

            # URL構築
            url = f"{self.base_url}?s={stooq_symbol}&d1={days}&i=d"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        text = await response.text()

                        # CSVデータをDataFrameに変換
                        from io import StringIO
                        df = pd.read_csv(StringIO(text))

                        if not df.empty and 'Close' in df.columns:
                            # Yahoo Finance形式に変換
                            df = self._convert_to_standard_format(df)
                            return df

        except Exception as e:
            self.logger.error(f"Stooq fetch error for {symbol}: {e}")

        return None

    def _convert_to_stooq_symbol(self, symbol: str) -> str:
        """Stooq用銘柄コード変換"""
        if symbol.isdigit():
            return f"{symbol}.jp"
        elif symbol.endswith('.T'):
            return symbol.replace('.T', '.jp')
        return symbol

    def _period_to_days(self, period: str) -> int:
        """期間文字列を日数に変換"""
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730
        }
        return period_map.get(period, 30)

    def _convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """標準フォーマットに変換"""
        # Stooq列名をYahoo Finance形式に変換
        column_map = {
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }

        df = df.rename(columns=column_map)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

        return df

class MultiSourceDataProvider:
    """複数ソース対応データプロバイダー"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データソース初期化
        self.providers = {
            DataSource.YAHOO_FINANCE: YahooFinanceProviderV2(),
            DataSource.STOOQ: StooqProvider(),
        }

        # データソース情報
        self.source_info = {
            DataSource.YAHOO_FINANCE: DataSourceInfo(
                source=DataSource.YAHOO_FINANCE,
                is_available=YFINANCE_AVAILABLE,
                delay_minutes=20,
                daily_limit=1000,
                cost_per_request=0.0,
                quality=DataQuality.GOOD
            ),
            DataSource.STOOQ: DataSourceInfo(
                source=DataSource.STOOQ,
                is_available=True,
                delay_minutes=15,
                daily_limit=10000,
                cost_per_request=0.0,
                quality=DataQuality.FAIR
            )
        }

        # キャッシュシステム
        self.cache_dir = Path("data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_expiry = 300  # 5分間キャッシュ

        # 統計情報
        self.request_stats = defaultdict(int)
        self.success_stats = defaultdict(int)

    async def get_stock_data(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """複数ソースからの株価データ取得"""

        # キャッシュチェック
        cached_data = self._get_from_cache(symbol, period)
        if cached_data is not None:
            return cached_data

        # 優先順位順でデータ取得試行
        priority_sources = [
            DataSource.YAHOO_FINANCE,
            DataSource.STOOQ
        ]

        for source in priority_sources:
            if not self.source_info[source].is_available:
                continue

            try:
                self.logger.debug(f"Trying {source.value} for {symbol}")
                self.request_stats[source] += 1

                provider = self.providers[source]
                data = await provider.get_stock_data(symbol, period)

                if data is not None and not data.empty:
                    self.success_stats[source] += 1
                    self.source_info[source].last_success = datetime.now()

                    # 成功率更新
                    success_rate = self.success_stats[source] / self.request_stats[source] * 100
                    self.source_info[source].success_rate = success_rate

                    # キャッシュ保存
                    self._save_to_cache(symbol, period, data, source)

                    self.logger.info(f"Successfully fetched {symbol} from {source.value}")
                    return data

            except Exception as e:
                self.logger.error(f"Error fetching from {source.value}: {e}")
                continue

        self.logger.warning(f"All data sources failed for {symbol}")
        return None

    async def get_multiple_stocks_data(self, symbols: List[str], period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """複数銘柄の並列データ取得"""

        results = {}

        # セマフォで同時リクエスト数制限
        semaphore = asyncio.Semaphore(5)  # 最大5並列

        async def fetch_single(symbol: str):
            async with semaphore:
                data = await self.get_stock_data(symbol, period)
                if data is not None:
                    results[symbol] = data
                await asyncio.sleep(0.1)  # レート制限対策

        # 全銘柄を並列取得
        tasks = [fetch_single(symbol) for symbol in symbols]
        await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def _get_from_cache(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """キャッシュからデータ取得"""

        cache_key = f"{symbol}_{period}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        if cache_file.exists():
            # キャッシュ有効期限チェック
            file_age = time.time() - cache_file.stat().st_mtime

            if file_age < self.cache_expiry:
                try:
                    return pd.read_parquet(cache_file)
                except Exception as e:
                    self.logger.error(f"Cache read error: {e}")
                    cache_file.unlink(missing_ok=True)

        return None

    def _save_to_cache(self, symbol: str, period: str, data: pd.DataFrame, source: DataSource):
        """キャッシュにデータ保存"""

        try:
            cache_key = f"{symbol}_{period}"
            cache_file = self.cache_dir / f"{cache_key}.parquet"

            # メタデータ追加
            data_copy = data.copy()
            data_copy.attrs = {
                'symbol': symbol,
                'source': source.value,
                'cached_at': datetime.now().isoformat()
            }

            data_copy.to_parquet(cache_file)

        except Exception as e:
            self.logger.error(f"Cache save error: {e}")

    def get_source_statistics(self) -> Dict[str, Any]:
        """データソース統計取得"""

        stats = {}

        for source, info in self.source_info.items():
            stats[source.value] = {
                'available': info.is_available,
                'requests': self.request_stats[source],
                'successes': self.success_stats[source],
                'success_rate': info.success_rate,
                'last_success': info.last_success.isoformat() if info.last_success else None,
                'delay_minutes': info.delay_minutes,
                'quality': info.quality.value
            }

        return stats

    def cleanup_cache(self, max_age_hours: int = 24):
        """古いキャッシュファイルの削除"""

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        deleted_count = 0

        for cache_file in self.cache_dir.glob("*.parquet"):
            file_age = current_time - cache_file.stat().st_mtime

            if file_age > max_age_seconds:
                cache_file.unlink()
                deleted_count += 1

        self.logger.info(f"Cleaned up {deleted_count} cache files")

# グローバルインスタンス
real_data_provider = MultiSourceDataProvider()

# 既存システムとの互換性関数
async def get_real_stock_data(symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
    """既存システム互換関数"""
    return await real_data_provider.get_stock_data(symbol, period)

# テスト関数
async def test_real_data_provider():
    """実データプロバイダーのテスト"""

    print("=== Real Data Provider V2 テスト ===")

    provider = MultiSourceDataProvider()

    # テスト銘柄
    test_symbols = ["7203", "4751", "6861", "8306", "9984"]

    print(f"\n[ {len(test_symbols)}銘柄のデータ取得テスト ]")

    start_time = time.time()

    # 並列取得テスト
    results = await provider.get_multiple_stocks_data(test_symbols, period="1mo")

    end_time = time.time()

    print(f"\n取得結果:")
    print(f"  成功: {len(results)}/{len(test_symbols)} 銘柄")
    print(f"  処理時間: {end_time - start_time:.2f}秒")

    for symbol, data in results.items():
        if data is not None:
            latest_price = data['Close'].iloc[-1] if len(data) > 0 else 0
            data_days = len(data)
            print(f"  {symbol}: {data_days}日分, 最新価格 ¥{latest_price:.2f}")
        else:
            print(f"  {symbol}: データ取得失敗")

    # データソース統計
    print(f"\n[ データソース統計 ]")
    stats = provider.get_source_statistics()
    for source_name, stat in stats.items():
        if stat['requests'] > 0:
            print(f"  {source_name}: {stat['successes']}/{stat['requests']} "
                  f"(成功率{stat['success_rate']:.1f}%) 遅延{stat['delay_minutes']}分")

    # 単一銘柄詳細テスト
    print(f"\n[ 詳細データテスト: 7203 トヨタ自動車 ]")
    toyota_data = await provider.get_stock_data("7203", "3mo")

    if toyota_data is not None and not toyota_data.empty:
        print(f"  データ期間: {len(toyota_data)}日分")
        print(f"  価格レンジ: ¥{toyota_data['Low'].min():.2f} - ¥{toyota_data['High'].max():.2f}")
        print(f"  平均出来高: {toyota_data['Volume'].mean():,.0f}株")
        print(f"  最新価格: ¥{toyota_data['Close'].iloc[-1]:.2f}")

        # データ品質チェック
        missing_data = toyota_data.isnull().sum().sum()
        data_quality = "優秀" if missing_data == 0 else f"欠損{missing_data}件"
        print(f"  データ品質: {data_quality}")

    else:
        print("  データ取得に失敗しました")

    print(f"\n=== Real Data Provider V2 テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_real_data_provider())