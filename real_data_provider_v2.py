#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Real Data Provider V2 - 改善版実用レベルデータプロバイダー
Issue #853対応：マルチソースデータプロバイダーの強化

主要改善点:
1. データソースの管理と設定の外部化
2. Yahoo Finance API利用の堅牢性向上
3. キャッシュシステムの改善
4. エラーハンドリングとロギングの強化
5. 動的データソース管理
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import requests
import time
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict
import aiohttp
import sqlite3
import pickle
import hashlib
from abc import ABC, abstractmethod

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

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class DataSource(Enum):
    """データソース種別"""
    YAHOO_FINANCE = "yahoo_finance"
    STOOQ = "stooq"
    ALPHA_VANTAGE = "alpha_vantage"
    MOCK = "mock"


class DataQualityLevel(Enum):
    """データ品質レベル"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FAILED = "failed"


@dataclass
class DataSourceConfig:
    """データソース設定"""
    name: str
    enabled: bool = True
    priority: int = 1
    timeout: int = 30
    retry_count: int = 3
    rate_limit_per_minute: int = 60
    rate_limit_per_day: int = 1000
    quality_threshold: float = 70.0

    # API固有設定
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

    # 品質設定
    min_data_points: int = 5
    max_price_threshold: float = 1000000
    price_consistency_check: bool = True

    # キャッシュ設定
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300


@dataclass
class DataFetchResult:
    """データ取得結果"""
    data: Optional[pd.DataFrame]
    source: DataSource
    quality_level: DataQualityLevel
    quality_score: float
    fetch_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False


class DataSourceConfigManager:
    """データソース設定管理"""

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or Path("config/data_sources.yaml")
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict[str, DataSourceConfig]:
        """設定ファイル読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)

                configs = {}
                for name, data in config_data.get('data_sources', {}).items():
                    configs[name] = DataSourceConfig(name=name, **data)

                self.logger.info(f"Loaded {len(configs)} data source configurations")
                return configs
            else:
                self.logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return self._get_default_configs()

        except Exception as e:
            self.logger.error(f"Failed to load data source configs: {e}")
            return self._get_default_configs()

    def _get_default_configs(self) -> Dict[str, DataSourceConfig]:
        """デフォルト設定"""
        return {
            'yahoo_finance': DataSourceConfig(
                name='yahoo_finance',
                enabled=True,
                priority=1,
                timeout=30,
                rate_limit_per_minute=60,
                rate_limit_per_day=1000,
                quality_threshold=80.0,
                max_price_threshold=1000000
            ),
            'stooq': DataSourceConfig(
                name='stooq',
                enabled=True,
                priority=2,
                timeout=20,
                base_url="https://stooq.com/q/d/l/",
                rate_limit_per_minute=30,
                quality_threshold=70.0
            ),
            'mock': DataSourceConfig(
                name='mock',
                enabled=True,
                priority=99,
                timeout=1,
                quality_threshold=50.0
            )
        }

    def get_config(self, source_name: str) -> Optional[DataSourceConfig]:
        """設定取得"""
        return self.configs.get(source_name)

    def is_enabled(self, source_name: str) -> bool:
        """有効性確認"""
        config = self.get_config(source_name)
        return config.enabled if config else False

    def enable_source(self, source_name: str):
        """データソース有効化"""
        if source_name in self.configs:
            self.configs[source_name].enabled = True
            self.logger.info(f"Enabled data source: {source_name}")

    def disable_source(self, source_name: str):
        """データソース無効化"""
        if source_name in self.configs:
            self.configs[source_name].enabled = False
            self.logger.info(f"Disabled data source: {source_name}")

    def get_enabled_sources(self) -> List[str]:
        """有効なデータソース一覧"""
        return [name for name, config in self.configs.items() if config.enabled]

    def save_configs(self):
        """設定保存"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            config_data = {
                'data_sources': {
                    name: {
                        'enabled': config.enabled,
                        'priority': config.priority,
                        'timeout': config.timeout,
                        'retry_count': config.retry_count,
                        'rate_limit_per_minute': config.rate_limit_per_minute,
                        'rate_limit_per_day': config.rate_limit_per_day,
                        'quality_threshold': config.quality_threshold,
                        'api_key': config.api_key,
                        'base_url': config.base_url,
                        'headers': config.headers,
                        'min_data_points': config.min_data_points,
                        'max_price_threshold': config.max_price_threshold,
                        'price_consistency_check': config.price_consistency_check,
                        'cache_enabled': config.cache_enabled,
                        'cache_ttl_seconds': config.cache_ttl_seconds
                    }
                    for name, config in self.configs.items()
                }
            }

            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

            self.logger.info(f"Saved data source configurations to {self.config_path}")

        except Exception as e:
            self.logger.error(f"Failed to save configs: {e}")


class ImprovedCacheManager:
    """改善版キャッシュ管理"""

    def __init__(self, cache_dir: Path = Path("data/cache"), use_redis: bool = False):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Redis設定
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.redis_client = None

        if self.use_redis:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
                self.redis_client.ping()
                self.logger.info("Connected to Redis cache")
            except Exception as e:
                self.logger.warning(f"Redis connection failed, using file cache: {e}")
                self.use_redis = False

        # メモリキャッシュ
        self.memory_cache = {}
        self.cache_timestamps = {}
        self.max_memory_items = 1000

    def _get_cache_key(self, symbol: str, period: str, source: str) -> str:
        """キャッシュキー生成"""
        key_str = f"{symbol}_{period}_{source}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def get_cached_data(self, symbol: str, period: str, source: str,
                            ttl_seconds: int = 300) -> Optional[pd.DataFrame]:
        """キャッシュからデータ取得"""
        cache_key = self._get_cache_key(symbol, period, source)

        try:
            # 1. メモリキャッシュ確認
            if cache_key in self.memory_cache:
                cached_time = self.cache_timestamps.get(cache_key, 0)
                if time.time() - cached_time < ttl_seconds:
                    self.logger.debug(f"Memory cache hit: {symbol}")
                    return self.memory_cache[cache_key]
                else:
                    # 期限切れのため削除
                    del self.memory_cache[cache_key]
                    del self.cache_timestamps[cache_key]

            # 2. Redis確認
            if self.use_redis and self.redis_client:
                try:
                    cached_data = self.redis_client.get(cache_key)
                    if cached_data:
                        data = pickle.loads(cached_data)
                        self.logger.debug(f"Redis cache hit: {symbol}")
                        # メモリキャッシュにも保存
                        self._store_in_memory(cache_key, data)
                        return data
                except Exception as e:
                    self.logger.warning(f"Redis cache read error: {e}")

            # 3. ファイルキャッシュ確認
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age < ttl_seconds:
                    try:
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                        self.logger.debug(f"File cache hit: {symbol}")
                        # メモリキャッシュにも保存
                        self._store_in_memory(cache_key, data)
                        return data
                    except Exception as e:
                        self.logger.warning(f"File cache read error: {e}")
                        cache_file.unlink(missing_ok=True)

            return None

        except Exception as e:
            self.logger.error(f"Cache get error for {symbol}: {e}")
            return None

    async def store_cached_data(self, symbol: str, period: str, source: str,
                              data: pd.DataFrame, ttl_seconds: int = 300):
        """キャッシュにデータ保存"""
        cache_key = self._get_cache_key(symbol, period, source)

        try:
            # 1. メモリキャッシュに保存
            self._store_in_memory(cache_key, data)

            # 2. Redis に保存
            if self.use_redis and self.redis_client:
                try:
                    serialized_data = pickle.dumps(data)
                    self.redis_client.setex(cache_key, ttl_seconds, serialized_data)
                    self.logger.debug(f"Stored in Redis cache: {symbol}")
                except Exception as e:
                    self.logger.warning(f"Redis cache write error: {e}")

            # 3. ファイルキャッシュに保存
            try:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                self.logger.debug(f"Stored in file cache: {symbol}")
            except Exception as e:
                self.logger.warning(f"File cache write error: {e}")

        except Exception as e:
            self.logger.error(f"Cache store error for {symbol}: {e}")

    def _store_in_memory(self, cache_key: str, data: pd.DataFrame):
        """メモリキャッシュ保存"""
        # メモリキャッシュサイズ制限
        if len(self.memory_cache) >= self.max_memory_items:
            # 最も古いエントリを削除
            oldest_key = min(self.cache_timestamps.keys(),
                           key=lambda k: self.cache_timestamps[k])
            del self.memory_cache[oldest_key]
            del self.cache_timestamps[oldest_key]

        self.memory_cache[cache_key] = data.copy()
        self.cache_timestamps[cache_key] = time.time()

    def clear_cache(self, symbol: Optional[str] = None):
        """キャッシュクリア"""
        if symbol:
            # 特定銘柄のキャッシュクリア
            keys_to_remove = []
            for key in self.memory_cache.keys():
                if symbol in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]

            self.logger.info(f"Cleared cache for symbol: {symbol}")
        else:
            # 全キャッシュクリア
            self.memory_cache.clear()
            self.cache_timestamps.clear()

            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    self.logger.warning(f"Redis cache clear error: {e}")

            self.logger.info("Cleared all cache")


class BaseDataProvider(ABC):
    """データプロバイダー基底クラス"""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
        self.daily_request_count = 0
        self.last_request_time = 0
        self.request_history = []

    @abstractmethod
    async def get_stock_data(self, symbol: str, period: str = "1mo") -> DataFetchResult:
        """株価データ取得（抽象メソッド）"""
        pass

    async def _wait_for_rate_limit(self):
        """レート制限対応"""
        current_time = time.time()

        # 1分間のリクエスト数制限
        minute_ago = current_time - 60
        recent_requests = [t for t in self.request_history if t > minute_ago]

        if len(recent_requests) >= self.config.rate_limit_per_minute:
            wait_time = 60 - (current_time - recent_requests[0])
            if wait_time > 0:
                self.logger.debug(f"Rate limit wait: {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        # 最小間隔制限
        if self.last_request_time > 0:
            min_interval = 60 / self.config.rate_limit_per_minute
            elapsed = current_time - self.last_request_time
            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                await asyncio.sleep(wait_time)

    def _record_request(self):
        """リクエスト記録"""
        current_time = time.time()
        self.request_history.append(current_time)
        self.last_request_time = current_time
        self.request_count += 1

        # 1日前より古い記録は削除
        day_ago = current_time - 86400
        self.request_history = [t for t in self.request_history if t > day_ago]
        self.daily_request_count = len(self.request_history)

    def _calculate_quality_score(self, data: pd.DataFrame) -> Tuple[DataQualityLevel, float]:
        """データ品質スコア計算"""
        if data is None or data.empty:
            return DataQualityLevel.FAILED, 0.0

        score = 0.0
        max_score = 100.0

        # データ量チェック (20点)
        if len(data) >= self.config.min_data_points:
            score += 20.0
        else:
            score += 20.0 * (len(data) / self.config.min_data_points)

        # 必要列の存在チェック (20点)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        existing_columns = sum(1 for col in required_columns if col in data.columns)
        score += 20.0 * (existing_columns / len(required_columns))

        # データ完全性チェック (20点)
        completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        score += 20.0 * completeness

        # 価格整合性チェック (20点)
        if self.config.price_consistency_check and all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            consistency_violations = (
                (data['High'] < data['Low']) |
                (data['High'] < data['Open']) |
                (data['High'] < data['Close']) |
                (data['Low'] > data['Open']) |
                (data['Low'] > data['Close'])
            ).sum()

            consistency_score = 1.0 - (consistency_violations / len(data))
            score += 20.0 * consistency_score
        else:
            score += 20.0  # 整合性チェック無効の場合は満点

        # 異常値チェック (20点)
        anomaly_score = 20.0
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                if (data[col] <= 0).any():
                    anomaly_score -= 5.0
                if (data[col] > self.config.max_price_threshold).any():
                    anomaly_score -= 5.0

        score += max(0.0, anomaly_score)

        # 品質レベル判定
        if score >= 90:
            level = DataQualityLevel.HIGH
        elif score >= 70:
            level = DataQualityLevel.MEDIUM
        elif score >= 50:
            level = DataQualityLevel.LOW
        else:
            level = DataQualityLevel.FAILED

        return level, score


class ImprovedYahooFinanceProvider(BaseDataProvider):
    """改善版Yahoo Finance プロバイダー"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.symbol_variations_cache = {}

    async def get_stock_data(self, symbol: str, period: str = "1mo") -> DataFetchResult:
        """株価データ取得（改良版）"""
        start_time = time.time()

        if not YFINANCE_AVAILABLE:
            return DataFetchResult(
                data=None,
                source=DataSource.YAHOO_FINANCE,
                quality_level=DataQualityLevel.FAILED,
                quality_score=0.0,
                fetch_time=0.0,
                error_message="yfinance not available"
            )

        try:
            # レート制限チェック
            if self.daily_request_count >= self.config.rate_limit_per_day:
                return DataFetchResult(
                    data=None,
                    source=DataSource.YAHOO_FINANCE,
                    quality_level=DataQualityLevel.FAILED,
                    quality_score=0.0,
                    fetch_time=0.0,
                    error_message="Daily rate limit exceeded"
                )

            await self._wait_for_rate_limit()

            # 複数の銘柄コード形式を試行（改善版）
            symbol_variations = self._generate_symbol_variations(symbol)
            last_error = None

            for ticker_symbol in symbol_variations:
                try:
                    self.logger.debug(f"Trying symbol variation: {ticker_symbol}")

                    ticker = yf.Ticker(ticker_symbol)
                    data = ticker.history(period=period)

                    self._record_request()

                    if not data.empty and len(data) > 0:
                        # データ品質チェック（改善版）
                        quality_level, quality_score = self._calculate_quality_score(data)

                        if quality_score >= self.config.quality_threshold:
                            fetch_time = time.time() - start_time

                            self.logger.info(f"Successfully fetched {symbol} as {ticker_symbol} "
                                           f"(quality: {quality_score:.1f})")

                            return DataFetchResult(
                                data=data,
                                source=DataSource.YAHOO_FINANCE,
                                quality_level=quality_level,
                                quality_score=quality_score,
                                fetch_time=fetch_time,
                                metadata={
                                    'symbol_used': ticker_symbol,
                                    'variations_tried': symbol_variations.index(ticker_symbol) + 1
                                }
                            )
                        else:
                            self.logger.debug(f"Data quality insufficient for {ticker_symbol}: {quality_score:.1f}")
                            last_error = f"Data quality too low: {quality_score:.1f}"

                except Exception as e:
                    last_error = str(e)
                    self.logger.debug(f"Failed to fetch {ticker_symbol}: {e}")
                    continue

            # 全てのバリエーションが失敗
            fetch_time = time.time() - start_time
            error_details = self._generate_error_details(symbol, symbol_variations, last_error)

            self.logger.warning(f"All symbol variations failed for {symbol}: {error_details}")

            return DataFetchResult(
                data=None,
                source=DataSource.YAHOO_FINANCE,
                quality_level=DataQualityLevel.FAILED,
                quality_score=0.0,
                fetch_time=fetch_time,
                error_message=error_details,
                metadata={'variations_tried': len(symbol_variations)}
            )

        except Exception as e:
            fetch_time = time.time() - start_time
            error_msg = f"Yahoo Finance fetch error: {e}"
            self.logger.error(error_msg)

            return DataFetchResult(
                data=None,
                source=DataSource.YAHOO_FINANCE,
                quality_level=DataQualityLevel.FAILED,
                quality_score=0.0,
                fetch_time=fetch_time,
                error_message=error_msg
            )

    def _generate_symbol_variations(self, symbol: str) -> List[str]:
        """銘柄コードバリエーション生成（改善版）"""
        # キャッシュ確認
        if symbol in self.symbol_variations_cache:
            return self.symbol_variations_cache[symbol]

        variations = []

        if symbol.isdigit():
            # 数字のみの銘柄コード
            variations.extend([
                f"{symbol}.T",      # 東京証券取引所（最優先）
                f"{symbol}.JP",     # 日本
                symbol,             # そのまま
                f"{symbol}.TO",     # 東京
                f"{symbol}.TYO"     # Tokyo
            ])
        else:
            # すでにサフィックス付き
            variations.append(symbol)

            if '.' in symbol:
                base_symbol = symbol.split('.')[0]
                if base_symbol.isdigit():
                    variations.extend([
                        f"{base_symbol}.T",
                        f"{base_symbol}.JP",
                        base_symbol,
                        f"{base_symbol}.TO",
                        f"{base_symbol}.TYO"
                    ])

        # 重複除去
        variations = list(dict.fromkeys(variations))

        # キャッシュに保存
        self.symbol_variations_cache[symbol] = variations

        return variations

    def _generate_error_details(self, symbol: str, variations: List[str], last_error: str) -> str:
        """エラー詳細生成"""
        details = [
            f"Symbol: {symbol}",
            f"Variations tried: {len(variations)}",
            f"Variations: {', '.join(variations)}",
            f"Last error: {last_error or 'Unknown'}",
            f"Daily requests: {self.daily_request_count}/{self.config.rate_limit_per_day}"
        ]
        return " | ".join(details)


class ImprovedStooqProvider(BaseDataProvider):
    """改善版Stooq プロバイダー"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)

    async def get_stock_data(self, symbol: str, period: str = "1mo") -> DataFetchResult:
        """Stooqから株価データ取得（改良版）"""
        start_time = time.time()

        try:
            await self._wait_for_rate_limit()

            # Stooq用の銘柄コード変換
            stooq_symbol = self._convert_to_stooq_symbol(symbol)
            days = self._period_to_days(period)

            # URL構築
            url = f"{self.config.base_url}?s={stooq_symbol}&d1={days}&i=d"

            timeout = aiohttp.ClientTimeout(total=self.config.timeout)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=self.config.headers) as response:
                    self._record_request()

                    if response.status == 200:
                        text = await response.text()

                        # CSVデータをDataFrameに変換
                        from io import StringIO
                        df = pd.read_csv(StringIO(text))

                        if not df.empty and 'Close' in df.columns:
                            # 標準フォーマットに変換
                            df = self._convert_to_standard_format(df)

                            # データ品質チェック
                            quality_level, quality_score = self._calculate_quality_score(df)
                            fetch_time = time.time() - start_time

                            self.logger.info(f"Stooq fetch successful for {symbol} "
                                           f"(quality: {quality_score:.1f})")

                            return DataFetchResult(
                                data=df,
                                source=DataSource.STOOQ,
                                quality_level=quality_level,
                                quality_score=quality_score,
                                fetch_time=fetch_time,
                                metadata={'stooq_symbol': stooq_symbol}
                            )
                    else:
                        error_msg = f"HTTP {response.status}"
                        self.logger.warning(f"Stooq HTTP error for {symbol}: {error_msg}")

        except Exception as e:
            error_msg = f"Stooq fetch error: {e}"
            self.logger.error(f"Stooq error for {symbol}: {error_msg}")

        fetch_time = time.time() - start_time
        return DataFetchResult(
            data=None,
            source=DataSource.STOOQ,
            quality_level=DataQualityLevel.FAILED,
            quality_score=0.0,
            fetch_time=fetch_time,
            error_message=error_msg if 'error_msg' in locals() else "Unknown error"
        )

    def _convert_to_stooq_symbol(self, symbol: str) -> str:
        """Stooq用銘柄コード変換"""
        if symbol.isdigit():
            return f"{symbol}.jp"
        elif symbol.endswith('.T'):
            return symbol.replace('.T', '.jp')
        elif symbol.endswith('.JP'):
            return symbol.lower()
        return symbol

    def _period_to_days(self, period: str) -> int:
        """期間文字列を日数に変換"""
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }
        return period_map.get(period, 30)

    def _convert_to_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """標準フォーマットに変換"""
        # 列名マッピング
        column_map = {
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }

        # 列名変換
        df = df.rename(columns=column_map)

        # 日付処理
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')

        # 数値型変換
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # NaN値を削除
        df = df.dropna()

        return df


class MockDataProvider(BaseDataProvider):
    """モックデータプロバイダー（テスト用）"""

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)

    async def get_stock_data(self, symbol: str, period: str = "1mo") -> DataFetchResult:
        """模擬データ生成"""
        start_time = time.time()

        try:
            await asyncio.sleep(0.1)  # 模擬遅延

            # データ期間決定
            days = self._period_to_days(period)

            # 模擬株価データ生成
            np.random.seed(hash(symbol) % 2**32)
            base_price = 1000 + (hash(symbol) % 10000)

            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            prices = []
            current_price = base_price

            for _ in range(days):
                change = np.random.normal(0, 0.02)  # 2%の標準偏差
                current_price *= (1 + change)
                current_price = max(current_price, 100)  # 最低価格
                prices.append(current_price)

            # OHLCV データ作成
            data = []
            for i, (date, close) in enumerate(zip(dates, prices)):
                open_price = prices[i-1] if i > 0 else close
                high = max(open_price, close) * np.random.uniform(1.0, 1.02)
                low = min(open_price, close) * np.random.uniform(0.98, 1.0)
                volume = np.random.randint(100000, 10000000)

                data.append({
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': volume
                })

            df = pd.DataFrame(data, index=dates)

            # 品質スコア計算
            quality_level, quality_score = self._calculate_quality_score(df)
            fetch_time = time.time() - start_time

            self._record_request()

            return DataFetchResult(
                data=df,
                source=DataSource.MOCK,
                quality_level=quality_level,
                quality_score=quality_score,
                fetch_time=fetch_time,
                metadata={'generated': True, 'base_price': base_price}
            )

        except Exception as e:
            fetch_time = time.time() - start_time
            return DataFetchResult(
                data=None,
                source=DataSource.MOCK,
                quality_level=DataQualityLevel.FAILED,
                quality_score=0.0,
                fetch_time=fetch_time,
                error_message=f"Mock data generation error: {e}"
            )

    def _period_to_days(self, period: str) -> int:
        """期間文字列を日数に変換"""
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }
        return period_map.get(period, 30)


class ImprovedMultiSourceDataProvider:
    """改善版複数ソース対応データプロバイダー"""

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)

        # 設定管理
        self.config_manager = DataSourceConfigManager(config_path)

        # キャッシュ管理
        self.cache_manager = ImprovedCacheManager()

        # プロバイダー初期化
        self.providers = self._initialize_providers()

        # 統計情報
        self.fetch_statistics = defaultdict(lambda: {
            'requests': 0, 'successes': 0, 'failures': 0,
            'total_time': 0.0, 'avg_quality': 0.0
        })

        self.logger.info(f"Initialized MultiSourceDataProvider with {len(self.providers)} providers")

    def _initialize_providers(self) -> Dict[str, BaseDataProvider]:
        """プロバイダー初期化"""
        providers = {}

        # Yahoo Finance
        if self.config_manager.is_enabled('yahoo_finance') and YFINANCE_AVAILABLE:
            config = self.config_manager.get_config('yahoo_finance')
            providers['yahoo_finance'] = ImprovedYahooFinanceProvider(config)

        # Stooq
        if self.config_manager.is_enabled('stooq'):
            config = self.config_manager.get_config('stooq')
            providers['stooq'] = ImprovedStooqProvider(config)

        # Mock (always available for testing)
        if self.config_manager.is_enabled('mock'):
            config = self.config_manager.get_config('mock')
            providers['mock'] = MockDataProvider(config)

        return providers

    async def get_stock_data(self, symbol: str, period: str = "1mo",
                           preferred_source: Optional[str] = None,
                           use_cache: bool = True) -> DataFetchResult:
        """株価データ取得（改善版）"""

        # キャッシュ確認
        if use_cache:
            for source_name in self._get_source_priority_order(preferred_source):
                if source_name not in self.providers:
                    continue

                config = self.config_manager.get_config(source_name)
                if config and config.cache_enabled:
                    cached_data = await self.cache_manager.get_cached_data(
                        symbol, period, source_name, config.cache_ttl_seconds
                    )

                    if cached_data is not None:
                        quality_level, quality_score = self.providers[source_name]._calculate_quality_score(cached_data)

                        self.logger.info(f"Cache hit for {symbol} from {source_name}")

                        return DataFetchResult(
                            data=cached_data,
                            source=DataSource(source_name),
                            quality_level=quality_level,
                            quality_score=quality_score,
                            fetch_time=0.0,
                            cached=True,
                            metadata={'source': source_name}
                        )

        # プロバイダーからデータ取得
        source_order = self._get_source_priority_order(preferred_source)
        best_result = None

        for source_name in source_order:
            if source_name not in self.providers:
                continue

            try:
                provider = self.providers[source_name]
                config = self.config_manager.get_config(source_name)

                self.logger.debug(f"Trying to fetch {symbol} from {source_name}")

                result = await provider.get_stock_data(symbol, period)

                # 統計更新
                self._update_statistics(source_name, result)

                if result.data is not None and result.quality_score >= config.quality_threshold:
                    # 成功：キャッシュに保存
                    if use_cache and config.cache_enabled:
                        await self.cache_manager.store_cached_data(
                            symbol, period, source_name, result.data, config.cache_ttl_seconds
                        )

                    self.logger.info(f"Successfully fetched {symbol} from {source_name} "
                                   f"(quality: {result.quality_score:.1f})")
                    return result

                # 品質が低いが、データは取得できた場合
                if result.data is not None and (best_result is None or
                                              result.quality_score > best_result.quality_score):
                    best_result = result

            except Exception as e:
                self.logger.error(f"Provider error for {source_name}: {e}")
                continue

        # 全プロバイダーが失敗した場合
        if best_result is not None:
            self.logger.warning(f"Returning low quality data for {symbol} "
                              f"(quality: {best_result.quality_score:.1f})")
            return best_result

        # 完全失敗
        self.logger.error(f"Failed to fetch data for {symbol} from all sources")
        return DataFetchResult(
            data=None,
            source=DataSource.MOCK,  # フォールバック
            quality_level=DataQualityLevel.FAILED,
            quality_score=0.0,
            fetch_time=0.0,
            error_message="All data sources failed"
        )

    def _get_source_priority_order(self, preferred_source: Optional[str] = None) -> List[str]:
        """データソース優先順序取得"""
        enabled_sources = self.config_manager.get_enabled_sources()

        if preferred_source and preferred_source in enabled_sources:
            # 優先ソースを最初に
            order = [preferred_source]
            order.extend([s for s in enabled_sources if s != preferred_source])
            return order

        # 優先度順にソート
        sources_with_priority = []
        for source_name in enabled_sources:
            config = self.config_manager.get_config(source_name)
            if config:
                sources_with_priority.append((source_name, config.priority))

        # 優先度でソート（小さい値が高優先度）
        sources_with_priority.sort(key=lambda x: x[1])

        return [source for source, _ in sources_with_priority]

    def _update_statistics(self, source_name: str, result: DataFetchResult):
        """統計情報更新"""
        stats = self.fetch_statistics[source_name]
        stats['requests'] += 1
        stats['total_time'] += result.fetch_time

        if result.data is not None:
            stats['successes'] += 1
            # 移動平均で品質スコア更新
            current_avg = stats['avg_quality']
            new_avg = (current_avg * (stats['successes'] - 1) + result.quality_score) / stats['successes']
            stats['avg_quality'] = new_avg
        else:
            stats['failures'] += 1

    def get_statistics(self) -> Dict[str, Dict]:
        """統計情報取得"""
        stats = {}
        for source_name, data in self.fetch_statistics.items():
            total_requests = data['requests']
            if total_requests > 0:
                stats[source_name] = {
                    'total_requests': total_requests,
                    'success_rate': data['successes'] / total_requests * 100,
                    'failure_rate': data['failures'] / total_requests * 100,
                    'avg_response_time': data['total_time'] / total_requests,
                    'avg_quality_score': data['avg_quality']
                }

        return stats

    def get_source_status(self) -> Dict[str, Dict]:
        """データソース状態取得"""
        status = {}
        for source_name, provider in self.providers.items():
            config = self.config_manager.get_config(source_name)
            stats = self.fetch_statistics[source_name]

            status[source_name] = {
                'enabled': config.enabled,
                'priority': config.priority,
                'daily_requests': provider.daily_request_count,
                'daily_limit': config.rate_limit_per_day,
                'requests_remaining': config.rate_limit_per_day - provider.daily_request_count,
                'success_rate': (stats['successes'] / stats['requests'] * 100) if stats['requests'] > 0 else 0,
                'avg_quality': stats['avg_quality']
            }

        return status

    def enable_source(self, source_name: str):
        """データソース有効化"""
        self.config_manager.enable_source(source_name)

        # プロバイダーを再初期化
        if source_name not in self.providers:
            self.providers = self._initialize_providers()

    def disable_source(self, source_name: str):
        """データソース無効化"""
        self.config_manager.disable_source(source_name)

        # プロバイダーから削除
        if source_name in self.providers:
            del self.providers[source_name]

    def clear_cache(self, symbol: Optional[str] = None):
        """キャッシュクリア"""
        self.cache_manager.clear_cache(symbol)


# グローバルインスタンス
improved_data_provider = ImprovedMultiSourceDataProvider()


# テスト関数
async def test_improved_data_provider():
    """改善版データプロバイダーのテスト"""
    print("=== Improved Multi-Source Data Provider Test ===")

    try:
        # プロバイダー初期化
        provider = ImprovedMultiSourceDataProvider()
        print(f"✓ Provider initialized with {len(provider.providers)} sources")

        # 有効なソース確認
        enabled_sources = provider.config_manager.get_enabled_sources()
        print(f"✓ Enabled sources: {', '.join(enabled_sources)}")

        # テスト銘柄でデータ取得
        test_symbols = ["7203", "4751"]

        for symbol in test_symbols:
            print(f"\n--- Testing symbol: {symbol} ---")

            # データ取得
            result = await provider.get_stock_data(symbol, "1mo")

            if result.data is not None:
                print(f"✓ Data fetched successfully")
                print(f"  - Source: {result.source.value}")
                print(f"  - Quality: {result.quality_level.value} ({result.quality_score:.1f})")
                print(f"  - Data points: {len(result.data)}")
                print(f"  - Fetch time: {result.fetch_time:.2f}s")
                print(f"  - Cached: {result.cached}")

                # データ内容確認
                if not result.data.empty:
                    latest = result.data.iloc[-1]
                    print(f"  - Latest close: {latest['Close']:.2f}")
            else:
                print(f"❌ Data fetch failed: {result.error_message}")

        # 統計情報表示
        print("\n--- Provider Statistics ---")
        stats = provider.get_statistics()
        for source, data in stats.items():
            print(f"{source}:")
            print(f"  - Success rate: {data['success_rate']:.1f}%")
            print(f"  - Avg response time: {data['avg_response_time']:.2f}s")
            print(f"  - Avg quality: {data['avg_quality_score']:.1f}")

        # ソース状態表示
        print("\n--- Source Status ---")
        status = provider.get_source_status()
        for source, data in status.items():
            print(f"{source}:")
            print(f"  - Enabled: {data['enabled']}")
            print(f"  - Priority: {data['priority']}")
            print(f"  - Daily requests: {data['daily_requests']}/{data['daily_limit']}")

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


# グローバルインスタンス
real_data_provider = ImprovedMultiSourceDataProvider()

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # テスト実行
    asyncio.run(test_improved_data_provider())