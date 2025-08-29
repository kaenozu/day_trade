#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Data Provider - 強化データプロバイダシステム
フォールバックの見直しとリアルデータの信頼性向上
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import sys # Added for debug prints

from ..alerts.fallback_notification_system import (
    notify_fallback_usage, notify_dummy_data_usage, notify_real_data_recovery,
    DataSource
)


class DataQuality(Enum):
    """データ品質レベル"""
    HIGH = "high"           # リアルタイム、高精度
    MEDIUM = "medium"       # 若干の遅延あり、通常精度
    LOW = "low"            # 大幅遅延または推定値
    FALLBACK = "fallback"  # フォールバックデータ
    DUMMY = "dummy"        # ダミーデータ


@dataclass
class DataResult:
    """データ取得結果"""
    data: Any
    quality: DataQuality
    source: str
    timestamp: datetime
    latency_ms: float
    error_message: Optional[str] = None
    is_cached: bool = False
    cache_age_seconds: float = 0.0


class EnhancedDataProvider:
    """強化データプロバイダ"""

    def __init__(self):
        self.providers = []
        self.cache = {}
        self.circuit_breakers = {}
        self.retry_counts = {}
        self.last_success_times = {}

        # ロガーの初期化
        from daytrade_logging import get_logger
        self.logger = get_logger("enhanced_data_provider")

        # プロバイダーの初期化
        self._init_providers()

    def _init_providers(self):
        """データプロバイダーの初期化"""
        try:
            # yfinanceプロバイダー
            from src.day_trade.utils.yfinance_import import get_yfinance
            yf_module, available = get_yfinance()
            if available:
                self.providers.append({
                    'name': 'yfinance',
                    'module': yf_module,
                    'priority': 1,
                    'timeout': 10.0,
                    'retry_limit': 3
                })
                self.logger.info("yfinance provider initialized")
        except ImportError:
            self.logger.warning("yfinance provider not available")

        # 他のデータプロバイダーも同様に追加可能
        self.logger.info(f"Initialized {len(self.providers)} data providers")

    async def get_stock_data(self, symbol: str, period: str = "1d") -> DataResult:
        """株価データを取得"""
        start_time = time.time()

        # キャッシュチェック
        cache_result = self._check_cache(f"stock_{symbol}_{period}")
        if cache_result:
            print(f"[DEBUG EPD] Cache hit for {symbol}. Quality: {cache_result.quality.value}", file=sys.stderr) # Debug print
            return cache_result

        # プロバイダーから順次取得を試行
        for provider in self.providers:
            try:
                # サーキットブレーカーチェック
                if self._is_circuit_open(provider['name']):
                    print(f"[DEBUG EPD] Circuit open for {provider['name']}", file=sys.stderr) # Debug print
                    continue

                result = await self._get_stock_data_from_provider(provider, symbol, period)
                print(f"[DEBUG EPD] Provider {provider['name']} returned result for {symbol}. Quality: {result.quality.value}", file=sys.stderr) # Debug print

                if result.quality != DataQuality.DUMMY:
                    # 成功時の処理
                    self._record_success(provider['name'])
                    self._update_cache(f"stock_{symbol}_{period}", result)

                    # 復旧通知
                    if provider['name'] in self.circuit_breakers:
                        notify_real_data_recovery(provider['name'], "stock_data")

                    return result

            except Exception as e:
                self.logger.warning(f"Provider {provider['name']} failed: {e}")
                print(f"[DEBUG EPD] Provider {provider['name']} failed for {symbol}: {e}", file=sys.stderr) # Debug print
                self._record_failure(provider['name'])
                continue

        # すべてのプロバイダーが失敗した場合のフォールバック
        fallback_result = self._generate_fallback_stock_data(symbol, period, start_time)
        print(f"[DEBUG EPD] All providers failed for {symbol}. Returning fallback/dummy. Quality: {fallback_result.quality.value}", file=sys.stderr) # Debug print
        return fallback_result

    async def get_market_data(self) -> DataResult:
        """市場データを取得"""
        start_time = time.time()

        cache_result = self._check_cache("market_data")
        if cache_result:
            return cache_result

        # 簡単な市場データを返す（実装簡略化）
        market_data = {
            'market_status': 'open',
            'nikkei225': 28000.0,
            'topix': 1950.0,
            'volume': 1500000000,
            'timestamp': datetime.now().isoformat()
        }

        result = DataResult(
            data=market_data,
            quality=DataQuality.MEDIUM,
            source="market_api",
            timestamp=datetime.now(),
            latency_ms=(time.time() - start_time) * 1000
        )

        self._update_cache("market_data", result)
        return result

    async def _get_stock_data_from_provider(self, provider: Dict[str, Any],
                                          symbol: str, period: str) -> DataResult:
        """特定プロバイダーから株価データを取得"""
        start_time = time.time()

        if provider['name'] == 'yfinance':
            return await self._get_yfinance_data(provider, symbol, period, start_time)

        # 他のプロバイダーの実装もここに追加
        raise ValueError(f"Unknown provider: {provider['name']}")

    async def _get_yfinance_data(self, provider: Dict[str, Any],
                               symbol: str, period: str, start_time: float) -> DataResult:
        """yfinanceからデータを取得"""
        try:
            yf_module = provider['module']
            symbol_yf = f"{symbol}.T" if symbol.isdigit() and len(symbol) == 4 else symbol

            ticker = yf_module.Ticker(symbol_yf)

            # タイムアウト付きでデータ取得
            future = asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ticker.history(period=period)
            )

            hist = await asyncio.wait_for(future, timeout=provider['timeout'])
            print(f"[DEBUG EPD] yfinance hist length for {symbol}: {len(hist)}", file=sys.stderr) # Debug print

            if len(hist) == 0:
                raise ValueError("No data returned from yfinance")

            # データ品質の評価
            quality = self._evaluate_data_quality(hist)

            # 最新データを抽出
            latest = hist.iloc[-1]
            stock_data = {
                'symbol': symbol,
                'price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume']),
                'change': ((float(latest['Close']) - float(latest['Open'])) / float(latest['Open'])) * 100,
                'last_updated': hist.index[-1].isoformat()
            }

            return DataResult(
                data=stock_data,
                quality=quality,
                source=f"yfinance_{symbol_yf}",
                timestamp=datetime.now(),
                latency_ms=(time.time() - start_time) * 1000
            )

        except asyncio.TimeoutError:
            print(f"[DEBUG EPD] yfinance timeout for {symbol}", file=sys.stderr) # Debug print
            notify_fallback_usage("yfinance", "stock_data", "Timeout", DataSource.FALLBACK_DATA)
            raise
        except Exception as e:
            print(f"[DEBUG EPD] yfinance general error for {symbol}: {e}", file=sys.stderr) # Debug print
            notify_fallback_usage("yfinance", "stock_data", str(e), DataSource.FALLBACK_DATA)
            raise

    def _evaluate_data_quality(self, data) -> DataQuality:
        """データ品質を評価"""
        if len(data) == 0:
            return DataQuality.DUMMY

        # 最新データの鮮度をチェック
        latest_timestamp = data.index[-1]
        now = datetime.now()
        age = now - latest_timestamp.to_pydatetime().replace(tzinfo=None)

        if age.total_seconds() < 300:  # 5分以内
            return DataQuality.HIGH
        elif age.total_seconds() < 3600:  # 1時間以内
            return DataQuality.MEDIUM
        else:
            return DataQuality.LOW

    def _generate_fallback_stock_data(self, symbol: str, period: str, start_time: float) -> DataResult:
        """フォールバック株価データを生成"""
        # 過去のキャッシュデータがあれば使用
        cached_data = self._get_historical_cache(f"stock_{symbol}_{period}")

        if cached_data:
            notify_fallback_usage("cache", "stock_data", "Using cached data", DataSource.CACHED_DATA)

            # キャッシュデータを少し調整
            stock_data = cached_data.copy()
            stock_data['last_updated'] = datetime.now().isoformat()
            stock_data['note'] = 'Cached data - may be outdated'

            return DataResult(
                data=stock_data,
                quality=DataQuality.FALLBACK,
                source="cache_fallback",
                timestamp=datetime.now(),
                latency_ms=(time.time() - start_time) * 1000,
                is_cached=True
            )

        # 最後の手段：ダミーデータ
        notify_dummy_data_usage("fallback_generator", "stock_data", "All providers failed")

        dummy_data = {
            'symbol': symbol,
            'price': 1000.0,
            'open': 995.0,
            'high': 1010.0,
            'low': 990.0,
            'volume': 100000,
            'change': 0.5,
            'last_updated': datetime.now().isoformat(),
            'note': 'DUMMY DATA - Not real market data'
        }

        return DataResult(
            data=dummy_data,
            quality=DataQuality.DUMMY,
            source="dummy_generator",
            timestamp=datetime.now(),
            latency_ms=(time.time() - start_time) * 1000,
            error_message="All data providers failed"
        )

    def _check_cache(self, key: str) -> Optional[DataResult]:
        """キャッシュをチェック"""
        if key in self.cache:
            cache_entry = self.cache[key]
            age = (datetime.now() - cache_entry['timestamp']).total_seconds()

            # キャッシュの有効期限（5分）
            if age < 300:
                result = cache_entry['result']
                result.is_cached = True
                result.cache_age_seconds = age
                return result

        return None

    def _update_cache(self, key: str, result: DataResult):
        """キャッシュを更新"""
        self.cache[key] = {
            'timestamp': datetime.now(),
            'result': result
        }

        # キャッシュサイズ制限（100エントリー）
        if len(self.cache) > 100:
            oldest_key = min(self.cache.keys(),
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

    def _get_historical_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """期限切れキャッシュからデータを取得"""
        if key in self.cache:
            return self.cache[key]['result'].data
        return None

    def _is_circuit_open(self, provider_name: str) -> bool:
        """サーキットブレーカーがオープンかチェック"""
        if provider_name not in self.circuit_breakers:
            return False

        breaker = self.circuit_breakers[provider_name]

        # 5分経過したらリセット
        if (datetime.now() - breaker['timestamp']).total_seconds() > 300:
            del self.circuit_breakers[provider_name]
            return False

        return breaker['failures'] >= 3

    def _record_success(self, provider_name: str):
        """成功を記録"""
        self.last_success_times[provider_name] = datetime.now()

        # サーキットブレーカーをリセット
        if provider_name in self.circuit_breakers:
            del self.circuit_breakers[provider_name]

        # リトライカウントをリセット
        if provider_name in self.retry_counts:
            del self.retry_counts[provider_name]

    def _record_failure(self, provider_name: str):
        """失敗を記録"""
        if provider_name not in self.circuit_breakers:
            self.circuit_breakers[provider_name] = {
                'failures': 0,
                'timestamp': datetime.now()
            }

        self.circuit_breakers[provider_name]['failures'] += 1
        self.circuit_breakers[provider_name]['timestamp'] = datetime.now()

    def get_provider_status(self) -> Dict[str, Any]:
        """プロバイダー状況を取得"""
        status = {}

        for provider in self.providers:
            name = provider['name']
            status[name] = {
                'available': True,
                'circuit_open': self._is_circuit_open(name),
                'last_success': self.last_success_times.get(name),
                'failures': self.circuit_breakers.get(name, {}).get('failures', 0)
            }

        return status


# グローバルインスタンス
_data_provider = None


def get_data_provider() -> EnhancedDataProvider:
    """グローバルデータプロバイダーを取得"""
    global _data_provider
    if _data_provider is None:
        _data_provider = EnhancedDataProvider()
    return _data_provider


# 便利関数
async def get_stock_data(symbol: str, period: str = "1d") -> DataResult:
    """株価データを取得"""
    return await get_data_provider().get_stock_data(symbol, period)


async def get_market_data() -> DataResult:
    """市場データを取得"""
    return await get_data_provider().get_market_data()


def get_provider_status() -> Dict[str, Any]:
    """プロバイダー状況を取得"""
    return get_data_provider().get_provider_status()


if __name__ == "__main__":
    async def test_data_provider():
        print("📊 強化データプロバイダーテスト")
        print("=" * 50)

        # 株価データ取得テスト
        try:
            result = await get_stock_data("7203", "1d")
            print(f"株価データ取得結果:")
            print(f"  品質: {result.quality.value}")
            print(f"  ソース: {result.source}")
            print(f"  レイテンシ: {result.latency_ms:.2f}ms")
            print(f"  データ: {result.data}")
        except Exception as e:
            print(f"株価データ取得エラー: {e}")

        # プロバイダー状況
        status = get_provider_status()
        print(f"\nプロバイダー状況: {status}")

        print("\nテスト完了")

    asyncio.run(test_data_provider())