#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Source Data Provider

複数ソース対応データプロバイダー
"""

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import concurrent.futures

from .enums import DataSource, DataSourceConfig, DataFetchResult, DataQualityLevel
from .config_manager import DataSourceConfigManager
from .cache_manager import ImprovedCacheManager
from .base_provider import BaseDataProvider
from .yahoo_provider import ImprovedYahooFinanceProvider
from .stooq_provider import ImprovedStooqProvider
from .mock_provider import MockDataProvider

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class ImprovedMultiSourceDataProvider:
    """改善版複数ソース対応データプロバイダー"""

    def __init__(self, config_path: Optional[Path] = None):
        """初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
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

        self.logger.info(
            f"Initialized MultiSourceDataProvider with {len(self.providers)} providers"
        )

    def _initialize_providers(self) -> Dict[str, BaseDataProvider]:
        """プロバイダー初期化"""
        providers = {}

        # Yahoo Finance
        if (self.config_manager.is_enabled('yahoo_finance') and
            YFINANCE_AVAILABLE):
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

    async def get_stock_data(
        self,
        symbol: str,
        period: str = "1mo",
        preferred_source: Optional[str] = None,
        use_cache: bool = True
    ) -> DataFetchResult:
        """株価データ取得（改善版）
        
        Args:
            symbol: 銘柄コード
            period: データ期間
            preferred_source: 優先データソース
            use_cache: キャッシュ使用フラグ
            
        Returns:
            データ取得結果
        """

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
                        quality_level, quality_score = (
                            self.providers[source_name]._calculate_quality_score(
                                cached_data
                            )
                        )

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

                if (result.data is not None and
                    result.quality_score >= config.quality_threshold):
                    # 成功：キャッシュに保存
                    if use_cache and config.cache_enabled:
                        await self.cache_manager.store_cached_data(
                            symbol, period, source_name, result.data,
                            config.cache_ttl_seconds
                        )

                    self.logger.info(
                        f"Successfully fetched {symbol} from {source_name} "
                        f"(quality: {result.quality_score:.1f})"
                    )
                    return result

                # 品質が低いが、データは取得できた場合
                if (result.data is not None and
                    (best_result is None or
                     result.quality_score > best_result.quality_score)):
                    best_result = result

            except Exception as e:
                self.logger.error(f"Provider error for {source_name}: {e}")
                continue

        # 全プロバイダーが失敗した場合
        if best_result is not None:
            self.logger.warning(
                f"Returning low quality data for {symbol} "
                f"(quality: {best_result.quality_score:.1f})"
            )
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

    def _get_source_priority_order(
        self,
        preferred_source: Optional[str] = None
    ) -> List[str]:
        """データソース優先順序取得
        
        Args:
            preferred_source: 優先ソース
            
        Returns:
            優先順序付きソースリスト
        """
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
        """統計情報更新
        
        Args:
            source_name: ソース名
            result: 取得結果
        """
        stats = self.fetch_statistics[source_name]
        stats['requests'] += 1
        stats['total_time'] += result.fetch_time

        if result.data is not None:
            stats['successes'] += 1
            # 移動平均で品質スコア更新
            current_avg = stats['avg_quality']
            new_avg = (
                (current_avg * (stats['successes'] - 1) + result.quality_score)
                / stats['successes']
            )
            stats['avg_quality'] = new_avg
        else:
            stats['failures'] += 1

    def get_statistics(self) -> Dict[str, Dict]:
        """統計情報取得
        
        Returns:
            統計情報の辞書
        """
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

    def get_stock_data_sync(
        self,
        symbol: str,
        period: str = "1mo",
        preferred_source: Optional[str] = None,
        use_cache: bool = True
    ) -> DataFetchResult:
        """株価データ取得（同期版）
        
        Args:
            symbol: 銘柄コード
            period: データ期間
            preferred_source: 優先ソース
            use_cache: キャッシュ使用フラグ
            
        Returns:
            データ取得結果
        """
        # 既存のイベントループがある場合の対応
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 別スレッドで実行
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.get_stock_data(symbol, period, preferred_source, use_cache)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.get_stock_data(symbol, period, preferred_source, use_cache)
                )
        except RuntimeError:
            # イベントループが存在しない場合
            return asyncio.run(
                self.get_stock_data(symbol, period, preferred_source, use_cache)
            )

    def get_source_status(self) -> Dict[str, Dict]:
        """データソース状態取得
        
        Returns:
            ソース状態の辞書
        """
        status = {}
        for source_name, provider in self.providers.items():
            config = self.config_manager.get_config(source_name)
            stats = self.fetch_statistics[source_name]

            status[source_name] = {
                'enabled': config.enabled,
                'priority': config.priority,
                'daily_requests': provider.daily_request_count,
                'daily_limit': config.rate_limit_per_day,
                'requests_remaining': (
                    config.rate_limit_per_day - provider.daily_request_count
                ),
                'success_rate': (
                    (stats['successes'] / stats['requests'] * 100)
                    if stats['requests'] > 0 else 0
                ),
                'avg_quality': stats['avg_quality']
            }

        return status

    def enable_source(self, source_name: str):
        """データソース有効化
        
        Args:
            source_name: ソース名
        """
        self.config_manager.enable_source(source_name)

        # プロバイダーを再初期化
        if source_name not in self.providers:
            self.providers = self._initialize_providers()

    def disable_source(self, source_name: str):
        """データソース無効化
        
        Args:
            source_name: ソース名
        """
        self.config_manager.disable_source(source_name)

        # プロバイダーから削除
        if source_name in self.providers:
            del self.providers[source_name]

    def clear_cache(self, symbol: Optional[str] = None):
        """キャッシュクリア
        
        Args:
            symbol: クリア対象銘柄コード
        """
        self.cache_manager.clear_cache(symbol)