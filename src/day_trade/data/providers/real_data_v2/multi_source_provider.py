#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider V2 - Multi-Source Provider
リアルデータプロバイダー V2 - 統合プロバイダー

複数のデータソースを統合管理するメインプロバイダー
"""

import asyncio
import concurrent.futures
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .models import (
    DataSource, DataFetchResult, DataQualityLevel, ProviderStatistics
)
from .config_manager import DataSourceConfigManager
from .cache_manager import ImprovedCacheManager
from .base_provider import BaseDataProvider
from .yahoo_provider import ImprovedYahooFinanceProvider
from .stooq_provider import ImprovedStooqProvider
from .mock_provider import MockDataProvider

# yfinance の可用性チェック
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class ImprovedMultiSourceDataProvider:
    """改善版複数ソース対応データプロバイダー"""

    def __init__(self, config_path: Optional[Path] = None):
        """統合プロバイダー初期化
        
        Args:
            config_path: 設定ファイルパス
        """
        self.logger = logging.getLogger(__name__)

        # 設定管理
        self.config_manager = DataSourceConfigManager(config_path)

        # キャッシュ管理
        self.cache_manager = ImprovedCacheManager()

        # プロバイダー初期化
        self.providers = self._initialize_providers()

        # 統計情報
        self.fetch_statistics = defaultdict(ProviderStatistics)

        self.logger.info(
            f"Initialized MultiSourceDataProvider with {len(self.providers)} providers"
        )

    def _initialize_providers(self) -> Dict[str, BaseDataProvider]:
        """プロバイダー初期化
        
        Returns:
            初期化されたプロバイダーの辞書
        """
        providers = {}

        # Yahoo Finance
        if self.config_manager.is_enabled('yahoo_finance') and YFINANCE_AVAILABLE:
            config = self.config_manager.get_config('yahoo_finance')
            providers['yahoo_finance'] = ImprovedYahooFinanceProvider(config)
            self.logger.debug("Initialized Yahoo Finance provider")

        # Stooq
        if self.config_manager.is_enabled('stooq'):
            config = self.config_manager.get_config('stooq')
            providers['stooq'] = ImprovedStooqProvider(config)
            self.logger.debug("Initialized Stooq provider")

        # Mock (always available for testing)
        if self.config_manager.is_enabled('mock'):
            config = self.config_manager.get_config('mock')
            providers['mock'] = MockDataProvider(config)
            self.logger.debug("Initialized Mock provider")

        return providers

    async def get_stock_data(self, 
                           symbol: str, 
                           period: str = "1mo",
                           preferred_source: Optional[str] = None,
                           use_cache: bool = True) -> DataFetchResult:
        """株価データ取得（統合版）
        
        Args:
            symbol: 銘柄コード
            period: 取得期間
            preferred_source: 優先データソース
            use_cache: キャッシュ使用フラグ
            
        Returns:
            データ取得結果
        """

        # キャッシュ確認
        if use_cache:
            cached_result = await self._get_from_cache(symbol, period)
            if cached_result:
                return cached_result

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

        # 全プロバイダーが失敗した場合の処理
        return self._handle_all_failed(symbol, best_result)

    async def _get_from_cache(self, 
                            symbol: str, 
                            period: str) -> Optional[DataFetchResult]:
        """キャッシュからデータ取得試行
        
        Args:
            symbol: 銘柄コード
            period: 取得期間
            
        Returns:
            キャッシュされたデータ結果 or None
        """
        for source_name in self.config_manager.get_enabled_sources():
            if source_name not in self.providers:
                continue

            config = self.config_manager.get_config(source_name)
            if not config or not config.cache_enabled:
                continue

            cached_data = await self.cache_manager.get_cached_data(
                symbol, period, source_name, config.cache_ttl_seconds
            )

            if cached_data is not None:
                provider = self.providers[source_name]
                quality_level, quality_score = provider._calculate_quality_score(
                    cached_data
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

        return None

    def _get_source_priority_order(self, 
                                 preferred_source: Optional[str] = None) -> List[str]:
        """データソース優先順序取得
        
        Args:
            preferred_source: 優先データソース名
            
        Returns:
            優先順序でソートされたソース名のリスト
        """
        enabled_sources = self.config_manager.get_enabled_sources()

        if preferred_source and preferred_source in enabled_sources:
            # 優先ソースを最初に
            order = [preferred_source]
            order.extend([s for s in enabled_sources if s != preferred_source])
            return order

        # 設定ファイルの優先度順でソート
        return self.config_manager.get_sources_by_priority()

    def _update_statistics(self, source_name: str, result: DataFetchResult):
        """統計情報更新
        
        Args:
            source_name: データソース名
            result: データ取得結果
        """
        stats = self.fetch_statistics[source_name]
        stats.requests += 1
        stats.total_time += result.fetch_time

        if result.data is not None:
            stats.successes += 1
            # 移動平均で品質スコア更新
            current_avg = stats.avg_quality
            new_avg = ((current_avg * (stats.successes - 1) + result.quality_score) / 
                      stats.successes)
            stats.avg_quality = new_avg
        else:
            stats.failures += 1

    def _handle_all_failed(self, 
                         symbol: str, 
                         best_result: Optional[DataFetchResult]) -> DataFetchResult:
        """全プロバイダー失敗時の処理
        
        Args:
            symbol: 銘柄コード
            best_result: 最良の結果（品質低）
            
        Returns:
            最終的なデータ取得結果
        """
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

    def get_stock_data_sync(self, 
                          symbol: str, 
                          period: str = "1mo",
                          preferred_source: Optional[str] = None,
                          use_cache: bool = True) -> DataFetchResult:
        """株価データ取得（同期版）
        
        Args:
            symbol: 銘柄コード
            period: 取得期間
            preferred_source: 優先データソース
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

    def get_statistics(self) -> Dict[str, Dict]:
        """統計情報取得
        
        Returns:
            プロバイダー別統計情報
        """
        stats = {}
        for source_name, provider_stats in self.fetch_statistics.items():
            if provider_stats.requests > 0:
                stats[source_name] = {
                    'total_requests': provider_stats.requests,
                    'success_rate': provider_stats.success_rate,
                    'failure_rate': provider_stats.failure_rate,
                    'avg_response_time': provider_stats.avg_response_time,
                    'avg_quality_score': provider_stats.avg_quality
                }

        return stats

    def get_source_status(self) -> Dict[str, Dict]:
        """データソース状態取得
        
        Returns:
            ソース別状態情報
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
                'requests_remaining': max(
                    0, config.rate_limit_per_day - provider.daily_request_count
                ),
                'success_rate': stats.success_rate,
                'avg_quality': stats.avg_quality
            }

        return status

    def enable_source(self, source_name: str):
        """データソース有効化
        
        Args:
            source_name: データソース名
        """
        self.config_manager.enable_source(source_name)

        # プロバイダーを再初期化
        if source_name not in self.providers:
            self.providers = self._initialize_providers()

    def disable_source(self, source_name: str):
        """データソース無効化
        
        Args:
            source_name: データソース名
        """
        self.config_manager.disable_source(source_name)

        # プロバイダーから削除
        if source_name in self.providers:
            del self.providers[source_name]

    def clear_cache(self, symbol: Optional[str] = None):
        """キャッシュクリア
        
        Args:
            symbol: 特定銘柄のクリア（Noneの場合は全クリア）
        """
        self.cache_manager.clear_cache(symbol)

    def reset_statistics(self):
        """統計情報リセット"""
        self.fetch_statistics.clear()
        self.logger.info("Reset provider statistics")

    def get_provider_info(self) -> Dict[str, Dict]:
        """プロバイダー情報取得
        
        Returns:
            プロバイダー情報
        """
        info = {}
        for source_name, provider in self.providers.items():
            config = self.config_manager.get_config(source_name)
            info[source_name] = {
                'class_name': provider.__class__.__name__,
                'enabled': config.enabled,
                'priority': config.priority,
                'timeout': config.timeout,
                'quality_threshold': config.quality_threshold,
                'cache_enabled': config.cache_enabled
            }
        
        return info