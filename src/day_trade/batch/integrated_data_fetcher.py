#!/usr/bin/env python3
"""
統合データフェッチャー
APIリクエスト統合システムを使用した高性能データ取得システム

主要機能:
- 複数データソースの統合
- 自動的なAPIリクエスト統合
- キャッシュ最適化
- 失敗時の自動フォールバック
- リアルタイム性能監視
"""

import asyncio
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..data.batch_data_fetcher import (
    AdvancedBatchDataFetcher,
    DataRequest,
    DataResponse,
)
from ..data.stock_fetcher import StockDataFetcher
from ..utils.logging_config import get_context_logger
from .api_request_consolidator import (
    APIRequest,
    APIRequestConsolidator,
    APIResponse,
    RequestPriority,
)

logger = get_context_logger(__name__)


class DataSource(Enum):
    """データソース"""

    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    LOCAL_DB = "local_db"
    CACHE = "cache"


@dataclass
class IntegratedDataRequest:
    """統合データリクエスト"""

    symbols: List[str]
    period: str = "60d"
    interval: str = "1d"
    data_sources: List[DataSource] = field(
        default_factory=lambda: [DataSource.YFINANCE]
    )
    priority: RequestPriority = RequestPriority.NORMAL
    enable_fallback: bool = True
    max_age_seconds: int = 3600  # キャッシュ最大経過時間
    quality_threshold: float = 0.8  # データ品質閾値
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedDataResponse:
    """統合データレスポンス"""

    symbols: List[str]
    data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    success_count: int = 0
    failed_symbols: List[str] = field(default_factory=list)
    data_sources_used: Dict[str, List[str]] = field(default_factory=dict)
    total_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    average_data_quality: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntegratedDataFetcher:
    """
    統合データフェッチャー

    複数のデータソースから効率的にデータを取得し、
    APIリクエスト統合システムを使用してパフォーマンスを最適化
    """

    def __init__(
        self,
        consolidator_config: Dict[str, Any] = None,
        enable_advanced_batch_fetcher: bool = True,
        fallback_timeout: float = 10.0,
        max_concurrent_requests: int = 50,
    ):
        # APIリクエスト統合システム
        consolidator_config = consolidator_config or {}
        self.consolidator = APIRequestConsolidator(**consolidator_config)

        # 従来のデータフェッチャー
        self.stock_fetcher = StockDataFetcher()

        self.advanced_batch_fetcher = None
        if enable_advanced_batch_fetcher:
            try:
                self.advanced_batch_fetcher = AdvancedBatchDataFetcher(
                    max_workers=4, enable_kafka=False, enable_redis=False
                )
                logger.info("高度バッチフェッチャー有効化")
            except Exception as e:
                logger.warning(f"高度バッチフェッチャー初期化失敗: {e}")

        # 設定
        self.fallback_timeout = fallback_timeout
        self.max_concurrent_requests = max_concurrent_requests

        # パフォーマンス統計
        self.total_requests = 0
        self.successful_requests = 0
        self.cache_hits = 0
        self.fallback_uses = 0
        self.average_response_time = 0.0

        # 並行処理管理
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.active_requests = {}  # request_id -> Future

        logger.info("統合データフェッチャー初期化完了")

    def start(self):
        """システム開始"""
        self.consolidator.start()
        logger.info("統合データフェッチャー開始")

    def stop(self):
        """システム停止"""
        self.consolidator.stop()
        self.executor.shutdown(wait=True)
        if self.advanced_batch_fetcher:
            self.advanced_batch_fetcher.close()
        logger.info("統合データフェッチャー停止")

    def fetch_data(
        self,
        symbols: List[str],
        period: str = "60d",
        interval: str = "1d",
        data_sources: List[DataSource] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        enable_fallback: bool = True,
        timeout: float = 30.0,
    ) -> IntegratedDataResponse:
        """
        統合データ取得

        Args:
            symbols: 銘柄コードリスト
            period: データ期間
            interval: データ間隔
            data_sources: データソースリスト
            priority: 優先度
            enable_fallback: フォールバック有効化
            timeout: タイムアウト

        Returns:
            統合データレスポンス
        """
        start_time = time.time()
        self.total_requests += 1

        if data_sources is None:
            data_sources = [DataSource.YFINANCE]

        request = IntegratedDataRequest(
            symbols=symbols,
            period=period,
            interval=interval,
            data_sources=data_sources,
            priority=priority,
            enable_fallback=enable_fallback,
        )

        try:
            # プライマリー データ取得
            response = self._fetch_primary_data(request, timeout)

            # フォールバック処理
            if enable_fallback and response.success_count < len(symbols):
                fallback_response = self._fetch_fallback_data(
                    request, response, timeout / 2
                )
                response = self._merge_responses(response, fallback_response)

            # 統計更新
            response.total_processing_time = time.time() - start_time
            self._update_stats(response)

            logger.info(
                f"統合データ取得完了: {response.success_count}/{len(symbols)} symbols, "
                f"time={response.total_processing_time:.2f}s, "
                f"cache_hit_rate={response.cache_hit_rate:.1%}"
            )

            return response

        except Exception as e:
            logger.error(f"統合データ取得エラー: {e}")
            return IntegratedDataResponse(
                symbols=symbols,
                failed_symbols=symbols,
                total_processing_time=time.time() - start_time,
            )

    def fetch_data_async(
        self, symbols: List[str], callback: Optional[callable] = None, **kwargs
    ) -> str:
        """
        非同期データ取得

        Args:
            symbols: 銘柄コードリスト
            callback: 完了時コールバック
            **kwargs: その他のパラメータ

        Returns:
            リクエストID
        """
        request_id = f"integrated_{int(time.time() * 1000)}_{len(self.active_requests)}"

        # 非同期処理投入
        future = self.executor.submit(self.fetch_data, symbols, **kwargs)
        self.active_requests[request_id] = future

        # コールバック設定
        if callback:

            def handle_completion(f):
                try:
                    result = f.result()
                    callback(request_id, result)
                except Exception as e:
                    logger.error(f"非同期処理エラー {request_id}: {e}")
                finally:
                    if request_id in self.active_requests:
                        del self.active_requests[request_id]

            future.add_done_callback(handle_completion)

        logger.debug(f"非同期データ取得開始: {request_id} - {len(symbols)} symbols")
        return request_id

    def _fetch_primary_data(
        self, request: IntegratedDataRequest, timeout: float
    ) -> IntegratedDataResponse:
        """プライマリーデータ取得"""
        response = IntegratedDataResponse(symbols=request.symbols)

        for data_source in request.data_sources:
            if data_source == DataSource.YFINANCE:
                # APIリクエスト統合システム使用
                primary_response = self._fetch_via_consolidator(request, timeout)
                response = self._merge_responses(response, primary_response)

            elif data_source == DataSource.LOCAL_DB and self.advanced_batch_fetcher:
                # 高度バッチフェッチャー使用
                batch_response = self._fetch_via_advanced_batch(request)
                response = self._merge_responses(response, batch_response)

            # 他のデータソースも同様に実装可能

            # 十分なデータが取得できた場合は終了
            if response.success_count >= len(request.symbols) * 0.9:
                break

        return response

    def _fetch_via_consolidator(
        self, request: IntegratedDataRequest, timeout: float
    ) -> IntegratedDataResponse:
        """統合システム経由でデータ取得"""
        response = IntegratedDataResponse(symbols=request.symbols)
        response_data = {}
        completed_requests = 0
        total_requests = 0
        cache_hits = 0

        # 複数のAPIエンドポイントにリクエスト分散
        endpoints = ["stock_data", "price_data", "volume_data"]

        for endpoint in endpoints:
            # APIリクエスト統合システムにリクエスト投入
            result_container = {"data": None, "completed": False}

            def response_callback(api_response: APIResponse):
                result_container["data"] = api_response
                result_container["completed"] = True
                nonlocal completed_requests, cache_hits
                completed_requests += 1
                if api_response.cache_hit:
                    cache_hits += 1

            request_id = self.consolidator.submit_request(
                endpoint=endpoint,
                symbols=request.symbols,
                parameters={"period": request.period, "interval": request.interval},
                priority=request.priority,
                timeout=timeout,
                callback=response_callback,
            )

            total_requests += 1

            # 結果待機
            start_wait = time.time()
            while (
                not result_container["completed"]
                and (time.time() - start_wait) < timeout
            ):
                time.sleep(0.01)

            # データ処理
            if result_container["completed"] and result_container["data"].success:
                api_data = result_container["data"].data
                if api_data:
                    response_data.update(api_data)

        # データフレーム変換
        for symbol, data_dict in response_data.items():
            if data_dict and isinstance(data_dict, dict):
                # ダミーデータフレーム作成（実際の実装では適切な変換を行う）
                df = pd.DataFrame([data_dict])
                response.data[symbol] = df
                response.success_count += 1

                if symbol not in response.data_sources_used:
                    response.data_sources_used[symbol] = []
                response.data_sources_used[symbol].append("yfinance_consolidated")

        # 失敗した銘柄
        response.failed_symbols = [s for s in request.symbols if s not in response.data]

        # キャッシュヒット率
        response.cache_hit_rate = (
            cache_hits / total_requests if total_requests > 0 else 0.0
        )

        return response

    def _fetch_via_advanced_batch(
        self, request: IntegratedDataRequest
    ) -> IntegratedDataResponse:
        """高度バッチフェッチャー経由でデータ取得"""
        if not self.advanced_batch_fetcher:
            return IntegratedDataResponse(symbols=request.symbols)

        # DataRequestsに変換
        batch_requests = [
            DataRequest(
                symbol=symbol,
                period=request.period,
                preprocessing=True,
                priority=5 if request.priority == RequestPriority.HIGH else 3,
            )
            for symbol in request.symbols
        ]

        # バッチ取得実行
        try:
            batch_responses = self.advanced_batch_fetcher.fetch_batch(
                batch_requests, use_parallel=True
            )

            # IntegratedDataResponseに変換
            response = IntegratedDataResponse(symbols=request.symbols)
            cache_hits = 0

            for symbol, data_response in batch_responses.items():
                if data_response.success and data_response.data is not None:
                    response.data[symbol] = data_response.data
                    response.success_count += 1

                    if symbol not in response.data_sources_used:
                        response.data_sources_used[symbol] = []
                    response.data_sources_used[symbol].append("advanced_batch")

                    if data_response.cache_hit:
                        cache_hits += 1
                else:
                    response.failed_symbols.append(symbol)

            response.cache_hit_rate = (
                cache_hits / len(batch_responses) if batch_responses else 0.0
            )

            # データ品質平均
            quality_scores = [
                r.data_quality_score for r in batch_responses.values() if r.success
            ]
            response.average_data_quality = (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            )

            return response

        except Exception as e:
            logger.error(f"高度バッチフェッチャーエラー: {e}")
            return IntegratedDataResponse(
                symbols=request.symbols, failed_symbols=request.symbols
            )

    def _fetch_fallback_data(
        self,
        request: IntegratedDataRequest,
        primary_response: IntegratedDataResponse,
        timeout: float,
    ) -> IntegratedDataResponse:
        """フォールバックデータ取得"""
        failed_symbols = primary_response.failed_symbols
        if not failed_symbols:
            return IntegratedDataResponse(symbols=[])

        logger.info(f"フォールバック処理開始: {len(failed_symbols)} symbols")
        self.fallback_uses += 1

        # 従来のStockDataFetcherを使用
        response = IntegratedDataResponse(symbols=failed_symbols)

        try:
            # バルク取得実行
            bulk_results = self.stock_fetcher.bulk_get_current_prices_optimized(
                codes=failed_symbols, batch_size=20, delay=0.05
            )

            for symbol, price_data in bulk_results.items():
                if price_data:
                    # 簡易データフレーム作成
                    df = pd.DataFrame(
                        [
                            {
                                "Close": price_data.get("price", 0.0),
                                "Volume": price_data.get("volume", 0),
                                "Timestamp": time.time(),
                            }
                        ]
                    )

                    response.data[symbol] = df
                    response.success_count += 1

                    if symbol not in response.data_sources_used:
                        response.data_sources_used[symbol] = []
                    response.data_sources_used[symbol].append("fallback_yfinance")

            response.failed_symbols = [
                s for s in failed_symbols if s not in response.data
            ]

        except Exception as e:
            logger.error(f"フォールバック処理エラー: {e}")
            response.failed_symbols = failed_symbols

        return response

    def _merge_responses(
        self, primary: IntegratedDataResponse, secondary: IntegratedDataResponse
    ) -> IntegratedDataResponse:
        """レスポンス統合"""
        merged = IntegratedDataResponse(
            symbols=list(set(primary.symbols + secondary.symbols))
        )

        # データ統合
        merged.data.update(primary.data)
        merged.data.update(secondary.data)  # 後優先

        # 統計統合
        merged.success_count = len(merged.data)
        merged.failed_symbols = [s for s in merged.symbols if s not in merged.data]

        # データソース情報統合
        merged.data_sources_used.update(primary.data_sources_used)
        for symbol, sources in secondary.data_sources_used.items():
            if symbol not in merged.data_sources_used:
                merged.data_sources_used[symbol] = []
            merged.data_sources_used[symbol].extend(sources)

        # キャッシュヒット率統合（重み付き平均）
        primary_weight = len(primary.data) if primary.data else 0
        secondary_weight = len(secondary.data) if secondary.data else 0
        total_weight = primary_weight + secondary_weight

        if total_weight > 0:
            merged.cache_hit_rate = (
                primary.cache_hit_rate * primary_weight
                + secondary.cache_hit_rate * secondary_weight
            ) / total_weight

        # データ品質統合
        if primary.average_data_quality > 0 and secondary.average_data_quality > 0:
            merged.average_data_quality = (
                (
                    primary.average_data_quality * primary_weight
                    + secondary.average_data_quality * secondary_weight
                )
                / total_weight
                if total_weight > 0
                else 0.0
            )
        elif primary.average_data_quality > 0:
            merged.average_data_quality = primary.average_data_quality
        elif secondary.average_data_quality > 0:
            merged.average_data_quality = secondary.average_data_quality

        return merged

    def _update_stats(self, response: IntegratedDataResponse):
        """統計更新"""
        if response.success_count > 0:
            self.successful_requests += 1

        if response.cache_hit_rate > 0:
            self.cache_hits += int(response.cache_hit_rate * len(response.symbols))

        # 平均レスポンス時間更新
        if response.total_processing_time > 0:
            total_time = self.average_response_time * (self.total_requests - 1)
            self.average_response_time = (
                total_time + response.total_processing_time
            ) / self.total_requests

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        consolidator_stats = self.consolidator.get_stats()

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": (
                self.successful_requests / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": (
                self.cache_hits / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
            "fallback_uses": self.fallback_uses,
            "fallback_rate": (
                self.fallback_uses / self.total_requests
                if self.total_requests > 0
                else 0.0
            ),
            "average_response_time": self.average_response_time,
            "active_requests": len(self.active_requests),
            "consolidator_stats": consolidator_stats.__dict__,
            "consolidator_status": self.consolidator.get_status(),
        }

    def clear_caches(self) -> Dict[str, Any]:
        """全キャッシュクリア"""
        cleared = {}

        # 統合システムのキャッシュクリア
        if hasattr(self.consolidator, "cache") and self.consolidator.cache:
            try:
                cleared["consolidator"] = self.consolidator.cache.clear()
            except Exception as e:
                logger.error(f"統合システムキャッシュクリアエラー: {e}")
                cleared["consolidator"] = 0

        # 高度バッチフェッチャーのキャッシュクリア
        if self.advanced_batch_fetcher:
            try:
                cleared.update(self.advanced_batch_fetcher.clear_cache())
            except Exception as e:
                logger.error(f"高度バッチフェッチャーキャッシュクリアエラー: {e}")

        # 従来フェッチャーのキャッシュクリア
        if hasattr(self.stock_fetcher, "memory_cache"):
            cleared["stock_fetcher"] = len(self.stock_fetcher.memory_cache)
            self.stock_fetcher.memory_cache.clear()

        logger.info(f"キャッシュクリア完了: {cleared}")
        return cleared


# 便利関数
def create_integrated_fetcher(
    batch_size: int = 50, max_workers: int = 6, enable_caching: bool = True, **kwargs
) -> IntegratedDataFetcher:
    """統合データフェッチャー作成"""
    consolidator_config = {
        "base_batch_size": batch_size,
        "max_workers": max_workers,
        "enable_caching": enable_caching,
        **kwargs,
    }

    return IntegratedDataFetcher(
        consolidator_config=consolidator_config, enable_advanced_batch_fetcher=True
    )


if __name__ == "__main__":
    # テスト実行
    print("=== 統合データフェッチャー テスト ===")

    fetcher = create_integrated_fetcher(batch_size=30, max_workers=4)
    fetcher.start()

    try:
        # テストデータ取得
        test_symbols = ["7203", "8306", "9984", "6758", "4689"]

        print(f"データ取得開始: {test_symbols}")

        # 同期取得
        response = fetcher.fetch_data(
            symbols=test_symbols,
            period="5d",
            priority=RequestPriority.HIGH,
            enable_fallback=True,
        )

        print("\n結果:")
        print(f"  成功: {response.success_count}/{len(test_symbols)}")
        print(f"  失敗銘柄: {response.failed_symbols}")
        print(f"  処理時間: {response.total_processing_time:.2f}秒")
        print(f"  キャッシュヒット率: {response.cache_hit_rate:.1%}")
        print(f"  データ品質: {response.average_data_quality:.1f}")
        print(f"  データソース: {response.data_sources_used}")

        # パフォーマンス統計
        stats = fetcher.get_performance_stats()
        print("\nパフォーマンス統計:")
        print(f"  総リクエスト: {stats['total_requests']}")
        print(f"  成功率: {stats['success_rate']:.1%}")
        print(f"  キャッシュヒット率: {stats['cache_hit_rate']:.1%}")
        print(f"  フォールバック使用率: {stats['fallback_rate']:.1%}")
        print(f"  平均レスポンス時間: {stats['average_response_time']:.3f}秒")

    finally:
        fetcher.stop()

    print("\n=== テスト完了 ===")
