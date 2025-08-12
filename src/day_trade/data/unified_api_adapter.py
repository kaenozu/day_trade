#!/usr/bin/env python3
"""
統一APIアダプター
Issue #376: バッチ処理の強化 - APIリクエスト統合

複数のAPIリクエストを統合し、効率的な一括処理を実現
"""

import asyncio
import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import aiohttp
import yfinance as yf

# プロジェクトモジュール
try:
    from ..utils.cache_utils import generate_safe_cache_key
    from ..utils.exceptions import APIError, NetworkError, ValidationError
    from ..utils.logging_config import get_context_logger, log_performance_metric
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    def log_performance_metric(*args, **kwargs):
        pass

    def generate_safe_cache_key(*args, **kwargs):
        return str(hash(str(args) + str(kwargs)))

    class APIError(Exception):
        pass

    class NetworkError(Exception):
        pass

    class ValidationError(Exception):
        pass


logger = get_context_logger(__name__)


class APIProvider(Enum):
    """APIプロバイダー"""

    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    CUSTOM = "custom"


class RequestMethod(Enum):
    """リクエストメソッド"""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass
class APIRequest:
    """APIリクエスト定義"""

    provider: APIProvider
    endpoint: str
    method: RequestMethod = RequestMethod.GET
    parameters: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    priority: int = 1
    cache_ttl: int = 300  # 5分
    batch_key: Optional[str] = None  # バッチ統合用キー
    created_at: float = field(default_factory=time.time)
    request_id: str = field(default_factory=lambda: f"req_{int(time.time() * 1000)}")

    def get_cache_key(self) -> str:
        """キャッシュキー生成"""
        key_data = f"{self.provider.value}:{self.endpoint}:{self.method.value}:{json.dumps(self.parameters, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]


@dataclass
class APIResponse:
    """APIレスポンス"""

    request_id: str
    success: bool
    data: Any
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    response_time_ms: float = 0.0
    from_cache: bool = False
    timestamp: float = field(default_factory=time.time)


class APIBatch:
    """API一括処理バッチ"""

    def __init__(self, batch_key: str, max_size: int = 100):
        self.batch_key = batch_key
        self.max_size = max_size
        self.requests: List[APIRequest] = []
        self.created_at = time.time()
        self.timeout_seconds = 60

    def add_request(self, request: APIRequest) -> bool:
        """リクエスト追加"""
        if len(self.requests) >= self.max_size:
            return False

        self.requests.append(request)
        return True

    def is_ready(self) -> bool:
        """処理準備完了判定"""
        return (
            len(self.requests) >= 10 or time.time() - self.created_at > 5  # 10個以上
        )  # または5秒経過

    def is_expired(self) -> bool:
        """期限切れ判定"""
        return time.time() - self.created_at > self.timeout_seconds


class APIAdapter(ABC):
    """APIアダプター基底クラス"""

    @abstractmethod
    async def execute_request(self, request: APIRequest) -> APIResponse:
        """単一リクエスト実行"""
        pass

    @abstractmethod
    async def execute_batch(self, batch: APIBatch) -> List[APIResponse]:
        """バッチリクエスト実行"""
        pass

    @abstractmethod
    def can_batch(self, requests: List[APIRequest]) -> bool:
        """バッチ処理可能性判定"""
        pass


class YFinanceAdapter(APIAdapter):
    """Yahoo Finance APIアダプター"""

    def __init__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        self.rate_limiter = asyncio.Semaphore(10)  # 同時10リクエスト制限

    async def execute_request(self, request: APIRequest) -> APIResponse:
        """単一リクエスト実行"""
        start_time = time.time()

        async with self.rate_limiter:
            try:
                if request.endpoint == "current_price":
                    symbol = request.parameters.get("symbol")
                    if not symbol:
                        raise ValidationError("symbol parameter required")

                    # yfinanceを使用した価格取得
                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    response_data = {
                        "symbol": symbol,
                        "price": info.get("regularMarketPrice", info.get("ask", 0)),
                        "change": info.get("regularMarketChange", 0),
                        "change_percent": info.get("regularMarketChangePercent", 0),
                        "volume": info.get("regularMarketVolume", 0),
                        "timestamp": time.time(),
                    }

                elif request.endpoint == "historical_data":
                    symbol = request.parameters.get("symbol")
                    period = request.parameters.get("period", "1mo")
                    interval = request.parameters.get("interval", "1d")

                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period, interval=interval)

                    response_data = {
                        "symbol": symbol,
                        "data": hist.to_dict("records") if not hist.empty else [],
                        "period": period,
                        "interval": interval,
                    }

                else:
                    raise ValidationError(f"Unsupported endpoint: {request.endpoint}")

                response_time_ms = (time.time() - start_time) * 1000

                return APIResponse(
                    request_id=request.request_id,
                    success=True,
                    data=response_data,
                    status_code=200,
                    response_time_ms=response_time_ms,
                )

            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                logger.error(f"YFinance API error ({request.request_id}): {e}")

                return APIResponse(
                    request_id=request.request_id,
                    success=False,
                    data=None,
                    error_message=str(e),
                    response_time_ms=response_time_ms,
                )

    async def execute_batch(self, batch: APIBatch) -> List[APIResponse]:
        """バッチリクエスト実行"""
        if not batch.requests:
            return []

        # エンドポイント別にグループ化
        price_requests = []
        historical_requests = []

        for req in batch.requests:
            if req.endpoint == "current_price":
                price_requests.append(req)
            elif req.endpoint == "historical_data":
                historical_requests.append(req)

        responses = []

        # 価格取得バッチ処理
        if price_requests:
            symbols = [req.parameters.get("symbol") for req in price_requests]
            batch_responses = await self._batch_price_fetch(symbols, price_requests)
            responses.extend(batch_responses)

        # ヒストリカルデータは個別処理（YFinanceの制限）
        if historical_requests:
            tasks = [self.execute_request(req) for req in historical_requests]
            individual_responses = await asyncio.gather(*tasks, return_exceptions=True)

            for resp in individual_responses:
                if isinstance(resp, APIResponse):
                    responses.append(resp)
                else:
                    # 例外が発生した場合のエラーレスポンス
                    responses.append(
                        APIResponse(
                            request_id="unknown",
                            success=False,
                            data=None,
                            error_message=str(resp),
                        )
                    )

        return responses

    async def _batch_price_fetch(
        self, symbols: List[str], requests: List[APIRequest]
    ) -> List[APIResponse]:
        """価格一括取得"""
        start_time = time.time()

        try:
            # yfinanceで一括取得
            tickers = yf.Tickers(" ".join(symbols))
            responses = []

            for i, symbol in enumerate(symbols):
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info

                    response_data = {
                        "symbol": symbol,
                        "price": info.get("regularMarketPrice", info.get("ask", 0)),
                        "change": info.get("regularMarketChange", 0),
                        "change_percent": info.get("regularMarketChangePercent", 0),
                        "volume": info.get("regularMarketVolume", 0),
                        "timestamp": time.time(),
                    }

                    responses.append(
                        APIResponse(
                            request_id=requests[i].request_id,
                            success=True,
                            data=response_data,
                            status_code=200,
                            response_time_ms=(time.time() - start_time) * 1000,
                        )
                    )

                except Exception as e:
                    responses.append(
                        APIResponse(
                            request_id=requests[i].request_id,
                            success=False,
                            data=None,
                            error_message=str(e),
                            response_time_ms=(time.time() - start_time) * 1000,
                        )
                    )

            return responses

        except Exception as e:
            # 全体的なエラー
            return [
                APIResponse(
                    request_id=req.request_id,
                    success=False,
                    data=None,
                    error_message=str(e),
                )
                for req in requests
            ]

    def can_batch(self, requests: List[APIRequest]) -> bool:
        """バッチ処理可能性判定"""
        # 同一エンドポイントの価格取得はバッチ可能
        endpoints = set(req.endpoint for req in requests)
        return len(endpoints) == 1 and "current_price" in endpoints


class GenericHTTPAdapter(APIAdapter):
    """汎用HTTPアダプター"""

    def __init__(self, base_url: str = "", default_headers: Dict[str, str] = None):
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30), headers=self.default_headers
        )

    async def execute_request(self, request: APIRequest) -> APIResponse:
        """単一リクエスト実行"""
        start_time = time.time()

        try:
            url = urljoin(self.base_url, request.endpoint)
            headers = {**self.default_headers, **request.headers}

            async with self.session.request(
                request.method.value,
                url,
                params=request.parameters if request.method == RequestMethod.GET else None,
                json=(
                    request.parameters
                    if request.method in [RequestMethod.POST, RequestMethod.PUT]
                    else None
                ),
                headers=headers,
                timeout=request.timeout,
            ) as response:
                response_data = (
                    await response.json()
                    if response.content_type == "application/json"
                    else await response.text()
                )

                response_time_ms = (time.time() - start_time) * 1000

                return APIResponse(
                    request_id=request.request_id,
                    success=response.status < 400,
                    data=response_data,
                    status_code=response.status,
                    response_time_ms=response_time_ms,
                )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            return APIResponse(
                request_id=request.request_id,
                success=False,
                data=None,
                error_message=str(e),
                response_time_ms=response_time_ms,
            )

    async def execute_batch(self, batch: APIBatch) -> List[APIResponse]:
        """バッチリクエスト実行（並列実行）"""
        tasks = [self.execute_request(req) for req in batch.requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        result = []
        for resp in responses:
            if isinstance(resp, APIResponse):
                result.append(resp)
            else:
                result.append(
                    APIResponse(
                        request_id="unknown",
                        success=False,
                        data=None,
                        error_message=str(resp),
                    )
                )

        return result

    def can_batch(self, requests: List[APIRequest]) -> bool:
        """バッチ処理可能性判定（常に可能）"""
        return True


class UnifiedAPIAdapter:
    """統一APIアダプター"""

    def __init__(
        self,
        enable_caching: bool = True,
        max_batch_size: int = 100,
        batch_timeout_seconds: int = 5,
    ):
        """
        初期化

        Args:
            enable_caching: キャッシュ有効化
            max_batch_size: 最大バッチサイズ
            batch_timeout_seconds: バッチタイムアウト
        """
        self.enable_caching = enable_caching
        self.max_batch_size = max_batch_size
        self.batch_timeout_seconds = batch_timeout_seconds

        # アダプターマップ
        self.adapters = {
            APIProvider.YFINANCE: YFinanceAdapter(),
            APIProvider.CUSTOM: GenericHTTPAdapter(),
        }

        # キャッシュ
        self.cache = {} if enable_caching else None
        self.cache_lock = threading.RLock()

        # バッチ管理
        self.pending_batches: Dict[str, APIBatch] = {}
        self.batch_lock = threading.RLock()

        # 統計情報
        self.stats = {
            "total_requests": 0,
            "cached_responses": 0,
            "batched_requests": 0,
            "average_response_time_ms": 0.0,
            "error_rate": 0.0,
        }
        self.stats_lock = threading.RLock()

        # バッチ処理タスク
        self._batch_task = None
        self._running = True

        logger.info(
            f"UnifiedAPIAdapter初期化完了: caching={enable_caching}, " f"max_batch={max_batch_size}"
        )

    async def execute(self, request: APIRequest) -> APIResponse:
        """リクエスト実行"""
        # 統計更新
        with self.stats_lock:
            self.stats["total_requests"] += 1

        # キャッシュ確認
        if self.enable_caching:
            cached_response = self._get_cached_response(request)
            if cached_response:
                with self.stats_lock:
                    self.stats["cached_responses"] += 1
                return cached_response

        # アダプター取得
        adapter = self.adapters.get(request.provider)
        if not adapter:
            return APIResponse(
                request_id=request.request_id,
                success=False,
                data=None,
                error_message=f"Unsupported provider: {request.provider}",
            )

        # バッチ処理判定
        if request.batch_key and adapter.can_batch([request]):
            return await self._handle_batch_request(request, adapter)
        else:
            response = await adapter.execute_request(request)

            # キャッシュ保存
            if self.enable_caching and response.success:
                self._cache_response(request, response)

            # 統計更新
            self._update_stats(response)

            return response

    async def execute_multiple(self, requests: List[APIRequest]) -> List[APIResponse]:
        """複数リクエスト実行"""
        if not requests:
            return []

        # プロバイダー別にグループ化
        provider_groups = defaultdict(list)
        for req in requests:
            provider_groups[req.provider].append(req)

        all_responses = []

        # プロバイダーごとに処理
        for provider, provider_requests in provider_groups.items():
            adapter = self.adapters.get(provider)
            if not adapter:
                # 未サポートプロバイダーエラー
                error_responses = [
                    APIResponse(
                        request_id=req.request_id,
                        success=False,
                        data=None,
                        error_message=f"Unsupported provider: {provider}",
                    )
                    for req in provider_requests
                ]
                all_responses.extend(error_responses)
                continue

            # バッチ処理可能性確認
            if adapter.can_batch(provider_requests):
                # バッチ作成と実行
                batch = APIBatch("multi_batch", len(provider_requests))
                for req in provider_requests:
                    batch.add_request(req)

                batch_responses = await adapter.execute_batch(batch)
                all_responses.extend(batch_responses)

                with self.stats_lock:
                    self.stats["batched_requests"] += len(provider_requests)
            else:
                # 個別実行
                tasks = [adapter.execute_request(req) for req in provider_requests]
                individual_responses = await asyncio.gather(*tasks, return_exceptions=True)

                for resp in individual_responses:
                    if isinstance(resp, APIResponse):
                        all_responses.append(resp)
                        # 統計更新
                        self._update_stats(resp)

        return all_responses

    async def _handle_batch_request(self, request: APIRequest, adapter: APIAdapter) -> APIResponse:
        """バッチリクエスト処理"""
        with self.batch_lock:
            batch_key = request.batch_key

            # 既存バッチを探す
            batch = self.pending_batches.get(batch_key)
            if not batch:
                batch = APIBatch(batch_key, self.max_batch_size)
                self.pending_batches[batch_key] = batch

            # リクエスト追加
            if not batch.add_request(request):
                # バッチが満杯の場合は新しいバッチを作成
                batch = APIBatch(f"{batch_key}_{int(time.time())}", self.max_batch_size)
                batch.add_request(request)
                self.pending_batches[f"{batch_key}_{int(time.time())}"] = batch

        # バッチ処理準備完了まで待機
        timeout = self.batch_timeout_seconds
        start_time = time.time()

        while time.time() - start_time < timeout:
            if batch.is_ready():
                break
            await asyncio.sleep(0.1)

        # バッチ実行
        with self.batch_lock:
            if batch_key in self.pending_batches:
                del self.pending_batches[batch_key]

        batch_responses = await adapter.execute_batch(batch)

        # 対応するレスポンスを検索
        for response in batch_responses:
            if response.request_id == request.request_id:
                # キャッシュ保存
                if self.enable_caching and response.success:
                    self._cache_response(request, response)

                # 統計更新
                self._update_stats(response)
                with self.stats_lock:
                    self.stats["batched_requests"] += 1

                return response

        # レスポンスが見つからない場合
        return APIResponse(
            request_id=request.request_id,
            success=False,
            data=None,
            error_message="Batch response not found",
        )

    def _get_cached_response(self, request: APIRequest) -> Optional[APIResponse]:
        """キャッシュレスポンス取得"""
        if not self.cache:
            return None

        cache_key = request.get_cache_key()

        with self.cache_lock:
            if cache_key in self.cache:
                cached_response, cached_at = self.cache[cache_key]

                # TTL チェック
                if time.time() - cached_at < request.cache_ttl:
                    # キャッシュヒット
                    cached_response.from_cache = True
                    return cached_response
                else:
                    # 期限切れ削除
                    del self.cache[cache_key]

        return None

    def _cache_response(self, request: APIRequest, response: APIResponse):
        """レスポンスキャッシュ"""
        if not self.cache:
            return

        cache_key = request.get_cache_key()

        with self.cache_lock:
            self.cache[cache_key] = (response, time.time())

    def _update_stats(self, response: APIResponse):
        """統計更新"""
        with self.stats_lock:
            # 移動平均でレスポンス時間更新
            alpha = 0.1
            self.stats["average_response_time_ms"] = (
                self.stats["average_response_time_ms"] * (1 - alpha)
                + response.response_time_ms * alpha
            )

            # エラー率更新
            if not response.success:
                current_error_rate = self.stats.get("error_rate", 0)
                self.stats["error_rate"] = current_error_rate * (1 - alpha) + 1.0 * alpha
            else:
                current_error_rate = self.stats.get("error_rate", 0)
                self.stats["error_rate"] = current_error_rate * (1 - alpha)

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self.stats_lock:
            stats = self.stats.copy()

        with self.batch_lock:
            stats["pending_batches"] = len(self.pending_batches)

        if self.cache:
            with self.cache_lock:
                stats["cache_size"] = len(self.cache)
                stats["cache_hit_rate"] = self.stats["cached_responses"] / max(
                    self.stats["total_requests"], 1
                )

        return stats

    def clear_cache(self):
        """キャッシュクリア"""
        if self.cache:
            with self.cache_lock:
                self.cache.clear()
                logger.info("APIキャッシュクリア完了")

    async def cleanup(self):
        """クリーンアップ"""
        self._running = False

        # アダプタークリーンアップ
        for adapter in self.adapters.values():
            if hasattr(adapter, "session"):
                await adapter.session.close()

        logger.info("UnifiedAPIAdapter クリーンアップ完了")


# 便利関数
async def fetch_stock_prices(
    symbols: List[str], adapter: UnifiedAPIAdapter = None
) -> Dict[str, Any]:
    """株価一括取得"""
    if not adapter:
        adapter = UnifiedAPIAdapter()

    requests = []
    for symbol in symbols:
        requests.append(
            APIRequest(
                provider=APIProvider.YFINANCE,
                endpoint="current_price",
                parameters={"symbol": symbol},
                batch_key="price_batch",
            )
        )

    responses = await adapter.execute_multiple(requests)

    result = {}
    for response in responses:
        if response.success and response.data:
            symbol = response.data.get("symbol")
            if symbol:
                result[symbol] = response.data

    return result


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #376 統一APIアダプターテスト ===")

        adapter = UnifiedAPIAdapter(enable_caching=True)

        try:
            # 価格取得テスト
            print("\n1. 価格一括取得テスト")
            test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
            prices = await fetch_stock_prices(test_symbols, adapter)

            print(f"取得銘柄数: {len(prices)}")
            for symbol, data in prices.items():
                print(f"  {symbol}: ${data.get('price', 0):.2f}")

            # 統計情報
            print("\n2. 統計情報")
            stats = adapter.get_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")

        finally:
            await adapter.cleanup()

    asyncio.run(main())
    print("\n=== 統一APIアダプターテスト完了 ===")
