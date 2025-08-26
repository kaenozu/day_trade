#!/usr/bin/env python3
"""
外部APIクライアント - メインクライアント
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import ClientTimeout

from ...utils.logging_config import get_context_logger
from .auth_manager import AuthenticationManager
from .cache_manager import CacheManager
from .data_normalizers import DataNormalizer
from .enums import APIProvider, DataType
from .error_handlers import ErrorHandler
from .models import APIConfig, APIEndpoint, APIRequest, APIResponse
from .rate_limiter import RateLimiter
from .request_executor import RequestExecutor
from .url_builder import URLBuilder

logger = get_context_logger(__name__)


class ExternalAPIClient:
    """外部APIクライアント（セキュリティ強化版）"""

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.endpoints: Dict[str, APIEndpoint] = {}

        # 各種マネージャーの初期化
        self.auth_manager = AuthenticationManager(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.cache_manager = CacheManager(self.config)
        self.data_normalizer = DataNormalizer()
        self.error_handler = ErrorHandler()
        self.url_builder = URLBuilder(self.config)

        # HTTP セッション
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self.request_executor: Optional[RequestExecutor] = None

        # 統計情報
        self.request_stats: Dict[str, int] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "rate_limited_requests": 0,
            "security_violations": 0,
            "sanitized_errors": 0,
        }

        # デフォルトエンドポイント設定
        self._setup_default_endpoints()

    def _setup_default_endpoints(self) -> None:
        """デフォルトエンドポイント設定"""
        default_endpoints = [
            # Yahoo Finance（無料・公開API）
            APIEndpoint(
                provider=APIProvider.YAHOO_FINANCE,
                data_type=DataType.STOCK_PRICE,
                endpoint_url="https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                rate_limit_per_second=2.0,
                rate_limit_per_minute=100,
                response_format="json",
                data_path="chart.result[0].indicators.quote[0]",
            ),
            # Mock Provider（開発・テスト用）
            APIEndpoint(
                provider=APIProvider.MOCK_PROVIDER,
                data_type=DataType.STOCK_PRICE,
                endpoint_url="https://api.mock-provider.com/v1/stocks/{symbol}/price",
                rate_limit_per_second=10.0,
                rate_limit_per_minute=600,
                max_retries=2,
                timeout_seconds=10,
            ),
            # Alpha Vantage（要APIキー）
            APIEndpoint(
                provider=APIProvider.ALPHA_VANTAGE,
                data_type=DataType.STOCK_PRICE,
                endpoint_url="https://www.alphavantage.co/query",
                requires_auth=True,
                auth_param_name="apikey",
                rate_limit_per_second=0.2,  # 5 calls per minute for free tier
                rate_limit_per_minute=5,
                response_format="json",
            ),
            # IEX Cloud（要APIキー）
            APIEndpoint(
                provider=APIProvider.IEX_CLOUD,
                data_type=DataType.STOCK_PRICE,
                endpoint_url="https://cloud.iexapis.com/stable/stock/{symbol}/quote",
                requires_auth=True,
                auth_param_name="token",
                rate_limit_per_second=10.0,
                rate_limit_per_minute=500,
            ),
        ]

        for endpoint in default_endpoints:
            self.register_endpoint(endpoint)

    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """エンドポイント登録"""
        endpoint_key = f"{endpoint.provider.value}_{endpoint.data_type.value}"
        self.endpoints[endpoint_key] = endpoint

        # レート制限状態初期化
        self.rate_limiter.initialize_provider(endpoint.provider)

        logger.info(f"APIエンドポイント登録: {endpoint_key}")

    async def initialize(self) -> None:
        """クライアント初期化"""
        if self.session is None:
            timeout = ClientTimeout(total=self.config.default_timeout_seconds)
            connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)

            headers = {
                "User-Agent": self.config.user_agent,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            }

            self.session = aiohttp.ClientSession(
                timeout=timeout, connector=connector, headers=headers
            )

            # リクエストエグゼキューターの初期化
            self.request_executor = RequestExecutor(
                self.session,
                self.url_builder,
                self.auth_manager,
                self.data_normalizer,
                self.error_handler
            )

        logger.info("外部APIクライアント初期化完了")

    async def cleanup(self) -> None:
        """クライアントクリーンアップ"""
        if self.session:
            await self.session.close()
            self.session = None
            self.request_executor = None

        logger.info("外部APIクライアントクリーンアップ完了")

    async def fetch_stock_data(
        self, symbol: str, provider: APIProvider = APIProvider.MOCK_PROVIDER, **kwargs
    ) -> Optional[APIResponse]:
        """株価データ取得"""
        endpoint_key = f"{provider.value}_{DataType.STOCK_PRICE.value}"

        if endpoint_key not in self.endpoints:
            logger.error(f"未登録のエンドポイント: {endpoint_key}")
            return None

        endpoint = self.endpoints[endpoint_key]

        # リクエストパラメータ構築
        params = {"symbol": symbol, **kwargs}

        # 認証パラメータ追加
        if endpoint.requires_auth:
            params = self.auth_manager.add_auth_to_params(params, endpoint)

        # APIリクエスト作成
        request = APIRequest(
            endpoint=endpoint,
            params=params,
            request_id=f"stock_{symbol}_{int(time.time())}",
            timestamp=datetime.now(),
        )

        return await self.execute_request(request)

    async def fetch_market_index_data(
        self,
        index_symbol: str,
        provider: APIProvider = APIProvider.MOCK_PROVIDER,
        **kwargs,
    ) -> Optional[APIResponse]:
        """市場指数データ取得"""
        endpoint_key = f"{provider.value}_{DataType.MARKET_INDEX.value}"

        # 株価エンドポイントを代用（多くのAPIで同じエンドポイント）
        if endpoint_key not in self.endpoints:
            endpoint_key = f"{provider.value}_{DataType.STOCK_PRICE.value}"

        if endpoint_key not in self.endpoints:
            logger.error(f"未登録のエンドポイント: {endpoint_key}")
            return None

        endpoint = self.endpoints[endpoint_key]

        params = {"symbol": index_symbol, **kwargs}
        if endpoint.requires_auth:
            params = self.auth_manager.add_auth_to_params(params, endpoint)

        request = APIRequest(
            endpoint=endpoint,
            params=params,
            request_id=f"index_{index_symbol}_{int(time.time())}",
            timestamp=datetime.now(),
        )

        return await self.execute_request(request)

    async def execute_request(self, request: APIRequest) -> Optional[APIResponse]:
        """APIリクエスト実行"""
        if not self.session:
            await self.initialize()

        # 認証要件検証
        if not self.auth_manager.validate_auth_requirements(request.endpoint):
            return None

        # キャッシュチェック
        cache_key = self.cache_manager.generate_cache_key(request)
        cached_response = self.cache_manager.get_cached_response(cache_key)
        if cached_response:
            self.request_stats["cached_responses"] += 1
            return cached_response

        # レート制限チェック
        if not await self.rate_limiter.check_rate_limit(request.endpoint.provider):
            self.request_stats["rate_limited_requests"] += 1
            logger.warning(f"レート制限により遅延: {request.endpoint.provider.value}")

            # レート制限解除まで待機
            await self.rate_limiter.wait_for_rate_limit(request.endpoint.provider)

        # セマフォで同時リクエスト数制御
        async with self.semaphore:
            response = await self.request_executor.execute_with_retry(request)

        # レスポンス統計更新
        self.request_stats["total_requests"] += 1
        if response and response.success:
            self.request_stats["successful_requests"] += 1
        else:
            self.request_stats["failed_requests"] += 1

        # レート制限状態更新
        await self.rate_limiter.update_rate_limit_state(request.endpoint.provider)

        # キャッシュに保存
        if response and response.success and self.config.enable_response_caching:
            self.cache_manager.cache_response(cache_key, response)

        return response

    def get_request_statistics(self) -> Dict[str, Any]:
        """リクエスト統計取得"""
        total = self.request_stats["total_requests"]
        success_rate = (
            (self.request_stats["successful_requests"] / total * 100)
            if total > 0
            else 0
        )
        cache_hit_rate = (
            (self.request_stats["cached_responses"] / total * 100) if total > 0 else 0
        )

        return {
            "total_requests": total,
            "successful_requests": self.request_stats["successful_requests"],
            "failed_requests": self.request_stats["failed_requests"],
            "cached_responses": self.request_stats["cached_responses"],
            "rate_limited_requests": self.request_stats["rate_limited_requests"],
            "success_rate_percent": success_rate,
            "cache_hit_rate_percent": cache_hit_rate,
            "registered_endpoints": len(self.endpoints),
            "active_providers": len(self.rate_limiter.rate_limits),
        }

    def get_rate_limit_status(self) -> Dict[str, Dict[str, Any]]:
        """レート制限状況取得"""
        return self.rate_limiter.get_rate_limit_status()

    async def clear_cache(self) -> None:
        """キャッシュクリア"""
        self.cache_manager.clear_cache()

    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        health_status = {
            "session_active": self.session is not None and not self.session.closed,
            "endpoints_registered": len(self.endpoints),
            "cache_entries": len(self.cache_manager.response_cache),
            "total_requests": self.request_stats["total_requests"],
            "timestamp": datetime.now().isoformat(),
        }

        # 各プロバイダーの簡易テスト
        provider_health = {}
        for provider in APIProvider:
            if provider == APIProvider.MOCK_PROVIDER:
                try:
                    # モックプロバイダーのテスト
                    response = await self.fetch_stock_data("TEST", provider)
                    provider_health[provider.value] = {
                        "status": (
                            "healthy" if response and response.success else "unhealthy"
                        ),
                        "last_test": datetime.now().isoformat(),
                    }
                except Exception as e:
                    provider_health[provider.value] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "last_test": datetime.now().isoformat(),
                    }

        health_status["provider_health"] = provider_health

        return health_status