"""
外部APIクライアント メインクラス
統合APIクライアントとヘルスチェック機能
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp
from aiohttp import ClientError, ClientTimeout

from ...utils.logging_config import get_context_logger
from .auth_manager import AuthenticationManager
from .cache_manager import CacheManager
from .data_normalizers import DataNormalizer
from .enums import APIProvider, DataType, RequestMethod
from .error_handlers import ErrorHandler
from .models import APIConfig, APIEndpoint, APIRequest, APIResponse
from .rate_limiter import RateLimiter
from .url_builder import URLBuilder

logger = get_context_logger(__name__)


class ExternalAPIClient:
    """外部APIクライアント（セキュリティ強化版）"""

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        
        # コンポーネント初期化
        self.auth_manager = AuthenticationManager(self.config)
        self.rate_limiter = RateLimiter()
        self.cache_manager = CacheManager(
            self.config.enable_response_caching,
            self.config.cache_ttl_seconds,
            self.config.cache_size_limit
        )
        self.data_normalizer = DataNormalizer()
        self.error_handler = ErrorHandler()
        self.url_builder = URLBuilder(self.config.secure_url_builder)

        # HTTP セッション
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # 統計情報とセキュリティメトリクス
        self.request_stats: Dict[str, int] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "rate_limited_requests": 0,
            "security_violations": 0,
            "sanitized_errors": 0,
        }

        # デフォルトエンドポイント設定（セキュリティ強化版）
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
        self.rate_limiter.register_endpoint(endpoint)
        logger.info(f"APIエンドポイント登録: {endpoint.provider.value}_{endpoint.data_type.value}")

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

        logger.info("外部APIクライアント初期化完了")

    async def cleanup(self) -> None:
        """クライアントクリーンアップ"""
        if self.session:
            await self.session.close()
            self.session = None

        logger.info("外部APIクライアントクリーンアップ完了")

    async def fetch_stock_data(
        self, symbol: str, provider: APIProvider = APIProvider.MOCK_PROVIDER, **kwargs
    ) -> Optional[APIResponse]:
        """株価データ取得"""
        endpoint_key = f"{provider.value}_{DataType.STOCK_PRICE.value}"

        if endpoint_key not in self.rate_limiter.endpoints:
            logger.error(f"未登録のエンドポイント: {endpoint_key}")
            return None

        endpoint = self.rate_limiter.endpoints[endpoint_key]

        # リクエストパラメータ構築
        params = {"symbol": symbol, **kwargs}
        headers = {}

        # 認証情報を追加
        if not self.auth_manager.add_auth_to_request(endpoint, params, headers):
            logger.error(f"認証キーが見つかりません: {provider.value}")
            return None

        # APIリクエスト作成
        request = APIRequest(
            endpoint=endpoint,
            params=params,
            headers=headers,
            request_id=f"stock_{symbol}_{int(time.time())}",
            timestamp=datetime.now(),
        )

        return await self.execute_request(request)

    async def execute_request(self, request: APIRequest) -> Optional[APIResponse]:
        """APIリクエスト実行"""
        if not self.session:
            await self.initialize()

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
            response = await self._execute_with_retry(request)

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

    async def _execute_with_retry(self, request: APIRequest) -> Optional[APIResponse]:
        """リトライ付きリクエスト実行"""
        last_error = None

        for attempt in range(request.endpoint.max_retries + 1):
            try:
                response = await self._execute_single_request(request)
                if response.success:
                    return response

                # エラーレスポンスの場合、リトライするかどうか判定
                if not self.error_handler.should_retry(response, attempt):
                    return response

                last_error = response.error_message

            except Exception as e:
                # セキュリティ強化: エラーメッセージをサニタイズして保存
                raw_error = str(e)
                last_error = self.error_handler.sanitize_error_message(raw_error, type(e).__name__)
                logger.warning(
                    f"APIリクエストエラー (試行 {attempt + 1}): {last_error}"
                )

                if attempt >= request.endpoint.max_retries:
                    break

            # リトライ前の待機
            if attempt < request.endpoint.max_retries:
                delay = self.error_handler.calculate_retry_delay(
                    attempt, 
                    request.endpoint.retry_delay_seconds,
                    self.config.exponential_backoff,
                    self.config.max_backoff_seconds
                )
                await asyncio.sleep(delay)

        # 全てのリトライが失敗
        logger.error(f"APIリクエスト最終失敗: {last_error}")
        return APIResponse(
            request=request,
            status_code=0,
            response_data=None,
            headers={},
            response_time_ms=0,
            timestamp=datetime.now(),
            success=False,
            error_message=last_error,
        )

    async def _execute_single_request(self, request: APIRequest) -> APIResponse:
        """単一APIリクエスト実行"""
        start_time = time.time()

        # URL構築
        url = self.url_builder.build_url(request)

        # ヘッダー構築
        headers = dict(request.headers)

        try:
            # HTTPリクエスト実行
            async with self.session.request(
                method=request.endpoint.method.value,
                url=url,
                params=(
                    request.params
                    if request.endpoint.method == RequestMethod.GET
                    else None
                ),
                json=(
                    request.data
                    if request.endpoint.method != RequestMethod.GET
                    else None
                ),
                headers=headers,
                timeout=ClientTimeout(total=request.endpoint.timeout_seconds),
            ) as response:
                response_time = (time.time() - start_time) * 1000
                response_headers = dict(response.headers)

                # レスポンスデータ取得
                if request.endpoint.response_format == "json":
                    response_data = await response.json()
                elif request.endpoint.response_format == "csv":
                    text_data = await response.text()
                    response_data = self.data_normalizer.parse_csv_response(text_data)
                else:
                    response_data = await response.text()

                # レスポンス作成
                api_response = APIResponse(
                    request=request,
                    status_code=response.status,
                    response_data=response_data,
                    headers=response_headers,
                    response_time_ms=response_time,
                    timestamp=datetime.now(),
                    success=response.status == 200,
                    error_message=(
                        None if response.status == 200 else f"HTTP {response.status}"
                    ),
                )

                # データ正規化
                if api_response.success:
                    api_response.normalized_data = await self.data_normalizer.normalize_response_data(
                        api_response
                    )

                return api_response

        except asyncio.TimeoutError:
            return APIResponse(
                request=request,
                status_code=0,
                response_data=None,
                headers={},
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                success=False,
                error_message="Request timeout",
            )

        except ClientError as e:
            # セキュリティ強化: エラーメッセージの機密情報をサニタイズ
            safe_error_message = self.error_handler.sanitize_error_message(str(e), "ClientError")

            return APIResponse(
                request=request,
                status_code=0,
                response_data=None,
                headers={},
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                success=False,
                error_message=safe_error_message,
            )

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

        # 各コンポーネントの統計を統合
        url_stats = self.url_builder.get_request_stats()
        error_stats = self.error_handler.get_error_stats()
        cache_stats = self.cache_manager.get_cache_stats()

        return {
            "total_requests": total,
            "successful_requests": self.request_stats["successful_requests"],
            "failed_requests": self.request_stats["failed_requests"],
            "cached_responses": self.request_stats["cached_responses"],
            "rate_limited_requests": self.request_stats["rate_limited_requests"],
            "success_rate_percent": success_rate,
            "cache_hit_rate_percent": cache_hit_rate,
            "registered_endpoints": len(self.rate_limiter.endpoints),
            "security_violations": self.request_stats["security_violations"] + url_stats.get("security_violations", 0),
            "sanitized_errors": self.request_stats["sanitized_errors"] + error_stats.get("sanitized_errors", 0),
            "cache_info": cache_stats,
        }

    def get_rate_limit_status(self) -> Dict[str, Dict[str, Any]]:
        """レート制限状況取得"""
        return self.rate_limiter.get_rate_limit_status()

    async def clear_cache(self) -> None:
        """キャッシュクリア"""
        await self.cache_manager.clear_cache()

    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        health_status = {
            "session_active": self.session is not None and not self.session.closed,
            "endpoints_registered": len(self.rate_limiter.endpoints),
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