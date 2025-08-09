#!/usr/bin/env python3
"""
外部APIクライアントシステム
Issue #331: API・外部統合システム - Phase 1

外部市場データAPI統合・RESTfulクライアント・レート制限・エラー回復
- 複数APIプロバイダー対応
- 自動レート制限管理
- 堅牢なエラー回復システム
- データ形式統一・正規化
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import aiohttp
import pandas as pd
from aiohttp import ClientError, ClientTimeout

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class APIProvider(Enum):
    """APIプロバイダー"""

    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    IEX_CLOUD = "iex_cloud"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    TWELVE_DATA = "twelve_data"
    MOCK_PROVIDER = "mock_provider"  # 開発・テスト用


class DataType(Enum):
    """データタイプ"""

    STOCK_PRICE = "stock_price"
    MARKET_INDEX = "market_index"
    COMPANY_INFO = "company_info"
    FINANCIAL_STATEMENT = "financial_statement"
    NEWS = "news"
    ECONOMIC_INDICATOR = "economic_indicator"
    EXCHANGE_RATE = "exchange_rate"
    SECTOR_PERFORMANCE = "sector_performance"


class RequestMethod(Enum):
    """HTTPリクエストメソッド"""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass
class APIEndpoint:
    """APIエンドポイント設定"""

    provider: APIProvider
    data_type: DataType
    endpoint_url: str
    method: RequestMethod = RequestMethod.GET

    # レート制限設定
    rate_limit_per_second: float = 1.0
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # 認証設定
    requires_auth: bool = False
    auth_header_name: Optional[str] = None
    auth_param_name: Optional[str] = None

    # データ変換設定
    response_format: str = "json"  # json, csv, xml
    data_path: Optional[str] = None  # JSONレスポンス内のデータパス

    # エラー回復設定
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: int = 30


@dataclass
class APIRequest:
    """APIリクエスト"""

    endpoint: APIEndpoint
    params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    data: Optional[Dict[str, Any]] = None

    # リクエストメタデータ
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    priority: int = 1  # 1=高, 2=中, 3=低


@dataclass
class APIResponse:
    """APIレスポンス"""

    request: APIRequest
    status_code: int
    response_data: Any
    headers: Dict[str, str]

    # 実行メタデータ
    response_time_ms: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

    # 処理結果
    normalized_data: Optional[pd.DataFrame] = None
    cache_key: Optional[str] = None


@dataclass
class RateLimitState:
    """レート制限状態"""

    provider: APIProvider
    requests_per_second: int = 0
    requests_per_minute: int = 0
    requests_per_hour: int = 0

    # タイムウィンドウ
    second_window_start: datetime = field(default_factory=datetime.now)
    minute_window_start: datetime = field(default_factory=datetime.now)
    hour_window_start: datetime = field(default_factory=datetime.now)

    # 制限情報
    last_request_time: Optional[datetime] = None
    blocked_until: Optional[datetime] = None


@dataclass
class APIConfig:
    """API設定"""

    # 基本設定
    user_agent: str = "DayTrade-API-Client/1.0"
    default_timeout_seconds: int = 30
    max_concurrent_requests: int = 10

    # レート制限設定
    global_rate_limit_per_second: float = 5.0
    respect_server_rate_limits: bool = True
    rate_limit_buffer_factor: float = 0.8  # 80%の余裕を持つ

    # キャッシュ設定
    enable_response_caching: bool = True
    cache_ttl_seconds: int = 300  # 5分
    cache_size_limit: int = 1000

    # エラー回復設定
    default_max_retries: int = 3
    exponential_backoff: bool = True
    max_backoff_seconds: float = 60.0

    # 認証設定
    api_keys: Dict[str, str] = field(default_factory=dict)
    oauth_tokens: Dict[str, str] = field(default_factory=dict)


class ExternalAPIClient:
    """外部APIクライアント"""

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.rate_limits: Dict[APIProvider, RateLimitState] = {}
        self.response_cache: Dict[str, APIResponse] = {}

        # HTTP セッション
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # 統計情報
        self.request_stats: Dict[str, int] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "rate_limited_requests": 0,
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
        if endpoint.provider not in self.rate_limits:
            self.rate_limits[endpoint.provider] = RateLimitState(
                provider=endpoint.provider
            )

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

        if endpoint_key not in self.endpoints:
            logger.error(f"未登録のエンドポイント: {endpoint_key}")
            return None

        endpoint = self.endpoints[endpoint_key]

        # リクエストパラメータ構築
        params = {"symbol": symbol, **kwargs}

        # 認証パラメータ追加
        if endpoint.requires_auth:
            auth_key = self._get_auth_key(endpoint)
            if auth_key and endpoint.auth_param_name:
                params[endpoint.auth_param_name] = auth_key
            elif not auth_key:
                logger.error(f"認証キーが見つかりません: {provider.value}")
                return None

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

        # キャッシュチェック
        cache_key = self._generate_cache_key(request)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            self.request_stats["cached_responses"] += 1
            return cached_response

        # レート制限チェック
        if not await self._check_rate_limit(request.endpoint.provider):
            self.request_stats["rate_limited_requests"] += 1
            logger.warning(f"レート制限により遅延: {request.endpoint.provider.value}")

            # レート制限解除まで待機
            await self._wait_for_rate_limit(request.endpoint.provider)

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
        await self._update_rate_limit_state(request.endpoint.provider)

        # キャッシュに保存
        if response and response.success and self.config.enable_response_caching:
            self._cache_response(cache_key, response)

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
                if not self._should_retry(response, attempt):
                    return response

                last_error = response.error_message

            except Exception as e:
                last_error = str(e)
                logger.warning(f"APIリクエストエラー (試行 {attempt + 1}): {e}")

                if attempt >= request.endpoint.max_retries:
                    break

            # リトライ前の待機
            if attempt < request.endpoint.max_retries:
                delay = self._calculate_retry_delay(attempt, request.endpoint)
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
        url = self._build_url(request)

        # ヘッダー構築
        headers = dict(request.headers)
        if request.endpoint.requires_auth and request.endpoint.auth_header_name:
            auth_key = self._get_auth_key(request.endpoint)
            if auth_key:
                headers[request.endpoint.auth_header_name] = auth_key

        try:
            # HTTPリクエスト実行
            async with self.session.request(
                method=request.endpoint.method.value,
                url=url,
                params=request.params
                if request.endpoint.method == RequestMethod.GET
                else None,
                json=request.data
                if request.endpoint.method != RequestMethod.GET
                else None,
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
                    response_data = self._parse_csv_response(text_data)
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
                    error_message=None
                    if response.status == 200
                    else f"HTTP {response.status}",
                )

                # データ正規化
                if api_response.success:
                    api_response.normalized_data = await self._normalize_response_data(
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
            return APIResponse(
                request=request,
                status_code=0,
                response_data=None,
                headers={},
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                success=False,
                error_message=str(e),
            )

    def _build_url(self, request: APIRequest) -> str:
        """URL構築"""
        url = request.endpoint.endpoint_url

        # パラメータ置換（{symbol}など）
        for key, value in request.params.items():
            placeholder = f"{{{key}}}"
            if placeholder in url:
                url = url.replace(placeholder, str(value))

        return url

    def _get_auth_key(self, endpoint: APIEndpoint) -> Optional[str]:
        """認証キー取得"""
        provider_key = endpoint.provider.value
        return self.config.api_keys.get(provider_key)

    def _should_retry(self, response: APIResponse, attempt: int) -> bool:
        """リトライ判定"""
        # 成功レスポンスはリトライしない
        if response.success:
            return False

        # 最大リトライ数に達した場合はリトライしない
        if attempt >= response.request.endpoint.max_retries:
            return False

        # 4xx エラー（クライアントエラー）はリトライしない
        if 400 <= response.status_code < 500:
            return False

        # 5xx エラーやネットワークエラーはリトライする
        return True

    def _calculate_retry_delay(self, attempt: int, endpoint: APIEndpoint) -> float:
        """リトライ遅延時間計算"""
        base_delay = endpoint.retry_delay_seconds

        if self.config.exponential_backoff:
            # 指数バックオフ
            delay = base_delay * (2**attempt)
            return min(delay, self.config.max_backoff_seconds)
        else:
            return base_delay

    async def _check_rate_limit(self, provider: APIProvider) -> bool:
        """レート制限チェック"""
        if provider not in self.rate_limits:
            return True

        rate_limit = self.rate_limits[provider]
        current_time = datetime.now()

        # ブロック期間チェック
        if rate_limit.blocked_until and current_time < rate_limit.blocked_until:
            return False

        # タイムウィンドウリセット
        self._reset_time_windows(rate_limit, current_time)

        # 各種制限チェック
        endpoint_key = f"{provider.value}_{DataType.STOCK_PRICE.value}"
        if endpoint_key in self.endpoints:
            endpoint = self.endpoints[endpoint_key]

            # 秒あたり制限
            if rate_limit.requests_per_second >= endpoint.rate_limit_per_second:
                return False

            # 分あたり制限
            if rate_limit.requests_per_minute >= endpoint.rate_limit_per_minute:
                return False

            # 時間あたり制限
            if rate_limit.requests_per_hour >= endpoint.rate_limit_per_hour:
                return False

        return True

    async def _wait_for_rate_limit(self, provider: APIProvider) -> None:
        """レート制限解除まで待機"""
        if provider not in self.rate_limits:
            return

        rate_limit = self.rate_limits[provider]

        # 最小待機時間を計算
        wait_seconds = 1.0  # デフォルト

        # 秒制限の場合
        if rate_limit.last_request_time:
            elapsed = (datetime.now() - rate_limit.last_request_time).total_seconds()
            if elapsed < 1.0:
                wait_seconds = max(wait_seconds, 1.0 - elapsed)

        await asyncio.sleep(wait_seconds)

    async def _update_rate_limit_state(self, provider: APIProvider) -> None:
        """レート制限状態更新"""
        if provider not in self.rate_limits:
            return

        rate_limit = self.rate_limits[provider]
        current_time = datetime.now()

        # リクエストカウント更新
        rate_limit.requests_per_second += 1
        rate_limit.requests_per_minute += 1
        rate_limit.requests_per_hour += 1
        rate_limit.last_request_time = current_time

    def _reset_time_windows(
        self, rate_limit: RateLimitState, current_time: datetime
    ) -> None:
        """タイムウィンドウリセット"""
        # 秒ウィンドウリセット
        if (current_time - rate_limit.second_window_start).total_seconds() >= 1:
            rate_limit.requests_per_second = 0
            rate_limit.second_window_start = current_time

        # 分ウィンドウリセット
        if (current_time - rate_limit.minute_window_start).total_seconds() >= 60:
            rate_limit.requests_per_minute = 0
            rate_limit.minute_window_start = current_time

        # 時間ウィンドウリセット
        if (current_time - rate_limit.hour_window_start).total_seconds() >= 3600:
            rate_limit.requests_per_hour = 0
            rate_limit.hour_window_start = current_time

    def _generate_cache_key(self, request: APIRequest) -> str:
        """キャッシュキー生成"""
        key_parts = [
            request.endpoint.provider.value,
            request.endpoint.data_type.value,
            str(sorted(request.params.items())),
        ]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[APIResponse]:
        """キャッシュレスポンス取得"""
        if not self.config.enable_response_caching:
            return None

        if cache_key not in self.response_cache:
            return None

        cached_response = self.response_cache[cache_key]

        # TTLチェック
        age = (datetime.now() - cached_response.timestamp).total_seconds()
        if age > self.config.cache_ttl_seconds:
            del self.response_cache[cache_key]
            return None

        return cached_response

    def _cache_response(self, cache_key: str, response: APIResponse) -> None:
        """レスポンスキャッシュ"""
        if not self.config.enable_response_caching:
            return

        # キャッシュサイズ制限
        if len(self.response_cache) >= self.config.cache_size_limit:
            # 最も古いエントリを削除（簡易LRU）
            oldest_key = min(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k].timestamp,
            )
            del self.response_cache[oldest_key]

        response.cache_key = cache_key
        self.response_cache[cache_key] = response

    def _parse_csv_response(self, csv_text: str) -> pd.DataFrame:
        """CSVレスポンス解析"""
        try:
            from io import StringIO

            return pd.read_csv(StringIO(csv_text))
        except Exception as e:
            logger.error(f"CSV解析エラー: {e}")
            return pd.DataFrame()

    async def _normalize_response_data(
        self, response: APIResponse
    ) -> Optional[pd.DataFrame]:
        """レスポンスデータ正規化"""
        try:
            provider = response.request.endpoint.provider
            data_type = response.request.endpoint.data_type

            if provider == APIProvider.MOCK_PROVIDER:
                return await self._normalize_mock_data(response)
            elif provider == APIProvider.YAHOO_FINANCE:
                return await self._normalize_yahoo_finance_data(response)
            elif provider == APIProvider.ALPHA_VANTAGE:
                return await self._normalize_alpha_vantage_data(response)
            else:
                # 汎用正規化
                return await self._normalize_generic_data(response)

        except Exception as e:
            logger.error(f"データ正規化エラー: {e}")
            return None

    async def _normalize_mock_data(self, response: APIResponse) -> pd.DataFrame:
        """モックデータ正規化"""
        # 模擬的なデータ正規化
        data = response.response_data

        if isinstance(data, dict) and "price_data" in data:
            price_data = data["price_data"]

            df = pd.DataFrame(
                {
                    "timestamp": [datetime.now()],
                    "open": [price_data.get("open", 1000)],
                    "high": [price_data.get("high", 1050)],
                    "low": [price_data.get("low", 950)],
                    "close": [price_data.get("close", 1025)],
                    "volume": [price_data.get("volume", 1000000)],
                    "symbol": [response.request.params.get("symbol", "UNKNOWN")],
                }
            )

            return df

        # フォールバック: 模擬データ生成
        import random

        symbol = response.request.params.get("symbol", "MOCK")
        base_price = 1000 + hash(symbol) % 2000

        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [base_price * random.uniform(0.98, 1.02)],
                "high": [base_price * random.uniform(1.01, 1.05)],
                "low": [base_price * random.uniform(0.95, 0.99)],
                "close": [base_price * random.uniform(0.99, 1.03)],
                "volume": [random.randint(100000, 500000)],
                "symbol": [symbol],
            }
        )

        return df

    async def _normalize_yahoo_finance_data(
        self, response: APIResponse
    ) -> pd.DataFrame:
        """Yahoo Finance データ正規化"""
        try:
            data = response.response_data

            # Yahoo Finance API レスポンス構造に基づく解析
            if isinstance(data, dict) and "chart" in data:
                chart_data = data["chart"]["result"][0]
                timestamps = chart_data["timestamp"]
                quotes = chart_data["indicators"]["quote"][0]

                df = pd.DataFrame(
                    {
                        "timestamp": [datetime.fromtimestamp(ts) for ts in timestamps],
                        "open": quotes["open"],
                        "high": quotes["high"],
                        "low": quotes["low"],
                        "close": quotes["close"],
                        "volume": quotes["volume"],
                        "symbol": [chart_data["meta"]["symbol"]] * len(timestamps),
                    }
                )

                # 欠損値処理
                df = df.fillna(method="ffill").fillna(method="bfill")

                return df

        except Exception as e:
            logger.error(f"Yahoo Finance データ正規化エラー: {e}")

        return pd.DataFrame()

    async def _normalize_alpha_vantage_data(
        self, response: APIResponse
    ) -> pd.DataFrame:
        """Alpha Vantage データ正規化"""
        try:
            data = response.response_data

            # Alpha Vantage API レスポンス構造に基づく解析
            if isinstance(data, dict) and "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                symbol = data["Meta Data"]["2. Symbol"]

                records = []
                for date_str, prices in time_series.items():
                    records.append(
                        {
                            "timestamp": datetime.strptime(date_str, "%Y-%m-%d"),
                            "open": float(prices["1. open"]),
                            "high": float(prices["2. high"]),
                            "low": float(prices["3. low"]),
                            "close": float(prices["4. close"]),
                            "volume": int(prices["5. volume"]),
                            "symbol": symbol,
                        }
                    )

                df = pd.DataFrame(records)
                df = df.sort_values("timestamp")

                return df

        except Exception as e:
            logger.error(f"Alpha Vantage データ正規化エラー: {e}")

        return pd.DataFrame()

    async def _normalize_generic_data(self, response: APIResponse) -> pd.DataFrame:
        """汎用データ正規化"""
        # 基本的な正規化処理
        data = response.response_data

        if isinstance(data, list) and data:
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            return pd.DataFrame()

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
            "active_providers": len(self.rate_limits),
        }

    def get_rate_limit_status(self) -> Dict[str, Dict[str, Any]]:
        """レート制限状況取得"""
        status = {}

        for provider, rate_limit in self.rate_limits.items():
            status[provider.value] = {
                "requests_per_second": rate_limit.requests_per_second,
                "requests_per_minute": rate_limit.requests_per_minute,
                "requests_per_hour": rate_limit.requests_per_hour,
                "last_request": rate_limit.last_request_time.isoformat()
                if rate_limit.last_request_time
                else None,
                "blocked_until": rate_limit.blocked_until.isoformat()
                if rate_limit.blocked_until
                else None,
            }

        return status

    async def clear_cache(self) -> None:
        """キャッシュクリア"""
        self.response_cache.clear()
        logger.info("APIレスポンスキャッシュをクリアしました")

    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        health_status = {
            "session_active": self.session is not None and not self.session.closed,
            "endpoints_registered": len(self.endpoints),
            "cache_entries": len(self.response_cache),
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
                        "status": "healthy"
                        if response and response.success
                        else "unhealthy",
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


# 使用例・テスト関数


async def setup_api_client() -> ExternalAPIClient:
    """APIクライアントセットアップ"""
    config = APIConfig(
        max_concurrent_requests=5, cache_ttl_seconds=300, default_max_retries=2
    )

    client = ExternalAPIClient(config)
    await client.initialize()

    return client


async def test_stock_data_fetching():
    """株価データ取得テスト"""
    client = await setup_api_client()

    try:
        # 複数銘柄のデータ取得テスト
        test_symbols = ["7203", "8306", "9984"]  # トヨタ、三菱UFJ、SBG

        for symbol in test_symbols:
            print(f"\n{symbol} データ取得テスト:")

            response = await client.fetch_stock_data(symbol, APIProvider.MOCK_PROVIDER)

            if response and response.success:
                print(f"  ✅ 成功: {response.response_time_ms:.1f}ms")
                if response.normalized_data is not None:
                    print(f"  📊 データ: {len(response.normalized_data)} レコード")
                    print(f"  💰 価格: {response.normalized_data['close'].iloc[0]:.2f}")
            else:
                print(
                    f"  ❌ 失敗: {response.error_message if response else 'レスポンスなし'}"
                )

        # 統計情報表示
        stats = client.get_request_statistics()
        print("\n📈 統計情報:")
        print(f"  総リクエスト数: {stats['total_requests']}")
        print(f"  成功率: {stats['success_rate_percent']:.1f}%")
        print(f"  キャッシュヒット率: {stats['cache_hit_rate_percent']:.1f}%")

    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(test_stock_data_fetching())
