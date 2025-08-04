"""
API通信の耐障害性強化モジュール
Issue 185: 外部API通信の耐障害性強化

高度なリトライ機構、サーキットブレーカー、フェイルオーバー機能を提供
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    NetworkError,
    RateLimitError,
    ResourceNotFoundError,
    ServerError,
    TimeoutError,
)
from .logging_config import (
    get_context_logger,
    log_api_call,
    log_business_event,
    log_error_with_context,
    log_performance_metric,
    log_security_event,
)


class CircuitState(Enum):
    """サーキットブレーカーの状態"""

    CLOSED = "closed"  # 正常状態
    OPEN = "open"  # 遮断状態
    HALF_OPEN = "half_open"  # 半開状態


@dataclass
class RetryConfig:
    """リトライ設定"""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_factor: float = 1.0
    status_forcelist: List[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )
    allowed_methods: List[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE"]
    )


@dataclass
class CircuitBreakerConfig:
    """サーキットブレーカー設定"""

    failure_threshold: int = 5  # 失敗閾値
    success_threshold: int = 3  # 成功閾値（半開状態で）
    timeout: float = 60.0  # 開状態のタイムアウト（秒）
    monitor_window: float = 300.0  # 監視ウィンドウ（秒）


@dataclass
class APIEndpoint:
    """APIエンドポイント設定"""

    name: str
    base_url: str
    timeout: float = 30.0
    priority: int = 1  # 優先度（小さいほど高優先度）
    health_check_path: str = "/"
    is_active: bool = True


class CircuitBreaker:
    """サーキットブレーカー実装"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = get_context_logger(__name__)

    def can_execute(self) -> bool:
        """実行可能かどうかを判定"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.logger.info(
                    "サーキットブレーカーが半開状態に移行", previous_state="open"
                )
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self) -> None:
        """成功を記録"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info(
                    "サーキットブレーカーが閉状態に復帰",
                    success_count=self.success_count,
                )
        elif self.state == CircuitState.CLOSED:
            # 成功時は失敗回数をリセット
            self.failure_count = 0

    def record_failure(self) -> None:
        """失敗を記録"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.warning(
                    "サーキットブレーカーが開状態に移行",
                    failure_count=self.failure_count,
                    threshold=self.config.failure_threshold,
                )
                log_security_event(
                    "circuit_breaker_opened",
                    "warning",
                    failure_count=self.failure_count,
                )
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.logger.warning("サーキットブレーカーが再び開状態に移行")

    def _should_attempt_reset(self) -> bool:
        """リセットを試行すべきかどうかを判定"""
        if self.last_failure_time is None:
            return True
        return (
            datetime.now() - self.last_failure_time
        ).total_seconds() >= self.config.timeout


class ResilientAPIClient:
    """耐障害性を持つAPIクライアント"""

    def __init__(
        self,
        endpoints: List[APIEndpoint],
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        enable_health_check: bool = True,
        health_check_interval: float = 300.0,
    ):
        """
        Args:
            endpoints: APIエンドポイントのリスト
            retry_config: リトライ設定
            circuit_config: サーキットブレーカー設定
            enable_health_check: ヘルスチェック有効化
            health_check_interval: ヘルスチェック間隔（秒）
        """
        self.endpoints = sorted(endpoints, key=lambda x: x.priority)
        self.retry_config = retry_config or RetryConfig()
        self.circuit_config = circuit_config or CircuitBreakerConfig()

        # 各エンドポイントにサーキットブレーカーを作成
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            endpoint.name: CircuitBreaker(self.circuit_config)
            for endpoint in self.endpoints
        }

        self.enable_health_check = enable_health_check
        self.health_check_interval = health_check_interval
        self.last_health_check: Dict[str, datetime] = {}

        self.logger = get_context_logger(__name__)

        # HTTPセッションの設定
        self.session = self._create_session()

        # 初期ヘルスチェック
        if self.enable_health_check:
            self._initial_health_check()

    def _create_session(self) -> requests.Session:
        """HTTPセッションを作成"""
        session = requests.Session()

        # リトライ戦略の設定
        retry_strategy = Retry(
            total=self.retry_config.max_attempts - 1,  # 最初の試行を除くリトライ回数
            backoff_factor=self.retry_config.base_delay,  # 修正箇所
            status_forcelist=self.retry_config.status_forcelist,
            allowed_methods=self.retry_config.allowed_methods,
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def _initial_health_check(self) -> None:
        """初期ヘルスチェック"""
        self.logger.info("初期ヘルスチェックを開始")

        for endpoint in self.endpoints:
            try:
                self._check_endpoint_health(endpoint)
            except Exception as e:
                self.logger.warning(
                    f"エンドポイント {endpoint.name} の初期ヘルスチェックに失敗",
                    endpoint=endpoint.name,
                    error=str(e),
                )

    def _check_endpoint_health(self, endpoint: APIEndpoint) -> bool:
        """エンドポイントのヘルスチェック"""
        try:
            health_url = f"{endpoint.base_url.rstrip('/')}{endpoint.health_check_path}"

            start_time = time.time()
            response = self.session.get(
                health_url,
                timeout=min(endpoint.timeout, 10.0),  # ヘルスチェックは短いタイムアウト
            )
            response_time = (time.time() - start_time) * 1000

            is_healthy = response.status_code < 400
            endpoint.is_active = is_healthy
            self.last_health_check[endpoint.name] = datetime.now()

            log_performance_metric(
                "health_check_time",
                response_time,
                "ms",
                endpoint=endpoint.name,
                status_code=response.status_code,
                is_healthy=is_healthy,
            )

            if is_healthy:
                self.logger.debug(f"ヘルスチェック成功: {endpoint.name}")
                return True
            else:
                self.logger.warning(
                    f"ヘルスチェック失敗: {endpoint.name}",
                    status_code=response.status_code,
                )
                return False

        except Exception as e:
            endpoint.is_active = False
            self.logger.error(f"ヘルスチェックエラー: {endpoint.name}", error=str(e))
            return False

    def _should_health_check(self, endpoint: APIEndpoint) -> bool:
        """ヘルスチェックが必要かどうかを判定"""
        if not self.enable_health_check:
            return False

        last_check = self.last_health_check.get(endpoint.name)
        if last_check is None:
            return True

        return (
            datetime.now() - last_check
        ).total_seconds() >= self.health_check_interval

    def _get_available_endpoints(self) -> List[APIEndpoint]:
        """利用可能なエンドポイントを取得"""
        available = []

        for endpoint in self.endpoints:
            # ヘルスチェックが必要な場合は実行
            if self._should_health_check(endpoint):
                self._check_endpoint_health(endpoint)

            # サーキットブレーカーとアクティブ状態をチェック
            circuit_breaker = self.circuit_breakers[endpoint.name]
            if endpoint.is_active and circuit_breaker.can_execute():
                available.append(endpoint)

        return available

    def _execute_with_retry(
        self, endpoint: APIEndpoint, method: str, url: str, **kwargs
    ) -> requests.Response:
        """リトライ機構付きでリクエストを実行 (urllib3のRetryに任せる)"""
        circuit_breaker = self.circuit_breakers[endpoint.name]

        try:
            start_time = time.time()

            # タイムアウトを設定
            kwargs.setdefault("timeout", endpoint.timeout)

            # リクエスト実行 (Retryオブジェクトがリトライを処理)
            response = self.session.request(method, url, **kwargs)

            response_time = (time.time() - start_time) * 1000

            # API呼び出しログ
            # response.raw.retries は requests 内部の Retry オブジェクト
            # _retry_counts はリトライ回数を保持する辞書
            retry_count_for_method = 0
            if hasattr(response.raw, 'retries') and hasattr(response.raw.retries, '_retry_counts'):
                # _retry_counts は {method.lower(): count} の形式
                retry_count_for_method = response.raw.retries._retry_counts.get(method.lower(), 0)

            log_api_call(
                endpoint.name,
                method,
                url,
                response.status_code,
                response_time=response_time,
            )

            # 成功判定
            if response.status_code < 400:
                circuit_breaker.record_success()
                log_performance_metric(
                    "api_response_time",
                    response_time,
                    "ms",
                    endpoint=endpoint.name,
                    method=method,
                    status_code=response.status_code,
                )
                return response

        except requests.exceptions.HTTPError as http_exc:
            circuit_breaker.record_failure()
            self.logger.error(f"HTTPエラー発生: {http_exc.response.status_code}",
                            endpoint=endpoint.name,
                            error=str(http_exc))
            self._raise_http_error(http_exc.response)

        except requests.exceptions.RequestException as req_exc:
            circuit_breaker.record_failure()
            # ネットワークエラー（リトライ後も失敗）
            self._raise_network_error(req_exc)

    def _raise_http_error(self, response: requests.Response) -> None:
        """HTTPエラーを適切な例外に変換"""
        status_code = response.status_code
        message = f"HTTP {status_code}: {response.reason}"

        if status_code == 400:
            raise BadRequestError(
                message, "API_BAD_REQUEST", {"status_code": status_code}
            )
        elif status_code in (401, 403):
            raise AuthenticationError(
                message, "API_AUTH_ERROR", {"status_code": status_code}
            )
        elif status_code == 404:
            raise ResourceNotFoundError(
                message, "API_NOT_FOUND", {"status_code": status_code}
            )
        elif status_code == 429:
            raise RateLimitError(
                message, "API_RATE_LIMIT", {"status_code": status_code}
            )
        elif status_code >= 500:
            raise ServerError(message, "API_SERVER_ERROR", {"status_code": status_code})
        else:
            raise APIError(message, "API_HTTP_ERROR", {"status_code": status_code})

    def _raise_network_error(self, exc: requests.exceptions.RequestException) -> None:
        """ネットワークエラーを適切な例外に変換"""
        if isinstance(exc, requests.exceptions.Timeout):
            raise TimeoutError(f"リクエストタイムアウト: {exc}", "NETWORK_TIMEOUT")
        elif isinstance(exc, requests.exceptions.ConnectionError):
            raise NetworkError(f"接続エラー: {exc}", "NETWORK_CONNECTION")
        else:
            raise NetworkError(f"ネットワークエラー: {exc}", "NETWORK_UNKNOWN")

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        """
        耐障害性を持つHTTPリクエスト

        Args:
            method: HTTPメソッド
            path: パス（エンドポイントのbase_urlに続く部分）
            **kwargs: requests.request に渡すその他の引数

        Returns:
            HTTPレスポンス

        Raises:
            APIError: API関連エラー
        """
        available_endpoints = self._get_available_endpoints()

        if not available_endpoints:
            log_security_event("all_endpoints_unavailable", "critical")
            raise NetworkError(
                "利用可能なAPIエンドポイントがありません", "NO_AVAILABLE_ENDPOINTS"
            )

        last_exception = None

        for endpoint in available_endpoints:
            try:
                url = f"{endpoint.base_url.rstrip('/')}/{path.lstrip('/')}"

                self.logger.debug(
                    f"エンドポイント {endpoint.name} でリクエスト試行",
                    endpoint=endpoint.name,
                    method=method,
                    url=url,
                )

                response = self._execute_with_retry(endpoint, method, url, **kwargs)

                # 成功
                log_business_event(
                    "api_request_success",
                    endpoint=endpoint.name,
                    method=method,
                    path=path,
                )

                return response

            except Exception as e:
                last_exception = e
                log_error_with_context(
                    e, {"endpoint": endpoint.name, "method": method, "path": path}
                )

                self.logger.warning(
                    f"エンドポイント {endpoint.name} でエラー、次を試行",
                    endpoint=endpoint.name,
                    error=str(e),
                )
                continue

        # すべてのエンドポイントで失敗
        log_business_event(
            "api_request_failed_all_endpoints",
            method=method,
            path=path,
            available_endpoints=[ep.name for ep in available_endpoints],
        )

        if last_exception:
            raise last_exception
        else:
            raise APIError("すべてのエンドポイントで失敗しました")

    def get(self, path: str, **kwargs) -> requests.Response:
        """GETリクエスト"""
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> requests.Response:
        """POSTリクエスト"""
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs) -> requests.Response:
        """PUTリクエスト"""
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        """DELETEリクエスト"""
        return self.request("DELETE", path, **kwargs)

    def get_status(self) -> Dict[str, Any]:
        """APIクライアントの状態を取得"""
        status = {"endpoints": [], "overall_health": True}

        for endpoint in self.endpoints:
            circuit_breaker = self.circuit_breakers[endpoint.name]
            endpoint_status = {
                "name": endpoint.name,
                "base_url": endpoint.base_url,
                "is_active": endpoint.is_active,
                "circuit_state": circuit_breaker.state.value,
                "failure_count": circuit_breaker.failure_count,
                "last_health_check": self.last_health_check.get(endpoint.name),
            }
            status["endpoints"].append(endpoint_status)

            if not endpoint.is_active or circuit_breaker.state == CircuitState.OPEN:
                status["overall_health"] = False

        return status


def resilient_api_call(
    endpoints: List[APIEndpoint],
    retry_config: Optional[RetryConfig] = None,
    circuit_config: Optional[CircuitBreakerConfig] = None,
):
    """
    関数デコレータ：API呼び出しに耐障害性を追加

    Usage:
        @resilient_api_call([endpoint1, endpoint2])
        def fetch_data():
            # API呼び出し処理
            pass
    """

    def decorator(func: Callable) -> Callable:
        client = ResilientAPIClient(endpoints, retry_config, circuit_config)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 関数にクライアントを注入
            return func(client, *args, **kwargs)

        return wrapper

    return decorator


# 使用例とデモ
if __name__ == "__main__":
    from .logging_config import setup_logging

    setup_logging()
    logger = get_context_logger(__name__)

    # エンドポイント設定例
    endpoints = [
        APIEndpoint("primary", "https://api.example.com", priority=1),
        APIEndpoint("secondary", "https://api-backup.example.com", priority=2),
        APIEndpoint("tertiary", "https://api-fallback.example.com", priority=3),
    ]

    # リトライ設定例
    retry_config = RetryConfig(
        max_attempts=3, base_delay=1.0, max_delay=30.0, exponential_base=2.0
    )

    # サーキットブレーカー設定例
    circuit_config = CircuitBreakerConfig(
        failure_threshold=5, success_threshold=3, timeout=60.0
    )

    # クライアント作成
    client = ResilientAPIClient(endpoints, retry_config, circuit_config)

    # 状態確認
    status = client.get_status()
    logger.info("APIクライアント状態", **status)

    logger.info("API耐障害性モジュールの初期化完了")
