"""
外部APIクライアント レート制限管理
API プロバイダー別のレート制限チェックと制御
"""

import asyncio
from datetime import datetime
from typing import Dict

from ...utils.logging_config import get_context_logger
from .enums import APIProvider, DataType
from .models import APIEndpoint, RateLimitState

logger = get_context_logger(__name__)


class RateLimiter:
    """レート制限管理クラス"""

    def __init__(self):
        self.rate_limits: Dict[APIProvider, RateLimitState] = {}
        self.endpoints: Dict[str, APIEndpoint] = {}

    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        """エンドポイント登録とレート制限状態初期化"""
        endpoint_key = f"{endpoint.provider.value}_{endpoint.data_type.value}"
        self.endpoints[endpoint_key] = endpoint

        # レート制限状態初期化
        if endpoint.provider not in self.rate_limits:
            self.rate_limits[endpoint.provider] = RateLimitState(
                provider=endpoint.provider
            )

    async def check_rate_limit(self, provider: APIProvider) -> bool:
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

    async def wait_for_rate_limit(self, provider: APIProvider) -> None:
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

    async def update_rate_limit_state(self, provider: APIProvider) -> None:
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

    def get_rate_limit_status(self) -> Dict[str, Dict[str, any]]:
        """レート制限状況取得"""
        status = {}

        for provider, rate_limit in self.rate_limits.items():
            status[provider.value] = {
                "requests_per_second": rate_limit.requests_per_second,
                "requests_per_minute": rate_limit.requests_per_minute,
                "requests_per_hour": rate_limit.requests_per_hour,
                "last_request": (
                    rate_limit.last_request_time.isoformat()
                    if rate_limit.last_request_time
                    else None
                ),
                "blocked_until": (
                    rate_limit.blocked_until.isoformat()
                    if rate_limit.blocked_until
                    else None
                ),
            }

        return status