#!/usr/bin/env python3
"""
外部APIクライアント - レート制限管理
"""

import asyncio
from datetime import datetime
from typing import Dict

from ...utils.logging_config import get_context_logger
from .enums import APIProvider, DataType
from .models import APIConfig, RateLimitState

logger = get_context_logger(__name__)


class RateLimiter:
    """レート制限管理クラス"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.rate_limits: Dict[APIProvider, RateLimitState] = {}

    def initialize_provider(self, provider: APIProvider) -> None:
        """プロバイダーのレート制限状態を初期化"""
        if provider not in self.rate_limits:
            self.rate_limits[provider] = RateLimitState(provider=provider)
            logger.info(f"レート制限状態初期化: {provider.value}")

    async def check_rate_limit(self, provider: APIProvider) -> bool:
        """レート制限チェック"""
        if provider not in self.rate_limits:
            self.initialize_provider(provider)
            return True

        rate_limit = self.rate_limits[provider]
        current_time = datetime.now()

        # ブロック期間チェック
        if rate_limit.blocked_until and current_time < rate_limit.blocked_until:
            return False

        # タイムウィンドウリセット
        self._reset_time_windows(rate_limit, current_time)

        # 各種制限チェック（仮のエンドポイント設定から取得）
        # 実際の実装では、各プロバイダーのエンドポイント情報から制限を取得
        return self._check_provider_limits(provider, rate_limit)

    def _check_provider_limits(
        self, provider: APIProvider, rate_limit: RateLimitState
    ) -> bool:
        """プロバイダー固有の制限チェック"""
        # デフォルト制限値（実際にはエンドポイント設定から取得）
        default_limits = {
            APIProvider.YAHOO_FINANCE: {
                "per_second": 2.0,
                "per_minute": 100,
                "per_hour": 1000,
            },
            APIProvider.ALPHA_VANTAGE: {
                "per_second": 0.2,
                "per_minute": 5,
                "per_hour": 500,
            },
            APIProvider.IEX_CLOUD: {
                "per_second": 10.0,
                "per_minute": 500,
                "per_hour": 10000,
            },
            APIProvider.MOCK_PROVIDER: {
                "per_second": 10.0,
                "per_minute": 600,
                "per_hour": 10000,
            },
        }

        limits = default_limits.get(
            provider,
            {"per_second": 1.0, "per_minute": 60, "per_hour": 1000},
        )

        # 秒あたり制限
        if rate_limit.requests_per_second >= limits["per_second"]:
            logger.debug(f"秒あたり制限に到達: {provider.value}")
            return False

        # 分あたり制限
        if rate_limit.requests_per_minute >= limits["per_minute"]:
            logger.debug(f"分あたり制限に到達: {provider.value}")
            return False

        # 時間あたり制限
        if rate_limit.requests_per_hour >= limits["per_hour"]:
            logger.debug(f"時間あたり制限に到達: {provider.value}")
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

        logger.info(f"レート制限により {wait_seconds:.2f} 秒待機: {provider.value}")
        await asyncio.sleep(wait_seconds)

    async def update_rate_limit_state(self, provider: APIProvider) -> None:
        """レート制限状態更新"""
        if provider not in self.rate_limits:
            self.initialize_provider(provider)

        rate_limit = self.rate_limits[provider]
        current_time = datetime.now()

        # リクエストカウント更新
        rate_limit.requests_per_second += 1
        rate_limit.requests_per_minute += 1
        rate_limit.requests_per_hour += 1
        rate_limit.last_request_time = current_time

        logger.debug(
            f"レート制限状態更新: {provider.value} - "
            f"秒:{rate_limit.requests_per_second}, "
            f"分:{rate_limit.requests_per_minute}, "
            f"時:{rate_limit.requests_per_hour}"
        )

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

    def reset_provider_limits(self, provider: APIProvider) -> None:
        """特定プロバイダーの制限をリセット"""
        if provider in self.rate_limits:
            self.rate_limits[provider] = RateLimitState(provider=provider)
            logger.info(f"レート制限状態をリセット: {provider.value}")

    def reset_all_limits(self) -> None:
        """全プロバイダーの制限をリセット"""
        for provider in list(self.rate_limits.keys()):
            self.reset_provider_limits(provider)
        logger.info("全プロバイダーのレート制限状態をリセット")