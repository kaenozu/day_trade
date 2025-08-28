#!/usr/bin/env python3
"""
レート制限サービス実装
Issue #918 項目9対応: セキュリティ強化

APIレート制限、アクセス頻度制御機能
"""

import threading
from datetime import datetime, timedelta
from typing import Dict, Any

from ..dependency_injection import ILoggingService, injectable, singleton
from .interfaces import IRateLimitService
from .types import RateLimitInfo


@singleton(IRateLimitService)
@injectable
class RateLimitService(IRateLimitService):
    """レート制限サービス実装"""

    def __init__(self, logging_service: ILoggingService):
        self.logging_service = logging_service
        self.logger = logging_service.get_logger(__name__, "RateLimitService")

        # カウンター管理
        self._counters: Dict[str, Dict[str, Any]] = {}
        self._counter_lock = threading.RLock()

    def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> RateLimitInfo:
        """レート制限チェック"""
        try:
            with self._counter_lock:
                now = datetime.now()

                # カウンター初期化または期限切れリセット
                if key not in self._counters:
                    self._counters[key] = {
                        'count': 0,
                        'window_start': now,
                        'window_seconds': window_seconds
                    }

                counter_data = self._counters[key]

                # ウィンドウリセットチェック
                if now >= counter_data['window_start'] + timedelta(seconds=window_seconds):
                    counter_data['count'] = 0
                    counter_data['window_start'] = now
                    counter_data['window_seconds'] = window_seconds

                # レート制限チェック
                current_count = counter_data['count']
                reset_time = counter_data['window_start'] + timedelta(seconds=window_seconds)
                is_exceeded = current_count >= limit

                if is_exceeded:
                    self.logger.warning(f"Rate limit exceeded for key: {key}")

                return RateLimitInfo(
                    limit=limit,
                    window_seconds=window_seconds,
                    current_count=current_count,
                    reset_time=reset_time,
                    is_exceeded=is_exceeded
                )

        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            return RateLimitInfo(
                limit=limit,
                window_seconds=window_seconds,
                current_count=0,
                reset_time=datetime.now(),
                is_exceeded=False
            )

    def increment_counter(self, key: str) -> int:
        """カウンター増分"""
        try:
            with self._counter_lock:
                if key in self._counters:
                    self._counters[key]['count'] += 1
                    return self._counters[key]['count']
                return 0

        except Exception as e:
            self.logger.error(f"Counter increment error: {e}")
            return 0

    def reset_counter(self, key: str) -> bool:
        """カウンターリセット"""
        try:
            with self._counter_lock:
                if key in self._counters:
                    self._counters[key]['count'] = 0
                    self._counters[key]['window_start'] = datetime.now()
                    return True
                return False

        except Exception as e:
            self.logger.error(f"Counter reset error: {e}")
            return False