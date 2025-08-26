"""
エラーハンドラーの統計情報管理
スレッドセーフな統計収集機能を提供
"""

import logging
import threading
from typing import Dict, Optional

from .config import EnhancedErrorHandlerConfig

logger = logging.getLogger(__name__)


class ErrorHandlerStats:
    """エラーハンドラーの統計情報（スレッドセーフ版）"""

    def __init__(self, config: Optional[EnhancedErrorHandlerConfig] = None):
        """
        Args:
            config: エラーハンドラー設定（依存性注入）
        """
        self._config = config
        self._lock = threading.RLock()
        self._errors_handled = 0
        self._sanitization_count = 0
        self._i18n_fallback_count = 0
        self._rich_display_count = 0
        self._log_errors_count = 0

        if self._config:
            self._lock_timeout = self._config.lock_timeout_seconds
        else:
            self._lock_timeout = 1.0

    def _safe_lock_operation(self, operation_func, default_value=0):
        """安全なロック操作（タイムアウト付き）"""
        try:
            if self._lock.acquire(timeout=self._lock_timeout):
                try:
                    return operation_func()
                except Exception as e:
                    if logger.isEnabledFor(logging.ERROR):
                        logger.error(f"ErrorHandlerStats operation failed: {e}")
                    return default_value
                finally:
                    try:
                        self._lock.release()
                    except Exception as release_error:
                        if logger.isEnabledFor(logging.ERROR):
                            logger.error(
                                f"Failed to release ErrorHandlerStats lock: {release_error}"
                            )
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"ErrorHandlerStats lock timeout ({self._lock_timeout}s)"
                )
            return default_value
        except Exception as e:
            if logger.isEnabledFor(logging.ERROR):
                logger.error(f"ErrorHandlerStats lock acquisition failed: {e}")
            return default_value

    def record_error_handled(self, count: int = 1):
        """エラー処理回数を記録"""
        self._safe_lock_operation(
            lambda: setattr(self, "_errors_handled", self._errors_handled + count)
        )

    def record_sanitization(self, count: int = 1):
        """サニタイズ実行回数を記録"""
        self._safe_lock_operation(
            lambda: setattr(
                self, "_sanitization_count", self._sanitization_count + count
            )
        )

    def record_i18n_fallback(self, count: int = 1):
        """i18nフォールバック回数を記録"""
        self._safe_lock_operation(
            lambda: setattr(
                self, "_i18n_fallback_count", self._i18n_fallback_count + count
            )
        )

    def record_rich_display(self, count: int = 1):
        """Rich表示回数を記録"""
        self._safe_lock_operation(
            lambda: setattr(
                self, "_rich_display_count", self._rich_display_count + count
            )
        )

    def record_log_error(self, count: int = 1):
        """ログエラー回数を記録"""
        self._safe_lock_operation(
            lambda: setattr(self, "_log_errors_count", self._log_errors_count + count)
        )

    def get_stats(self) -> Dict[str, int]:
        """統計情報を取得"""

        def create_stats():
            return {
                "errors_handled": self._errors_handled,
                "sanitization_count": self._sanitization_count,
                "i18n_fallback_count": self._i18n_fallback_count,
                "rich_display_count": self._rich_display_count,
                "log_errors_count": self._log_errors_count,
            }

        result = self._safe_lock_operation(create_stats)
        return result if isinstance(result, dict) else {}

    def reset(self):
        """統計をリセット"""

        def reset_counters():
            self._errors_handled = 0
            self._sanitization_count = 0
            self._i18n_fallback_count = 0
            self._rich_display_count = 0
            self._log_errors_count = 0

        self._safe_lock_operation(reset_counters)