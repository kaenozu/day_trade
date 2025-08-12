"""
キャッシュ統計・メトリクス収集

統合キャッシュシステムの詳細な統計情報とメトリクス収集機能を提供します。
元のcache_utils.pyから統計関連機能を分離・拡張。
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

from .config import get_cache_config

logger = logging.getLogger(__name__)


class CacheMetrics:
    """
    詳細なキャッシュメトリクス収集クラス

    スレッドセーフで高精度な統計情報を提供
    """

    def __init__(self, config=None):
        """
        Args:
            config: キャッシュ設定（Noneの場合はグローバル設定を使用）
        """
        self._config = config or get_cache_config()
        self._lock = threading.RLock()  # 再帰可能ロック

        # 基本統計
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0
        self._evictions = 0
        self._errors = 0
        self._timeouts = 0

        # 時間統計
        self._start_time = time.time()
        self._last_reset_time = self._start_time

        # パフォーマンス統計
        self._operation_times = deque(maxlen=self._config.max_operation_history)
        self._get_times = deque(maxlen=self._config.max_operation_history)
        self._set_times = deque(maxlen=self._config.max_operation_history)

        # 移動平均統計
        self._hit_rate_window = deque(maxlen=self._config.hit_rate_window_size)

        # 詳細統計
        self._cache_sizes = deque(maxlen=100)  # キャッシュサイズ履歴
        self._error_details = deque(maxlen=100)  # エラー詳細履歴

        # 設定
        self._lock_timeout = self._config.lock_timeout

    def _safe_lock_operation(self, operation_func, default_value=0):
        """安全なロック操作（タイムアウト付き・詳細エラーハンドリング）"""
        try:
            if self._lock.acquire(timeout=self._lock_timeout):
                try:
                    return operation_func()
                except Exception as e:
                    # 操作実行中のエラー
                    if logger.isEnabledFor(logging.ERROR):
                        logger.error(
                            f"CacheMetrics operation failed: {e}", exc_info=True
                        )
                    return default_value
                finally:
                    try:
                        self._lock.release()
                    except Exception as release_error:
                        if logger.isEnabledFor(logging.ERROR):
                            logger.error(
                                f"Failed to release CacheMetrics lock: {release_error}"
                            )
            else:
                # ロック取得タイムアウト
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        f"CacheMetrics lock timeout ({self._lock_timeout}s) - returning default value"
                    )
                return default_value
        except Exception as e:
            # ロック取得自体のエラー
            if logger.isEnabledFor(logging.ERROR):
                logger.error(
                    f"CacheMetrics lock acquisition failed: {e}", exc_info=True
                )
            return default_value

    @property
    def hits(self) -> int:
        """ヒット数を取得"""
        return self._safe_lock_operation(lambda: self._hits)

    @property
    def misses(self) -> int:
        """ミス数を取得"""
        return self._safe_lock_operation(lambda: self._misses)

    @property
    def sets(self) -> int:
        """セット数を取得"""
        return self._safe_lock_operation(lambda: self._sets)

    @property
    def deletes(self) -> int:
        """削除数を取得"""
        return self._safe_lock_operation(lambda: self._deletes)

    @property
    def evictions(self) -> int:
        """エビクション数を取得"""
        return self._safe_lock_operation(lambda: self._evictions)

    @property
    def errors(self) -> int:
        """エラー数を取得"""
        return self._safe_lock_operation(lambda: self._errors)

    @property
    def timeouts(self) -> int:
        """タイムアウト数を取得"""
        return self._safe_lock_operation(lambda: self._timeouts)

    @property
    def total_requests(self) -> int:
        """総リクエスト数"""
        return self._safe_lock_operation(lambda: self._hits + self._misses)

    @property
    def hit_rate(self) -> float:
        """ヒット率を計算"""

        def calculate_hit_rate():
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

        result = self._safe_lock_operation(calculate_hit_rate)
        return result if isinstance(result, float) else 0.0

    @property
    def miss_rate(self) -> float:
        """ミス率を計算"""
        return 1.0 - self.hit_rate

    @property
    def windowed_hit_rate(self) -> float:
        """移動平均ヒット率"""

        def calculate_windowed():
            if not self._hit_rate_window:
                return 0.0
            return sum(self._hit_rate_window) / len(self._hit_rate_window)

        result = self._safe_lock_operation(calculate_windowed)
        return result if isinstance(result, float) else 0.0

    def record_hit(self, operation_time: Optional[float] = None) -> None:
        """キャッシュヒットを記録"""

        def record():
            self._hits += 1
            if operation_time is not None:
                self._operation_times.append(operation_time)
                self._get_times.append(operation_time)
            self._update_hit_rate_window(True)

        self._safe_lock_operation(record)

    def record_miss(self, operation_time: Optional[float] = None) -> None:
        """キャッシュミスを記録"""

        def record():
            self._misses += 1
            if operation_time is not None:
                self._operation_times.append(operation_time)
                self._get_times.append(operation_time)
            self._update_hit_rate_window(False)

        self._safe_lock_operation(record)

    def record_set(self, operation_time: Optional[float] = None) -> None:
        """キャッシュセットを記録"""

        def record():
            self._sets += 1
            if operation_time is not None:
                self._operation_times.append(operation_time)
                self._set_times.append(operation_time)

        self._safe_lock_operation(record)

    def record_delete(self) -> None:
        """キャッシュ削除を記録"""

        def record():
            self._deletes += 1

        self._safe_lock_operation(record)

    def record_eviction(self) -> None:
        """エビクションを記録"""

        def record():
            self._evictions += 1

        self._safe_lock_operation(record)

    def record_error(self, error_details: Optional[str] = None) -> None:
        """エラーを記録"""

        def record():
            self._errors += 1
            if error_details:
                self._error_details.append(
                    {"timestamp": time.time(), "details": error_details}
                )

        self._safe_lock_operation(record)

    def record_timeout(self) -> None:
        """タイムアウトを記録"""

        def record():
            self._timeouts += 1

        self._safe_lock_operation(record)

    def record_cache_size(self, size: int) -> None:
        """キャッシュサイズを記録"""

        def record():
            self._cache_sizes.append({"timestamp": time.time(), "size": size})

        self._safe_lock_operation(record)

    def _update_hit_rate_window(self, was_hit: bool) -> None:
        """移動平均ヒット率を更新（内部メソッド、ロック必須）"""
        self._hit_rate_window.append(1.0 if was_hit else 0.0)

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""

        def get_stats():
            stats = {}

            # 平均操作時間
            if self._operation_times:
                stats["avg_operation_time"] = sum(self._operation_times) / len(
                    self._operation_times
                )
                stats["max_operation_time"] = max(self._operation_times)
                stats["min_operation_time"] = min(self._operation_times)
            else:
                stats["avg_operation_time"] = 0.0
                stats["max_operation_time"] = 0.0
                stats["min_operation_time"] = 0.0

            # GET操作の統計
            if self._get_times:
                stats["avg_get_time"] = sum(self._get_times) / len(self._get_times)
                stats["max_get_time"] = max(self._get_times)
            else:
                stats["avg_get_time"] = 0.0
                stats["max_get_time"] = 0.0

            # SET操作の統計
            if self._set_times:
                stats["avg_set_time"] = sum(self._set_times) / len(self._set_times)
                stats["max_set_time"] = max(self._set_times)
            else:
                stats["avg_set_time"] = 0.0
                stats["max_set_time"] = 0.0

            return stats

        return self._safe_lock_operation(get_stats, {})

    def get_error_summary(self) -> Dict[str, Any]:
        """エラーサマリーを取得"""

        def get_summary():
            if not self._error_details:
                return {"total_errors": self._errors, "recent_errors": []}

            recent_errors = list(self._error_details)[-10:]  # 最近の10件
            return {"total_errors": self._errors, "recent_errors": recent_errors}

        return self._safe_lock_operation(
            get_summary, {"total_errors": 0, "recent_errors": []}
        )

    def get_cache_size_trend(self) -> List[Dict[str, Any]]:
        """キャッシュサイズのトレンドを取得"""

        def get_trend():
            return list(self._cache_sizes)

        return self._safe_lock_operation(get_trend, [])

    def get_uptime(self) -> float:
        """稼働時間（秒）を取得"""
        return time.time() - self._start_time

    def get_uptime_since_reset(self) -> float:
        """最後のリセット以降の稼働時間（秒）を取得"""
        return time.time() - self._last_reset_time

    def reset(self) -> None:
        """統計をリセット"""

        def reset_stats():
            self._hits = 0
            self._misses = 0
            self._sets = 0
            self._deletes = 0
            self._evictions = 0
            self._errors = 0
            self._timeouts = 0

            self._operation_times.clear()
            self._get_times.clear()
            self._set_times.clear()
            self._hit_rate_window.clear()
            self._cache_sizes.clear()
            self._error_details.clear()

            self._last_reset_time = time.time()

        self._safe_lock_operation(reset_stats)
        logger.info("Cache metrics reset")

    def to_dict(self) -> Dict[str, Any]:
        """統計情報を辞書形式で取得"""

        def get_dict():
            base_stats = {
                "hits": self._hits,
                "misses": self._misses,
                "sets": self._sets,
                "deletes": self._deletes,
                "evictions": self._evictions,
                "errors": self._errors,
                "timeouts": self._timeouts,
                "total_requests": self._hits + self._misses,
                "hit_rate": self.hit_rate,
                "miss_rate": self.miss_rate,
                "windowed_hit_rate": self.windowed_hit_rate,
                "uptime": self.get_uptime(),
                "uptime_since_reset": self.get_uptime_since_reset(),
            }

            # パフォーマンス統計を追加
            base_stats.update(self.get_performance_stats())

            return base_stats

        return self._safe_lock_operation(get_dict, {})


# core.pyのCacheStatsクラスとの互換性のために
class CacheStats(CacheMetrics):
    """
    基本的なキャッシュ統計情報（core.pyとの互換性維持）
    """

    def __init__(self, config=None):
        """
        Args:
            config: キャッシュ設定（Noneの場合はグローバル設定を使用）
        """
        super().__init__(config)

    def record_hit(self) -> None:
        """ヒットを記録"""
        super().record_hit()

    def record_miss(self) -> None:
        """ミスを記録"""
        super().record_miss()

    def record_set(self) -> None:
        """設定を記録"""
        super().record_set()

    def record_delete(self) -> None:
        """削除を記録"""
        super().record_delete()

    def record_error(self) -> None:
        """エラーを記録"""
        super().record_error()


class CacheStatsCollector:
    """
    複数のキャッシュインスタンスの統計を集約するコレクター
    """

    def __init__(self):
        """統計コレクターを初期化"""
        self._cache_stats: Dict[str, CacheMetrics] = {}
        self._lock = threading.RLock()

    def register_cache(self, cache_name: str, metrics: CacheMetrics) -> None:
        """キャッシュの統計を登録"""
        with self._lock:
            self._cache_stats[cache_name] = metrics

    def unregister_cache(self, cache_name: str) -> None:
        """キャッシュの統計を登録解除"""
        with self._lock:
            if cache_name in self._cache_stats:
                del self._cache_stats[cache_name]

    def get_cache_stats(self, cache_name: str) -> Optional[CacheMetrics]:
        """特定のキャッシュの統計を取得"""
        with self._lock:
            return self._cache_stats.get(cache_name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """全てのキャッシュの統計を取得"""
        with self._lock:
            return {name: stats.to_dict() for name, stats in self._cache_stats.items()}

    def get_aggregated_stats(self) -> Dict[str, Any]:
        """全キャッシュの統計を集約"""
        with self._lock:
            if not self._cache_stats:
                return {}

            aggregated = {
                "total_caches": len(self._cache_stats),
                "total_hits": 0,
                "total_misses": 0,
                "total_sets": 0,
                "total_deletes": 0,
                "total_evictions": 0,
                "total_errors": 0,
                "total_timeouts": 0,
            }

            operation_times = []

            for stats in self._cache_stats.values():
                stats_dict = stats.to_dict()
                aggregated["total_hits"] += stats_dict.get("hits", 0)
                aggregated["total_misses"] += stats_dict.get("misses", 0)
                aggregated["total_sets"] += stats_dict.get("sets", 0)
                aggregated["total_deletes"] += stats_dict.get("deletes", 0)
                aggregated["total_evictions"] += stats_dict.get("evictions", 0)
                aggregated["total_errors"] += stats_dict.get("errors", 0)
                aggregated["total_timeouts"] += stats_dict.get("timeouts", 0)

                # 操作時間の収集
                if "avg_operation_time" in stats_dict:
                    operation_times.append(stats_dict["avg_operation_time"])

            # 集約統計の計算
            total_requests = aggregated["total_hits"] + aggregated["total_misses"]
            aggregated["total_requests"] = total_requests
            aggregated["overall_hit_rate"] = (
                aggregated["total_hits"] / total_requests if total_requests > 0 else 0.0
            )
            aggregated["overall_miss_rate"] = 1.0 - aggregated["overall_hit_rate"]

            if operation_times:
                aggregated["avg_operation_time"] = sum(operation_times) / len(
                    operation_times
                )
            else:
                aggregated["avg_operation_time"] = 0.0

            return aggregated


# グローバル統計コレクター
_global_stats_collector: Optional[CacheStatsCollector] = None
_collector_lock = threading.RLock()


def get_global_stats_collector() -> CacheStatsCollector:
    """グローバル統計コレクターを取得"""
    global _global_stats_collector

    if _global_stats_collector is None:
        with _collector_lock:
            if _global_stats_collector is None:
                _global_stats_collector = CacheStatsCollector()

    return _global_stats_collector


def reset_global_stats_collector() -> None:
    """グローバル統計コレクターをリセット（テスト用）"""
    global _global_stats_collector

    with _collector_lock:
        _global_stats_collector = None
