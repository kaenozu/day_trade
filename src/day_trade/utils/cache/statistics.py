"""
キャッシュ統計管理モジュール

キャッシュの統計情報を収集、管理、分析するための機能を提供します。
スレッドセーフな実装で、ヒット率、エラー率、パフォーマンス指標などを追跡します。
"""

import threading
import time
from typing import Dict, List, Union

from ..logging_config import get_logger
from .config import get_cache_config
from .constants import CacheConstants

logger = get_logger(__name__)


class CacheStats:
    """キャッシュ統計情報（スレッドセーフ版・デッドロック対策強化・パフォーマンス最適化）"""

    def __init__(self, config=None):
        """
        Args:
            config: キャッシュ設定（Noneの場合はデフォルト設定を使用）
        """
        self._config = config or get_cache_config()
        self._lock = threading.RLock()  # 再帰可能ロック
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
        self._errors = 0
        self._lock_timeout = self._config.lock_timeout

        # パフォーマンス最適化: アトミック操作用の追加統計
        self._start_time = time.time()
        self._last_reset_time = self._start_time

        # 高精度統計用（時間加重平均など）
        self._operation_times = []
        self._max_operation_history = self._config.max_operation_history

        # キャッシュ効率統計
        self._hit_rate_window = []
        self._window_size = self._config.hit_rate_window_size

    def _safe_lock_operation(self, operation_func, default_value=0):
        """安全なロック操作（タイムアウト付き・詳細エラーハンドリング）"""
        try:
            if self._lock.acquire(timeout=self._lock_timeout):
                try:
                    return operation_func()
                except Exception as e:
                    # 操作実行中のエラー
                    if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logger.ERROR):
                        logger.error(f"CacheStats operation failed: {e}", exc_info=True)
                    return default_value
                finally:
                    try:
                        self._lock.release()
                    except Exception as release_error:
                        if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logger.ERROR):
                            logger.error(f"Failed to release CacheStats lock: {release_error}")
            else:
                # ロック取得タイムアウト
                if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logger.WARNING):
                    logger.warning(
                        f"CacheStats lock timeout ({self._lock_timeout}s) - returning default value"
                    )
                return default_value
        except Exception as e:
            # ロック取得自体のエラー
            if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logger.ERROR):
                logger.error(f"CacheStats lock acquisition failed: {e}", exc_info=True)
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
    def evictions(self) -> int:
        """エビクション数を取得"""
        return self._safe_lock_operation(lambda: self._evictions)

    @property
    def errors(self) -> int:
        """エラー数を取得"""
        return self._safe_lock_operation(lambda: self._errors)

    @property
    def hit_rate(self) -> float:
        """ヒット率を計算"""

        def calculate_hit_rate():
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0

        result = self._safe_lock_operation(calculate_hit_rate)
        return result if isinstance(result, float) else 0.0

    def record_hit(self, count: int = 1) -> None:
        """キャッシュヒットを記録（移動平均更新付き）"""

        def record_with_moving_avg():
            self._increment_counter("_hits", count)
            # 移動平均ヒット率の更新
            total_requests = self._hits + self._misses
            if total_requests > 0:
                current_hit_rate = self._hits / total_requests
                self._update_hit_rate_window(current_hit_rate)
            # 操作時刻を記録
            self._record_operation_time()

        self._safe_lock_operation(record_with_moving_avg)

    def record_miss(self, count: int = 1) -> None:
        """キャッシュミスを記録（移動平均更新付き）"""

        def record_with_moving_avg():
            self._increment_counter("_misses", count)
            # 移動平均ヒット率の更新
            total_requests = self._hits + self._misses
            if total_requests > 0:
                current_hit_rate = self._hits / total_requests
                self._update_hit_rate_window(current_hit_rate)
            # 操作時刻を記録
            self._record_operation_time()

        self._safe_lock_operation(record_with_moving_avg)

    def record_set(self, count: int = 1) -> None:
        """キャッシュセットを記録"""
        self._safe_lock_operation(lambda: self._increment_counter("_sets", count))

    def record_eviction(self, count: int = 1) -> None:
        """キャッシュエビクションを記録"""
        self._safe_lock_operation(lambda: self._increment_counter("_evictions", count))

    def record_error(self, count: int = 1) -> None:
        """キャッシュエラーを記録"""
        self._safe_lock_operation(lambda: self._increment_counter("_errors", count))

    def _increment_counter(self, counter_name: str, count: int) -> None:
        """カウンターをインクリメント（ロック内で呼び出し・エラーハンドリング強化）"""
        try:
            if not hasattr(self, counter_name):
                raise AttributeError(f"Counter '{counter_name}' does not exist")

            if not isinstance(count, int) or count < 0:
                raise ValueError(f"Count must be a non-negative integer, got: {count}")

            current_value = getattr(self, counter_name)
            new_value = current_value + count

            # オーバーフロー検査（Python の int は無限精度だが、異常に大きな値をチェック）
            if new_value > CacheConstants.MAX_COUNTER_VALUE:
                if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logger.WARNING):
                    logger.warning(
                        f"Counter '{counter_name}' approaching overflow, resetting to 0"
                    )
                setattr(self, counter_name, 0)
            else:
                setattr(self, counter_name, new_value)

        except (AttributeError, ValueError, TypeError) as e:
            if hasattr(logger, "isEnabledFor") and logger.isEnabledFor(logger.ERROR):
                logger.error(f"Failed to increment counter '{counter_name}': {e}")
            # エラーが発生した場合はカウントを無視

    def record_fallback(self, count: int = 1) -> None:
        """フォールバック使用を記録（エラーとして扱う）"""
        self.record_error(count)

    def reset(self) -> None:
        """統計をリセット（履歴情報も含む）"""

        def reset_counters():
            self._hits = 0
            self._misses = 0
            self._sets = 0
            self._evictions = 0
            self._errors = 0

            # 履歴情報もリセット
            self._operation_times.clear()
            self._hit_rate_window.clear()
            self._last_reset_time = time.time()

        self._safe_lock_operation(reset_counters)

    def to_dict(self) -> Dict[str, Union[int, float]]:
        """統計情報を辞書として返す（スレッドセーフ・詳細統計付き）"""

        def create_stats_dict():
            current_time = time.time()
            uptime = current_time - self._start_time
            time_since_reset = current_time - self._last_reset_time

            # 基本統計
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            # 効率指標
            operations_per_second = total_requests / uptime if uptime > 0 else 0.0
            recent_ops_per_second = (
                total_requests / time_since_reset if time_since_reset > 0 else 0.0
            )

            # 移動平均ヒット率
            moving_avg_hit_rate = (
                sum(self._hit_rate_window) / len(self._hit_rate_window)
                if self._hit_rate_window
                else hit_rate
            )

            return {
                # 基本統計
                "hits": self._hits,
                "misses": self._misses,
                "sets": self._sets,
                "evictions": self._evictions,
                "errors": self._errors,
                "total_requests": total_requests,
                # 効率指標
                "hit_rate": hit_rate,
                "moving_avg_hit_rate": moving_avg_hit_rate,
                "miss_rate": 1.0 - hit_rate if total_requests > 0 else 0.0,
                # パフォーマンス指標
                "operations_per_second": operations_per_second,
                "recent_operations_per_second": recent_ops_per_second,
                "uptime_seconds": uptime,
                "time_since_reset_seconds": time_since_reset,
                # 高度な指標
                "error_rate": (
                    self._errors / total_requests if total_requests > 0 else 0.0
                ),
                "eviction_rate": (
                    self._evictions / self._sets if self._sets > 0 else 0.0
                ),
                "efficiency_score": self._calculate_efficiency_score(
                    hit_rate,
                    self._errors / total_requests if total_requests > 0 else 0.0,
                ),
                # メタデータ
                "last_reset_time": self._last_reset_time,
                "start_time": self._start_time,
                "operation_history_size": len(self._operation_times),
                "hit_rate_window_size": len(self._hit_rate_window),
            }

        result = self._safe_lock_operation(create_stats_dict)
        return result if isinstance(result, dict) else {}

    def _calculate_efficiency_score(self, hit_rate: float, error_rate: float) -> float:
        """キャッシュ効率スコアを計算（0-100の範囲）"""
        # ヒット率を基本スコアとし、エラー率でペナルティを課す
        base_score = hit_rate * 100
        error_penalty = error_rate * CacheConstants.ERROR_PENALTY_MULTIPLIER
        return max(0.0, base_score - error_penalty)

    def _update_hit_rate_window(self, hit_rate: float) -> None:
        """移動平均ヒット率ウィンドウを更新（ロック内で呼び出し前提）"""
        self._hit_rate_window.append(hit_rate)
        # ウィンドウサイズを超えた場合は古いデータを削除
        if len(self._hit_rate_window) > self._window_size:
            self._hit_rate_window.pop(0)

    def _record_operation_time(self) -> None:
        """操作時刻を記録（ロック内で呼び出し前提）"""
        current_time = time.time()
        self._operation_times.append(current_time)
        # 履歴サイズを制限
        if len(self._operation_times) > self._max_operation_history:
            self._operation_times.pop(0)

    def get_recent_operations_per_second(self, window_seconds: float = 60.0) -> float:
        """指定時間内の操作数/秒を取得（スレッドセーフ）"""

        def calculate_recent_ops():
            if not self._operation_times:
                return 0.0

            current_time = time.time()
            cutoff_time = current_time - window_seconds

            # 指定時間内の操作数をカウント
            recent_ops = sum(
                1 for op_time in self._operation_times if op_time >= cutoff_time
            )

            return recent_ops / window_seconds if window_seconds > 0 else 0.0

        result = self._safe_lock_operation(calculate_recent_ops)
        return result if isinstance(result, (int, float)) else 0.0

    def get_peak_operations_per_second(self, window_seconds: float = 10.0) -> float:
        """ピーク時の操作数/秒を取得（スレッドセーフ）"""

        def calculate_peak_ops():
            if len(self._operation_times) < 2:
                return 0.0

            max_ops_per_second = 0.0
            current_time = time.time()

            # 時間窓をスライドさせながらピークを探す
            for i in range(len(self._operation_times)):
                window_start = self._operation_times[i]
                if (
                    current_time - window_start
                    > CacheConstants.DEFAULT_STATS_HISTORY_CLEANUP_SECONDS
                ):
                    continue

                window_end = window_start + window_seconds
                ops_in_window = sum(
                    1
                    for op_time in self._operation_times
                    if window_start <= op_time <= window_end
                )

                ops_per_second = ops_in_window / window_seconds
                max_ops_per_second = max(max_ops_per_second, ops_per_second)

            return max_ops_per_second

        result = self._safe_lock_operation(calculate_peak_ops)
        return result if isinstance(result, (int, float)) else 0.0

    def add_stats(self, other_stats: "CacheStats") -> None:
        """他の統計情報を追加（デッドロック回避）"""
        # デッドロックを避けるため、まず他の統計を安全に取得
        other_dict = other_stats.to_dict()

        def add_other_stats():
            self._hits += other_dict.get("hits", 0)
            self._misses += other_dict.get("misses", 0)
            self._sets += other_dict.get("sets", 0)
            self._evictions += other_dict.get("evictions", 0)
            self._errors += other_dict.get("errors", 0)

        self._safe_lock_operation(add_other_stats)

    def get_summary_report(self) -> str:
        """統計情報の要約レポートを生成"""
        stats = self.to_dict()
        
        report_lines = [
            "=== Cache Statistics Summary ===",
            f"Total Requests: {stats['total_requests']:,}",
            f"Hit Rate: {stats['hit_rate']:.2%}",
            f"Miss Rate: {stats['miss_rate']:.2%}",
            f"Error Rate: {stats['error_rate']:.2%}",
            f"Efficiency Score: {stats['efficiency_score']:.1f}/100",
            "",
            f"Operations/Second: {stats['operations_per_second']:.2f}",
            f"Recent Operations/Second: {stats['recent_operations_per_second']:.2f}",
            f"Uptime: {stats['uptime_seconds']:.1f} seconds",
            "",
            f"Cache Operations:",
            f"  Hits: {stats['hits']:,}",
            f"  Misses: {stats['misses']:,}",
            f"  Sets: {stats['sets']:,}",
            f"  Evictions: {stats['evictions']:,}",
            f"  Errors: {stats['errors']:,}",
        ]
        
        return "\n".join(report_lines)