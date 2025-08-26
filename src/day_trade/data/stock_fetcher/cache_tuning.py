"""
キャッシュ自動調整機能
パフォーマンス最適化のための適応的調整機能
"""

from .cache_core import DataCache


class CacheTuningMixin:
    """
    DataCacheクラス用の自動調整機能ミックスイン
    """

    def auto_tune_cache_settings(self) -> dict:
        """
        キャッシュ統計に基づいて設定を自動調整
        パフォーマンス最適化のための適応的調整機能

        Returns:
            調整結果と統計情報
        """
        stats = self.get_cache_stats()
        adjustments = {}

        # 最小要求数に達していない場合は調整しない
        min_requests_for_tuning = 100
        if stats["total_requests"] < min_requests_for_tuning:
            return {
                "adjusted": False,
                "reason": f"insufficient_data (min: {min_requests_for_tuning})",
                "stats": stats,
            }

        original_ttl = self.ttl_seconds
        original_max_size = self.max_size

        # TTL調整ロジック
        if stats["hit_rate"] < 0.4 and stats["stale_hit_rate"] > 0.15:
            # ヒット率が低く、stale hitが多い場合はTTLを延長
            new_ttl = min(int(self.ttl_seconds * 1.5), 900)  # 最大15分
            if new_ttl != self.ttl_seconds:
                self.ttl_seconds = new_ttl
                adjustments["ttl"] = {"old": original_ttl, "new": new_ttl}

        elif stats["hit_rate"] > 0.8 and stats["stale_hit_rate"] < 0.05:
            # ヒット率が高く、stale hitが少ない場合はTTLを短縮
            new_ttl = max(int(self.ttl_seconds * 0.8), 60)  # 最小1分
            if new_ttl != self.ttl_seconds:
                self.ttl_seconds = new_ttl
                adjustments["ttl"] = {"old": original_ttl, "new": new_ttl}

        # キャッシュサイズ調整ロジック
        if (
            stats["cache_utilization"] > 0.9
            and stats["eviction_count"] > stats["hit_count"] * 0.2
        ):
            # 使用率が高く、退避が多い場合はサイズを増加
            new_max_size = min(int(self.max_size * 1.3), 5000)  # 最大5000エントリ
            if new_max_size != self.max_size:
                self.max_size = new_max_size
                adjustments["max_size"] = {
                    "old": original_max_size,
                    "new": new_max_size,
                }

        elif stats["cache_utilization"] < 0.3 and self.max_size > 500:
            # 使用率が低い場合はサイズを減少
            new_max_size = max(int(self.max_size * 0.7), 500)  # 最小500エントリ
            if new_max_size != self.max_size:
                self.max_size = new_max_size
                adjustments["max_size"] = {
                    "old": original_max_size,
                    "new": new_max_size,
                }

                # サイズ縮小時は古いエントリを削除
                self._evict_excess_entries()

        return {
            "adjusted": len(adjustments) > 0,
            "adjustments": adjustments,
            "stats": stats,
            "improvement_expected": self._calculate_expected_improvement(adjustments),
        }

    def _evict_excess_entries(self):
        """キャッシュサイズ縮小時の余剰エントリ削除"""
        if len(self._cache) <= self.max_size:
            return

        # アクセス順序に基づいてLRU削除
        excess_count = len(self._cache) - self.max_size
        keys_to_remove = self._access_order[:excess_count]

        for key in keys_to_remove:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

        self._eviction_count += excess_count

    def _calculate_expected_improvement(self, adjustments: dict) -> dict:
        """調整による期待される改善効果を推定"""
        improvement = {"hit_rate": 0.0, "memory_efficiency": 0.0}

        if "ttl" in adjustments:
            ttl_change = adjustments["ttl"]["new"] / adjustments["ttl"]["old"]
            if ttl_change > 1.0:
                # TTL延長の場合はヒット率向上を期待
                improvement["hit_rate"] = (ttl_change - 1.0) * 0.1  # 控えめな見積もり
            else:
                # TTL短縮の場合はメモリ効率向上
                improvement["memory_efficiency"] = (1.0 - ttl_change) * 0.2

        if "max_size" in adjustments:
            size_change = (
                adjustments["max_size"]["new"] / adjustments["max_size"]["old"]
            )
            if size_change > 1.0:
                # サイズ増加の場合はヒット率向上
                improvement["hit_rate"] += (size_change - 1.0) * 0.05
            else:
                # サイズ縮小の場合はメモリ効率向上
                improvement["memory_efficiency"] += (1.0 - size_change) * 0.3

        return improvement


# DataCacheクラスにミックスインを適用
class TunableDataCache(DataCache, CacheTuningMixin):
    """
    自動調整機能付きDataCache
    """
    pass