"""
株価データ取得用コアキャッシュシステム
DataCacheクラスの中核機能
"""

import os
import time
from typing import Dict, Optional, Any


class DataCache:
    """高度なデータキャッシュクラス（フォールバック機能付き）"""

    def __init__(
        self,
        ttl_seconds: int = None,  # デフォルトは環境変数から取得
        max_size: int = None,  # デフォルトは環境変数から取得
        stale_while_revalidate: int = None,  # デフォルトは環境変数から取得
    ):
        """
        Args:
            ttl_seconds: キャッシュの有効期限（秒）
            max_size: 最大キャッシュサイズ（LRU eviction）
            stale_while_revalidate: 期限切れ後もフォールバックとして利用可能な期間（秒）
        """
        # 環境変数から設定を取得（デフォルト値付き）
        self.ttl_seconds = ttl_seconds or int(
            os.getenv("STOCK_CACHE_TTL_SECONDS", "300")
        )  # 5分
        self.max_size = max_size or int(os.getenv("STOCK_CACHE_MAX_SIZE", "2000"))
        self.stale_while_revalidate = stale_while_revalidate or int(
            os.getenv("STOCK_CACHE_STALE_SECONDS", "600")
        )  # 10分
        self._cache = {}
        self._access_order = []  # LRU tracking

        # パフォーマンス統計
        self._hit_count = 0
        self._miss_count = 0
        self._stale_hit_count = 0
        self._eviction_count = 0

    def get(self, key: str, allow_stale: bool = False) -> Optional[any]:
        """
        キャッシュから値を取得

        Args:
            key: キャッシュキー
            allow_stale: 期限切れキャッシュも許可するか
        """
        if key in self._cache:
            cache_data = self._cache[key]
            current_time = time.time()
            
            # 新形式（タイムスタンプ、値、TTL）と旧形式（値、タイムスタンプ）の両方をサポート
            if len(cache_data) >= 3:
                value, timestamp, expiry_time = cache_data
                age = current_time - timestamp
            else:
                value, timestamp = cache_data
                age = current_time - timestamp

            # フレッシュなキャッシュ
            if age < self.ttl_seconds:
                self._update_access_order(key)
                self._hit_count += 1
                return value

            # stale-while-revalidate期間内のキャッシュ
            elif allow_stale and age < self.ttl_seconds + self.stale_while_revalidate:
                self._update_access_order(key)
                self._stale_hit_count += 1
                return value

            # 完全に期限切れ
            else:
                self._remove_key(key)

        self._miss_count += 1
        return None

    def set(self, key: str, value: any, ttl: int = None) -> None:
        """キャッシュに値を設定（LRU eviction付き）"""
        current_time = time.time()

        # キャッシュサイズ制限の確認
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        # TTLが指定されている場合は使用、そうでなければデフォルトTTLを使用
        effective_ttl = ttl or self.ttl_seconds
        expiry_time = current_time + effective_ttl
        self._cache[key] = (value, current_time, expiry_time)
        self._update_access_order(key)

    def _update_access_order(self, key: str) -> None:
        """アクセス順序を更新（LRU tracking）"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict_lru(self) -> None:
        """最も古いキャッシュエントリを削除"""
        if self._access_order:
            lru_key = self._access_order[0]
            self._remove_key(lru_key)
            self._eviction_count += 1

    def _remove_key(self, key: str) -> None:
        """キーをキャッシュから完全に削除"""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)

    def clear(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
        self._access_order.clear()
        # 統計もリセット
        self._hit_count = 0
        self._miss_count = 0
        self._stale_hit_count = 0
        self._eviction_count = 0

    def size(self) -> int:
        """キャッシュサイズを取得"""
        return len(self._cache)

    def get_cache_stats(self) -> dict:
        """キャッシュ統計を取得"""
        total_requests = self._hit_count + self._miss_count + self._stale_hit_count

        if total_requests == 0:
            hit_rate = 0.0
            stale_hit_rate = 0.0
        else:
            hit_rate = self._hit_count / total_requests
            stale_hit_rate = self._stale_hit_count / total_requests

        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "stale_hit_count": self._stale_hit_count,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "stale_hit_rate": stale_hit_rate,
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "eviction_count": self._eviction_count,
            "cache_utilization": (
                len(self._cache) / self.max_size if self.max_size > 0 else 0.0
            ),
        }

    def get_cache_info(self) -> Dict[str, any]:
        """キャッシュ統計情報を取得"""
        current_time = time.time()
        fresh_count = 0
        stale_count = 0

        for _, cache_data in self._cache.items():
            if len(cache_data) >= 3:
                _, timestamp, _ = cache_data
            else:
                _, timestamp = cache_data
                
            age = current_time - timestamp
            if age < self.ttl_seconds:
                fresh_count += 1
            elif age < self.ttl_seconds + self.stale_while_revalidate:
                stale_count += 1

        return {
            "total_entries": len(self._cache),
            "fresh_entries": fresh_count,
            "stale_entries": stale_count,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "stale_while_revalidate": self.stale_while_revalidate,
        }

    def optimize_cache_settings(self) -> dict:
        """キャッシュ統計に基づいた設定最適化の提案"""
        stats = self.get_cache_stats()
        recommendations = {}

        # ヒット率が低い場合の提案
        if stats["hit_rate"] < 0.5:
            recommendations["ttl_increase"] = (
                "TTLを延長してキャッシュヒット率を向上させることを検討"
            )

        # 退避回数が多い場合の提案
        if stats["eviction_count"] > stats["hit_count"] * 0.1:
            recommendations["size_increase"] = "キャッシュサイズを増加させることを検討"

        # キャッシュ使用率が低い場合の提案
        if stats["cache_utilization"] < 0.3:
            recommendations["size_decrease"] = (
                "キャッシュサイズを減少させてメモリ効率を向上"
            )

        # stale hitが多い場合の提案
        if stats["stale_hit_rate"] > 0.2:
            recommendations["stale_period_adjust"] = (
                "stale-while-revalidate期間の調整を検討"
            )

        return {"current_stats": stats, "recommendations": recommendations}