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

    def auto_tune_cache_settings(self) -> Dict[str, Any]:
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

    def _calculate_expected_improvement(self, adjustments: Dict[str, Any]) -> Dict[str, float]:
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