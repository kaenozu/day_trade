"""
キャッシュ統計管理モジュール

高性能キャッシュの統計収集と分析機能を提供します。
"""

import time
from typing import Any, Dict, List

from ..logging_config import get_logger

logger = get_logger(__name__)


class CacheStatsManager:
    """
    キャッシュ統計情報を管理するクラス
    """

    def __init__(self):
        """統計管理クラスの初期化"""
        # 統計情報（軽量版）
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
        self._start_time = time.time()
        self._time = time.time

    def record_hit(self) -> None:
        """ヒットを記録"""
        self._hits += 1

    def record_miss(self) -> None:
        """ミスを記録"""
        self._misses += 1

    def record_set(self) -> None:
        """セットを記録"""
        self._sets += 1

    def record_evictions(self, count: int) -> None:
        """エビクションを記録"""
        self._evictions += count

    def get_hit_rate(self) -> float:
        """ヒット率を取得"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self, cache_size: int, max_size: int, cleanup_threshold: float) -> Dict[str, Any]:
        """統計情報を取得"""
        current_time = self._time()
        uptime = current_time - self._start_time
        total_requests = self._hits + self._misses
        
        return {
            # 基本統計
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "evictions": self._evictions,
            "total_requests": total_requests,
            
            # 効率指標
            "hit_rate": self.get_hit_rate(),
            "miss_rate": 1.0 - self.get_hit_rate() if total_requests > 0 else 0.0,
            
            # キャッシュ状態
            "cache_size": cache_size,
            "max_size": max_size,
            "cleanup_threshold": cleanup_threshold,
            "fill_ratio": cache_size / max_size if max_size > 0 else 0.0,
            
            # パフォーマンス指標
            "operations_per_second": total_requests / uptime if uptime > 0 else 0.0,
            "uptime_seconds": uptime,
            "start_time": self._start_time,
        }

    def reset_stats(self) -> None:
        """統計情報をリセット"""
        try:
            self._hits = 0
            self._misses = 0
            self._sets = 0
            self._evictions = 0
            self._start_time = self._time()
            logger.debug("CacheStatsManager stats reset")
            
        except Exception as e:
            logger.error(f"Error resetting stats: {e}", exc_info=True)


class CacheAccessAnalyzer:
    """
    キャッシュアクセスパターンを分析するクラス
    """

    def __init__(self, access_times: Dict[str, float]):
        """
        Args:
            access_times: キーとアクセス時刻の辞書への参照
        """
        self._access_times = access_times

    def get_most_accessed_keys(self, limit: int = 10) -> List[str]:
        """
        最もアクセスされたキーのリストを取得
        
        Args:
            limit: 取得するキーの最大数
            
        Returns:
            アクセス頻度順のキーリスト（新しい順）
        """
        try:
            sorted_items = sorted(
                self._access_times.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            return [key for key, _ in sorted_items[:limit]]
                
        except Exception as e:
            logger.error(f"Error getting most accessed keys: {e}", exc_info=True)
            return []

    def get_least_accessed_keys(self, limit: int = 10) -> List[str]:
        """
        最もアクセスされていないキーのリストを取得
        
        Args:
            limit: 取得するキーの最大数
            
        Returns:
            アクセス頻度順のキーリスト（古い順）
        """
        try:
            sorted_items = sorted(
                self._access_times.items(), 
                key=lambda x: x[1]
            )
            return [key for key, _ in sorted_items[:limit]]
                
        except Exception as e:
            logger.error(f"Error getting least accessed keys: {e}", exc_info=True)
            return []

    def get_access_pattern_summary(self) -> Dict[str, Any]:
        """
        アクセスパターンの要約を取得
        
        Returns:
            アクセスパターンの統計情報
        """
        try:
            if not self._access_times:
                return {
                    "total_keys": 0,
                    "access_time_range": 0.0,
                    "most_recent_access": None,
                    "least_recent_access": None,
                }

            access_times = list(self._access_times.values())
            most_recent = max(access_times)
            least_recent = min(access_times)
            
            return {
                "total_keys": len(self._access_times),
                "access_time_range": most_recent - least_recent,
                "most_recent_access": most_recent,
                "least_recent_access": least_recent,
                "average_access_time": sum(access_times) / len(access_times),
            }
            
        except Exception as e:
            logger.error(f"Error getting access pattern summary: {e}", exc_info=True)
            return {}

    def identify_cold_keys(self, threshold_seconds: float = 3600) -> List[str]:
        """
        長時間アクセスされていないキーを特定
        
        Args:
            threshold_seconds: しきい値（秒）
            
        Returns:
            長時間アクセスされていないキーのリスト
        """
        try:
            current_time = time.time()
            cold_keys = []
            
            for key, access_time in self._access_times.items():
                if current_time - access_time > threshold_seconds:
                    cold_keys.append(key)
            
            return sorted(cold_keys, key=lambda k: self._access_times[k])
            
        except Exception as e:
            logger.error(f"Error identifying cold keys: {e}", exc_info=True)
            return []