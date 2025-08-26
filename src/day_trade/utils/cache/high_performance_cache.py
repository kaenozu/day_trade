"""
高性能キャッシュモジュール

超高性能キャッシュ実装を提供します。
最小限のロックと最適化されたデータ構造を使用し、
read-heavy workloadに最適化されています。
"""

from typing import Any, Dict, List, Optional

from ..logging_config import get_logger
from .basic_operations import BasicCacheOperations
from .config import get_cache_config
from .stats_manager import CacheStatsManager, CacheAccessAnalyzer

logger = get_logger(__name__)


class HighPerformanceCache:
    """
    超高性能キャッシュ実装
    最小限のロックと最適化されたデータ構造を使用
    """

    def __init__(
        self, 
        max_size: Optional[int] = None, 
        config: Optional["CacheConfig"] = None
    ):
        """
        Args:
            max_size: 最大キャッシュサイズ（Noneの場合は設定から取得）
            config: キャッシュ設定（Noneの場合はデフォルト設定を使用）
        """
        self._config = config or get_cache_config()
        
        # 基本操作を委譲するクラス
        self._operations = BasicCacheOperations(max_size, config)
        
        # 統計管理
        self._stats = CacheStatsManager()
        
        # アクセス分析（基本操作クラスのアクセス時間辞書を参照）
        self._analyzer = CacheAccessAnalyzer(self._operations._access_times)
        
        logger.debug("HighPerformanceCache initialized with modular architecture")

    def get(self, key: str) -> Any:
        """
        超高速get操作
        
        Args:
            key: 取得するキー
            
        Returns:
            キャッシュされた値、存在しない場合はNone
        """
        result = self._operations.get(key)
        if result is not None:
            self._stats.record_hit()
        else:
            self._stats.record_miss()
        return result

    def set(self, key: str, value: Any) -> bool:
        """
        高速set操作
        
        Args:
            key: 設定するキー
            value: 設定する値
            
        Returns:
            設定に成功したかどうか
        """
        success = self._operations.set(key, value)
        if success:
            self._stats.record_set()
        return success

    def delete(self, key: str) -> bool:
        """
        キーの削除
        
        Args:
            key: 削除するキー
            
        Returns:
            削除に成功したかどうか
        """
        return self._operations.delete(key)

    def clear(self) -> None:
        """キャッシュのクリア"""
        self._operations.clear()

    def size(self) -> int:
        """現在のキャッシュサイズ"""
        return self._operations.size()

    def is_empty(self) -> bool:
        """キャッシュが空かどうか"""
        return self._operations.is_empty()

    def contains(self, key: str) -> bool:
        """キーが存在するかどうか"""
        return self._operations.contains(key)

    def get_hit_rate(self) -> float:
        """ヒット率を取得"""
        return self._stats.get_hit_rate()

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return self._stats.get_stats(
            self.size(), 
            self._operations._max_size, 
            self._operations._cleanup_threshold
        )

    def get_keys(self) -> List[str]:
        """現在のキーリストを取得（デバッグ用）"""
        return self._operations.get_keys()

    def get_most_accessed_keys(self, limit: int = 10) -> List[str]:
        """
        最もアクセスされたキーのリストを取得
        
        Args:
            limit: 取得するキーの最大数
            
        Returns:
            アクセス頻度順のキーリスト（新しい順）
        """
        return self._analyzer.get_most_accessed_keys(limit)

    def get_least_accessed_keys(self, limit: int = 10) -> List[str]:
        """
        最もアクセスされていないキーのリストを取得
        
        Args:
            limit: 取得するキーの最大数
            
        Returns:
            アクセス頻度順のキーリスト（古い順）
        """
        return self._analyzer.get_least_accessed_keys(limit)

    def trim_to_size(self, target_size: int) -> int:
        """
        指定サイズまでキャッシュを縮小
        
        Args:
            target_size: 目標サイズ
            
        Returns:
            削除されたエントリの数
        """
        evicted_count = self._operations.trim_to_size(target_size)
        if evicted_count > 0:
            self._stats.record_evictions(evicted_count)
        return evicted_count

    def reset_stats(self) -> None:
        """統計情報をリセット"""
        self._stats.reset_stats()

    def get_access_pattern_summary(self) -> Dict[str, Any]:
        """アクセスパターンの要約を取得"""
        return self._analyzer.get_access_pattern_summary()

    def identify_cold_keys(self, threshold_seconds: float = 3600) -> List[str]:
        """
        長時間アクセスされていないキーを特定
        
        Args:
            threshold_seconds: しきい値（秒）
            
        Returns:
            長時間アクセスされていないキーのリスト
        """
        return self._analyzer.identify_cold_keys(threshold_seconds)

    def optimize(self) -> Dict[str, Any]:
        """
        キャッシュを最適化し、結果を返す
        
        Returns:
            最適化の結果情報
        """
        try:
            initial_size = self.size()
            initial_stats = self.get_stats()
            
            # コールドキーを特定して削除
            cold_keys = self.identify_cold_keys(threshold_seconds=1800)  # 30分
            removed_cold_keys = 0
            
            for key in cold_keys:
                if self.delete(key):
                    removed_cold_keys += 1
            
            # 必要に応じてサイズを調整
            target_size = int(self._operations._max_size * 0.8)  # 80%に調整
            trimmed_count = self.trim_to_size(target_size)
            
            final_size = self.size()
            final_stats = self.get_stats()
            
            result = {
                "initial_size": initial_size,
                "final_size": final_size,
                "removed_cold_keys": removed_cold_keys,
                "trimmed_entries": trimmed_count,
                "total_removed": removed_cold_keys + trimmed_count,
                "size_reduction_ratio": (initial_size - final_size) / initial_size if initial_size > 0 else 0.0,
                "hit_rate_before": initial_stats.get("hit_rate", 0.0),
                "hit_rate_after": final_stats.get("hit_rate", 0.0),
            }
            
            logger.info(f"HighPerformanceCache optimized: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during cache optimization: {e}", exc_info=True)
            return {"error": str(e)}