"""
キャッシュコアインターフェース

統合キャッシュシステムの基本インターフェースと抽象クラスを定義します。
他のキャッシュ実装の基盤となります。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

from ..utils.logging_config import get_context_logger

logger = get_logger(__name__)

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class CacheInterface(Generic[K, V], ABC):
    """キャッシュの基本インターフェース"""

    @abstractmethod
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        キーに対応する値を取得

        Args:
            key: キー
            default: キーが存在しない場合のデフォルト値

        Returns:
            値またはデフォルト値
        """
        pass

    @abstractmethod
    def set(self, key: K, value: V, ttl: Optional[int] = None) -> bool:
        """
        キーと値のペアを設定

        Args:
            key: キー
            value: 値
            ttl: 生存時間（秒）

        Returns:
            設定が成功したかどうか
        """
        pass

    @abstractmethod
    def delete(self, key: K) -> bool:
        """
        キーとその値を削除

        Args:
            key: 削除するキー

        Returns:
            削除が成功したかどうか
        """
        pass

    @abstractmethod
    def exists(self, key: K) -> bool:
        """
        キーが存在するかチェック

        Args:
            key: チェックするキー

        Returns:
            キーが存在するかどうか
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """全てのキャッシュエントリを削除"""
        pass

    @abstractmethod
    def size(self) -> int:
        """キャッシュに保存されているアイテム数を返す"""
        pass

    def get_many(self, keys: list[K]) -> Dict[K, V]:
        """
        複数のキーに対応する値を一括取得

        Args:
            keys: キーのリスト

        Returns:
            キーと値のマッピング
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def set_many(self, mapping: Dict[K, V], ttl: Optional[int] = None) -> int:
        """
        複数のキーと値のペアを一括設定

        Args:
            mapping: キーと値のマッピング
            ttl: 生存時間（秒）

        Returns:
            成功した設定数
        """
        success_count = 0
        for key, value in mapping.items():
            if self.set(key, value, ttl):
                success_count += 1
        return success_count

    def delete_many(self, keys: list[K]) -> int:
        """
        複数のキーを一括削除

        Args:
            keys: 削除するキーのリスト

        Returns:
            成功した削除数
        """
        success_count = 0
        for key in keys:
            if self.delete(key):
                success_count += 1
        return success_count


class CacheStats:
    """基本的なキャッシュ統計情報"""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0

    @property
    def total_requests(self) -> int:
        """総リクエスト数"""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """ヒット率（0.0-1.0）"""
        total = self.total_requests
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        """ミス率（0.0-1.0）"""
        return 1.0 - self.hit_rate

    def record_hit(self) -> None:
        """ヒットを記録"""
        self.hits += 1

    def record_miss(self) -> None:
        """ミスを記録"""
        self.misses += 1

    def record_set(self) -> None:
        """設定を記録"""
        self.sets += 1

    def record_delete(self) -> None:
        """削除を記録"""
        self.deletes += 1

    def record_error(self) -> None:
        """エラーを記録"""
        self.errors += 1

    def reset(self) -> None:
        """統計をリセット"""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0

    def to_dict(self) -> Dict[str, Any]:
        """統計情報を辞書形式で取得"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
        }


class BaseCacheManager(CacheInterface[str, Any]):
    """
    基本キャッシュマネージャー

    統計情報収集機能付きの基本実装を提供します。
    """

    def __init__(self, enable_stats: bool = True):
        """
        Args:
            enable_stats: 統計収集を有効にするか
        """
        self.enable_stats = enable_stats
        self._stats = CacheStats() if enable_stats else None

    @property
    def stats(self) -> Optional[CacheStats]:
        """統計情報を取得"""
        return self._stats

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を辞書形式で取得"""
        if self._stats:
            return self._stats.to_dict()
        return {}

    def reset_stats(self) -> None:
        """統計情報をリセット"""
        if self._stats:
            self._stats.reset()

    def _record_hit(self) -> None:
        """ヒットを記録（内部用）"""
        if self._stats:
            self._stats.record_hit()

    def _record_miss(self) -> None:
        """ミスを記録（内部用）"""
        if self._stats:
            self._stats.record_miss()

    def _record_set(self) -> None:
        """設定を記録（内部用）"""
        if self._stats:
            self._stats.record_set()

    def _record_delete(self) -> None:
        """削除を記録（内部用）"""
        if self._stats:
            self._stats.record_delete()

    def _record_error(self) -> None:
        """エラーを記録（内部用）"""
        if self._stats:
            self._stats.record_error()

    # 抽象メソッドのデフォルト実装（サブクラスでオーバーライド必須）
    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        raise NotImplementedError("Subclass must implement get method")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        raise NotImplementedError("Subclass must implement set method")

    def delete(self, key: str) -> bool:
        raise NotImplementedError("Subclass must implement delete method")

    def exists(self, key: str) -> bool:
        raise NotImplementedError("Subclass must implement exists method")

    def clear(self) -> None:
        raise NotImplementedError("Subclass must implement clear method")

    def size(self) -> int:
        raise NotImplementedError("Subclass must implement size method")


class CacheEntry:
    """キャッシュエントリ（TTL対応）"""

    def __init__(
        self, value: Any, ttl: Optional[int] = None, timestamp: Optional[float] = None
    ):
        """
        Args:
            value: 保存する値
            ttl: 生存時間（秒）
            timestamp: 作成時刻（指定しない場合は現在時刻）
        """
        import time

        self.value = value
        self.ttl = ttl
        self.created_at = timestamp if timestamp is not None else time.time()
        self.accessed_at = self.created_at
        self.access_count = 0

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """
        エントリが期限切れかチェック

        Args:
            current_time: 現在時刻（指定しない場合は現在時刻を使用）

        Returns:
            期限切れかどうか
        """
        if self.ttl is None:
            return False

        import time

        if current_time is None:
            current_time = time.time()

        return (current_time - self.created_at) > self.ttl

    def touch(self, current_time: Optional[float] = None) -> None:
        """
        エントリにアクセスしたことを記録

        Args:
            current_time: 現在時刻
        """
        import time

        self.accessed_at = current_time if current_time is not None else time.time()
        self.access_count += 1

    def get_age(self, current_time: Optional[float] = None) -> float:
        """
        エントリの生存時間を取得

        Args:
            current_time: 現在時刻

        Returns:
            生存時間（秒）
        """
        import time

        if current_time is None:
            current_time = time.time()

        return current_time - self.created_at

    def get_remaining_ttl(
        self, current_time: Optional[float] = None
    ) -> Optional[float]:
        """
        残りTTLを取得

        Args:
            current_time: 現在時刻

        Returns:
            残りTTL（秒）、TTLが設定されていない場合はNone
        """
        if self.ttl is None:
            return None

        import time

        if current_time is None:
            current_time = time.time()

        remaining = self.ttl - (current_time - self.created_at)
        return max(0, remaining)

    def __repr__(self) -> str:
        return (
            f"CacheEntry(value={self.value}, ttl={self.ttl}, age={self.get_age():.2f})"
        )
