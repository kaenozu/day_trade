#!/usr/bin/env python3
"""
スマートキャッシュ無効化戦略
Issue #377: 高度なキャッシング戦略の導入

TTL, LRU, LFU, 依存性ベース、イベント駆動など高度な無効化戦略を実装
"""

import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class InvalidationTrigger(Enum):
    """無効化トリガー種類"""

    TTL_EXPIRED = auto()  # TTL期限切れ
    LRU_EVICTION = auto()  # LRU削除
    LFU_EVICTION = auto()  # LFU削除
    DEPENDENCY_CHANGED = auto()  # 依存性変更
    EVENT_TRIGGERED = auto()  # イベント駆動
    MANUAL_INVALIDATION = auto()  # 手動無効化
    MEMORY_PRESSURE = auto()  # メモリ圧迫
    STALE_DETECTION = auto()  # 陳腐化検出


@dataclass
class CacheEntry:
    """拡張キャッシュエントリ"""

    key: str
    value: Any
    created_at: float
    expires_at: float
    last_accessed: float
    access_count: int = 0
    access_frequency: float = 0.0  # アクセス頻度（アクセス数/経過時間）
    size_bytes: int = 0
    priority: float = 1.0
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)  # 依存キー
    version: int = 1
    stale_threshold: float = 0.5  # 陳腐化しきい値

    def __post_init__(self):
        if self.access_count == 0:
            self.access_count = 1
            self.last_accessed = self.created_at
        self.update_frequency()

    def update_frequency(self):
        """アクセス頻度更新"""
        age = max(time.time() - self.created_at, 1.0)  # 最低1秒
        self.access_frequency = self.access_count / age

    def record_access(self):
        """アクセス記録"""
        self.last_accessed = time.time()
        self.access_count += 1
        self.update_frequency()

    @property
    def is_expired(self) -> bool:
        """TTL期限切れチェック"""
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """経過時間（秒）"""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """アイドル時間（秒）"""
        return time.time() - self.last_accessed

    @property
    def is_stale(self) -> bool:
        """陳腐化判定"""
        # アクセス頻度が低下しているか
        recent_access_rate = 1.0 / max(self.idle_seconds, 1.0)
        return recent_access_rate < self.stale_threshold


class InvalidationStrategy(ABC):
    """無効化戦略抽象基底クラス"""

    @abstractmethod
    def should_evict(self, entry: CacheEntry) -> Tuple[bool, InvalidationTrigger]:
        """無効化すべきかどうかの判定"""
        pass

    @abstractmethod
    def get_eviction_candidates(
        self, entries: Dict[str, CacheEntry], count: int
    ) -> List[str]:
        """削除候補選択"""
        pass

    @abstractmethod
    def on_access(self, entry: CacheEntry):
        """アクセス時処理"""
        pass

    @abstractmethod
    def get_strategy_stats(self) -> Dict[str, Any]:
        """戦略固有統計情報"""
        pass


class TTLStrategy(InvalidationStrategy):
    """TTL（Time To Live）無効化戦略"""

    def __init__(self):
        self._stats = {"ttl_evictions": 0, "expired_checks": 0}

    def should_evict(self, entry: CacheEntry) -> Tuple[bool, InvalidationTrigger]:
        """TTL期限切れチェック"""
        self._stats["expired_checks"] += 1

        if entry.is_expired:
            self._stats["ttl_evictions"] += 1
            return True, InvalidationTrigger.TTL_EXPIRED

        return False, None

    def get_eviction_candidates(
        self, entries: Dict[str, CacheEntry], count: int
    ) -> List[str]:
        """期限切れエントリを削除候補として選択"""
        expired_entries = [
            (key, entry.expires_at)
            for key, entry in entries.items()
            if entry.is_expired
        ]

        # 有効期限順でソート
        expired_entries.sort(key=lambda x: x[1])

        return [key for key, _ in expired_entries[:count]]

    def on_access(self, entry: CacheEntry):
        """アクセス時処理（TTLでは特に処理なし）"""
        pass

    def get_strategy_stats(self) -> Dict[str, Any]:
        return {"ttl_strategy": self._stats}


class LRUStrategy(InvalidationStrategy):
    """LRU（Least Recently Used）無効化戦略"""

    def __init__(self, max_idle_seconds: int = 3600):
        self.max_idle_seconds = max_idle_seconds
        self._access_order = OrderedDict()  # アクセス順序管理
        self._lock = threading.RLock()
        self._stats = {"lru_evictions": 0, "access_updates": 0}

    def should_evict(self, entry: CacheEntry) -> Tuple[bool, InvalidationTrigger]:
        """アイドル時間による無効化判定"""
        if entry.idle_seconds > self.max_idle_seconds:
            self._stats["lru_evictions"] += 1
            return True, InvalidationTrigger.LRU_EVICTION

        return False, None

    def get_eviction_candidates(
        self, entries: Dict[str, CacheEntry], count: int
    ) -> List[str]:
        """LRU順で削除候補選択"""
        with self._lock:
            # アクセス時間順でソート（古い順）
            lru_entries = sorted(entries.items(), key=lambda x: x[1].last_accessed)

            return [key for key, _ in lru_entries[:count]]

    def on_access(self, entry: CacheEntry):
        """アクセス順序更新"""
        with self._lock:
            # アクセス順序を最新に更新
            if entry.key in self._access_order:
                del self._access_order[entry.key]
            self._access_order[entry.key] = time.time()
            self._stats["access_updates"] += 1

    def get_strategy_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "lru_strategy": {
                    **self._stats,
                    "tracked_keys": len(self._access_order),
                    "max_idle_seconds": self.max_idle_seconds,
                }
            }


class LFUStrategy(InvalidationStrategy):
    """LFU（Least Frequently Used）無効化戦略"""

    def __init__(self, min_frequency_threshold: float = 0.01):
        self.min_frequency_threshold = min_frequency_threshold
        self._frequency_tracker = {}
        self._lock = threading.RLock()
        self._stats = {"lfu_evictions": 0, "frequency_updates": 0}

    def should_evict(self, entry: CacheEntry) -> Tuple[bool, InvalidationTrigger]:
        """アクセス頻度による無効化判定"""
        if entry.access_frequency < self.min_frequency_threshold:
            self._stats["lfu_evictions"] += 1
            return True, InvalidationTrigger.LFU_EVICTION

        return False, None

    def get_eviction_candidates(
        self, entries: Dict[str, CacheEntry], count: int
    ) -> List[str]:
        """LFU順で削除候補選択"""
        with self._lock:
            # アクセス頻度順でソート（低い順）
            lfu_entries = sorted(entries.items(), key=lambda x: x[1].access_frequency)

            return [key for key, _ in lfu_entries[:count]]

    def on_access(self, entry: CacheEntry):
        """アクセス頻度更新"""
        with self._lock:
            entry.update_frequency()
            self._frequency_tracker[entry.key] = entry.access_frequency
            self._stats["frequency_updates"] += 1

    def get_strategy_stats(self) -> Dict[str, Any]:
        with self._lock:
            avg_frequency = sum(self._frequency_tracker.values()) / max(
                len(self._frequency_tracker), 1
            )

            return {
                "lfu_strategy": {
                    **self._stats,
                    "tracked_keys": len(self._frequency_tracker),
                    "avg_frequency": avg_frequency,
                    "min_frequency_threshold": self.min_frequency_threshold,
                }
            }


class DependencyBasedStrategy(InvalidationStrategy):
    """依存性ベース無効化戦略"""

    def __init__(self):
        self._dependency_graph = defaultdict(set)  # key -> 依存するキーのセット
        self._dependents_graph = defaultdict(
            set
        )  # key -> このキーに依存するキーのセット
        self._lock = threading.RLock()
        self._stats = {
            "dependency_evictions": 0,
            "dependency_updates": 0,
            "cascade_evictions": 0,
        }

    def add_dependency(self, key: str, depends_on: str):
        """依存関係追加"""
        with self._lock:
            self._dependency_graph[key].add(depends_on)
            self._dependents_graph[depends_on].add(key)
            self._stats["dependency_updates"] += 1

            logger.debug(f"依存関係追加: {key} -> {depends_on}")

    def remove_dependency(self, key: str, depends_on: str = None):
        """依存関係削除"""
        with self._lock:
            if depends_on:
                # 特定の依存関係削除
                self._dependency_graph[key].discard(depends_on)
                self._dependents_graph[depends_on].discard(key)
            else:
                # 全依存関係削除
                for dep in self._dependency_graph[key]:
                    self._dependents_graph[dep].discard(key)
                del self._dependency_graph[key]

                # このキーに依存するキーの依存関係も削除
                for dependent in self._dependents_graph[key]:
                    self._dependency_graph[dependent].discard(key)
                del self._dependents_graph[key]

    def should_evict(self, entry: CacheEntry) -> Tuple[bool, InvalidationTrigger]:
        """依存関係による無効化判定"""
        # 依存関係の無効化は通常、カスケード削除で処理される
        return False, None

    def get_eviction_candidates(
        self, entries: Dict[str, CacheEntry], count: int
    ) -> List[str]:
        """依存性に基づく削除候補選択"""
        # 依存関係が多いキーを優先的に削除候補とする
        with self._lock:
            dependency_scores = [
                (key, len(self._dependency_graph.get(key, set())))
                for key in entries.keys()
            ]

            dependency_scores.sort(key=lambda x: x[1], reverse=True)
            return [key for key, _ in dependency_scores[:count]]

    def on_access(self, entry: CacheEntry):
        """アクセス時処理（依存性戦略では特に処理なし）"""
        pass

    def invalidate_dependents(
        self, key: str, entries: Dict[str, CacheEntry]
    ) -> List[str]:
        """依存先の無効化（カスケード削除）"""
        with self._lock:
            invalidated_keys = []

            def cascade_invalidate(target_key):
                if target_key not in self._dependents_graph:
                    return

                for dependent_key in list(self._dependents_graph[target_key]):
                    if dependent_key in entries:
                        invalidated_keys.append(dependent_key)
                        self._stats["cascade_evictions"] += 1

                        # 再帰的にカスケード削除
                        cascade_invalidate(dependent_key)

            cascade_invalidate(key)

            if invalidated_keys:
                logger.info(f"カスケード無効化: {key} -> {len(invalidated_keys)}件")

            return invalidated_keys

    def get_strategy_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "dependency_strategy": {
                    **self._stats,
                    "total_dependencies": sum(
                        len(deps) for deps in self._dependency_graph.values()
                    ),
                    "keys_with_dependencies": len(self._dependency_graph),
                    "keys_with_dependents": len(self._dependents_graph),
                }
            }


class EventDrivenStrategy(InvalidationStrategy):
    """イベント駆動無効化戦略"""

    def __init__(self):
        self._event_handlers = defaultdict(list)  # event_type -> [handler_functions]
        self._tag_subscriptions = defaultdict(set)  # tag -> set of keys
        self._key_subscriptions = defaultdict(set)  # key -> set of event_types
        self._lock = threading.RLock()
        self._stats = {
            "event_evictions": 0,
            "events_processed": 0,
            "subscriptions_added": 0,
        }

    def subscribe_to_event(self, key: str, event_type: str, handler: Callable = None):
        """イベント購読"""
        with self._lock:
            self._key_subscriptions[key].add(event_type)
            if handler:
                self._event_handlers[event_type].append(handler)
            self._stats["subscriptions_added"] += 1

            logger.debug(f"イベント購読: {key} -> {event_type}")

    def subscribe_tag_to_event(self, tag: str, event_type: str):
        """タグベースイベント購読"""
        with self._lock:
            self._tag_subscriptions[tag].add(event_type)
            self._stats["subscriptions_added"] += 1

    def trigger_event(
        self, event_type: str, entries: Dict[str, CacheEntry], **event_data
    ) -> List[str]:
        """イベント発火"""
        with self._lock:
            self._stats["events_processed"] += 1
            invalidated_keys = []

            # キー固有のイベント処理
            for key, subscribed_events in self._key_subscriptions.items():
                if event_type in subscribed_events and key in entries:
                    invalidated_keys.append(key)
                    self._stats["event_evictions"] += 1

            # タグベースのイベント処理
            for tag, subscribed_events in self._tag_subscriptions.items():
                if event_type in subscribed_events:
                    for key, entry in entries.items():
                        if tag in entry.tags and key not in invalidated_keys:
                            invalidated_keys.append(key)
                            self._stats["event_evictions"] += 1

            # カスタムハンドラー実行
            for handler in self._event_handlers.get(event_type, []):
                try:
                    additional_keys = handler(entries, **event_data)
                    if additional_keys:
                        invalidated_keys.extend(additional_keys)
                except Exception as e:
                    logger.error(f"イベントハンドラーエラー {event_type}: {e}")

            if invalidated_keys:
                logger.info(
                    f"イベント駆動無効化: {event_type} -> {len(invalidated_keys)}件"
                )

            return invalidated_keys

    def should_evict(self, entry: CacheEntry) -> Tuple[bool, InvalidationTrigger]:
        """イベント駆動戦略では直接的な無効化判定は行わない"""
        return False, None

    def get_eviction_candidates(
        self, entries: Dict[str, CacheEntry], count: int
    ) -> List[str]:
        """イベント購読数が多いキーを削除候補とする"""
        with self._lock:
            subscription_scores = [
                (key, len(self._key_subscriptions.get(key, set())))
                for key in entries.keys()
            ]

            subscription_scores.sort(key=lambda x: x[1], reverse=True)
            return [key for key, _ in subscription_scores[:count]]

    def on_access(self, entry: CacheEntry):
        """アクセス時処理（イベント戦略では特に処理なし）"""
        pass

    def get_strategy_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "event_driven_strategy": {
                    **self._stats,
                    "total_subscriptions": sum(
                        len(events) for events in self._key_subscriptions.values()
                    ),
                    "keys_with_subscriptions": len(self._key_subscriptions),
                    "tag_subscriptions": len(self._tag_subscriptions),
                    "registered_handlers": sum(
                        len(handlers) for handlers in self._event_handlers.values()
                    ),
                }
            }


class SmartInvalidationManager:
    """スマート無効化管理システム"""

    def __init__(
        self,
        strategies: List[InvalidationStrategy] = None,
        max_cache_size: int = 10000,
        memory_pressure_threshold: float = 0.8,
    ):
        self.max_cache_size = max_cache_size
        self.memory_pressure_threshold = memory_pressure_threshold

        # デフォルト戦略設定
        if strategies is None:
            strategies = [
                TTLStrategy(),
                LRUStrategy(max_idle_seconds=3600),
                LFUStrategy(min_frequency_threshold=0.01),
            ]

        self.strategies = strategies
        self._cache_entries = {}
        self._lock = threading.RLock()

        # 統計情報
        self._stats = {
            "total_evictions": 0,
            "manual_invalidations": 0,
            "pressure_evictions": 0,
            "strategy_evictions": defaultdict(int),
            "last_cleanup": time.time(),
        }

        logger.info(
            f"SmartInvalidationManager初期化: {len(strategies)}戦略, 最大サイズ={max_cache_size}"
        )

    def add_entry(self, key: str, entry: CacheEntry):
        """エントリ追加"""
        with self._lock:
            self._cache_entries[key] = entry

            # 容量チェック
            if len(self._cache_entries) > self.max_cache_size:
                self._handle_memory_pressure()

    def get_entry(self, key: str) -> Optional[CacheEntry]:
        """エントリ取得"""
        with self._lock:
            entry = self._cache_entries.get(key)
            if entry:
                # アクセス記録
                entry.record_access()

                # 戦略にアクセス通知
                for strategy in self.strategies:
                    strategy.on_access(entry)

            return entry

    def remove_entry(
        self,
        key: str,
        trigger: InvalidationTrigger = InvalidationTrigger.MANUAL_INVALIDATION,
    ) -> bool:
        """エントリ削除"""
        with self._lock:
            if key in self._cache_entries:
                del self._cache_entries[key]
                self._stats["total_evictions"] += 1
                self._stats["strategy_evictions"][trigger] += 1

                if trigger == InvalidationTrigger.MANUAL_INVALIDATION:
                    self._stats["manual_invalidations"] += 1

                logger.debug(f"エントリ削除: {key}, 理由: {trigger.name}")
                return True

            return False

    def cleanup_expired(self) -> int:
        """期限切れエントリクリーンアップ"""
        with self._lock:
            removed_count = 0
            keys_to_remove = []

            for key, entry in self._cache_entries.items():
                for strategy in self.strategies:
                    should_evict, trigger = strategy.should_evict(entry)
                    if should_evict:
                        keys_to_remove.append((key, trigger))
                        break

            # 削除実行
            for key, trigger in keys_to_remove:
                if self.remove_entry(key, trigger):
                    removed_count += 1

            # 依存性・イベント駆動のカスケード削除処理
            self._handle_cascade_invalidation(keys_to_remove)

            self._stats["last_cleanup"] = time.time()

            if removed_count > 0:
                logger.info(f"期限切れクリーンアップ完了: {removed_count}件削除")

            return removed_count

    def invalidate_by_tag(self, tag: str) -> int:
        """タグベース無効化"""
        with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache_entries.items() if tag in entry.tags
            ]

            removed_count = 0
            for key in keys_to_remove:
                if self.remove_entry(key, InvalidationTrigger.MANUAL_INVALIDATION):
                    removed_count += 1

            if removed_count > 0:
                logger.info(f"タグベース無効化完了: {tag} -> {removed_count}件削除")

            return removed_count

    def invalidate_by_dependency(self, dependency_key: str) -> int:
        """依存性ベース無効化"""
        with self._lock:
            removed_count = 0

            # 依存性戦略を検索
            for strategy in self.strategies:
                if isinstance(strategy, DependencyBasedStrategy):
                    invalidated_keys = strategy.invalidate_dependents(
                        dependency_key, self._cache_entries
                    )

                    for key in invalidated_keys:
                        if self.remove_entry(
                            key, InvalidationTrigger.DEPENDENCY_CHANGED
                        ):
                            removed_count += 1

                    break

            return removed_count

    def trigger_event(self, event_type: str, **event_data) -> int:
        """イベント駆動無効化"""
        with self._lock:
            removed_count = 0

            # イベント駆動戦略を検索
            for strategy in self.strategies:
                if isinstance(strategy, EventDrivenStrategy):
                    invalidated_keys = strategy.trigger_event(
                        event_type, self._cache_entries, **event_data
                    )

                    for key in invalidated_keys:
                        if self.remove_entry(key, InvalidationTrigger.EVENT_TRIGGERED):
                            removed_count += 1

                    break

            return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        with self._lock:
            stats = {
                "cache_size": len(self._cache_entries),
                "max_cache_size": self.max_cache_size,
                "memory_usage_ratio": len(self._cache_entries) / self.max_cache_size,
                "last_cleanup_age": time.time() - self._stats["last_cleanup"],
                **self._stats,
            }

            # 戦略固有統計情報
            for strategy in self.strategies:
                strategy_stats = strategy.get_strategy_stats()
                stats.update(strategy_stats)

            return stats

    def _handle_memory_pressure(self):
        """メモリ圧迫時の処理"""
        current_ratio = len(self._cache_entries) / self.max_cache_size

        if current_ratio >= self.memory_pressure_threshold:
            # 削除すべきエントリ数計算
            target_size = int(self.max_cache_size * 0.7)  # 70%まで削減
            evict_count = len(self._cache_entries) - target_size

            logger.warning(f"メモリ圧迫: {current_ratio:.1%}, {evict_count}件削除予定")

            # 各戦略から削除候補を取得
            all_candidates = []
            candidates_per_strategy = max(1, evict_count // len(self.strategies))

            for strategy in self.strategies:
                candidates = strategy.get_eviction_candidates(
                    self._cache_entries, candidates_per_strategy
                )
                all_candidates.extend(candidates)

            # 重複削除・優先度順ソート
            unique_candidates = list(dict.fromkeys(all_candidates))[:evict_count]

            # 削除実行
            removed_count = 0
            for key in unique_candidates:
                if self.remove_entry(key, InvalidationTrigger.MEMORY_PRESSURE):
                    removed_count += 1
                    self._stats["pressure_evictions"] += 1

            logger.info(f"メモリ圧迫対応完了: {removed_count}件削除")

    def _handle_cascade_invalidation(
        self, initial_removals: List[Tuple[str, InvalidationTrigger]]
    ):
        """カスケード無効化処理"""
        # 依存性ベースのカスケード削除
        for strategy in self.strategies:
            if isinstance(strategy, DependencyBasedStrategy):
                for key, trigger in initial_removals:
                    cascade_keys = strategy.invalidate_dependents(
                        key, self._cache_entries
                    )
                    for cascade_key in cascade_keys:
                        self.remove_entry(
                            cascade_key, InvalidationTrigger.DEPENDENCY_CHANGED
                        )


# グローバルインスタンス
_global_invalidation_manager: Optional[SmartInvalidationManager] = None
_manager_lock = threading.Lock()


def get_invalidation_manager(
    strategies: List[InvalidationStrategy] = None, max_cache_size: int = 10000
) -> SmartInvalidationManager:
    """グローバル無効化マネージャー取得"""
    global _global_invalidation_manager

    if _global_invalidation_manager is None:
        with _manager_lock:
            if _global_invalidation_manager is None:
                _global_invalidation_manager = SmartInvalidationManager(
                    strategies=strategies, max_cache_size=max_cache_size
                )

    return _global_invalidation_manager


if __name__ == "__main__":
    # テスト実行
    print("=== Issue #377 スマートキャッシュ無効化戦略テスト ===")

    # 戦略設定
    strategies = [
        TTLStrategy(),
        LRUStrategy(max_idle_seconds=10),
        LFUStrategy(min_frequency_threshold=0.1),
        DependencyBasedStrategy(),
        EventDrivenStrategy(),
    ]

    manager = SmartInvalidationManager(strategies=strategies, max_cache_size=100)

    print("\n1. エントリ追加・アクセステスト")

    # テストエントリ作成
    entry1 = CacheEntry(
        key="test1",
        value={"data": "value1"},
        created_at=time.time(),
        expires_at=time.time() + 60,
        last_accessed=time.time(),
        tags={"user_data", "important"},
    )

    entry2 = CacheEntry(
        key="test2",
        value={"data": "value2"},
        created_at=time.time() - 20,
        expires_at=time.time() + 5,  # 5秒後に期限切れ
        last_accessed=time.time() - 20,
        tags={"temp_data"},
    )

    manager.add_entry("test1", entry1)
    manager.add_entry("test2", entry2)

    print("エントリ追加完了")

    # アクセステスト
    accessed_entry = manager.get_entry("test1")
    print(f"アクセステスト: {accessed_entry.key if accessed_entry else 'None'}")

    print("\n2. 無効化テスト")

    # TTL期限切れをシミュレート
    time.sleep(6)  # entry2が期限切れになる

    removed_count = manager.cleanup_expired()
    print(f"期限切れクリーンアップ: {removed_count}件削除")

    # タグベース無効化
    tag_removed = manager.invalidate_by_tag("temp_data")
    print(f"タグベース無効化: {tag_removed}件削除")

    print("\n3. 統計情報")
    stats = manager.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

    print("\n=== スマートキャッシュ無効化戦略テスト完了 ===")
