#!/usr/bin/env python3
"""
Smart Cache Invalidation System
Issue #377: 高度なキャッシング戦略の導入

スマートキャッシュ無効化戦略:
- データ依存性グラフの自動構築
- 関連データの連鎖無効化
- 段階的更新とバッチ処理
- パフォーマンス影響の最小化
"""

import asyncio
import hashlib
import json
import threading
import time
import warnings
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

try:
    from ..utils.logging_config import get_context_logger
    from ..utils.unified_cache_manager import UnifiedCacheManager
    from .enhanced_persistent_cache import EnhancedPersistentCache
    from .redis_enhanced_cache import RedisEnhancedCache
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # Fallback classes
    class EnhancedPersistentCache:
        pass

    class RedisEnhancedCache:
        pass

    class UnifiedCacheManager:
        pass


logger = get_context_logger(__name__)


class InvalidationType(Enum):
    """無効化タイプ"""

    IMMEDIATE = "immediate"  # 即座に無効化
    LAZY = "lazy"  # 次回アクセス時に無効化
    BATCH = "batch"  # バッチで無効化
    TIME_BASED = "time_based"  # 時間ベース無効化
    EVENT_DRIVEN = "event_driven"  # イベント駆動無効化


class DependencyType(Enum):
    """依存性タイプ"""

    DIRECT = "direct"  # 直接依存
    DERIVED = "derived"  # 派生依存
    AGGREGATED = "aggregated"  # 集約依存
    TEMPORAL = "temporal"  # 時間依存
    HIERARCHICAL = "hierarchical"  # 階層依存


@dataclass
class DependencyRule:
    """依存性ルール"""

    source_pattern: str
    target_patterns: List[str]
    dependency_type: DependencyType
    invalidation_type: InvalidationType
    priority: float = 1.0
    delay_seconds: float = 0.0
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InvalidationEvent:
    """無効化イベント"""

    event_id: str
    source_key: str
    affected_keys: List[str]
    event_type: InvalidationType
    timestamp: float
    rule_id: Optional[str] = None
    batch_id: Optional[str] = None
    processed: bool = False
    error_message: Optional[str] = None


@dataclass
class InvalidationStats:
    """無効化統計"""

    total_events: int = 0
    immediate_invalidations: int = 0
    lazy_invalidations: int = 0
    batch_invalidations: int = 0
    cascade_invalidations: int = 0
    failed_invalidations: int = 0
    average_processing_time_ms: float = 0.0
    dependency_graph_size: int = 0


class DependencyGraph:
    """データ依存性グラフ"""

    def __init__(self):
        # key -> Set[key] (依存するキーのセット)
        self.dependencies = defaultdict(set)
        # key -> Set[key] (依存されているキーのセット)
        self.dependents = defaultdict(set)
        self.lock = threading.RLock()
        self.access_history = deque(maxlen=10000)  # アクセス履歴

    def add_dependency(self, source_key: str, target_key: str):
        """依存関係追加"""
        with self.lock:
            self.dependencies[target_key].add(source_key)
            self.dependents[source_key].add(target_key)

            logger.debug(f"依存関係追加: {source_key} -> {target_key}")

    def remove_dependency(self, source_key: str, target_key: str):
        """依存関係削除"""
        with self.lock:
            self.dependencies[target_key].discard(source_key)
            self.dependents[source_key].discard(target_key)

            # 空のセットをクリーンアップ
            if not self.dependencies[target_key]:
                del self.dependencies[target_key]
            if not self.dependents[source_key]:
                del self.dependents[source_key]

    def get_dependent_keys(self, source_key: str, max_depth: int = 3) -> Set[str]:
        """依存しているキー取得（カスケード）"""
        with self.lock:
            visited = set()
            to_process = deque([(source_key, 0)])
            result = set()

            while to_process:
                current_key, depth = to_process.popleft()

                if current_key in visited or depth > max_depth:
                    continue

                visited.add(current_key)

                # 直接依存するキーを追加
                dependents = self.dependents.get(current_key, set())
                for dependent in dependents:
                    result.add(dependent)
                    to_process.append((dependent, depth + 1))

            return result

    def get_dependency_keys(self, target_key: str) -> Set[str]:
        """依存元キー取得"""
        with self.lock:
            return self.dependencies.get(target_key, set()).copy()

    def remove_key(self, key: str):
        """キーとその依存関係を完全削除"""
        with self.lock:
            # 依存している全てのキーから削除
            for source_key in self.dependencies.get(key, set()).copy():
                self.remove_dependency(source_key, key)

            # 依存されている全てのキーから削除
            for target_key in self.dependents.get(key, set()).copy():
                self.remove_dependency(key, target_key)

    def record_access(self, key: str, operation: str = "get"):
        """アクセス記録（パターン学習用）"""
        self.access_history.append({"key": key, "operation": operation, "timestamp": time.time()})

    def analyze_access_patterns(self) -> Dict[str, List[str]]:
        """アクセスパターン分析（自動依存関係推定）"""
        patterns = defaultdict(list)

        # 時間窓での共起分析
        time_window = 60.0  # 1分以内のアクセスを関連とみなす

        for i in range(len(self.access_history) - 1):
            current = self.access_history[i]
            next_access = self.access_history[i + 1]

            time_diff = next_access["timestamp"] - current["timestamp"]

            if time_diff <= time_window and current["key"] != next_access["key"]:
                patterns[current["key"]].append(next_access["key"])

        return dict(patterns)

    def get_stats(self) -> Dict[str, Any]:
        """依存グラフ統計"""
        with self.lock:
            return {
                "total_dependencies": sum(len(deps) for deps in self.dependencies.values()),
                "total_dependents": sum(len(deps) for deps in self.dependents.values()),
                "keys_with_dependencies": len(self.dependencies),
                "keys_with_dependents": len(self.dependents),
                "access_history_size": len(self.access_history),
                "max_fanout": (
                    max(len(deps) for deps in self.dependents.values()) if self.dependents else 0
                ),
                "max_fanin": (
                    max(len(deps) for deps in self.dependencies.values())
                    if self.dependencies
                    else 0
                ),
            }


class SmartCacheInvalidator:
    """スマートキャッシュ無効化システム"""

    def __init__(
        self,
        cache_instances: List[Any] = None,
        enable_auto_dependency_detection: bool = True,
        batch_processing_interval: float = 5.0,
        max_cascade_depth: int = 3,
    ):
        self.cache_instances = cache_instances or []
        self.dependency_graph = DependencyGraph()
        self.rules = {}  # rule_id -> DependencyRule
        self.pending_events = deque()
        self.processing_events = {}  # batch_id -> List[InvalidationEvent]

        self.enable_auto_dependency_detection = enable_auto_dependency_detection
        self.batch_processing_interval = batch_processing_interval
        self.max_cascade_depth = max_cascade_depth

        self.stats = InvalidationStats()
        self.lock = threading.RLock()

        # バックグラウンドタスク
        self.batch_processor_task = None
        self.pattern_analyzer_task = None
        self.running = False

        # イベントハンドラー
        self.event_handlers = {}  # event_type -> List[Callable]

        logger.info("スマートキャッシュ無効化システム初期化完了")

    def register_cache(self, cache_instance: Any):
        """キャッシュインスタンス登録"""
        self.cache_instances.append(cache_instance)
        logger.info(f"キャッシュインスタンス登録: {type(cache_instance).__name__}")

    def add_rule(self, rule: DependencyRule) -> str:
        """依存性ルール追加"""
        rule_id = hashlib.md5(f"{rule.source_pattern}:{rule.target_patterns}".encode()).hexdigest()[
            :8
        ]

        with self.lock:
            self.rules[rule_id] = rule
            logger.info(
                f"依存性ルール追加: {rule_id} ({rule.source_pattern} -> {rule.target_patterns})"
            )

        return rule_id

    def remove_rule(self, rule_id: str) -> bool:
        """依存性ルール削除"""
        with self.lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                logger.info(f"依存性ルール削除: {rule_id}")
                return True
        return False

    def invalidate(
        self,
        key: str,
        invalidation_type: InvalidationType = InvalidationType.IMMEDIATE,
        batch_id: Optional[str] = None,
        cascade: bool = True,
    ) -> str:
        """キー無効化実行"""
        event_id = f"{int(time.time() * 1000)}_{hashlib.md5(key.encode()).hexdigest()[:8]}"

        with self.lock:
            # 影響を受けるキー特定
            affected_keys = [key]

            if cascade:
                dependent_keys = self.dependency_graph.get_dependent_keys(
                    key, self.max_cascade_depth
                )
                affected_keys.extend(list(dependent_keys))

            # マッチするルール適用
            for rule_id, rule in self.rules.items():
                if self._matches_pattern(key, rule.source_pattern):
                    # ルールに基づいて追加の影響キーを特定
                    for pattern in rule.target_patterns:
                        additional_keys = self._find_keys_by_pattern(pattern)
                        affected_keys.extend(additional_keys)

                    # ルールの無効化タイプを優先
                    if invalidation_type == InvalidationType.IMMEDIATE:
                        invalidation_type = rule.invalidation_type

            # 重複排除
            affected_keys = list(set(affected_keys))

            # 無効化イベント作成
            event = InvalidationEvent(
                event_id=event_id,
                source_key=key,
                affected_keys=affected_keys,
                event_type=invalidation_type,
                timestamp=time.time(),
                batch_id=batch_id,
            )

            # 統計更新
            self.stats.total_events += 1
            if cascade and len(affected_keys) > 1:
                self.stats.cascade_invalidations += 1

            # イベント処理
            if invalidation_type == InvalidationType.IMMEDIATE:
                self._process_immediate_invalidation(event)
            elif invalidation_type == InvalidationType.BATCH:
                self.pending_events.append(event)
            elif invalidation_type == InvalidationType.LAZY:
                self._mark_for_lazy_invalidation(event)

            logger.debug(f"無効化実行: {key} -> {len(affected_keys)}件影響")

        return event_id

    def _process_immediate_invalidation(self, event: InvalidationEvent):
        """即座無効化処理"""
        start_time = time.time()

        try:
            for cache in self.cache_instances:
                for key in event.affected_keys:
                    # キャッシュタイプに応じた無効化
                    if hasattr(cache, "delete"):
                        cache.delete(key)
                    elif hasattr(cache, "invalidate"):
                        cache.invalidate(key)

                    # 依存グラフからアクセス記録
                    self.dependency_graph.record_access(key, "invalidate")

            self.stats.immediate_invalidations += 1
            event.processed = True

            logger.debug(f"即座無効化完了: {event.event_id}")

        except Exception as e:
            self.stats.failed_invalidations += 1
            event.error_message = str(e)
            logger.error(f"即座無効化エラー ({event.event_id}): {e}")

        finally:
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_time(processing_time)

    def _mark_for_lazy_invalidation(self, event: InvalidationEvent):
        """遅延無効化マーク"""
        # メタデータに遅延無効化フラグを設定
        for cache in self.cache_instances:
            if hasattr(cache, "set_metadata"):
                for key in event.affected_keys:
                    cache.set_metadata(key, {"lazy_invalid": True, "invalid_since": time.time()})

        self.stats.lazy_invalidations += 1
        event.processed = True
        logger.debug(f"遅延無効化マーク完了: {event.event_id}")

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """パターンマッチング"""
        # 簡単なワイルドカードパターンマッチング
        if "*" in pattern:
            import fnmatch

            return fnmatch.fnmatch(key, pattern)
        return key == pattern

    def _find_keys_by_pattern(self, pattern: str) -> List[str]:
        """パターンに一致するキー検索"""
        # 実際の実装では各キャッシュインスタンスからキー一覧を取得
        # ここでは例として空のリストを返す
        return []

    def _update_processing_time(self, processing_time_ms: float):
        """処理時間統計更新"""
        current_avg = self.stats.average_processing_time_ms
        total_events = self.stats.total_events

        if total_events > 1:
            self.stats.average_processing_time_ms = (
                current_avg * (total_events - 1) + processing_time_ms
            ) / total_events
        else:
            self.stats.average_processing_time_ms = processing_time_ms

    async def start_background_processing(self):
        """バックグラウンド処理開始"""
        if self.running:
            return

        self.running = True

        # バッチ処理タスク
        self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())

        # パターン分析タスク
        if self.enable_auto_dependency_detection:
            self.pattern_analyzer_task = asyncio.create_task(self._pattern_analyzer_loop())

        logger.info("バックグラウンド処理開始")

    async def stop_background_processing(self):
        """バックグラウンド処理停止"""
        self.running = False

        if self.batch_processor_task:
            self.batch_processor_task.cancel()
        if self.pattern_analyzer_task:
            self.pattern_analyzer_task.cancel()

        logger.info("バックグラウンド処理停止")

    async def _batch_processor_loop(self):
        """バッチ処理ループ"""
        while self.running:
            try:
                await asyncio.sleep(self.batch_processing_interval)
                await self._process_batch_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"バッチ処理ループエラー: {e}")

    async def _process_batch_events(self):
        """バッチイベント処理"""
        if not self.pending_events:
            return

        with self.lock:
            batch_events = []
            # 最大100件まで一度に処理
            for _ in range(min(100, len(self.pending_events))):
                if self.pending_events:
                    batch_events.append(self.pending_events.popleft())

        if not batch_events:
            return

        batch_id = f"batch_{int(time.time() * 1000)}"
        self.processing_events[batch_id] = batch_events

        start_time = time.time()

        try:
            # バッチで無効化実行
            all_keys_to_invalidate = set()
            for event in batch_events:
                all_keys_to_invalidate.update(event.affected_keys)

            # 各キャッシュインスタンスで一括無効化
            for cache in self.cache_instances:
                if hasattr(cache, "batch_delete"):
                    # バッチ削除対応
                    cache.batch_delete(list(all_keys_to_invalidate))
                else:
                    # 個別削除
                    for key in all_keys_to_invalidate:
                        if hasattr(cache, "delete"):
                            cache.delete(key)

            # 統計更新
            self.stats.batch_invalidations += len(batch_events)

            # イベント完了マーク
            for event in batch_events:
                event.processed = True

            logger.info(
                f"バッチ無効化完了: {batch_id} ({len(batch_events)}イベント, {len(all_keys_to_invalidate)}キー)"
            )

        except Exception as e:
            self.stats.failed_invalidations += len(batch_events)
            for event in batch_events:
                event.error_message = str(e)
            logger.error(f"バッチ無効化エラー ({batch_id}): {e}")

        finally:
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_time(processing_time)

            # 処理済みイベントをクリア
            if batch_id in self.processing_events:
                del self.processing_events[batch_id]

    async def _pattern_analyzer_loop(self):
        """パターン分析ループ"""
        while self.running:
            try:
                await asyncio.sleep(300)  # 5分間隔
                self._analyze_and_create_rules()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"パターン分析ループエラー: {e}")

    def _analyze_and_create_rules(self):
        """パターン分析と自動ルール作成"""
        try:
            patterns = self.dependency_graph.analyze_access_patterns()

            for source_key, related_keys in patterns.items():
                if len(related_keys) >= 3:  # 3回以上の共起で関連性ありとみなす
                    # 自動ルール作成
                    rule = DependencyRule(
                        source_pattern=source_key,
                        target_patterns=list(set(related_keys)),
                        dependency_type=DependencyType.DERIVED,
                        invalidation_type=InvalidationType.BATCH,
                        priority=0.5,  # 自動生成ルールは低優先度
                        metadata={
                            "auto_generated": True,
                            "confidence": len(related_keys) / 10,
                        },
                    )

                    rule_id = self.add_rule(rule)
                    logger.info(
                        f"自動ルール生成: {rule_id} ({source_key} -> {len(related_keys)}関連)"
                    )

        except Exception as e:
            logger.error(f"パターン分析エラー: {e}")

    def add_event_handler(self, event_type: str, handler: Callable):
        """イベントハンドラー追加"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """包括的統計情報"""
        dependency_stats = self.dependency_graph.get_stats()

        return {
            "invalidation_stats": {
                "total_events": self.stats.total_events,
                "immediate_invalidations": self.stats.immediate_invalidations,
                "lazy_invalidations": self.stats.lazy_invalidations,
                "batch_invalidations": self.stats.batch_invalidations,
                "cascade_invalidations": self.stats.cascade_invalidations,
                "failed_invalidations": self.stats.failed_invalidations,
                "success_rate": (self.stats.total_events - self.stats.failed_invalidations)
                / max(self.stats.total_events, 1)
                * 100,
                "average_processing_time_ms": self.stats.average_processing_time_ms,
            },
            "dependency_graph": dependency_stats,
            "rules": {
                "total_rules": len(self.rules),
                "auto_generated_rules": sum(
                    1 for rule in self.rules.values() if rule.metadata.get("auto_generated")
                ),
                "manual_rules": sum(
                    1 for rule in self.rules.values() if not rule.metadata.get("auto_generated")
                ),
            },
            "processing": {
                "pending_events": len(self.pending_events),
                "active_batches": len(self.processing_events),
                "background_running": self.running,
            },
            "cache_instances": len(self.cache_instances),
        }


# テスト関数
async def test_smart_cache_invalidation():
    """スマートキャッシュ無効化テスト"""
    print("=== Issue #377 スマートキャッシュ無効化システムテスト ===")

    try:
        # テスト用キャッシュ（モック）
        class MockCache:
            def __init__(self):
                self.data = {}
                self.deleted_keys = []

            def get(self, key):
                return self.data.get(key)

            def put(self, key, value):
                self.data[key] = value

            def delete(self, key):
                if key in self.data:
                    del self.data[key]
                    self.deleted_keys.append(key)
                    return True
                return False

            def batch_delete(self, keys):
                deleted = 0
                for key in keys:
                    if self.delete(key):
                        deleted += 1
                return deleted

        # キャッシュインスタンス作成
        cache1 = MockCache()
        cache2 = MockCache()

        # テストデータ
        cache1.put("user:123", {"name": "TestUser"})
        cache1.put("user:123:profile", {"bio": "Test bio"})
        cache1.put("user:123:settings", {"theme": "dark"})
        cache2.put("user_list", ["123", "456"])

        # 無効化システム初期化
        invalidator = SmartCacheInvalidator([cache1, cache2], batch_processing_interval=2.0)

        print("\n1. スマートキャッシュ無効化システム初期化完了")

        # 依存性ルール追加
        rule1 = DependencyRule(
            source_pattern="user:*",
            target_patterns=["user:*:profile", "user:*:settings", "user_list"],
            dependency_type=DependencyType.HIERARCHICAL,
            invalidation_type=InvalidationType.IMMEDIATE,
        )

        rule_id = invalidator.add_rule(rule1)
        print(f"\n2. 依存性ルール追加: {rule_id}")

        # バックグラウンド処理開始
        await invalidator.start_background_processing()
        print("   バックグラウンド処理開始")

        # 即座無効化テスト
        print("\n3. 即座無効化テスト...")
        event_id = invalidator.invalidate("user:123", InvalidationType.IMMEDIATE)
        print(f"   無効化実行: {event_id}")
        print(f"   cache1削除キー: {cache1.deleted_keys}")
        print(f"   cache2削除キー: {cache2.deleted_keys}")

        # バッチ無効化テスト
        print("\n4. バッチ無効化テスト...")
        cache1.put("batch_test:1", "data1")
        cache1.put("batch_test:2", "data2")
        cache1.put("batch_test:3", "data3")

        invalidator.invalidate("batch_test:1", InvalidationType.BATCH)
        invalidator.invalidate("batch_test:2", InvalidationType.BATCH)
        invalidator.invalidate("batch_test:3", InvalidationType.BATCH)
        print("   バッチイベント追加完了")

        # バッチ処理を待機
        await asyncio.sleep(3)

        # 統計情報表示
        print("\n5. 統計情報:")
        stats = invalidator.get_comprehensive_stats()
        for category, data in stats.items():
            if isinstance(data, dict):
                print(f"   {category}:")
                for k, v in data.items():
                    print(f"     {k}: {v}")
            else:
                print(f"   {category}: {data}")

        # バックグラウンド処理停止
        await invalidator.stop_background_processing()
        print("\n[OK] スマートキャッシュ無効化システム テスト完了！")

    except Exception as e:
        print(f"[ERROR] テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_smart_cache_invalidation())
