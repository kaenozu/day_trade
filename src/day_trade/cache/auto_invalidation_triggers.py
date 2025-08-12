#!/usr/bin/env python3
"""
Automatic Cache Invalidation Triggers System
Issue #377: 高度なキャッシング戦略の導入

自動キャッシュ無効化トリガーシステム:
- データベース変更監視
- ファイルシステム変更監視
- 時間ベース自動無効化
- API/外部システム変更通知
- メモリ使用量ベーストリガー
"""

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

try:
    from ..utils.logging_config import get_context_logger
    from .smart_cache_invalidation import InvalidationType, SmartCacheInvalidator
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class SmartCacheInvalidator:
        pass

    class InvalidationType(Enum):
        IMMEDIATE = "immediate"
        BATCH = "batch"


logger = get_context_logger(__name__)


class TriggerType(Enum):
    """トリガータイプ"""

    TIME_BASED = "time_based"
    FILE_CHANGE = "file_change"
    DB_CHANGE = "db_change"
    MEMORY_PRESSURE = "memory_pressure"
    API_WEBHOOK = "api_webhook"
    MANUAL = "manual"
    DEPENDENCY = "dependency"


class TriggerCondition(Enum):
    """トリガー条件"""

    ON_CHANGE = "on_change"
    ON_CREATE = "on_create"
    ON_DELETE = "on_delete"
    ON_SCHEDULE = "on_schedule"
    ON_THRESHOLD = "on_threshold"
    ON_EVENT = "on_event"


@dataclass
class TriggerConfig:
    """トリガー設定"""

    trigger_id: str
    trigger_type: TriggerType
    trigger_condition: TriggerCondition
    cache_pattern: str  # 無効化対象のキャッシュパターン

    # 設定パラメータ
    schedule_cron: Optional[str] = None  # cron式
    file_paths: List[str] = field(default_factory=list)
    db_tables: List[str] = field(default_factory=list)
    memory_threshold_percent: float = 85.0
    webhook_url: Optional[str] = None

    # 動作設定
    invalidation_type: InvalidationType = InvalidationType.BATCH
    delay_seconds: float = 0.0
    batch_window_seconds: float = 30.0
    enabled: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerEvent:
    """トリガーイベント"""

    event_id: str
    trigger_id: str
    trigger_type: TriggerType
    timestamp: float
    details: Dict[str, Any]
    processed: bool = False
    cache_keys_affected: List[str] = field(default_factory=list)


class TriggerHandler(ABC):
    """トリガーハンドラー抽象基底クラス"""

    def __init__(self, config: TriggerConfig, invalidator: SmartCacheInvalidator):
        self.config = config
        self.invalidator = invalidator
        self.enabled = config.enabled
        self.last_triggered = 0.0
        self.stats = {"triggered": 0, "errors": 0, "cache_keys_invalidated": 0}

    @abstractmethod
    async def start(self):
        """トリガー監視開始"""
        pass

    @abstractmethod
    async def stop(self):
        """トリガー監視停止"""
        pass

    async def trigger(self, event_details: Dict[str, Any] = None):
        """トリガー実行"""
        if not self.enabled:
            return

        try:
            event_id = f"{self.config.trigger_id}_{int(time.time() * 1000)}"

            event = TriggerEvent(
                event_id=event_id,
                trigger_id=self.config.trigger_id,
                trigger_type=self.config.trigger_type,
                timestamp=time.time(),
                details=event_details or {},
            )

            # 遅延実行
            if self.config.delay_seconds > 0:
                await asyncio.sleep(self.config.delay_seconds)

            # キャッシュ無効化実行
            affected_keys = await self._find_affected_cache_keys(event)

            for key in affected_keys:
                self.invalidator.invalidate(key, self.config.invalidation_type, cascade=True)

            event.cache_keys_affected = affected_keys
            event.processed = True

            self.stats["triggered"] += 1
            self.stats["cache_keys_invalidated"] += len(affected_keys)
            self.last_triggered = time.time()

            logger.info(f"トリガー実行完了: {event_id} ({len(affected_keys)}キー無効化)")

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"トリガー実行エラー ({self.config.trigger_id}): {e}")

    async def _find_affected_cache_keys(self, event: TriggerEvent) -> List[str]:
        """影響を受けるキャッシュキー検索"""
        # パターンマッチングでキー検索
        # 実際の実装では各キャッシュインスタンスからキー一覧を取得する必要がある
        affected_keys = []

        # 簡単なパターンマッチング（実装例）
        pattern = self.config.cache_pattern
        if "*" in pattern:
            # ワイルドカードパターンの場合、実際のキーと照合
            # ここでは例として固定キーを返す
            if pattern == "user:*":
                affected_keys = ["user:123", "user:456"]
            elif pattern == "market_data:*":
                affected_keys = ["market_data:AAPL", "market_data:MSFT"]
        else:
            affected_keys = [pattern]

        return affected_keys


class TimeBasedTrigger(TriggerHandler):
    """時間ベーストリガー"""

    def __init__(self, config: TriggerConfig, invalidator: SmartCacheInvalidator):
        super().__init__(config, invalidator)
        self.scheduler_task = None
        self.running = False

    async def start(self):
        """スケジュール開始"""
        if self.running:
            return

        self.running = True

        if self.config.schedule_cron:
            # Cron式スケジュール（簡単な実装）
            self.scheduler_task = asyncio.create_task(self._cron_scheduler())
        else:
            # 固定間隔スケジュール
            interval = self.config.metadata.get("interval_seconds", 3600)
            self.scheduler_task = asyncio.create_task(self._interval_scheduler(interval))

        logger.info(f"時間ベーストリガー開始: {self.config.trigger_id}")

    async def stop(self):
        """スケジュール停止"""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
        logger.info(f"時間ベーストリガー停止: {self.config.trigger_id}")

    async def _interval_scheduler(self, interval_seconds: float):
        """固定間隔スケジューラー"""
        while self.running:
            try:
                await asyncio.sleep(interval_seconds)
                await self.trigger({"scheduled_time": datetime.now().isoformat()})
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"間隔スケジューラーエラー: {e}")

    async def _cron_scheduler(self):
        """Cronスケジューラー（簡易版）"""
        while self.running:
            try:
                # 次回実行時間計算（簡易実装）
                next_run = self._calculate_next_cron_run()
                sleep_seconds = max(0, next_run - time.time())

                await asyncio.sleep(sleep_seconds)
                await self.trigger({"cron_scheduled": self.config.schedule_cron})

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cronスケジューラーエラー: {e}")

    def _calculate_next_cron_run(self) -> float:
        """次回Cron実行時間計算（簡易版）"""
        # 実際の実装ではcroniterなどのライブラリを使用
        # ここでは1時間後を返す簡易実装
        return time.time() + 3600


class FileSystemTrigger(TriggerHandler):
    """ファイルシステム変更トリガー"""

    def __init__(self, config: TriggerConfig, invalidator: SmartCacheInvalidator):
        super().__init__(config, invalidator)
        self.observer = None
        self.event_handlers = []

    async def start(self):
        """ファイル監視開始"""
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdogライブラリが利用できません。ファイル監視をスキップします。")
            return

        if self.observer:
            return

        self.observer = Observer()

        for file_path in self.config.file_paths:
            path = Path(file_path)

            if path.is_dir():
                watch_path = str(path)
                recursive = True
            else:
                watch_path = str(path.parent)
                recursive = False

            handler = FileChangeHandler(self, str(path))
            self.event_handlers.append(handler)
            self.observer.schedule(handler, watch_path, recursive=recursive)

        self.observer.start()
        logger.info(f"ファイルシステムトリガー開始: {self.config.trigger_id}")

    async def stop(self):
        """ファイル監視停止"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        logger.info(f"ファイルシステムトリガー停止: {self.config.trigger_id}")


class FileChangeHandler(FileSystemEventHandler):
    """ファイル変更ハンドラー"""

    def __init__(self, trigger: FileSystemTrigger, watch_path: str):
        self.trigger = trigger
        self.watch_path = watch_path

    def on_modified(self, event):
        if not event.is_directory:
            asyncio.create_task(
                self.trigger.trigger(
                    {
                        "event_type": "modified",
                        "file_path": event.src_path,
                        "timestamp": time.time(),
                    }
                )
            )

    def on_created(self, event):
        if not event.is_directory:
            asyncio.create_task(
                self.trigger.trigger(
                    {
                        "event_type": "created",
                        "file_path": event.src_path,
                        "timestamp": time.time(),
                    }
                )
            )

    def on_deleted(self, event):
        if not event.is_directory:
            asyncio.create_task(
                self.trigger.trigger(
                    {
                        "event_type": "deleted",
                        "file_path": event.src_path,
                        "timestamp": time.time(),
                    }
                )
            )


class DatabaseTrigger(TriggerHandler):
    """データベース変更トリガー"""

    def __init__(self, config: TriggerConfig, invalidator: SmartCacheInvalidator):
        super().__init__(config, invalidator)
        self.polling_task = None
        self.running = False
        self.last_check_time = 0.0

    async def start(self):
        """DB監視開始"""
        if self.running:
            return

        self.running = True
        self.last_check_time = time.time()

        # ポーリングベースの変更監視
        poll_interval = self.config.metadata.get("poll_interval_seconds", 30)
        self.polling_task = asyncio.create_task(self._polling_monitor(poll_interval))

        logger.info(f"データベーストリガー開始: {self.config.trigger_id}")

    async def stop(self):
        """DB監視停止"""
        self.running = False
        if self.polling_task:
            self.polling_task.cancel()
        logger.info(f"データベーストリガー停止: {self.config.trigger_id}")

    async def _polling_monitor(self, interval_seconds: float):
        """ポーリング監視"""
        while self.running:
            try:
                await asyncio.sleep(interval_seconds)

                # DB変更チェック（簡易実装）
                changes = await self._check_database_changes()
                if changes:
                    await self.trigger(
                        {
                            "changes_detected": len(changes),
                            "affected_tables": list(changes.keys()),
                            "timestamp": time.time(),
                        }
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DBポーリング監視エラー: {e}")

    async def _check_database_changes(self) -> Dict[str, Any]:
        """データベース変更チェック"""
        changes = {}

        # SQLiteの例：最終更新時間をチェック
        db_path = self.config.metadata.get("db_path", "data/cache.db")

        try:
            if os.path.exists(db_path):
                mtime = os.path.getmtime(db_path)
                if mtime > self.last_check_time:
                    changes["cache_db"] = {"last_modified": mtime}
                    self.last_check_time = mtime

        except Exception as e:
            logger.error(f"DB変更チェックエラー: {e}")

        return changes


class MemoryPressureTrigger(TriggerHandler):
    """メモリ圧迫トリガー"""

    def __init__(self, config: TriggerConfig, invalidator: SmartCacheInvalidator):
        super().__init__(config, invalidator)
        self.monitoring_task = None
        self.running = False

    async def start(self):
        """メモリ監視開始"""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutilライブラリが利用できません。メモリ監視をスキップします。")
            return

        if self.running:
            return

        self.running = True

        check_interval = self.config.metadata.get("check_interval_seconds", 60)
        self.monitoring_task = asyncio.create_task(self._memory_monitor(check_interval))

        logger.info(f"メモリ圧迫トリガー開始: {self.config.trigger_id}")

    async def stop(self):
        """メモリ監視停止"""
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info(f"メモリ圧迫トリガー停止: {self.config.trigger_id}")

    async def _memory_monitor(self, interval_seconds: float):
        """メモリ監視"""
        while self.running:
            try:
                await asyncio.sleep(interval_seconds)

                memory = psutil.virtual_memory()
                if memory.percent >= self.config.memory_threshold_percent:
                    await self.trigger(
                        {
                            "memory_percent": memory.percent,
                            "memory_available_mb": memory.available / 1024 / 1024,
                            "threshold_percent": self.config.memory_threshold_percent,
                            "timestamp": time.time(),
                        }
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"メモリ監視エラー: {e}")


class AutoInvalidationTriggerManager:
    """自動無効化トリガー管理システム"""

    def __init__(self, invalidator: SmartCacheInvalidator):
        self.invalidator = invalidator
        self.triggers = {}  # trigger_id -> TriggerHandler
        self.configs = {}  # trigger_id -> TriggerConfig
        self.running = False
        self.lock = threading.RLock()
        self.stats = {
            "active_triggers": 0,
            "total_events": 0,
            "total_invalidations": 0,
            "errors": 0,
        }

        logger.info("自動無効化トリガー管理システム初期化完了")

    def add_trigger(self, config: TriggerConfig) -> str:
        """トリガー追加"""
        with self.lock:
            # トリガーハンドラー作成
            handler = self._create_trigger_handler(config)

            self.configs[config.trigger_id] = config
            self.triggers[config.trigger_id] = handler
            self.stats["active_triggers"] += 1

            # 実行中なら即座に開始
            if self.running:
                asyncio.create_task(handler.start())

            logger.info(f"トリガー追加: {config.trigger_id} ({config.trigger_type.value})")

        return config.trigger_id

    def remove_trigger(self, trigger_id: str) -> bool:
        """トリガー削除"""
        with self.lock:
            if trigger_id in self.triggers:
                handler = self.triggers[trigger_id]
                asyncio.create_task(handler.stop())

                del self.triggers[trigger_id]
                del self.configs[trigger_id]
                self.stats["active_triggers"] -= 1

                logger.info(f"トリガー削除: {trigger_id}")
                return True

        return False

    def _create_trigger_handler(self, config: TriggerConfig) -> TriggerHandler:
        """トリガーハンドラー作成"""
        if config.trigger_type == TriggerType.TIME_BASED:
            return TimeBasedTrigger(config, self.invalidator)
        elif config.trigger_type == TriggerType.FILE_CHANGE:
            return FileSystemTrigger(config, self.invalidator)
        elif config.trigger_type == TriggerType.DB_CHANGE:
            return DatabaseTrigger(config, self.invalidator)
        elif config.trigger_type == TriggerType.MEMORY_PRESSURE:
            return MemoryPressureTrigger(config, self.invalidator)
        else:
            raise ValueError(f"未サポートのトリガータイプ: {config.trigger_type}")

    async def start_all_triggers(self):
        """全トリガー開始"""
        if self.running:
            return

        self.running = True

        with self.lock:
            tasks = []
            for handler in self.triggers.values():
                tasks.append(handler.start())

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"全トリガー開始完了: {len(self.triggers)}個")

    async def stop_all_triggers(self):
        """全トリガー停止"""
        self.running = False

        with self.lock:
            tasks = []
            for handler in self.triggers.values():
                tasks.append(handler.stop())

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("全トリガー停止完了")

    def get_trigger_stats(self) -> Dict[str, Any]:
        """トリガー統計情報"""
        with self.lock:
            trigger_details = {}

            for trigger_id, handler in self.triggers.items():
                config = self.configs[trigger_id]
                trigger_details[trigger_id] = {
                    "type": config.trigger_type.value,
                    "enabled": handler.enabled,
                    "last_triggered": handler.last_triggered,
                    "stats": handler.stats,
                }

            return {
                "manager_stats": self.stats,
                "trigger_details": trigger_details,
                "running": self.running,
            }


# テスト関数
async def test_auto_invalidation_triggers():
    """自動無効化トリガーテスト"""
    print("=== Issue #377 自動無効化トリガーシステムテスト ===")

    try:
        # モックインバリデーター
        class MockInvalidator:
            def __init__(self):
                self.invalidated_keys = []

            def invalidate(self, key, invalidation_type, cascade=False):
                self.invalidated_keys.append(key)
                return f"event_{len(self.invalidated_keys)}"

        # システム初期化
        invalidator = MockInvalidator()
        trigger_manager = AutoInvalidationTriggerManager(invalidator)

        print("\n1. 自動無効化トリガーシステム初期化完了")

        # 時間ベーストリガー追加
        time_trigger_config = TriggerConfig(
            trigger_id="time_trigger_1",
            trigger_type=TriggerType.TIME_BASED,
            trigger_condition=TriggerCondition.ON_SCHEDULE,
            cache_pattern="user:*",
            invalidation_type=InvalidationType.BATCH,
            metadata={"interval_seconds": 2},
        )

        trigger_manager.add_trigger(time_trigger_config)
        print("\n2. 時間ベーストリガー追加完了")

        # メモリ圧迫トリガー追加
        if PSUTIL_AVAILABLE:
            memory_trigger_config = TriggerConfig(
                trigger_id="memory_trigger_1",
                trigger_type=TriggerType.MEMORY_PRESSURE,
                trigger_condition=TriggerCondition.ON_THRESHOLD,
                cache_pattern="temp:*",
                memory_threshold_percent=95.0,  # 高い閾値でテスト用
                metadata={"check_interval_seconds": 1},
            )

            trigger_manager.add_trigger(memory_trigger_config)
            print("3. メモリ圧迫トリガー追加完了")

        # トリガー開始
        await trigger_manager.start_all_triggers()
        print("\n4. 全トリガー開始完了")

        # 少し待機して実行を確認
        print("   トリガー実行を待機中...")
        await asyncio.sleep(5)

        # 統計情報表示
        print("\n5. 統計情報:")
        stats = trigger_manager.get_trigger_stats()
        for category, data in stats.items():
            if isinstance(data, dict):
                print(f"   {category}:")
                for k, v in data.items():
                    if isinstance(v, dict):
                        print(f"     {k}:")
                        for kk, vv in v.items():
                            print(f"       {kk}: {vv}")
                    else:
                        print(f"     {k}: {v}")
            else:
                print(f"   {category}: {data}")

        print(f"\n6. 無効化されたキー: {invalidator.invalidated_keys}")

        # トリガー停止
        await trigger_manager.stop_all_triggers()
        print("\n[OK] 自動無効化トリガーシステム テスト完了！")

    except Exception as e:
        print(f"[ERROR] テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_auto_invalidation_triggers())
