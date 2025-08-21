#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Processing System - 非同期処理・キャッシュシステム
Issue #939 対応: 非同期タスクキューとキャッシュによる処理速度向上
"""

import time
import json
import threading
import asyncio
import concurrent.futures
import hashlib
import pickle
from queue import Queue, PriorityQueue, Empty
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import sqlite3

# Redis（オプション）
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    redis = None

# カスタムモジュール
try:
    from performance_monitor import performance_monitor, track_performance
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False
    def track_performance(func):
        return func

try:
    from audit_logger import audit_logger
    HAS_AUDIT_LOGGER = True
except ImportError:
    HAS_AUDIT_LOGGER = False


class TaskPriority(Enum):
    """タスク優先度"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class TaskStatus(Enum):
    """タスク状態"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncTask:
    """非同期タスク"""
    task_id: str
    function: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3

    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING

    result: Any = None
    error: Optional[str] = None

    def __lt__(self, other):
        """優先度での比較（PriorityQueue用）"""
        return self.priority.value < other.priority.value


@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0


class InMemoryCache:
    """インメモリキャッシュシステム"""

    def __init__(self, max_size_mb: int = 100, default_ttl_seconds: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.current_size_bytes = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """キャッシュからデータを取得"""
        with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                return None

            # 有効期限チェック
            if datetime.now() > entry.expires_at:
                self._remove_entry(key)
                return None

            # アクセス統計更新
            entry.access_count += 1
            entry.last_accessed = datetime.now()

            return entry.value

    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """データをキャッシュに保存"""
        with self._lock:
            try:
                # サイズ計算
                serialized = pickle.dumps(value)
                entry_size = len(serialized)

                # サイズ制限チェック
                if entry_size > self.max_size_bytes:
                    return False

                # 既存エントリがある場合は削除
                if key in self.cache:
                    self._remove_entry(key)

                # 容量確保
                while (self.current_size_bytes + entry_size) > self.max_size_bytes:
                    if not self._evict_lru():
                        return False

                # エントリ作成
                ttl = ttl_seconds or self.default_ttl_seconds
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=ttl),
                    size_bytes=entry_size
                )

                self.cache[key] = entry
                self.current_size_bytes += entry_size

                return True

            except Exception as e:
                print(f"キャッシュ保存エラー: {e}")
                return False

    def _remove_entry(self, key: str):
        """エントリを削除"""
        entry = self.cache.get(key)
        if entry:
            self.current_size_bytes -= entry.size_bytes
            del self.cache[key]

    def _evict_lru(self) -> bool:
        """LRU方式で最も使用されていないエントリを削除"""
        if not self.cache:
            return False

        # 最も古いアクセス時刻のエントリを見つける
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
        self._remove_entry(lru_key)
        return True

    def clear_expired(self):
        """期限切れエントリをクリア"""
        with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, entry in self.cache.items()
                if now > entry.expires_at
            ]

            for key in expired_keys:
                self._remove_entry(key)

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        with self._lock:
            total_accesses = sum(entry.access_count for entry in self.cache.values())

            return {
                'entries': len(self.cache),
                'size_mb': self.current_size_bytes / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'usage_percent': (self.current_size_bytes / self.max_size_bytes) * 100,
                'total_accesses': total_accesses,
                'average_access_per_entry': total_accesses / len(self.cache) if self.cache else 0
            }


class AsyncTaskManager:
    """非同期タスク管理システム"""

    def __init__(self, max_workers: int = 4, queue_size: int = 1000):
        self.max_workers = max_workers
        self.queue_size = queue_size

        # タスクキュー
        self.task_queue = PriorityQueue(maxsize=queue_size)
        self.tasks: Dict[str, AsyncTask] = {}

        # ワーカー管理
        self.workers: List[threading.Thread] = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.running = False

        # ロック
        self._lock = threading.Lock()

        # 統計
        self.stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_cancelled': 0
        }

    def start(self):
        """タスクマネージャー開始"""
        if self.running:
            return

        self.running = True

        # ワーカースレッド開始
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"AsyncWorker-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        print(f"AsyncTaskManager started with {self.max_workers} workers")

    def stop(self):
        """タスクマネージャー停止"""
        if not self.running:
            return

        self.running = False

        # ワーカースレッド停止
        for worker in self.workers:
            worker.join(timeout=5)

        self.executor.shutdown(wait=True)
        print("AsyncTaskManager stopped")

    def submit_task(self,
                   function: Callable,
                   args: tuple = (),
                   kwargs: dict = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   timeout_seconds: int = 300,
                   task_id: Optional[str] = None) -> str:
        """タスクを投入"""
        if not self.running:
            self.start()

        # タスクID生成
        if task_id is None:
            task_data = f"{function.__name__}_{args}_{kwargs}_{time.time()}"
            task_id = hashlib.md5(task_data.encode()).hexdigest()

        # タスク作成
        task = AsyncTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            timeout_seconds=timeout_seconds
        )

        try:
            # キューに投入
            self.task_queue.put(task, block=False)

            with self._lock:
                self.tasks[task_id] = task
                self.stats['total_submitted'] += 1

            # ログ記録
            if HAS_AUDIT_LOGGER:
                audit_logger.log_business_event(
                    "async_task_submitted",
                    {
                        "task_id": task_id,
                        "function": function.__name__,
                        "priority": priority.name,
                        "queue_size": self.task_queue.qsize()
                    }
                )

            return task_id

        except Exception as e:
            print(f"タスク投入エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"task_id": task_id, "context": "task_submission"})
            raise

    def _worker_loop(self):
        """ワーカーループ"""
        while self.running:
            try:
                # タスク取得（1秒でタイムアウト）
                task = self.task_queue.get(timeout=1)

                # タスク実行
                self._execute_task(task)

                self.task_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                print(f"ワーカーエラー: {e}")
                if HAS_AUDIT_LOGGER:
                    audit_logger.log_error_with_context(e, {"context": "worker_loop"})

    def _execute_task(self, task: AsyncTask):
        """タスク実行"""
        start_time = time.time()

        with self._lock:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()

        try:
            # タイムアウト付きでタスク実行
            future = self.executor.submit(task.function, *task.args, **task.kwargs)
            result = future.result(timeout=task.timeout_seconds)

            # 成功
            with self._lock:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                self.stats['total_completed'] += 1

            execution_time = (time.time() - start_time) * 1000

            # パフォーマンス記録
            if HAS_PERFORMANCE_MONITOR:
                performance_monitor.track_analysis_time(
                    symbol=f"async_task_{task.function.__name__}",
                    duration=execution_time / 1000,
                    analysis_type="async_task"
                )

            # ログ記録
            if HAS_AUDIT_LOGGER:
                audit_logger.log_business_event(
                    "async_task_completed",
                    {
                        "task_id": task.task_id,
                        "function": task.function.__name__,
                        "execution_time_ms": execution_time
                    }
                )

        except concurrent.futures.TimeoutError:
            # タイムアウト
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = f"Timeout after {task.timeout_seconds} seconds"
                task.completed_at = datetime.now()
                self.stats['total_failed'] += 1

            print(f"タスクタイムアウト: {task.task_id}")

        except Exception as e:
            # エラー
            with self._lock:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()

            # リトライ処理
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.error = None

                try:
                    self.task_queue.put(task, block=False)
                    print(f"タスクリトライ {task.retry_count}/{task.max_retries}: {task.task_id}")
                except:
                    task.status = TaskStatus.FAILED
                    task.error = f"Retry failed: {str(e)}"
                    self.stats['total_failed'] += 1
            else:
                self.stats['total_failed'] += 1
                print(f"タスク失敗: {task.task_id} - {e}")

            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {
                    "task_id": task.task_id,
                    "function": task.function.__name__,
                    "retry_count": task.retry_count
                })

    def get_task_status(self, task_id: str) -> Optional[AsyncTask]:
        """タスク状態を取得"""
        with self._lock:
            return self.tasks.get(task_id)

    def get_task_result(self, task_id: str, wait_timeout: float = None) -> Any:
        """タスク結果を取得（完了まで待機）"""
        task = self.get_task_status(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        # 完了を待機
        start_wait = time.time()
        while task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            if wait_timeout and (time.time() - start_wait) > wait_timeout:
                raise TimeoutError(f"Wait timeout for task: {task_id}")

            time.sleep(0.1)
            task = self.get_task_status(task_id)

        if task.status == TaskStatus.COMPLETED:
            return task.result
        elif task.status == TaskStatus.FAILED:
            raise RuntimeError(f"Task failed: {task.error}")
        elif task.status == TaskStatus.CANCELLED:
            raise RuntimeError("Task was cancelled")
        else:
            raise RuntimeError(f"Unknown task status: {task.status}")

    def cancel_task(self, task_id: str) -> bool:
        """タスクをキャンセル"""
        with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False

            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                self.stats['total_cancelled'] += 1
                return True

            return False

    def get_queue_stats(self) -> Dict[str, Any]:
        """キュー統計を取得"""
        with self._lock:
            pending_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.PENDING)
            running_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.RUNNING)

            return {
                'queue_size': self.task_queue.qsize(),
                'max_queue_size': self.queue_size,
                'workers': len(self.workers),
                'max_workers': self.max_workers,
                'tasks': {
                    'total': len(self.tasks),
                    'pending': pending_tasks,
                    'running': running_tasks,
                    'completed': self.stats['total_completed'],
                    'failed': self.stats['total_failed'],
                    'cancelled': self.stats['total_cancelled']
                },
                'statistics': dict(self.stats)
            }


class HighPerformanceCacheSystem:
    """高性能キャッシュシステム"""

    def __init__(self,
                 memory_cache_mb: int = 100,
                 redis_host: Optional[str] = None,
                 redis_port: int = 6379,
                 redis_db: int = 0):

        # インメモリキャッシュ
        self.memory_cache = InMemoryCache(max_size_mb=memory_cache_mb)

        # Redisキャッシュ（オプション）
        self.redis_client = None
        if HAS_REDIS and redis_host:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False,
                    socket_timeout=1.0
                )
                # 接続テスト
                self.redis_client.ping()
                print(f"Redis connected: {redis_host}:{redis_port}")
            except Exception as e:
                print(f"Redis connection failed: {e}")
                self.redis_client = None

        # 統計
        self.cache_hits = 0
        self.cache_misses = 0
        self._stats_lock = threading.Lock()

        # クリーンアップスレッド
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def get(self, key: str) -> Optional[Any]:
        """キャッシュからデータを取得"""
        # 1. インメモリキャッシュをチェック
        value = self.memory_cache.get(key)
        if value is not None:
            with self._stats_lock:
                self.cache_hits += 1
            return value

        # 2. Redisキャッシュをチェック
        if self.redis_client:
            try:
                redis_value = self.redis_client.get(key)
                if redis_value is not None:
                    # デシリアライズしてメモリキャッシュにも保存
                    value = pickle.loads(redis_value)
                    self.memory_cache.put(key, value, ttl_seconds=1800)  # 30分

                    with self._stats_lock:
                        self.cache_hits += 1
                    return value
            except Exception as e:
                print(f"Redis get error: {e}")

        # キャッシュミス
        with self._stats_lock:
            self.cache_misses += 1
        return None

    def put(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """データをキャッシュに保存"""
        success = True

        # 1. インメモリキャッシュに保存
        if not self.memory_cache.put(key, value, ttl_seconds):
            success = False

        # 2. Redisキャッシュに保存
        if self.redis_client:
            try:
                serialized = pickle.dumps(value)
                self.redis_client.setex(key, ttl_seconds, serialized)
            except Exception as e:
                print(f"Redis put error: {e}")
                success = False

        return success

    def delete(self, key: str) -> bool:
        """キャッシュからデータを削除"""
        # インメモリから削除
        with self.memory_cache._lock:
            if key in self.memory_cache.cache:
                self.memory_cache._remove_entry(key)

        # Redisから削除
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                print(f"Redis delete error: {e}")
                return False

        return True

    def clear(self):
        """すべてのキャッシュをクリア"""
        # インメモリクリア
        with self.memory_cache._lock:
            self.memory_cache.cache.clear()
            self.memory_cache.current_size_bytes = 0

        # Redisクリア
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                print(f"Redis clear error: {e}")

    def _cleanup_loop(self):
        """定期クリーンアップ"""
        while True:
            try:
                time.sleep(300)  # 5分間隔
                self.memory_cache.clear_expired()
            except Exception as e:
                print(f"Cleanup error: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        with self._stats_lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

            stats = {
                'requests': {
                    'hits': self.cache_hits,
                    'misses': self.cache_misses,
                    'total': total_requests,
                    'hit_rate_percent': hit_rate
                },
                'memory_cache': self.memory_cache.get_stats(),
                'redis_connected': self.redis_client is not None
            }

            # Redis統計
            if self.redis_client:
                try:
                    redis_info = self.redis_client.info()
                    stats['redis_stats'] = {
                        'used_memory_mb': redis_info.get('used_memory', 0) / 1024 / 1024,
                        'keyspace_hits': redis_info.get('keyspace_hits', 0),
                        'keyspace_misses': redis_info.get('keyspace_misses', 0)
                    }
                except Exception:
                    stats['redis_stats'] = {'error': 'Unable to fetch Redis stats'}

            return stats


class AsyncProcessingSystem:
    """統合非同期処理システム"""

    def __init__(self, max_workers: int = 4, cache_mb: int = 100, redis_host: Optional[str] = None):
        self.task_manager = AsyncTaskManager(max_workers=max_workers)
        self.cache_system = HighPerformanceCacheSystem(
            memory_cache_mb=cache_mb,
            redis_host=redis_host
        )

        # システム開始
        self.task_manager.start()

        print("Async Processing System initialized")

    def submit_cached_task(self,
                          function: Callable,
                          args: tuple = (),
                          kwargs: dict = None,
                          cache_key: Optional[str] = None,
                          cache_ttl: int = 3600,
                          priority: TaskPriority = TaskPriority.NORMAL,
                          force_refresh: bool = False) -> Union[str, Any]:
        """キャッシュ付きタスクを投入"""

        # キャッシュキー生成
        if cache_key is None:
            cache_data = f"{function.__name__}_{args}_{kwargs}"
            cache_key = hashlib.md5(cache_data.encode()).hexdigest()

        # キャッシュチェック
        if not force_refresh:
            cached_result = self.cache_system.get(cache_key)
            if cached_result is not None:
                return cached_result

        # タスク投入
        task_id = self.task_manager.submit_task(
            function=function,
            args=args,
            kwargs=kwargs or {},
            priority=priority
        )

        return task_id

    def get_cached_result(self, task_id_or_key: str, wait_timeout: float = None) -> Any:
        """結果を取得（キャッシュまたはタスク完了待ち）"""
        # まずキャッシュをチェック
        cached_result = self.cache_system.get(task_id_or_key)
        if cached_result is not None:
            return cached_result

        # タスク結果を取得
        try:
            result = self.task_manager.get_task_result(task_id_or_key, wait_timeout)

            # 結果をキャッシュに保存
            self.cache_system.put(task_id_or_key, result)

            return result

        except Exception as e:
            print(f"Result retrieval error: {e}")
            raise

    def shutdown(self):
        """システム終了"""
        self.task_manager.stop()
        print("Async Processing System shutdown")

    def get_system_stats(self) -> Dict[str, Any]:
        """システム統計を取得"""
        return {
            'task_manager': self.task_manager.get_queue_stats(),
            'cache_system': self.cache_system.get_cache_stats(),
            'system_info': {
                'max_workers': self.task_manager.max_workers,
                'redis_available': HAS_REDIS,
                'redis_connected': self.cache_system.redis_client is not None
            }
        }


# グローバルインスタンス
async_processing_system = AsyncProcessingSystem()


# デコレーター
def async_cached(cache_ttl: int = 3600, priority: TaskPriority = TaskPriority.NORMAL):
    """非同期キャッシュデコレーター"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return async_processing_system.submit_cached_task(
                function=func,
                args=args,
                kwargs=kwargs,
                cache_ttl=cache_ttl,
                priority=priority
            )
        return wrapper
    return decorator


if __name__ == "__main__":
    # テスト実行
    print("Async Processing System テスト開始")

    # テスト用の重い処理
    def heavy_computation(n: int, sleep_time: float = 1.0) -> int:
        """重い計算のシミュレーション"""
        time.sleep(sleep_time)
        result = sum(i * i for i in range(n))
        print(f"Computation completed: n={n}, result={result}")
        return result

    def fetch_stock_data(symbol: str) -> dict:
        """株価データ取得のシミュレーション"""
        time.sleep(2.0)  # API呼び出しのシミュレーション
        return {
            'symbol': symbol,
            'price': 1500 + hash(symbol) % 1000,
            'volume': 1000000 + hash(symbol) % 500000,
            'timestamp': datetime.now().isoformat()
        }

    # システムインスタンス作成
    system = AsyncProcessingSystem(max_workers=3, cache_mb=50)

    print("\n1. 非同期タスクテスト")

    # 複数のタスクを投入
    task_ids = []
    for i in range(5):
        task_id = system.submit_cached_task(
            function=heavy_computation,
            args=(1000,),
            kwargs={'sleep_time': 1.0},
            priority=TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.HIGH
        )
        task_ids.append(task_id)
        print(f"Task submitted: {task_id}")

    print("\n2. タスク結果取得")

    # 結果を取得
    start_time = time.time()
    for task_id in task_ids:
        try:
            if isinstance(task_id, str):  # タスクIDの場合
                result = system.get_cached_result(task_id, wait_timeout=10.0)
                print(f"Task {task_id[:8]}... result: {result}")
            else:  # キャッシュヒットの場合
                print(f"Cache hit: {task_id}")
        except Exception as e:
            print(f"Task error: {e}")

    elapsed = time.time() - start_time
    print(f"All tasks completed in {elapsed:.2f} seconds")

    print("\n3. キャッシュテスト")

    # 同じタスクを再実行（キャッシュヒット）
    cached_task = system.submit_cached_task(
        function=heavy_computation,
        args=(1000,),
        kwargs={'sleep_time': 1.0}
    )
    print(f"Cached task result: {cached_task}")

    print("\n4. 株価データ取得テスト")

    # 株価データ取得
    stock_symbols = ['7203', '9984', '4751']
    stock_tasks = []

    for symbol in stock_symbols:
        task_id = system.submit_cached_task(
            function=fetch_stock_data,
            args=(symbol,),
            priority=TaskPriority.HIGH
        )
        stock_tasks.append((symbol, task_id))

    # 結果取得
    for symbol, task_id in stock_tasks:
        try:
            if isinstance(task_id, str):
                result = system.get_cached_result(task_id, wait_timeout=5.0)
                print(f"Stock data for {symbol}: {result}")
            else:
                print(f"Cached stock data for {symbol}: {task_id}")
        except Exception as e:
            print(f"Stock data error for {symbol}: {e}")

    print("\n5. システム統計")
    stats = system.get_system_stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))

    print("\n6. システム終了")
    system.shutdown()

    print("テスト完了 ✅")