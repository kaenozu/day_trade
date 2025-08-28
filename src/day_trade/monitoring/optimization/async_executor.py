#!/usr/bin/env python3
"""
非同期処理最適化

効率的な非同期処理とコルーチン管理
"""

import asyncio
import concurrent.futures
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional
from functools import wraps


class AsyncTaskManager:
    """非同期タスクマネージャー"""

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.running_tasks = set()
        self.completed_tasks = []
        self.failed_tasks = []

    async def run_task(self, coro: Awaitable) -> Any:
        """タスク実行"""
        async with self.semaphore:
            task = asyncio.create_task(coro)
            self.running_tasks.add(task)

            try:
                result = await task
                self.completed_tasks.append(task)
                return result
            except Exception as e:
                self.failed_tasks.append((task, e))
                raise
            finally:
                self.running_tasks.discard(task)

    async def run_batch(self, coros: List[Awaitable]) -> List[Any]:
        """バッチ実行"""
        tasks = [self.run_task(coro) for coro in coros]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self) -> Dict[str, int]:
        """統計取得"""
        return {
            'running': len(self.running_tasks),
            'completed': len(self.completed_tasks),
            'failed': len(self.failed_tasks),
            'max_concurrent': self.max_concurrent
        }

    async def cancel_all_tasks(self):
        """全タスクをキャンセル"""
        for task in self.running_tasks.copy():
            task.cancel()
        
        # キャンセルの完了を待機
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks, return_exceptions=True)


class HybridExecutor:
    """ハイブリッド実行器（同期・非同期）"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    async def run_cpu_bound(self, func: Callable, *args, **kwargs) -> Any:
        """CPU集約的タスクを別プロセスで実行"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_pool, func, *args, **kwargs)

    async def run_io_bound(self, func: Callable, *args, **kwargs) -> Any:
        """I/O集約的タスクを別スレッドで実行"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)

    def cleanup(self):
        """リソースクリーンアップ"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class AsyncCache:
    """非同期キャッシュ"""

    def __init__(self, ttl: float = 3600):
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
        self._locks = {}

    async def get_or_set(self, key: str, coro_func: Callable[[], Awaitable]) -> Any:
        """キャッシュ取得または設定"""
        # キー専用ロック取得
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:
            # キャッシュチェック
            if key in self._cache:
                timestamp = self._timestamps.get(key, 0)
                if time.time() - timestamp < self.ttl:
                    return self._cache[key]

            # キャッシュミス：値を計算
            result = await coro_func()
            self._cache[key] = result
            self._timestamps[key] = time.time()
            return result

    def invalidate(self, key: str):
        """キャッシュ無効化"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._locks.pop(key, None)

    def clear(self):
        """全キャッシュクリア"""
        self._cache.clear()
        self._timestamps.clear()
        self._locks.clear()


class RateLimiter:
    """レート制限器"""

    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """レート制限チェック"""
        async with self.lock:
            now = time.time()

            # 古い呼び出し記録を削除
            self.calls = [call_time for call_time in self.calls
                         if now - call_time < self.time_window]

            # 制限チェック
            if len(self.calls) >= self.max_calls:
                # 待機時間計算
                oldest_call = min(self.calls)
                wait_time = self.time_window - (now - oldest_call)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            # 現在の呼び出しを記録
            self.calls.append(now)


def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """非同期リトライデコレータ"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        raise last_exception

        return wrapper
    return decorator


def async_timeout(timeout: float):
    """非同期タイムアウトデコレータ"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        return wrapper
    return decorator


class AsyncBatchProcessor:
    """非同期バッチ処理器"""

    def __init__(self, batch_size: int = 100, max_wait: float = 1.0):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.queue = asyncio.Queue()
        self.processor_task = None
        self.running = False

    async def add_item(self, item: Any) -> Any:
        """アイテム追加"""
        future = asyncio.Future()
        await self.queue.put((item, future))
        return await future

    async def start_processing(self, processor_func: Callable):
        """処理開始"""
        self.running = True
        self.processor_task = asyncio.create_task(
            self._process_batches(processor_func)
        )

    async def stop_processing(self):
        """処理停止"""
        self.running = False
        if self.processor_task:
            await self.processor_task

    async def _process_batches(self, processor_func: Callable):
        """バッチ処理ループ"""
        while self.running:
            batch = []
            futures = []

            # バッチ収集
            try:
                # 最初のアイテムを待機
                item, future = await asyncio.wait_for(
                    self.queue.get(), timeout=self.max_wait
                )
                batch.append(item)
                futures.append(future)

                # 追加アイテムを収集（ノンブロッキング）
                while len(batch) < self.batch_size:
                    try:
                        item, future = self.queue.get_nowait()
                        batch.append(item)
                        futures.append(future)
                    except asyncio.QueueEmpty:
                        break

            except asyncio.TimeoutError:
                continue

            # バッチ処理実行
            try:
                results = await processor_func(batch)

                # 結果を対応するFutureに設定
                for future, result in zip(futures, results):
                    if not future.cancelled():
                        future.set_result(result)

            except Exception as e:
                # エラーを全Futureに設定
                for future in futures:
                    if not future.cancelled():
                        future.set_exception(e)


class AsyncTaskScheduler:
    """非同期タスクスケジューラー"""
    
    def __init__(self):
        self.scheduled_tasks = {}
        self.running = False
    
    def schedule_task(self, name: str, coro_func: Callable, 
                     interval: float) -> str:
        """定期実行タスクをスケジュール"""
        task_info = {
            'name': name,
            'coro_func': coro_func,
            'interval': interval,
            'task': None,
            'last_run': 0
        }
        self.scheduled_tasks[name] = task_info
        return name
    
    async def start_scheduler(self):
        """スケジューラー開始"""
        self.running = True
        await self._scheduler_loop()
    
    def stop_scheduler(self):
        """スケジューラー停止"""
        self.running = False
        for task_info in self.scheduled_tasks.values():
            if task_info['task']:
                task_info['task'].cancel()
    
    async def _scheduler_loop(self):
        """スケジューラーループ"""
        while self.running:
            current_time = time.time()
            
            for name, task_info in self.scheduled_tasks.items():
                if (current_time - task_info['last_run'] >= task_info['interval'] 
                    and (not task_info['task'] or task_info['task'].done())):
                    
                    # タスク実行
                    task_info['task'] = asyncio.create_task(
                        task_info['coro_func']()
                    )
                    task_info['last_run'] = current_time
            
            await asyncio.sleep(1.0)  # 1秒間隔でチェック


# グローバルインスタンス
task_manager = AsyncTaskManager()
hybrid_executor = HybridExecutor()
async_cache = AsyncCache()


# 便利な関数
async def run_parallel(coros: List[Awaitable], max_concurrent: int = 10) -> List[Any]:
    """並列実行"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(coro):
        async with semaphore:
            return await coro

    tasks = [run_with_semaphore(coro) for coro in coros]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def async_map(func: Callable, items: List[Any], 
                   max_concurrent: int = 10) -> List[Any]:
    """非同期マップ処理"""
    coros = [func(item) for item in items]
    return await run_parallel(coros, max_concurrent)


def async_cached(cache: AsyncCache, key_func: Callable = None):
    """非同期キャッシュデコレータ"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # キー生成
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # キャッシュから取得または実行
            return await cache.get_or_set(cache_key, 
                                        lambda: func(*args, **kwargs))
        
        return wrapper
    return decorator


# 使用例:
# @async_retry(max_retries=3)
# @async_timeout(10.0)
# async def fetch_stock_data(symbol):
#     # API呼び出し等
#     pass