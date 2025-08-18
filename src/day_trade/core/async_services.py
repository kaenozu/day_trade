#!/usr/bin/env python3
"""
非同期・並行処理サービス - 依存性注入版
Issue #918 項目8対応: 並行性・非同期処理の改善

高度な非同期処理とワーカー管理システム
"""

import asyncio
import concurrent.futures
import time
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil

from .dependency_injection import (
    IConfigurationService, ILoggingService, injectable, singleton, get_container
)
from .database_services import IDatabaseService, ICacheService
from ..utils.logging_config import get_context_logger


class TaskPriority(Enum):
    """タスク優先度"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class WorkerType(Enum):
    """ワーカータイプ"""
    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    MIXED = "mixed"


class ExecutionStrategy(Enum):
    """実行戦略"""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class TaskResult:
    """タスク実行結果"""
    task_id: str
    status: str
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkerStats:
    """ワーカー統計"""
    worker_id: str
    worker_type: WorkerType
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    current_load: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0


@dataclass
class AsyncTask:
    """非同期タスク定義"""
    task_id: str
    func: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    worker_type: WorkerType = WorkerType.IO_BOUND
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


class IAsyncExecutorService(ABC):
    """非同期実行サービスインターフェース"""

    @abstractmethod
    async def submit_task(self, task: AsyncTask) -> str:
        """タスク投入"""
        pass

    @abstractmethod
    async def submit_batch(self, tasks: List[AsyncTask]) -> List[str]:
        """バッチタスク投入"""
        pass

    @abstractmethod
    async def get_result(self, task_id: str) -> TaskResult:
        """結果取得"""
        pass

    @abstractmethod
    async def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> List[TaskResult]:
        """完了待機"""
        pass

    @abstractmethod
    def get_worker_stats(self) -> Dict[str, WorkerStats]:
        """ワーカー統計取得"""
        pass


class IProgressMonitorService(ABC):
    """進捗監視サービスインターフェース"""

    @abstractmethod
    def start_progress(self, task_id: str, total_items: int):
        """進捗開始"""
        pass

    @abstractmethod
    def update_progress(self, task_id: str, completed_items: int):
        """進捗更新"""
        pass

    @abstractmethod
    def get_progress(self, task_id: str) -> Dict[str, Any]:
        """進捗取得"""
        pass


class ISchedulerService(ABC):
    """スケジューラーサービスインターフェース"""

    @abstractmethod
    def schedule_periodic(self, func: Callable, interval: timedelta, **kwargs):
        """定期実行スケジュール"""
        pass

    @abstractmethod
    def schedule_at(self, func: Callable, run_at: datetime, **kwargs):
        """指定時刻実行スケジュール"""
        pass

    @abstractmethod
    def cancel_scheduled(self, schedule_id: str):
        """スケジュールキャンセル"""
        pass


@singleton(IAsyncExecutorService)
@injectable
class AdvancedAsyncExecutorService(IAsyncExecutorService):
    """高度非同期実行サービス実装"""

    def __init__(self, 
                 config_service: IConfigurationService,
                 logging_service: ILoggingService,
                 db_service: Optional[IDatabaseService] = None,
                 cache_service: Optional[ICacheService] = None):
        self.config_service = config_service
        self.logging_service = logging_service
        self.db_service = db_service
        self.cache_service = cache_service
        
        self.logger = logging_service.get_logger(__name__, "AdvancedAsyncExecutorService")
        
        # 実行器設定
        self._setup_executors()
        
        # タスク管理
        self._tasks: Dict[str, AsyncTask] = {}
        self._task_futures: Dict[str, asyncio.Future] = {}
        self._results: Dict[str, TaskResult] = {}
        
        # ワーカー統計
        self._worker_stats: Dict[str, WorkerStats] = {}
        
        # 制御設定
        self._max_concurrent_tasks = self._get_max_concurrent_tasks()
        self._running_tasks = 0
        self._task_semaphore = asyncio.Semaphore(self._max_concurrent_tasks)
        
        # バックグラウンド監視タスク
        self._monitoring_task = None
        self._start_monitoring()
        
        self.logger.info(f"AdvancedAsyncExecutorService initialized with {self._max_concurrent_tasks} max concurrent tasks")

    def _setup_executors(self):
        """実行器セットアップ"""
        config = self.config_service.get_config()
        async_config = config.get('async_processing', {})
        
        # CPU集約処理用
        cpu_workers = async_config.get('cpu_workers', min(4, mp.cpu_count()))
        self._cpu_executor = ProcessPoolExecutor(max_workers=cpu_workers)
        
        # I/O集約処理用  
        io_workers = async_config.get('io_workers', min(20, mp.cpu_count() * 4))
        self._io_executor = ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="AsyncIO")
        
        self.logger.info(f"Executors initialized: CPU={cpu_workers}, IO={io_workers}")

    def _get_max_concurrent_tasks(self) -> int:
        """最大同時実行タスク数取得"""
        config = self.config_service.get_config()
        return config.get('async_processing', {}).get('max_concurrent_tasks', 50)

    def _start_monitoring(self):
        """監視タスク開始"""
        try:
            # イベントループが実行中の場合のみ監視タスクを開始
            loop = asyncio.get_running_loop()
            if self._monitoring_task is None or self._monitoring_task.done():
                self._monitoring_task = asyncio.create_task(self._monitor_workers())
        except RuntimeError:
            # イベントループが実行中でない場合は監視タスクをスキップ
            self.logger.warning("No running event loop - monitoring task will start when async operations begin")
            self._monitoring_task = None

    async def _monitor_workers(self):
        """ワーカー監視ループ"""
        while True:
            try:
                await self._update_worker_stats()
                await asyncio.sleep(30)  # 30秒間隔で監視
            except Exception as e:
                self.logger.error(f"Worker monitoring error: {e}")
                await asyncio.sleep(60)

    async def _update_worker_stats(self):
        """ワーカー統計更新"""
        try:
            # システムリソース取得
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # 実行器統計更新
            for worker_id, stats in self._worker_stats.items():
                stats.cpu_usage = cpu_percent / len(self._worker_stats) if self._worker_stats else cpu_percent
                stats.memory_usage = memory_percent / len(self._worker_stats) if self._worker_stats else memory_percent
                
        except Exception as e:
            self.logger.warning(f"Failed to update worker stats: {e}")

    async def submit_task(self, task: AsyncTask) -> str:
        """タスク投入"""
        task_id = task.task_id
        self._tasks[task_id] = task
        
        # 監視タスクが開始されていない場合は開始
        if self._monitoring_task is None:
            try:
                self._monitoring_task = asyncio.create_task(self._monitor_workers())
            except Exception as e:
                self.logger.warning(f"Could not start monitoring task: {e}")
        
        # 非同期実行開始
        future = asyncio.create_task(self._execute_task(task))
        self._task_futures[task_id] = future
        
        self.logger.debug(f"Task submitted: {task_id}")
        return task_id

    async def submit_batch(self, tasks: List[AsyncTask]) -> List[str]:
        """バッチタスク投入"""
        task_ids = []
        
        for task in tasks:
            task_id = await self.submit_task(task)
            task_ids.append(task_id)
        
        self.logger.info(f"Batch submitted: {len(tasks)} tasks")
        return task_ids

    async def _execute_task(self, task: AsyncTask) -> TaskResult:
        """タスク実行"""
        async with self._task_semaphore:
            self._running_tasks += 1
            start_time = time.time()
            
            try:
                # ワーカータイプに応じた実行器選択
                if task.worker_type == WorkerType.CPU_BOUND:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._cpu_executor, task.func, *task.args, **task.kwargs
                    )
                elif task.worker_type == WorkerType.IO_BOUND:
                    if asyncio.iscoroutinefunction(task.func):
                        result = await task.func(*task.args, **task.kwargs)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self._io_executor, task.func, *task.args, **task.kwargs
                        )
                else:  # MIXED
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._io_executor, task.func, *task.args, **task.kwargs
                    )
                
                execution_time = time.time() - start_time
                
                # 結果保存
                task_result = TaskResult(
                    task_id=task.task_id,
                    status="completed",
                    result=result,
                    execution_time=execution_time
                )
                
                self._results[task.task_id] = task_result
                self._update_worker_stat(task, execution_time, success=True)
                
                return task_result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # リトライ処理
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    self.logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
                    await asyncio.sleep(2 ** task.retry_count)  # 指数バックオフ
                    return await self._execute_task(task)
                
                # 失敗結果保存
                task_result = TaskResult(
                    task_id=task.task_id,
                    status="failed",
                    error=e,
                    execution_time=execution_time
                )
                
                self._results[task.task_id] = task_result
                self._update_worker_stat(task, execution_time, success=False)
                
                self.logger.error(f"Task {task.task_id} failed permanently: {e}")
                return task_result
                
            finally:
                self._running_tasks -= 1

    def _update_worker_stat(self, task: AsyncTask, execution_time: float, success: bool):
        """ワーカー統計更新"""
        worker_id = f"{task.worker_type.value}_worker"
        
        if worker_id not in self._worker_stats:
            self._worker_stats[worker_id] = WorkerStats(
                worker_id=worker_id,
                worker_type=task.worker_type
            )
        
        stats = self._worker_stats[worker_id]
        stats.total_tasks += 1
        
        if success:
            stats.completed_tasks += 1
        else:
            stats.failed_tasks += 1
        
        # 平均実行時間更新
        stats.avg_execution_time = (
            (stats.avg_execution_time * (stats.total_tasks - 1) + execution_time) / stats.total_tasks
        )

    async def get_result(self, task_id: str) -> TaskResult:
        """結果取得"""
        if task_id in self._results:
            return self._results[task_id]
        
        if task_id in self._task_futures:
            future = self._task_futures[task_id]
            if future.done():
                return future.result()
            else:
                # まだ実行中
                return TaskResult(task_id=task_id, status="running")
        
        # タスクが見つからない
        return TaskResult(task_id=task_id, status="not_found")

    async def wait_for_completion(self, task_ids: List[str], timeout: Optional[float] = None) -> List[TaskResult]:
        """完了待機"""
        futures = [self._task_futures[task_id] for task_id in task_ids if task_id in self._task_futures]
        
        if not futures:
            return [TaskResult(task_id=task_id, status="not_found") for task_id in task_ids]
        
        try:
            if timeout:
                await asyncio.wait_for(asyncio.gather(*futures), timeout=timeout)
            else:
                await asyncio.gather(*futures)
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for tasks: {task_ids}")
        
        return [await self.get_result(task_id) for task_id in task_ids]

    def get_worker_stats(self) -> Dict[str, WorkerStats]:
        """ワーカー統計取得"""
        # 現在の負荷情報更新
        for stats in self._worker_stats.values():
            stats.current_load = self._running_tasks / self._max_concurrent_tasks
        
        return self._worker_stats.copy()

    def shutdown(self):
        """シャットダウン"""
        self.logger.info("Shutting down AsyncExecutorService")
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        self._cpu_executor.shutdown(wait=True)
        self._io_executor.shutdown(wait=True)


@singleton(IProgressMonitorService)  
@injectable
class ProgressMonitorService(IProgressMonitorService):
    """進捗監視サービス実装"""

    def __init__(self, logging_service: ILoggingService):
        self.logging_service = logging_service
        self.logger = logging_service.get_logger(__name__, "ProgressMonitorService")
        
        self._progress_data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def start_progress(self, task_id: str, total_items: int):
        """進捗開始"""
        with self._lock:
            self._progress_data[task_id] = {
                'total_items': total_items,
                'completed_items': 0,
                'start_time': datetime.now(),
                'last_update': datetime.now(),
                'status': 'running'
            }
        
        self.logger.info(f"Progress started for task {task_id}: {total_items} items")

    def update_progress(self, task_id: str, completed_items: int):
        """進捗更新"""
        with self._lock:
            if task_id not in self._progress_data:
                self.logger.warning(f"Progress not initialized for task {task_id}")
                return
            
            progress = self._progress_data[task_id]
            progress['completed_items'] = completed_items
            progress['last_update'] = datetime.now()
            
            # 完了チェック
            if completed_items >= progress['total_items']:
                progress['status'] = 'completed'

    def get_progress(self, task_id: str) -> Dict[str, Any]:
        """進捗取得"""
        with self._lock:
            if task_id not in self._progress_data:
                return {'status': 'not_found'}
            
            progress = self._progress_data[task_id].copy()
            
            # 計算フィールド追加
            if progress['total_items'] > 0:
                progress['percentage'] = (progress['completed_items'] / progress['total_items']) * 100
            else:
                progress['percentage'] = 0
            
            # 経過時間
            elapsed = datetime.now() - progress['start_time']
            progress['elapsed_time'] = elapsed.total_seconds()
            
            # 残り時間推定
            if progress['completed_items'] > 0 and progress['status'] == 'running' and elapsed.total_seconds() > 0:
                rate = progress['completed_items'] / elapsed.total_seconds()
                remaining_items = progress['total_items'] - progress['completed_items']
                estimated_remaining = remaining_items / rate if rate > 0 else 0
                progress['estimated_remaining'] = estimated_remaining
            else:
                progress['estimated_remaining'] = 0
            
            return progress


@singleton(ISchedulerService)
@injectable
class SchedulerService(ISchedulerService):
    """スケジューラーサービス実装"""

    def __init__(self, 
                 logging_service: ILoggingService,
                 async_executor: IAsyncExecutorService):
        self.logging_service = logging_service
        self.async_executor = async_executor
        self.logger = logging_service.get_logger(__name__, "SchedulerService")
        
        self._scheduled_tasks: Dict[str, Dict[str, Any]] = {}
        self._scheduler_task = None
        self._running = False
        
        self._start_scheduler()

    def _start_scheduler(self):
        """スケジューラー開始"""
        if not self._running:
            self._running = True
            try:
                # イベントループが実行中の場合のみスケジューラーを開始
                loop = asyncio.get_running_loop()
                self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            except RuntimeError:
                # イベントループが実行中でない場合はスケジューラーをスキップ
                self.logger.warning("No running event loop - scheduler will start when async operations begin")
                self._scheduler_task = None

    async def _scheduler_loop(self):
        """スケジューラーメインループ"""
        while self._running:
            try:
                current_time = datetime.now()
                
                # 実行予定のタスクをチェック
                for schedule_id, scheduled_item in list(self._scheduled_tasks.items()):
                    if scheduled_item['next_run'] <= current_time:
                        await self._execute_scheduled_task(schedule_id, scheduled_item)
                
                await asyncio.sleep(1)  # 1秒間隔でチェック
                
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(10)

    async def _execute_scheduled_task(self, schedule_id: str, scheduled_item: Dict[str, Any]):
        """スケジュールタスク実行"""
        try:
            func = scheduled_item['func']
            kwargs = scheduled_item.get('kwargs', {})
            
            # 非同期タスクとして実行
            task = AsyncTask(
                task_id=f"scheduled_{schedule_id}_{int(time.time())}",
                func=func,
                kwargs=kwargs,
                priority=TaskPriority.NORMAL
            )
            
            await self.async_executor.submit_task(task)
            
            # 次回実行時刻更新（定期実行の場合）
            if scheduled_item.get('interval'):
                scheduled_item['next_run'] = datetime.now() + scheduled_item['interval']
            else:
                # 単発実行の場合は削除
                del self._scheduled_tasks[schedule_id]
            
        except Exception as e:
            self.logger.error(f"Scheduled task execution error [{schedule_id}]: {e}")

    def schedule_periodic(self, func: Callable, interval: timedelta, **kwargs):
        """定期実行スケジュール"""
        schedule_id = f"periodic_{int(time.time())}_{id(func)}"
        
        self._scheduled_tasks[schedule_id] = {
            'func': func,
            'interval': interval,
            'kwargs': kwargs,
            'next_run': datetime.now() + interval,
            'type': 'periodic'
        }
        
        # スケジューラーがまだ開始されていない場合は開始
        if self._scheduler_task is None and self._running:
            try:
                self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            except Exception as e:
                self.logger.warning(f"Could not start scheduler task: {e}")
        
        self.logger.info(f"Scheduled periodic task: {schedule_id} (interval: {interval})")
        return schedule_id

    def schedule_at(self, func: Callable, run_at: datetime, **kwargs):
        """指定時刻実行スケジュール"""
        schedule_id = f"at_{int(time.time())}_{id(func)}"
        
        self._scheduled_tasks[schedule_id] = {
            'func': func,
            'kwargs': kwargs,
            'next_run': run_at,
            'type': 'one_time'
        }
        
        self.logger.info(f"Scheduled one-time task: {schedule_id} (run at: {run_at})")
        return schedule_id

    def cancel_scheduled(self, schedule_id: str):
        """スケジュールキャンセル"""
        if schedule_id in self._scheduled_tasks:
            del self._scheduled_tasks[schedule_id]
            self.logger.info(f"Cancelled scheduled task: {schedule_id}")

    def shutdown(self):
        """シャットダウン"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()


def register_async_services():
    """非同期サービスを登録"""
    container = get_container()
    
    # 非同期実行サービス
    if not container.is_registered(IAsyncExecutorService):
        container.register_singleton(IAsyncExecutorService, AdvancedAsyncExecutorService)
    
    # 進捗監視サービス
    if not container.is_registered(IProgressMonitorService):
        container.register_singleton(IProgressMonitorService, ProgressMonitorService)
    
    # スケジューラーサービス
    if not container.is_registered(ISchedulerService):
        container.register_singleton(ISchedulerService, SchedulerService)


# 便利な非同期ヘルパー関数
async def run_parallel_analysis(symbols: List[str], analysis_func: Callable, max_concurrent: int = 10) -> Dict[str, Any]:
    """並列分析実行ヘルパー"""
    container = get_container()
    executor = container.resolve(IAsyncExecutorService)
    progress_monitor = container.resolve(IProgressMonitorService)
    
    task_id = f"parallel_analysis_{int(time.time())}"
    progress_monitor.start_progress(task_id, len(symbols))
    
    # タスク作成
    tasks = []
    for i, symbol in enumerate(symbols):
        task = AsyncTask(
            task_id=f"{task_id}_{i}",
            func=analysis_func,
            args=(symbol,),
            worker_type=WorkerType.IO_BOUND
        )
        tasks.append(task)
    
    # バッチ実行
    task_ids = await executor.submit_batch(tasks)
    
    # 結果収集
    results = {}
    for i, symbol in enumerate(symbols):
        result = await executor.get_result(task_ids[i])
        results[symbol] = result.result if result.status == "completed" else result.error
        progress_monitor.update_progress(task_id, i + 1)
    
    return results