#!/usr/bin/env python3
"""
並列バッチ処理エンジン
大規模データ処理のための高性能並列バッチシステム

主要機能:
- 動的ワーカー管理
- 負荷分散とタスクキューイング
- 失敗時の自動リトライ
- リアルタイム性能監視
- 適応的リソース調整
- メモリ効率的なストリーミング処理
"""

import asyncio
import gc
import multiprocessing
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from enum import Enum
from queue import PriorityQueue, Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import psutil

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class TaskPriority(Enum):
    """タスク優先度"""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    REALTIME = 5


class ProcessingMode(Enum):
    """処理モード"""

    THREAD_BASED = "thread"  # スレッドベース（I/Oバウンド）
    PROCESS_BASED = "process"  # プロセスベース（CPUバウンド）
    HYBRID = "hybrid"  # ハイブリッド（自動選択）
    ASYNC = "async"  # 非同期（高並行性）


class TaskStatus(Enum):
    """タスクステータス"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class BatchTask:
    """バッチタスク定義"""

    task_id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: float = 60.0
    retry_count: int = 3
    retry_delay: float = 1.0
    memory_limit_mb: int = 512
    cpu_weight: float = 1.0  # CPU負荷重み
    io_weight: float = 1.0  # I/O負荷重み
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """優先度による比較"""
        return self.priority.value > other.priority.value


@dataclass
class TaskResult:
    """タスク実行結果"""

    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_peak_mb: float = 0.0
    retry_count: int = 0
    worker_id: str = ""
    status: TaskStatus = TaskStatus.COMPLETED
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerStats:
    """ワーカー統計"""

    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_activity: float = field(default_factory=time.time)
    is_active: bool = True


@dataclass
class EngineStats:
    """エンジン統計"""

    total_tasks_submitted: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    throughput_tps: float = 0.0  # Tasks Per Second
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_workers: int = 0
    queue_size: int = 0
    uptime_seconds: float = 0.0


class AdaptiveResourceManager:
    """適応的リソース管理"""

    def __init__(self, initial_workers: int = 4, max_workers: int = None):
        self.initial_workers = initial_workers
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2
        self.min_workers = max(1, initial_workers // 2)

        # システムリソース監視
        self.cpu_threshold_high = 85.0  # CPU使用率上限
        self.cpu_threshold_low = 30.0  # CPU使用率下限
        self.memory_threshold_high = 85.0  # メモリ使用率上限
        self.memory_threshold_low = 50.0  # メモリ使用率下限

        # 適応制御
        self.adjustment_interval = 10.0  # 調整間隔（秒）
        self.last_adjustment = 0.0
        self.performance_history = deque(maxlen=50)

    def should_scale_up(self, current_workers: int, stats: EngineStats) -> bool:
        """スケールアップ判定"""
        if current_workers >= self.max_workers:
            return False

        # CPU使用率チェック
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent

        # キュー滞留チェック
        queue_pressure = stats.queue_size / max(current_workers, 1)

        # スケールアップ条件
        conditions = [
            cpu_usage < self.cpu_threshold_high,
            memory_usage < self.memory_threshold_high,
            queue_pressure > 2.0,  # キューに滞留がある
            stats.success_rate > 0.8,  # 成功率が良好
        ]

        return all(conditions[:3]) and any(conditions[2:])  # リソースOK かつ 性能要求あり

    def should_scale_down(self, current_workers: int, stats: EngineStats) -> bool:
        """スケールダウン判定"""
        if current_workers <= self.min_workers:
            return False

        # システム負荷チェック
        cpu_usage = psutil.cpu_percent(interval=None)
        queue_pressure = stats.queue_size / max(current_workers, 1)

        # スケールダウン条件
        conditions = [
            cpu_usage < self.cpu_threshold_low,
            queue_pressure < 0.5,  # キューに余裕がある
            stats.throughput_tps < current_workers * 0.5,  # ワーカーが遊んでいる
        ]

        return all(conditions)

    def get_optimal_workers(self, current_workers: int, stats: EngineStats) -> int:
        """最適ワーカー数取得"""
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return current_workers

        if self.should_scale_up(current_workers, stats):
            self.last_adjustment = now
            return min(current_workers + 1, self.max_workers)
        elif self.should_scale_down(current_workers, stats):
            self.last_adjustment = now
            return max(current_workers - 1, self.min_workers)

        return current_workers


class ParallelBatchEngine:
    """
    並列バッチ処理エンジン

    大規模データ処理のための高性能並列システム
    動的ワーカー管理と適応的リソース調整を提供
    """

    def __init__(
        self,
        initial_workers: int = 4,
        max_workers: int = None,
        processing_mode: ProcessingMode = ProcessingMode.THREAD_BASED,
        enable_adaptive_scaling: bool = True,
        task_queue_size: int = 1000,
        result_queue_size: int = 1000,
        monitoring_interval: float = 5.0,
    ):
        self.processing_mode = processing_mode
        self.enable_adaptive_scaling = enable_adaptive_scaling
        self.task_queue_size = task_queue_size
        self.result_queue_size = result_queue_size
        self.monitoring_interval = monitoring_interval

        # タスクキューとワーカー管理
        self.task_queue = PriorityQueue(maxsize=task_queue_size)
        self.result_queue = Queue(maxsize=result_queue_size)
        self.active_tasks = {}  # task_id -> Future
        self.completed_tasks = {}  # task_id -> TaskResult

        # ワーカー管理
        self.thread_executor = None
        self.process_executor = None
        self.current_workers = 0
        self.worker_stats = {}  # worker_id -> WorkerStats

        # 適応的リソース管理
        self.resource_manager = AdaptiveResourceManager(
            initial_workers=initial_workers, max_workers=max_workers
        )

        # 統計とモニタリング
        self.stats = EngineStats()
        self.start_time = time.time()
        self.performance_history = deque(maxlen=1000)

        # 制御フラグ
        self.running = False
        self.monitor_thread = None
        self.result_processor_thread = None

        # ロック
        self.stats_lock = threading.Lock()
        self.workers_lock = threading.Lock()

        logger.info(
            f"並列バッチエンジン初期化: workers={initial_workers}, "
            f"mode={processing_mode.value}, adaptive={enable_adaptive_scaling}"
        )

    def start(self):
        """エンジン開始"""
        if self.running:
            return

        self.running = True
        self.start_time = time.time()

        # ワーカー初期化
        self._initialize_workers(self.resource_manager.initial_workers)

        # バックグラウンドスレッド開始
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.result_processor_thread = threading.Thread(
            target=self._result_processing_loop, daemon=True
        )

        self.monitor_thread.start()
        self.result_processor_thread.start()

        logger.info("並列バッチエンジン開始")

    def stop(self, timeout: float = 30.0):
        """エンジン停止"""
        if not self.running:
            return

        logger.info("並列バッチエンジン停止開始...")

        self.running = False

        # バックグラウンドスレッド終了待機
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        if self.result_processor_thread:
            self.result_processor_thread.join(timeout=5.0)

        # アクティブタスクの完了待機
        self._wait_for_active_tasks(timeout=timeout)

        # ワーカー終了
        self._shutdown_workers()

        logger.info("並列バッチエンジン停止完了")

    def submit_task(
        self,
        function: Callable,
        *args,
        task_id: str = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = 60.0,
        retry_count: int = 3,
        **kwargs,
    ) -> str:
        """
        タスク投入

        Args:
            function: 実行する関数
            *args: 関数の引数
            task_id: タスクID（自動生成可能）
            priority: 優先度
            timeout: タイムアウト
            retry_count: リトライ回数
            **kwargs: 関数のキーワード引数

        Returns:
            タスクID
        """
        if not self.running:
            raise RuntimeError("エンジンが開始されていません")

        if task_id is None:
            task_id = f"task_{int(time.time() * 1000)}_{self.stats.total_tasks_submitted}"

        task = BatchTask(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            retry_count=retry_count,
        )

        try:
            self.task_queue.put(task, timeout=1.0)

            with self.stats_lock:
                self.stats.total_tasks_submitted += 1
                self.stats.queue_size = self.task_queue.qsize()

            logger.debug(f"タスク投入: {task_id}, priority={priority.name}")
            return task_id

        except Exception as e:
            logger.error(f"タスク投入エラー {task_id}: {e}")
            raise

    def submit_batch_tasks(
        self,
        tasks: List[Dict[str, Any]],
        default_priority: TaskPriority = TaskPriority.NORMAL,
    ) -> List[str]:
        """
        バッチタスク投入

        Args:
            tasks: タスク定義のリスト
            default_priority: デフォルト優先度

        Returns:
            タスクIDのリスト
        """
        task_ids = []

        for task_def in tasks:
            task_id = self.submit_task(
                function=task_def["function"],
                args=task_def.get("args", ()),
                kwargs=task_def.get("kwargs", {}),
                task_id=task_def.get("task_id"),
                priority=task_def.get("priority", default_priority),
                timeout=task_def.get("timeout", 60.0),
                retry_count=task_def.get("retry_count", 3),
            )
            task_ids.append(task_id)

        logger.info(f"バッチタスク投入完了: {len(task_ids)} tasks")
        return task_ids

    def get_result(self, task_id: str, timeout: float = None) -> Optional[TaskResult]:
        """
        タスク結果取得

        Args:
            task_id: タスクID
            timeout: 待機タイムアウト

        Returns:
            タスク結果
        """
        # 完了済みタスクから検索
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]

        # アクティブタスクの場合は完了待機
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            try:
                result = future.result(timeout=timeout)
                return self.completed_tasks.get(task_id)
            except Exception as e:
                logger.error(f"タスク結果取得エラー {task_id}: {e}")
                return None

        # タスクが存在しない
        return None

    def get_results_batch(
        self, task_ids: List[str], timeout: float = None
    ) -> Dict[str, TaskResult]:
        """
        バッチタスク結果取得

        Args:
            task_ids: タスクIDのリスト
            timeout: 待機タイムアウト

        Returns:
            タスクID -> 結果のDict
        """
        results = {}

        for task_id in task_ids:
            result = self.get_result(task_id, timeout=timeout)
            if result:
                results[task_id] = result

        return results

    def cancel_task(self, task_id: str) -> bool:
        """
        タスクキャンセル

        Args:
            task_id: タスクID

        Returns:
            キャンセル成功可否
        """
        # アクティブタスクのキャンセル
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            success = future.cancel()
            if success:
                # 結果記録
                result = TaskResult(
                    task_id=task_id,
                    success=False,
                    error="Task cancelled",
                    status=TaskStatus.CANCELLED,
                )
                self.completed_tasks[task_id] = result
                del self.active_tasks[task_id]

            return success

        return False

    def _initialize_workers(self, worker_count: int):
        """ワーカー初期化"""
        with self.workers_lock:
            if self.processing_mode == ProcessingMode.THREAD_BASED:
                self.thread_executor = ThreadPoolExecutor(
                    max_workers=worker_count, thread_name_prefix="BatchEngine"
                )
                executor = self.thread_executor

            elif self.processing_mode == ProcessingMode.PROCESS_BASED:
                self.process_executor = ProcessPoolExecutor(max_workers=worker_count)
                executor = self.process_executor

            elif self.processing_mode == ProcessingMode.HYBRID:
                # 混合モード（CPUとI/O負荷に応じて動的選択）
                thread_workers = max(1, worker_count // 2)
                process_workers = max(1, worker_count - thread_workers)

                self.thread_executor = ThreadPoolExecutor(
                    max_workers=thread_workers, thread_name_prefix="BatchEngine-Thread"
                )
                self.process_executor = ProcessPoolExecutor(max_workers=process_workers)
                executor = self.thread_executor  # デフォルト

            self.current_workers = worker_count

            # ワーカー統計初期化
            for i in range(worker_count):
                worker_id = f"worker_{self.processing_mode.value}_{i}"
                self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)

            with self.stats_lock:
                self.stats.active_workers = worker_count

        # ワーカータスク開始
        for _ in range(worker_count):
            self._start_worker_task()

    def _start_worker_task(self):
        """ワーカータスク開始"""
        if self.thread_executor:
            future = self.thread_executor.submit(self._worker_loop)
            # 必要に応じてFutureを管理

    def _worker_loop(self):
        """ワーカーループ"""
        worker_id = f"worker_{threading.current_thread().name}_{id(threading.current_thread())}"

        while self.running:
            try:
                # タスク取得
                task = self.task_queue.get(timeout=1.0)

                # タスク実行
                result = self._execute_task(task, worker_id)

                # 結果をキューに投入
                self.result_queue.put((task.task_id, result))

                # ワーカー統計更新
                if worker_id in self.worker_stats:
                    stats = self.worker_stats[worker_id]
                    stats.last_activity = time.time()

                    if result.success:
                        stats.tasks_completed += 1
                    else:
                        stats.tasks_failed += 1

                    stats.total_execution_time += result.execution_time
                    if stats.tasks_completed > 0:
                        stats.average_execution_time = (
                            stats.total_execution_time / stats.tasks_completed
                        )

                self.task_queue.task_done()

            except Exception as e:
                if self.running:  # 正常終了時のエラーは無視
                    logger.error(f"ワーカーループエラー {worker_id}: {e}")
                break

    def _execute_task(self, task: BatchTask, worker_id: str) -> TaskResult:
        """タスク実行"""
        start_time = time.time()
        retry_count = 0

        while retry_count <= task.retry_count:
            try:
                # メモリ使用量監視開始
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024

                # タスク実行
                if asyncio.iscoroutinefunction(task.function):
                    # 非同期関数の場合
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            asyncio.wait_for(
                                task.function(*task.args, **task.kwargs),
                                timeout=task.timeout,
                            )
                        )
                    finally:
                        loop.close()
                else:
                    # 通常の関数
                    result = task.function(*task.args, **task.kwargs)

                # メモリ使用量計算
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_peak = memory_after - memory_before

                execution_time = time.time() - start_time

                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    memory_peak_mb=memory_peak,
                    retry_count=retry_count,
                    worker_id=worker_id,
                    status=TaskStatus.COMPLETED,
                )

            except Exception as e:
                retry_count += 1
                error_msg = str(e)

                if retry_count <= task.retry_count:
                    logger.warning(
                        f"タスク実行失敗 {task.task_id} (retry {retry_count}/{task.retry_count}): {error_msg}"
                    )
                    time.sleep(task.retry_delay * retry_count)  # 指数バックオフ
                else:
                    logger.error(f"タスク実行最終失敗 {task.task_id}: {error_msg}")

                    execution_time = time.time() - start_time

                    return TaskResult(
                        task_id=task.task_id,
                        success=False,
                        error=error_msg,
                        execution_time=execution_time,
                        retry_count=retry_count - 1,
                        worker_id=worker_id,
                        status=TaskStatus.FAILED,
                    )

    def _result_processing_loop(self):
        """結果処理ループ"""
        while self.running:
            try:
                task_id, result = self.result_queue.get(timeout=1.0)

                # 結果保存
                self.completed_tasks[task_id] = result

                # アクティブタスクから削除
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]

                # 統計更新
                with self.stats_lock:
                    self.stats.total_tasks_completed += 1

                    if result.success:
                        pass  # 成功カウントは別途管理
                    else:
                        self.stats.total_tasks_failed += 1

                    # 成功率計算
                    total_finished = (
                        self.stats.total_tasks_completed + self.stats.total_tasks_failed
                    )
                    if total_finished > 0:
                        self.stats.success_rate = self.stats.total_tasks_completed / total_finished

                    # 平均実行時間更新
                    if result.execution_time > 0:
                        current_total_time = self.stats.average_execution_time * (
                            total_finished - 1
                        )
                        self.stats.average_execution_time = (
                            current_total_time + result.execution_time
                        ) / total_finished

                self.result_queue.task_done()

            except Exception as e:
                if self.running:
                    logger.error(f"結果処理ループエラー: {e}")

    def _monitoring_loop(self):
        """モニタリングループ"""
        while self.running:
            try:
                self._update_system_stats()

                # 適応的スケーリング
                if self.enable_adaptive_scaling:
                    self._adaptive_scaling()

                # パフォーマンス履歴記録
                self._record_performance_snapshot()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                if self.running:
                    logger.error(f"モニタリングループエラー: {e}")

    def _update_system_stats(self):
        """システム統計更新"""
        with self.stats_lock:
            # アップタイム
            self.stats.uptime_seconds = time.time() - self.start_time

            # キューサイズ
            self.stats.queue_size = self.task_queue.qsize()

            # スループット計算
            if self.stats.uptime_seconds > 0:
                self.stats.throughput_tps = (
                    self.stats.total_tasks_completed / self.stats.uptime_seconds
                )

            # システムリソース
            self.stats.memory_usage_mb = psutil.virtual_memory().used / 1024 / 1024
            self.stats.cpu_usage_percent = psutil.cpu_percent(interval=None)

    def _adaptive_scaling(self):
        """適応的スケーリング"""
        with self.workers_lock:
            optimal_workers = self.resource_manager.get_optimal_workers(
                self.current_workers, self.stats
            )

            if optimal_workers != self.current_workers:
                logger.info(f"ワーカー数調整: {self.current_workers} -> {optimal_workers}")
                self._adjust_workers(optimal_workers)

    def _adjust_workers(self, target_workers: int):
        """ワーカー数調整"""
        if target_workers > self.current_workers:
            # スケールアップ
            additional_workers = target_workers - self.current_workers
            for _ in range(additional_workers):
                self._start_worker_task()
        elif target_workers < self.current_workers:
            # スケールダウンは自然終了を待つ
            # 必要に応じて強制終了機能を実装
            pass

        self.current_workers = target_workers

        with self.stats_lock:
            self.stats.active_workers = target_workers

    def _record_performance_snapshot(self):
        """パフォーマンススナップショット記録"""
        snapshot = {
            "timestamp": time.time(),
            "throughput_tps": self.stats.throughput_tps,
            "success_rate": self.stats.success_rate,
            "queue_size": self.stats.queue_size,
            "active_workers": self.stats.active_workers,
            "cpu_usage": self.stats.cpu_usage_percent,
            "memory_usage": self.stats.memory_usage_mb,
        }

        self.performance_history.append(snapshot)

    def _wait_for_active_tasks(self, timeout: float):
        """アクティブタスクの完了待機"""
        start_time = time.time()

        while self.active_tasks and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self.active_tasks:
            logger.warning(f"タイムアウト: {len(self.active_tasks)} tasks still active")

    def _shutdown_workers(self):
        """ワーカー終了"""
        with self.workers_lock:
            if self.thread_executor:
                self.thread_executor.shutdown(wait=True)
                self.thread_executor = None

            if self.process_executor:
                self.process_executor.shutdown(wait=True)
                self.process_executor = None

    def get_stats(self) -> EngineStats:
        """統計取得"""
        with self.stats_lock:
            return self.stats

    def get_performance_history(self, last_n: int = None) -> List[Dict[str, Any]]:
        """パフォーマンス履歴取得"""
        if last_n:
            return list(self.performance_history)[-last_n:]
        return list(self.performance_history)

    def get_worker_stats(self) -> Dict[str, WorkerStats]:
        """ワーカー統計取得"""
        return self.worker_stats.copy()


# 便利関数
def create_batch_engine(
    workers: int = 4,
    processing_mode: ProcessingMode = ProcessingMode.THREAD_BASED,
    enable_adaptive_scaling: bool = True,
    **kwargs,
) -> ParallelBatchEngine:
    """バッチエンジン作成"""
    return ParallelBatchEngine(
        initial_workers=workers,
        processing_mode=processing_mode,
        enable_adaptive_scaling=enable_adaptive_scaling,
        **kwargs,
    )


if __name__ == "__main__":
    # テスト実行
    print("=== 並列バッチ処理エンジン テスト ===")

    # テスト関数定義
    def cpu_intensive_task(n: int, duration: float = 0.1) -> int:
        """CPU集約的タスク"""
        time.sleep(duration)
        return sum(i * i for i in range(n))

    def io_intensive_task(delay: float = 0.2) -> str:
        """I/O集約的タスク"""
        time.sleep(delay)
        return f"Task completed at {time.time()}"

    def failing_task(should_fail: bool = True) -> str:
        """失敗するタスク"""
        if should_fail:
            raise ValueError("Intentional failure for testing")
        return "Task succeeded"

    # エンジン作成・開始
    engine = create_batch_engine(
        workers=4,
        processing_mode=ProcessingMode.THREAD_BASED,
        enable_adaptive_scaling=True,
        monitoring_interval=2.0,
    )

    engine.start()

    try:
        # テストタスク投入
        print("テストタスク投入中...")

        task_ids = []

        # CPU集約的タスク
        for i in range(5):
            task_id = engine.submit_task(
                cpu_intensive_task,
                1000 + i * 100,
                duration=0.1,
                priority=TaskPriority.HIGH,
                task_id=f"cpu_task_{i}",
            )
            task_ids.append(task_id)

        # I/O集約的タスク
        for i in range(10):
            task_id = engine.submit_task(
                io_intensive_task,
                delay=0.05,
                priority=TaskPriority.NORMAL,
                task_id=f"io_task_{i}",
            )
            task_ids.append(task_id)

        # 失敗するタスク（リトライテスト）
        for i in range(3):
            task_id = engine.submit_task(
                failing_task,
                should_fail=(i < 2),  # 最初の2つは失敗
                priority=TaskPriority.LOW,
                retry_count=2,
                task_id=f"fail_task_{i}",
            )
            task_ids.append(task_id)

        print(f"投入完了: {len(task_ids)} tasks")

        # 結果待機・取得
        print("結果待機中...")
        time.sleep(3.0)

        # バッチ結果取得
        results = engine.get_results_batch(task_ids, timeout=10.0)

        # 結果分析
        successful_results = [r for r in results.values() if r.success]
        failed_results = [r for r in results.values() if not r.success]

        print("\n結果:")
        print(f"  完了: {len(results)}/{len(task_ids)}")
        print(f"  成功: {len(successful_results)}")
        print(f"  失敗: {len(failed_results)}")

        # 成功したタスクの詳細
        for result in successful_results[:5]:  # 最初の5つのみ表示
            print(
                f"  ✓ {result.task_id}: {result.execution_time:.3f}s, "
                f"memory={result.memory_peak_mb:.1f}MB, worker={result.worker_id}"
            )

        # 失敗したタスクの詳細
        for result in failed_results:
            print(f"  ✗ {result.task_id}: {result.error}, retries={result.retry_count}")

        # エンジン統計
        stats = engine.get_stats()
        print("\nエンジン統計:")
        print(f"  総投入: {stats.total_tasks_submitted}")
        print(f"  総完了: {stats.total_tasks_completed}")
        print(f"  総失敗: {stats.total_tasks_failed}")
        print(f"  成功率: {stats.success_rate:.1%}")
        print(f"  平均実行時間: {stats.average_execution_time:.3f}s")
        print(f"  スループット: {stats.throughput_tps:.2f} TPS")
        print(f"  アクティブワーカー: {stats.active_workers}")
        print(f"  CPU使用率: {stats.cpu_usage_percent:.1f}%")
        print(f"  メモリ使用量: {stats.memory_usage_mb:.1f} MB")
        print(f"  稼働時間: {stats.uptime_seconds:.1f}s")

        # ワーカー統計
        worker_stats = engine.get_worker_stats()
        print("\nワーカー統計:")
        for worker_id, stats in list(worker_stats.items())[:3]:  # 最初の3つのみ
            print(
                f"  {worker_id}: completed={stats.tasks_completed}, "
                f"failed={stats.tasks_failed}, avg_time={stats.average_execution_time:.3f}s"
            )

        # パフォーマンス履歴
        history = engine.get_performance_history(last_n=3)
        print("\n最近のパフォーマンス:")
        for snapshot in history:
            print(
                f"  {snapshot['timestamp']:.0f}: TPS={snapshot['throughput_tps']:.2f}, "
                f"workers={snapshot['active_workers']}, queue={snapshot['queue_size']}"
            )

    finally:
        # エンジン停止
        print("\nエンジン停止中...")
        engine.stop(timeout=10.0)

    print("\n=== テスト完了 ===")
