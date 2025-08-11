#!/usr/bin/env python3
"""
並列実行マネージャー
Issue #383: CPU/I/Oバウンドタスクの適切な分離

PythonのGIL制約を回避し、タスクの性質に応じて最適なExecutorを選択する
統一された並列処理抽象化レイヤー
"""

import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from multiprocessing import cpu_count
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .logging_config import get_context_logger
from .performance_monitor import PerformanceMonitor

logger = get_context_logger(__name__)


class TaskType(Enum):
    """タスク分類"""

    IO_BOUND = auto()  # I/Oバウンド（ネットワーク、ファイル）
    CPU_BOUND = auto()  # CPUバウンド（計算集約）
    MIXED = auto()  # 混在（適応的選択）
    ASYNC_IO = auto()  # 非同期I/O


class ExecutorType(Enum):
    """実行エンジン種別"""

    THREAD_POOL = auto()  # ThreadPoolExecutor
    PROCESS_POOL = auto()  # ProcessPoolExecutor
    ASYNC_IO = auto()  # asyncio


@dataclass
class TaskProfile:
    """タスクプロファイル"""

    task_type: TaskType
    expected_duration_ms: float
    memory_intensive: bool = False
    io_operations: int = 0
    cpu_operations: int = 0
    data_size_mb: float = 0.0
    serializable: bool = True  # プロセス間通信可能か


@dataclass
class ExecutionResult:
    """実行結果"""

    task_id: str
    result: Any
    execution_time_ms: float
    executor_type: ExecutorType
    success: bool = True
    error: Optional[Exception] = None
    memory_peak_mb: float = 0.0
    cpu_utilization: float = 0.0


class TaskClassifier:
    """タスク自動分類器"""

    def __init__(self):
        self._classification_cache = {}
        self._performance_history = {}

    def classify_task(
        self, func: Callable, args: tuple, kwargs: dict, hint: Optional[TaskType] = None
    ) -> TaskProfile:
        """タスクを自動分類"""

        # 明示的ヒントがある場合
        if hint:
            return TaskProfile(
                task_type=hint,
                expected_duration_ms=self._estimate_duration(func, args, kwargs),
                memory_intensive=self._is_memory_intensive(func, args, kwargs),
                serializable=self._is_serializable(func, args, kwargs),
            )

        # 関数名・モジュールからの推測
        func_name = func.__name__
        module_name = func.__module__ or ""

        # I/Oバウンドの特徴
        io_indicators = [
            "fetch",
            "download",
            "request",
            "http",
            "api",
            "file",
            "read",
            "write",
            "database",
            "query",
            "connect",
            "socket",
            "network",
            "yfinance",
        ]

        # CPUバウンドの特徴
        cpu_indicators = [
            "calculate",
            "compute",
            "optimize",
            "train",
            "predict",
            "feature",
            "transform",
            "analysis",
            "backtest",
            "simulate",
            "numpy",
            "scipy",
        ]

        # 関数名/モジュール名から判定
        func_text = f"{func_name} {module_name}".lower()

        io_score = sum(1 for indicator in io_indicators if indicator in func_text)
        cpu_score = sum(1 for indicator in cpu_indicators if indicator in func_text)

        # データサイズから判定
        data_size = self._estimate_data_size(args, kwargs)

        if io_score > cpu_score:
            task_type = TaskType.IO_BOUND
        elif cpu_score > io_score:
            task_type = TaskType.CPU_BOUND
        elif data_size > 50.0:  # 50MB以上は重い処理と判定
            task_type = TaskType.CPU_BOUND
        else:
            task_type = TaskType.MIXED

        return TaskProfile(
            task_type=task_type,
            expected_duration_ms=self._estimate_duration(func, args, kwargs),
            memory_intensive=data_size > 100.0,  # 100MB以上
            data_size_mb=data_size,
            serializable=self._is_serializable(func, args, kwargs),
        )

    def _estimate_duration(self, func: Callable, args: tuple, kwargs: dict) -> float:
        """実行時間推定"""
        # 履歴がある場合
        func_key = f"{func.__module__}.{func.__name__}"
        if func_key in self._performance_history:
            return np.mean(self._performance_history[func_key])

        # データサイズから推定
        data_size = self._estimate_data_size(args, kwargs)
        if data_size > 100:
            return 5000.0  # 5秒
        elif data_size > 10:
            return 1000.0  # 1秒
        else:
            return 100.0  # 100ms

    def _estimate_data_size(self, args: tuple, kwargs: dict) -> float:
        """データサイズ推定（MB）"""
        total_size = 0.0

        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, pd.DataFrame):
                total_size += arg.memory_usage(deep=True).sum() / 1024 / 1024
            elif isinstance(arg, np.ndarray):
                total_size += arg.nbytes / 1024 / 1024
            elif isinstance(arg, (list, tuple)) and len(arg) > 1000:
                total_size += len(arg) * 8 / 1024 / 1024  # 概算

        return total_size

    def _is_memory_intensive(self, func: Callable, args: tuple, kwargs: dict) -> bool:
        """メモリ集約的かどうか"""
        return self._estimate_data_size(args, kwargs) > 50.0

    def _is_serializable(self, func: Callable, args: tuple, kwargs: dict) -> bool:
        """シリアライゼーション可能かどうか"""
        try:
            # 基本的なチェック
            import pickle

            pickle.dumps(func)
            pickle.dumps(args)
            pickle.dumps(kwargs)
            return True
        except (TypeError, AttributeError, pickle.PicklingError):
            return False

    def update_performance_history(self, func: Callable, execution_time_ms: float):
        """性能履歴更新"""
        func_key = f"{func.__module__}.{func.__name__}"
        if func_key not in self._performance_history:
            self._performance_history[func_key] = []

        self._performance_history[func_key].append(execution_time_ms)

        # 履歴は最新100件のみ保持
        if len(self._performance_history[func_key]) > 100:
            self._performance_history[func_key] = self._performance_history[func_key][
                -100:
            ]


class ParallelExecutorManager:
    """並列実行管理システム"""

    def __init__(
        self,
        max_thread_workers: Optional[int] = None,
        max_process_workers: Optional[int] = None,
        enable_adaptive_sizing: bool = True,
        performance_monitoring: bool = True,
    ):
        """
        Args:
            max_thread_workers: スレッドプール最大ワーカー数
            max_process_workers: プロセスプール最大ワーカー数
            enable_adaptive_sizing: 適応的サイジング
            performance_monitoring: 性能監視
        """

        # デフォルト設定
        self.max_thread_workers = max_thread_workers or min(32, (cpu_count() or 1) + 4)
        self.max_process_workers = max_process_workers or cpu_count() or 1
        self.enable_adaptive_sizing = enable_adaptive_sizing

        # コンポーネント
        self.classifier = TaskClassifier()
        self.performance_monitor = (
            PerformanceMonitor() if performance_monitoring else None
        )

        # Executor プール
        self._thread_executor: Optional[ThreadPoolExecutor] = None
        self._process_executor: Optional[ProcessPoolExecutor] = None

        # 統計情報
        self._execution_stats = {
            ExecutorType.THREAD_POOL: {"count": 0, "total_time": 0.0, "errors": 0},
            ExecutorType.PROCESS_POOL: {"count": 0, "total_time": 0.0, "errors": 0},
        }

        self._active_tasks = {}
        self._lock = threading.RLock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @property
    def thread_executor(self) -> ThreadPoolExecutor:
        """スレッドプールExecutor（遅延初期化）"""
        if self._thread_executor is None:
            self._thread_executor = ThreadPoolExecutor(
                max_workers=self.max_thread_workers,
                thread_name_prefix="ParallelExec-Thread",
            )
        return self._thread_executor

    @property
    def process_executor(self) -> ProcessPoolExecutor:
        """プロセスプールExecutor（遅延初期化）"""
        if self._process_executor is None:
            self._process_executor = ProcessPoolExecutor(
                max_workers=self.max_process_workers
            )
        return self._process_executor

    def execute_task(
        self,
        func: Callable,
        *args,
        task_type_hint: Optional[TaskType] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        タスクを並列実行

        Args:
            func: 実行する関数
            *args: 関数の引数
            task_type_hint: タスク種別のヒント
            timeout_seconds: タイムアウト（秒）
            **kwargs: 関数のキーワード引数

        Returns:
            実行結果
        """

        task_id = f"{func.__name__}_{int(time.time() * 1000)}"
        start_time = time.perf_counter()

        try:
            # タスク分類
            profile = self.classifier.classify_task(func, args, kwargs, task_type_hint)

            # 適切なExecutorを選択
            executor_type = self._select_executor(profile)

            # 実行
            if executor_type == ExecutorType.THREAD_POOL:
                result = self._execute_with_thread_pool(
                    func, args, kwargs, timeout_seconds
                )
            elif executor_type == ExecutorType.PROCESS_POOL:
                result = self._execute_with_process_pool(
                    func, args, kwargs, timeout_seconds
                )
            else:
                raise ValueError(f"Unsupported executor type: {executor_type}")

            execution_time = (time.perf_counter() - start_time) * 1000

            # 性能履歴更新
            self.classifier.update_performance_history(func, execution_time)

            # 統計更新
            self._update_stats(executor_type, execution_time, success=True)

            return ExecutionResult(
                task_id=task_id,
                result=result,
                execution_time_ms=execution_time,
                executor_type=executor_type,
                success=True,
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Task execution failed: {task_id}, error: {e}")

            # エラー統計更新
            executor_type = getattr(
                self, "_last_executor_type", ExecutorType.THREAD_POOL
            )
            self._update_stats(executor_type, execution_time, success=False)

            return ExecutionResult(
                task_id=task_id,
                result=None,
                execution_time_ms=execution_time,
                executor_type=executor_type,
                success=False,
                error=e,
            )

    def execute_batch(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        max_concurrent: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """
        バッチタスク実行

        Args:
            tasks: (func, args, kwargs) のタスクリスト
            max_concurrent: 最大同時実行数
            timeout_seconds: 全体タイムアウト

        Returns:
            実行結果リスト
        """
        if not tasks:
            return []

        results = []
        futures = {}

        # タスクを分類してExecutorに振り分け
        thread_tasks = []
        process_tasks = []

        for i, (func, args, kwargs) in enumerate(tasks):
            profile = self.classifier.classify_task(func, args, kwargs)
            executor_type = self._select_executor(profile)

            task_info = (i, func, args, kwargs)
            if executor_type == ExecutorType.THREAD_POOL:
                thread_tasks.append(task_info)
            else:
                process_tasks.append(task_info)

        # ThreadPool実行
        if thread_tasks:
            for i, func, args, kwargs in thread_tasks:
                future = self.thread_executor.submit(func, *args, **kwargs)
                futures[future] = (i, func, ExecutorType.THREAD_POOL)

        # ProcessPool実行
        if process_tasks:
            for i, func, args, kwargs in process_tasks:
                if self.classifier._is_serializable(func, args, kwargs):
                    future = self.process_executor.submit(func, *args, **kwargs)
                    futures[future] = (i, func, ExecutorType.PROCESS_POOL)
                else:
                    # シリアライゼーション不可の場合はThreadPoolにフォールバック
                    future = self.thread_executor.submit(func, *args, **kwargs)
                    futures[future] = (i, func, ExecutorType.THREAD_POOL)

        # 結果収集
        results = [None] * len(tasks)
        start_time = time.perf_counter()

        for future in as_completed(futures, timeout=timeout_seconds):
            task_index, func, executor_type = futures[future]
            task_id = f"batch_{task_index}_{func.__name__}"

            try:
                result = future.result()
                execution_time = (time.perf_counter() - start_time) * 1000

                results[task_index] = ExecutionResult(
                    task_id=task_id,
                    result=result,
                    execution_time_ms=execution_time,
                    executor_type=executor_type,
                    success=True,
                )

                self._update_stats(executor_type, execution_time, success=True)

            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.error(f"Batch task failed: {task_id}, error: {e}")

                results[task_index] = ExecutionResult(
                    task_id=task_id,
                    result=None,
                    execution_time_ms=execution_time,
                    executor_type=executor_type,
                    success=False,
                    error=e,
                )

                self._update_stats(executor_type, execution_time, success=False)

        return results

    def _select_executor(self, profile: TaskProfile) -> ExecutorType:
        """適切なExecutorを選択"""

        # シリアライゼーション不可の場合はThreadPool強制
        if not profile.serializable:
            return ExecutorType.THREAD_POOL

        # タスク種別による判定
        if profile.task_type == TaskType.IO_BOUND:
            return ExecutorType.THREAD_POOL
        elif profile.task_type == TaskType.CPU_BOUND:
            return ExecutorType.PROCESS_POOL
        elif profile.task_type == TaskType.MIXED:
            # 混在の場合は適応的選択
            if profile.memory_intensive or profile.expected_duration_ms > 1000:
                return ExecutorType.PROCESS_POOL
            else:
                return ExecutorType.THREAD_POOL

        # デフォルト
        return ExecutorType.THREAD_POOL

    def _execute_with_thread_pool(
        self, func: Callable, args: tuple, kwargs: dict, timeout: Optional[float]
    ) -> Any:
        """ThreadPoolで実行"""
        self._last_executor_type = ExecutorType.THREAD_POOL
        future = self.thread_executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)

    def _execute_with_process_pool(
        self, func: Callable, args: tuple, kwargs: dict, timeout: Optional[float]
    ) -> Any:
        """ProcessPoolで実行"""
        self._last_executor_type = ExecutorType.PROCESS_POOL
        future = self.process_executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)

    def _update_stats(
        self, executor_type: ExecutorType, execution_time: float, success: bool
    ):
        """統計情報更新"""
        with self._lock:
            stats = self._execution_stats[executor_type]
            stats["count"] += 1
            stats["total_time"] += execution_time
            if not success:
                stats["errors"] += 1

    def get_performance_stats(self) -> Dict[str, Any]:
        """性能統計取得"""
        with self._lock:
            stats = {}
            for executor_type, data in self._execution_stats.items():
                if data["count"] > 0:
                    stats[executor_type.name] = {
                        "total_executions": data["count"],
                        "total_time_ms": data["total_time"],
                        "average_time_ms": data["total_time"] / data["count"],
                        "error_rate": data["errors"] / data["count"],
                        "success_rate": 1.0 - (data["errors"] / data["count"]),
                    }
            return stats

    def shutdown(self, wait: bool = True):
        """Executor群をシャットダウン"""
        if self._thread_executor:
            self._thread_executor.shutdown(wait=wait)
            self._thread_executor = None

        if self._process_executor:
            self._process_executor.shutdown(wait=wait)
            self._process_executor = None

        logger.info("ParallelExecutorManager shutdown completed")


# デコレーターAPI


def parallel_task(
    task_type: Optional[TaskType] = None,
    timeout_seconds: Optional[float] = None,
    enable_caching: bool = False,
):
    """
    並列実行デコレーター

    Args:
        task_type: タスクタイプヒント
        timeout_seconds: タイムアウト
        enable_caching: 結果キャッシュ有効化
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # グローバルエグゼキューターマネージャーを取得/作成
            manager = getattr(wrapper, "_executor_manager", None)
            if manager is None:
                manager = ParallelExecutorManager()
                wrapper._executor_manager = manager

            result = manager.execute_task(
                func,
                *args,
                task_type_hint=task_type,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )

            if result.success:
                return result.result
            else:
                raise result.error or RuntimeError("Task execution failed")

        # クリーンアップ関数を追加
        def cleanup():
            manager = getattr(wrapper, "_executor_manager", None)
            if manager:
                manager.shutdown()

        wrapper.cleanup = cleanup
        return wrapper

    return decorator


# グローバル singleton インスタンス
_global_manager: Optional[ParallelExecutorManager] = None
_global_manager_lock = threading.Lock()


def get_global_executor_manager() -> ParallelExecutorManager:
    """グローバルExecutorManager取得"""
    global _global_manager

    if _global_manager is None:
        with _global_manager_lock:
            if _global_manager is None:
                _global_manager = ParallelExecutorManager()

    return _global_manager


def shutdown_global_executor_manager():
    """グローバルExecutorManagerシャットダウン"""
    global _global_manager

    with _global_manager_lock:
        if _global_manager:
            _global_manager.shutdown()
            _global_manager = None


# 便利関数
def execute_parallel(
    func: Callable, *args, task_type: Optional[TaskType] = None, **kwargs
) -> Any:
    """関数を並列実行"""
    manager = get_global_executor_manager()
    result = manager.execute_task(func, *args, task_type_hint=task_type, **kwargs)

    if result.success:
        return result.result
    else:
        raise result.error or RuntimeError("Parallel execution failed")


if __name__ == "__main__":
    # テスト実行
    def cpu_intensive_task(n: int) -> float:
        """CPU集約的タスク"""
        return sum(i * i for i in range(n))

    def io_intensive_task(duration: float) -> str:
        """I/O集約的タスク（スリープで模擬）"""
        time.sleep(duration)
        return f"Slept for {duration}s"

    # テスト
    with ParallelExecutorManager() as manager:
        print("=== Issue #383 並列実行マネージャーテスト ===")

        # CPU集約的タスク
        print("\n1. CPU集約的タスクテスト")
        result = manager.execute_task(cpu_intensive_task, 100000)
        print(
            f"結果: {result.result}, 実行時間: {result.execution_time_ms:.2f}ms, Executor: {result.executor_type.name}"
        )

        # I/O集約的タスク
        print("\n2. I/O集約的タスクテスト")
        result = manager.execute_task(io_intensive_task, 0.1)
        print(
            f"結果: {result.result}, 実行時間: {result.execution_time_ms:.2f}ms, Executor: {result.executor_type.name}"
        )

        # 性能統計
        print("\n3. 性能統計")
        stats = manager.get_performance_stats()
        for executor_name, executor_stats in stats.items():
            print(
                f"{executor_name}: 実行数={executor_stats['total_executions']}, "
                f"平均時間={executor_stats['average_time_ms']:.2f}ms, "
                f"成功率={executor_stats['success_rate']:.1%}"
            )

        print("\n=== Issue #383 並列実行マネージャーテスト完了 ===")
