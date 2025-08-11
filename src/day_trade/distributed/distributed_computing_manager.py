#!/usr/bin/env python3
"""
統一分散コンピューティングマネージャー
Issue #384: 並列処理のさらなる強化 - 統合管理層

DaskとRayの統一インターフェース、動的バックエンド選択、
最適化されたタスク分散システム
"""

import asyncio
import tempfile
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

import pandas as pd

# 分散処理フレームワーク（オプショナル）
try:
    import dask
    import ray

    DASK_AVAILABLE = True
    RAY_AVAILABLE = True
except ImportError as e:
    missing_lib = (
        str(e).split("'")[1] if "'" in str(e) else "distributed computing library"
    )
    warnings.warn(
        f"{missing_lib}が利用できません。該当機能は制限されます。",
        UserWarning,
        stacklevel=2,
    )
    DASK_AVAILABLE = False
    RAY_AVAILABLE = False

# プロジェクトモジュール
try:
    from ..utils.logging_config import get_context_logger, log_performance_metric
    from ..utils.parallel_executor_manager import ParallelExecutorManager, TaskType
    from ..utils.performance_monitor import PerformanceMonitor
    from .dask_data_processor import DaskDataProcessor, create_dask_data_processor
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    def log_performance_metric(*args, **kwargs):
        pass

    # モッククラス
    class DaskDataProcessor:
        def __init__(self, **kwargs):
            pass

        async def process_multiple_symbols_parallel(self, *args, **kwargs):
            return pd.DataFrame()

        def cleanup(self):
            pass

    def create_dask_data_processor(**kwargs):
        return DaskDataProcessor()

    class PerformanceMonitor:
        def __init__(self):
            pass

        def start_monitoring(self):
            pass

        def stop_monitoring(self):
            return {}

    class ParallelExecutorManager:
        def __init__(self, **kwargs):
            pass

        async def execute_async(self, *args, **kwargs):
            return None

    class TaskType:
        CPU_BOUND = "cpu_bound"
        IO_BOUND = "io_bound"


logger = get_context_logger(__name__)

T = TypeVar("T")


class ComputingBackend(Enum):
    """コンピューティングバックエンド"""

    AUTO = "auto"  # 自動選択
    SEQUENTIAL = "sequential"  # シーケンシャル処理
    THREADING = "threading"  # マルチスレッド
    MULTIPROCESSING = "multiprocessing"  # マルチプロセス
    DASK = "dask"  # Dask分散処理
    RAY = "ray"  # Ray分散処理
    HYBRID = "hybrid"  # ハイブリッド（タスクに応じて選択）


class TaskDistributionStrategy(Enum):
    """タスク分散戦略"""

    ROUND_ROBIN = "round_robin"  # ラウンドロビン
    LOAD_BALANCED = "load_balanced"  # 負荷分散
    AFFINITY = "affinity"  # アフィニティベース
    DYNAMIC = "dynamic"  # 動的最適化


@dataclass
class DistributedTask(Generic[T]):
    """分散タスク定義"""

    task_id: str
    task_function: Callable[..., T]
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    task_type: TaskType = TaskType.CPU_BOUND
    priority: int = 1
    timeout_seconds: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class DistributedResult(Generic[T]):
    """分散処理結果"""

    task_id: str
    result: Optional[T] = None
    success: bool = False
    error: Optional[Exception] = None
    execution_time_seconds: float = 0.0
    backend_used: ComputingBackend = ComputingBackend.SEQUENTIAL
    worker_id: Optional[str] = None
    memory_peak_mb: float = 0.0
    completed_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedBackendManager(ABC):
    """分散バックエンド基底クラス"""

    @abstractmethod
    async def initialize(self, **config) -> bool:
        """バックエンド初期化"""
        pass

    @abstractmethod
    async def execute_task(self, task: DistributedTask) -> DistributedResult:
        """タスク実行"""
        pass

    @abstractmethod
    async def execute_batch(
        self, tasks: List[DistributedTask]
    ) -> List[DistributedResult]:
        """バッチタスク実行"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        pass

    @abstractmethod
    async def cleanup(self):
        """リソース解放"""
        pass


class DaskBackendManager(DistributedBackendManager):
    """Daskバックエンドマネージャー"""

    def __init__(self):
        self.dask_processor = None
        self.initialized = False

    async def initialize(self, **config) -> bool:
        """Dask初期化"""
        if not DASK_AVAILABLE:
            logger.warning("Daskが利用できません")
            return False

        try:
            self.dask_processor = create_dask_data_processor(
                enable_distributed=config.get("enable_distributed", True),
                n_workers=config.get("n_workers"),
                memory_limit=config.get("memory_limit", "2GB"),
                temp_dir=config.get("temp_dir"),
            )

            self.initialized = True
            logger.info("Daskバックエンド初期化完了")
            return True

        except Exception as e:
            logger.error(f"Daskバックエンド初期化失敗: {e}")
            return False

    async def execute_task(self, task: DistributedTask) -> DistributedResult:
        """Daskタスク実行"""
        if not self.initialized:
            return DistributedResult(
                task_id=task.task_id,
                success=False,
                error=RuntimeError("Daskバックエンドが初期化されていません"),
                backend_used=ComputingBackend.DASK,
            )

        start_time = time.time()

        try:
            # Dask delayed を使用してタスク実行
            if DASK_AVAILABLE:
                from dask.delayed import delayed

                delayed_task = delayed(task.task_function)(*task.args, **task.kwargs)
                result = delayed_task.compute()
            else:
                # フォールバック
                result = task.task_function(*task.args, **task.kwargs)

            execution_time = time.time() - start_time

            return DistributedResult(
                task_id=task.task_id,
                result=result,
                success=True,
                execution_time_seconds=execution_time,
                backend_used=ComputingBackend.DASK,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Daskタスク実行エラー {task.task_id}: {e}")

            return DistributedResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time_seconds=execution_time,
                backend_used=ComputingBackend.DASK,
            )

    async def execute_batch(
        self, tasks: List[DistributedTask]
    ) -> List[DistributedResult]:
        """Daskバッチ実行"""
        if not self.initialized or not DASK_AVAILABLE:
            return [await self.execute_task(task) for task in tasks]

        try:
            from dask import compute
            from dask.delayed import delayed

            # 遅延タスク作成
            delayed_tasks = []
            for task in tasks:
                delayed_task = delayed(task.task_function)(*task.args, **task.kwargs)
                delayed_tasks.append((task.task_id, delayed_task))

            # 一括計算
            start_time = time.time()
            results = compute(*[dt for _, dt in delayed_tasks])
            execution_time = time.time() - start_time

            # 結果構築
            distributed_results = []
            for i, (task_id, _) in enumerate(delayed_tasks):
                distributed_results.append(
                    DistributedResult(
                        task_id=task_id,
                        result=results[i] if i < len(results) else None,
                        success=True,
                        execution_time_seconds=execution_time / len(tasks),
                        backend_used=ComputingBackend.DASK,
                    )
                )

            return distributed_results

        except Exception as e:
            logger.error(f"Daskバッチ実行エラー: {e}")
            return [await self.execute_task(task) for task in tasks]

    def get_stats(self) -> Dict[str, Any]:
        """Dask統計情報"""
        if self.dask_processor:
            return self.dask_processor.get_stats()
        return {}

    async def cleanup(self):
        """Daskクリーンアップ"""
        if self.dask_processor:
            self.dask_processor.cleanup()
        self.initialized = False


class RayBackendManager(DistributedBackendManager):
    """Rayバックエンドマネージャー"""

    def __init__(self):
        self.initialized = False
        self.ray_context = None

    async def initialize(self, **config) -> bool:
        """Ray初期化"""
        if not RAY_AVAILABLE:
            logger.warning("Rayが利用できません")
            return False

        try:
            # Ray初期化（既に初期化済みの場合はスキップ）
            if not ray.is_initialized():
                ray_config = {
                    "num_cpus": config.get("num_cpus"),
                    "object_store_memory": config.get("object_store_memory"),
                    "log_to_driver": config.get("log_to_driver", False),
                }

                # Noneの値を除去
                ray_config = {k: v for k, v in ray_config.items() if v is not None}

                ray.init(**ray_config)

            self.ray_context = ray.get_runtime_context()
            self.initialized = True
            logger.info("Rayバックエンド初期化完了")
            return True

        except Exception as e:
            logger.error(f"Rayバックエンド初期化失敗: {e}")
            return False

    async def execute_task(self, task: DistributedTask) -> DistributedResult:
        """Rayタスク実行"""
        if not self.initialized or not RAY_AVAILABLE:
            return DistributedResult(
                task_id=task.task_id,
                success=False,
                error=RuntimeError("Rayバックエンドが初期化されていません"),
                backend_used=ComputingBackend.RAY,
            )

        start_time = time.time()

        try:
            # Ray remote function として実行
            @ray.remote
            def execute_ray_task(func, args, kwargs):
                return func(*args, **kwargs)

            # リモート実行
            future = execute_ray_task.remote(task.task_function, task.args, task.kwargs)

            # タイムアウト考慮で結果取得
            if task.timeout_seconds:
                result = ray.get(future, timeout=task.timeout_seconds)
            else:
                result = ray.get(future)

            execution_time = time.time() - start_time

            return DistributedResult(
                task_id=task.task_id,
                result=result,
                success=True,
                execution_time_seconds=execution_time,
                backend_used=ComputingBackend.RAY,
                worker_id=str(self.ray_context.get_worker_id())
                if self.ray_context
                else None,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Rayタスク実行エラー {task.task_id}: {e}")

            return DistributedResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time_seconds=execution_time,
                backend_used=ComputingBackend.RAY,
            )

    async def execute_batch(
        self, tasks: List[DistributedTask]
    ) -> List[DistributedResult]:
        """Rayバッチ実行"""
        if not self.initialized or not RAY_AVAILABLE:
            return [await self.execute_task(task) for task in tasks]

        try:

            @ray.remote
            def execute_ray_batch_task(func, args, kwargs, task_id):
                try:
                    result = func(*args, **kwargs)
                    return task_id, result, True, None
                except Exception as e:
                    return task_id, None, False, e

            # 並列実行
            futures = []
            start_time = time.time()

            for task in tasks:
                future = execute_ray_batch_task.remote(
                    task.task_function, task.args, task.kwargs, task.task_id
                )
                futures.append(future)

            # 結果収集
            batch_results = ray.get(futures)
            execution_time = time.time() - start_time

            # DistributedResult構築
            distributed_results = []
            for task_id, result, success, error in batch_results:
                distributed_results.append(
                    DistributedResult(
                        task_id=task_id,
                        result=result,
                        success=success,
                        error=error,
                        execution_time_seconds=execution_time / len(tasks),
                        backend_used=ComputingBackend.RAY,
                    )
                )

            return distributed_results

        except Exception as e:
            logger.error(f"Rayバッチ実行エラー: {e}")
            return [await self.execute_task(task) for task in tasks]

    def get_stats(self) -> Dict[str, Any]:
        """Ray統計情報"""
        if not self.initialized or not RAY_AVAILABLE:
            return {}

        try:
            stats = ray.cluster_resources()
            return {
                "ray_cluster_resources": stats,
                "ray_available_resources": ray.available_resources(),
                "ray_initialized": ray.is_initialized(),
                "ray_worker_id": str(self.ray_context.get_worker_id())
                if self.ray_context
                else None,
            }
        except Exception as e:
            logger.debug(f"Ray統計取得エラー: {e}")
            return {}

    async def cleanup(self):
        """Rayクリーンアップ"""
        try:
            if RAY_AVAILABLE and ray.is_initialized():
                ray.shutdown()
            self.initialized = False
            logger.info("Rayクリーンアップ完了")
        except Exception as e:
            logger.error(f"Rayクリーンアップエラー: {e}")


class SequentialBackendManager(DistributedBackendManager):
    """シーケンシャルバックエンドマネージャー（フォールバック）"""

    async def initialize(self, **config) -> bool:
        """シーケンシャル初期化（常に成功）"""
        return True

    async def execute_task(self, task: DistributedTask) -> DistributedResult:
        """シーケンシャルタスク実行"""
        start_time = time.time()

        try:
            result = task.task_function(*task.args, **task.kwargs)
            execution_time = time.time() - start_time

            return DistributedResult(
                task_id=task.task_id,
                result=result,
                success=True,
                execution_time_seconds=execution_time,
                backend_used=ComputingBackend.SEQUENTIAL,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return DistributedResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time_seconds=execution_time,
                backend_used=ComputingBackend.SEQUENTIAL,
            )

    async def execute_batch(
        self, tasks: List[DistributedTask]
    ) -> List[DistributedResult]:
        """シーケンシャルバッチ実行"""
        return [await self.execute_task(task) for task in tasks]

    def get_stats(self) -> Dict[str, Any]:
        """シーケンシャル統計"""
        return {"backend": "sequential", "parallel_processing": False}

    async def cleanup(self):
        """シーケンシャルクリーンアップ（何もしない）"""
        pass


class DistributedComputingManager:
    """
    統一分散コンピューティングマネージャー

    複数の分散処理バックエンド（Dask、Ray、標準並列処理）を統一的に管理し、
    タスクの性質に応じて最適なバックエンドを動的に選択する
    """

    def __init__(
        self,
        preferred_backend: ComputingBackend = ComputingBackend.AUTO,
        fallback_strategy: bool = True,
        enable_performance_tracking: bool = True,
        temp_dir: Optional[str] = None,
    ):
        """
        初期化

        Args:
            preferred_backend: 優先バックエンド
            fallback_strategy: フォールバック戦略有効化
            enable_performance_tracking: パフォーマンス追跡有効化
            temp_dir: 一時ディレクトリ
        """
        self.preferred_backend = preferred_backend
        self.fallback_strategy = fallback_strategy
        self.enable_performance_tracking = enable_performance_tracking

        # 一時ディレクトリ設定
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="distributed_computing_"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # バックエンドマネージャー
        self.backend_managers = {
            ComputingBackend.DASK: DaskBackendManager(),
            ComputingBackend.RAY: RayBackendManager(),
            ComputingBackend.SEQUENTIAL: SequentialBackendManager(),
        }

        # 利用可能バックエンド
        self.available_backends = set()

        # パフォーマンス統計
        self.performance_stats = {
            "tasks_executed": 0,
            "total_execution_time": 0.0,
            "backend_usage": defaultdict(int),
            "success_rate": 0.0,
            "average_task_time": 0.0,
        }

        # 標準並列処理マネージャー
        self.parallel_executor = ParallelExecutorManager()

        # パフォーマンスモニター
        if enable_performance_tracking:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None

        logger.info(
            f"DistributedComputingManager初期化: preferred_backend={preferred_backend.value}"
        )

    async def initialize(self, **config) -> Dict[ComputingBackend, bool]:
        """
        分散処理環境初期化

        Args:
            **config: バックエンド固有設定

        Returns:
            バックエンド別初期化結果
        """
        logger.info("分散処理環境初期化開始")

        initialization_results = {}

        # 各バックエンドの初期化試行
        for backend, manager in self.backend_managers.items():
            try:
                backend_config = config.get(backend.value, {})
                success = await manager.initialize(**backend_config)
                initialization_results[backend] = success

                if success:
                    self.available_backends.add(backend)
                    logger.info(f"{backend.value}バックエンド初期化成功")
                else:
                    logger.warning(f"{backend.value}バックエンド初期化失敗")

            except Exception as e:
                logger.error(f"{backend.value}バックエンド初期化エラー: {e}")
                initialization_results[backend] = False

        # フォールバック確保
        if not self.available_backends:
            self.available_backends.add(ComputingBackend.SEQUENTIAL)
            initialization_results[ComputingBackend.SEQUENTIAL] = True
            logger.warning(
                "全分散バックエンド初期化失敗。シーケンシャル処理にフォールバック"
            )

        logger.info(
            f"初期化完了。利用可能バックエンド: {[b.value for b in self.available_backends]}"
        )
        return initialization_results

    def select_optimal_backend(self, task: DistributedTask) -> ComputingBackend:
        """
        タスクに最適なバックエンド選択

        Args:
            task: 分散タスク

        Returns:
            選択されたバックエンド
        """
        # 優先バックエンドが AUTO でない場合はそれを使用
        if (
            self.preferred_backend != ComputingBackend.AUTO
            and self.preferred_backend in self.available_backends
        ):
            return self.preferred_backend

        # タスクタイプに基づく自動選択
        if task.task_type == TaskType.CPU_BOUND:
            # CPU集約的タスク：Ray > Dask > マルチプロセス
            for backend in [
                ComputingBackend.RAY,
                ComputingBackend.DASK,
                ComputingBackend.MULTIPROCESSING,
            ]:
                if backend in self.available_backends:
                    return backend

        elif task.task_type == TaskType.IO_BOUND:
            # I/O集約的タスク：Dask > マルチスレッド > Ray
            for backend in [
                ComputingBackend.DASK,
                ComputingBackend.THREADING,
                ComputingBackend.RAY,
            ]:
                if backend in self.available_backends:
                    return backend

        # デフォルト選択
        if ComputingBackend.DASK in self.available_backends:
            return ComputingBackend.DASK
        elif ComputingBackend.RAY in self.available_backends:
            return ComputingBackend.RAY
        else:
            return ComputingBackend.SEQUENTIAL

    async def execute_distributed_task(
        self, task: DistributedTask
    ) -> DistributedResult:
        """
        分散タスク実行

        Args:
            task: 実行タスク

        Returns:
            実行結果
        """
        # バックエンド選択
        selected_backend = self.select_optimal_backend(task)

        # パフォーマンス監視開始
        if self.performance_monitor:
            self.performance_monitor.start_monitoring()

        start_time = time.time()

        try:
            # バックエンド別実行
            if selected_backend in self.backend_managers:
                result = await self.backend_managers[selected_backend].execute_task(
                    task
                )
            else:
                # 標準並列処理フォールバック
                result = await self._execute_with_parallel_executor(
                    task, selected_backend
                )

            # 統計更新
            execution_time = time.time() - start_time
            self._update_performance_stats(
                selected_backend, execution_time, result.success
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"分散タスク実行エラー {task.task_id}: {e}")

            # フォールバック処理
            if (
                self.fallback_strategy
                and selected_backend != ComputingBackend.SEQUENTIAL
            ):
                logger.info(f"フォールバック実行: {task.task_id}")
                fallback_task = DistributedTask(
                    task_id=f"{task.task_id}_fallback",
                    task_function=task.task_function,
                    args=task.args,
                    kwargs=task.kwargs,
                    task_type=task.task_type,
                )
                return await self.backend_managers[
                    ComputingBackend.SEQUENTIAL
                ].execute_task(fallback_task)

            return DistributedResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time_seconds=execution_time,
                backend_used=selected_backend,
            )
        finally:
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()

    async def execute_distributed_batch(
        self,
        tasks: List[DistributedTask],
        strategy: TaskDistributionStrategy = TaskDistributionStrategy.DYNAMIC,
    ) -> List[DistributedResult]:
        """
        分散バッチ実行

        Args:
            tasks: タスクリスト
            strategy: 分散戦略

        Returns:
            実行結果リスト
        """
        if not tasks:
            return []

        logger.info(
            f"分散バッチ実行開始: {len(tasks)}タスク, strategy={strategy.value}"
        )

        # 分散戦略に応じてタスクをグループ化
        task_groups = self._group_tasks_by_strategy(tasks, strategy)

        # バックエンド別バッチ実行
        all_results = []

        for backend, backend_tasks in task_groups.items():
            if not backend_tasks:
                continue

            try:
                if backend in self.backend_managers:
                    batch_results = await self.backend_managers[backend].execute_batch(
                        backend_tasks
                    )
                else:
                    # 標準並列処理
                    batch_results = await self._execute_batch_with_parallel_executor(
                        backend_tasks, backend
                    )

                all_results.extend(batch_results)

                # 統計更新
                for result in batch_results:
                    self._update_performance_stats(
                        backend, result.execution_time_seconds, result.success
                    )

            except Exception as e:
                logger.error(f"バックエンド {backend.value} バッチ実行エラー: {e}")

                # エラー結果生成
                error_results = [
                    DistributedResult(
                        task_id=task.task_id,
                        success=False,
                        error=e,
                        backend_used=backend,
                    )
                    for task in backend_tasks
                ]
                all_results.extend(error_results)

        logger.info(f"分散バッチ実行完了: {len(all_results)}結果")
        return all_results

    def _group_tasks_by_strategy(
        self, tasks: List[DistributedTask], strategy: TaskDistributionStrategy
    ) -> Dict[ComputingBackend, List[DistributedTask]]:
        """分散戦略に応じたタスクグループ化"""

        task_groups = defaultdict(list)

        if strategy == TaskDistributionStrategy.DYNAMIC:
            # 動的最適化：各タスクに最適なバックエンドを選択
            for task in tasks:
                backend = self.select_optimal_backend(task)
                task_groups[backend].append(task)

        elif strategy == TaskDistributionStrategy.ROUND_ROBIN:
            # ラウンドロビン：利用可能バックエンドに均等分散
            available_backends = list(self.available_backends)
            for i, task in enumerate(tasks):
                backend = available_backends[i % len(available_backends)]
                task_groups[backend].append(task)

        elif strategy == TaskDistributionStrategy.LOAD_BALANCED:
            # 負荷分散：現在の負荷を考慮して分散
            backend_loads = self._get_backend_loads()
            available_backends = sorted(
                self.available_backends, key=lambda b: backend_loads.get(b, 0)
            )

            for i, task in enumerate(tasks):
                backend = available_backends[i % len(available_backends)]
                task_groups[backend].append(task)

        else:  # AFFINITY
            # アフィニティベース：タスクタイプに基づく分散
            cpu_bound_tasks = [t for t in tasks if t.task_type == TaskType.CPU_BOUND]
            io_bound_tasks = [t for t in tasks if t.task_type == TaskType.IO_BOUND]

            # CPU集約的タスクを高性能バックエンドに
            if ComputingBackend.RAY in self.available_backends:
                task_groups[ComputingBackend.RAY].extend(cpu_bound_tasks)
            elif ComputingBackend.DASK in self.available_backends:
                task_groups[ComputingBackend.DASK].extend(cpu_bound_tasks)

            # I/O集約的タスクをI/O特化バックエンドに
            if ComputingBackend.DASK in self.available_backends:
                task_groups[ComputingBackend.DASK].extend(io_bound_tasks)
            elif ComputingBackend.THREADING in self.available_backends:
                task_groups[ComputingBackend.THREADING].extend(io_bound_tasks)

        return task_groups

    def _get_backend_loads(self) -> Dict[ComputingBackend, float]:
        """バックエンド負荷取得"""
        loads = {}

        for backend in self.available_backends:
            try:
                if backend in self.backend_managers:
                    stats = self.backend_managers[backend].get_stats()
                    # 簡易的な負荷指標（実装依存）
                    loads[backend] = stats.get("current_load", 0.0)
                else:
                    loads[backend] = 0.0
            except Exception as e:
                logger.debug(f"バックエンド負荷取得エラー {backend.value}: {e}")
                loads[backend] = 0.0

        return loads

    async def _execute_with_parallel_executor(
        self, task: DistributedTask, backend: ComputingBackend
    ) -> DistributedResult:
        """標準並列処理実行"""

        start_time = time.time()

        try:
            if backend == ComputingBackend.THREADING:
                result = await self.parallel_executor.execute_async(
                    task.task_function,
                    *task.args,
                    task_type=task.task_type,
                    timeout_seconds=task.timeout_seconds,
                    **task.kwargs,
                )
            else:
                # シーケンシャル実行
                result = task.task_function(*task.args, **task.kwargs)

            execution_time = time.time() - start_time

            return DistributedResult(
                task_id=task.task_id,
                result=result,
                success=True,
                execution_time_seconds=execution_time,
                backend_used=backend,
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return DistributedResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time_seconds=execution_time,
                backend_used=backend,
            )

    async def _execute_batch_with_parallel_executor(
        self, tasks: List[DistributedTask], backend: ComputingBackend
    ) -> List[DistributedResult]:
        """標準並列処理バッチ実行"""

        results = []

        for task in tasks:
            result = await self._execute_with_parallel_executor(task, backend)
            results.append(result)

        return results

    def _update_performance_stats(
        self, backend: ComputingBackend, execution_time: float, success: bool
    ):
        """パフォーマンス統計更新"""
        self.performance_stats["tasks_executed"] += 1
        self.performance_stats["total_execution_time"] += execution_time
        self.performance_stats["backend_usage"][backend.value] += 1

        if self.performance_stats["tasks_executed"] > 0:
            self.performance_stats["average_task_time"] = (
                self.performance_stats["total_execution_time"]
                / self.performance_stats["tasks_executed"]
            )

            success_count = sum(1 for result_success in [success] if result_success)
            self.performance_stats["success_rate"] = (
                success_count / self.performance_stats["tasks_executed"]
            )

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """包括的統計情報取得"""
        stats = {
            "manager_stats": self.performance_stats.copy(),
            "available_backends": [b.value for b in self.available_backends],
            "preferred_backend": self.preferred_backend.value,
            "backend_stats": {},
        }

        # バックエンド別統計
        for backend, manager in self.backend_managers.items():
            if backend in self.available_backends:
                try:
                    stats["backend_stats"][backend.value] = manager.get_stats()
                except Exception as e:
                    logger.debug(f"バックエンド統計取得エラー {backend.value}: {e}")
                    stats["backend_stats"][backend.value] = {"error": str(e)}

        return stats

    def get_health_status(self) -> Dict[str, Any]:
        """ヘルスステータス"""
        health = {
            "status": "healthy",
            "available_backends_count": len(self.available_backends),
            "distributed_processing_enabled": len(self.available_backends) > 1,
            "fallback_strategy_enabled": self.fallback_strategy,
            "performance_tracking_enabled": self.enable_performance_tracking,
        }

        # 最低限のバックエンドが利用可能かチェック
        if ComputingBackend.SEQUENTIAL not in self.available_backends:
            health["status"] = "critical"
            health["issue"] = "No backends available"
        elif (
            len(self.available_backends) == 1
            and ComputingBackend.SEQUENTIAL in self.available_backends
        ):
            health["status"] = "degraded"
            health["issue"] = "Only sequential processing available"

        return health

    async def cleanup(self):
        """全リソースクリーンアップ"""
        logger.info("DistributedComputingManager クリーンアップ開始")

        # 各バックエンドのクリーンアップ
        for backend, manager in self.backend_managers.items():
            try:
                await manager.cleanup()
            except Exception as e:
                logger.error(f"バックエンドクリーンアップエラー {backend.value}: {e}")

        # 一時ディレクトリクリーンアップ
        try:
            if self.temp_dir.exists():
                import shutil

                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"一時ディレクトリクリーンアップ警告: {e}")

        logger.info("クリーンアップ完了")


# 便利関数
def create_distributed_computing_manager(
    preferred_backend: ComputingBackend = ComputingBackend.AUTO, **config
) -> DistributedComputingManager:
    """DistributedComputingManagerファクトリ関数"""
    return DistributedComputingManager(preferred_backend=preferred_backend, **config)


# defaultdictインポート
from collections import defaultdict

if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #384 統一分散コンピューティングテスト ===")

        manager = None
        try:
            # マネージャー初期化
            manager = create_distributed_computing_manager(
                preferred_backend=ComputingBackend.AUTO, fallback_strategy=True
            )

            # 初期化
            init_results = await manager.initialize(
                {
                    "dask": {"enable_distributed": True, "n_workers": 2},
                    "ray": {"num_cpus": 2},
                }
            )
            print(f"初期化結果: {init_results}")

            # テスト関数
            def cpu_intensive_task(n: int) -> float:
                """CPU集約的タスク"""
                return sum(i**2 for i in range(n))

            def io_simulation_task(delay: float) -> str:
                """I/O模擬タスク"""
                import time

                time.sleep(delay)
                return f"Completed after {delay}s"

            # 1. 単一タスクテスト
            print("\n1. 単一タスクテスト")
            task = DistributedTask(
                task_id="cpu_test_1",
                task_function=cpu_intensive_task,
                args=(10000,),
                task_type=TaskType.CPU_BOUND,
            )

            result = await manager.execute_distributed_task(task)
            print(
                f"CPU集約的タスク結果: success={result.success}, backend={result.backend_used.value}"
            )
            print(f"実行時間: {result.execution_time_seconds:.3f}秒")

            # 2. バッチタスクテスト
            print("\n2. バッチタスクテスト")
            batch_tasks = []

            # CPU集約的タスク
            for i in range(3):
                task = DistributedTask(
                    task_id=f"cpu_batch_{i}",
                    task_function=cpu_intensive_task,
                    args=(5000,),
                    task_type=TaskType.CPU_BOUND,
                )
                batch_tasks.append(task)

            # I/O集約的タスク
            for i in range(2):
                task = DistributedTask(
                    task_id=f"io_batch_{i}",
                    task_function=io_simulation_task,
                    args=(0.1,),
                    task_type=TaskType.IO_BOUND,
                )
                batch_tasks.append(task)

            batch_results = await manager.execute_distributed_batch(
                batch_tasks, strategy=TaskDistributionStrategy.DYNAMIC
            )

            print(f"バッチ処理完了: {len(batch_results)}結果")
            success_count = sum(1 for r in batch_results if r.success)
            print(f"成功率: {success_count}/{len(batch_results)}")

            # バックエンド使用状況
            backend_usage = {}
            for result in batch_results:
                backend = result.backend_used.value
                backend_usage[backend] = backend_usage.get(backend, 0) + 1
            print(f"バックエンド使用状況: {backend_usage}")

            # 3. 統計情報
            print("\n3. 統計情報")
            stats = manager.get_comprehensive_stats()
            print(f"実行タスク数: {stats['manager_stats']['tasks_executed']}")
            print(f"平均実行時間: {stats['manager_stats']['average_task_time']:.3f}秒")
            print(f"成功率: {stats['manager_stats']['success_rate']:.2%}")

            # 4. ヘルスチェック
            print("\n4. ヘルスチェック")
            health = manager.get_health_status()
            print(f"ステータス: {health['status']}")
            print(f"利用可能バックエンド数: {health['available_backends_count']}")
            print(f"分散処理有効: {health['distributed_processing_enabled']}")

        except Exception as e:
            print(f"テスト実行エラー: {e}")

        finally:
            if manager:
                await manager.cleanup()

    asyncio.run(main())
    print("\n=== 統一分散コンピューティングテスト完了 ===")
