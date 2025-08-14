#!/usr/bin/env python3
"""
並列推論エンジン
Parallel Inference Engine for ML Model Acceleration

Issue #761: MLモデル推論パイプラインの高速化と最適化 Phase 3
"""

import asyncio
import logging
import multiprocessing as mp
import threading
import time
import queue
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from collections import defaultdict, deque
import numpy as np
import psutil
import weakref

# GPU 並列処理
try:
    import torch
    import torch.multiprocessing as torch_mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# ログ設定
logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """並列処理設定"""
    # CPU並列設定
    cpu_workers: int = mp.cpu_count()
    cpu_batch_size: int = 16
    cpu_queue_size: int = 1000

    # GPU並列設定
    gpu_workers: int = 1  # GPUの数
    gpu_batch_size: int = 32
    gpu_queue_size: int = 500
    enable_gpu_parallel: bool = False

    # 非同期設定
    async_workers: int = 10
    async_batch_timeout_ms: int = 50

    # ロードバランシング設定
    enable_load_balancing: bool = True
    load_balance_strategy: str = "round_robin"  # "round_robin", "least_loaded", "weighted"
    health_check_interval: float = 5.0

    # タスク分散設定
    enable_task_distribution: bool = True
    task_timeout_seconds: float = 30.0


@dataclass
class WorkerStats:
    """ワーカー統計"""
    worker_id: str
    worker_type: str  # "cpu", "gpu", "async"
    tasks_processed: int = 0
    total_processing_time: float = 0.0
    current_load: int = 0
    avg_processing_time_ms: float = 0.0
    error_count: int = 0
    last_activity: float = field(default_factory=time.time)
    health_status: str = "healthy"  # "healthy", "busy", "error", "offline"


@dataclass
class InferenceTask:
    """推論タスク"""
    task_id: str
    model_id: str
    input_data: np.ndarray
    callback: Optional[Callable] = None
    priority: int = 0  # 高い値が高優先度
    created_at: float = field(default_factory=time.time)
    timeout: float = 30.0


@dataclass
class InferenceResult:
    """推論結果"""
    task_id: str
    model_id: str
    output: np.ndarray
    processing_time_ms: float
    worker_id: str
    success: bool = True
    error_message: Optional[str] = None


class WorkerPool:
    """ワーカープール基底クラス"""

    def __init__(self, config: ParallelConfig, worker_type: str):
        self.config = config
        self.worker_type = worker_type
        self.workers: Dict[str, Any] = {}
        self.worker_stats: Dict[str, WorkerStats] = {}
        self.task_queues: Dict[str, queue.Queue] = {}
        self.result_queue = queue.Queue()
        self.shutdown_event = threading.Event()

    def start_workers(self) -> None:
        """ワーカー開始"""
        pass

    def stop_workers(self) -> None:
        """ワーカー停止"""
        self.shutdown_event.set()

    def submit_task(self, task: InferenceTask) -> Future:
        """タスク投入"""
        pass

    def get_worker_stats(self) -> Dict[str, WorkerStats]:
        """ワーカー統計取得"""
        return self.worker_stats.copy()


class CPUWorkerPool(WorkerPool):
    """CPUワーカープール"""

    def __init__(self, config: ParallelConfig):
        super().__init__(config, "cpu")
        self.executor = ProcessPoolExecutor(max_workers=config.cpu_workers)
        self.worker_processes: Dict[str, mp.Process] = {}

    def start_workers(self) -> None:
        """CPU ワーカー開始"""
        for i in range(self.config.cpu_workers):
            worker_id = f"cpu_worker_{i}"

            # ワーカー統計初期化
            self.worker_stats[worker_id] = WorkerStats(
                worker_id=worker_id,
                worker_type="cpu"
            )

            # タスクキュー作成
            self.task_queues[worker_id] = queue.Queue(maxsize=self.config.cpu_queue_size)

            # ワーカープロセス開始
            process = mp.Process(
                target=self._cpu_worker_loop,
                args=(worker_id, self.task_queues[worker_id], self.result_queue)
            )
            process.start()
            self.worker_processes[worker_id] = process

        logger.info(f"Started {len(self.worker_processes)} CPU workers")

    def _cpu_worker_loop(self, worker_id: str, task_queue: queue.Queue, result_queue: queue.Queue) -> None:
        """CPU ワーカーループ"""
        try:
            # モデルローダー辞書（プロセス内で保持）
            loaded_models: Dict[str, Any] = {}

            while not self.shutdown_event.is_set():
                try:
                    # タスク取得（タイムアウト付き）
                    task = task_queue.get(timeout=1.0)

                    start_time = time.perf_counter()

                    # モデル読み込み（必要に応じて）
                    if task.model_id not in loaded_models:
                        # 実際の実装では、モデルローダーを使用
                        loaded_models[task.model_id] = self._load_cpu_model(task.model_id)

                    model = loaded_models[task.model_id]

                    # 推論実行
                    output = self._cpu_inference(model, task.input_data)

                    processing_time = (time.perf_counter() - start_time) * 1000

                    # 結果作成
                    result = InferenceResult(
                        task_id=task.task_id,
                        model_id=task.model_id,
                        output=output,
                        processing_time_ms=processing_time,
                        worker_id=worker_id,
                        success=True
                    )

                    result_queue.put(result)

                except queue.Empty:
                    continue
                except Exception as e:
                    # エラー結果作成
                    error_result = InferenceResult(
                        task_id=getattr(task, 'task_id', 'unknown'),
                        model_id=getattr(task, 'model_id', 'unknown'),
                        output=np.array([]),
                        processing_time_ms=0.0,
                        worker_id=worker_id,
                        success=False,
                        error_message=str(e)
                    )
                    result_queue.put(error_result)

        except Exception as e:
            logger.error(f"CPU worker {worker_id} error: {e}")

    def _load_cpu_model(self, model_id: str) -> Any:
        """CPU モデル読み込み（ダミー実装）"""
        # 実際の実装では、モデルファイルから読み込み
        return f"CPU_Model_{model_id}"

    def _cpu_inference(self, model: Any, input_data: np.ndarray) -> np.ndarray:
        """CPU 推論実行（ダミー実装）"""
        # 実際の実装では、モデルで推論実行
        time.sleep(0.001)  # 推論時間のシミュレーション
        return np.random.randn(1, 10).astype(np.float32)

    def submit_task(self, task: InferenceTask) -> Future:
        """タスク投入"""
        return self.executor.submit(self._process_task, task)

    def _process_task(self, task: InferenceTask) -> InferenceResult:
        """タスク処理"""
        # ワーカー選択（ラウンドロビン）
        worker_ids = list(self.worker_stats.keys())
        if not worker_ids:
            raise RuntimeError("No CPU workers available")

        # 最も負荷の少ないワーカーを選択
        best_worker = min(worker_ids, key=lambda w: self.worker_stats[w].current_load)

        # タスクをワーカーキューに投入
        self.task_queues[best_worker].put(task)

        # 統計更新
        self.worker_stats[best_worker].current_load += 1

        # 結果取得
        result = self.result_queue.get()

        # 統計更新
        self.worker_stats[best_worker].current_load -= 1
        self.worker_stats[best_worker].tasks_processed += 1
        self.worker_stats[best_worker].total_processing_time += result.processing_time_ms
        self.worker_stats[best_worker].avg_processing_time_ms = (
            self.worker_stats[best_worker].total_processing_time /
            self.worker_stats[best_worker].tasks_processed
        )

        return result

    def stop_workers(self) -> None:
        """CPU ワーカー停止"""
        super().stop_workers()

        # プロセス終了
        for process in self.worker_processes.values():
            process.terminate()
            process.join(timeout=5.0)

        # エグゼキュータ終了
        self.executor.shutdown(wait=True)

        logger.info("Stopped CPU workers")


class GPUWorkerPool(WorkerPool):
    """GPU ワーカープール"""

    def __init__(self, config: ParallelConfig):
        super().__init__(config, "gpu")
        self.device_count = 0
        self.device_queues: Dict[int, asyncio.Queue] = {}
        self.gpu_processes: Dict[str, mp.Process] = {}

        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
            logger.info(f"Found {self.device_count} GPU devices")
        else:
            logger.warning("GPU not available")

    def start_workers(self) -> None:
        """GPU ワーカー開始"""
        if self.device_count == 0:
            logger.warning("No GPU devices available")
            return

        # 各GPUデバイスにワーカー作成
        for device_id in range(min(self.device_count, self.config.gpu_workers)):
            worker_id = f"gpu_worker_{device_id}"

            # ワーカー統計初期化
            self.worker_stats[worker_id] = WorkerStats(
                worker_id=worker_id,
                worker_type="gpu"
            )

            # タスクキュー作成
            self.task_queues[worker_id] = queue.Queue(maxsize=self.config.gpu_queue_size)

            # GPU ワーカープロセス開始
            process = mp.Process(
                target=self._gpu_worker_loop,
                args=(worker_id, device_id, self.task_queues[worker_id], self.result_queue)
            )
            process.start()
            self.gpu_processes[worker_id] = process

        logger.info(f"Started {len(self.gpu_processes)} GPU workers")

    def _gpu_worker_loop(self, worker_id: str, device_id: int, task_queue: queue.Queue, result_queue: queue.Queue) -> None:
        """GPU ワーカーループ"""
        try:
            if not TORCH_AVAILABLE:
                logger.error("PyTorch not available for GPU worker")
                return

            # GPU デバイス設定
            torch.cuda.set_device(device_id)
            device = torch.device(f'cuda:{device_id}')

            # モデル辞書（GPU上で保持）
            loaded_models: Dict[str, Any] = {}

            while not self.shutdown_event.is_set():
                try:
                    # タスク取得
                    task = task_queue.get(timeout=1.0)

                    start_time = time.perf_counter()

                    # モデル読み込み（必要に応じて）
                    if task.model_id not in loaded_models:
                        loaded_models[task.model_id] = self._load_gpu_model(task.model_id, device)

                    model = loaded_models[task.model_id]

                    # 推論実行
                    output = self._gpu_inference(model, task.input_data, device)

                    processing_time = (time.perf_counter() - start_time) * 1000

                    # 結果作成
                    result = InferenceResult(
                        task_id=task.task_id,
                        model_id=task.model_id,
                        output=output,
                        processing_time_ms=processing_time,
                        worker_id=worker_id,
                        success=True
                    )

                    result_queue.put(result)

                except queue.Empty:
                    continue
                except Exception as e:
                    # エラー結果作成
                    error_result = InferenceResult(
                        task_id=getattr(task, 'task_id', 'unknown'),
                        model_id=getattr(task, 'model_id', 'unknown'),
                        output=np.array([]),
                        processing_time_ms=0.0,
                        worker_id=worker_id,
                        success=False,
                        error_message=str(e)
                    )
                    result_queue.put(error_result)

        except Exception as e:
            logger.error(f"GPU worker {worker_id} error: {e}")

    def _load_gpu_model(self, model_id: str, device: torch.device) -> Any:
        """GPU モデル読み込み（ダミー実装）"""
        # 実際の実装では、モデルファイルから読み込み、GPU に転送
        return f"GPU_Model_{model_id}_on_{device}"

    def _gpu_inference(self, model: Any, input_data: np.ndarray, device: torch.device) -> np.ndarray:
        """GPU 推論実行（ダミー実装）"""
        # 実際の実装では、PyTorch テンソルで推論
        if TORCH_AVAILABLE:
            # CPU -> GPU データ転送のシミュレーション
            tensor_input = torch.from_numpy(input_data).to(device)

            # 推論のシミュレーション
            with torch.no_grad():
                time.sleep(0.0005)  # GPU 推論時間のシミュレーション
                tensor_output = torch.randn(1, 10, device=device)

            # GPU -> CPU データ転送
            output = tensor_output.cpu().numpy()
            return output.astype(np.float32)
        else:
            return np.random.randn(1, 10).astype(np.float32)

    def submit_task(self, task: InferenceTask) -> Future:
        """タスク投入"""
        # GPU が利用できない場合はエラー
        if self.device_count == 0:
            future = Future()
            future.set_exception(RuntimeError("No GPU devices available"))
            return future

        return self.executor.submit(self._process_task, task)

    def _process_task(self, task: InferenceTask) -> InferenceResult:
        """タスク処理"""
        # ワーカー選択
        worker_ids = list(self.worker_stats.keys())
        if not worker_ids:
            raise RuntimeError("No GPU workers available")

        # 最も負荷の少ないワーカーを選択
        best_worker = min(worker_ids, key=lambda w: self.worker_stats[w].current_load)

        # タスクをワーカーキューに投入
        self.task_queues[best_worker].put(task)

        # 統計更新
        self.worker_stats[best_worker].current_load += 1

        # 結果取得
        result = self.result_queue.get()

        # 統計更新
        self.worker_stats[best_worker].current_load -= 1
        self.worker_stats[best_worker].tasks_processed += 1

        return result

    def stop_workers(self) -> None:
        """GPU ワーカー停止"""
        super().stop_workers()

        # プロセス終了
        for process in self.gpu_processes.values():
            process.terminate()
            process.join(timeout=5.0)

        logger.info("Stopped GPU workers")


class AsyncWorkerPool:
    """非同期ワーカープール"""

    def __init__(self, config: ParallelConfig):
        self.config = config
        self.workers: List[asyncio.Task] = []
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.result_callbacks: Dict[str, Callable] = {}
        self.running = False

    async def start_workers(self) -> None:
        """非同期ワーカー開始"""
        self.running = True

        # 複数の非同期ワーカー開始
        for i in range(self.config.async_workers):
            worker_task = asyncio.create_task(self._async_worker_loop(f"async_worker_{i}"))
            self.workers.append(worker_task)

        logger.info(f"Started {len(self.workers)} async workers")

    async def _async_worker_loop(self, worker_id: str) -> None:
        """非同期ワーカーループ"""
        try:
            while self.running:
                try:
                    # タスク取得
                    task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )

                    start_time = time.perf_counter()

                    # 非同期推論実行
                    output = await self._async_inference(task.input_data)

                    processing_time = (time.perf_counter() - start_time) * 1000

                    # 結果作成
                    result = InferenceResult(
                        task_id=task.task_id,
                        model_id=task.model_id,
                        output=output,
                        processing_time_ms=processing_time,
                        worker_id=worker_id,
                        success=True
                    )

                    # コールバック実行
                    if task.task_id in self.result_callbacks:
                        callback = self.result_callbacks.pop(task.task_id)
                        await callback(result)

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Async worker {worker_id} error: {e}")

        except Exception as e:
            logger.error(f"Async worker {worker_id} loop error: {e}")

    async def _async_inference(self, input_data: np.ndarray) -> np.ndarray:
        """非同期推論（ダミー実装）"""
        # 実際の実装では、非同期対応のモデルで推論
        await asyncio.sleep(0.001)  # 非同期推論のシミュレーション
        return np.random.randn(1, 10).astype(np.float32)

    async def submit_task(self, task: InferenceTask, callback: Optional[Callable] = None) -> None:
        """非同期タスク投入"""
        if callback:
            self.result_callbacks[task.task_id] = callback

        await self.task_queue.put(task)

    async def stop_workers(self) -> None:
        """非同期ワーカー停止"""
        self.running = False

        # 全ワーカータスクの完了待機
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        logger.info("Stopped async workers")


class LoadBalancer:
    """ロードバランサー"""

    def __init__(self, config: ParallelConfig):
        self.config = config
        self.worker_pools: Dict[str, WorkerPool] = {}
        self.strategy = config.load_balance_strategy
        self.health_monitor = HealthMonitor(config)

    def register_worker_pool(self, pool_type: str, worker_pool: WorkerPool) -> None:
        """ワーカープール登録"""
        self.worker_pools[pool_type] = worker_pool
        logger.info(f"Registered worker pool: {pool_type}")

    def select_worker_pool(self, task: InferenceTask) -> Tuple[str, WorkerPool]:
        """ワーカープール選択"""
        if not self.worker_pools:
            raise RuntimeError("No worker pools available")

        if self.strategy == "round_robin":
            return self._round_robin_selection()
        elif self.strategy == "least_loaded":
            return self._least_loaded_selection()
        elif self.strategy == "weighted":
            return self._weighted_selection(task)
        else:
            # デフォルトはラウンドロビン
            return self._round_robin_selection()

    def _round_robin_selection(self) -> Tuple[str, WorkerPool]:
        """ラウンドロビン選択"""
        # 簡単な実装：最初の利用可能なプールを選択
        for pool_type, pool in self.worker_pools.items():
            return pool_type, pool

    def _least_loaded_selection(self) -> Tuple[str, WorkerPool]:
        """最小負荷選択"""
        best_pool = None
        best_load = float('inf')

        for pool_type, pool in self.worker_pools.items():
            stats = pool.get_worker_stats()
            total_load = sum(stat.current_load for stat in stats.values())

            if total_load < best_load:
                best_load = total_load
                best_pool = (pool_type, pool)

        if best_pool:
            return best_pool
        else:
            return next(iter(self.worker_pools.items()))

    def _weighted_selection(self, task: InferenceTask) -> Tuple[str, WorkerPool]:
        """重み付き選択"""
        # GPU タスクは GPU プールを優先
        if "gpu" in self.worker_pools and hasattr(task, 'prefer_gpu') and task.prefer_gpu:
            return "gpu", self.worker_pools["gpu"]

        # その他は最小負荷選択
        return self._least_loaded_selection()


class HealthMonitor:
    """ヘルスモニター"""

    def __init__(self, config: ParallelConfig):
        self.config = config
        self.worker_health: Dict[str, Dict[str, Any]] = {}
        self.monitoring = False

    async def start_monitoring(self) -> None:
        """ヘルス監視開始"""
        self.monitoring = True

        while self.monitoring:
            try:
                await self._check_worker_health()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    async def _check_worker_health(self) -> None:
        """ワーカーヘルスチェック"""
        # システムリソース確認
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        # GPU 使用率確認（可能な場合）
        gpu_percent = 0.0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_percent = torch.cuda.utilization()
            except:
                pass

        health_info = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "gpu_percent": gpu_percent
        }

        logger.debug(f"System health: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, GPU {gpu_percent:.1f}%")

    def stop_monitoring(self) -> None:
        """ヘルス監視停止"""
        self.monitoring = False


class ParallelInferenceEngine:
    """並列推論エンジン"""

    def __init__(self, config: ParallelConfig):
        self.config = config
        self.cpu_pool = CPUWorkerPool(config) if config.cpu_workers > 0 else None
        self.gpu_pool = GPUWorkerPool(config) if config.enable_gpu_parallel else None
        self.async_pool = AsyncWorkerPool(config)
        self.load_balancer = LoadBalancer(config)
        self.health_monitor = HealthMonitor(config)

        # 統計
        self.total_tasks: int = 0
        self.completed_tasks: int = 0
        self.failed_tasks: int = 0
        self.start_time: float = time.time()

    async def start(self) -> None:
        """エンジン開始"""
        try:
            # CPU ワーカープール開始
            if self.cpu_pool:
                self.cpu_pool.start_workers()
                self.load_balancer.register_worker_pool("cpu", self.cpu_pool)

            # GPU ワーカープール開始
            if self.gpu_pool:
                self.gpu_pool.start_workers()
                self.load_balancer.register_worker_pool("gpu", self.gpu_pool)

            # 非同期ワーカープール開始
            await self.async_pool.start_workers()

            # ヘルスモニター開始
            asyncio.create_task(self.health_monitor.start_monitoring())

            logger.info("Parallel inference engine started")

        except Exception as e:
            logger.error(f"Failed to start parallel inference engine: {e}")
            raise

    async def stop(self) -> None:
        """エンジン停止"""
        try:
            # ヘルスモニター停止
            self.health_monitor.stop_monitoring()

            # ワーカープール停止
            if self.cpu_pool:
                self.cpu_pool.stop_workers()

            if self.gpu_pool:
                self.gpu_pool.stop_workers()

            await self.async_pool.stop_workers()

            logger.info("Parallel inference engine stopped")

        except Exception as e:
            logger.error(f"Error stopping parallel inference engine: {e}")

    async def infer_async(self, task: InferenceTask) -> InferenceResult:
        """非同期推論"""
        self.total_tasks += 1

        try:
            # ワーカープール選択
            pool_type, worker_pool = self.load_balancer.select_worker_pool(task)

            # タスク実行
            if pool_type == "async":
                result_future = asyncio.Future()

                async def callback(result: InferenceResult):
                    result_future.set_result(result)

                await self.async_pool.submit_task(task, callback)
                result = await result_future
            else:
                # 同期プールを非同期で実行
                loop = asyncio.get_event_loop()
                future = worker_pool.submit_task(task)
                result = await loop.run_in_executor(None, future.result)

            if result.success:
                self.completed_tasks += 1
            else:
                self.failed_tasks += 1

            return result

        except Exception as e:
            self.failed_tasks += 1
            logger.error(f"Inference task {task.task_id} failed: {e}")

            return InferenceResult(
                task_id=task.task_id,
                model_id=task.model_id,
                output=np.array([]),
                processing_time_ms=0.0,
                worker_id="error",
                success=False,
                error_message=str(e)
            )

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """包括的統計取得"""
        uptime = time.time() - self.start_time

        stats = {
            "uptime_seconds": uptime,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.completed_tasks / max(1, self.total_tasks),
            "throughput_tasks_per_second": self.completed_tasks / max(1, uptime)
        }

        # ワーカープール統計
        if self.cpu_pool:
            stats["cpu_workers"] = self.cpu_pool.get_worker_stats()

        if self.gpu_pool:
            stats["gpu_workers"] = self.gpu_pool.get_worker_stats()

        return stats


# 使用例とテスト
async def test_parallel_inference():
    """並列推論テスト"""

    # 設定
    config = ParallelConfig(
        cpu_workers=2,
        enable_gpu_parallel=TORCH_AVAILABLE and torch.cuda.is_available(),
        gpu_workers=1 if TORCH_AVAILABLE and torch.cuda.is_available() else 0,
        async_workers=3
    )

    # エンジン初期化
    engine = ParallelInferenceEngine(config)

    try:
        await engine.start()
        print("=== Parallel Inference Test ===")

        # テストタスク作成
        tasks = []
        for i in range(10):
            task = InferenceTask(
                task_id=f"task_{i}",
                model_id=f"model_{i % 3}",  # 3つのモデルを循環
                input_data=np.random.randn(1, 10).astype(np.float32),
                priority=i % 3
            )
            tasks.append(task)

        print(f"Created {len(tasks)} test tasks")

        # 並列推論実行
        start_time = time.perf_counter()

        # 全タスクを並列実行
        results = await asyncio.gather(*[
            engine.infer_async(task) for task in tasks
        ])

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # 結果分析
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        print(f"\n=== Results ===")
        print(f"Total time: {total_time:.2f}s")
        print(f"Successful tasks: {len(successful_results)}")
        print(f"Failed tasks: {len(failed_results)}")
        print(f"Throughput: {len(tasks) / total_time:.2f} tasks/sec")

        if successful_results:
            avg_processing_time = np.mean([r.processing_time_ms for r in successful_results])
            print(f"Average processing time: {avg_processing_time:.2f}ms")

            # ワーカー別統計
            worker_usage = defaultdict(int)
            for result in successful_results:
                worker_usage[result.worker_id] += 1

            print(f"\nWorker usage:")
            for worker_id, count in worker_usage.items():
                print(f"  {worker_id}: {count} tasks")

        # エンジン統計
        print(f"\n=== Engine Statistics ===")
        engine_stats = engine.get_comprehensive_stats()
        print(f"Success rate: {engine_stats['success_rate']:.1%}")
        print(f"Throughput: {engine_stats['throughput_tasks_per_second']:.2f} tasks/sec")

        if "cpu_workers" in engine_stats:
            print(f"CPU workers: {len(engine_stats['cpu_workers'])}")

        if "gpu_workers" in engine_stats:
            print(f"GPU workers: {len(engine_stats['gpu_workers'])}")

    except Exception as e:
        print(f"Parallel inference test failed: {e}")

    finally:
        await engine.stop()
        print("Test completed")


if __name__ == "__main__":
    asyncio.run(test_parallel_inference())