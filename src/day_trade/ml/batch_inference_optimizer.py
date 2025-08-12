#!/usr/bin/env python3
"""
バッチ推論最適化システム
Issue #379: ML Model Inference Performance Optimization

インテリジェント動的バッチ処理
- 適応的バッチサイズ調整
- レイテンシー・スループット最適化バランス
- メモリ効率バッチスケジューリング
- 優先度ベースバッチ構成
- パイプライン並列バッチ処理
"""

import asyncio
import heapq
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from ..trading.high_frequency_engine import MemoryPool, MicrosecondTimer
from ..utils.logging_config import get_context_logger
from .gpu_accelerated_inference import GPUAcceleratedInferenceEngine, GPUInferenceResult

# 既存システムとの統合
from .optimized_inference_engine import (
    InferenceResult,
    OptimizedInferenceEngine,
)

logger = get_context_logger(__name__)


class BatchStrategy(Enum):
    """バッチ処理戦略"""

    LATENCY_OPTIMIZED = "latency_optimized"  # レイテンシー最適化
    THROUGHPUT_OPTIMIZED = "throughput_optimized"  # スループット最適化
    BALANCED = "balanced"  # バランス型
    ADAPTIVE = "adaptive"  # 適応型
    PRIORITY_BASED = "priority_based"  # 優先度ベース


class BatchSchedulingPolicy(Enum):
    """バッチスケジューリングポリシー"""

    FIFO = "fifo"  # First In First Out
    PRIORITY_QUEUE = "priority_queue"  # 優先度キュー
    SHORTEST_JOB_FIRST = "shortest_job_first"  # 最短処理時間優先
    DEADLINE_AWARE = "deadline_aware"  # 締切考慮
    LOAD_BALANCING = "load_balancing"  # 負荷分散


@dataclass
class BatchRequest:
    """バッチリクエスト"""

    id: str
    model_name: str
    input_data: np.ndarray
    priority: float = 1.0
    max_latency_ms: Optional[int] = None
    callback: Optional[Callable] = None

    # 内部管理用
    created_at: float = field(default_factory=time.time)
    estimated_processing_time_us: int = 1000
    memory_requirement_mb: float = 1.0

    # 結果管理用
    future: Optional[asyncio.Future] = None

    def __post_init__(self):
        if self.future is None:
            self.future = asyncio.Future()

        # メモリ要求量推定
        self.memory_requirement_mb = (
            self.input_data.nbytes / 1024 / 1024 if self.input_data is not None else 1.0
        )


@dataclass
class BatchConfiguration:
    """バッチ設定"""

    strategy: BatchStrategy = BatchStrategy.BALANCED
    scheduling_policy: BatchSchedulingPolicy = BatchSchedulingPolicy.PRIORITY_QUEUE

    # バッチサイズ制御
    min_batch_size: int = 1
    max_batch_size: int = 128
    target_batch_size: int = 32
    adaptive_batch_sizing: bool = True

    # タイミング制御
    max_wait_time_ms: int = 10
    collection_timeout_ms: int = 5
    processing_timeout_ms: int = 1000

    # リソース制御
    max_memory_per_batch_mb: int = 512
    max_concurrent_batches: int = 4
    enable_memory_optimization: bool = True

    # 適応制御
    latency_threshold_ms: int = 100
    throughput_threshold_per_sec: int = 1000
    adaptation_interval_ms: int = 1000
    performance_history_size: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式に変換"""
        return {
            "strategy": self.strategy.value,
            "scheduling_policy": self.scheduling_policy.value,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "target_batch_size": self.target_batch_size,
            "adaptive_batch_sizing": self.adaptive_batch_sizing,
            "max_wait_time_ms": self.max_wait_time_ms,
            "collection_timeout_ms": self.collection_timeout_ms,
            "processing_timeout_ms": self.processing_timeout_ms,
            "max_memory_per_batch_mb": self.max_memory_per_batch_mb,
            "max_concurrent_batches": self.max_concurrent_batches,
            "enable_memory_optimization": self.enable_memory_optimization,
            "latency_threshold_ms": self.latency_threshold_ms,
            "throughput_threshold_per_sec": self.throughput_threshold_per_sec,
            "adaptation_interval_ms": self.adaptation_interval_ms,
            "performance_history_size": self.performance_history_size,
        }


@dataclass
class BatchPerformanceMetrics:
    """バッチパフォーマンス指標"""

    batch_id: str
    batch_size: int
    total_processing_time_us: int
    avg_request_latency_us: float

    memory_usage_mb: float
    throughput_requests_per_sec: float

    # 詳細指標
    queue_wait_time_us: int = 0
    batch_formation_time_us: int = 0
    inference_time_us: int = 0
    result_distribution_time_us: int = 0

    success_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """メトリクスを辞書形式に変換"""
        return {
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "total_processing_time_us": self.total_processing_time_us,
            "avg_request_latency_us": self.avg_request_latency_us,
            "memory_usage_mb": self.memory_usage_mb,
            "throughput_requests_per_sec": self.throughput_requests_per_sec,
            "queue_wait_time_us": self.queue_wait_time_us,
            "batch_formation_time_us": self.batch_formation_time_us,
            "inference_time_us": self.inference_time_us,
            "result_distribution_time_us": self.result_distribution_time_us,
            "success_count": self.success_count,
            "error_count": self.error_count,
        }


class AdaptiveBatchSizer:
    """適応的バッチサイザー"""

    def __init__(self, config: BatchConfiguration):
        self.config = config
        self.performance_history = deque(maxlen=config.performance_history_size)
        self.current_batch_size = config.target_batch_size

        # 適応制御パラメータ
        self.latency_trend = deque(maxlen=10)
        self.throughput_trend = deque(maxlen=10)
        self.last_adaptation_time = time.time()

    def record_performance(self, metrics: BatchPerformanceMetrics):
        """パフォーマンス記録"""
        self.performance_history.append(metrics)

        # トレンド更新
        self.latency_trend.append(metrics.avg_request_latency_us / 1000)  # ms
        self.throughput_trend.append(metrics.throughput_requests_per_sec)

    def should_adapt(self) -> bool:
        """適応判定"""
        current_time = time.time()
        time_since_last_adaptation = (current_time - self.last_adaptation_time) * 1000

        return (
            time_since_last_adaptation >= self.config.adaptation_interval_ms
            and len(self.performance_history) >= 5
        )

    def compute_optimal_batch_size(self) -> int:
        """最適バッチサイズ計算"""
        if not self.should_adapt():
            return self.current_batch_size

        try:
            recent_metrics = list(self.performance_history)[-10:]  # 直近10回

            if self.config.strategy == BatchStrategy.LATENCY_OPTIMIZED:
                optimal_size = self._optimize_for_latency(recent_metrics)
            elif self.config.strategy == BatchStrategy.THROUGHPUT_OPTIMIZED:
                optimal_size = self._optimize_for_throughput(recent_metrics)
            elif self.config.strategy == BatchStrategy.BALANCED:
                optimal_size = self._optimize_balanced(recent_metrics)
            elif self.config.strategy == BatchStrategy.ADAPTIVE:
                optimal_size = self._adaptive_optimization(recent_metrics)
            else:
                optimal_size = self.current_batch_size

            # 制約適用
            optimal_size = max(
                self.config.min_batch_size,
                min(self.config.max_batch_size, optimal_size),
            )

            if optimal_size != self.current_batch_size:
                logger.debug(
                    f"バッチサイズ適応: {self.current_batch_size} → {optimal_size}"
                )
                self.current_batch_size = optimal_size
                self.last_adaptation_time = time.time()

            return optimal_size

        except Exception as e:
            logger.warning(f"バッチサイズ最適化エラー: {e}")
            return self.current_batch_size

    def _optimize_for_latency(self, metrics: List[BatchPerformanceMetrics]) -> int:
        """レイテンシー最適化"""
        if not metrics:
            return self.current_batch_size

        # 平均レイテンシーとバッチサイズの関係分析
        latency_by_size = defaultdict(list)
        for metric in metrics:
            latency_by_size[metric.batch_size].append(
                metric.avg_request_latency_us / 1000
            )

        # 最低レイテンシーのバッチサイズ選択
        best_size = self.current_batch_size
        best_latency = float("inf")

        for batch_size, latencies in latency_by_size.items():
            avg_latency = statistics.mean(latencies)
            if avg_latency < best_latency:
                best_latency = avg_latency
                best_size = batch_size

        # 現在のサイズから段階的に変更
        if best_size < self.current_batch_size:
            return max(best_size, self.current_batch_size - 4)
        elif best_size > self.current_batch_size:
            return min(best_size, self.current_batch_size + 4)
        else:
            return self.current_batch_size

    def _optimize_for_throughput(self, metrics: List[BatchPerformanceMetrics]) -> int:
        """スループット最適化"""
        if not metrics:
            return self.current_batch_size

        # スループットとバッチサイズの関係分析
        throughput_by_size = defaultdict(list)
        for metric in metrics:
            throughput_by_size[metric.batch_size].append(
                metric.throughput_requests_per_sec
            )

        # 最高スループットのバッチサイズ選択
        best_size = self.current_batch_size
        best_throughput = 0.0

        for batch_size, throughputs in throughput_by_size.items():
            avg_throughput = statistics.mean(throughputs)
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_size = batch_size

        # より大きなバッチサイズへの傾向
        if best_size > self.current_batch_size:
            return min(best_size, self.current_batch_size + 8)
        else:
            return self.current_batch_size

    def _optimize_balanced(self, metrics: List[BatchPerformanceMetrics]) -> int:
        """バランス最適化"""
        if not metrics:
            return self.current_batch_size

        # レイテンシーとスループットの重み付きスコア
        scores_by_size = defaultdict(list)

        for metric in metrics:
            latency_score = 1.0 / (
                1.0 + metric.avg_request_latency_us / 10000
            )  # 正規化
            throughput_score = metric.throughput_requests_per_sec / 1000  # 正規化

            # 重み付き合成スコア（レイテンシー:スループット = 0.6:0.4）
            composite_score = 0.6 * latency_score + 0.4 * throughput_score
            scores_by_size[metric.batch_size].append(composite_score)

        # 最高スコアのバッチサイズ選択
        best_size = self.current_batch_size
        best_score = 0.0

        for batch_size, scores in scores_by_size.items():
            avg_score = statistics.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_size = batch_size

        return best_size

    def _adaptive_optimization(self, metrics: List[BatchPerformanceMetrics]) -> int:
        """適応的最適化"""
        if not metrics:
            return self.current_batch_size

        recent_latency = (
            statistics.mean(self.latency_trend) if self.latency_trend else 50
        )
        recent_throughput = (
            statistics.mean(self.throughput_trend) if self.throughput_trend else 100
        )

        # 現在の性能レベル判定
        latency_ok = recent_latency <= self.config.latency_threshold_ms
        throughput_ok = recent_throughput >= self.config.throughput_threshold_per_sec

        if latency_ok and throughput_ok:
            # 両方満足 → 小幅増加でスループット向上を狙う
            return min(self.current_batch_size + 2, self.config.max_batch_size)
        elif latency_ok and not throughput_ok:
            # レイテンシーOK, スループット不足 → バッチサイズ増加
            return min(self.current_batch_size + 4, self.config.max_batch_size)
        elif not latency_ok and throughput_ok:
            # レイテンシー問題あり → バッチサイズ削減
            return max(self.current_batch_size - 4, self.config.min_batch_size)
        else:
            # 両方問題あり → より積極的なバッチサイズ削減
            return max(self.current_batch_size - 8, self.config.min_batch_size)


class BatchScheduler:
    """バッチスケジューラー"""

    def __init__(self, config: BatchConfiguration):
        self.config = config

        # リクエスト管理
        if config.scheduling_policy == BatchSchedulingPolicy.PRIORITY_QUEUE:
            self.request_queue = []  # heapq用
        else:
            self.request_queue = deque()  # FIFO用

        self.pending_requests: Dict[str, BatchRequest] = {}
        self.queue_lock = threading.RLock()

        # バッチ管理
        self.active_batches: Dict[str, List[BatchRequest]] = {}
        self.batch_counter = 0

    def add_request(self, request: BatchRequest) -> bool:
        """リクエスト追加"""
        with self.queue_lock:
            try:
                self.pending_requests[request.id] = request

                if (
                    self.config.scheduling_policy
                    == BatchSchedulingPolicy.PRIORITY_QUEUE
                ):
                    # 優先度キュー（負の値でheapqを最大ヒープとして使用）
                    heapq.heappush(
                        self.request_queue,
                        (-request.priority, request.created_at, request.id),
                    )
                elif (
                    self.config.scheduling_policy
                    == BatchSchedulingPolicy.SHORTEST_JOB_FIRST
                ):
                    # 推定処理時間順
                    heapq.heappush(
                        self.request_queue,
                        (
                            request.estimated_processing_time_us,
                            request.created_at,
                            request.id,
                        ),
                    )
                else:
                    # FIFO
                    self.request_queue.append(request.id)

                return True

            except Exception as e:
                logger.error(f"リクエスト追加エラー: {e}")
                return False

    def form_batch(self, target_size: int) -> Optional[List[BatchRequest]]:
        """バッチ構成"""
        with self.queue_lock:
            if not self.request_queue:
                return None

            batch_requests = []
            total_memory_mb = 0.0

            # リクエスト選択
            while (
                len(batch_requests) < target_size
                and len(batch_requests) < self.config.max_batch_size
                and self.request_queue
            ):
                # 次のリクエスト取得
                if self.config.scheduling_policy in [
                    BatchSchedulingPolicy.PRIORITY_QUEUE,
                    BatchSchedulingPolicy.SHORTEST_JOB_FIRST,
                ]:
                    if self.request_queue:
                        _, _, request_id = heapq.heappop(self.request_queue)
                    else:
                        break
                else:
                    if self.request_queue:
                        request_id = self.request_queue.popleft()
                    else:
                        break

                if request_id not in self.pending_requests:
                    continue

                request = self.pending_requests[request_id]

                # メモリ制約チェック
                if (
                    self.config.enable_memory_optimization
                    and total_memory_mb + request.memory_requirement_mb
                    > self.config.max_memory_per_batch_mb
                ):
                    # メモリ不足 → リクエストを戻す
                    if self.config.scheduling_policy in [
                        BatchSchedulingPolicy.PRIORITY_QUEUE,
                        BatchSchedulingPolicy.SHORTEST_JOB_FIRST,
                    ]:
                        heapq.heappush(
                            self.request_queue,
                            (-request.priority, request.created_at, request_id),
                        )
                    else:
                        self.request_queue.appendleft(request_id)
                    break

                # バッチに追加
                batch_requests.append(request)
                total_memory_mb += request.memory_requirement_mb
                del self.pending_requests[request_id]

            # バッチサイズ制約チェック
            if (
                len(batch_requests) < self.config.min_batch_size
                and len(self.request_queue) == 0
            ):
                # 最小バッチサイズに達しないがキューが空 → 現在のバッチを処理
                pass
            elif len(batch_requests) < self.config.min_batch_size:
                # 最小バッチサイズに達しない → リクエストを戻してNoneを返す
                for req in batch_requests:
                    self.pending_requests[req.id] = req
                    if (
                        self.config.scheduling_policy
                        == BatchSchedulingPolicy.PRIORITY_QUEUE
                    ):
                        heapq.heappush(
                            self.request_queue, (-req.priority, req.created_at, req.id)
                        )
                    else:
                        self.request_queue.appendleft(req.id)
                return None

            if batch_requests:
                # バッチID生成
                self.batch_counter += 1
                batch_id = f"batch_{self.batch_counter}_{int(time.time() * 1000)}"
                self.active_batches[batch_id] = batch_requests

                logger.debug(f"バッチ構成完了: {batch_id} ({len(batch_requests)}件)")
                return batch_requests

            return None

    def get_queue_stats(self) -> Dict[str, Any]:
        """キュー統計取得"""
        with self.queue_lock:
            return {
                "pending_requests": len(self.pending_requests),
                "queue_size": len(self.request_queue),
                "active_batches": len(self.active_batches),
                "total_batches_processed": self.batch_counter,
            }


class BatchInferenceOptimizer:
    """バッチ推論最適化システム（メイン）"""

    def __init__(
        self,
        config: BatchConfiguration = None,
        inference_engine: Optional[OptimizedInferenceEngine] = None,
        gpu_engine: Optional[GPUAcceleratedInferenceEngine] = None,
    ):
        self.config = config or BatchConfiguration()
        self.inference_engine = inference_engine
        self.gpu_engine = gpu_engine

        # コンポーネント初期化
        self.batch_sizer = AdaptiveBatchSizer(self.config)
        self.scheduler = BatchScheduler(self.config)

        # 処理管理
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.batch_processors = []
        self.is_running = False

        # 統計
        self.performance_stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_latency_ms": 0.0,
            "avg_throughput_per_sec": 0.0,
            "total_processing_time_ms": 0.0,
            "success_rate": 0.0,
        }

        # メモリプール統合
        try:
            self.memory_pool = MemoryPool(self.config.max_memory_per_batch_mb * 2)
        except Exception as e:
            logger.warning(f"メモリプール初期化失敗: {e}")
            self.memory_pool = None

        logger.info(f"バッチ推論最適化システム初期化完了: {self.config.to_dict()}")

    async def start(self):
        """バッチ処理開始"""
        if self.is_running:
            logger.warning("バッチ処理既に実行中")
            return

        self.is_running = True
        logger.info("バッチ推論最適化システム開始")

        # バッチプロセッサー起動
        for i in range(self.config.max_concurrent_batches):
            processor_task = asyncio.create_task(
                self._batch_processor_loop(f"processor_{i}")
            )
            self.batch_processors.append(processor_task)

    async def stop(self):
        """バッチ処理停止"""
        if not self.is_running:
            return

        logger.info("バッチ推論最適化システム停止中...")
        self.is_running = False

        # プロセッサー停止
        for processor in self.batch_processors:
            processor.cancel()

        # 実行中タスク完了待ち
        if self.processing_tasks:
            await asyncio.gather(
                *self.processing_tasks.values(), return_exceptions=True
            )

        self.batch_processors.clear()
        self.processing_tasks.clear()

        logger.info("バッチ推論最適化システム停止完了")

    async def submit_request(
        self,
        model_name: str,
        input_data: np.ndarray,
        priority: float = 1.0,
        max_latency_ms: Optional[int] = None,
    ) -> Union[InferenceResult, GPUInferenceResult]:
        """推論リクエスト送信"""
        request_id = (
            f"req_{int(time.time() * 1000000)}_{hash(input_data.tobytes()) & 0xFFFF}"
        )

        request = BatchRequest(
            id=request_id,
            model_name=model_name,
            input_data=input_data,
            priority=priority,
            max_latency_ms=max_latency_ms,
        )

        # リクエスト追加
        success = self.scheduler.add_request(request)
        if not success:
            raise RuntimeError(f"リクエスト追加失敗: {request_id}")

        self.performance_stats["total_requests"] += 1

        # 結果待ち
        try:
            if max_latency_ms:
                result = await asyncio.wait_for(
                    request.future, timeout=max_latency_ms / 1000.0
                )
            else:
                result = await request.future

            return result

        except asyncio.TimeoutError:
            logger.warning(f"リクエストタイムアウト: {request_id}")
            raise
        except Exception as e:
            logger.error(f"リクエスト処理エラー: {request_id}, {e}")
            raise

    async def _batch_processor_loop(self, processor_name: str):
        """バッチプロセッサーループ"""
        logger.debug(f"バッチプロセッサー開始: {processor_name}")

        while self.is_running:
            try:
                # 最適バッチサイズ計算
                optimal_batch_size = self.batch_sizer.compute_optimal_batch_size()

                # バッチ構成
                batch_requests = self.scheduler.form_batch(optimal_batch_size)

                if batch_requests:
                    # バッチ処理実行
                    await self._process_batch(batch_requests, processor_name)
                else:
                    # バッチなし → 短時間待機
                    await asyncio.sleep(self.config.collection_timeout_ms / 1000.0)

            except asyncio.CancelledError:
                logger.debug(f"バッチプロセッサー停止: {processor_name}")
                break
            except Exception as e:
                logger.error(f"バッチプロセッサーエラー ({processor_name}): {e}")
                await asyncio.sleep(0.1)  # エラー時の短時間待機

    async def _process_batch(
        self, batch_requests: List[BatchRequest], processor_name: str
    ):
        """バッチ処理実行"""
        batch_start_time = MicrosecondTimer.now_ns()
        batch_id = f"{processor_name}_{int(time.time() * 1000)}"

        logger.debug(f"バッチ処理開始: {batch_id} ({len(batch_requests)}件)")

        try:
            # バッチデータ構築
            formation_start = MicrosecondTimer.now_ns()

            # モデル別グループ化
            model_groups = defaultdict(list)
            for request in batch_requests:
                model_groups[request.model_name].append(request)

            batch_formation_time = MicrosecondTimer.elapsed_us(formation_start)

            # モデル別並列推論
            inference_start = MicrosecondTimer.now_ns()
            model_results = {}

            # 各モデルのバッチ推論実行
            for model_name, requests in model_groups.items():
                try:
                    # 入力データ結合
                    batch_input = np.vstack([req.input_data for req in requests])

                    # 推論実行
                    if self.gpu_engine:
                        result = await self.gpu_engine.predict_gpu(
                            model_name, batch_input, use_cache=True
                        )
                    elif self.inference_engine:
                        result = await self.inference_engine.predict(
                            model_name, batch_input, use_cache=True
                        )
                    else:
                        raise RuntimeError("推論エンジンが設定されていません")

                    model_results[model_name] = (result, requests)

                except Exception as e:
                    logger.error(f"モデル {model_name} バッチ推論エラー: {e}")
                    # エラー時は個別リクエストにエラー設定
                    for request in requests:
                        if not request.future.done():
                            request.future.set_exception(e)

            inference_time = MicrosecondTimer.elapsed_us(inference_start)

            # 結果分配
            distribution_start = MicrosecondTimer.now_ns()
            success_count = 0
            error_count = 0

            for model_name, (batch_result, requests) in model_results.items():
                try:
                    # 結果分割
                    batch_size = len(requests)
                    predictions_per_request = (
                        batch_result.predictions.shape[0] // batch_size
                    )

                    for i, request in enumerate(requests):
                        try:
                            start_idx = i * predictions_per_request
                            end_idx = (i + 1) * predictions_per_request

                            # 個別結果作成
                            if isinstance(batch_result, GPUInferenceResult):
                                individual_result = GPUInferenceResult(
                                    predictions=batch_result.predictions[
                                        start_idx:end_idx
                                    ],
                                    execution_time_us=batch_result.execution_time_us,
                                    batch_size=1,
                                    device_id=batch_result.device_id,
                                    backend_used=batch_result.backend_used,
                                    gpu_memory_used_mb=batch_result.gpu_memory_used_mb
                                    / batch_size,
                                    gpu_utilization_percent=batch_result.gpu_utilization_percent,
                                    model_name=model_name,
                                    input_shape=request.input_data.shape,
                                )
                            else:
                                individual_result = InferenceResult(
                                    predictions=batch_result.predictions[
                                        start_idx:end_idx
                                    ],
                                    execution_time_us=batch_result.execution_time_us,
                                    batch_size=1,
                                    backend_used=batch_result.backend_used,
                                    model_name=model_name,
                                    input_shape=request.input_data.shape,
                                )

                            # 結果設定
                            if not request.future.done():
                                request.future.set_result(individual_result)
                                success_count += 1

                        except Exception as e:
                            logger.error(f"個別結果設定エラー: {e}")
                            if not request.future.done():
                                request.future.set_exception(e)
                                error_count += 1

                except Exception as e:
                    logger.error(f"結果分配エラー ({model_name}): {e}")
                    for request in requests:
                        if not request.future.done():
                            request.future.set_exception(e)
                            error_count += 1

            result_distribution_time = MicrosecondTimer.elapsed_us(distribution_start)
            total_processing_time = MicrosecondTimer.elapsed_us(batch_start_time)

            # パフォーマンス統計記録
            avg_latency = (
                total_processing_time / len(batch_requests) if batch_requests else 0
            )
            throughput = (
                len(batch_requests) / (total_processing_time / 1_000_000)
                if total_processing_time > 0
                else 0
            )

            metrics = BatchPerformanceMetrics(
                batch_id=batch_id,
                batch_size=len(batch_requests),
                total_processing_time_us=total_processing_time,
                avg_request_latency_us=avg_latency,
                memory_usage_mb=sum(
                    req.memory_requirement_mb for req in batch_requests
                ),
                throughput_requests_per_sec=throughput,
                queue_wait_time_us=0,  # 簡略化
                batch_formation_time_us=batch_formation_time,
                inference_time_us=inference_time,
                result_distribution_time_us=result_distribution_time,
                success_count=success_count,
                error_count=error_count,
            )

            self.batch_sizer.record_performance(metrics)
            self._update_global_stats(metrics)

            logger.debug(
                f"バッチ処理完了: {batch_id} "
                f"({len(batch_requests)}件, {total_processing_time/1000:.1f}ms, "
                f"{throughput:.1f}req/sec)"
            )

        except Exception as e:
            logger.error(f"バッチ処理エラー: {batch_id}, {e}")
            # 全リクエストにエラー設定
            for request in batch_requests:
                if not request.future.done():
                    request.future.set_exception(e)

    def _update_global_stats(self, metrics: BatchPerformanceMetrics):
        """グローバル統計更新"""
        try:
            total_batches = self.performance_stats["total_batches"]

            # 移動平均更新
            self.performance_stats["total_batches"] += 1
            self.performance_stats["avg_batch_size"] = (
                self.performance_stats["avg_batch_size"] * total_batches
                + metrics.batch_size
            ) / (total_batches + 1)
            self.performance_stats["avg_latency_ms"] = (
                self.performance_stats["avg_latency_ms"] * total_batches
                + metrics.avg_request_latency_us / 1000
            ) / (total_batches + 1)
            self.performance_stats["avg_throughput_per_sec"] = (
                self.performance_stats["avg_throughput_per_sec"] * total_batches
                + metrics.throughput_requests_per_sec
            ) / (total_batches + 1)
            self.performance_stats["total_processing_time_ms"] += (
                metrics.total_processing_time_us / 1000
            )

            # 成功率更新
            total_processed = metrics.success_count + metrics.error_count
            if total_processed > 0:
                batch_success_rate = metrics.success_count / total_processed
                self.performance_stats["success_rate"] = (
                    self.performance_stats["success_rate"] * total_batches
                    + batch_success_rate
                ) / (total_batches + 1)

        except Exception as e:
            logger.warning(f"統計更新エラー: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        stats = self.performance_stats.copy()

        # 追加統計
        stats.update(
            {
                "queue_stats": self.scheduler.get_queue_stats(),
                "current_batch_size": self.batch_sizer.current_batch_size,
                "config": self.config.to_dict(),
                "is_running": self.is_running,
                "active_processors": len(self.batch_processors),
            }
        )

        return stats

    async def benchmark_batch_performance(
        self, model_name: str, test_data: List[np.ndarray], iterations: int = 100
    ) -> Dict[str, Any]:
        """バッチパフォーマンス ベンチマーク"""
        logger.info(f"バッチ推論ベンチマーク開始: {model_name}, {iterations}回")

        if not self.is_running:
            await self.start()

        # ベンチマーク実行
        latencies = []
        throughput_measurements = []
        batch_start_time = time.time()

        # 並列リクエスト送信
        tasks = []
        for i in range(iterations):
            data = test_data[i % len(test_data)]
            task = asyncio.create_task(self.submit_request(model_name, data))
            tasks.append(task)

        # 結果収集
        results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_end_time = time.time()

        # 統計計算
        successful_results = [r for r in results if not isinstance(r, Exception)]

        if successful_results:
            latencies = [r.execution_time_us for r in successful_results]
            total_time = batch_end_time - batch_start_time
            throughput = len(successful_results) / total_time

            benchmark_results = {
                "model_name": model_name,
                "iterations": iterations,
                "successful_requests": len(successful_results),
                "failed_requests": len(results) - len(successful_results),
                "avg_latency_us": np.mean(latencies),
                "min_latency_us": np.min(latencies),
                "max_latency_us": np.max(latencies),
                "p95_latency_us": np.percentile(latencies, 95),
                "p99_latency_us": np.percentile(latencies, 99),
                "total_throughput_per_sec": throughput,
                "total_benchmark_time_sec": total_time,
                "current_performance_stats": self.get_performance_stats(),
            }
        else:
            benchmark_results = {
                "model_name": model_name,
                "iterations": iterations,
                "successful_requests": 0,
                "failed_requests": len(results),
                "error": "全リクエスト失敗",
            }

        logger.info(f"バッチ推論ベンチマーク完了: {benchmark_results}")
        return benchmark_results


# エクスポート用ファクトリ関数
async def create_batch_inference_optimizer(
    strategy: BatchStrategy = BatchStrategy.BALANCED,
    max_batch_size: int = 64,
    inference_engine: Optional[OptimizedInferenceEngine] = None,
    gpu_engine: Optional[GPUAcceleratedInferenceEngine] = None,
) -> BatchInferenceOptimizer:
    """バッチ推論最適化システム作成"""
    config = BatchConfiguration(
        strategy=strategy, max_batch_size=max_batch_size, adaptive_batch_sizing=True
    )

    optimizer = BatchInferenceOptimizer(
        config=config, inference_engine=inference_engine, gpu_engine=gpu_engine
    )

    return optimizer


if __name__ == "__main__":
    # テスト実行
    async def test_batch_optimizer():
        print("=== バッチ推論最適化システム テスト ===")

        # 最適化システム作成
        optimizer = await create_batch_inference_optimizer()

        # 統計表示
        stats = optimizer.get_performance_stats()
        print(f"初期化完了: {stats}")

        print("✅ バッチ推論最適化システム テスト完了")

    import asyncio

    asyncio.run(test_batch_optimizer())
