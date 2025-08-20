"""
GPU ストリーム管理とバッチ処理

gpu_accelerated_inference.py からのリファクタリング抽出
GPU ストリーム管理、動的バッチング処理機能を提供
"""

import time
import queue
import asyncio
import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np

from .gpu_config import GPUInferenceConfig, GPUInferenceResult

# GPU計算ライブラリ (フォールバック対応)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available - stream management disabled", stacklevel=2)

# MicrosecondTimer導入
try:
    from day_trade.utils.timer import MicrosecondTimer
except ImportError:
    # フォールバック実装
    class MicrosecondTimer:
        @staticmethod
        def now_ns():
            return time.time_ns()

        @staticmethod
        def elapsed_us(start_ns):
            return (time.time_ns() - start_ns) // 1000

# ロギング設定
import logging
logger = logging.getLogger(__name__)


class GPUStreamManager:
    """GPU ストリーム管理"""

    def __init__(self, config: GPUInferenceConfig):
        self.config = config
        self.streams = {}
        self.stream_queues = {}
        self.active_streams = set()
        self.stream_lock = threading.RLock()

        self._initialize_streams()

    def _initialize_streams(self):
        """ストリーム初期化"""
        for device_id in self.config.device_ids:
            self.streams[device_id] = []
            self.stream_queues[device_id] = queue.Queue()

            try:
                if CUPY_AVAILABLE:
                    with cp.cuda.Device(device_id):
                        for i in range(self.config.max_concurrent_streams):
                            stream = cp.cuda.Stream()
                            self.streams[device_id].append(stream)
                            self.stream_queues[device_id].put((i, stream))

                    logger.info(
                        f"デバイス {device_id}: {self.config.max_concurrent_streams} ストリーム初期化完了"
                    )

            except Exception as e:
                logger.error(f"ストリーム初期化エラー (デバイス {device_id}): {e}")

    def acquire_stream(
        self, device_id: int, timeout_ms: int = 1000
    ) -> Optional[Tuple[int, Any]]:
        """ストリーム取得"""
        try:
            timeout_seconds = timeout_ms / 1000.0
            stream_info = self.stream_queues[device_id].get(timeout=timeout_seconds)

            with self.stream_lock:
                self.active_streams.add((device_id, stream_info[0]))

            return stream_info

        except queue.Empty:
            logger.warning(f"ストリーム取得タイムアウト (デバイス {device_id})")
            return None

    def release_stream(self, device_id: int, stream_id: int, stream: Any):
        """ストリーム返却"""
        try:
            # ストリーム同期
            if self.config.enable_stream_synchronization and CUPY_AVAILABLE:
                stream.synchronize()

            self.stream_queues[device_id].put((stream_id, stream))

            with self.stream_lock:
                self.active_streams.discard((device_id, stream_id))

        except Exception as e:
            logger.error(f"ストリーム返却エラー: {e}")

    def synchronize_all_streams(self):
        """全ストリーム同期"""
        if not CUPY_AVAILABLE:
            return

        for device_id, streams in self.streams.items():
            try:
                with cp.cuda.Device(device_id):
                    for stream in streams:
                        stream.synchronize()
            except Exception as e:
                logger.error(f"ストリーム同期エラー (デバイス {device_id}): {e}")

    def get_stream_stats(self) -> Dict[str, Any]:
        """ストリーム統計取得"""
        return {
            "total_streams": sum(len(streams) for streams in self.streams.values()),
            "active_streams": len(self.active_streams),
            "device_streams": {
                device_id: {
                    "total": len(streams),
                    "available": self.stream_queues[device_id].qsize(),
                    "active": len([s for s in self.active_streams if s[0] == device_id])
                }
                for device_id, streams in self.streams.items()
            }
        }

    def cleanup(self):
        """ストリームリソースクリーンアップ"""
        try:
            self.synchronize_all_streams()

            # アクティブストリームのクリア
            with self.stream_lock:
                self.active_streams.clear()

            # ストリームの削除
            for device_id in self.streams:
                self.streams[device_id].clear()
                # キューのクリア
                while not self.stream_queues[device_id].empty():
                    try:
                        self.stream_queues[device_id].get_nowait()
                    except queue.Empty:
                        break

            logger.debug("GPU ストリームマネージャークリーンアップ完了")

        except Exception as e:
            logger.error(f"GPU ストリームマネージャークリーンアップエラー: {e}")


class GPUBatchProcessor:
    """GPU 動的バッチ処理"""

    def __init__(self, config: GPUInferenceConfig):
        self.config = config
        self.pending_requests = {}  # device_id -> List[request]
        self.batch_timers = {}  # device_id -> timer
        self.batch_lock = threading.RLock()

        # デバイス毎の初期化
        for device_id in config.device_ids:
            self.pending_requests[device_id] = []
            self.batch_timers[device_id] = None

        # バッチ処理統計
        self.batch_stats = {
            "total_batches_processed": 0,
            "total_requests_batched": 0,
            "avg_batch_size": 0.0,
            "batch_timeouts": 0,
            "batch_processing_time_us": 0,
        }

    async def add_inference_request(
        self, device_id: int, input_data: np.ndarray, callback: Callable
    ) -> Optional[GPUInferenceResult]:
        """推論リクエスト追加"""
        if not self.config.dynamic_batching:
            # 動的バッチング無効時は即座処理
            return await callback(input_data)

        request = {
            "input": input_data,
            "callback": callback,
            "timestamp": MicrosecondTimer.now_ns(),
            "future": asyncio.Future(),
        }

        with self.batch_lock:
            self.pending_requests[device_id].append(request)

            # バッチサイズチェック
            if len(self.pending_requests[device_id]) >= self.config.max_batch_size:
                await self._process_batch(device_id)
            else:
                # タイマー設定
                if self.batch_timers[device_id] is None:
                    self.batch_timers[device_id] = asyncio.create_task(
                        self._batch_timeout(device_id)
                    )

        return await request["future"]

    async def _process_batch(self, device_id: int):
        """バッチ処理実行"""
        batch_start_time = MicrosecondTimer.now_ns()

        with self.batch_lock:
            if not self.pending_requests[device_id]:
                return

            # バッチ取得
            current_batch = self.pending_requests[device_id]
            self.pending_requests[device_id] = []

            # タイマーキャンセル
            if self.batch_timers[device_id]:
                self.batch_timers[device_id].cancel()
                self.batch_timers[device_id] = None

        try:
            # バッチ推論実行
            batch_inputs = np.stack([req["input"] for req in current_batch])

            # 最初のコールバックでバッチ推論
            batch_result = await current_batch[0]["callback"](batch_inputs)

            # 結果分割
            batch_size = len(current_batch)
            predictions_per_sample = batch_result.predictions.shape[0] // batch_size

            for i, request in enumerate(current_batch):
                start_idx = i * predictions_per_sample
                end_idx = (i + 1) * predictions_per_sample

                individual_result = GPUInferenceResult(
                    predictions=batch_result.predictions[start_idx:end_idx],
                    execution_time_us=batch_result.execution_time_us,
                    batch_size=1,
                    device_id=batch_result.device_id,
                    backend_used=batch_result.backend_used,
                    gpu_memory_used_mb=batch_result.gpu_memory_used_mb,
                    gpu_utilization_percent=batch_result.gpu_utilization_percent,
                    model_name=batch_result.model_name,
                    input_shape=request["input"].shape,
                )

                request["future"].set_result(individual_result)

            # 統計更新
            batch_processing_time = MicrosecondTimer.elapsed_us(batch_start_time)
            self._update_batch_stats(batch_size, batch_processing_time)

        except Exception as e:
            logger.error(f"バッチ処理エラー (デバイス {device_id}): {e}")
            for request in current_batch:
                if not request["future"].done():
                    request["future"].set_exception(e)

    async def _batch_timeout(self, device_id: int):
        """バッチタイムアウト処理"""
        await asyncio.sleep(self.config.batch_timeout_ms / 1000.0)

        with self.batch_lock:
            if self.pending_requests[device_id]:
                self.batch_stats["batch_timeouts"] += 1
                await self._process_batch(device_id)

    def _update_batch_stats(self, batch_size: int, processing_time_us: int):
        """バッチ処理統計更新"""
        self.batch_stats["total_batches_processed"] += 1
        self.batch_stats["total_requests_batched"] += batch_size
        self.batch_stats["avg_batch_size"] = (
            self.batch_stats["total_requests_batched"] / self.batch_stats["total_batches_processed"]
        )
        self.batch_stats["batch_processing_time_us"] += processing_time_us

    def get_batch_stats(self) -> Dict[str, Any]:
        """バッチ処理統計取得"""
        stats = self.batch_stats.copy()

        # 現在の保留中リクエスト数
        pending_counts = {}
        with self.batch_lock:
            for device_id, requests in self.pending_requests.items():
                pending_counts[device_id] = len(requests)

        stats["pending_requests"] = pending_counts
        stats["total_pending"] = sum(pending_counts.values())

        # 平均処理時間
        if self.batch_stats["total_batches_processed"] > 0:
            stats["avg_processing_time_us"] = (
                self.batch_stats["batch_processing_time_us"] / self.batch_stats["total_batches_processed"]
            )
        else:
            stats["avg_processing_time_us"] = 0.0

        return stats

    def force_process_pending_batches(self):
        """保留中バッチの強制処理"""
        async def process_all():
            tasks = []
            for device_id in self.pending_requests:
                if self.pending_requests[device_id]:
                    tasks.append(self._process_batch(device_id))

            if tasks:
                await asyncio.gather(*tasks)

        # 現在のイベントループで実行
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 既にイベントループが実行中の場合はタスクとして追加
                asyncio.create_task(process_all())
            else:
                # イベントループが実行中でない場合は直接実行
                loop.run_until_complete(process_all())
        except RuntimeError:
            # イベントループがない場合は新しいループを作成
            asyncio.run(process_all())

    def cleanup(self):
        """バッチプロセッサークリーンアップ"""
        try:
            # 保留中のバッチを強制処理
            self.force_process_pending_batches()

            # タイマーのキャンセル
            with self.batch_lock:
                for device_id, timer in self.batch_timers.items():
                    if timer and not timer.done():
                        timer.cancel()
                self.batch_timers.clear()

                # 保留中リクエストのクリア
                for device_id in self.pending_requests:
                    self.pending_requests[device_id].clear()

            logger.debug("GPU バッチプロセッサークリーンアップ完了")

        except Exception as e:
            logger.error(f"GPU バッチプロセッサークリーンアップエラー: {e}")

    @property
    def is_processing(self) -> bool:
        """バッチ処理中かどうか"""
        with self.batch_lock:
            return any(len(requests) > 0 for requests in self.pending_requests.values())

    def get_performance_metrics(self) -> Dict[str, Any]:
        """バッチ処理パフォーマンスメトリクス"""
        stats = self.get_batch_stats()

        metrics = {
            "efficiency": {
                "batch_utilization": stats["avg_batch_size"] / max(self.config.max_batch_size, 1),
                "timeout_rate": (
                    stats["batch_timeouts"] / max(stats["total_batches_processed"], 1)
                ),
                "throughput_batches_per_sec": 0.0,
                "throughput_requests_per_sec": 0.0
            },
            "latency": {
                "avg_batch_processing_us": stats["avg_processing_time_us"],
                "estimated_queueing_time_ms": self.config.batch_timeout_ms
            }
        }

        # スループット計算（統計期間を仮定）
        if stats["avg_processing_time_us"] > 0:
            metrics["efficiency"]["throughput_batches_per_sec"] = (
                1_000_000 / stats["avg_processing_time_us"]
            )
            metrics["efficiency"]["throughput_requests_per_sec"] = (
                metrics["efficiency"]["throughput_batches_per_sec"] * stats["avg_batch_size"]
            )

        return metrics