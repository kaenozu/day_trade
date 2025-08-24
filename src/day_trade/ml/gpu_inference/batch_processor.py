#!/usr/bin/env python3
"""
GPU 動的バッチ処理
Issue #379: ML Model Inference Performance Optimization
"""

import asyncio
import threading
from typing import Optional

import numpy as np

from ..trading.high_frequency_engine import MicrosecondTimer
from ..utils.logging_config import get_context_logger
from .types import GPUInferenceConfig, GPUInferenceResult, GPUBackend

logger = get_context_logger(__name__)


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

    async def add_inference_request(
        self, device_id: int, input_data: np.ndarray, callback: callable
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
                await self._process_batch(device_id)

    def get_pending_request_count(self, device_id: int) -> int:
        """指定デバイスの保留中リクエスト数を取得"""
        with self.batch_lock:
            return len(self.pending_requests.get(device_id, []))

    def clear_pending_requests(self, device_id: int):
        """指定デバイスの保留中リクエストをクリア"""
        with self.batch_lock:
            if device_id in self.pending_requests:
                # 保留中のリクエストにエラーを設定
                for request in self.pending_requests[device_id]:
                    if not request["future"].done():
                        request["future"].set_exception(
                            RuntimeError("バッチプロセッサーシャットダウンのためリクエストキャンセル")
                        )
                self.pending_requests[device_id] = []
                
            # タイマーキャンセル
            if self.batch_timers.get(device_id):
                self.batch_timers[device_id].cancel()
                self.batch_timers[device_id] = None

    def cleanup(self):
        """バッチプロセッサークリーンアップ"""
        try:
            with self.batch_lock:
                # 全デバイスの保留リクエストをクリア
                for device_id in list(self.pending_requests.keys()):
                    self.clear_pending_requests(device_id)
                    
                self.pending_requests.clear()
                self.batch_timers.clear()
                
            logger.debug("GPUバッチプロセッサークリーンアップ完了")
            
        except Exception as e:
            logger.error(f"GPUバッチプロセッサークリーンアップエラー: {e}")