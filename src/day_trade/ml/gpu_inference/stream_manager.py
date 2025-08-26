#!/usr/bin/env python3
"""
GPU ストリーム管理
Issue #379: ML Model Inference Performance Optimization
"""

import queue
import threading
import warnings
from typing import Any, Optional, Tuple

from ..utils.logging_config import get_context_logger
from .types import GPUInferenceConfig

logger = get_context_logger(__name__)

# CUDA支援ライブラリ (フォールバック対応)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available - CPU fallback", stacklevel=2)


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

    def cleanup(self):
        """ストリームリソースクリーンアップ"""
        try:
            # 全ストリーム同期
            self.synchronize_all_streams()
            
            # ストリームクリーンアップ
            with self.stream_lock:
                self.streams.clear()
                for device_id in self.stream_queues:
                    while not self.stream_queues[device_id].empty():
                        try:
                            self.stream_queues[device_id].get_nowait()
                        except queue.Empty:
                            break
                self.stream_queues.clear()
                self.active_streams.clear()
                
            logger.debug("GPUストリームマネージャークリーンアップ完了")
            
        except Exception as e:
            logger.error(f"GPUストリームマネージャークリーンアップエラー: {e}")