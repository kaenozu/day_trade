#!/usr/bin/env python3
"""
GPU加速推論エンジン（メイン・コンパクト版）
Issue #379: ML Model Inference Performance Optimization
"""

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..trading.high_frequency_engine import MicrosecondTimer
from ..utils.logging_config import get_context_logger
from ..utils.unified_cache_manager import UnifiedCacheManager
from .types import GPUInferenceConfig, GPUInferenceResult, GPUBackend, GPUMonitoringData
from .device_manager import GPUDeviceManager
from .stream_manager import GPUStreamManager
from .batch_processor import GPUBatchProcessor
from .inference_session import GPUInferenceSession

logger = get_context_logger(__name__)


class GPUAcceleratedInferenceEngine:
    """GPU加速推論エンジン（メイン）"""

    def __init__(self, config: GPUInferenceConfig = None):
        self.config = config or GPUInferenceConfig()

        # コンポーネント初期化
        self.device_manager = GPUDeviceManager()
        self.stream_manager = GPUStreamManager(self.config)
        self.batch_processor = GPUBatchProcessor(self.config)

        # セッション管理
        self.sessions: Dict[str, GPUInferenceSession] = {}
        self.session_device_mapping: Dict[str, int] = {}

        # キャッシュシステム統合
        try:
            self.cache_manager = UnifiedCacheManager(
                l1_memory_mb=100, l2_memory_mb=200, l3_disk_mb=1000
            )
        except Exception as e:
            logger.warning(f"キャッシュマネージャー初期化失敗: {e}")
            self.cache_manager = None

        # エンジン統計
        self.engine_stats = {
            "models_loaded": 0,
            "total_gpu_inferences": 0,
            "total_gpu_time_us": 0,
            "avg_gpu_time_us": 0.0,
            "total_gpu_memory_mb": 0.0,
            "peak_gpu_utilization": 0.0,
        }

        logger.info(
            f"GPU加速推論エンジン初期化完了: {len(self.device_manager.available_devices)} GPU"
        )

    async def load_model(
        self, model_path: str, model_name: str, device_id: Optional[int] = None
    ) -> bool:
        """モデル読み込み"""
        try:
            # デバイス選択
            if device_id is None:
                optimal_device = self.device_manager.get_optimal_device()
                device_id = optimal_device["id"]

            # セッション作成
            session = GPUInferenceSession(
                model_path, self.config, device_id, model_name
            )

            if session.session is not None:
                self.sessions[model_name] = session
                self.session_device_mapping[model_name] = device_id
                self.engine_stats["models_loaded"] += 1

                logger.info(
                    f"GPU モデル読み込み完了: {model_name} (デバイス {device_id})"
                )
                return True
            else:
                logger.error(f"GPU モデル読み込み失敗: {model_name}")
                return False

        except Exception as e:
            logger.error(f"GPU モデル読み込みエラー: {model_name}, {e}")
            return False

    async def predict_gpu(
        self,
        model_name: str,
        input_data: np.ndarray,
        use_cache: bool = True,
        stream_id: Optional[int] = None,
    ) -> GPUInferenceResult:
        """GPU推論実行"""
        start_time = MicrosecondTimer.now_ns()

        try:
            # キャッシュチェック
            cache_key = None
            if use_cache and self.cache_manager:
                cache_key = f"gpu_{model_name}_{hash(input_data.tobytes())}"
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    result = GPUInferenceResult(**cached_result)
                    result.cache_hit = True
                    result.execution_time_us = MicrosecondTimer.elapsed_us(start_time)
                    return result

            # モデル取得
            if model_name not in self.sessions:
                raise ValueError(f"GPU モデルが読み込まれていません: {model_name}")

            session = self.sessions[model_name]
            device_id = self.session_device_mapping[model_name]

            # 動的バッチング
            if self.config.dynamic_batching:
                result = await self.batch_processor.add_inference_request(
                    device_id, input_data, lambda data: session.predict_gpu(data)
                )
            else:
                result = await session.predict_gpu(input_data)

            # キャッシュ保存
            if use_cache and self.cache_manager and cache_key:
                self.cache_manager.put(
                    cache_key,
                    result.to_dict(),
                    priority=8.0,  # GPU結果は高優先度
                )

            # エンジン統計更新
            self.engine_stats["total_gpu_inferences"] += 1
            self.engine_stats["total_gpu_time_us"] += result.execution_time_us
            self.engine_stats["avg_gpu_time_us"] = (
                self.engine_stats["total_gpu_time_us"]
                / self.engine_stats["total_gpu_inferences"]
            )
            self.engine_stats["total_gpu_memory_mb"] += result.gpu_memory_used_mb
            self.engine_stats["peak_gpu_utilization"] = max(
                self.engine_stats["peak_gpu_utilization"],
                result.gpu_utilization_percent,
            )

            return result

        except Exception as e:
            execution_time = MicrosecondTimer.elapsed_us(start_time)
            logger.error(f"GPU推論実行エラー: {model_name}, {e}")

            # エラー時フォールバック結果
            return GPUInferenceResult(
                predictions=np.zeros((input_data.shape[0], 1)),
                execution_time_us=execution_time,
                batch_size=input_data.shape[0],
                device_id=-1,
                backend_used=GPUBackend.CPU_FALLBACK,
                model_name=model_name,
                input_shape=input_data.shape,
            )

    async def predict_multi_gpu(
        self, requests: List[Tuple[str, np.ndarray]]
    ) -> List[GPUInferenceResult]:
        """複数GPU並列推論"""
        if len(requests) <= 1:
            if requests:
                model_name, input_data = requests[0]
                return [await self.predict_gpu(model_name, input_data)]
            else:
                return []

        # デバイス別にリクエスト分散
        device_groups = {}
        for i, (model_name, input_data) in enumerate(requests):
            device_id = self.session_device_mapping.get(model_name, 0)
            if device_id not in device_groups:
                device_groups[device_id] = []
            device_groups[device_id].append((i, model_name, input_data))

        # 並列実行
        tasks = []
        for device_id, group_requests in device_groups.items():
            for idx, model_name, input_data in group_requests:
                task = asyncio.create_task(self.predict_gpu(model_name, input_data))
                tasks.append((idx, task))

        # 結果収集
        results = [None] * len(requests)
        for idx, task in tasks:
            try:
                results[idx] = await task
            except Exception as e:
                logger.error(f"マルチGPU推論エラー (インデックス {idx}): {e}")
                # フォールバック結果
                model_name, input_data = requests[idx]
                results[idx] = GPUInferenceResult(
                    predictions=np.zeros((input_data.shape[0], 1)),
                    execution_time_us=0,
                    batch_size=input_data.shape[0],
                    device_id=-1,
                    backend_used=GPUBackend.CPU_FALLBACK,
                    model_name=model_name,
                    input_shape=input_data.shape,
                )

        return results

    async def benchmark_gpu_performance(
        self, model_name: str, test_data: np.ndarray, iterations: int = 100
    ) -> Dict[str, Any]:
        """GPU推論ベンチマーク"""
        logger.info(f"GPU推論ベンチマーク開始: {model_name}")

        # ウォームアップ
        for _ in range(5):
            await self.predict_gpu(model_name, test_data, use_cache=False)

        # ベンチマーク実行
        times = []
        for _ in range(iterations):
            result = await self.predict_gpu(model_name, test_data, use_cache=False)
            times.append(result.execution_time_us)

        times_array = np.array(times)
        return {
            "model_name": model_name,
            "iterations": iterations,
            "avg_time_us": np.mean(times_array),
            "throughput_per_sec": 1_000_000 / np.mean(times_array),
        }

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """総合統計取得"""
        stats = self.engine_stats.copy()
        stats["devices"] = self.device_manager.available_devices
        stats["sessions"] = {name: session.get_session_stats() for name, session in self.sessions.items()}
        stats["active_streams"] = len(self.stream_manager.active_streams)
        if self.cache_manager:
            stats["cache_stats"] = self.cache_manager.get_comprehensive_stats()
        return stats

    def cleanup(self):
        """リソースクリーンアップ"""
        try:
            # ストリーム同期
            self.stream_manager.synchronize_all_streams()

            # バッチプロセッサークリーンアップ
            self.batch_processor.cleanup()

            # ストリームマネージャークリーンアップ
            self.stream_manager.cleanup()

            # セッション削除
            for session in self.sessions.values():
                session.cleanup()
            self.sessions.clear()

            # キャッシュクリア
            if self.cache_manager:
                self.cache_manager.clear_all()

            logger.info("GPU推論エンジン クリーンアップ完了")

        except Exception as e:
            logger.error(f"GPU推論エンジン クリーンアップエラー: {e}")


# エクスポート用ファクトリ関数
async def create_gpu_inference_engine(
    backend: GPUBackend = GPUBackend.CUDA,
    device_ids: List[int] = None,
    memory_pool_size_mb: int = 2048,
    enable_dynamic_batching: bool = True,
) -> GPUAcceleratedInferenceEngine:
    """GPU加速推論エンジン作成"""
    config = GPUInferenceConfig(
        backend=backend,
        device_ids=device_ids or [0],
        memory_pool_size_mb=memory_pool_size_mb,
        dynamic_batching=enable_dynamic_batching,
    )

    engine = GPUAcceleratedInferenceEngine(config)
    return engine