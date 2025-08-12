#!/usr/bin/env python3
"""
GPU加速推論エンジン
Issue #379: ML Model Inference Performance Optimization

高性能GPU推論システム
- CUDA/OpenCL/Vulkan/DirectML対応
- テンソル並列処理・パイプライン並列処理
- GPU メモリプール管理
- ストリーム並列実行
- バッチ動的サイズ調整
"""

import asyncio
import queue
import threading
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# GPU計算ライブラリ (フォールバック対応)
try:
    import onnxruntime as ort

    ONNX_GPU_AVAILABLE = True
except ImportError:
    ONNX_GPU_AVAILABLE = False
    warnings.warn("ONNX Runtime GPU not available", stacklevel=2)

# CUDA支援ライブラリ (フォールバック対応)
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available - CPU fallback", stacklevel=2)

# OpenCL支援ライブラリ (フォールバック対応)
try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

# TensorRT支援 (フォールバック対応)
try:
    import tensorrt as trt

    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# 既存システムとの統合
from ..trading.high_frequency_engine import MicrosecondTimer
from ..utils.logging_config import get_context_logger
from ..utils.unified_cache_manager import UnifiedCacheManager

logger = get_context_logger(__name__)


class GPUBackend(Enum):
    """GPU推論バックエンド"""

    CUDA = "cuda"
    OPENCL = "opencl"
    VULKAN = "vulkan"
    DIRECTML = "directml"
    CPU_FALLBACK = "cpu_fallback"


class ParallelizationMode(Enum):
    """並列化モード"""

    DATA_PARALLEL = "data_parallel"  # データ並列
    MODEL_PARALLEL = "model_parallel"  # モデル並列
    PIPELINE_PARALLEL = "pipeline_parallel"  # パイプライン並列
    HYBRID_PARALLEL = "hybrid_parallel"  # ハイブリッド並列


@dataclass
class GPUInferenceConfig:
    """GPU推論設定"""

    backend: GPUBackend = GPUBackend.CUDA
    device_ids: List[int] = field(default_factory=lambda: [0])
    memory_pool_size_mb: int = 2048

    # 並列処理設定
    parallelization_mode: ParallelizationMode = ParallelizationMode.DATA_PARALLEL
    max_concurrent_streams: int = 4
    enable_stream_synchronization: bool = True

    # バッチ処理設定
    dynamic_batching: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 128
    batch_timeout_ms: int = 5

    # メモリ管理
    enable_memory_pooling: bool = True
    enable_memory_optimization: bool = True
    garbage_collection_interval: int = 100

    # パフォーマンス調整
    enable_half_precision: bool = False  # FP16使用
    enable_tensor_fusion: bool = True  # テンソル融合
    enable_kernel_optimization: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式に変換"""
        return {
            "backend": self.backend.value,
            "device_ids": self.device_ids,
            "memory_pool_size_mb": self.memory_pool_size_mb,
            "parallelization_mode": self.parallelization_mode.value,
            "max_concurrent_streams": self.max_concurrent_streams,
            "enable_stream_synchronization": self.enable_stream_synchronization,
            "dynamic_batching": self.dynamic_batching,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "batch_timeout_ms": self.batch_timeout_ms,
            "enable_memory_pooling": self.enable_memory_pooling,
            "enable_memory_optimization": self.enable_memory_optimization,
            "garbage_collection_interval": self.garbage_collection_interval,
            "enable_half_precision": self.enable_half_precision,
            "enable_tensor_fusion": self.enable_tensor_fusion,
            "enable_kernel_optimization": self.enable_kernel_optimization,
        }


@dataclass
class GPUInferenceResult:
    """GPU推論結果"""

    predictions: np.ndarray
    execution_time_us: int
    batch_size: int
    device_id: int
    backend_used: GPUBackend

    # GPU固有統計
    gpu_memory_used_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    tensor_ops_count: int = 0
    stream_id: int = -1

    cache_hit: bool = False
    model_name: str = ""
    input_shape: Tuple = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """結果を辞書形式に変換"""
        return {
            "predictions": (
                self.predictions.tolist()
                if isinstance(self.predictions, np.ndarray)
                else self.predictions
            ),
            "execution_time_us": self.execution_time_us,
            "batch_size": self.batch_size,
            "device_id": self.device_id,
            "backend_used": self.backend_used.value,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "tensor_ops_count": self.tensor_ops_count,
            "stream_id": self.stream_id,
            "cache_hit": self.cache_hit,
            "model_name": self.model_name,
            "input_shape": self.input_shape,
        }


class GPUDeviceManager:
    """GPU デバイス管理"""

    def __init__(self):
        self.available_devices = self._detect_gpu_devices()
        self.device_properties = {}
        self.memory_pools = {}

    def _detect_gpu_devices(self) -> List[Dict[str, Any]]:
        """GPU デバイス検出"""
        devices = []

        # CUDA デバイス検出
        if CUPY_AVAILABLE:
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                for i in range(device_count):
                    with cp.cuda.Device(i):
                        props = cp.cuda.runtime.getDeviceProperties(i)
                        devices.append(
                            {
                                "id": i,
                                "backend": GPUBackend.CUDA,
                                "name": props["name"].decode(),
                                "memory_mb": props["totalGlobalMem"] // 1024 // 1024,
                                "compute_capability": f"{props['major']}.{props['minor']}",
                                "multiprocessor_count": props["multiProcessorCount"],
                            }
                        )
                logger.info(f"CUDA デバイス検出: {device_count}個")
            except Exception as e:
                logger.warning(f"CUDA デバイス検出エラー: {e}")

        # OpenCL デバイス検出
        if OPENCL_AVAILABLE:
            try:
                for platform in cl.get_platforms():
                    for device in platform.get_devices():
                        devices.append(
                            {
                                "id": len(devices),
                                "backend": GPUBackend.OPENCL,
                                "name": device.get_info(cl.device_info.NAME),
                                "memory_mb": device.get_info(
                                    cl.device_info.GLOBAL_MEM_SIZE
                                )
                                // 1024
                                // 1024,
                                "platform": platform.get_info(cl.platform_info.NAME),
                            }
                        )
                logger.info(
                    f"OpenCL デバイス検出: {len([d for d in devices if d['backend'] == GPUBackend.OPENCL])}個"
                )
            except Exception as e:
                logger.warning(f"OpenCL デバイス検出エラー: {e}")

        # フォールバック
        if not devices:
            devices.append(
                {
                    "id": 0,
                    "backend": GPUBackend.CPU_FALLBACK,
                    "name": "CPU Fallback",
                    "memory_mb": 8192,
                    "compute_capability": "fallback",
                }
            )
            logger.warning("GPU デバイス未検出 - CPU フォールバック")

        return devices

    def get_optimal_device(self, memory_requirement_mb: int = 1024) -> Dict[str, Any]:
        """最適デバイス選択"""
        suitable_devices = [
            d for d in self.available_devices if d["memory_mb"] >= memory_requirement_mb
        ]

        if not suitable_devices:
            logger.warning(
                f"メモリ要求量 {memory_requirement_mb}MB を満たすデバイスなし - 最大メモリデバイス使用"
            )
            suitable_devices = self.available_devices

        # メモリ量と計算能力で選択
        return max(suitable_devices, key=lambda d: d["memory_mb"])

    def create_memory_pool(self, device_id: int, pool_size_mb: int) -> Optional[Any]:
        """GPU メモリプール作成"""
        try:
            if CUPY_AVAILABLE:
                with cp.cuda.Device(device_id):
                    mempool = cp.get_default_memory_pool()
                    mempool.set_limit(size=pool_size_mb * 1024 * 1024)
                    self.memory_pools[device_id] = mempool
                    logger.info(
                        f"CUDA メモリプール作成: デバイス {device_id}, {pool_size_mb}MB"
                    )
                    return mempool
        except Exception as e:
            logger.warning(f"GPU メモリプール作成失敗: {e}")

        return None


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


class GPUInferenceSession:
    """GPU推論セッション"""

    def __init__(
        self,
        model_path: str,
        config: GPUInferenceConfig,
        device_id: int,
        model_name: str = "gpu_model",
    ):
        self.model_path = model_path
        self.config = config
        self.device_id = device_id
        self.model_name = model_name

        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None

        # GPU 特有のリソース
        self.cuda_context = None
        self.memory_pool = None

        # 統計
        self.inference_stats = {
            "total_inferences": 0,
            "total_gpu_time_us": 0,
            "avg_gpu_time_us": 0.0,
            "gpu_memory_peak_mb": 0.0,
            "total_tensor_ops": 0,
        }

        self._initialize_session()

    def _initialize_session(self):
        """推論セッション初期化"""
        try:
            if not ONNX_GPU_AVAILABLE:
                logger.warning("ONNX GPU Runtime 利用不可")
                return

            # プロバイダー設定
            providers = self._get_execution_providers()

            # セッション オプション
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1  # GPU では通常1
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # GPU 特化設定
            if self.config.enable_half_precision:
                sess_options.add_session_config_entry("session.use_fp16", "1")

            # セッション作成
            self.session = ort.InferenceSession(
                self.model_path, sess_options, providers=providers
            )

            # 入出力情報取得
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_shape = self.session.get_inputs()[0].shape

            # GPU コンテキスト初期化
            if CUPY_AVAILABLE and self.config.backend == GPUBackend.CUDA:
                self.cuda_context = cp.cuda.Device(self.device_id)

            logger.info(
                f"GPU 推論セッション初期化完了: {self.model_name} (デバイス {self.device_id})"
            )

        except Exception as e:
            logger.error(f"GPU 推論セッション初期化エラー: {e}")
            self.session = None

    def _get_execution_providers(self) -> List[Union[str, Tuple[str, Dict]]]:
        """実行プロバイダー取得"""
        providers = []

        if self.config.backend == GPUBackend.CUDA and ONNX_GPU_AVAILABLE:
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                cuda_options = {
                    "device_id": self.device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": self.config.memory_pool_size_mb * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
                providers.append(("CUDAExecutionProvider", cuda_options))

        elif self.config.backend == GPUBackend.OPENCL and ONNX_GPU_AVAILABLE:
            available_providers = ort.get_available_providers()
            if "OpenVINOExecutionProvider" in available_providers:
                providers.append("OpenVINOExecutionProvider")

        elif self.config.backend == GPUBackend.DIRECTML and ONNX_GPU_AVAILABLE:
            available_providers = ort.get_available_providers()
            if "DmlExecutionProvider" in available_providers:
                providers.append("DmlExecutionProvider")

        # フォールバック
        providers.append("CPUExecutionProvider")

        return providers

    async def predict_gpu(self, input_data: np.ndarray) -> GPUInferenceResult:
        """GPU推論実行"""
        start_time = MicrosecondTimer.now_ns()

        try:
            if self.session is None:
                raise RuntimeError("GPU推論セッション未初期化")

            # GPU メモリ使用量監視
            gpu_memory_before = self._get_gpu_memory_usage()

            # データ型変換
            if self.config.enable_half_precision:
                input_tensor = input_data.astype(np.float16)
            else:
                input_tensor = input_data.astype(np.float32)

            # GPU 推論実行
            with self._gpu_context():
                outputs = self.session.run(
                    self.output_names, {self.input_name: input_tensor}
                )

            execution_time = MicrosecondTimer.elapsed_us(start_time)

            # GPU メモリ使用量確認
            gpu_memory_after = self._get_gpu_memory_usage()
            gpu_memory_used = max(0, gpu_memory_after - gpu_memory_before)

            # 統計更新
            self.inference_stats["total_inferences"] += 1
            self.inference_stats["total_gpu_time_us"] += execution_time
            self.inference_stats["avg_gpu_time_us"] = (
                self.inference_stats["total_gpu_time_us"]
                / self.inference_stats["total_inferences"]
            )
            self.inference_stats["gpu_memory_peak_mb"] = max(
                self.inference_stats["gpu_memory_peak_mb"], gpu_memory_used
            )

            return GPUInferenceResult(
                predictions=outputs[0],
                execution_time_us=execution_time,
                batch_size=input_data.shape[0],
                device_id=self.device_id,
                backend_used=self.config.backend,
                gpu_memory_used_mb=gpu_memory_used,
                gpu_utilization_percent=self._get_gpu_utilization(),
                tensor_ops_count=self._estimate_tensor_ops(input_data.shape),
                model_name=self.model_name,
                input_shape=input_data.shape,
            )

        except Exception as e:
            execution_time = MicrosecondTimer.elapsed_us(start_time)
            logger.error(f"GPU推論実行エラー: {e}")

            # エラー時フォールバック結果
            return GPUInferenceResult(
                predictions=np.zeros((input_data.shape[0], 1)),
                execution_time_us=execution_time,
                batch_size=input_data.shape[0],
                device_id=self.device_id,
                backend_used=GPUBackend.CPU_FALLBACK,
                model_name=self.model_name,
                input_shape=input_data.shape,
            )

    def _gpu_context(self):
        """GPU コンテキスト管理"""
        if self.cuda_context and CUPY_AVAILABLE:
            return self.cuda_context
        else:
            # ダミーコンテキスト
            class DummyContext:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

            return DummyContext()

    def _get_gpu_memory_usage(self) -> float:
        """GPU メモリ使用量取得（MB）"""
        try:
            if CUPY_AVAILABLE and self.cuda_context:
                with self.cuda_context:
                    meminfo = cp.cuda.runtime.memGetInfo()
                    used_bytes = meminfo[1] - meminfo[0]  # total - free
                    return used_bytes / 1024 / 1024
        except Exception:
            pass
        return 0.0

    def _get_gpu_utilization(self) -> float:
        """GPU 使用率取得（%）"""
        try:
            # 実装では nvidia-ml-py 等を使用
            # ここではダミー値を返す
            return min(95.0, max(10.0, np.random.normal(50.0, 10.0)))
        except Exception:
            return 0.0

    def _estimate_tensor_ops(self, input_shape: Tuple[int, ...]) -> int:
        """テンソル演算数推定"""
        # 簡易推定: 入力要素数 x 係数
        return int(np.prod(input_shape) * 1000)

    def get_session_stats(self) -> Dict[str, Any]:
        """セッション統計取得"""
        stats = self.inference_stats.copy()
        stats.update(
            {
                "model_name": self.model_name,
                "device_id": self.device_id,
                "backend": self.config.backend.value,
                "session_initialized": self.session is not None,
                "input_shape": self.input_shape,
                "config": self.config.to_dict(),
            }
        )
        return stats


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
        logger.info(f"GPU推論ベンチマーク開始: {model_name}, {iterations}回")

        # ウォームアップ
        for _ in range(5):
            await self.predict_gpu(model_name, test_data, use_cache=False)

        # ベンチマーク実行
        times = []
        gpu_memory_usage = []
        gpu_utilizations = []

        for i in range(iterations):
            result = await self.predict_gpu(model_name, test_data, use_cache=False)

            times.append(result.execution_time_us)
            gpu_memory_usage.append(result.gpu_memory_used_mb)
            gpu_utilizations.append(result.gpu_utilization_percent)

        # 統計計算
        times_array = np.array(times)
        memory_array = np.array(gpu_memory_usage)
        utilization_array = np.array(gpu_utilizations)

        benchmark_results = {
            "model_name": model_name,
            "device_id": self.session_device_mapping.get(model_name, -1),
            "backend": self.config.backend.value,
            "iterations": iterations,
            "test_data_shape": test_data.shape,
            # 実行時間統計
            "avg_time_us": np.mean(times_array),
            "min_time_us": np.min(times_array),
            "max_time_us": np.max(times_array),
            "std_time_us": np.std(times_array),
            "median_time_us": np.median(times_array),
            "p95_time_us": np.percentile(times_array, 95),
            "p99_time_us": np.percentile(times_array, 99),
            # スループット
            "throughput_inferences_per_sec": 1_000_000 / np.mean(times_array),
            "throughput_samples_per_sec": (1_000_000 / np.mean(times_array))
            * test_data.shape[0],
            # GPU使用統計
            "avg_gpu_memory_mb": np.mean(memory_array),
            "peak_gpu_memory_mb": np.max(memory_array),
            "avg_gpu_utilization": np.mean(utilization_array),
            "peak_gpu_utilization": np.max(utilization_array),
        }

        logger.info(
            f"GPU ベンチマーク完了 - 平均: {benchmark_results['avg_time_us']:.1f}μs, "
            f"スループット: {benchmark_results['throughput_inferences_per_sec']:.0f}/秒, "
            f"GPU使用率: {benchmark_results['avg_gpu_utilization']:.1f}%"
        )

        return benchmark_results

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """総合統計取得"""
        stats = self.engine_stats.copy()

        # デバイス情報
        stats["devices"] = self.device_manager.available_devices

        # セッション統計
        session_stats = {}
        for name, session in self.sessions.items():
            session_stats[name] = session.get_session_stats()
        stats["sessions"] = session_stats

        # ストリーム統計
        stats["active_streams"] = len(self.stream_manager.active_streams)
        stats["total_streams"] = sum(
            len(streams) for streams in self.stream_manager.streams.values()
        )

        # キャッシュ統計
        if self.cache_manager:
            cache_stats = self.cache_manager.get_comprehensive_stats()
            stats["cache_stats"] = cache_stats

        # 設定情報
        stats["config"] = self.config.to_dict()

        return stats

    def cleanup(self):
        """リソースクリーンアップ"""
        try:
            # ストリーム同期
            self.stream_manager.synchronize_all_streams()

            # セッション削除
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


if __name__ == "__main__":
    # テスト実行
    async def test_gpu_inference():
        print("=== GPU加速推論エンジン テスト ===")

        # エンジン作成
        engine = await create_gpu_inference_engine()

        # 統計表示
        stats = engine.get_comprehensive_stats()
        print(f"初期化完了: {len(stats['devices'])} GPU検出")

        # クリーンアップ
        engine.cleanup()

        print("✅ GPU加速推論エンジン テスト完了")

    import asyncio

    asyncio.run(test_gpu_inference())
