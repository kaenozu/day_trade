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
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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

# 既存システムとの統合
from ..trading.high_frequency_engine import MicrosecondTimer
from ..utils.logging_config import get_context_logger
from ..utils.unified_cache_manager import UnifiedCacheManager

logger = get_context_logger(__name__)

# TensorRT支援 (フォールバック対応)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # 自動初期化

    TENSORRT_AVAILABLE = True
    PYCUDA_AVAILABLE = True

    # TensorRTバージョン確認
    trt_version = f"{trt.__version__}"
    logger.info(f"TensorRT利用可能 - バージョン: {trt_version}")
except ImportError as e:
    TENSORRT_AVAILABLE = False
    PYCUDA_AVAILABLE = False
    logger.info(f"TensorRT利用不可: {e}")

# Issue #720対応: GPU監視ライブラリ (フォールバック対応)
try:
    import pynvml

    PYNVML_AVAILABLE = True
    # NVML初期化
    try:
        pynvml.nvmlInit()
        logger.info("NVIDIA Management Library (pynvml) 初期化成功")
    except Exception as e:
        PYNVML_AVAILABLE = False
        logger.warning(f"NVML初期化失敗: {e}")
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available - GPU監視機能制限")


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

    # Issue #720対応: GPU監視設定
    enable_realtime_monitoring: bool = True  # リアルタイムGPU監視
    monitoring_interval_ms: int = 100  # 監視間隔（ミリ秒）
    gpu_utilization_threshold: float = 90.0  # GPU使用率警告閾値（%）
    gpu_memory_threshold: float = 90.0  # GPUメモリ使用率警告閾値（%）
    temperature_threshold: float = 80.0  # GPU温度警告閾値（℃）
    power_threshold: float = 250.0  # GPU電力消費警告閾値（W）

    # Issue #721対応: TensorRT設定
    enable_tensorrt: bool = True  # TensorRT使用有効化
    tensorrt_precision: str = "fp16"  # "fp32", "fp16", "int8"
    tensorrt_max_workspace_size: int = 1024  # ワークスペースサイズ（MB）
    tensorrt_max_batch_size: int = 32  # 最大バッチサイズ
    tensorrt_enable_dla: bool = False  # DLA (Deep Learning Accelerator) 使用
    tensorrt_dla_core: int = -1  # DLAコア指定（-1で自動選択）
    tensorrt_optimization_level: int = 3  # 最適化レベル (0-5)
    tensorrt_enable_timing_cache: bool = True  # タイミングキャッシュ

    # Issue #722対応: CPUフォールバック最適化設定
    cpu_threads: int = 0  # CPU推論スレッド数（0で自動選択）
    enable_cpu_optimizations: bool = True  # CPU最適化有効化
    cpu_memory_arena_mb: int = 512  # CPUメモリアリーナサイズ（MB）
    enable_cpu_vectorization: bool = True  # CPU ベクトル化最適化
    cpu_execution_mode: str = "sequential"  # "sequential" or "parallel"

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
            # Issue #720対応: GPU監視設定
            "enable_realtime_monitoring": self.enable_realtime_monitoring,
            "monitoring_interval_ms": self.monitoring_interval_ms,
            "gpu_utilization_threshold": self.gpu_utilization_threshold,
            "gpu_memory_threshold": self.gpu_memory_threshold,
            "temperature_threshold": self.temperature_threshold,
            "power_threshold": self.power_threshold,
            # Issue #721対応: TensorRT設定
            "enable_tensorrt": self.enable_tensorrt,
            "tensorrt_precision": self.tensorrt_precision,
            "tensorrt_max_workspace_size": self.tensorrt_max_workspace_size,
            "tensorrt_max_batch_size": self.tensorrt_max_batch_size,
            "tensorrt_enable_dla": self.tensorrt_enable_dla,
            "tensorrt_dla_core": self.tensorrt_dla_core,
            "tensorrt_optimization_level": self.tensorrt_optimization_level,
            "tensorrt_enable_timing_cache": self.tensorrt_enable_timing_cache,
            # Issue #722対応: CPUフォールバック最適化設定
            "cpu_threads": self.cpu_threads,
            "enable_cpu_optimizations": self.enable_cpu_optimizations,
            "cpu_memory_arena_mb": self.cpu_memory_arena_mb,
            "enable_cpu_vectorization": self.enable_cpu_vectorization,
            "cpu_execution_mode": self.cpu_execution_mode,
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


# Issue #721対応: TensorRTエンジンクラス
class TensorRTEngine:
    """TensorRT推論エンジン"""

    def __init__(self, config: GPUInferenceConfig, device_id: int = 0):
        self.config = config
        self.device_id = device_id
        self.engine = None
        self.context = None
        self.bindings = None
        self.stream = None

        # メモリ管理
        self.inputs = []
        self.outputs = []
        self.allocations = []

        # TensorRT設定
        self.logger = trt.Logger(trt.Logger.INFO)
        self.builder = None
        self.network = None

    def build_engine_from_onnx(self, onnx_model_path: str) -> bool:
        """ONNXモデルからTensorRTエンジンを構築"""
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT利用不可 - エンジン構築スキップ")
            return False

        try:
            # TensorRTビルダー作成
            self.builder = trt.Builder(self.logger)
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            self.network = self.builder.create_network(network_flags)

            # ONNX パーサー
            parser = trt.OnnxParser(self.network, self.logger)

            # ONNXファイル読み込み
            with open(onnx_model_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("ONNX モデル解析失敗")
                    for error in range(parser.num_errors):
                        logger.error(f"  エラー {error}: {parser.get_error(error)}")
                    return False

            # ビルダー設定
            builder_config = self.builder.create_builder_config()

            # ワークスペースサイズ設定
            builder_config.max_workspace_size = self.config.tensorrt_max_workspace_size * (1024 ** 2)

            # 精度設定
            if self.config.tensorrt_precision == "fp16":
                builder_config.set_flag(trt.BuilderFlag.FP16)
                logger.info("TensorRT FP16精度有効化")
            elif self.config.tensorrt_precision == "int8":
                builder_config.set_flag(trt.BuilderFlag.INT8)
                logger.info("TensorRT INT8精度有効化")

            # DLA設定（Jetson等でのみ有効）
            if self.config.tensorrt_enable_dla and self.config.tensorrt_dla_core >= 0:
                builder_config.default_device_type = trt.DeviceType.DLA
                builder_config.DLA_core = self.config.tensorrt_dla_core
                builder_config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
                logger.info(f"TensorRT DLA有効化: コア {self.config.tensorrt_dla_core}")

            # タイミングキャッシュ
            if self.config.tensorrt_enable_timing_cache:
                cache = builder_config.create_timing_cache(b"")
                builder_config.set_timing_cache(cache, ignore_mismatch=False)
                logger.info("TensorRT タイミングキャッシュ有効化")

            # 最適化プロファイル設定（動的バッチサイズ対応）
            profile = self.builder.create_optimization_profile()

            # 入力テンソルのプロファイル設定
            for i in range(self.network.num_inputs):
                input_tensor = self.network.get_input(i)
                input_shape = input_tensor.shape

                # 動的バッチサイズの設定
                min_shape = list(input_shape)
                opt_shape = list(input_shape)
                max_shape = list(input_shape)

                if min_shape[0] == -1:  # バッチ次元が動的
                    min_shape[0] = 1
                    opt_shape[0] = self.config.tensorrt_max_batch_size // 2
                    max_shape[0] = self.config.tensorrt_max_batch_size

                profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
                logger.debug(f"入力プロファイル {input_tensor.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

            builder_config.add_optimization_profile(profile)

            # エンジン構築
            logger.info("TensorRTエンジン構築開始...")
            start_time = time.time()

            self.engine = self.builder.build_engine(self.network, builder_config)

            build_time = time.time() - start_time

            if self.engine is None:
                logger.error("TensorRTエンジン構築失敗")
                return False

            logger.info(f"TensorRTエンジン構築完了: {build_time:.2f}秒")

            # 実行コンテキスト作成
            self.context = self.engine.create_execution_context()

            # メモリ割り当て準備
            self._prepare_memory_allocations()

            return True

        except Exception as e:
            logger.error(f"TensorRTエンジン構築エラー: {e}")
            return False

    def _prepare_memory_allocations(self):
        """メモリ割り当て準備"""
        if not PYCUDA_AVAILABLE or self.engine is None:
            return

        try:
            # CUDAストリーム作成
            self.stream = cuda.Stream()

            # 入力・出力テンソル情報取得
            self.inputs = []
            self.outputs = []
            self.bindings = []
            self.allocations = []

            for binding in self.engine:
                binding_idx = self.engine.get_binding_index(binding)
                size = trt.volume(self.context.get_binding_shape(binding_idx))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))

                # GPUメモリ割り当て
                device_mem = cuda.mem_alloc(size * dtype().itemsize)
                self.allocations.append(device_mem)
                self.bindings.append(int(device_mem))

                if self.engine.binding_is_input(binding):
                    self.inputs.append({
                        'name': binding,
                        'index': binding_idx,
                        'size': size,
                        'dtype': dtype,
                        'device_mem': device_mem
                    })
                else:
                    self.outputs.append({
                        'name': binding,
                        'index': binding_idx,
                        'size': size,
                        'dtype': dtype,
                        'device_mem': device_mem
                    })

            logger.info(f"TensorRTメモリ割り当て完了: 入力={len(self.inputs)}, 出力={len(self.outputs)}")

        except Exception as e:
            logger.error(f"TensorRTメモリ割り当てエラー: {e}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """TensorRT推論実行"""
        if not PYCUDA_AVAILABLE or self.engine is None or self.context is None:
            raise RuntimeError("TensorRT エンジン未初期化")

        try:
            # 入力データの前処理
            batch_size = input_data.shape[0]

            # バッチサイズに応じて動的形状設定
            if len(self.inputs) > 0:
                input_binding = self.inputs[0]
                input_shape = list(input_data.shape)

                if not self.context.set_binding_shape(input_binding['index'], input_shape):
                    raise RuntimeError(f"入力形状設定失敗: {input_shape}")

            # 入力データをGPUメモリにコピー
            for i, input_info in enumerate(self.inputs):
                host_mem = np.ascontiguousarray(input_data.astype(input_info['dtype']))
                cuda.memcpy_htod_async(input_info['device_mem'], host_mem, self.stream)

            # 推論実行
            success = self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )

            if not success:
                raise RuntimeError("TensorRT推論実行失敗")

            # 出力データをCPUメモリにコピー
            outputs = []
            for output_info in self.outputs:
                # 出力形状取得
                output_shape = self.context.get_binding_shape(output_info['index'])
                output_size = trt.volume(output_shape)

                # CPUメモリ準備
                host_mem = np.empty(output_size, dtype=output_info['dtype'])

                # GPUからCPUへコピー
                cuda.memcpy_dtoh_async(host_mem, output_info['device_mem'], self.stream)

                # 結果をリシェイプ
                host_mem = host_mem.reshape(output_shape)
                outputs.append(host_mem)

            # ストリーム同期
            self.stream.synchronize()

            # 結果返却（複数出力の場合は最初の出力）
            return outputs[0] if len(outputs) > 0 else np.array([])

        except Exception as e:
            logger.error(f"TensorRT推論エラー: {e}")
            raise

    def save_engine(self, engine_path: str) -> bool:
        """TensorRTエンジンをファイルに保存"""
        if self.engine is None:
            return False

        try:
            with open(engine_path, 'wb') as f:
                f.write(self.engine.serialize())

            logger.info(f"TensorRTエンジン保存完了: {engine_path}")
            return True

        except Exception as e:
            logger.error(f"TensorRTエンジン保存エラー: {e}")
            return False

    def load_engine(self, engine_path: str) -> bool:
        """保存されたTensorRTエンジンを読み込み"""
        if not TENSORRT_AVAILABLE:
            return False

        try:
            runtime = trt.Runtime(self.logger)

            with open(engine_path, 'rb') as f:
                engine_data = f.read()

            self.engine = runtime.deserialize_cuda_engine(engine_data)

            if self.engine is None:
                logger.error("TensorRTエンジン読み込み失敗")
                return False

            self.context = self.engine.create_execution_context()
            self._prepare_memory_allocations()

            logger.info(f"TensorRTエンジン読み込み完了: {engine_path}")
            return True

        except Exception as e:
            logger.error(f"TensorRTエンジン読み込みエラー: {e}")
            return False

    def cleanup(self):
        """リソースクリーンアップ"""
        try:
            # GPU メモリ解放
            for allocation in self.allocations:
                if allocation:
                    allocation.free()
            self.allocations.clear()

            # ストリーム削除
            if self.stream:
                del self.stream
                self.stream = None

            # コンテキスト削除
            if self.context:
                del self.context
                self.context = None

            # エンジン削除
            if self.engine:
                del self.engine
                self.engine = None

            logger.debug("TensorRTエンジンクリーンアップ完了")

        except Exception as e:
            logger.error(f"TensorRTエンジンクリーンアップエラー: {e}")


# Issue #720対応: GPU監視データ構造
@dataclass
class GPUMonitoringData:
    """GPU監視データ"""

    device_id: int
    timestamp: float

    # GPU使用率統計
    gpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0

    # メモリ使用量（MB）
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_free_mb: float = 0.0

    # 温度・電力
    temperature_celsius: float = 0.0
    power_consumption_watts: float = 0.0

    # プロセス情報
    running_processes: int = 0
    compute_mode: str = "Default"

    # エラー状態
    has_errors: bool = False
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "memory_utilization_percent": self.memory_utilization_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "memory_free_mb": self.memory_free_mb,
            "temperature_celsius": self.temperature_celsius,
            "power_consumption_watts": self.power_consumption_watts,
            "running_processes": self.running_processes,
            "compute_mode": self.compute_mode,
            "has_errors": self.has_errors,
            "error_message": self.error_message,
        }

    @property
    def is_overloaded(self) -> bool:
        """GPU過負荷状態の判定"""
        return (
            self.gpu_utilization_percent > 95.0
            or self.memory_utilization_percent > 95.0
            or self.temperature_celsius > 85.0
        )

    @property
    def is_healthy(self) -> bool:
        """GPU健全性の判定"""
        return (
            not self.has_errors
            and self.gpu_utilization_percent < 90.0
            and self.memory_utilization_percent < 90.0
            and self.temperature_celsius < 80.0
        )


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

        # Issue #721対応: TensorRT エンジン
        self.tensorrt_engine = None
        self.use_tensorrt = False

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

            # Issue #722対応: CPUフォールバック最適化設定
            if self.config.backend == GPUBackend.CPU_FALLBACK or not ONNX_GPU_AVAILABLE:
                # CPU最適化設定適用
                if self.config.enable_cpu_optimizations:
                    cpu_threads = self._get_optimal_cpu_threads()
                    sess_options.intra_op_num_threads = cpu_threads
                    sess_options.inter_op_num_threads = 1

                    # CPU実行モード設定
                    if self.config.cpu_execution_mode == "sequential":
                        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    else:
                        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

                    logger.info(f"CPU最適化設定適用: {cpu_threads}スレッド, {self.config.cpu_execution_mode}実行")
                else:
                    sess_options.intra_op_num_threads = 1
            else:
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

            # Issue #722対応: CPU最適化設定追加適用
            if self.config.backend == GPUBackend.CPU_FALLBACK or not ONNX_GPU_AVAILABLE:
                self._enable_cpu_optimized_inference(self.session)

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

        # Issue #721対応: TensorRT初期化を試行
        self._try_initialize_tensorrt()

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
                # Issue #722対応: OpenVINO最適化設定
                openvino_options = {
                    "device_type": "CPU",
                    "precision": "FP32",
                    "num_of_threads": self._get_optimal_cpu_threads(),
                }
                providers.append(("OpenVINOExecutionProvider", openvino_options))
            else:
                providers.append("OpenVINOExecutionProvider")

        elif self.config.backend == GPUBackend.DIRECTML and ONNX_GPU_AVAILABLE:
            available_providers = ort.get_available_providers()
            if "DmlExecutionProvider" in available_providers:
                providers.append("DmlExecutionProvider")

        # Issue #722対応: CPUフォールバック最適化
        cpu_options = self._get_optimized_cpu_options()
        providers.append(("CPUExecutionProvider", cpu_options))

        return providers

    # Issue #722対応: CPUフォールバック最適化メソッド

    def _get_optimal_cpu_threads(self) -> int:
        """最適CPU スレッド数取得"""
        import os

        # システムCPU数取得
        cpu_count = os.cpu_count() or 1

        # メモリベースの調整
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

            # メモリ量に基づく制限
            if available_memory_gb < 4:
                max_threads = min(2, cpu_count)
            elif available_memory_gb < 8:
                max_threads = min(4, cpu_count)
            else:
                max_threads = min(cpu_count, 8)  # 最大8スレッド

        except ImportError:
            # psutil利用不可時は控えめに設定
            max_threads = min(4, cpu_count)

        # 設定値があれば考慮
        if hasattr(self.config, 'cpu_threads') and self.config.cpu_threads > 0:
            max_threads = min(self.config.cpu_threads, max_threads)

        logger.debug(f"最適CPU スレッド数: {max_threads} (システム: {cpu_count})")
        return max_threads

    def _get_optimized_cpu_options(self) -> Dict[str, Any]:
        """最適化CPU実行プロバイダーオプション"""
        options = {
            # スレッド数設定
            "intra_op_num_threads": self._get_optimal_cpu_threads(),
            "inter_op_num_threads": 1,  # モデル間並列は1に制限

            # メモリアロケーション最適化
            "arena_extend_strategy": "kSameAsRequested",

            # CPU最適化設定
            "enable_cpu_mem_arena": True,
            "enable_mem_pattern": True,
            "enable_mem_reuse": True,

            # SIMD/ベクトル化最適化
            "use_arena": True,
        }

        # CPU固有の高速化設定
        try:
            import platform
            if platform.machine().lower() in ['x86_64', 'amd64']:
                # x86_64 固有最適化
                options.update({
                    # AVX/AVX2/AVX512などの活用（ONNX Runtime自動選択）
                    "execution_mode": "ORT_SEQUENTIAL",  # 順次実行で一貫性確保
                })
        except Exception:
            pass

        logger.debug(f"CPU実行プロバイダーオプション: {options}")
        return options

    def _enable_cpu_optimized_inference(self, session) -> None:
        """CPU推論最適化設定適用"""
        try:
            # セッション統計情報有効化（パフォーマンス監視用）
            if hasattr(session, 'enable_profiling'):
                session.enable_profiling('cpu_profile.json')

            # CPU最適化ログ出力
            providers = session.get_providers()
            logger.info(f"CPU推論セッション初期化完了: プロバイダー {providers}")

            # CPU推論パフォーマンス情報収集準備
            self.cpu_inference_stats = {
                'total_inferences': 0,
                'total_time_ms': 0.0,
                'avg_time_ms': 0.0,
                'thread_count': self._get_optimal_cpu_threads(),
            }

        except Exception as e:
            logger.debug(f"CPU推論最適化設定適用エラー: {e}")

    def _update_cpu_performance_stats(self, inference_time_ms: float) -> None:
        """CPU推論パフォーマンス統計更新"""
        if hasattr(self, 'cpu_inference_stats'):
            stats = self.cpu_inference_stats
            stats['total_inferences'] += 1
            stats['total_time_ms'] += inference_time_ms
            stats['avg_time_ms'] = stats['total_time_ms'] / stats['total_inferences']

            # 100回毎にパフォーマンス状況をログ出力
            if stats['total_inferences'] % 100 == 0:
                logger.info(
                    f"CPU推論統計 (#{stats['total_inferences']}): "
                    f"平均 {stats['avg_time_ms']:.2f}ms, "
                    f"スレッド数 {stats['thread_count']}"
                )

    # Issue #721対応: TensorRT統合メソッド

    def _try_initialize_tensorrt(self):
        """TensorRT初期化を試行"""
        if (not self.config.enable_tensorrt or
            not TENSORRT_AVAILABLE or
            not self.model_path.endswith('.onnx')):
            logger.debug("TensorRT初期化スキップ - 条件不適合")
            return

        try:
            # TensorRTエンジン作成
            self.tensorrt_engine = TensorRTEngine(self.config, self.device_id)

            # エンジンファイル確認（キャッシュ）
            engine_path = self._get_tensorrt_engine_path()

            if engine_path.exists():
                logger.info(f"既存TensorRTエンジン読み込み: {engine_path}")
                if self.tensorrt_engine.load_engine(str(engine_path)):
                    self.use_tensorrt = True
                    logger.info("TensorRT推論有効化")
                    return

            # ONNXからエンジン構築
            logger.info("ONNXからTensorRTエンジン構築開始...")
            if self.tensorrt_engine.build_engine_from_onnx(self.model_path):
                # エンジンをキャッシュとして保存
                self.tensorrt_engine.save_engine(str(engine_path))
                self.use_tensorrt = True
                logger.info("TensorRT推論有効化")
            else:
                logger.warning("TensorRTエンジン構築失敗 - ONNX Runtime使用")

        except Exception as e:
            logger.warning(f"TensorRT初期化エラー: {e} - ONNX Runtime使用")
            self.tensorrt_engine = None

    def _get_tensorrt_engine_path(self) -> Path:
        """TensorRTエンジンファイルパス生成"""
        from pathlib import Path
        model_path = Path(self.model_path)

        # エンジンファイル名生成（モデル名+設定ハッシュ）
        config_str = (f"{self.config.tensorrt_precision}_"
                     f"{self.config.tensorrt_max_batch_size}_"
                     f"{self.config.tensorrt_max_workspace_size}_"
                     f"{self.config.tensorrt_optimization_level}")

        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        engine_name = f"{model_path.stem}_{config_hash}.trt"

        # キャッシュディレクトリ
        cache_dir = model_path.parent / "tensorrt_cache"
        cache_dir.mkdir(exist_ok=True)

        return cache_dir / engine_name

    async def predict_gpu(self, input_data: np.ndarray) -> GPUInferenceResult:
        """GPU推論実行"""
        start_time = MicrosecondTimer.now_ns()

        try:
            # GPU メモリ使用量監視
            gpu_memory_before = self._get_gpu_memory_usage()

            # Issue #721対応: TensorRT推論優先実行
            if self.use_tensorrt and self.tensorrt_engine:
                # データ型変換
                if self.config.enable_half_precision:
                    input_tensor = input_data.astype(np.float16)
                else:
                    input_tensor = input_data.astype(np.float32)

                # TensorRT推論実行
                outputs = [self.tensorrt_engine.predict(input_tensor)]
                backend_used = GPUBackend.CUDA  # TensorRTはCUDAベース
                logger.debug("TensorRT推論実行")
            else:
                if self.session is None:
                    raise RuntimeError("GPU推論セッション未初期化")

                # データ型変換
                if self.config.enable_half_precision:
                    input_tensor = input_data.astype(np.float16)
                else:
                    input_tensor = input_data.astype(np.float32)

                # ONNX Runtime GPU 推論実行
                with self._gpu_context():
                    inference_start = time.time()
                    outputs = self.session.run(
                        self.output_names, {self.input_name: input_tensor}
                    )
                    inference_time_ms = (time.time() - inference_start) * 1000

                # Issue #722対応: CPUフォールバック時の統計更新
                if (self.config.backend == GPUBackend.CPU_FALLBACK or
                    not ONNX_GPU_AVAILABLE or
                    'CPUExecutionProvider' in self.session.get_providers()):
                    self._update_cpu_performance_stats(inference_time_ms)

                backend_used = self.config.backend
                logger.debug("ONNX Runtime推論実行")

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
                backend_used=backend_used,  # Issue #721: TensorRT対応
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
            # Issue #720対応: 実際のGPU使用率取得
            if PYNVML_AVAILABLE:
                return self._get_gpu_utilization_nvml()
            else:
                # nvidia-smiコマンドフォールバック
                return self._get_gpu_utilization_nvidia_smi()
        except Exception as e:
            logger.warning(f"GPU使用率取得エラー: {e}")
            # フォールバックとしてダミー値を返す
            return min(95.0, max(10.0, np.random.normal(50.0, 10.0)))

    def _get_gpu_utilization_nvml(self) -> float:
        """NVML経由でのGPU使用率取得"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except Exception as e:
            logger.debug(f"NVML GPU使用率取得エラー: {e}")
            return 0.0

    def _get_gpu_utilization_nvidia_smi(self) -> float:
        """nvidia-smiコマンド経由でのGPU使用率取得"""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits',
                f'--id={self.device_id}'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                utilization_str = result.stdout.strip()
                return float(utilization_str)
            else:
                logger.debug(f"nvidia-smi エラー: {result.stderr}")
                return 0.0
        except Exception as e:
            logger.debug(f"nvidia-smi実行エラー: {e}")
            return 0.0

    # Issue #720対応: リアルタイムGPU監視機能

    def get_comprehensive_gpu_monitoring(self) -> GPUMonitoringData:
        """包括的なGPU監視データの取得"""
        monitoring_data = GPUMonitoringData(
            device_id=self.device_id,
            timestamp=time.time()
        )

        try:
            if PYNVML_AVAILABLE:
                self._populate_nvml_monitoring_data(monitoring_data)
            else:
                self._populate_nvidia_smi_monitoring_data(monitoring_data)
        except Exception as e:
            monitoring_data.has_errors = True
            monitoring_data.error_message = str(e)
            logger.warning(f"GPU監視データ取得エラー: {e}")

        return monitoring_data

    def _populate_nvml_monitoring_data(self, monitoring_data: GPUMonitoringData):
        """NVMLを使用して監視データを取得"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

            # GPU使用率
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            monitoring_data.gpu_utilization_percent = float(utilization.gpu)
            monitoring_data.memory_utilization_percent = float(utilization.memory)

            # メモリ情報
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            monitoring_data.memory_total_mb = memory_info.total / (1024 ** 2)
            monitoring_data.memory_used_mb = memory_info.used / (1024 ** 2)
            monitoring_data.memory_free_mb = memory_info.free / (1024 ** 2)

            # 温度情報
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                monitoring_data.temperature_celsius = float(temperature)
            except:
                pass

            # 電力消費
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                monitoring_data.power_consumption_watts = power / 1000.0  # mWから変換
            except:
                pass

            # 実行中プロセス数
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                monitoring_data.running_processes = len(processes)
            except:
                pass

            # コンピュートモード
            try:
                compute_mode = pynvml.nvmlDeviceGetComputeMode(handle)
                mode_names = {
                    pynvml.NVML_COMPUTEMODE_DEFAULT: "Default",
                    pynvml.NVML_COMPUTEMODE_EXCLUSIVE_THREAD: "Exclusive Thread",
                    pynvml.NVML_COMPUTEMODE_PROHIBITED: "Prohibited",
                    pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS: "Exclusive Process"
                }
                monitoring_data.compute_mode = mode_names.get(compute_mode, "Unknown")
            except:
                pass

        except Exception as e:
            raise Exception(f"NVML監視データ取得エラー: {e}")

    def _populate_nvidia_smi_monitoring_data(self, monitoring_data: GPUMonitoringData):
        """nvidia-smiを使用して監視データを取得"""
        try:
            # 複数の情報を一度に取得
            query_items = [
                'utilization.gpu',
                'utilization.memory',
                'memory.total',
                'memory.used',
                'memory.free',
                'temperature.gpu',
                'power.draw'
            ]
            query_string = ','.join(query_items)

            result = subprocess.run([
                'nvidia-smi',
                f'--query-gpu={query_string}',
                '--format=csv,noheader,nounits',
                f'--id={self.device_id}'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                values = result.stdout.strip().split(', ')

                if len(values) >= 7:
                    monitoring_data.gpu_utilization_percent = self._safe_float_conversion(values[0])
                    monitoring_data.memory_utilization_percent = self._safe_float_conversion(values[1])
                    monitoring_data.memory_total_mb = self._safe_float_conversion(values[2])
                    monitoring_data.memory_used_mb = self._safe_float_conversion(values[3])
                    monitoring_data.memory_free_mb = self._safe_float_conversion(values[4])
                    monitoring_data.temperature_celsius = self._safe_float_conversion(values[5])
                    monitoring_data.power_consumption_watts = self._safe_float_conversion(values[6])
            else:
                raise Exception(f"nvidia-smi実行失敗: {result.stderr}")

        except Exception as e:
            raise Exception(f"nvidia-smi監視データ取得エラー: {e}")

    def _safe_float_conversion(self, value_str: str) -> float:
        """安全な文字列→浮動小数点変換"""
        try:
            # "N/A"や空文字列の処理
            if value_str.strip() in ['N/A', '', '[N/A]']:
                return 0.0
            return float(value_str.strip())
        except (ValueError, AttributeError):
            return 0.0

    def check_gpu_health(self, monitoring_data: GPUMonitoringData) -> Dict[str, Any]:
        """GPU健全性チェック"""
        health_status = {
            "is_healthy": monitoring_data.is_healthy,
            "is_overloaded": monitoring_data.is_overloaded,
            "warnings": [],
            "critical_alerts": []
        }

        # 警告レベルのチェック
        if monitoring_data.gpu_utilization_percent > self.config.gpu_utilization_threshold:
            health_status["warnings"].append(
                f"GPU使用率が高い: {monitoring_data.gpu_utilization_percent:.1f}%"
            )

        if monitoring_data.memory_utilization_percent > self.config.gpu_memory_threshold:
            health_status["warnings"].append(
                f"GPUメモリ使用率が高い: {monitoring_data.memory_utilization_percent:.1f}%"
            )

        if monitoring_data.temperature_celsius > self.config.temperature_threshold:
            health_status["warnings"].append(
                f"GPU温度が高い: {monitoring_data.temperature_celsius:.1f}°C"
            )

        if monitoring_data.power_consumption_watts > self.config.power_threshold:
            health_status["warnings"].append(
                f"GPU電力消費が高い: {monitoring_data.power_consumption_watts:.1f}W"
            )

        # クリティカルレベルのチェック
        if monitoring_data.gpu_utilization_percent > 98.0:
            health_status["critical_alerts"].append("GPU使用率が限界に達しています")

        if monitoring_data.memory_utilization_percent > 98.0:
            health_status["critical_alerts"].append("GPUメモリ使用率が限界に達しています")

        if monitoring_data.temperature_celsius > 90.0:
            health_status["critical_alerts"].append("GPU温度が危険水準です")

        if monitoring_data.has_errors:
            health_status["critical_alerts"].append(f"GPU監視エラー: {monitoring_data.error_message}")

        return health_status

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

    # Issue #721対応: セッションクリーンアップメソッド
    def cleanup(self):
        """セッションリソースクリーンアップ"""
        try:
            # TensorRTエンジンクリーンアップ
            if self.tensorrt_engine:
                self.tensorrt_engine.cleanup()
                self.tensorrt_engine = None

            # ONNX Runtimeセッションクリーンアップ
            if self.session:
                del self.session
                self.session = None

            logger.debug(f"GPUセッションクリーンアップ完了: {self.model_name}")

        except Exception as e:
            logger.error(f"GPUセッションクリーンアップエラー: {e}")


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

        # Issue #720対応: GPU監視機能の初期化
        self.monitoring_enabled = self.config.enable_realtime_monitoring
        self.monitoring_data_history: Dict[int, List[GPUMonitoringData]] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_stop_event = threading.Event()

        # 監視データのバッファサイズ（最新N件を保持）
        self.monitoring_history_size = 100

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

    # Issue #720対応: GPU監視機能メソッド

    def start_realtime_monitoring(self):
        """リアルタイムGPU監視開始"""
        if not self.monitoring_enabled:
            logger.info("GPU監視が無効化されています")
            return

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("GPU監視は既に実行中です")
            return

        logger.info("リアルタイムGPU監視を開始")
        self.monitoring_stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            name="GPU-Monitor",
            daemon=True
        )
        self.monitoring_thread.start()

    def stop_realtime_monitoring(self):
        """リアルタイムGPU監視停止"""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            return

        logger.info("リアルタイムGPU監視を停止")
        self.monitoring_stop_event.set()
        self.monitoring_thread.join(timeout=5.0)

        if self.monitoring_thread.is_alive():
            logger.warning("GPU監視スレッド停止タイムアウト")

    def _monitoring_worker(self):
        """GPU監視ワーカースレッド"""
        logger.debug("GPU監視ワーカースレッド開始")

        # 各デバイスの監視データ履歴初期化
        for device_id in self.config.device_ids:
            self.monitoring_data_history[device_id] = []

        interval_seconds = self.config.monitoring_interval_ms / 1000.0

        while not self.monitoring_stop_event.wait(interval_seconds):
            try:
                # 各GPUデバイスの監視データ収集
                for device_id in self.config.device_ids:
                    # セッションからGPU監視データ取得
                    monitoring_data = self._collect_device_monitoring_data(device_id)

                    if monitoring_data:
                        # 履歴に追加
                        self.monitoring_data_history[device_id].append(monitoring_data)

                        # 履歴サイズ制限
                        if len(self.monitoring_data_history[device_id]) > self.monitoring_history_size:
                            self.monitoring_data_history[device_id].pop(0)

                        # 健全性チェック
                        health_status = self._check_device_health(device_id, monitoring_data)

                        # 警告・アラートのログ出力
                        self._handle_monitoring_alerts(device_id, health_status)

            except Exception as e:
                logger.error(f"GPU監視ワーカーエラー: {e}")

        logger.debug("GPU監視ワーカースレッド終了")

    def _collect_device_monitoring_data(self, device_id: int) -> Optional[GPUMonitoringData]:
        """指定デバイスの監視データ収集"""
        try:
            # デバイス用のセッションを探す
            session = None
            for model_name, sess in self.sessions.items():
                if self.session_device_mapping.get(model_name) == device_id:
                    session = sess
                    break

            if session:
                return session.get_comprehensive_gpu_monitoring()
            else:
                # セッションがない場合は基本的な監視データのみ
                return self._get_basic_device_monitoring(device_id)

        except Exception as e:
            logger.debug(f"デバイス{device_id}の監視データ収集エラー: {e}")
            return None

    def _get_basic_device_monitoring(self, device_id: int) -> GPUMonitoringData:
        """基本的なデバイス監視データ取得（セッション無し）"""
        monitoring_data = GPUMonitoringData(
            device_id=device_id,
            timestamp=time.time()
        )

        try:
            if PYNVML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

                # GPU使用率
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    monitoring_data.gpu_utilization_percent = float(utilization.gpu)
                    monitoring_data.memory_utilization_percent = float(utilization.memory)
                except:
                    pass

                # メモリ情報
                try:
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    monitoring_data.memory_total_mb = memory_info.total / (1024 ** 2)
                    monitoring_data.memory_used_mb = memory_info.used / (1024 ** 2)
                    monitoring_data.memory_free_mb = memory_info.free / (1024 ** 2)
                except:
                    pass

                # 温度
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    monitoring_data.temperature_celsius = float(temperature)
                except:
                    pass

                # 電力
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle)
                    monitoring_data.power_consumption_watts = power / 1000.0
                except:
                    pass

        except Exception as e:
            monitoring_data.has_errors = True
            monitoring_data.error_message = str(e)

        return monitoring_data

    def _check_device_health(self, device_id: int, monitoring_data: GPUMonitoringData) -> Dict[str, Any]:
        """デバイス健全性チェック"""
        health_status = {
            "device_id": device_id,
            "is_healthy": monitoring_data.is_healthy,
            "is_overloaded": monitoring_data.is_overloaded,
            "warnings": [],
            "critical_alerts": []
        }

        # 警告レベルチェック
        if monitoring_data.gpu_utilization_percent > self.config.gpu_utilization_threshold:
            health_status["warnings"].append(
                f"GPU使用率が閾値を超過: {monitoring_data.gpu_utilization_percent:.1f}% > {self.config.gpu_utilization_threshold}%"
            )

        if monitoring_data.memory_utilization_percent > self.config.gpu_memory_threshold:
            health_status["warnings"].append(
                f"GPUメモリ使用率が閾値を超過: {monitoring_data.memory_utilization_percent:.1f}% > {self.config.gpu_memory_threshold}%"
            )

        if monitoring_data.temperature_celsius > self.config.temperature_threshold:
            health_status["warnings"].append(
                f"GPU温度が閾値を超過: {monitoring_data.temperature_celsius:.1f}°C > {self.config.temperature_threshold}°C"
            )

        if monitoring_data.power_consumption_watts > self.config.power_threshold:
            health_status["warnings"].append(
                f"GPU電力消費が閾値を超過: {monitoring_data.power_consumption_watts:.1f}W > {self.config.power_threshold}W"
            )

        # クリティカルアラート
        if monitoring_data.gpu_utilization_percent > 98.0:
            health_status["critical_alerts"].append("GPU使用率が限界レベル (>98%)")

        if monitoring_data.memory_utilization_percent > 98.0:
            health_status["critical_alerts"].append("GPUメモリ使用率が限界レベル (>98%)")

        if monitoring_data.temperature_celsius > 90.0:
            health_status["critical_alerts"].append("GPU温度が危険レベル (>90°C)")

        if monitoring_data.has_errors:
            health_status["critical_alerts"].append(f"GPU監視エラー: {monitoring_data.error_message}")

        return health_status

    def _handle_monitoring_alerts(self, device_id: int, health_status: Dict[str, Any]):
        """監視アラートの処理"""
        # 警告ログ出力
        for warning in health_status["warnings"]:
            logger.warning(f"GPU {device_id} 警告: {warning}")

        # クリティカルアラートログ出力
        for alert in health_status["critical_alerts"]:
            logger.error(f"GPU {device_id} クリティカル: {alert}")

    def get_monitoring_data(self, device_id: Optional[int] = None) -> Dict[int, List[GPUMonitoringData]]:
        """監視データ履歴取得"""
        if device_id is not None:
            return {device_id: self.monitoring_data_history.get(device_id, [])}
        return self.monitoring_data_history.copy()

    def get_latest_monitoring_snapshot(self) -> Dict[str, Any]:
        """最新の監視データスナップショット取得"""
        snapshot = {
            "timestamp": time.time(),
            "devices": {},
            "summary": {
                "total_devices": len(self.config.device_ids),
                "healthy_devices": 0,
                "overloaded_devices": 0,
                "devices_with_errors": 0,
                "avg_gpu_utilization": 0.0,
                "avg_memory_utilization": 0.0,
                "avg_temperature": 0.0,
                "total_power_consumption": 0.0
            }
        }

        gpu_utils = []
        memory_utils = []
        temperatures = []
        power_consumptions = []

        for device_id in self.config.device_ids:
            history = self.monitoring_data_history.get(device_id, [])
            if history:
                latest_data = history[-1]
                snapshot["devices"][device_id] = latest_data.to_dict()

                # サマリ統計用
                gpu_utils.append(latest_data.gpu_utilization_percent)
                memory_utils.append(latest_data.memory_utilization_percent)
                temperatures.append(latest_data.temperature_celsius)
                power_consumptions.append(latest_data.power_consumption_watts)

                if latest_data.is_healthy:
                    snapshot["summary"]["healthy_devices"] += 1
                if latest_data.is_overloaded:
                    snapshot["summary"]["overloaded_devices"] += 1
                if latest_data.has_errors:
                    snapshot["summary"]["devices_with_errors"] += 1

        # サマリ統計計算
        if gpu_utils:
            snapshot["summary"]["avg_gpu_utilization"] = np.mean(gpu_utils)
            snapshot["summary"]["avg_memory_utilization"] = np.mean(memory_utils)
            snapshot["summary"]["avg_temperature"] = np.mean(temperatures)
            snapshot["summary"]["total_power_consumption"] = np.sum(power_consumptions)

        return snapshot

    def cleanup(self):
        """リソースクリーンアップ"""
        try:
            # Issue #720対応: GPU監視停止
            self.stop_realtime_monitoring()

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
