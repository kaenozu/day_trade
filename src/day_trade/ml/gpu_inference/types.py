#!/usr/bin/env python3
"""
GPU推論の基本型と設定
Issue #379: ML Model Inference Performance Optimization
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

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
    warnings.warn("OpenCL not available", stacklevel=2)

# TensorRT支援 (フォールバック対応)
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    warnings.warn("TensorRT not available - 最適化制限", stacklevel=2)

# PyCUDA支援 (フォールバック対応)
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    warnings.warn("PyCUDA not available - TensorRT推論制限", stacklevel=2)

# NVIDIA GPU管理 (フォールバック対応)
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not available - GPU監視機能制限", stacklevel=2)


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


@dataclass
class GPUMonitoringData:
    """GPU監視データ"""
    
    device_id: int
    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_free_mb: float = 0.0
    temperature_celsius: float = 0.0
    power_draw_watts: float = 0.0
    fan_speed_percent: float = 0.0
    clock_graphics_mhz: int = 0
    clock_memory_mhz: int = 0
    
    # 時系列データ
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """監視データを辞書形式に変換"""
        return {
            "device_id": self.device_id,
            "utilization_percent": self.utilization_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "memory_free_mb": self.memory_free_mb,
            "temperature_celsius": self.temperature_celsius,
            "power_draw_watts": self.power_draw_watts,
            "fan_speed_percent": self.fan_speed_percent,
            "clock_graphics_mhz": self.clock_graphics_mhz,
            "clock_memory_mhz": self.clock_memory_mhz,
            "timestamp": self.timestamp,
        }