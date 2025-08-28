#!/usr/bin/env python3
"""
GPU推論パッケージ
Issue #379: ML Model Inference Performance Optimization
"""

from .types import (
    GPUBackend,
    ParallelizationMode,
    GPUInferenceConfig,
    GPUInferenceResult,
    GPUMonitoringData,
    # 互換性定数
    ONNX_GPU_AVAILABLE,
    CUPY_AVAILABLE,
    OPENCL_AVAILABLE,
    TENSORRT_AVAILABLE,
    PYCUDA_AVAILABLE,
    PYNVML_AVAILABLE,
)
from .device_manager import GPUDeviceManager
from .tensorrt_engine import TensorRTEngine
from .gpu_monitoring import GPUMonitor
from .stream_manager import GPUStreamManager
from .batch_processor import GPUBatchProcessor
from .inference_session import GPUInferenceSession
from .inference_engine import GPUAcceleratedInferenceEngine, create_gpu_inference_engine

__all__ = [
    "GPUBackend",
    "ParallelizationMode", 
    "GPUInferenceConfig",
    "GPUInferenceResult",
    "GPUMonitoringData",
    "GPUDeviceManager",
    "TensorRTEngine",
    "GPUMonitor",
    "GPUStreamManager",
    "GPUBatchProcessor", 
    "GPUInferenceSession",
    "GPUAcceleratedInferenceEngine",
    "create_gpu_inference_engine",
    # 互換性定数
    "ONNX_GPU_AVAILABLE",
    "CUPY_AVAILABLE",
    "OPENCL_AVAILABLE", 
    "TENSORRT_AVAILABLE",
    "PYCUDA_AVAILABLE",
    "PYNVML_AVAILABLE",
]