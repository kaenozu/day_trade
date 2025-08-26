#!/usr/bin/env python3
"""
GPU加速推論エンジン - 後方互換性維持
Issue #379: ML Model Inference Performance Optimization

このファイルは新しいモジュラー構造への後方互換性のために保持されています。
新しいコードでは gpu_inference パッケージを直接使用してください。

元の2167行の実装は以下のモジュールに分割されました：
- gpu_inference.types: 型定義とEnum (247行)
- gpu_inference.config: 設定クラス (125行)  
- gpu_inference.device_manager: デバイス管理 (171行)
- gpu_inference.stream_manager: ストリーム管理 (124行)
- gpu_inference.batch_processor: バッチ処理 (157行)
- gpu_inference.gpu_monitoring: GPU監視 (197行)
- gpu_inference.tensorrt_engine: TensorRT エンジン (326行)
- gpu_inference.inference_session: 推論セッション (287行)
- gpu_inference.inference_engine: メイン推論エンジン (294行)
"""

# 新しいモジュール構造からの完全インポート
from .gpu_inference import (
    # 型定義とEnum
    GPUBackend,
    ParallelizationMode,
    GPUInferenceConfig,
    GPUInferenceResult,
    GPUMonitoringData,
    
    # コアクラス
    GPUDeviceManager,
    GPUStreamManager,
    GPUBatchProcessor,
    GPUMonitor,
    TensorRTEngine,
    GPUInferenceSession,
    GPUAcceleratedInferenceEngine,
    
    # ファクトリ関数
    create_gpu_inference_engine,
    
    # 互換性定数
    ONNX_GPU_AVAILABLE,
    CUPY_AVAILABLE,
    OPENCL_AVAILABLE,
    TENSORRT_AVAILABLE,
    PYCUDA_AVAILABLE,
    PYNVML_AVAILABLE,
)

# 既存システムとの統合（必要に応じて）
from ..trading.high_frequency_engine import MicrosecondTimer
from ..utils.logging_config import get_context_logger
from ..utils.unified_cache_manager import UnifiedCacheManager

# 後方互換性のための警告出力
import warnings
warnings.warn(
    "gpu_accelerated_inference.py は非推奨です。"
    "新しいコードでは day_trade.ml.gpu_inference パッケージを直接使用してください。",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'GPUBackend',
    'ParallelizationMode', 
    'GPUInferenceConfig',
    'GPUInferenceResult',
    'GPUMonitoringData',
    'GPUDeviceManager',
    'GPUStreamManager',
    'GPUBatchProcessor',
    'GPUMonitor',
    'TensorRTEngine',
    'GPUInferenceSession',
    'GPUAcceleratedInferenceEngine',
    'create_gpu_inference_engine',
    'ONNX_GPU_AVAILABLE',
    'CUPY_AVAILABLE', 
    'OPENCL_AVAILABLE',
    'TENSORRT_AVAILABLE',
    'PYCUDA_AVAILABLE',
    'PYNVML_AVAILABLE',
]