#!/usr/bin/env python3
"""
ML推論最適化システム統合モジュール
ML Inference Optimization System Integration Module

Issue #761: MLモデル推論パイプラインの高速化と最適化
"""

from .model_optimizer import (
    ModelOptimizationConfig,
    OptimizationMetrics,
    ModelOptimizationEngine,
    ONNXOptimizer,
    TensorRTOptimizer,
    QuantizationOptimizer,
    BatchInferenceEngine
)

from .memory_optimizer import (
    MemoryConfig,
    MemoryStats,
    MemoryOptimizer,
    ModelPool,
    FeatureCache,
    MemoryMonitor,
    ZeroCopyManager
)

from .parallel_engine import (
    ParallelConfig,
    WorkerStats,
    InferenceTask,
    InferenceResult,
    ParallelInferenceEngine,
    CPUWorkerPool,
    GPUWorkerPool,
    AsyncWorkerPool
)

from .advanced_optimizer import (
    OptimizationConfig,
    ModelPerformanceMetrics,
    DynamicModelSelector,
    InferenceCache,
    PerformanceProfiler,
    ABTester,
    AdvancedOptimizer
)

from .integrated_system import (
    InferenceSystemConfig,
    OptimizedInferenceSystem,
    create_optimized_inference_system
)

__version__ = "1.0.0"
__author__ = "Day Trade ML System"

# モジュール公開API
__all__ = [
    # Model Optimization
    "ModelOptimizationConfig",
    "OptimizationMetrics",
    "ModelOptimizationEngine",
    "ONNXOptimizer",
    "TensorRTOptimizer",
    "QuantizationOptimizer",
    "BatchInferenceEngine",

    # Memory Optimization
    "MemoryConfig",
    "MemoryStats",
    "MemoryOptimizer",
    "ModelPool",
    "FeatureCache",
    "MemoryMonitor",
    "ZeroCopyManager",

    # Parallel Processing
    "ParallelConfig",
    "WorkerStats",
    "InferenceTask",
    "InferenceResult",
    "ParallelInferenceEngine",
    "CPUWorkerPool",
    "GPUWorkerPool",
    "AsyncWorkerPool",

    # Advanced Optimization
    "OptimizationConfig",
    "ModelPerformanceMetrics",
    "DynamicModelSelector",
    "InferenceCache",
    "PerformanceProfiler",
    "ABTester",
    "AdvancedOptimizer",

    # Integrated System
    "InferenceSystemConfig",
    "OptimizedInferenceSystem",
    "create_optimized_inference_system"
]


def get_system_info() -> dict:
    """
    システム情報取得

    Returns:
        システム情報辞書
    """
    return {
        "module": "day_trade.inference",
        "version": __version__,
        "components": [
            "ModelOptimizationEngine",
            "MemoryOptimizer",
            "ParallelInferenceEngine",
            "AdvancedOptimizer",
            "OptimizedInferenceSystem"
        ],
        "features": [
            "ONNX Runtime最適化",
            "TensorRT GPU最適化",
            "モデル量子化",
            "バッチ推論",
            "メモリプール管理",
            "特徴量キャッシング",
            "並列処理",
            "動的モデル選択",
            "推論結果キャッシング",
            "A/Bテスト",
            "パフォーマンスプロファイリング"
        ],
        "performance_targets": {
            "latency_reduction": "80%以上",
            "throughput_improvement": "10倍以上",
            "memory_efficiency": "50%削減",
            "accuracy_retention": "97%以上"
        },
        "supported_frameworks": [
            "ONNX Runtime",
            "TensorRT",
            "PyTorch",
            "TensorFlow"
        ]
    }