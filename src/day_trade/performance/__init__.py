#!/usr/bin/env python3
"""
パフォーマンス最適化パッケージ
Issue #434: 本番環境パフォーマンス最終最適化
Issue #443: HFT超低レイテンシ最適化 - <10μs実現戦略

HFT超低レイテンシとGPU加速による極限性能システム
Rust FFI統合による究極の低レイテンシ(<10μs)実現
"""

from .gpu_accelerator import GPUAccelerator, GPUConfig, get_gpu_accelerator, gpu_accelerated
from .hft_optimizer import HFTOptimizer, HFTConfig, get_hft_optimizer, hft_optimized

# Issue #443: 超低レイテンシシステム
try:
    from .ultra_low_latency_core import (
        UltraLowLatencyCore,
        UltraLowLatencyConfig,
        create_ultra_low_latency_core
    )
    from .system_optimization import (
        SystemOptimizer,
        SystemOptimizationConfig,
        setup_ultra_low_latency_system
    )
    ULTRA_LOW_LATENCY_AVAILABLE = True
except ImportError:
    ULTRA_LOW_LATENCY_AVAILABLE = False

__all__ = [
    # 既存システム
    "GPUAccelerator",
    "GPUConfig",
    "get_gpu_accelerator",
    "gpu_accelerated",
    "HFTOptimizer",
    "HFTConfig",
    "get_hft_optimizer",
    "hft_optimized",
    # 超低レイテンシシステム可用性フラグ
    "ULTRA_LOW_LATENCY_AVAILABLE",
]

# 超低レイテンシシステムが利用可能な場合に追加
if ULTRA_LOW_LATENCY_AVAILABLE:
    __all__.extend([
        "UltraLowLatencyCore",
        "UltraLowLatencyConfig",
        "create_ultra_low_latency_core",
        "SystemOptimizer",
        "SystemOptimizationConfig",
        "setup_ultra_low_latency_system",
    ])

__version__ = "2.0.0"  # 超低レイテンシ対応によるメジャーバージョンアップ

# パフォーマンス情報取得関数
def get_performance_info():
    """パフォーマンス最適化システム情報取得"""
    info = {
        "version": __version__,
        "systems": {
            "hft_optimizer": "HFT <50μs最適化システム",
            "gpu_accelerator": "GPU加速処理システム",
        },
        "features": [
            "SIMD並列処理",
            "メモリプール事前割り当て",
            "CPUキャッシュ最適化",
            "CUDA GPU加速",
            "PyTorch/TensorFlow統合",
        ],
    }

    if ULTRA_LOW_LATENCY_AVAILABLE:
        info["systems"]["ultra_low_latency"] = "超低レイテンシ <10μs実現システム"
        info["features"].extend([
            "Rust FFI統合",
            "Lock-freeデータ構造",
            "システムレベル最適化",
            "リアルタイムスケジューラ",
            "CPU親和性制御",
            "RDTSC高精度タイミング",
        ])

    return info

# システム能力検証関数
def verify_system_capabilities():
    """システム能力検証"""
    capabilities = {
        "hft_basic": True,
        "gpu_acceleration": False,
        "ultra_low_latency": ULTRA_LOW_LATENCY_AVAILABLE,
        "system_optimization": False,
    }

    try:
        # GPU能力チェック
        gpu = get_gpu_accelerator()
        if gpu.cuda_available or gpu.pytorch_available:
            capabilities["gpu_acceleration"] = True
    except Exception:
        pass

    try:
        # システム最適化能力チェック
        if ULTRA_LOW_LATENCY_AVAILABLE:
            import platform
            if platform.system() in ['Linux', 'Windows']:
                capabilities["system_optimization"] = True
    except Exception:
        pass

    return capabilities