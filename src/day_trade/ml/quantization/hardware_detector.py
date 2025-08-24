#!/usr/bin/env python3
"""
ハードウェア特性検出モジュール
Issue #379: ML Model Inference Performance Optimization

CPU/GPU特性の自動検出と最適化設定推奨機能
- CPU特性検出（AVX2/AVX512/FMA対応チェック）
- GPU特性検出（CUDA/ROCm/OpenVINO対応チェック）
- メモリ情報取得
- ハードウェアに最適化された圧縮設定推奨
"""

import warnings
from typing import Any, Dict

from ..utils.logging_config import get_context_logger
from .data_structures import CompressionConfig, HardwareTarget, QuantizationType

logger = get_context_logger(__name__)

# ハードウェア検出用ライブラリ (フォールバック対応)
try:
    import cpuinfo

    CPU_INFO_AVAILABLE = True
except ImportError:
    CPU_INFO_AVAILABLE = False

# ONNX Runtime (GPU検出用)
try:
    import onnxruntime as ort

    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False


class HardwareDetector:
    """ハードウェア特性検出"""

    def __init__(self):
        """ハードウェア情報の初期化検出"""
        self.cpu_features = self._detect_cpu_features()
        self.gpu_features = self._detect_gpu_features()
        self.memory_info = self._detect_memory_info()

    def _detect_cpu_features(self) -> Dict[str, Any]:
        """CPU特性検出"""
        features = {
            "architecture": "unknown",
            "vendor": "unknown",
            "supports_avx2": False,
            "supports_avx512": False,
            "supports_fma": False,
            "core_count": 1,
            "cache_sizes": {},
        }

        try:
            if CPU_INFO_AVAILABLE:
                info = cpuinfo.get_cpu_info()
                features["architecture"] = info.get("arch", "unknown")
                features["vendor"] = info.get("vendor_id_raw", "unknown")

                flags = info.get("flags", [])
                features["supports_avx2"] = "avx2" in flags
                features["supports_avx512"] = any(
                    "avx512" in flag for flag in flags
                )
                features["supports_fma"] = "fma" in flags
                features["core_count"] = info.get("count", 1)

            # Intel/AMD特化判定
            if "intel" in features["vendor"].lower():
                features["optimization_target"] = HardwareTarget.INTEL_X86
            elif "amd" in features["vendor"].lower():
                features["optimization_target"] = HardwareTarget.AMD_X86
            elif "arm" in features["architecture"].lower():
                features["optimization_target"] = HardwareTarget.ARM_CORTEX
            else:
                features["optimization_target"] = HardwareTarget.GENERIC_CPU

        except Exception as e:
            logger.warning(f"CPU特性検出エラー: {e}")

        return features

    def _detect_gpu_features(self) -> Dict[str, Any]:
        """GPU特性検出"""
        features = {
            "has_gpu": False,
            "gpu_type": "none",
            "memory_mb": 0,
            "compute_capability": "unknown",
        }

        try:
            # CUDA GPU検出
            if ONNX_RUNTIME_AVAILABLE:
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    features["has_gpu"] = True
                    features["gpu_type"] = "nvidia_cuda"
                elif "ROCMExecutionProvider" in providers:
                    features["has_gpu"] = True
                    features["gpu_type"] = "amd_rocm"
                elif "OpenVINOExecutionProvider" in providers:
                    features["has_gpu"] = True
                    features["gpu_type"] = "intel_openvino"

        except Exception as e:
            logger.warning(f"GPU特性検出エラー: {e}")

        return features

    def _detect_memory_info(self) -> Dict[str, Any]:
        """メモリ情報検出"""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return {
                "total_mb": memory.total // 1024 // 1024,
                "available_mb": memory.available // 1024 // 1024,
                "usage_percent": memory.percent,
            }
        except ImportError:
            return {
                "total_mb": 8192,
                "available_mb": 4096,
                "usage_percent": 50.0,
            }

    def get_optimal_config(self) -> CompressionConfig:
        """最適圧縮設定の推奨"""
        config = CompressionConfig()

        # CPU特化最適化
        config.target_hardware = self.cpu_features["optimization_target"]

        # AVX512対応なら積極的量子化
        if self.cpu_features["supports_avx512"]:
            config.quantization_type = QuantizationType.STATIC_INT8
            config.quantization_ratio = 0.9
        elif self.cpu_features["supports_avx2"]:
            config.quantization_type = QuantizationType.DYNAMIC_INT8
            config.quantization_ratio = 0.8

        # GPU利用可能なら混合精度
        if (
            self.gpu_features["has_gpu"]
            and "nvidia" in self.gpu_features["gpu_type"]
        ):
            config.quantization_type = QuantizationType.MIXED_PRECISION_FP16

        # メモリ制約に応じたプルーニング
        if self.memory_info["available_mb"] < 2048:
            config.pruning_ratio = 0.7  # 積極的プルーニング
        elif self.memory_info["available_mb"] < 4096:
            config.pruning_ratio = 0.5
        else:
            config.pruning_ratio = 0.3

        logger.info(f"最適圧縮設定推奨完了: {config.to_dict()}")
        return config

    def get_hardware_info(self) -> Dict[str, Any]:
        """ハードウェア情報の完全な辞書を返す"""
        return {
            "cpu_features": self.cpu_features,
            "gpu_features": self.gpu_features,
            "memory_info": self.memory_info,
        }

    def is_high_performance_cpu(self) -> bool:
        """高性能CPU判定（AVX512対応など）"""
        return (
            self.cpu_features["supports_avx512"]
            or self.cpu_features["core_count"] >= 8
        )

    def supports_gpu_acceleration(self) -> bool:
        """GPU加速対応判定"""
        return self.gpu_features["has_gpu"]

    def get_recommended_quantization_type(self) -> QuantizationType:
        """推奨量子化タイプ取得"""
        if self.supports_gpu_acceleration():
            return QuantizationType.MIXED_PRECISION_FP16
        elif self.is_high_performance_cpu():
            return QuantizationType.STATIC_INT8
        else:
            return QuantizationType.DYNAMIC_INT8