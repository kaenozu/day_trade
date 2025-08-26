#!/usr/bin/env python3
"""
モデル量子化データ構造定義
Issue #379: ML Model Inference Performance Optimization

量子化・プルーニング・圧縮で使用される共通データ構造
- 列挙型（QuantizationType, PruningType, HardwareTarget）
- 設定データクラス（CompressionConfig）
- 結果データクラス（CompressionResult）
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class QuantizationType(Enum):
    """量子化タイプ"""

    NONE = "none"
    DYNAMIC_INT8 = "dynamic_int8"
    STATIC_INT8 = "static_int8"
    MIXED_PRECISION_FP16 = "mixed_precision_fp16"
    CUSTOM_QUANTIZATION = "custom_quantization"


class PruningType(Enum):
    """プルーニングタイプ"""

    NONE = "none"
    UNSTRUCTURED = "unstructured"  # 非構造化プルーニング
    STRUCTURED_CHANNEL = "structured_channel"  # チャネル単位構造化
    STRUCTURED_BLOCK = "structured_block"  # ブロック単位構造化
    MAGNITUDE_BASED = "magnitude_based"  # 重み大きさベース
    GRADIENT_BASED = "gradient_based"  # 勾配ベース


class HardwareTarget(Enum):
    """ハードウェア最適化ターゲット"""

    GENERIC_CPU = "generic_cpu"
    INTEL_X86 = "intel_x86"
    AMD_X86 = "amd_x86"
    ARM_CORTEX = "arm_cortex"
    NVIDIA_GPU = "nvidia_gpu"
    AMD_GPU = "amd_gpu"


@dataclass
class CompressionConfig:
    """モデル圧縮設定"""

    quantization_type: QuantizationType = QuantizationType.DYNAMIC_INT8
    pruning_type: PruningType = PruningType.MAGNITUDE_BASED
    target_hardware: HardwareTarget = HardwareTarget.GENERIC_CPU

    # 量子化設定
    quantization_ratio: float = 0.95  # 量子化する層の割合
    calibration_dataset_size: int = 1000

    # プルーニング設定
    pruning_ratio: float = 0.5  # プルーニングする重みの割合
    structured_pruning_granularity: int = 4  # 構造化プルーニング粒度

    # 知識蒸留設定
    enable_knowledge_distillation: bool = False
    teacher_model_path: str | None = None
    distillation_temperature: float = 3.0

    # 最適化設定
    optimize_for_inference: bool = True
    target_speedup_ratio: float = 3.0  # 目標速度向上倍率
    max_accuracy_drop: float = 0.02  # 許容精度低下

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式に変換"""
        return {
            "quantization_type": self.quantization_type.value,
            "pruning_type": self.pruning_type.value,
            "target_hardware": self.target_hardware.value,
            "quantization_ratio": self.quantization_ratio,
            "calibration_dataset_size": self.calibration_dataset_size,
            "pruning_ratio": self.pruning_ratio,
            "structured_pruning_granularity": self.structured_pruning_granularity,
            "enable_knowledge_distillation": self.enable_knowledge_distillation,
            "teacher_model_path": self.teacher_model_path,
            "distillation_temperature": self.distillation_temperature,
            "optimize_for_inference": self.optimize_for_inference,
            "target_speedup_ratio": self.target_speedup_ratio,
            "max_accuracy_drop": self.max_accuracy_drop,
        }


@dataclass
class CompressionResult:
    """モデル圧縮結果"""

    original_model_size_mb: float
    compressed_model_size_mb: float
    compression_ratio: float

    original_inference_time_us: float
    compressed_inference_time_us: float
    speedup_ratio: float

    original_accuracy: float
    compressed_accuracy: float
    accuracy_drop: float

    quantization_applied: bool = False
    pruning_applied: bool = False
    distillation_applied: bool = False

    optimization_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """結果を辞書形式に変換"""
        return {
            "original_model_size_mb": self.original_model_size_mb,
            "compressed_model_size_mb": self.compressed_model_size_mb,
            "compression_ratio": self.compression_ratio,
            "original_inference_time_us": self.original_inference_time_us,
            "compressed_inference_time_us": self.compressed_inference_time_us,
            "speedup_ratio": self.speedup_ratio,
            "original_accuracy": self.original_accuracy,
            "compressed_accuracy": self.compressed_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "quantization_applied": self.quantization_applied,
            "pruning_applied": self.pruning_applied,
            "distillation_applied": self.distillation_applied,
            "optimization_stats": self.optimization_stats,
        }