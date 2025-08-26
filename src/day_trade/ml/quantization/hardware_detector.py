#!/usr/bin/env python3
"""
ハードウェア特性検出システム
Issue #379: ML Model Inference Performance Optimization

CPU、GPU、メモリ特性を検出し、最適化設定を推奨
"""

from typing import Any, Dict

from .core import CompressionConfig, HardwareTarget, QuantizationType, check_dependencies
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class HardwareDetector:
    """ハードウェア特性検出"""

    def __init__(self):
        """ハードウェア検出を初期化"""
        self.dependencies = check_dependencies()
        self.cpu_features = self._detect_cpu_features()
        self.gpu_features = self._detect_gpu_features()
        self.memory_info = self._detect_memory_info()
        
        logger.info("ハードウェア検出完了")

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
            "optimization_target": HardwareTarget.GENERIC_CPU,
        }

        try:
            if self.dependencies.get("cpu_info", False):
                import cpuinfo
                
                info = cpuinfo.get_cpu_info()
                features["architecture"] = info.get("arch", "unknown")
                features["vendor"] = info.get("vendor_id_raw", "unknown")

                flags = info.get("flags", [])
                features["supports_avx2"] = "avx2" in flags
                features["supports_avx512"] = any("avx512" in flag for flag in flags)
                features["supports_fma"] = "fma" in flags
                features["core_count"] = info.get("count", 1)

            # Intel/AMD特化判定
            vendor_lower = features["vendor"].lower()
            if "intel" in vendor_lower:
                features["optimization_target"] = HardwareTarget.INTEL_X86
            elif "amd" in vendor_lower:
                features["optimization_target"] = HardwareTarget.AMD_X86
            elif "arm" in features["architecture"].lower():
                features["optimization_target"] = HardwareTarget.ARM_CORTEX
            else:
                features["optimization_target"] = HardwareTarget.GENERIC_CPU

            logger.debug(f"CPU特性検出結果: {features}")

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
            "providers": [],
        }

        try:
            if self.dependencies.get("onnx_quantization", False):
                import onnxruntime as ort
                
                providers = ort.get_available_providers()
                features["providers"] = providers
                
                if "CUDAExecutionProvider" in providers:
                    features["has_gpu"] = True
                    features["gpu_type"] = "nvidia_cuda"
                    features["optimization_target"] = HardwareTarget.NVIDIA_GPU
                elif "ROCMExecutionProvider" in providers:
                    features["has_gpu"] = True
                    features["gpu_type"] = "amd_rocm"
                    features["optimization_target"] = HardwareTarget.AMD_GPU
                elif "OpenVINOExecutionProvider" in providers:
                    features["has_gpu"] = True
                    features["gpu_type"] = "intel_openvino"
                    
            # PyTorchからGPU情報も取得
            if self.dependencies.get("pytorch_quantization", False):
                try:
                    import torch
                    if torch.cuda.is_available():
                        features["has_gpu"] = True
                        if features["gpu_type"] == "none":
                            features["gpu_type"] = "nvidia_cuda"
                            features["optimization_target"] = HardwareTarget.NVIDIA_GPU
                        
                        # GPU詳細情報
                        features["cuda_device_count"] = torch.cuda.device_count()
                        features["cuda_device_name"] = torch.cuda.get_device_name(0)
                except Exception as e:
                    logger.debug(f"PyTorch GPU検出エラー: {e}")

            logger.debug(f"GPU特性検出結果: {features}")

        except Exception as e:
            logger.warning(f"GPU特性検出エラー: {e}")

        return features

    def _detect_memory_info(self) -> Dict[str, Any]:
        """メモリ情報検出"""
        memory_info = {
            "total_mb": 8192,  # デフォルト値
            "available_mb": 4096,
            "usage_percent": 50.0,
        }
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            memory_info.update({
                "total_mb": memory.total // 1024 // 1024,
                "available_mb": memory.available // 1024 // 1024,
                "usage_percent": memory.percent,
            })
            
            logger.debug(f"メモリ情報: {memory_info}")
            
        except ImportError:
            logger.warning("psutilが利用不可 - デフォルトメモリ情報を使用")
        except Exception as e:
            logger.warning(f"メモリ情報取得エラー: {e}")

        return memory_info

    def get_optimal_config(self) -> CompressionConfig:
        """ハードウェアに最適化された圧縮設定を推奨"""
        config = CompressionConfig()

        try:
            # CPU特化最適化
            config.target_hardware = self.cpu_features["optimization_target"]

            # AVX512対応なら積極的量子化
            if self.cpu_features["supports_avx512"]:
                config.quantization_type = QuantizationType.STATIC_INT8
                config.quantization_ratio = 0.9
                logger.info("AVX512対応CPU検出 - 静的INT8量子化を推奨")
                
            elif self.cpu_features["supports_avx2"]:
                config.quantization_type = QuantizationType.DYNAMIC_INT8
                config.quantization_ratio = 0.8
                logger.info("AVX2対応CPU検出 - 動的INT8量子化を推奨")
                
            else:
                config.quantization_type = QuantizationType.DYNAMIC_INT8
                config.quantization_ratio = 0.6
                logger.info("標準CPU検出 - 軽量動的量子化を推奨")

            # GPU利用可能なら混合精度を優先
            if self.gpu_features["has_gpu"]:
                if "nvidia" in self.gpu_features["gpu_type"]:
                    config.quantization_type = QuantizationType.MIXED_PRECISION_FP16
                    config.target_hardware = HardwareTarget.NVIDIA_GPU
                    logger.info("NVIDIA GPU検出 - FP16混合精度量子化を推奨")
                elif "amd" in self.gpu_features["gpu_type"]:
                    config.target_hardware = HardwareTarget.AMD_GPU
                    logger.info("AMD GPU検出 - AMD GPU最適化を推奨")

            # メモリ制約に応じたプルーニング調整
            available_mb = self.memory_info["available_mb"]
            if available_mb < 2048:
                config.pruning_ratio = 0.7  # 積極的プルーニング
                logger.info("低メモリ環境検出 - 積極的プルーニング (70%) を推奨")
            elif available_mb < 4096:
                config.pruning_ratio = 0.5
                logger.info("中メモリ環境検出 - 標準プルーニング (50%) を推奨")
            else:
                config.pruning_ratio = 0.3
                logger.info("高メモリ環境検出 - 軽量プルーニング (30%) を推奨")

            # コア数に応じたキャリブレーションデータサイズ調整
            core_count = self.cpu_features["core_count"]
            if core_count >= 8:
                config.calibration_dataset_size = 2000
            elif core_count >= 4:
                config.calibration_dataset_size = 1000
            else:
                config.calibration_dataset_size = 500

            logger.info(f"最適圧縮設定推奨完了: {config.to_dict()}")
            
        except Exception as e:
            logger.error(f"最適設定推奨エラー: {e}")
            # エラー時はデフォルト設定を返す

        return config