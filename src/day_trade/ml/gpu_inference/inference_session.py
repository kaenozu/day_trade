#!/usr/bin/env python3
"""
GPU推論セッション（コンパクト版）
Issue #379: ML Model Inference Performance Optimization
"""

import hashlib
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..trading.high_frequency_engine import MicrosecondTimer
from ..utils.logging_config import get_context_logger
from .types import GPUInferenceConfig, GPUInferenceResult, GPUBackend, GPUMonitoringData
from .tensorrt_engine import TensorRTEngine
from .gpu_monitoring import GPUMonitor

logger = get_context_logger(__name__)

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
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available - CPU fallback", stacklevel=2)

# TensorRT支援 (フォールバック対応)
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


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

        # セッション関連
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None

        # GPU 特有のリソース
        self.cuda_context = None
        self.tensorrt_engine = None
        self.use_tensorrt = False

        # GPU監視機能
        self.gpu_monitor = GPUMonitor(device_id)

        # 統計
        self.inference_stats = {
            "total_inferences": 0,
            "total_gpu_time_us": 0,
            "avg_gpu_time_us": 0.0,
            "gpu_memory_peak_mb": 0.0,
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
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # セッション作成
            self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)

            # 入出力情報取得
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_shape = self.session.get_inputs()[0].shape

            # GPU コンテキスト初期化
            if CUPY_AVAILABLE and self.config.backend == GPUBackend.CUDA:
                self.cuda_context = cp.cuda.Device(self.device_id)

            logger.info(f"GPU 推論セッション初期化完了: {self.model_name}")

        except Exception as e:
            logger.error(f"GPU 推論セッション初期化エラー: {e}")
            self.session = None

        # TensorRT初期化を試行
        self._try_initialize_tensorrt()

    def _get_execution_providers(self) -> List[Union[str, Tuple[str, Dict]]]:
        """実行プロバイダー取得"""
        providers = []

        if self.config.backend == GPUBackend.CUDA and ONNX_GPU_AVAILABLE:
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                cuda_options = {
                    "device_id": self.device_id,
                    "gpu_mem_limit": self.config.memory_pool_size_mb * 1024 * 1024,
                }
                providers.append(("CUDAExecutionProvider", cuda_options))

        # CPU フォールバック
        providers.append("CPUExecutionProvider")
        return providers

    def _try_initialize_tensorrt(self):
        """TensorRT初期化を試行"""
        if (not self.config.enable_tensorrt or
            not TENSORRT_AVAILABLE or
            not self.model_path.endswith('.onnx')):
            return

        try:
            self.tensorrt_engine = TensorRTEngine(self.config, self.device_id)
            engine_path = self._get_tensorrt_engine_path()

            if engine_path.exists():
                if self.tensorrt_engine.load_engine(str(engine_path)):
                    self.use_tensorrt = True
                    logger.info("TensorRT推論有効化")
            elif self.tensorrt_engine.build_engine_from_onnx(self.model_path):
                self.tensorrt_engine.save_engine(str(engine_path))
                self.use_tensorrt = True
                logger.info("TensorRT推論有効化")

        except Exception as e:
            logger.warning(f"TensorRT初期化エラー: {e}")
            self.tensorrt_engine = None

    def _get_tensorrt_engine_path(self) -> Path:
        """TensorRTエンジンファイルパス生成"""
        model_path = Path(self.model_path)
        config_str = f"{self.config.tensorrt_precision}_{self.config.tensorrt_max_batch_size}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_dir = model_path.parent / "tensorrt_cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{model_path.stem}_{config_hash}.trt"

    async def predict_gpu(self, input_data: np.ndarray) -> GPUInferenceResult:
        """GPU推論実行"""
        start_time = MicrosecondTimer.now_ns()

        try:
            gpu_memory_before = self._get_gpu_memory_usage()

            # TensorRT推論優先実行
            if self.use_tensorrt and self.tensorrt_engine:
                input_tensor = input_data.astype(np.float16 if self.config.enable_half_precision else np.float32)
                outputs = [self.tensorrt_engine.predict(input_tensor)]
                backend_used = GPUBackend.CUDA
            else:
                if self.session is None:
                    raise RuntimeError("GPU推論セッション未初期化")

                input_tensor = input_data.astype(np.float16 if self.config.enable_half_precision else np.float32)
                with self._gpu_context():
                    outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
                backend_used = self.config.backend

            execution_time = MicrosecondTimer.elapsed_us(start_time)
            gpu_memory_after = self._get_gpu_memory_usage()
            gpu_memory_used = max(0, gpu_memory_after - gpu_memory_before)

            # 統計更新
            self.inference_stats["total_inferences"] += 1
            self.inference_stats["total_gpu_time_us"] += execution_time
            self.inference_stats["avg_gpu_time_us"] = (
                self.inference_stats["total_gpu_time_us"] / self.inference_stats["total_inferences"]
            )

            return GPUInferenceResult(
                predictions=outputs[0],
                execution_time_us=execution_time,
                batch_size=input_data.shape[0],
                device_id=self.device_id,
                backend_used=backend_used,
                gpu_memory_used_mb=gpu_memory_used,
                gpu_utilization_percent=self.gpu_monitor.get_gpu_utilization(),
                tensor_ops_count=self._estimate_tensor_ops(input_data.shape),
                model_name=self.model_name,
                input_shape=input_data.shape,
            )

        except Exception as e:
            execution_time = MicrosecondTimer.elapsed_us(start_time)
            logger.error(f"GPU推論実行エラー: {e}")

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
                    used_bytes = meminfo[1] - meminfo[0]
                    return used_bytes / 1024 / 1024
        except Exception:
            pass
        return 0.0

    def get_comprehensive_gpu_monitoring(self) -> GPUMonitoringData:
        """包括的なGPU監視データの取得"""
        return self.gpu_monitor.get_comprehensive_gpu_monitoring()

    def _estimate_tensor_ops(self, input_shape: Tuple[int, ...]) -> int:
        """テンソル演算数推定"""
        return int(np.prod(input_shape) * 1000)

    def get_session_stats(self) -> Dict[str, Any]:
        """セッション統計取得"""
        stats = self.inference_stats.copy()
        stats.update({
            "model_name": self.model_name,
            "device_id": self.device_id,
            "backend": self.config.backend.value,
            "session_initialized": self.session is not None,
            "input_shape": self.input_shape,
        })
        return stats

    def cleanup(self):
        """セッションリソースクリーンアップ"""
        try:
            if self.tensorrt_engine:
                self.tensorrt_engine.cleanup()
                self.tensorrt_engine = None

            if self.session:
                del self.session
                self.session = None

            logger.debug(f"GPUセッションクリーンアップ完了: {self.model_name}")

        except Exception as e:
            logger.error(f"GPUセッションクリーンアップエラー: {e}")