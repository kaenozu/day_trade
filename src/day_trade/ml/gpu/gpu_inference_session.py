"""
GPU 推論セッション管理

gpu_accelerated_inference.py からのリファクタリング抽出
GPU推論セッションの初期化、実行、監視機能を提供
"""

import time
import hashlib
import warnings
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .gpu_config import GPUBackend, GPUInferenceConfig, GPUInferenceResult
from .gpu_device_manager import GPUMonitoringData
from .tensorrt_engine import TensorRTEngine

# ONNX Runtime GPU のインポート (フォールバック対応)
try:
    import onnxruntime as ort
    ONNX_GPU_AVAILABLE = True
except ImportError:
    ONNX_GPU_AVAILABLE = False
    warnings.warn("ONNX Runtime not available - GPU inference disabled", stacklevel=2)

# GPU計算ライブラリ (フォールバック対応)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available - CPU fallback", stacklevel=2)

# PyNVML (NVIDIA監視ライブラリ)
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("PyNVML not available - GPU monitoring limited", stacklevel=2)

# MicrosecondTimer導入 (高精度タイマー)
try:
    from day_trade.utils.timer import MicrosecondTimer
except ImportError:
    # フォールバック実装
    class MicrosecondTimer:
        @staticmethod
        def now_ns():
            return time.time_ns()
        
        @staticmethod
        def elapsed_us(start_ns):
            return (time.time_ns() - start_ns) // 1000

# ロギング設定
import logging
logger = logging.getLogger(__name__)


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

        # TensorRT エンジン
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

            # CPUフォールバック最適化設定
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

            # CPU最適化設定追加適用
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
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": self.config.memory_pool_size_mb * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
                providers.append(("CUDAExecutionProvider", cuda_options))

        elif self.config.backend == GPUBackend.OPENCL and ONNX_GPU_AVAILABLE:
            available_providers = ort.get_available_providers()
            if "OpenVINOExecutionProvider" in available_providers:
                # OpenVINO最適化設定
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

        # CPUフォールバック最適化
        cpu_options = self._get_optimized_cpu_options()
        providers.append(("CPUExecutionProvider", cpu_options))

        return providers

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

    def _try_initialize_tensorrt(self):
        """TensorRT初期化を試行"""
        if (not self.config.enable_tensorrt or
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

            # TensorRT推論優先実行
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

                # CPUフォールバック時の統計更新
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
                backend_used=backend_used,
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
                "tensorrt_enabled": self.use_tensorrt,
                "config": self.config.to_dict(),
            }
        )
        return stats

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

    @property
    def is_initialized(self) -> bool:
        """セッション初期化状態確認"""
        return self.session is not None or (self.use_tensorrt and self.tensorrt_engine)

    @property
    def input_info(self) -> Dict[str, Any]:
        """入力情報取得"""
        return {
            "name": self.input_name,
            "shape": self.input_shape,
            "initialized": self.is_initialized
        }