#!/usr/bin/env python3
"""
MLモデル推論最適化エンジン
Model Inference Optimization Engine

Issue #761: MLモデル推論パイプラインの高速化と最適化 Phase 1
"""

import os
import time
import logging
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# ML Framework imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

# ログ設定
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ModelOptimizationConfig:
    """モデル最適化設定"""
    # ONNX設定
    enable_onnx: bool = True
    onnx_optimization_level: str = "all"  # "basic", "extended", "all"

    # 量子化設定
    enable_quantization: bool = True
    quantization_type: str = "dynamic"  # "dynamic", "static", "qat"
    quantization_dtype: str = "int8"

    # TensorRT設定
    enable_tensorrt: bool = False  # GPU環境でのみ有効
    tensorrt_precision: str = "fp16"  # "fp32", "fp16", "int8"

    # バッチ処理設定
    enable_batch_inference: bool = True
    max_batch_size: int = 32
    batch_timeout_ms: int = 10

    # パフォーマンス設定
    num_threads: int = mp.cpu_count()
    gpu_memory_fraction: float = 0.8
    enable_memory_optimization: bool = True


@dataclass
class OptimizationMetrics:
    """最適化メトリクス"""
    original_inference_time_ms: float = 0.0
    optimized_inference_time_ms: float = 0.0
    speedup_ratio: float = 0.0
    memory_reduction_ratio: float = 0.0
    model_size_reduction_ratio: float = 0.0
    accuracy_retention_ratio: float = 0.0
    throughput_improvement_ratio: float = 0.0


class BaseModelOptimizer(ABC):
    """モデル最適化基底クラス"""

    @abstractmethod
    def optimize_model(self, model: Any, config: ModelOptimizationConfig) -> Any:
        """モデル最適化"""
        pass

    @abstractmethod
    def benchmark_model(self, model: Any, test_data: np.ndarray) -> Dict[str, float]:
        """モデルベンチマーク"""
        pass


class ONNXOptimizer(BaseModelOptimizer):
    """ONNX Runtime最適化器"""

    def __init__(self):
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available. Install with: pip install onnxruntime")

        self.session_options = ort.SessionOptions()
        self.providers = self._get_available_providers()

    def _get_available_providers(self) -> List[str]:
        """利用可能なExecution Providerを取得"""
        available_providers = ort.get_available_providers()

        # 優先順位順にプロバイダーを選択
        preferred_providers = [
            'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]

        selected_providers = []
        for provider in preferred_providers:
            if provider in available_providers:
                selected_providers.append(provider)

        logger.info(f"Available ONNX providers: {selected_providers}")
        return selected_providers

    def optimize_model(self, model_path: str, config: ModelOptimizationConfig) -> ort.InferenceSession:
        """ONNXモデル最適化"""
        try:
            # セッションオプション設定
            self.session_options.intra_op_num_threads = config.num_threads
            self.session_options.inter_op_num_threads = config.num_threads

            # 最適化レベル設定
            if config.onnx_optimization_level == "basic":
                self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            elif config.onnx_optimization_level == "extended":
                self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            else:  # "all"
                self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # メモリ最適化
            if config.enable_memory_optimization:
                self.session_options.enable_mem_pattern = True
                self.session_options.enable_cpu_mem_arena = True

            # セッション作成
            session = ort.InferenceSession(
                model_path,
                sess_options=self.session_options,
                providers=self.providers
            )

            logger.info(f"ONNX model optimized: {model_path}")
            return session

        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")
            raise

    def benchmark_model(self, session: ort.InferenceSession, test_data: np.ndarray) -> Dict[str, float]:
        """ONNXモデルベンチマーク"""
        try:
            input_name = session.get_inputs()[0].name

            # ウォームアップ
            for _ in range(10):
                session.run(None, {input_name: test_data[:1]})

            # ベンチマーク実行
            times = []
            for _ in range(100):
                start_time = time.perf_counter()
                session.run(None, {input_name: test_data[:1]})
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # ms

            return {
                "avg_inference_time_ms": np.mean(times),
                "min_inference_time_ms": np.min(times),
                "max_inference_time_ms": np.max(times),
                "std_inference_time_ms": np.std(times)
            }

        except Exception as e:
            logger.error(f"ONNX benchmarking failed: {e}")
            return {}


class TensorRTOptimizer(BaseModelOptimizer):
    """TensorRT最適化器"""

    def __init__(self):
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT is not available. GPU optimization will be skipped.")
            self.available = False
        else:
            self.available = True

    def optimize_model(self, onnx_model_path: str, config: ModelOptimizationConfig) -> Optional[str]:
        """TensorRTエンジン作成"""
        if not self.available:
            return None

        try:
            # TensorRTロガー
            trt_logger = trt.Logger(trt.Logger.WARNING)

            # ビルダー作成
            builder = trt.Builder(trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_logger)

            # ONNXモデル読み込み
            with open(onnx_model_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    return None

            # ビルダー設定
            builder_config = builder.create_builder_config()
            builder_config.max_workspace_size = 1 << 30  # 1GB

            # 精度設定
            if config.tensorrt_precision == "fp16":
                builder_config.set_flag(trt.BuilderFlag.FP16)
            elif config.tensorrt_precision == "int8":
                builder_config.set_flag(trt.BuilderFlag.INT8)

            # エンジン構築
            engine = builder.build_engine(network, builder_config)

            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return None

            # エンジン保存
            engine_path = onnx_model_path.replace('.onnx', '.trt')
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())

            logger.info(f"TensorRT engine saved: {engine_path}")
            return engine_path

        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            return None

    def benchmark_model(self, engine_path: str, test_data: np.ndarray) -> Dict[str, float]:
        """TensorRTモデルベンチマーク"""
        if not self.available:
            return {}

        try:
            import pycuda.driver as cuda
            import pycuda.autoinit

            # エンジン読み込み
            with open(engine_path, 'rb') as f:
                engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(f.read())

            # コンテキスト作成
            context = engine.create_execution_context()

            # メモリ割り当て
            input_shape = test_data.shape
            output_shape = (test_data.shape[0], 1)  # 想定出力形状

            d_input = cuda.mem_alloc(test_data.nbytes)
            d_output = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)

            # ベンチマーク実行
            times = []
            for _ in range(100):
                cuda.memcpy_htod(d_input, test_data)

                start_time = time.perf_counter()
                context.execute_v2([int(d_input), int(d_output)])
                cuda.Context.synchronize()
                end_time = time.perf_counter()

                times.append((end_time - start_time) * 1000)  # ms

            return {
                "avg_inference_time_ms": np.mean(times),
                "min_inference_time_ms": np.min(times),
                "max_inference_time_ms": np.max(times),
                "std_inference_time_ms": np.std(times)
            }

        except Exception as e:
            logger.error(f"TensorRT benchmarking failed: {e}")
            return {}


class QuantizationOptimizer:
    """量子化最適化器"""

    def __init__(self):
        self.quantization_types = ["dynamic", "static", "qat"]

    def quantize_onnx_model(self, model_path: str, config: ModelOptimizationConfig) -> str:
        """ONNXモデル量子化"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = model_path.replace('.onnx', '_quantized.onnx')

            if config.quantization_type == "dynamic":
                quantize_dynamic(
                    model_input=model_path,
                    model_output=quantized_path,
                    weight_type=QuantType.QInt8 if config.quantization_dtype == "int8" else QuantType.QUInt8
                )

            logger.info(f"Model quantized: {quantized_path}")
            return quantized_path

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model_path

    def benchmark_quantization_impact(self, original_path: str, quantized_path: str, test_data: np.ndarray) -> Dict[str, float]:
        """量子化の影響評価"""
        try:
            # 元モデル
            original_session = ort.InferenceSession(original_path)
            input_name = original_session.get_inputs()[0].name

            # 量子化モデル
            quantized_session = ort.InferenceSession(quantized_path)

            # 精度比較
            original_output = original_session.run(None, {input_name: test_data})[0]
            quantized_output = quantized_session.run(None, {input_name: test_data})[0]

            # MSE計算
            mse = np.mean((original_output - quantized_output) ** 2)
            accuracy_retention = 1.0 - mse

            # サイズ比較
            original_size = os.path.getsize(original_path)
            quantized_size = os.path.getsize(quantized_path)
            size_reduction = (original_size - quantized_size) / original_size

            return {
                "accuracy_retention": accuracy_retention,
                "size_reduction_ratio": size_reduction,
                "mse": mse
            }

        except Exception as e:
            logger.error(f"Quantization benchmarking failed: {e}")
            return {}


class BatchInferenceEngine:
    """バッチ推論エンジン"""

    def __init__(self, config: ModelOptimizationConfig):
        self.config = config
        self.batch_queue = asyncio.Queue(maxsize=config.max_batch_size * 10)
        self.result_futures: Dict[str, asyncio.Future] = {}
        self.is_running = False

    async def start_batch_processing(self, model_session):
        """バッチ処理開始"""
        self.is_running = True

        while self.is_running:
            batch_data = []
            request_ids = []

            # バッチ収集
            try:
                # 最初のリクエスト待機
                request_id, data = await asyncio.wait_for(
                    self.batch_queue.get(),
                    timeout=self.config.batch_timeout_ms / 1000
                )
                batch_data.append(data)
                request_ids.append(request_id)

                # 追加リクエスト収集（ノンブロッキング）
                while len(batch_data) < self.config.max_batch_size:
                    try:
                        request_id, data = await asyncio.wait_for(
                            self.batch_queue.get(),
                            timeout=0.001  # 1ms
                        )
                        batch_data.append(data)
                        request_ids.append(request_id)
                    except asyncio.TimeoutError:
                        break

                # バッチ推論実行
                if batch_data:
                    await self._process_batch(model_session, batch_data, request_ids)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Batch processing error: {e}")

    async def _process_batch(self, model_session, batch_data: List[np.ndarray], request_ids: List[str]):
        """バッチ処理実行"""
        try:
            # バッチデータ結合
            batch_input = np.vstack(batch_data)

            # 推論実行
            input_name = model_session.get_inputs()[0].name
            batch_output = model_session.run(None, {input_name: batch_input})[0]

            # 結果を個別に分割して返す
            for i, request_id in enumerate(request_ids):
                if request_id in self.result_futures:
                    self.result_futures[request_id].set_result(batch_output[i])
                    del self.result_futures[request_id]

        except Exception as e:
            # エラーの場合は全ての Future にエラーセット
            for request_id in request_ids:
                if request_id in self.result_futures:
                    self.result_futures[request_id].set_exception(e)
                    del self.result_futures[request_id]

    async def predict_async(self, data: np.ndarray) -> np.ndarray:
        """非同期予測"""
        import uuid
        request_id = str(uuid.uuid4())

        # Future作成
        future = asyncio.Future()
        self.result_futures[request_id] = future

        # キューに追加
        await self.batch_queue.put((request_id, data))

        # 結果待機
        return await future

    def stop(self):
        """バッチ処理停止"""
        self.is_running = False


class ModelOptimizationEngine:
    """モデル最適化エンジン"""

    def __init__(self, config: ModelOptimizationConfig):
        self.config = config
        self.optimizers = {
            'onnx': ONNXOptimizer() if ONNX_AVAILABLE else None,
            'tensorrt': TensorRTOptimizer() if TENSORRT_AVAILABLE else None,
            'quantization': QuantizationOptimizer()
        }
        self.batch_engine = BatchInferenceEngine(config) if config.enable_batch_inference else None
        self.metrics = OptimizationMetrics()

    async def optimize_model_pipeline(self, model_path: str, validation_data: np.ndarray) -> Dict[str, Any]:
        """完全な最適化パイプライン実行"""
        results = {
            "original_model": model_path,
            "optimized_models": {},
            "benchmarks": {},
            "metrics": {}
        }

        try:
            # 元モデルベンチマーク
            logger.info("Benchmarking original model...")
            original_metrics = await self._benchmark_original_model(model_path, validation_data)
            results["benchmarks"]["original"] = original_metrics
            self.metrics.original_inference_time_ms = original_metrics.get("avg_inference_time_ms", 0)

            # ONNX最適化
            if self.config.enable_onnx and self.optimizers['onnx']:
                logger.info("Applying ONNX optimization...")
                onnx_session = self.optimizers['onnx'].optimize_model(model_path, self.config)
                onnx_metrics = self.optimizers['onnx'].benchmark_model(onnx_session, validation_data)
                results["optimized_models"]["onnx"] = onnx_session
                results["benchmarks"]["onnx"] = onnx_metrics

                # 量子化
                if self.config.enable_quantization:
                    logger.info("Applying quantization...")
                    quantized_path = self.optimizers['quantization'].quantize_onnx_model(model_path, self.config)
                    quantized_session = ort.InferenceSession(quantized_path)
                    quantized_metrics = self.optimizers['onnx'].benchmark_model(quantized_session, validation_data)
                    results["optimized_models"]["quantized"] = quantized_session
                    results["benchmarks"]["quantized"] = quantized_metrics

                    # 量子化影響評価
                    quant_impact = self.optimizers['quantization'].benchmark_quantization_impact(
                        model_path, quantized_path, validation_data
                    )
                    results["benchmarks"]["quantization_impact"] = quant_impact

            # TensorRT最適化
            if self.config.enable_tensorrt and self.optimizers['tensorrt']:
                logger.info("Applying TensorRT optimization...")
                tensorrt_engine = self.optimizers['tensorrt'].optimize_model(model_path, self.config)
                if tensorrt_engine:
                    tensorrt_metrics = self.optimizers['tensorrt'].benchmark_model(tensorrt_engine, validation_data)
                    results["benchmarks"]["tensorrt"] = tensorrt_metrics

            # 最適化メトリクス計算
            self._calculate_optimization_metrics(results["benchmarks"])
            results["metrics"] = asdict(self.metrics)

            # バッチ推論テスト
            if self.batch_engine and "onnx" in results["optimized_models"]:
                logger.info("Testing batch inference...")
                batch_metrics = await self._test_batch_inference(
                    results["optimized_models"]["onnx"], validation_data
                )
                results["benchmarks"]["batch"] = batch_metrics

            logger.info("Model optimization pipeline completed")
            return results

        except Exception as e:
            logger.error(f"Optimization pipeline failed: {e}")
            raise

    async def _benchmark_original_model(self, model_path: str, test_data: np.ndarray) -> Dict[str, float]:
        """元モデルベンチマーク"""
        try:
            # 拡張子に基づいてモデルタイプ判定
            if model_path.endswith('.onnx'):
                session = ort.InferenceSession(model_path)
                return self.optimizers['onnx'].benchmark_model(session, test_data)
            else:
                # その他のフォーマットは未実装
                logger.warning(f"Unsupported model format: {model_path}")
                return {}
        except Exception as e:
            logger.error(f"Original model benchmarking failed: {e}")
            return {}

    def _calculate_optimization_metrics(self, benchmarks: Dict[str, Dict[str, float]]):
        """最適化メトリクス計算"""
        try:
            original_time = benchmarks.get("original", {}).get("avg_inference_time_ms", 0)

            # 最も高速な最適化版を選択
            best_time = original_time
            best_variant = "original"

            for variant, metrics in benchmarks.items():
                if variant != "original" and "avg_inference_time_ms" in metrics:
                    time_ms = metrics["avg_inference_time_ms"]
                    if time_ms < best_time:
                        best_time = time_ms
                        best_variant = variant

            if original_time > 0:
                self.metrics.optimized_inference_time_ms = best_time
                self.metrics.speedup_ratio = original_time / best_time
                self.metrics.throughput_improvement_ratio = best_time / original_time if best_time > 0 else 0

            # 量子化メトリクス
            if "quantization_impact" in benchmarks:
                impact = benchmarks["quantization_impact"]
                self.metrics.model_size_reduction_ratio = impact.get("size_reduction_ratio", 0)
                self.metrics.accuracy_retention_ratio = impact.get("accuracy_retention", 0)

            logger.info(f"Optimization complete: {original_time:.2f}ms -> {best_time:.2f}ms ({self.metrics.speedup_ratio:.2f}x speedup)")

        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")

    async def _test_batch_inference(self, model_session, test_data: np.ndarray) -> Dict[str, float]:
        """バッチ推論テスト"""
        try:
            if not self.batch_engine:
                return {}

            # バッチ処理開始
            batch_task = asyncio.create_task(
                self.batch_engine.start_batch_processing(model_session)
            )

            # 複数の非同期リクエスト送信
            tasks = []
            start_time = time.perf_counter()

            for i in range(100):
                task = asyncio.create_task(
                    self.batch_engine.predict_async(test_data[i:i+1])
                )
                tasks.append(task)

            # 全リクエスト完了待機
            results = await asyncio.gather(*tasks)
            end_time = time.perf_counter()

            # バッチ処理停止
            self.batch_engine.stop()
            batch_task.cancel()

            total_time = (end_time - start_time) * 1000  # ms
            throughput = len(tasks) / (total_time / 1000)  # predictions/sec

            return {
                "batch_total_time_ms": total_time,
                "batch_throughput_predictions_per_sec": throughput,
                "batch_avg_latency_ms": total_time / len(tasks)
            }

        except Exception as e:
            logger.error(f"Batch inference testing failed: {e}")
            return {}


# 使用例とテスト
async def test_model_optimization():
    """モデル最適化テスト"""

    # 設定
    config = ModelOptimizationConfig(
        enable_onnx=True,
        enable_quantization=True,
        enable_batch_inference=True,
        max_batch_size=16,
        num_threads=4
    )

    # 最適化エンジン初期化
    engine = ModelOptimizationEngine(config)

    try:
        # テスト用ダミーデータ
        test_data = np.random.randn(100, 10).astype(np.float32)

        # 注意: 実際のモデルパスを指定する必要があります
        model_path = "dummy_model.onnx"  # 実際のパスに置き換え

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            logger.info("Creating dummy ONNX model for testing...")
            # ダミーモデル作成（実装省略）
            return

        print("Starting model optimization pipeline...")

        # 最適化パイプライン実行
        results = await engine.optimize_model_pipeline(model_path, test_data)

        # 結果表示
        print("\n=== Optimization Results ===")
        for variant, metrics in results["benchmarks"].items():
            print(f"\n{variant.upper()}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.3f}")

        print(f"\n=== Overall Metrics ===")
        metrics = results["metrics"]
        print(f"Speedup: {metrics['speedup_ratio']:.2f}x")
        print(f"Memory reduction: {metrics['memory_reduction_ratio']:.1%}")
        print(f"Model size reduction: {metrics['model_size_reduction_ratio']:.1%}")
        print(f"Accuracy retention: {metrics['accuracy_retention_ratio']:.1%}")

        return results

    except Exception as e:
        print(f"Optimization test failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(test_model_optimization())