"""
TensorRT 推論エンジン

gpu_accelerated_inference.py からのリファクタリング抽出
TensorRT エンジンの構築、管理、推論実行機能を提供
"""

import time
import warnings
import numpy as np
from typing import Any, Dict, List, Optional

from .gpu_config import GPUInferenceConfig

# TensorRT と PyCUDA のインポート (フォールバック対応)
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    warnings.warn("TensorRT not available - TensorRT features disabled", stacklevel=2)

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    warnings.warn("PyCUDA not available - CUDA memory management disabled", stacklevel=2)

# ロギング設定
import logging
logger = logging.getLogger(__name__)


class TensorRTEngine:
    """TensorRT推論エンジン"""

    def __init__(self, config: GPUInferenceConfig, device_id: int = 0):
        self.config = config
        self.device_id = device_id
        self.engine = None
        self.context = None
        self.bindings = None
        self.stream = None

        # メモリ管理
        self.inputs = []
        self.outputs = []
        self.allocations = []

        # TensorRT設定
        if TENSORRT_AVAILABLE:
            self.logger = trt.Logger(trt.Logger.INFO)
        else:
            self.logger = None
        self.builder = None
        self.network = None

    def build_engine_from_onnx(self, onnx_model_path: str) -> bool:
        """ONNXモデルからTensorRTエンジンを構築"""
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT利用不可 - エンジン構築スキップ")
            return False

        try:
            # TensorRTビルダー作成
            self.builder = trt.Builder(self.logger)
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            self.network = self.builder.create_network(network_flags)

            # ONNX パーサー
            parser = trt.OnnxParser(self.network, self.logger)

            # ONNXファイル読み込み
            with open(onnx_model_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("ONNX モデル解析失敗")
                    for error in range(parser.num_errors):
                        logger.error(f"  エラー {error}: {parser.get_error(error)}")
                    return False

            # ビルダー設定
            builder_config = self.builder.create_builder_config()

            # ワークスペースサイズ設定
            builder_config.max_workspace_size = self.config.tensorrt_max_workspace_size * (1024 ** 2)

            # 精度設定
            if self.config.tensorrt_precision == "fp16":
                builder_config.set_flag(trt.BuilderFlag.FP16)
                logger.info("TensorRT FP16精度有効化")
            elif self.config.tensorrt_precision == "int8":
                builder_config.set_flag(trt.BuilderFlag.INT8)
                logger.info("TensorRT INT8精度有効化")

            # DLA設定（Jetson等でのみ有効）
            if self.config.tensorrt_enable_dla and self.config.tensorrt_dla_core >= 0:
                builder_config.default_device_type = trt.DeviceType.DLA
                builder_config.DLA_core = self.config.tensorrt_dla_core
                builder_config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
                logger.info(f"TensorRT DLA有効化: コア {self.config.tensorrt_dla_core}")

            # タイミングキャッシュ
            if self.config.tensorrt_enable_timing_cache:
                cache = builder_config.create_timing_cache(b"")
                builder_config.set_timing_cache(cache, ignore_mismatch=False)
                logger.info("TensorRT タイミングキャッシュ有効化")

            # 最適化プロファイル設定（動的バッチサイズ対応）
            profile = self.builder.create_optimization_profile()

            # 入力テンソルのプロファイル設定
            for i in range(self.network.num_inputs):
                input_tensor = self.network.get_input(i)
                input_shape = input_tensor.shape

                # 動的バッチサイズの設定
                min_shape = list(input_shape)
                opt_shape = list(input_shape)
                max_shape = list(input_shape)

                if min_shape[0] == -1:  # バッチ次元が動的
                    min_shape[0] = 1
                    opt_shape[0] = self.config.tensorrt_max_batch_size // 2
                    max_shape[0] = self.config.tensorrt_max_batch_size

                profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
                logger.debug(f"入力プロファイル {input_tensor.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

            builder_config.add_optimization_profile(profile)

            # エンジン構築
            logger.info("TensorRTエンジン構築開始...")
            start_time = time.time()

            self.engine = self.builder.build_engine(self.network, builder_config)

            build_time = time.time() - start_time

            if self.engine is None:
                logger.error("TensorRTエンジン構築失敗")
                return False

            logger.info(f"TensorRTエンジン構築完了: {build_time:.2f}秒")

            # 実行コンテキスト作成
            self.context = self.engine.create_execution_context()

            # メモリ割り当て準備
            self._prepare_memory_allocations()

            return True

        except Exception as e:
            logger.error(f"TensorRTエンジン構築エラー: {e}")
            return False

    def _prepare_memory_allocations(self):
        """メモリ割り当て準備"""
        if not PYCUDA_AVAILABLE or self.engine is None:
            return

        try:
            # CUDAストリーム作成
            self.stream = cuda.Stream()

            # 入力・出力テンソル情報取得
            self.inputs = []
            self.outputs = []
            self.bindings = []
            self.allocations = []

            for binding in self.engine:
                binding_idx = self.engine.get_binding_index(binding)
                size = trt.volume(self.context.get_binding_shape(binding_idx))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))

                # GPUメモリ割り当て
                device_mem = cuda.mem_alloc(size * dtype().itemsize)
                self.allocations.append(device_mem)
                self.bindings.append(int(device_mem))

                if self.engine.binding_is_input(binding):
                    self.inputs.append({
                        'name': binding,
                        'index': binding_idx,
                        'size': size,
                        'dtype': dtype,
                        'device_mem': device_mem
                    })
                else:
                    self.outputs.append({
                        'name': binding,
                        'index': binding_idx,
                        'size': size,
                        'dtype': dtype,
                        'device_mem': device_mem
                    })

            logger.info(f"TensorRTメモリ割り当て完了: 入力={len(self.inputs)}, 出力={len(self.outputs)}")

        except Exception as e:
            logger.error(f"TensorRTメモリ割り当てエラー: {e}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """TensorRT推論実行"""
        if not PYCUDA_AVAILABLE or self.engine is None or self.context is None:
            raise RuntimeError("TensorRT エンジン未初期化")

        try:
            # 入力データの前処理
            batch_size = input_data.shape[0]

            # バッチサイズに応じて動的形状設定
            if len(self.inputs) > 0:
                input_binding = self.inputs[0]
                input_shape = list(input_data.shape)

                if not self.context.set_binding_shape(input_binding['index'], input_shape):
                    raise RuntimeError(f"入力形状設定失敗: {input_shape}")

            # 入力データをGPUメモリにコピー
            for i, input_info in enumerate(self.inputs):
                host_mem = np.ascontiguousarray(input_data.astype(input_info['dtype']))
                cuda.memcpy_htod_async(input_info['device_mem'], host_mem, self.stream)

            # 推論実行
            success = self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )

            if not success:
                raise RuntimeError("TensorRT推論実行失敗")

            # 出力データをCPUメモリにコピー
            outputs = []
            for output_info in self.outputs:
                # 出力形状取得
                output_shape = self.context.get_binding_shape(output_info['index'])
                output_size = trt.volume(output_shape)

                # CPUメモリ準備
                host_mem = np.empty(output_size, dtype=output_info['dtype'])

                # GPUからCPUへコピー
                cuda.memcpy_dtoh_async(host_mem, output_info['device_mem'], self.stream)

                # 結果をリシェイプ
                host_mem = host_mem.reshape(output_shape)
                outputs.append(host_mem)

            # ストリーム同期
            self.stream.synchronize()

            # 結果返却（複数出力の場合は最初の出力）
            return outputs[0] if len(outputs) > 0 else np.array([])

        except Exception as e:
            logger.error(f"TensorRT推論エラー: {e}")
            raise

    def save_engine(self, engine_path: str) -> bool:
        """TensorRTエンジンをファイルに保存"""
        if self.engine is None:
            return False

        try:
            with open(engine_path, 'wb') as f:
                f.write(self.engine.serialize())

            logger.info(f"TensorRTエンジン保存完了: {engine_path}")
            return True

        except Exception as e:
            logger.error(f"TensorRTエンジン保存エラー: {e}")
            return False

    def load_engine(self, engine_path: str) -> bool:
        """保存されたTensorRTエンジンを読み込み"""
        if not TENSORRT_AVAILABLE:
            return False

        try:
            runtime = trt.Runtime(self.logger)

            with open(engine_path, 'rb') as f:
                engine_data = f.read()

            self.engine = runtime.deserialize_cuda_engine(engine_data)

            if self.engine is None:
                logger.error("TensorRTエンジン読み込み失敗")
                return False

            self.context = self.engine.create_execution_context()
            self._prepare_memory_allocations()

            logger.info(f"TensorRTエンジン読み込み完了: {engine_path}")
            return True

        except Exception as e:
            logger.error(f"TensorRTエンジン読み込みエラー: {e}")
            return False

    def cleanup(self):
        """リソースクリーンアップ"""
        try:
            # GPU メモリ解放
            for allocation in self.allocations:
                if allocation:
                    allocation.free()
            self.allocations.clear()

            # ストリーム削除
            if self.stream:
                del self.stream
                self.stream = None

            # コンテキスト削除
            if self.context:
                del self.context
                self.context = None

            # エンジン削除
            if self.engine:
                del self.engine
                self.engine = None

            logger.debug("TensorRTエンジンクリーンアップ完了")

        except Exception as e:
            logger.error(f"TensorRTエンジンクリーンアップエラー: {e}")

    @property
    def is_initialized(self) -> bool:
        """初期化状態の確認"""
        return self.engine is not None and self.context is not None

    @property
    def batch_size_range(self) -> Dict[str, int]:
        """サポートされるバッチサイズ範囲"""
        return {
            "min": 1,
            "max": self.config.tensorrt_max_batch_size,
            "optimal": self.config.tensorrt_max_batch_size // 2
        }

    def get_engine_info(self) -> Dict[str, Any]:
        """エンジン情報取得"""
        if not self.is_initialized:
            return {"status": "未初期化"}

        try:
            return {
                "status": "初期化済み",
                "device_id": self.device_id,
                "input_count": len(self.inputs),
                "output_count": len(self.outputs),
                "precision": self.config.tensorrt_precision,
                "max_batch_size": self.config.tensorrt_max_batch_size,
                "workspace_size_mb": self.config.tensorrt_max_workspace_size,
                "dla_enabled": self.config.tensorrt_enable_dla,
                "inputs": [{"name": inp["name"], "size": inp["size"], "dtype": str(inp["dtype"])} for inp in self.inputs],
                "outputs": [{"name": out["name"], "size": out["size"], "dtype": str(out["dtype"])} for out in self.outputs]
            }
        except Exception as e:
            logger.error(f"エンジン情報取得エラー: {e}")
            return {"status": "エラー", "error": str(e)}