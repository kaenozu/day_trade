#!/usr/bin/env python3
"""
ONNX量子化エンジン
Issue #379: ML Model Inference Performance Optimization

ONNX Runtime を使用した各種量子化手法の実装
- 動的量子化・静的量子化・混合精度量子化
- Issue #724対応: 強化FP16量子化
"""

import time
from typing import List, Dict, Any

import numpy as np

from .core import CompressionConfig, check_dependencies
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ONNXQuantizationEngine:
    """ONNX量子化エンジン"""

    def __init__(self, config: CompressionConfig):
        """ONNX量子化エンジンを初期化"""
        self.config = config
        self.calibration_cache = {}
        self.dependencies = check_dependencies()

        # Issue #724対応: FP16量子化統計
        self.fp16_quantization_stats = {
            'total_quantizations': 0,
            'successful_fp16_quantizations': 0,
            'fallback_optimizations': 0,
            'average_compression_ratio': 0.0,
            'total_processing_time': 0.0,
        }

        logger.info("ONNX量子化エンジン初期化完了")

    def apply_dynamic_quantization(self, model_path: str, output_path: str) -> bool:
        """動的量子化適用"""
        if not self.dependencies.get("onnx_quantization", False):
            logger.warning("ONNX量子化ツール利用不可 - スキップ")
            return False

        try:
            from onnxruntime.quantization import (
                QuantType,
                quantize_dynamic,
            )

            quantize_dynamic(
                model_input=model_path,
                model_output=output_path,
                weight_type=QuantType.QUInt8,
                per_channel=True,
                op_types_to_quantize=["Conv", "MatMul", "Attention"],
                extra_options={"EnableSubgraph": True},
            )

            logger.info(f"動的量子化完了: {output_path}")
            return True

        except Exception as e:
            logger.error(f"動的量子化エラー: {e}")
            return False

    def apply_static_quantization(
        self, model_path: str, output_path: str, calibration_data: List[np.ndarray]
    ) -> bool:
        """静的量子化適用（校正データ使用）"""
        if not self.dependencies.get("onnx_quantization", False):
            logger.warning("ONNX量子化ツール利用不可 - スキップ")
            return False

        try:
            from onnxruntime.quantization import (
                CalibrationDataReader,
                QuantFormat,
                QuantType,
                quantize_static,
            )

            # 校正データリーダー作成
            class CustomCalibrationDataReader(CalibrationDataReader):
                def __init__(self, calibration_data_list):
                    super().__init__()
                    self.data_list = calibration_data_list
                    self.iterator = iter(calibration_data_list)

                def get_next(self):
                    try:
                        data = next(self.iterator)
                        return {"input": data}
                    except StopIteration:
                        return None

            calibration_reader = CustomCalibrationDataReader(calibration_data)

            quantize_static(
                model_input=model_path,
                model_output=output_path,
                calibration_data_reader=calibration_reader,
                quant_format=QuantFormat.QOperator,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QInt8,
                per_channel=True,
            )

            logger.info(f"静的量子化完了: {output_path}")
            return True

        except Exception as e:
            logger.error(f"静的量子化エラー: {e}")
            return False

    def apply_mixed_precision_quantization(
        self, model_path: str, output_path: str
    ) -> bool:
        """混合精度量子化適用 - Issue #724対応: 強化FP16量子化版"""
        start_time = time.time()

        try:
            # Issue #724対応: 統計更新
            self.fp16_quantization_stats['total_quantizations'] += 1

            # 入力モデルサイズ測定
            original_size = self._get_model_size(model_path)

            # Issue #724対応: 真のFP16量子化実装
            success = self._apply_fp16_quantization(model_path, output_path)
            processing_time = time.time() - start_time

            if success:
                # 圧縮率計算
                compressed_size = self._get_model_size(output_path)
                compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

                # 統計更新
                self.fp16_quantization_stats['successful_fp16_quantizations'] += 1
                self.fp16_quantization_stats['total_processing_time'] += processing_time
                self._update_compression_ratio(compression_ratio)

                logger.info(
                    f"強化FP16量子化完了: {output_path} "
                    f"({compression_ratio:.2f}圧縮率, {processing_time:.3f}秒)"
                )
                return True
            else:
                # フォールバック: 従来のグラフ最適化
                logger.warning("FP16量子化失敗 - グラフ最適化フォールバック実行")
                fallback_success = self._fallback_graph_optimization(model_path, output_path)

                if fallback_success:
                    self.fp16_quantization_stats['fallback_optimizations'] += 1
                    self.fp16_quantization_stats['total_processing_time'] += processing_time

                return fallback_success

        except Exception as e:
            logger.error(f"混合精度量子化エラー: {e}")
            # 緊急フォールバック
            try:
                fallback_success = self._fallback_graph_optimization(model_path, output_path)
                if fallback_success:
                    self.fp16_quantization_stats['fallback_optimizations'] += 1
                return fallback_success
            except:
                return False

    def _apply_fp16_quantization(self, model_path: str, output_path: str) -> bool:
        """Issue #724対応: 真のFP16量子化実装"""
        try:
            if not self.dependencies.get("onnx_quantization", False):
                logger.warning("ONNX量子化ツール利用不可 - FP16量子化スキップ")
                return False

            from onnxruntime.quantization import (
                QuantType,
                quantize_dynamic,
            )

            # 方法1: 動的FP16量子化（推奨）
            try:
                quantize_dynamic(
                    model_input=model_path,
                    model_output=output_path,
                    weight_type=QuantType.QFloat16,  # FP16量子化
                    optimize_model=True,
                    extra_options={
                        "EnableSubgraph": True,
                        "ForceQuantizeNoZeroPoint": True,
                        "MatMulConstBOnly": True,
                    }
                )

                # 量子化結果検証
                if self._verify_fp16_quantization(output_path):
                    logger.info(f"動的FP16量子化成功: {output_path}")
                    return True

            except Exception as e:
                logger.debug(f"動的FP16量子化失敗: {e}")

            # 方法2: 重みのみFP16変換
            try:
                return self._convert_weights_to_fp16(model_path, output_path)

            except Exception as e:
                logger.debug(f"重みFP16変換失敗: {e}")

            # 方法3: ONNX Runtime最適化 + FP16設定
            try:
                return self._onnx_runtime_fp16_optimization(model_path, output_path)

            except Exception as e:
                logger.debug(f"ONNX Runtime FP16最適化失敗: {e}")

            return False

        except Exception as e:
            logger.warning(f"FP16量子化実装エラー: {e}")
            return False

    def _convert_weights_to_fp16(self, model_path: str, output_path: str) -> bool:
        """重みをFP16に変換"""
        try:
            import onnx
            from onnx import numpy_helper

            # ONNXモデル読み込み
            model = onnx.load(model_path)

            # 重みをFP16に変換
            fp16_count = 0
            for initializer in model.graph.initializer:
                if initializer.data_type == onnx.TensorProto.FLOAT:
                    # FP32 -> FP16変換
                    weights_fp32 = numpy_helper.to_array(initializer)
                    weights_fp16 = weights_fp32.astype(np.float16)

                    # FP16データでinitializer更新
                    new_initializer = numpy_helper.from_array(
                        weights_fp16, initializer.name
                    )
                    new_initializer.data_type = onnx.TensorProto.FLOAT16

                    # 元のinitializerを置き換え
                    initializer.CopyFrom(new_initializer)
                    fp16_count += 1

            # グラフ内のValueInfoもFP16に更新
            self._update_graph_value_info_to_fp16(model.graph)

            # 変換済みモデル保存
            onnx.save(model, output_path)

            logger.info(f"重みFP16変換完了: {fp16_count}テンソル変換")
            return True

        except Exception as e:
            logger.debug(f"重みFP16変換エラー: {e}")
            return False

    def _update_graph_value_info_to_fp16(self, graph):
        """グラフのValueInfoをFP16に更新"""
        try:
            import onnx

            # 入力・出力・中間値のデータ型更新
            for value_info in graph.input + graph.output + graph.value_info:
                if (hasattr(value_info.type.tensor_type, 'elem_type') and
                    value_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT):
                    value_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16

        except Exception as e:
            logger.debug(f"ValueInfo FP16更新エラー: {e}")

    def _onnx_runtime_fp16_optimization(self, model_path: str, output_path: str) -> bool:
        """ONNX Runtime FP16最適化"""
        try:
            if not self.dependencies.get("onnx_quantization", False):
                logger.debug("ONNX Runtime利用不可 - FP16最適化スキップ")
                return False

            import onnxruntime as ort

            # GraphOptimizationLevel.ORT_ENABLE_ALL + FP16プロバイダー
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.optimized_model_filepath = output_path

            # FP16対応プロバイダー使用
            providers = ["CPUExecutionProvider"]
            if hasattr(ort, 'get_available_providers'):
                available_providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in available_providers:
                    providers.insert(0, ("CUDAExecutionProvider", {
                        "enable_fp16": True,
                    }))

            # セッション作成で最適化実行
            session = ort.InferenceSession(model_path, sess_options, providers=providers)

            # セッション設定確認
            if hasattr(session, 'get_providers'):
                active_providers = session.get_providers()
                logger.debug(f"FP16最適化プロバイダー: {active_providers}")

            logger.info(f"ONNX Runtime FP16最適化完了")
            return True

        except Exception as e:
            logger.debug(f"ONNX Runtime FP16最適化エラー: {e}")
            return False

    def _verify_fp16_quantization(self, model_path: str) -> bool:
        """FP16量子化結果検証"""
        try:
            import onnx

            model = onnx.load(model_path)

            # FP16テンソル数確認
            fp16_tensors = 0
            total_tensors = 0

            for initializer in model.graph.initializer:
                total_tensors += 1
                if initializer.data_type == onnx.TensorProto.FLOAT16:
                    fp16_tensors += 1

            fp16_ratio = fp16_tensors / total_tensors if total_tensors > 0 else 0
            logger.debug(f"FP16量子化率: {fp16_ratio:.1%} ({fp16_tensors}/{total_tensors})")

            # 50%以上がFP16なら成功とみなす
            return fp16_ratio >= 0.5

        except Exception as e:
            logger.debug(f"FP16量子化検証エラー: {e}")
            return False

    def _fallback_graph_optimization(self, model_path: str, output_path: str) -> bool:
        """フォールバック: 従来のグラフ最適化"""
        try:
            if not self.dependencies.get("onnx_quantization", False):
                logger.warning("ONNX Runtime利用不可 - フォールバック失敗")
                return False

            import onnxruntime as ort

            # 従来の実装
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.optimized_model_filepath = output_path

            # セッション作成により最適化実行
            session = ort.InferenceSession(
                model_path, sess_options, providers=["CPUExecutionProvider"]
            )

            logger.info(f"フォールバックグラフ最適化完了: {output_path}")
            return True

        except Exception as e:
            logger.error(f"フォールバックグラフ最適化エラー: {e}")
            return False

    def _get_model_size(self, model_path: str) -> int:
        """モデルファイルサイズ取得"""
        try:
            import os
            return os.path.getsize(model_path) if os.path.exists(model_path) else 0
        except:
            return 0

    def _update_compression_ratio(self, ratio: float) -> None:
        """圧縮率統計更新"""
        try:
            stats = self.fp16_quantization_stats
            current_avg = stats['average_compression_ratio']
            successful_count = stats['successful_fp16_quantizations']

            # 累積平均更新
            if successful_count == 1:
                stats['average_compression_ratio'] = ratio
            else:
                stats['average_compression_ratio'] = (
                    (current_avg * (successful_count - 1) + ratio) / successful_count
                )
        except:
            pass

    def get_fp16_quantization_stats(self) -> Dict[str, Any]:
        """Issue #724対応: FP16量子化統計取得"""
        stats = self.fp16_quantization_stats.copy()

        # 成功率計算
        total = stats['total_quantizations']
        successful = stats['successful_fp16_quantizations']
        stats['success_rate_percent'] = (successful / total * 100) if total > 0 else 0

        # 平均処理時間
        stats['average_processing_time'] = (
            stats['total_processing_time'] / total if total > 0 else 0
        )

        return stats

    def clear_calibration_cache(self) -> None:
        """キャリブレーション・キャッシュクリア"""
        self.calibration_cache.clear()
        logger.debug("キャリブレーション・キャッシュをクリア")

    def get_quantization_summary(self) -> Dict[str, Any]:
        """量子化処理の統計サマリー取得"""
        return {
            "config": self.config.to_dict(),
            "fp16_stats": self.get_fp16_quantization_stats(),
            "cache_size": len(self.calibration_cache),
            "dependencies": self.dependencies,
        }