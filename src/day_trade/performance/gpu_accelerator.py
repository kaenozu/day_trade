#!/usr/bin/env python3
"""
GPU加速処理エンジン
Issue #434: 本番環境パフォーマンス最終最適化 - GPU加速処理完全最適化

CUDAによる大規模並列計算とTensorFlowによる機械学習推論の最適化
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# GPU関連ライブラリ（オプショナル）
try:
    import cupy as cp
    import cupy.cuda.runtime as cuda_runtime

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import tensorflow as tf

    tf.config.experimental.set_memory_growth = True
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch

    if torch.cuda.is_available():
        TORCH_AVAILABLE = True
    else:
        TORCH_AVAILABLE = False
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


@dataclass
class GPUConfig:
    """GPU加速設定"""

    # GPU選択
    device_id: int = 0
    use_multi_gpu: bool = False
    gpu_memory_limit_mb: int = 2048

    # 並列処理設定
    batch_size: int = 1024
    max_concurrent_streams: int = 4

    # 最適化設定
    enable_tensor_cores: bool = True
    enable_mixed_precision: bool = True
    memory_pool_enabled: bool = True

    # フォールバック設定
    cpu_fallback: bool = True
    min_data_size_for_gpu: int = 100


class GPUMemoryManager:
    """GPU メモリ管理システム"""

    def __init__(self, config: GPUConfig):
        self.config = config
        self.memory_pools = {}
        self.allocation_stats = {
            "total_allocations": 0,
            "total_deallocations": 0,
            "current_usage_mb": 0,
            "peak_usage_mb": 0,
        }
        self._stats_lock = threading.RLock()

        if CUPY_AVAILABLE:
            self._init_cupy_memory_pool()

        logger.info("GPU メモリマネージャー初期化完了")

    def _init_cupy_memory_pool(self):
        """CuPy メモリプール初期化"""
        try:
            # デバイス設定
            cp.cuda.Device(self.config.device_id).use()

            # メモリプール設定
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()

            # メモリ制限設定
            if self.config.gpu_memory_limit_mb > 0:
                mempool.set_limit(size=self.config.gpu_memory_limit_mb * 1024 * 1024)

            self.memory_pools["cupy"] = mempool
            self.memory_pools["pinned"] = pinned_mempool

            logger.info(
                f"CuPy メモリプール初期化: {self.config.gpu_memory_limit_mb}MB制限"
            )

        except Exception as e:
            logger.error(f"CuPy メモリプール初期化失敗: {e}")

    def allocate_gpu_array(
        self, shape: Tuple, dtype=np.float32
    ) -> Optional["cp.ndarray"]:
        """GPU配列割り当て"""
        if not CUPY_AVAILABLE:
            return None

        try:
            with self._stats_lock:
                array = cp.zeros(shape, dtype=dtype)

                # 統計更新
                size_mb = array.nbytes / (1024 * 1024)
                self.allocation_stats["total_allocations"] += 1
                self.allocation_stats["current_usage_mb"] += size_mb
                self.allocation_stats["peak_usage_mb"] = max(
                    self.allocation_stats["peak_usage_mb"],
                    self.allocation_stats["current_usage_mb"],
                )

                return array

        except Exception as e:
            logger.error(f"GPU配列割り当て失敗: {e}")
            return None

    def deallocate_gpu_array(self, array: "cp.ndarray"):
        """GPU配列解放"""
        if not CUPY_AVAILABLE or array is None:
            return

        try:
            with self._stats_lock:
                size_mb = array.nbytes / (1024 * 1024)
                self.allocation_stats["total_deallocations"] += 1
                self.allocation_stats["current_usage_mb"] -= size_mb

                # 明示的メモリ解放
                del array

        except Exception as e:
            logger.error(f"GPU配列解放失敗: {e}")

    def get_memory_info(self) -> Dict[str, Any]:
        """メモリ情報取得"""
        info = {"cpu_fallback": True}

        if CUPY_AVAILABLE:
            try:
                meminfo = cp.cuda.runtime.memGetInfo()
                total_memory_mb = meminfo[1] / (1024 * 1024)
                free_memory_mb = meminfo[0] / (1024 * 1024)
                used_memory_mb = total_memory_mb - free_memory_mb

                info.update(
                    {
                        "gpu_available": True,
                        "total_memory_mb": total_memory_mb,
                        "free_memory_mb": free_memory_mb,
                        "used_memory_mb": used_memory_mb,
                        "utilization_percent": (used_memory_mb / total_memory_mb) * 100,
                        "allocation_stats": self.allocation_stats.copy(),
                    }
                )

            except Exception as e:
                logger.error(f"GPU メモリ情報取得失敗: {e}")

        return info

    def cleanup(self):
        """メモリクリーンアップ"""
        if CUPY_AVAILABLE and "cupy" in self.memory_pools:
            self.memory_pools["cupy"].free_all_blocks()
            logger.info("GPU メモリクリーンアップ完了")


class CudaKernelManager:
    """CUDA カーネル管理システム"""

    def __init__(self, config: GPUConfig):
        self.config = config
        self.compiled_kernels = {}
        self.streams = []

        if CUPY_AVAILABLE:
            self._init_cuda_streams()
            self._compile_kernels()

    def _init_cuda_streams(self):
        """CUDA ストリーム初期化"""
        try:
            for i in range(self.config.max_concurrent_streams):
                stream = cp.cuda.Stream(non_blocking=True)
                self.streams.append(stream)

            logger.info(f"CUDA ストリーム初期化: {len(self.streams)}ストリーム")

        except Exception as e:
            logger.error(f"CUDA ストリーム初期化失敗: {e}")

    def _compile_kernels(self):
        """カスタムCUDAカーネルコンパイル"""
        if not CUPY_AVAILABLE:
            return

        try:
            # 高速特徴量計算カーネル
            feature_kernel_code = """
            extern "C" __global__
            void compute_technical_features(
                const float* prices, const float* volumes,
                float* features, int n, int feature_dim
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= n) return;

                int base_idx = idx * feature_dim;

                if (idx >= 20) {  // 十分な履歴がある場合
                    // 移動平均5
                    float ma5 = 0.0f;
                    for (int i = 0; i < 5; i++) {
                        ma5 += prices[idx - i];
                    }
                    features[base_idx] = ma5 / 5.0f;

                    // 移動平均20
                    float ma20 = 0.0f;
                    for (int i = 0; i < 20; i++) {
                        ma20 += prices[idx - i];
                    }
                    features[base_idx + 1] = ma20 / 20.0f;

                    // RSI計算
                    float gain = 0.0f, loss = 0.0f;
                    for (int i = 1; i < 15; i++) {
                        float diff = prices[idx - i + 1] - prices[idx - i];
                        if (diff > 0) gain += diff;
                        else loss -= diff;
                    }
                    float rs = gain / fmaxf(loss, 1e-8f);
                    features[base_idx + 2] = 100.0f - (100.0f / (1.0f + rs));

                    // ボリューム比率
                    float vol_ma = 0.0f;
                    for (int i = 0; i < 5; i++) {
                        vol_ma += volumes[idx - i];
                    }
                    features[base_idx + 3] = volumes[idx] / fmaxf(vol_ma / 5.0f, 1e-8f);

                    // 価格変化率
                    features[base_idx + 4] = (prices[idx] - prices[idx - 1]) /
                                             fmaxf(prices[idx - 1], 1e-8f);

                    // 現在価格・ボリューム
                    features[base_idx + 5] = prices[idx];
                    features[base_idx + 6] = volumes[idx];
                } else {
                    // 不十分なデータの場合はゼロ埋め
                    for (int i = 0; i < feature_dim; i++) {
                        features[base_idx + i] = 0.0f;
                    }
                }
            }
            """

            self.compiled_kernels["technical_features"] = cp.RawKernel(
                feature_kernel_code, "compute_technical_features"
            )

            # 高速予測カーネル
            prediction_kernel_code = """
            extern "C" __global__
            void fast_linear_prediction(
                const float* features, const float* weights,
                float* predictions, int n, int feature_dim
            ) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= n) return;

                float prediction = 0.0f;
                int base_idx = idx * feature_dim;

                for (int i = 0; i < feature_dim; i++) {
                    prediction += features[base_idx + i] * weights[i];
                }

                predictions[idx] = prediction;
            }
            """

            self.compiled_kernels["linear_prediction"] = cp.RawKernel(
                prediction_kernel_code, "fast_linear_prediction"
            )

            logger.info("CUDA カーネルコンパイル完了")

        except Exception as e:
            logger.error(f"CUDA カーネルコンパイル失敗: {e}")

    def execute_feature_kernel(
        self,
        prices: "cp.ndarray",
        volumes: "cp.ndarray",
        features: "cp.ndarray",
        stream_id: int = 0,
    ) -> bool:
        """特徴量計算カーネル実行"""
        if "technical_features" not in self.compiled_kernels:
            return False

        try:
            n = len(prices)
            feature_dim = features.shape[1]

            # ブロック・グリッドサイズ計算
            block_size = 256
            grid_size = (n + block_size - 1) // block_size

            # ストリーム選択
            stream = (
                self.streams[stream_id % len(self.streams)] if self.streams else None
            )

            # カーネル実行
            self.compiled_kernels["technical_features"](
                (grid_size,),
                (block_size,),
                (prices, volumes, features, n, feature_dim),
                stream=stream,
            )

            return True

        except Exception as e:
            logger.error(f"特徴量カーネル実行失敗: {e}")
            return False

    def execute_prediction_kernel(
        self,
        features: "cp.ndarray",
        weights: "cp.ndarray",
        predictions: "cp.ndarray",
        stream_id: int = 0,
    ) -> bool:
        """予測カーネル実行"""
        if "linear_prediction" not in self.compiled_kernels:
            return False

        try:
            n = features.shape[0]
            feature_dim = features.shape[1]

            # ブロック・グリッドサイズ計算
            block_size = 256
            grid_size = (n + block_size - 1) // block_size

            # ストリーム選択
            stream = (
                self.streams[stream_id % len(self.streams)] if self.streams else None
            )

            # カーネル実行
            self.compiled_kernels["linear_prediction"](
                (grid_size,),
                (block_size,),
                (features, weights, predictions, n, feature_dim),
                stream=stream,
            )

            return True

        except Exception as e:
            logger.error(f"予測カーネル実行失敗: {e}")
            return False


class GPUAccelerator:
    """GPU加速処理メインエンジン"""

    def __init__(self, config: GPUConfig = None):
        self.config = config or GPUConfig()

        # GPU可用性チェック
        self.gpu_available = self._check_gpu_availability()

        # GPU管理システム初期化
        self.memory_manager = GPUMemoryManager(self.config)
        self.kernel_manager = (
            CudaKernelManager(self.config) if self.gpu_available else None
        )

        # モデル管理
        self.models = {}
        self.model_weights = {}

        # パフォーマンス統計
        self.performance_stats = {
            "gpu_operations": 0,
            "cpu_fallback_operations": 0,
            "avg_gpu_time_ms": 0.0,
            "avg_cpu_time_ms": 0.0,
            "speedup_ratio": 1.0,
        }
        self._stats_lock = threading.RLock()

        logger.info(f"GPU加速エンジン初期化完了 (GPU利用可能: {self.gpu_available})")

    def _check_gpu_availability(self) -> bool:
        """GPU可用性チェック"""
        gpu_info = {
            "cupy": CUPY_AVAILABLE,
            "tensorflow": TF_AVAILABLE,
            "torch": TORCH_AVAILABLE,
        }

        if CUPY_AVAILABLE:
            try:
                # CUDA デバイス情報取得
                device_count = cp.cuda.runtime.getDeviceCount()
                cp.cuda.Device(self.config.device_id).use()

                device_props = cp.cuda.runtime.getDeviceProperties(
                    self.config.device_id
                )
                logger.info(
                    f"CUDA デバイス検出: {device_count}台, 使用デバイス: {self.config.device_id}"
                )
                logger.info(f"デバイス名: {device_props['name'].decode()}")

                return True

            except Exception as e:
                logger.warning(f"CUDA初期化失敗: {e}")

        logger.info(f"GPU ライブラリ状況: {gpu_info}")
        return any(gpu_info.values())

    def compute_features_gpu(
        self, prices: np.ndarray, volumes: np.ndarray, feature_dim: int = 7
    ) -> Union[np.ndarray, None]:
        """GPU特徴量計算"""
        if not self.gpu_available or not CUPY_AVAILABLE:
            return self._compute_features_cpu(prices, volumes, feature_dim)

        start_time = time.perf_counter()

        try:
            # データサイズチェック
            if len(prices) < self.config.min_data_size_for_gpu:
                return self._compute_features_cpu(prices, volumes, feature_dim)

            # GPU データ転送
            gpu_prices = cp.asarray(prices, dtype=cp.float32)
            gpu_volumes = cp.asarray(volumes, dtype=cp.float32)
            gpu_features = cp.zeros((len(prices), feature_dim), dtype=cp.float32)

            # CUDA カーネル実行
            if self.kernel_manager and self.kernel_manager.execute_feature_kernel(
                gpu_prices, gpu_volumes, gpu_features
            ):
                # CPU に結果転送
                result = cp.asnumpy(gpu_features)

                # メモリ解放
                del gpu_prices, gpu_volumes, gpu_features

                # 統計更新
                gpu_time = (time.perf_counter() - start_time) * 1000
                self._update_gpu_stats(gpu_time, is_gpu=True)

                return result
            else:
                # カーネル実行失敗 - CPU フォールバック
                return self._compute_features_cpu(prices, volumes, feature_dim)

        except Exception as e:
            logger.error(f"GPU特徴量計算失敗: {e}")
            return self._compute_features_cpu(prices, volumes, feature_dim)

    def _compute_features_cpu(
        self, prices: np.ndarray, volumes: np.ndarray, feature_dim: int
    ) -> np.ndarray:
        """CPU フォールバック特徴量計算"""
        start_time = time.perf_counter()

        n = len(prices)
        features = np.zeros((n, feature_dim), dtype=np.float32)

        for i in range(20, n):  # 十分な履歴がある範囲
            # 移動平均
            features[i, 0] = np.mean(prices[i - 5 : i])  # MA5
            features[i, 1] = np.mean(prices[i - 20 : i])  # MA20

            # RSI簡略版
            changes = np.diff(prices[i - 14 : i])
            gains = changes[changes > 0]
            losses = -changes[changes < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            rs = avg_gain / max(avg_loss, 1e-8)
            features[i, 2] = 100 - (100 / (1 + rs))

            # ボリューム比率
            vol_ma = np.mean(volumes[i - 5 : i])
            features[i, 3] = volumes[i] / max(vol_ma, 1e-8)

            # 価格変化率
            features[i, 4] = (prices[i] - prices[i - 1]) / max(prices[i - 1], 1e-8)

            # 現在値
            features[i, 5] = prices[i]
            features[i, 6] = volumes[i]

        # 統計更新
        cpu_time = (time.perf_counter() - start_time) * 1000
        self._update_gpu_stats(cpu_time, is_gpu=False)

        return features

    def batch_predict_gpu(
        self, features_batch: np.ndarray, model_weights: np.ndarray
    ) -> Union[np.ndarray, None]:
        """GPU バッチ予測"""
        if not self.gpu_available or not CUPY_AVAILABLE:
            return self._batch_predict_cpu(features_batch, model_weights)

        start_time = time.perf_counter()

        try:
            # データサイズチェック
            if len(features_batch) < self.config.min_data_size_for_gpu:
                return self._batch_predict_cpu(features_batch, model_weights)

            # GPU データ転送
            gpu_features = cp.asarray(features_batch, dtype=cp.float32)
            gpu_weights = cp.asarray(model_weights, dtype=cp.float32)
            gpu_predictions = cp.zeros(len(features_batch), dtype=cp.float32)

            # CUDA カーネル実行
            if self.kernel_manager and self.kernel_manager.execute_prediction_kernel(
                gpu_features, gpu_weights, gpu_predictions
            ):
                # CPU に結果転送
                result = cp.asnumpy(gpu_predictions)

                # メモリ解放
                del gpu_features, gpu_weights, gpu_predictions

                # 統計更新
                gpu_time = (time.perf_counter() - start_time) * 1000
                self._update_gpu_stats(gpu_time, is_gpu=True)

                return result
            else:
                return self._batch_predict_cpu(features_batch, model_weights)

        except Exception as e:
            logger.error(f"GPU バッチ予測失敗: {e}")
            return self._batch_predict_cpu(features_batch, model_weights)

    def _batch_predict_cpu(
        self, features_batch: np.ndarray, model_weights: np.ndarray
    ) -> np.ndarray:
        """CPU フォールバック予測"""
        start_time = time.perf_counter()

        # 線形予測（行列積）
        predictions = np.dot(features_batch, model_weights)

        # 統計更新
        cpu_time = (time.perf_counter() - start_time) * 1000
        self._update_gpu_stats(cpu_time, is_gpu=False)

        return predictions

    async def async_gpu_pipeline(
        self, symbols_data: Dict[str, Dict[str, np.ndarray]], model_weights: np.ndarray
    ) -> Dict[str, Any]:
        """非同期GPU処理パイプライン"""
        pipeline_start = time.perf_counter()

        results = {}
        tasks = []

        # バッチ処理用データ準備
        all_features = []
        symbol_indices = {}
        current_index = 0

        for symbol, data in symbols_data.items():
            prices = data.get("prices", np.array([]))
            volumes = data.get("volumes", np.array([]))

            if len(prices) > 0 and len(volumes) > 0:
                # 特徴量計算
                features = self.compute_features_gpu(prices, volumes)
                if features is not None and len(features) > 0:
                    # 最新の特徴量を使用
                    latest_features = features[-1]
                    all_features.append(latest_features)
                    symbol_indices[symbol] = current_index
                    current_index += 1

        if all_features:
            # バッチ予測実行
            features_batch = np.array(all_features)
            predictions = self.batch_predict_gpu(features_batch, model_weights)

            # 結果をシンボル別に分割
            for symbol, index in symbol_indices.items():
                if predictions is not None and index < len(predictions):
                    results[symbol] = {
                        "prediction": float(predictions[index]),
                        "timestamp": time.time(),
                        "gpu_accelerated": self.gpu_available,
                    }

        pipeline_time = (time.perf_counter() - pipeline_start) * 1000

        return {
            "predictions": results,
            "pipeline_stats": {
                "total_symbols": len(results),
                "pipeline_time_ms": pipeline_time,
                "gpu_accelerated": self.gpu_available,
                "symbols_per_second": len(results) / max(pipeline_time / 1000, 1e-6),
            },
        }

    def _update_gpu_stats(self, execution_time_ms: float, is_gpu: bool):
        """GPU統計更新"""
        with self._stats_lock:
            if is_gpu:
                self.performance_stats["gpu_operations"] += 1
                current_avg = self.performance_stats["avg_gpu_time_ms"]
                total_ops = self.performance_stats["gpu_operations"]
                self.performance_stats["avg_gpu_time_ms"] = (
                    current_avg * (total_ops - 1) + execution_time_ms
                ) / total_ops
            else:
                self.performance_stats["cpu_fallback_operations"] += 1
                current_avg = self.performance_stats["avg_cpu_time_ms"]
                total_ops = self.performance_stats["cpu_fallback_operations"]
                self.performance_stats["avg_cpu_time_ms"] = (
                    current_avg * (total_ops - 1) + execution_time_ms
                ) / total_ops

            # 加速比計算
            gpu_time = self.performance_stats["avg_gpu_time_ms"]
            cpu_time = self.performance_stats["avg_cpu_time_ms"]
            if gpu_time > 0 and cpu_time > 0:
                self.performance_stats["speedup_ratio"] = cpu_time / gpu_time

    def get_gpu_report(self) -> Dict[str, Any]:
        """GPU加速レポート"""
        memory_info = self.memory_manager.get_memory_info()

        with self._stats_lock:
            perf_stats = self.performance_stats.copy()

        return {
            "gpu_config": {
                "device_id": self.config.device_id,
                "batch_size": self.config.batch_size,
                "memory_limit_mb": self.config.gpu_memory_limit_mb,
                "streams": self.config.max_concurrent_streams,
            },
            "availability": {
                "gpu_available": self.gpu_available,
                "cupy_available": CUPY_AVAILABLE,
                "tensorflow_available": TF_AVAILABLE,
                "torch_available": TORCH_AVAILABLE,
            },
            "performance": perf_stats,
            "memory": memory_info,
            "efficiency_score": min(100.0, perf_stats["speedup_ratio"] * 20),
        }

    def cleanup(self):
        """GPU リソースクリーンアップ"""
        try:
            self.memory_manager.cleanup()

            if CUPY_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()

            logger.info("GPU加速エンジンクリーンアップ完了")

        except Exception as e:
            logger.error(f"GPUクリーンアップエラー: {e}")


# グローバルインスタンス
_global_gpu_accelerator: Optional[GPUAccelerator] = None
_gpu_lock = threading.Lock()


def get_gpu_accelerator(config: GPUConfig = None) -> GPUAccelerator:
    """グローバルGPU加速エンジン取得"""
    global _global_gpu_accelerator

    if _global_gpu_accelerator is None:
        with _gpu_lock:
            if _global_gpu_accelerator is None:
                _global_gpu_accelerator = GPUAccelerator(config)

    return _global_gpu_accelerator


def gpu_accelerated(min_data_size: int = 100):
    """GPU加速デコレータ"""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            accelerator = get_gpu_accelerator()

            # データサイズチェック
            data_size = 0
            for arg in args:
                if isinstance(arg, (np.ndarray, list)):
                    data_size = max(data_size, len(arg))

            if data_size >= min_data_size and accelerator.gpu_available:
                # GPU処理パス
                return await func(*args, **kwargs)
            else:
                # CPU フォールバック
                return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            finally:
                loop.close()

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


if __name__ == "__main__":
    # GPU加速デモ
    print("=== GPU加速処理エンジン デモ ===")

    config = GPUConfig(
        batch_size=512,
        gpu_memory_limit_mb=1024,
        max_concurrent_streams=2,
    )

    accelerator = GPUAccelerator(config)

    # テストデータ生成
    np.random.seed(42)
    test_data = {}

    for i in range(20):
        symbol = f"STOCK_{i:03d}"
        test_data[symbol] = {
            "prices": np.random.normal(100, 5, 1000).astype(np.float32),
            "volumes": np.random.normal(10000, 1000, 1000).astype(np.float32),
        }

    model_weights = np.random.normal(0, 0.1, 7).astype(np.float32)

    print("\n1. 単一GPU特徴量計算テスト")
    prices = test_data["STOCK_001"]["prices"]
    volumes = test_data["STOCK_001"]["volumes"]

    features = accelerator.compute_features_gpu(prices, volumes)
    print(f"特徴量形状: {features.shape if features is not None else 'None'}")
    print(f"GPU利用可能: {accelerator.gpu_available}")

    print("\n2. バッチGPU予測テスト")
    if features is not None:
        latest_features = features[-10:]  # 最新10件
        predictions = accelerator.batch_predict_gpu(latest_features, model_weights)
        print(f"予測結果: {predictions[:5] if predictions is not None else 'None'}")

    print("\n3. 非同期パイプライン テスト")

    async def run_async_test():
        result = await accelerator.async_gpu_pipeline(test_data, model_weights)
        return result

    pipeline_result = asyncio.run(run_async_test())
    pipeline_stats = pipeline_result["pipeline_stats"]

    print(f"処理銘柄数: {pipeline_stats['total_symbols']}")
    print(f"パイプライン時間: {pipeline_stats['pipeline_time_ms']:.2f}ms")
    print(f"処理速度: {pipeline_stats['symbols_per_second']:.1f} symbols/sec")

    print("\n4. GPU レポート")
    report = accelerator.get_gpu_report()
    print(f"GPU利用可能: {report['availability']['gpu_available']}")
    print(f"GPU操作回数: {report['performance']['gpu_operations']}")
    print(f"CPU代替操作回数: {report['performance']['cpu_fallback_operations']}")
    print(f"平均GPU時間: {report['performance']['avg_gpu_time_ms']:.2f}ms")
    print(f"加速比: {report['performance']['speedup_ratio']:.1f}x")
    print(f"効率スコア: {report['efficiency_score']:.1f}/100")

    accelerator.cleanup()
    print("\n=== GPU加速デモ完了 ===")
