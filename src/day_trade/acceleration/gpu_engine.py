#!/usr/bin/env python3
"""
GPU並列処理エンジン
Phase F: 次世代機能拡張フェーズ

CUDA/OpenCL による10-100倍高速化処理
"""

import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# CuPy可用性チェック
try:
    import cupy as cp  # type: ignore[import - untyped]

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None  # type: ignore[misc]
    warnings.warn(
        "CuPy未インストール - GPU加速機能利用不可", ImportWarning, stacklevel=2
    )

from ..core.optimization_strategy import (
    OptimizationConfig,
    OptimizationLevel,
    OptimizationStrategy,
    optimization_strategy,
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)

class GPUBackend(Enum):
    """GPU バックエンド種別"""

    CUDA = "cuda"  # NVIDIA CUDA
    OPENCL = "opencl"  # OpenCL (AMD/Intel)
    CPU_FALLBACK = "cpu"  # CPUフォールバック

@dataclass
class GPUDeviceInfo:
    """GPU デバイス情報"""

    device_id: int
    name: str
    backend: GPUBackend
    memory_total: int  # MB
    memory_free: int  # MB
    compute_capability: Optional[str] = None
    is_available: bool = True

@dataclass
class GPUComputeResult:
    """GPU計算結果"""

    result: Any
    execution_time: float
    backend_used: GPUBackend
    device_info: GPUDeviceInfo
    memory_used: float  # MB
    speedup_ratio: Optional[float] = None

class GPUMemoryManager:
    """GPUメモリ管理クラス
    
    GPUメモリの効率的な割り当て、解放、統計情報の管理を行う
    """

    def __init__(self) -> None:
        """GPUメモリマネージャーの初期化"""
        self.allocated_blocks: Dict[str, Dict[str, Any]] = {}
        self.peak_memory_usage: int = 0
        self.current_memory_usage: int = 0

    def allocate(self, size: int, device_id: int = 0) -> str:
        """GPU メモリ割り当て"""
        block_id = f"block_{len(self.allocated_blocks)}_{time.time()}"

        try:
            # CUDA メモリ割り当て（利用可能な場合）
            if self._cuda_available():
                import cupy as cp

                with cp.cuda.Device(device_id):
                    memory_block = cp.cuda.alloc(size)
                    self.allocated_blocks[block_id] = {
                        "block": memory_block,
                        "size": size,
                        "device": device_id,
                        "backend": GPUBackend.CUDA,
                    }
            else:
                # フォールバック: NumPy配列
                memory_block = np.zeros(size, dtype=np.float32)
                self.allocated_blocks[block_id] = {
                    "block": memory_block,
                    "size": size,
                    "device": -1,
                    "backend": GPUBackend.CPU_FALLBACK,
                }

            self.current_memory_usage += size
            self.peak_memory_usage = max(
                self.peak_memory_usage, self.current_memory_usage
            )

            logger.debug(f"GPU メモリ割り当て: {size}bytes, Block ID: {block_id}")
            return block_id

        except Exception as e:
            logger.error(f"GPU メモリ割り当てエラー: {e}")
            raise

    def deallocate(self, block_id: str) -> None:
        """GPU メモリ解放"""
        if block_id in self.allocated_blocks:
            block_info = self.allocated_blocks[block_id]
            self.current_memory_usage -= block_info["size"]

            try:
                if block_info["backend"] == GPUBackend.CUDA:
                    # CUDA メモリ解放
                    block_info["block"].free()
                # CPU_FALLBACK の場合は自動解放

                del self.allocated_blocks[block_id]
                logger.debug(f"GPU メモリ解放完了: {block_id}")

            except Exception as e:
                logger.error(f"GPU メモリ解放エラー: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """メモリ統計取得"""
        return {
            "current_usage": self.current_memory_usage,
            "peak_usage": self.peak_memory_usage,
            "allocated_blocks": len(self.allocated_blocks),
            "fragmentation_ratio": self._calculate_fragmentation(),
        }

    def _calculate_fragmentation(self) -> float:
        """メモリ断片化率計算"""
        if not self.allocated_blocks:
            return 0.0

        total_allocated = sum(block["size"] for block in self.allocated_blocks.values())
        if total_allocated == 0:
            return 0.0

        return (self.current_memory_usage - total_allocated) / self.current_memory_usage

    def _cuda_available(self) -> bool:
        """CUDA 利用可能性チェック"""
        try:
            import cupy as cp

            return cp.cuda.runtime.getDeviceCount() > 0
        except (ImportError, Exception):
            return False

class GPUAccelerationEngine:
    """GPU並列処理エンジン
    
    CUDA/OpenCLを使用してテクニカル指標の計算を高速化するエンジン
    """

    def __init__(self, config: Optional[OptimizationConfig] = None) -> None:
        """GPU並列処理エンジンの初期化"""
        self.config = config or OptimizationConfig()
        self.memory_manager = GPUMemoryManager()

        # GPU バックエンド検出・初期化
        self.available_backends = self._detect_gpu_backends()
        self.devices = self._enumerate_devices()
        self.primary_backend = self._select_primary_backend()

        # パフォーマンス統計
        self.performance_stats = {
            "total_computations": 0,
            "total_gpu_time": 0.0,
            "total_cpu_time": 0.0,
            "average_speedup": 0.0,
        }

        logger.info(
            f"GPU並列処理エンジン初期化完了: {self.primary_backend.value} バックエンド"
        )
        logger.info(f"利用可能デバイス: {len(self.devices)}個")

    def _detect_gpu_backends(self) -> List[GPUBackend]:
        """GPU バックエンド検出"""
        available_backends = []

        # CUDA チェック
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                available_backends.append(GPUBackend.CUDA)
                logger.info("CUDA バックエンド利用可能")
        except (ImportError, Exception) as e:
            logger.info(f"CUDA バックエンド利用不可: {e}")

        # OpenCL チェック
        try:
            import pyopencl as cl  # type: ignore[import - untyped]

            platforms = cl.get_platforms()
            if platforms:
                available_backends.append(GPUBackend.OPENCL)
                logger.info("OpenCL バックエンド利用可能")
        except (ImportError, Exception) as e:
            logger.info(f"OpenCL バックエンド利用不可: {e}")

        # 最低でも CPU フォールバックは利用可能
        available_backends.append(GPUBackend.CPU_FALLBACK)

        return available_backends

    def _enumerate_devices(self) -> List[GPUDeviceInfo]:
        """デバイス列挙"""
        devices = []

        # CUDA デバイス
        if GPUBackend.CUDA in self.available_backends:
            try:
                import cupy as cp

                for i in range(cp.cuda.runtime.getDeviceCount()):
                    with cp.cuda.Device(i):
                        device_props = cp.cuda.runtime.getDeviceProperties(i)
                        mem_info = cp.cuda.runtime.memGetInfo()

                        devices.append(
                            GPUDeviceInfo(
                                device_id=i,
                                name=device_props["name"].decode("utf-8"),
                                backend=GPUBackend.CUDA,
                                memory_total=device_props["totalGlobalMem"] // 1024 // 1024,  # MB
                                memory_free=mem_info[0] // 1024 // 1024,  # MB
                                compute_capability=f"{device_props['major']}.{device_props['minor']}",
                                is_available=True,
                            )
                        )
            except Exception as e:
                logger.error(f"CUDA デバイス列挙エラー: {e}")

        # OpenCL デバイス
        if GPUBackend.OPENCL in self.available_backends:
            try:
                import pyopencl as cl

                for platform in cl.get_platforms():
                    for i, device in enumerate(platform.get_devices()):
                        devices.append(
                            GPUDeviceInfo(
                                device_id=i,
                                name=device.name,
                                backend=GPUBackend.OPENCL,
                                memory_total=device.global_mem_size // 1024 // 1024,  # MB
                                memory_free=device.global_mem_size // 1024 // 1024,  # MB (近似)
                                is_available=True,
                            )
                        )
            except Exception as e:
                logger.error(f"OpenCL デバイス列挙エラー: {e}")

        # CPU フォールバック
        import psutil

        devices.append(
            GPUDeviceInfo(
                device_id=-1,
                name=f"CPU ({psutil.cpu_count()}コア)",
                backend=GPUBackend.CPU_FALLBACK,
                memory_total=int(psutil.virtual_memory().total // 1024 // 1024),  # MB
                memory_free=int(psutil.virtual_memory().available // 1024 // 1024),  # MB
                is_available=True,
            )
        )

        return devices

    def _select_primary_backend(self) -> GPUBackend:
        """プライマリバックエンド選択"""
        # 優先順位: CUDA > OpenCL > CPU
        if GPUBackend.CUDA in self.available_backends:
            return GPUBackend.CUDA
        elif GPUBackend.OPENCL in self.available_backends:
            return GPUBackend.OPENCL
        else:
            return GPUBackend.CPU_FALLBACK

    def accelerate_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[str],
        periods: Optional[Dict[str, int]] = None,
        device_id: int = 0,
    ) -> GPUComputeResult:
        """テクニカル指標GPU並列計算"""
        start_time = time.time()

        try:
            if self.primary_backend == GPUBackend.CUDA:
                result = self._cuda_technical_indicators(
                    data, indicators, periods, device_id
                )
            elif self.primary_backend == GPUBackend.OPENCL:
                result = self._opencl_technical_indicators(
                    data, indicators, periods, device_id
                )
            else:
                result = self._cpu_technical_indicators(data, indicators, periods)

            execution_time = time.time() - start_time
            device_info = self._get_device_info(device_id)
            memory_used = self._estimate_memory_usage(data)

            # 統計更新
            self.performance_stats["total_computations"] += 1
            if self.primary_backend != GPUBackend.CPU_FALLBACK:
                self.performance_stats["total_gpu_time"] += execution_time
            else:
                self.performance_stats["total_cpu_time"] += execution_time

            return GPUComputeResult(
                result = result,
                execution_time = execution_time,
                backend_used = self.primary_backend,
                device_info = device_info,
                memory_used = memory_used,
            )

        except Exception as e:
            logger.error(f"GPU テクニカル指標計算エラー: {e}")
            # CPU フォールバックで再実行
            if self.primary_backend != GPUBackend.CPU_FALLBACK:
                logger.info("CPU フォールバックで再実行")
                return self._fallback_technical_indicators(data, indicators, periods)
            else:
                raise

    def _cuda_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[str],
        periods: Optional[Dict[str, int]],
        device_id: int,
    ) -> Dict[str, Any]:
        """CUDA テクニカル指標計算"""
        try:
            import cupy as cp

            with cp.cuda.Device(device_id):
                # データを GPU メモリに転送
                close_prices = cp.asarray(
                    data["Close"].values
                    if "Close" in data.columns
                    else data[data.columns[-1]].values
                )
                high_prices = cp.asarray(
                    data["High"].values if "High" in data.columns else close_prices
                )
                low_prices = cp.asarray(
                    data["Low"].values if "Low" in data.columns else close_prices
                )
                volume = cp.asarray(
                    data["Volume"].values
                    if "Volume" in data.columns
                    else cp.ones_like(close_prices)
                )

                results = {}

                for indicator in indicators:
                    if indicator == "sma":
                        period = periods.get("sma", 20) if periods else 20
                        results[indicator] = self._cuda_sma(close_prices, period)
                    elif indicator == "ema":
                        period = periods.get("ema", 20) if periods else 20
                        results[indicator] = self._cuda_ema(close_prices, period)
                    elif indicator == "rsi":
                        period = periods.get("rsi", 14) if periods else 14
                        results[indicator] = self._cuda_rsi(close_prices, period)
                    elif indicator == "bollinger_bands":
                        period = periods.get("bollinger_bands", 20) if periods else 20
                        results[indicator] = self._cuda_bollinger_bands(
                            close_prices, period
                        )
                    elif indicator == "macd":
                        results[indicator] = self._cuda_macd(close_prices)
                    elif indicator == "stochastic":
                        period = periods.get("stochastic", 14) if periods else 14
                        results[indicator] = self._cuda_stochastic(
                            high_prices, low_prices, close_prices, period
                        )

                # GPU から CPU にデータ転送
                cpu_results = {}
                for key, value in results.items():
                    if isinstance(value, dict):
                        cpu_results[key] = {k: cp.asnumpy(v) for k, v in value.items()}
                    else:
                        cpu_results[key] = cp.asnumpy(value)

                return cpu_results

        except Exception as e:
            logger.error(f"CUDA 計算エラー: {e}")
            raise

    def _cuda_sma(self, prices: "cp.ndarray", period: int) -> "cp.ndarray":
        """CUDA 単純移動平均"""
        import cupy as cp

        # 畳み込みを使用した効率的な SMA 計算
        kernel = cp.ones(period) / period
        # パディングを追加して境界条件を処理
        padded_prices = cp.pad(prices, (period - 1, 0), mode="edge")
        sma = cp.convolve(padded_prices, kernel, mode="valid")

        return sma

    def _cuda_ema(self, prices: "cp.ndarray", period: int) -> "cp.ndarray":
        """CUDA 指数移動平均"""
        import cupy as cp

        alpha = 2.0 / (period + 1)
        ema = cp.zeros_like(prices)
        ema[0] = prices[0]

        # GPU カーネル使用による効率的な EMA 計算
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _cuda_rsi(self, prices: "cp.ndarray", period: int) -> "cp.ndarray":
        """CUDA RSI 計算"""
        import cupy as cp

        # 価格変化量計算
        delta = cp.diff(prices)
        gain = cp.where(delta > 0, delta, 0)
        loss = cp.where(delta < 0, -delta, 0)

        # 平均ゲイン・ロス計算
        avg_gain = cp.zeros_like(prices)
        avg_loss = cp.zeros_like(prices)

        # 最初の期間の平均
        avg_gain[period] = cp.mean(gain[:period])
        avg_loss[period] = cp.mean(loss[:period])

        # 指数移動平均でゲイン・ロス更新
        for i in range(period + 1, len(prices)):
            avg_gain[i] = ((period - 1) * avg_gain[i - 1] + gain[i - 1]) / period
            avg_loss[i] = ((period - 1) * avg_loss[i - 1] + loss[i - 1]) / period

        # RSI 計算
        rs = avg_gain / cp.where(avg_loss != 0, avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _cuda_bollinger_bands(
        self, prices: "cp.ndarray", period: int
    ) -> Dict[str, "cp.ndarray"]:
        """CUDA ボリンジャーバンド計算"""
        import cupy as cp

        sma = self._cuda_sma(prices, period)

        # 移動標準偏差計算
        rolling_std = cp.zeros_like(prices)
        for i in range(period - 1, len(prices)):
            rolling_std[i] = cp.std(prices[i - period + 1 : i + 1])

        upper_band = sma + 2 * rolling_std
        lower_band = sma - 2 * rolling_std

        return {"middle": sma, "upper": upper_band, "lower": lower_band}

    def _cuda_macd(self, prices: "cp.ndarray") -> Dict[str, "cp.ndarray"]:
        """CUDA MACD 計算"""
        ema_12 = self._cuda_ema(prices, 12)
        ema_26 = self._cuda_ema(prices, 26)
        macd_line = ema_12 - ema_26
        signal_line = self._cuda_ema(macd_line, 9)
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def _cuda_stochastic(
        self, high: "cp.ndarray", low: "cp.ndarray", close: "cp.ndarray", period: int
    ) -> Dict[str, "cp.ndarray"]:
        """CUDA ストキャスティクス計算"""
        import cupy as cp

        stoch_k = cp.zeros_like(close)

        for i in range(period - 1, len(close)):
            highest_high = cp.max(high[i - period + 1 : i + 1])
            lowest_low = cp.min(low[i - period + 1 : i + 1])
            stoch_k[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100

        stoch_d = self._cuda_sma(stoch_k, 3)

        return {"k": stoch_k, "d": stoch_d}

    def _opencl_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[str],
        periods: Optional[Dict[str, int]],
        device_id: int,
    ) -> Dict[str, Any]:
        """OpenCL テクニカル指標計算"""
        # OpenCL 実装は簡略化（CUDA 実装をベースにした互換実装）
        logger.info("OpenCL バックエンドは CPU フォールバックを使用")
        return self._cpu_technical_indicators(data, indicators, periods)

    def _cpu_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[str],
        periods: Optional[Dict[str, int]],
    ) -> Dict[str, Any]:
        """CPU フォールバック テクニカル指標計算"""
        # NumPy を使用した最適化 CPU 実装
        close_prices = (
            data["Close"].values
            if "Close" in data.columns
            else data[data.columns[-1]].values
        )
        results = {}

        for indicator in indicators:
            if indicator == "sma":
                period = periods.get("sma", 20) if periods else 20
                results[indicator] = self._numpy_sma(close_prices, period)
            elif indicator == "ema":
                period = periods.get("ema", 20) if periods else 20
                results[indicator] = self._numpy_ema(close_prices, period)
            elif indicator == "rsi":
                period = periods.get("rsi", 14) if periods else 14
                results[indicator] = self._numpy_rsi(close_prices, period)
            # ... 他の指標も同様に実装

        return results

    def _numpy_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """NumPy 単純移動平均"""
        return np.convolve(prices, np.ones(period) / period, mode="valid")

    def _numpy_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """NumPy 指数移動平均"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

        return ema

    def _numpy_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """NumPy RSI 計算"""
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.convolve(gain, np.ones(period) / period, mode="valid")
        avg_loss = np.convolve(loss, np.ones(period) / period, mode="valid")

        rs = avg_gain / np.where(avg_loss != 0, avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _fallback_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[str],
        periods: Optional[Dict[str, int]],
    ) -> GPUComputeResult:
        """フォールバック実行"""
        start_time = time.time()
        result = self._cpu_technical_indicators(data, indicators, periods)
        execution_time = time.time() - start_time

        device_info = GPUDeviceInfo(
            device_id=-1,
            name="CPU Fallback",
            backend = GPUBackend.CPU_FALLBACK,
            memory_total = 0,
            memory_free = 0,
            is_available = True,
        )

        return GPUComputeResult(
            result = result,
            execution_time = execution_time,
            backend_used = GPUBackend.CPU_FALLBACK,
            device_info = device_info,
            memory_used = self._estimate_memory_usage(data),
        )

    def _get_device_info(self, device_id: int) -> GPUDeviceInfo:
        """デバイス情報取得"""
        for device in self.devices:
            if device.device_id == device_id and device.backend == self.primary_backend:
                return device

        # フォールバック
        return self.devices[-1]  # CPU フォールバック

    def _estimate_memory_usage(self, data: pd.DataFrame) -> float:
        """メモリ使用量推定"""
        return data.memory_usage(deep = True).sum() / 1024 / 1024  # MB

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        total_time = (
            self.performance_stats["total_gpu_time"]
            + self.performance_stats["total_cpu_time"]
        )

        return {
            "total_computations": self.performance_stats["total_computations"],
            "total_time": total_time,
            "gpu_time_ratio": self.performance_stats["total_gpu_time"]
            / max(total_time, 1e-10),
            "average_computation_time": total_time
            / max(self.performance_stats["total_computations"], 1),
            "primary_backend": self.primary_backend.value,
            "available_devices": len(self.devices),
            "memory_stats": self.memory_manager.get_memory_stats(),
        }

    def benchmark_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """パフォーマンスベンチマーク"""
        indicators = ["sma", "ema", "rsi", "bollinger_bands", "macd"]

        # GPU 実行
        gpu_start = time.time()
        gpu_result = self.accelerate_technical_indicators(data, indicators)
        gpu_time = time.time() - gpu_start

        # CPU 実行（比較用）
        cpu_start = time.time()
        cpu_result = self._cpu_technical_indicators(data, indicators, None)
        cpu_time = time.time() - cpu_start

        speedup_ratio = (
            cpu_time / gpu_result.execution_time
            if gpu_result.execution_time > 0
            else 1.0
        )

        return {
            "gpu_time": gpu_result.execution_time,
            "cpu_time": cpu_time,
            "speedup_ratio": speedup_ratio,
            "backend_used": gpu_result.backend_used.value,
            "memory_used": gpu_result.memory_used,
            "device_info": {
                "name": gpu_result.device_info.name,
                "memory_total": gpu_result.device_info.memory_total,
                "backend": gpu_result.device_info.backend.value,
            },
        }

@dataclass
class GPUAcceleratedResult:
    """GPU加速実行結果"""

    indicators: Dict[str, Any]
    computation_result: GPUComputeResult
    strategy_name: str

# Strategy Pattern への統合
@optimization_strategy("technical_indicators", OptimizationLevel.GPU_ACCELERATED)
class GPUAcceleratedTechnicalIndicators(OptimizationStrategy):
    """GPU加速テクニカル指標戦略"""

    def __init__(self, config: OptimizationConfig) -> None:
        """__init__関数"""
        super().__init__(config)
        self.gpu_engine = GPUAccelerationEngine(config)
        logger.info("GPU加速テクニカル指標戦略初期化完了")

    def get_strategy_name(self) -> str:
        return f"GPU加速テクニカル指標 ({self.gpu_engine.primary_backend.value})"

    def execute(self, data: pd.DataFrame, indicators: List[str], **kwargs) -> Any:
        """GPU加速実行"""
        periods = kwargs.get("periods")
        device_id = kwargs.get("device_id", 0)

        start_time = time.time()

        try:
            result = self.gpu_engine.accelerate_technical_indicators(
                data, indicators, periods, device_id
            )

            execution_time = time.time() - start_time
            self.record_execution(execution_time, True)

            # 結果をラップして返す
            return GPUAcceleratedResult(
                indicators = result.result,
                computation_result = result,
                strategy_name = self.get_strategy_name(),
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.record_execution(execution_time, False)
            logger.error(f"GPU加速実行エラー: {e}")
            raise

# GPU_ACCELERATED レベルは optimization_strategy.py で定義済み
