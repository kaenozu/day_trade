"""
GPU デバイス管理とモニタリング

gpu_accelerated_inference.py からのリファクタリング抽出
GPU デバイスの検出、管理、監視機能を提供
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .gpu_config import GPUBackend

# GPU計算ライブラリ (フォールバック対応)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    warnings.warn("CuPy not available - CPU fallback", stacklevel=2)

# OpenCL支援ライブラリ (フォールバック対応)
try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    warnings.warn("PyOpenCL not available", stacklevel=2)

# ロギング設定
import logging
logger = logging.getLogger(__name__)


@dataclass
class GPUMonitoringData:
    """GPU監視データ"""

    device_id: int
    timestamp: float

    # GPU使用率統計
    gpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0

    # メモリ使用量（MB）
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_free_mb: float = 0.0

    # 温度・電力
    temperature_celsius: float = 0.0
    power_consumption_watts: float = 0.0

    # プロセス情報
    running_processes: int = 0
    compute_mode: str = "Default"

    # エラー状態
    has_errors: bool = False
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "memory_utilization_percent": self.memory_utilization_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "memory_free_mb": self.memory_free_mb,
            "temperature_celsius": self.temperature_celsius,
            "power_consumption_watts": self.power_consumption_watts,
            "running_processes": self.running_processes,
            "compute_mode": self.compute_mode,
            "has_errors": self.has_errors,
            "error_message": self.error_message,
        }

    @property
    def is_overloaded(self) -> bool:
        """GPU過負荷状態の判定"""
        return (
            self.gpu_utilization_percent > 95.0
            or self.memory_utilization_percent > 95.0
            or self.temperature_celsius > 85.0
        )

    @property
    def is_healthy(self) -> bool:
        """GPU健全性の判定"""
        return (
            not self.has_errors
            and self.gpu_utilization_percent < 90.0
            and self.memory_utilization_percent < 90.0
            and self.temperature_celsius < 80.0
        )

    @property
    def memory_usage_ratio(self) -> float:
        """メモリ使用率（0.0-1.0）"""
        if self.memory_total_mb > 0:
            return self.memory_used_mb / self.memory_total_mb
        return 0.0

    def get_status_summary(self) -> str:
        """ステータスサマリーを取得"""
        if not self.is_healthy:
            return "不健全"
        elif self.is_overloaded:
            return "過負荷"
        elif self.gpu_utilization_percent > 70.0:
            return "高負荷"
        elif self.gpu_utilization_percent > 30.0:
            return "中負荷"
        else:
            return "軽負荷"


class GPUDeviceManager:
    """GPU デバイス管理"""

    def __init__(self):
        self.available_devices = self._detect_gpu_devices()
        self.device_properties = {}
        self.memory_pools = {}
        self.monitoring_data = {}

    def _detect_gpu_devices(self) -> List[Dict[str, Any]]:
        """GPU デバイス検出"""
        devices = []

        # CUDA デバイス検出
        if CUPY_AVAILABLE:
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                for i in range(device_count):
                    with cp.cuda.Device(i):
                        props = cp.cuda.runtime.getDeviceProperties(i)
                        devices.append(
                            {
                                "id": i,
                                "backend": GPUBackend.CUDA,
                                "name": props["name"].decode(),
                                "memory_mb": props["totalGlobalMem"] // 1024 // 1024,
                                "compute_capability": f"{props['major']}.{props['minor']}",
                                "multiprocessor_count": props["multiProcessorCount"],
                            }
                        )
                logger.info(f"CUDA デバイス検出: {device_count}個")
            except Exception as e:
                logger.warning(f"CUDA デバイス検出エラー: {e}")

        # OpenCL デバイス検出
        if OPENCL_AVAILABLE:
            try:
                for platform in cl.get_platforms():
                    for device in platform.get_devices():
                        devices.append(
                            {
                                "id": len(devices),
                                "backend": GPUBackend.OPENCL,
                                "name": device.get_info(cl.device_info.NAME),
                                "memory_mb": device.get_info(
                                    cl.device_info.GLOBAL_MEM_SIZE
                                )
                                // 1024
                                // 1024,
                                "platform": platform.get_info(cl.platform_info.NAME),
                            }
                        )
                logger.info(
                    f"OpenCL デバイス検出: {len([d for d in devices if d['backend'] == GPUBackend.OPENCL])}個"
                )
            except Exception as e:
                logger.warning(f"OpenCL デバイス検出エラー: {e}")

        # フォールバック
        if not devices:
            devices.append(
                {
                    "id": 0,
                    "backend": GPUBackend.CPU_FALLBACK,
                    "name": "CPU Fallback",
                    "memory_mb": 8192,
                    "compute_capability": "fallback",
                }
            )
            logger.warning("GPU デバイス未検出 - CPU フォールバック")

        return devices

    def get_optimal_device(self, memory_requirement_mb: int = 1024) -> Dict[str, Any]:
        """最適デバイス選択"""
        suitable_devices = [
            d for d in self.available_devices if d["memory_mb"] >= memory_requirement_mb
        ]

        if not suitable_devices:
            logger.warning(
                f"メモリ要求量 {memory_requirement_mb}MB を満たすデバイスなし - 最大メモリデバイス使用"
            )
            suitable_devices = self.available_devices

        # メモリ量と計算能力で選択
        return max(suitable_devices, key=lambda d: d["memory_mb"])

    def create_memory_pool(self, device_id: int, pool_size_mb: int) -> Optional[Any]:
        """GPU メモリプール作成"""
        try:
            if CUPY_AVAILABLE:
                with cp.cuda.Device(device_id):
                    mempool = cp.get_default_memory_pool()
                    mempool.set_limit(size=pool_size_mb * 1024 * 1024)
                    self.memory_pools[device_id] = mempool
                    logger.info(
                        f"CUDA メモリプール作成: デバイス {device_id}, {pool_size_mb}MB"
                    )
                    return mempool
        except Exception as e:
            logger.warning(f"GPU メモリプール作成失敗: {e}")

        return None

    def get_device_info(self, device_id: int) -> Optional[Dict[str, Any]]:
        """デバイス情報取得"""
        for device in self.available_devices:
            if device["id"] == device_id:
                return device
        return None

    def list_devices(self) -> List[Dict[str, Any]]:
        """利用可能デバイス一覧"""
        return self.available_devices.copy()

    def is_device_available(self, device_id: int) -> bool:
        """デバイス利用可能性チェック"""
        return any(device["id"] == device_id for device in self.available_devices)

    def get_memory_info(self, device_id: int) -> Optional[Dict[str, int]]:
        """メモリ情報取得"""
        try:
            if CUPY_AVAILABLE and self.is_device_available(device_id):
                with cp.cuda.Device(device_id):
                    meminfo = cp.cuda.runtime.memGetInfo()
                    return {
                        "free_mb": meminfo[0] // 1024 // 1024,
                        "total_mb": meminfo[1] // 1024 // 1024,
                        "used_mb": (meminfo[1] - meminfo[0]) // 1024 // 1024,
                    }
        except Exception as e:
            logger.warning(f"メモリ情報取得失敗 (デバイス {device_id}): {e}")

        return None

    def collect_monitoring_data(self, device_id: int) -> Optional[GPUMonitoringData]:
        """GPU監視データ収集"""
        try:
            timestamp = time.time()

            # 基本情報
            monitoring_data = GPUMonitoringData(
                device_id=device_id,
                timestamp=timestamp
            )

            # メモリ情報
            memory_info = self.get_memory_info(device_id)
            if memory_info:
                monitoring_data.memory_used_mb = memory_info["used_mb"]
                monitoring_data.memory_total_mb = memory_info["total_mb"]
                monitoring_data.memory_free_mb = memory_info["free_mb"]
                monitoring_data.memory_utilization_percent = (
                    memory_info["used_mb"] / memory_info["total_mb"] * 100
                    if memory_info["total_mb"] > 0 else 0.0
                )

            # GPU使用率（簡易実装 - 実際にはnvidia-smi等が必要）
            monitoring_data.gpu_utilization_percent = self._estimate_gpu_utilization(device_id)

            # 温度・電力（ダミー値 - 実際にはハードウェア依存）
            monitoring_data.temperature_celsius = 45.0  # プレースホルダー
            monitoring_data.power_consumption_watts = 150.0  # プレースホルダー

            # エラーチェック
            if monitoring_data.memory_utilization_percent > 100:
                monitoring_data.has_errors = True
                monitoring_data.error_message = "メモリ使用率が異常です"

            # 監視データ保存
            self.monitoring_data[device_id] = monitoring_data

            return monitoring_data

        except Exception as e:
            logger.error(f"GPU監視データ収集失敗 (デバイス {device_id}): {e}")
            return GPUMonitoringData(
                device_id=device_id,
                timestamp=time.time(),
                has_errors=True,
                error_message=str(e)
            )

    def _estimate_gpu_utilization(self, device_id: int) -> float:
        """GPU使用率推定（簡易実装）"""
        # 実際の実装では nvidia-ml-py や nvidia-smi を使用
        # ここでは memory pool の使用状況から推定
        try:
            if device_id in self.memory_pools:
                # メモリプールの使用状況から推定
                memory_info = self.get_memory_info(device_id)
                if memory_info and memory_info["total_mb"] > 0:
                    usage_ratio = memory_info["used_mb"] / memory_info["total_mb"]
                    return min(usage_ratio * 100, 100.0)
            return 0.0
        except Exception:
            return 0.0

    def get_latest_monitoring_data(self, device_id: int) -> Optional[GPUMonitoringData]:
        """最新の監視データを取得"""
        return self.monitoring_data.get(device_id)

    def cleanup_memory_pools(self):
        """メモリプールのクリーンアップ"""
        try:
            if CUPY_AVAILABLE:
                for device_id in self.memory_pools:
                    with cp.cuda.Device(device_id):
                        cp.get_default_memory_pool().free_all_blocks()
                logger.info("GPU メモリプールクリーンアップ完了")
        except Exception as e:
            logger.warning(f"GPU メモリプールクリーンアップ失敗: {e}")

    def get_device_summary(self) -> Dict[str, Any]:
        """デバイスサマリー取得"""
        total_memory_mb = sum(device["memory_mb"] for device in self.available_devices)
        cuda_devices = len([d for d in self.available_devices if d["backend"] == GPUBackend.CUDA])
        opencl_devices = len([d for d in self.available_devices if d["backend"] == GPUBackend.OPENCL])

        return {
            "total_devices": len(self.available_devices),
            "cuda_devices": cuda_devices,
            "opencl_devices": opencl_devices,
            "total_memory_mb": total_memory_mb,
            "fallback_mode": any(d["backend"] == GPUBackend.CPU_FALLBACK for d in self.available_devices),
            "devices": self.available_devices
        }