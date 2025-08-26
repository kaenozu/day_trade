#!/usr/bin/env python3
"""
GPU監視機能
Issue #720: GPU監視機能追加
"""

import subprocess
import time
import warnings
from typing import Dict, List, Optional

from ..utils.logging_config import get_context_logger
from .types import GPUMonitoringData

logger = get_context_logger(__name__)

# NVIDIA GPU管理 (フォールバック対応)
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
    logger.info("NVIDIA Management Library (pynvml) 初期化成功")
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available - GPU監視機能制限")
except Exception as e:
    PYNVML_AVAILABLE = False
    logger.warning(f"NVML初期化失敗: {e}")


class GPUMonitor:
    """GPU監視機能"""

    def __init__(self, device_id: int):
        self.device_id = device_id

    def get_comprehensive_gpu_monitoring(self) -> GPUMonitoringData:
        """包括的なGPU監視データの取得"""
        monitoring_data = GPUMonitoringData(device_id=self.device_id)

        try:
            if PYNVML_AVAILABLE:
                self._populate_nvml_monitoring_data(monitoring_data)
            else:
                self._populate_nvidia_smi_monitoring_data(monitoring_data)
        except Exception as e:
            logger.warning(f"GPU監視データ取得エラー: {e}")
            monitoring_data.timestamp = time.time()

        return monitoring_data

    def _populate_nvml_monitoring_data(self, monitoring_data: GPUMonitoringData):
        """NVMLを使用して監視データを取得"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

            # GPU使用率
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            monitoring_data.utilization_percent = float(utilization.gpu)

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
                monitoring_data.power_draw_watts = power / 1000.0  # mWから変換
            except:
                pass

            # ファン速度（可能であれば）
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                monitoring_data.fan_speed_percent = float(fan_speed)
            except:
                pass

            # クロック周波数
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                monitoring_data.clock_graphics_mhz = graphics_clock
                monitoring_data.clock_memory_mhz = memory_clock
            except:
                pass

            # タイムスタンプ設定
            monitoring_data.timestamp = time.time()

        except Exception as e:
            raise Exception(f"NVML監視データ取得エラー: {e}")

    def _populate_nvidia_smi_monitoring_data(self, monitoring_data: GPUMonitoringData):
        """nvidia-smiを使用して監視データを取得"""
        try:
            # 複数の情報を一度に取得
            query_items = [
                'utilization.gpu',
                'memory.total',
                'memory.used',
                'memory.free',
                'temperature.gpu',
                'power.draw',
                'fan.speed',
                'clocks.current.graphics',
                'clocks.current.memory'
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

                if len(values) >= 6:
                    monitoring_data.utilization_percent = self._safe_float_conversion(values[0])
                    monitoring_data.memory_total_mb = self._safe_float_conversion(values[1])
                    monitoring_data.memory_used_mb = self._safe_float_conversion(values[2])
                    monitoring_data.memory_free_mb = self._safe_float_conversion(values[3])
                    monitoring_data.temperature_celsius = self._safe_float_conversion(values[4])
                    monitoring_data.power_draw_watts = self._safe_float_conversion(values[5])

                if len(values) >= 9:
                    monitoring_data.fan_speed_percent = self._safe_float_conversion(values[6])
                    monitoring_data.clock_graphics_mhz = int(self._safe_float_conversion(values[7]))
                    monitoring_data.clock_memory_mhz = int(self._safe_float_conversion(values[8]))

            # タイムスタンプ設定
            monitoring_data.timestamp = time.time()

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

    def get_gpu_utilization(self) -> float:
        """GPU使用率取得"""
        try:
            if PYNVML_AVAILABLE:
                return self._get_gpu_utilization_nvml()
            else:
                return self._get_gpu_utilization_nvidia_smi()
        except Exception as e:
            logger.warning(f"GPU使用率取得エラー: {e}")
            return 0.0

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