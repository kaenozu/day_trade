#!/usr/bin/env python3
"""
GPU デバイス管理
Issue #379: ML Model Inference Performance Optimization
"""

from typing import List, Dict, Any, Optional

from .types import GPUBackend, CUPY_AVAILABLE, OPENCL_AVAILABLE
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# CUDA関連のインポート (フォールバック対応)
if CUPY_AVAILABLE:
    import cupy as cp

# OpenCL関連のインポート (フォールバック対応)
if OPENCL_AVAILABLE:
    import pyopencl as cl


class GPUDeviceManager:
    """GPU デバイス管理"""
    
    def __init__(self):
        self.available_devices = self._detect_gpu_devices()
        self.device_properties = {}
        self.memory_pools = {}
    
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
    
    def get_device_info(self, device_id: int) -> Dict[str, Any]:
        """デバイス情報取得"""
        for device in self.available_devices:
            if device["id"] == device_id:
                return device
        
        logger.warning(f"デバイス {device_id} が見つかりません")
        return {}
    
    def is_device_available(self, device_id: int) -> bool:
        """デバイス利用可能性チェック"""
        return any(d["id"] == device_id for d in self.available_devices)
    
    def get_memory_usage(self, device_id: int) -> Dict[str, float]:
        """デバイスメモリ使用量取得"""
        if not CUPY_AVAILABLE:
            return {"used_mb": 0.0, "free_mb": 0.0, "total_mb": 0.0}
        
        try:
            with cp.cuda.Device(device_id):
                meminfo = cp.cuda.runtime.memGetInfo()
                free_bytes = meminfo[0]
                total_bytes = meminfo[1]
                used_bytes = total_bytes - free_bytes
                
                return {
                    "used_mb": used_bytes / 1024 / 1024,
                    "free_mb": free_bytes / 1024 / 1024,
                    "total_mb": total_bytes / 1024 / 1024,
                }
        except Exception as e:
            logger.warning(f"デバイス {device_id} メモリ情報取得失敗: {e}")
            return {"used_mb": 0.0, "free_mb": 0.0, "total_mb": 0.0}
    
    def cleanup_memory_pools(self):
        """メモリプールクリーンアップ"""
        try:
            for device_id, pool in self.memory_pools.items():
                if CUPY_AVAILABLE and pool:
                    pool.free_all_blocks()
                    logger.debug(f"デバイス {device_id} メモリプールクリーンアップ完了")
            
            self.memory_pools.clear()
        
        except Exception as e:
            logger.error(f"メモリプールクリーンアップエラー: {e}")