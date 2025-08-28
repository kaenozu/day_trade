#!/usr/bin/env python3
"""
TOPIX500 Analysis System - Performance Manager

パフォーマンス管理・統計機能
"""

from typing import Any, Dict

import numpy as np

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class PerformanceManager:
    """パフォーマンス管理機能"""

    def __init__(self, enable_cache: bool = True, enable_parallel: bool = True):
        """
        パフォーマンスマネージャー初期化

        Args:
            enable_cache: キャッシュ有効化
            enable_parallel: 並列処理有効化
        """
        self.enable_cache = enable_cache
        self.enable_parallel = enable_parallel

        # パフォーマンス統計
        self.stats = {
            "total_analyses": 0,
            "batch_analyses": 0,
            "sector_analyses": 0,
            "cache_hits": 0,
            "processing_times": [],
            "memory_usage": [],
            "successful_symbols": 0,
            "failed_symbols": 0,
        }

        # システム設定情報
        self.system_info = {
            "cache_enabled": enable_cache,
            "parallel_enabled": enable_parallel,
        }

        logger.info("パフォーマンスマネージャー初期化完了")

    def update_stats(
        self,
        successful_symbols: int = 0,
        failed_symbols: int = 0,
        processing_time: float = 0.0,
        memory_usage: float = 0.0,
        cache_hits: int = 0,
        batch_analyses: int = 0,
        sector_analyses: int = 0,
    ):
        """統計更新"""
        self.stats["successful_symbols"] += successful_symbols
        self.stats["failed_symbols"] += failed_symbols
        self.stats["cache_hits"] += cache_hits
        self.stats["batch_analyses"] += batch_analyses
        self.stats["sector_analyses"] += sector_analyses
        self.stats["total_analyses"] += 1

        if processing_time > 0:
            self.stats["processing_times"].append(processing_time)

        if memory_usage > 0:
            self.stats["memory_usage"].append(memory_usage)

    def get_performance_stats(
        self,
        max_concurrent_symbols: int = 50,
        memory_limit_gb: float = 1.0,
        processing_timeout: int = 20,
        topix500_symbols_count: int = 0,
        sector_mapping_count: int = 0,
        symbol_data_cache_count: int = 0,
    ) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        return {
            "total_analyses": self.stats["total_analyses"],
            "batch_analyses": self.stats["batch_analyses"],
            "sector_analyses": self.stats["sector_analyses"],
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(self.stats["total_analyses"], 1)
            ),
            "successful_symbols": self.stats["successful_symbols"],
            "failed_symbols": self.stats["failed_symbols"],
            "success_rate": (
                self.stats["successful_symbols"]
                / max(self.stats["successful_symbols"] + self.stats["failed_symbols"], 1)
            ),
            "avg_processing_time": (
                np.mean(self.stats["processing_times"])
                if self.stats["processing_times"]
                else 0.0
            ),
            "system_status": {
                "cache_enabled": self.enable_cache,
                "parallel_enabled": self.enable_parallel,
                "max_concurrent_symbols": max_concurrent_symbols,
                "memory_limit_gb": memory_limit_gb,
                "processing_timeout": processing_timeout,
            },
            "topix500_data": {
                "loaded_symbols": topix500_symbols_count,
                "sectors": sector_mapping_count,
                "cache_size": symbol_data_cache_count,
            },
            "optimization_benefits": {
                "scale_expansion": "85銘柄 → 500銘柄 (6倍拡張)",
                "processing_target": "500銘柄を20秒以内",
                "memory_efficiency": "1GB以内メモリ使用",
                "sector_analysis": "セクター別最適化投資戦略",
            },
        }

    def reset_stats(self):
        """統計リセット"""
        self.stats = {
            "total_analyses": 0,
            "batch_analyses": 0,
            "sector_analyses": 0,
            "cache_hits": 0,
            "processing_times": [],
            "memory_usage": [],
            "successful_symbols": 0,
            "failed_symbols": 0,
        }
        logger.info("パフォーマンス統計リセット完了")

    def get_memory_efficiency_report(self) -> Dict[str, Any]:
        """メモリ効率レポート取得"""
        if not self.stats["memory_usage"]:
            return {"message": "メモリ使用データなし"}

        memory_stats = {
            "avg_memory_mb": np.mean(self.stats["memory_usage"]),
            "max_memory_mb": np.max(self.stats["memory_usage"]),
            "min_memory_mb": np.min(self.stats["memory_usage"]),
            "memory_efficiency": "良好" if np.mean(self.stats["memory_usage"]) < 1024 else "要改善",
        }

        return memory_stats

    def get_throughput_report(self) -> Dict[str, Any]:
        """スループットレポート取得"""
        if not self.stats["processing_times"]:
            return {"message": "処理時間データなし"}

        total_symbols = self.stats["successful_symbols"] + self.stats["failed_symbols"]
        total_time = sum(self.stats["processing_times"])

        throughput_stats = {
            "symbols_per_second": total_symbols / max(total_time, 0.001),
            "avg_processing_time": np.mean(self.stats["processing_times"]),
            "throughput_efficiency": (
                "優秀" if (total_symbols / max(total_time, 0.001)) > 25 else "標準"
            ),
            "target_achievement": (
                "達成" if (total_symbols / max(total_time, 0.001)) > 25 else "未達成"
            ),
        }

        return throughput_stats
