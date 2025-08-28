#!/usr/bin/env python3
"""
パフォーマンスメトリクス

パフォーマンス測定結果を格納するためのデータ構造を提供します。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceMetrics:
    """パフォーマンス測定結果"""

    function_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: float
    data_size: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
