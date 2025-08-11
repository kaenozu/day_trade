"""
分散処理パッケージ
Issue #384: 並列処理のさらなる強化

Dask & Ray による高度分散コンピューティング
"""

from .dask_data_processor import (
    DaskBatchProcessor,
    DaskDataProcessor,
    DaskStockAnalyzer,
    create_dask_data_processor,
)
from .distributed_computing_manager import (
    ComputingBackend,
    DistributedComputingManager,
    DistributedResult,
    DistributedTask,
    TaskDistributionStrategy,
)

__all__ = [
    "DaskDataProcessor",
    "DaskStockAnalyzer",
    "DaskBatchProcessor",
    "create_dask_data_processor",
    "DistributedComputingManager",
    "ComputingBackend",
    "TaskDistributionStrategy",
    "DistributedTask",
    "DistributedResult",
]
