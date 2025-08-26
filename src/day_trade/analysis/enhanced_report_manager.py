"""
強化された分析レポート管理システム - 互換性レイヤー

【重要】完全セーフモード - 分析・教育・研究専用
実際の取引は一切実行されません

このファイルは後方互換性のために保持されています。
実際の実装は reporting パッケージに移行されました。
"""

from typing import Optional

from .reporting import (
    DetailedMarketReport,
    EnhancedReportManager as NewEnhancedReportManager,
    ReportFormat,
    ReportType,
)
from ..automation.analysis_only_engine import AnalysisOnlyEngine
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 後方互換性のためのエクスポート
__all__ = [
    "ReportFormat",
    "ReportType", 
    "DetailedMarketReport",
    "EnhancedReportManager",
]


class EnhancedReportManager(NewEnhancedReportManager):
    """後方互換性のためのラッパークラス"""

    def __init__(self, analysis_engine: Optional[AnalysisOnlyEngine] = None) -> None:
        """互換性を保持した初期化"""
        logger.info("後方互換性レイヤー経由でEnhancedReportManagerを初期化")
        super().__init__(analysis_engine)