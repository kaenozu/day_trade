"""
レポート管理システム - 統合モジュール

【重要】完全セーフモード - 分析・教育・研究専用
実際の取引は一切実行されません

このモジュールは、以前のenhanced_report_manager.pyの機能を
複数のモジュールに分割し、保守性と可読性を向上させたものです。

後方互換性のため、既存のAPIは維持されています。
"""

# 主要クラスとモデルのエクスポート
from .core_manager import EnhancedReportManager
from .models import DetailedMarketReport, ReportFormat, ReportType

# 個別コンポーネントもエクスポート（上級ユーザー向け）
from .market_analyzers import MarketAnalyzers
from .signal_analyzers import SignalAnalyzers
from .insight_generators import InsightGenerators
from .report_exporters import ReportExporters
from .history_manager import HistoryManager

__all__ = [
    # 主要API（後方互換性）
    "EnhancedReportManager",
    "DetailedMarketReport",
    "ReportFormat",
    "ReportType",
    # 個別コンポーネント
    "MarketAnalyzers",
    "SignalAnalyzers",
    "InsightGenerators",
    "ReportExporters",
    "HistoryManager",
]