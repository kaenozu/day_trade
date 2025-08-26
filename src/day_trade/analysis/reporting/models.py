"""
レポート管理システム - データモデル定義

【重要】完全セーフモード - 分析・教育・研究専用
実際の取引は一切実行されません
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List


class ReportFormat(Enum):
    """レポート形式"""

    JSON = "json"
    HTML = "html"
    CSV = "csv"
    MARKDOWN = "markdown"


class ReportType(Enum):
    """レポートタイプ"""

    DAILY_SUMMARY = "daily_summary"
    WEEKLY_ANALYSIS = "weekly_analysis"
    MARKET_OVERVIEW = "market_overview"
    SIGNAL_PERFORMANCE = "signal_performance"
    EDUCATIONAL_REPORT = "educational_report"
    COMPARATIVE_ANALYSIS = "comparative_analysis"


@dataclass
class DetailedMarketReport:
    """詳細市場分析レポート"""

    report_id: str
    report_type: ReportType
    generated_at: datetime
    symbols_analyzed: List[str]

    # 市場概要
    market_summary: Dict[str, Any]

    # 個別銘柄分析
    individual_analyses: Dict[str, Dict[str, Any]]

    # トレンド分析
    trend_analysis: Dict[str, Any]

    # 相関分析
    correlation_analysis: Dict[str, Any]

    # ボラティリティ分析
    volatility_analysis: Dict[str, Any]

    # シグナル統計
    signal_statistics: Dict[str, Any]

    # 教育的洞察
    educational_insights: List[str]

    # 推奨事項
    recommendations: List[str]

    # メタデータ
    metadata: Dict[str, Any]