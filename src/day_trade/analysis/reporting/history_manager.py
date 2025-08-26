"""
レポート管理システム - 履歴・統計管理モジュール

【重要】完全セーフモード - 分析・教育・研究専用
実際の取引は一切実行されません
"""

from typing import Any, Dict, List

from .models import DetailedMarketReport
from ...config.trading_mode_config import is_safe_mode
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class HistoryManager:
    """履歴・統計管理機能を提供するクラス"""

    def __init__(self):
        self.report_history: List[DetailedMarketReport] = []
        self.export_directory_str = "reports"

    def add_report(self, report: DetailedMarketReport) -> None:
        """レポートを履歴に追加"""
        self.report_history.append(report)
        logger.info(f"レポート履歴に追加: {report.report_id}")

    def get_report_history(self, limit: int = 10) -> List[DetailedMarketReport]:
        """レポート履歴取得"""
        return self.report_history[-limit:]

    def clear_history(self) -> None:
        """履歴クリア"""
        self.report_history.clear()
        logger.info("レポート履歴をクリアしました")

    def get_summary_statistics(self) -> Dict[str, Any]:
        """要約統計取得"""
        return {
            "total_reports_generated": len(self.report_history),
            "safe_mode": is_safe_mode(),
            "export_directory": self.export_directory_str,
            "last_report_time": (
                self.report_history[-1].generated_at.isoformat()
                if self.report_history
                else None
            ),
        }