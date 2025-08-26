"""
レポート管理システム - コア管理機能

【重要】完全セーフモード - 分析・教育・研究専用
実際の取引は一切実行されません
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .history_manager import HistoryManager
from .insight_generators import InsightGenerators
from .market_analyzers import MarketAnalyzers
from .models import DetailedMarketReport, ReportFormat, ReportType
from .report_exporters import ReportExporters
from .signal_analyzers import SignalAnalyzers
from ...automation.analysis_only_engine import AnalysisOnlyEngine
from ...config.trading_mode_config import is_safe_mode
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class EnhancedReportManager:
    """強化された分析レポート管理システム"""

    def __init__(self, analysis_engine: Optional[AnalysisOnlyEngine] = None) -> None:
        """初期化"""
        # セーフモードチェック
        if not is_safe_mode():
            raise ValueError("セーフモードでない場合は使用できません")

        self.analysis_engine = analysis_engine
        self.export_directory = Path("reports")
        self.export_directory.mkdir(exist_ok=True)

        # コンポーネント初期化
        self.history_manager = HistoryManager()
        self.market_analyzers = MarketAnalyzers()
        self.signal_analyzers = SignalAnalyzers()
        self.insight_generators = InsightGenerators()
        self.report_exporters = ReportExporters(self.export_directory)

        logger.info("強化された分析レポート管理システム初期化完了")
        logger.info("※ 完全セーフモード - 分析・教育・研究専用")

    def generate_comprehensive_report(
        self,
        report_type: ReportType = ReportType.MARKET_OVERVIEW,
        symbols: Optional[List[str]] = None,
    ) -> DetailedMarketReport:
        """包括的分析レポート生成"""

        if not self.analysis_engine:
            raise ValueError("分析エンジンが設定されていません")

        # 分析対象銘柄
        target_symbols = symbols or self.analysis_engine.symbols

        logger.info(f"包括的分析レポート生成開始: {report_type.value}")

        # 基本データ収集
        all_analyses = self.analysis_engine.get_all_analyses()
        latest_report = self.analysis_engine.get_latest_report()
        market_summary = self.analysis_engine.get_market_summary()

        # レポートID生成
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 詳細分析実行
        trend_analysis = self.market_analyzers.analyze_market_trends(
            all_analyses, target_symbols
        )
        correlation_analysis = self.market_analyzers.analyze_correlations(
            all_analyses, target_symbols
        )
        volatility_analysis = self.market_analyzers.analyze_volatility_patterns(
            all_analyses, target_symbols
        )
        signal_stats = self.signal_analyzers.calculate_signal_statistics(
            all_analyses, latest_report
        )
        educational_insights = self.insight_generators.generate_educational_insights(
            all_analyses, trend_analysis, signal_stats
        )
        recommendations = self.insight_generators.generate_smart_recommendations(
            all_analyses, trend_analysis, signal_stats
        )

        # 個別銘柄詳細分析
        detailed_individual = self.signal_analyzers.create_detailed_individual_analyses(
            all_analyses, target_symbols
        )

        # 個別銘柄に教育ノートを追加
        for symbol in target_symbols:
            if symbol in all_analyses and symbol in detailed_individual:
                analysis = all_analyses[symbol]
                detailed_individual[symbol][
                    "educational_notes"
                ] = self.insight_generators.generate_individual_educational_notes(
                    analysis
                )

        # メタデータ
        metadata = {
            "analysis_engine_version": "1.0",
            "safe_mode": True,
            "trading_disabled": True,
            "generation_time_seconds": 0,  # 実装時に計測
            "data_freshness": self.market_analyzers.calculate_data_freshness(
                all_analyses
            ),
            "disclaimer": "このレポートは分析情報です。投資判断は自己責任で行ってください。",
        }

        # レポート作成
        report = DetailedMarketReport(
            report_id=report_id,
            report_type=report_type,
            generated_at=datetime.now(),
            symbols_analyzed=target_symbols,
            market_summary=market_summary,
            individual_analyses=detailed_individual,
            trend_analysis=trend_analysis,
            correlation_analysis=correlation_analysis,
            volatility_analysis=volatility_analysis,
            signal_statistics=signal_stats,
            educational_insights=educational_insights,
            recommendations=recommendations,
            metadata=metadata,
        )

        # 履歴に追加
        self.history_manager.add_report(report)

        logger.info(f"包括的分析レポート生成完了: {report_id}")
        return report

    def export_report(
        self,
        report: DetailedMarketReport,
        format: ReportFormat = ReportFormat.JSON,
        filename: Optional[str] = None,
    ) -> Path:
        """レポートエクスポート"""
        return self.report_exporters.export_report(report, format, filename)

    def get_report_history(self, limit: int = 10) -> List[DetailedMarketReport]:
        """レポート履歴取得"""
        return self.history_manager.get_report_history(limit)

    def clear_history(self) -> None:
        """履歴クリア"""
        self.history_manager.clear_history()

    def get_summary_statistics(self) -> dict:
        """要約統計取得"""
        stats = self.history_manager.get_summary_statistics()
        stats["export_directory"] = str(self.export_directory)
        return stats