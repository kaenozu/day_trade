"""
メインオーケストレータークラス
分割されたモジュールを統合し、高度AI分析を提供
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...utils.logging_config import get_context_logger
from .ai_analyzer import AIAnalyzer
from .core import OrchestratorCore
from .data_processor import DataProcessor
from .parallel_processor import ParallelProcessor
from .portfolio_analyzer import PortfolioAnalyzer
from .resource_manager import ResourceManager
from .signal_generator import SignalGenerator
from .types import AIAnalysisResult, ExecutionReport, ModelConfig, OrchestrationConfig

logger = get_context_logger(__name__)


class NextGenAIOrchestrator:
    """
    次世代AI取引エンジン オーケストレーター

    【重要】完全セーフモード - 自動取引機能は一切含まれていません

    高度機能：
    1. LSTM-Transformer ハイブリッドモデル統合
    2. 大規模並列AI分析パイプライン
    3. リアルタイムデータ処理・品質管理
    4. 高度リスク評価・ポートフォリオ最適化
    5. 包括的システム監視・フォールトトレラント

    ※ 実際の取引実行は一切行いません（分析・教育目的のみ）
    """

    def __init__(
        self,
        config: Optional[OrchestrationConfig] = None,
        ml_config: Optional[ModelConfig] = None,
        config_path: Optional[str] = None,
    ):
        """
        初期化

        Args:
            config: オーケストレーション設定
            ml_config: MLモデル設定
            config_path: 設定ファイルパス（オプション）
        """
        # コア初期化
        self.core = OrchestratorCore(config, ml_config, config_path)
        
        # モジュール初期化
        self.parallel_processor = ParallelProcessor(
            self.core.config, self.core.parallel_manager
        )
        
        self.data_processor = DataProcessor(
            self.core, self.core.config, self.core.stock_helper
        )
        
        self.ai_analyzer = AIAnalyzer(
            self.core, self.core.config, self.core.stock_helper
        )
        
        self.signal_generator = SignalGenerator(self.core.config)
        
        self.portfolio_analyzer = PortfolioAnalyzer(
            self.core, self.core.config
        )
        
        self.resource_manager = ResourceManager(self.core)

        logger.info("Next-Gen AI Orchestrator 初期化完了 - モジュール分割版")

    def run_advanced_analysis(
        self,
        symbols: Optional[List[str]] = None,
        analysis_type: str = "comprehensive",
        include_predictions: bool = True,
    ) -> ExecutionReport:
        """
        高度AI分析実行

        Args:
            symbols: 分析対象銘柄リスト
            analysis_type: 分析タイプ ("basic", "comprehensive", "ml_focus")
            include_predictions: 予測分析を含むか

        Returns:
            ExecutionReport: 詳細実行結果レポート
        """
        start_time = datetime.now()

        logger.info(f"Next-Gen AI分析開始 - タイプ: {analysis_type}")
        logger.info(f"対象銘柄: {len(symbols) if symbols else 0}")

        if not symbols:
            symbols = ["7203", "8306", "9984", "6758", "4689"]  # デフォルト銘柄

        # 分析実行
        generated_signals = []
        triggered_alerts = []
        ai_analysis_results = []
        successful_symbols = 0
        failed_symbols = 0
        errors = []
        successful_symbol_names = []
        failed_symbol_names = []

        try:
            # ポートフォリオ情報取得
            actual_portfolio_summary = self._get_portfolio_summary()

            # 高度バッチデータ取得
            batch_results = self.data_processor.execute_batch_data_collection(symbols)

            # 分析エンジン初期化
            for symbol in symbols:
                try:
                    # 分析エンジン作成
                    if symbol not in self.core.analysis_engines:
                        from ..analysis_only_engine import AnalysisOnlyEngine
                        self.core.analysis_engines[symbol] = AnalysisOnlyEngine([symbol])
                except Exception as e:
                    logger.warning(
                        f"Failed to create analysis engine for {symbol}: {e}"
                    )

            # 並列AI分析実行
            if (self.parallel_processor.is_parallel_available() and len(symbols) > 1):
                results = self.parallel_processor.execute_parallel_ai_analysis(
                    symbols, 
                    lambda symbol: self.ai_analyzer.analyze_single_symbol(
                        symbol, 
                        batch_results.get(symbol), 
                        analysis_type, 
                        include_predictions
                    )
                )
            else:
                results = self.parallel_processor.execute_sequential_ai_analysis(
                    symbols,
                    lambda symbol: self.ai_analyzer.analyze_single_symbol(
                        symbol, 
                        batch_results.get(symbol), 
                        analysis_type, 
                        include_predictions
                    )
                )

            # 結果集計
            for symbol, result in results.items():
                if result["success"]:
                    successful_symbols += 1
                    successful_symbol_names.append(
                        self.core.stock_helper.format_stock_display(symbol)
                    )
                    ai_analysis_results.append(result["analysis"])
                    
                    # シグナル・アラート生成
                    if result["analysis"]:
                        signals = self.signal_generator.generate_ai_signals(result["analysis"])
                        alerts = self.signal_generator.generate_smart_alerts(result["analysis"])
                        generated_signals.extend(signals)
                        triggered_alerts.extend(alerts)
                else:
                    failed_symbols += 1
                    failed_symbol_names.append(
                        self.core.stock_helper.format_stock_display(symbol)
                    )
                    errors.extend(result["errors"])

            # ポートフォリオ分析統合
            if actual_portfolio_summary:
                portfolio_summary = actual_portfolio_summary
            else:
                portfolio_summary = self.portfolio_analyzer.generate_portfolio_analysis(
                    ai_analysis_results
                )

            # システムヘルス分析
            system_health = self.portfolio_analyzer.analyze_system_health()

            # パフォーマンス統計
            performance_stats = self.portfolio_analyzer.calculate_performance_stats(start_time)

        except Exception as e:
            error_msg = f"高度AI分析実行エラー: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)

        end_time = datetime.now()

        # 実行レポート作成
        report = ExecutionReport(
            start_time=start_time,
            end_time=end_time,
            total_symbols=len(symbols),
            successful_symbols=successful_symbols,
            failed_symbols=failed_symbols,
            generated_signals=generated_signals,
            triggered_alerts=triggered_alerts,
            ai_analysis_results=ai_analysis_results,
            portfolio_summary=portfolio_summary,
            performance_stats=performance_stats,
            system_health=system_health,
            errors=errors,
        )

        # 実行履歴に保存
        self.core.execution_history.append(report)
        self.core.execution_history = self.core.execution_history[-50:]  # 最新50件のみ保持

        # 詳細なサマリー情報を表示
        self._log_execution_summary(
            successful_symbols, successful_symbol_names,
            failed_symbols, failed_symbol_names
        )

        logger.info(f"Next-Gen AI分析完了")
        return report

    def _get_portfolio_summary(self) -> Optional[Dict[str, Any]]:
        """ポートフォリオ情報取得"""
        try:
            from ...core.portfolio import PortfolioManager
            
            with self.core.db_manager.session_scope() as session:
                portfolio_manager = PortfolioManager(session)
                return portfolio_manager.get_portfolio_summary()
                
        except Exception as e:
            logger.warning(f"Portfolio summary not available: {e}")
            return None

    def _log_execution_summary(
        self, 
        successful_symbols: int, 
        successful_symbol_names: List[str],
        failed_symbols: int, 
        failed_symbol_names: List[str]
    ):
        """実行サマリーログ出力"""
        success_summary = f"成功: {successful_symbols}銘柄"
        if successful_symbol_names:
            success_list = ", ".join(successful_symbol_names[:3])
            if len(successful_symbol_names) > 3:
                success_list += f" 他{len(successful_symbol_names)-3}銘柄"
            success_summary += f" ({success_list})"

        fail_summary = f"失敗: {failed_symbols}銘柄"
        if failed_symbol_names:
            fail_list = ", ".join(failed_symbol_names[:3])
            if len(failed_symbol_names) > 3:
                fail_list += f" 他{len(failed_symbol_names)-3}銘柄"
            fail_summary += f" ({fail_list})"

        logger.info(f"実行結果 - {success_summary}, {fail_summary}")

    async def run_async_advanced_analysis(
        self, symbols: List[str], analysis_type: str = "comprehensive"
    ) -> ExecutionReport:
        """非同期高度分析実行"""
        logger.info("非同期Next-Gen AI分析開始")

        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            None, self.run_advanced_analysis, symbols, analysis_type, True
        )

        logger.info("非同期Next-Gen AI分析完了")
        return report

    def get_execution_history(self, limit: int = 10) -> List[ExecutionReport]:
        """実行履歴取得"""
        return self.core.get_execution_history(limit)

    def get_status(self) -> Dict[str, Any]:
        """オーケストレーターステータス取得"""
        core_status = self.core.get_status()
        
        # 追加のモジュール情報
        core_status.update({
            "modules": {
                "parallel_processor": bool(self.parallel_processor),
                "data_processor": bool(self.data_processor),
                "ai_analyzer": bool(self.ai_analyzer),
                "signal_generator": bool(self.signal_generator),
                "portfolio_analyzer": bool(self.portfolio_analyzer),
                "resource_manager": bool(self.resource_manager),
            },
            "architecture": "modular_design",
            "version": "2.0.0",
        })
        
        return core_status

    def get_resource_usage(self) -> Dict[str, Any]:
        """リソース使用状況取得"""
        return self.resource_manager.get_resource_usage()

    def cleanup(self) -> Dict[str, Any]:
        """リソースクリーンアップ"""
        return self.resource_manager.cleanup()

    def emergency_cleanup(self) -> Dict[str, Any]:
        """緊急時クリーンアップ"""
        return self.resource_manager.emergency_cleanup()

    def __enter__(self):
        """コンテキストマネージャー開始"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了時のリソースクリーンアップ"""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"__exit__ クリーンアップエラー: {e}")


# 後方互換性のためのエイリアス
DayTradeOrchestrator = NextGenAIOrchestrator