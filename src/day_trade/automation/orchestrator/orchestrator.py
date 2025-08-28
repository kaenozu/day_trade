"""
メインオーケストレータークラス
Next-Gen AI Trading Engine の中核管理クラス
"""

import asyncio
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...automation.analysis_only_engine import AnalysisOnlyEngine
from ...config.trading_mode_config import get_current_trading_config, is_safe_mode
from ...core.portfolio import PortfolioManager
from ...data.stock_fetcher import StockFetcher
from ...models.database import get_default_database_manager
from ...utils.logging_config import get_context_logger
from ...utils.stock_name_helper import get_stock_helper
from .analysis_engine import AnalysisEngine
from .config import (
    AIAnalysisResult,
    CI_MODE,
    ExecutionReport,
    OrchestrationConfig,
    get_default_config,
)
from .monitor import SystemMonitor
from .signal_generator import SignalGenerator
from .task_manager import TaskManager

logger = get_context_logger(__name__)

# 条件付きインポート
if not CI_MODE:
    try:
        from ...data.advanced_ml_engine import (
            AdvancedMLEngine,
            ModelConfig,
            create_advanced_ml_engine,
        )
        from ...data.batch_data_fetcher import (
            AdvancedBatchDataFetcher,
            DataRequest,
            DataResponse,
        )
        from ...utils.performance_monitor import PerformanceMonitor
    except ImportError:
        logger.warning("高度機能のインポートに失敗。基本機能のみ利用可能。")
        AdvancedMLEngine = None
        ModelConfig = None
        create_advanced_ml_engine = None
        AdvancedBatchDataFetcher = None
        DataRequest = None
        DataResponse = None
        PerformanceMonitor = None
else:
    # CI環境では軽量ダミークラス使用
    AdvancedMLEngine = None
    ModelConfig = None
    create_advanced_ml_engine = None
    AdvancedBatchDataFetcher = None
    DataRequest = None
    DataResponse = None
    PerformanceMonitor = None


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
        ml_config: Optional['ModelConfig'] = None,
        config_path: Optional[str] = None,
    ):
        """
        初期化
        
        Args:
            config: オーケストレーション設定
            ml_config: MLモデル設定
            config_path: 設定ファイルパス（オプション）
        """
        # セーフモードチェック
        if not is_safe_mode():
            raise ValueError(
                "セーフモードでない場合は、このオーケストレーターは使用できません"
            )

        self.config = config or get_default_config()
        self.config_path = config_path
        self.trading_config = get_current_trading_config()

        # CI環境では軽量化
        if CI_MODE:
            self.config.enable_ml_engine = False
            self.config.enable_advanced_batch = False
            self.ml_config = None
            logger.info("CI軽量モード: ML機能を無効化")
        else:
            self.ml_config = ml_config or (ModelConfig() if ModelConfig else None)

        # コアコンポーネント初期化
        self._initialize_core_components()

        # 高度AIコンポーネント初期化
        self._initialize_advanced_components()

        # 実行統計
        self.execution_history = []
        self.performance_metrics = {}

        logger.info("Next-Gen AI Orchestrator 初期化完了 - 完全セーフモード")
        logger.info("※ 自動取引機能は一切含まれていません")
        logger.info(
            f"設定: ML={self.config.enable_ml_engine}, "
            f"Batch={self.config.enable_advanced_batch}"
        )

    def _initialize_core_components(self) -> None:
        """コアコンポーネント初期化"""
        self.stock_fetcher = StockFetcher()
        self.stock_helper = get_stock_helper()
        self.db_manager = get_default_database_manager()
        
        # サブコンポーネント初期化
        self.task_manager = TaskManager(self.config)
        self.analysis_engine = AnalysisEngine(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.system_monitor = SystemMonitor(self.config)

    def _initialize_advanced_components(self) -> None:
        """高度AIコンポーネント初期化"""
        # MLエンジン
        if self.config.enable_ml_engine and not CI_MODE and create_advanced_ml_engine:
            try:
                self.ml_engine = create_advanced_ml_engine(asdict(self.ml_config))
                self.analysis_engine.set_ml_engine(self.ml_engine)
                self.system_monitor.set_ml_engine(self.ml_engine)
                logger.info("MLエンジン初期化完了")
            except Exception as e:
                logger.warning(f"MLエンジン初期化失敗: {e}")
                self.ml_engine = None
        else:
            self.ml_engine = None

        # バッチフェッチャー
        if self.config.enable_advanced_batch and not CI_MODE and AdvancedBatchDataFetcher:
            try:
                self.batch_fetcher = AdvancedBatchDataFetcher(
                    max_workers=self.config.max_workers,
                    enable_kafka=False,  # セーフモードではKafka無効
                    enable_redis=False,  # セーフモードではRedis無効
                )
                self.system_monitor.set_batch_fetcher(self.batch_fetcher)
                logger.info("バッチフェッチャー初期化完了")
            except Exception as e:
                logger.warning(f"バッチフェッチャー初期化失敗: {e}")
                self.batch_fetcher = None
        else:
            self.batch_fetcher = None

        # パフォーマンス監視
        if self.config.enable_performance_monitoring and PerformanceMonitor:
            try:
                self.performance_monitor = PerformanceMonitor()
                self.system_monitor.set_performance_monitor(self.performance_monitor)
                logger.info("パフォーマンス監視初期化完了")
            except Exception as e:
                logger.warning(f"パフォーマンス監視初期化失敗: {e}")
                self.performance_monitor = None
        else:
            self.performance_monitor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了時の包括的リソースクリーンアップ"""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"__exit__ クリーンアップエラー: {e}")

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

        try:
            # ポートフォリオ情報取得
            portfolio_summary = self._get_portfolio_summary()

            # 高度バッチデータ取得
            batch_results = self._execute_batch_data_collection(symbols)

            # AI分析実行
            results = self._execute_analysis_pipeline(
                symbols, batch_results, analysis_type, include_predictions
            )

            # 結果集計とシグナル/アラート生成
            for symbol, result in results.items():
                if result["success"]:
                    successful_symbols += 1
                    analysis_result = result["analysis"]
                    
                    ai_analysis_results.append(analysis_result)
                    
                    # シグナル生成
                    signals = self.signal_generator.generate_ai_signals(analysis_result)
                    generated_signals.extend(signals)
                    
                    # アラート生成
                    alerts = self.signal_generator.generate_smart_alerts(analysis_result)
                    triggered_alerts.extend(alerts)
                else:
                    failed_symbols += 1
                    errors.extend(result["errors"])

            # システムヘルス分析
            system_health = self.system_monitor.analyze_system_health()

            # パフォーマンス統計
            performance_stats = self.system_monitor.calculate_performance_stats(start_time)

            # ポートフォリオ分析更新
            if ai_analysis_results and not portfolio_summary:
                portfolio_summary = self.system_monitor.generate_portfolio_analysis(
                    ai_analysis_results
                )

        except Exception as e:
            error_msg = f"高度AI分析実行エラー: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            # デフォルト値設定
            system_health = {"overall_status": "error", "error": str(e)}
            performance_stats = {"error": str(e)}
            portfolio_summary = {"status": "error", "error": str(e)}

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
        self.execution_history.append(report)
        self.execution_history = self.execution_history[-50:]  # 最新50件のみ保持

        # サマリーログ
        self._log_execution_summary(report)

        return report

    def _get_portfolio_summary(self) -> Optional[Dict[str, Any]]:
        """ポートフォリオ情報取得"""
        try:
            with self.db_manager.session_scope() as session:
                portfolio_manager = PortfolioManager(session)
                return portfolio_manager.get_portfolio_summary()
        except Exception as e:
            logger.warning(f"ポートフォリオ情報取得失敗: {e}")
            return None

    def _execute_batch_data_collection(
        self, symbols: List[str]
    ) -> Dict[str, 'DataResponse']:
        """高度バッチデータ収集"""
        if not self.batch_fetcher:
            return {}

        # 銘柄名を含む詳細情報を表示
        symbol_names = [
            self.stock_helper.format_stock_display(symbol, include_code=False)
            for symbol in symbols[:5]
        ]
        if len(symbols) > 5:
            symbol_names.append(f"他{len(symbols)-5}銘柄")
        
        logger.info(f"バッチデータ収集開始: {len(symbols)} 銘柄 ({', '.join(symbol_names)})")

        try:
            # データリクエスト作成
            requests = [
                DataRequest(
                    symbol=symbol,
                    period="1y",  # より長期間のデータ
                    preprocessing=True,
                    features=[
                        "trend_strength",
                        "momentum", 
                        "price_channel",
                        "gap_analysis",
                    ],
                    priority=5 if symbol in ["7203", "8306"] else 3,
                    cache_ttl=3600,
                )
                for symbol in symbols
            ]

            # バッチ実行
            return self.batch_fetcher.fetch_batch(requests, use_parallel=True)

        except Exception as e:
            logger.error(f"バッチデータ収集エラー: {e}")
            return {}

    def _execute_analysis_pipeline(
        self,
        symbols: List[str],
        batch_data: Dict[str, 'DataResponse'],
        analysis_type: str,
        include_predictions: bool,
    ) -> Dict[str, Dict]:
        """分析パイプライン実行"""
        if len(symbols) > 1:
            # 並列分析実行
            return self.task_manager.execute_parallel_ai_analysis(
                symbols,
                batch_data,
                analysis_type,
                include_predictions,
                self.analysis_engine.analyze_single_symbol,
            )
        else:
            # 逐次分析実行
            return self.task_manager.execute_sequential_ai_analysis(
                symbols,
                batch_data,
                analysis_type,
                include_predictions,
                self.analysis_engine.analyze_single_symbol,
            )

    def _log_execution_summary(self, report: ExecutionReport) -> None:
        """実行サマリーログ出力"""
        successful_symbol_names = []
        failed_symbol_names = []
        
        # 成功した銘柄の表示名を生成
        for result in report.ai_analysis_results:
            symbol_display = self.stock_helper.format_stock_display(result.symbol)
            successful_symbol_names.append(symbol_display)
            
        # 失敗した銘柄の処理（エラーから推測）
        total_processed = len(successful_symbol_names)
        total_failed = report.failed_symbols
        
        if total_failed > 0:
            failed_symbol_names = [f"失敗{i+1}" for i in range(total_failed)]

        # サマリー情報生成
        success_summary = f"成功: {report.successful_symbols}銘柄"
        if successful_symbol_names:
            success_list = ", ".join(successful_symbol_names[:3])
            if len(successful_symbol_names) > 3:
                success_list += f" 他{len(successful_symbol_names)-3}銘柄"
            success_summary += f" ({success_list})"

        fail_summary = f"失敗: {report.failed_symbols}銘柄"
        if failed_symbol_names:
            fail_list = ", ".join(failed_symbol_names[:3])
            if len(failed_symbol_names) > 3:
                fail_list += f" 他{len(failed_symbol_names)-3}銘柄"
            fail_summary += f" ({fail_list})"

        logger.info(f"Next-Gen AI分析完了 - {success_summary}, {fail_summary}")

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
        return self.execution_history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """オーケストレーターステータス取得"""
        status = {
            "safe_mode": is_safe_mode(),
            "trading_disabled": True,
            "automatic_trading": False,
            "config_path": self.config_path,
            "mode": "next_gen_ai_analysis",
            "components": {
                "ml_engine_enabled": self.config.enable_ml_engine,
                "advanced_batch_enabled": self.config.enable_advanced_batch,
                "performance_monitoring": self.config.enable_performance_monitoring,
                "fault_tolerance": self.config.enable_fault_tolerance,
            },
            "execution_count": len(self.execution_history),
            "last_execution": (
                self.execution_history[-1].start_time.isoformat()
                if self.execution_history
                else None
            ),
        }
        
        # サブコンポーネントのステータスを追加
        try:
            status["task_manager"] = self.task_manager.get_status()
            status["signal_generator"] = self.signal_generator.get_status()
            status["system_monitor"] = self.system_monitor.get_monitoring_summary()
        except Exception as e:
            status["component_status_error"] = str(e)
            
        return status

    def cleanup(self) -> Dict[str, Any]:
        """リソースクリーンアップ"""
        logger.info("Next-Gen AI Orchestrator クリーンアップ開始")

        cleanup_summary = {
            "ml_engine": False,
            "batch_fetcher": False,
            "analysis_engine": {},
            "task_manager": {},
            "system_monitor": {},
            "stock_fetcher": False,
            "errors": []
        }

        try:
            # MLエンジンクリーンアップ
            if hasattr(self, 'ml_engine') and self.ml_engine:
                try:
                    if hasattr(self.ml_engine, "model") and self.ml_engine.model:
                        if hasattr(self.ml_engine.model, "cpu"):
                            self.ml_engine.model.cpu()
                        del self.ml_engine.model

                    if hasattr(self.ml_engine, "close"):
                        self.ml_engine.close()
                    if hasattr(self.ml_engine, "cleanup"):
                        self.ml_engine.cleanup()

                    if hasattr(self.ml_engine, "performance_history"):
                        self.ml_engine.performance_history.clear()

                    self.ml_engine = None
                    cleanup_summary["ml_engine"] = True
                except Exception as e:
                    error_msg = f"MLエンジン クリーンアップエラー: {e}"
                    cleanup_summary["errors"].append(error_msg)

            # バッチフェッチャークリーンアップ  
            if hasattr(self, 'batch_fetcher') and self.batch_fetcher:
                try:
                    self.batch_fetcher.close()
                    self.batch_fetcher = None
                    cleanup_summary["batch_fetcher"] = True
                except Exception as e:
                    error_msg = f"バッチフェッチャー クリーンアップエラー: {e}"
                    cleanup_summary["errors"].append(error_msg)

            # サブコンポーネントクリーンアップ
            if hasattr(self, 'analysis_engine'):
                cleanup_summary["analysis_engine"] = self.analysis_engine.cleanup()
                
            if hasattr(self, 'task_manager'):
                cleanup_summary["task_manager"] = self.task_manager.cleanup()
                
            if hasattr(self, 'system_monitor'):
                cleanup_summary["system_monitor"] = self.system_monitor.cleanup()

            # ストックフェッチャークリーンアップ
            if hasattr(self, 'stock_fetcher') and self.stock_fetcher:
                try:
                    if hasattr(self.stock_fetcher, "close"):
                        self.stock_fetcher.close()
                    self.stock_fetcher = None
                    cleanup_summary["stock_fetcher"] = True
                except Exception as e:
                    error_msg = f"ストックフェッチャー クリーンアップエラー: {e}"
                    cleanup_summary["errors"].append(error_msg)

            # 実行履歴クリア
            if hasattr(self, 'execution_history'):
                self.execution_history.clear()

            # ガベージコレクション強制実行
            import gc
            gc.collect()

            # クリーンアップサマリーログ
            if cleanup_summary["errors"]:
                logger.warning(
                    f"クリーンアップ完了（エラー{len(cleanup_summary['errors'])}件あり）"
                )
            else:
                logger.info("Next-Gen AI Orchestrator クリーンアップ完了")

        except Exception as e:
            logger.error(f"クリーンアップ致命的エラー: {e}")
            cleanup_summary["errors"].append(f"致命的エラー: {e}")

        return cleanup_summary


# 後方互換性のためのエイリアス
DayTradeOrchestrator = NextGenAIOrchestrator