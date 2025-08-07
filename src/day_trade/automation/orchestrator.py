"""
DayTradeOrchestrator - 分析専用オーケストレーター

完全セーフモード - 自動取引機能は一切含まれていません
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..automation.analysis_only_engine import AnalysisOnlyEngine
from ..config.trading_mode_config import get_current_trading_config, is_safe_mode
from ..data.stock_fetcher import StockFetcher
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class ExecutionReport:
    """実行レポート"""

    start_time: datetime
    end_time: datetime
    total_symbols: int
    successful_symbols: int
    failed_symbols: int
    generated_signals: List[Dict[str, Any]]
    triggered_alerts: List[Dict[str, Any]]
    portfolio_summary: Optional[Dict[str, Any]] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class DayTradeOrchestrator:
    """
    デイトレード分析専用オーケストレーター

    【重要】完全セーフモード - 自動取引機能は一切含まれていません

    機能：
    1. 市場データ分析の統合管理
    2. シグナル生成とアラート
    3. レポート生成
    4. 教育的インサイトの提供

    ※ 実際の取引実行は一切行いません
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルパス（オプション）
        """
        # セーフモードチェック
        if not is_safe_mode():
            raise ValueError(
                "セーフモードでない場合は、このオーケストレーターは使用できません"
            )

        self.config_path = config_path
        self.trading_config = get_current_trading_config()
        self.stock_fetcher = StockFetcher()
        self.analysis_engines: Dict[str, AnalysisOnlyEngine] = {}

        logger.info("DayTradeOrchestrator初期化完了 - 完全セーフモード")
        logger.info("※ 自動取引機能は一切含まれていません")

    def run_full_automation(
        self, symbols: Optional[List[str]] = None, report_only: bool = False
    ) -> ExecutionReport:
        """
        完全分析オートメーション実行

        Args:
            symbols: 分析対象銘柄リスト
            report_only: レポート生成のみフラグ

        Returns:
            ExecutionReport: 実行結果レポート
        """
        start_time = datetime.now()

        logger.info(
            f"分析オートメーション開始 - 対象銘柄: {len(symbols) if symbols else 0}"
        )

        if not symbols:
            symbols = ["7203", "8306", "9984"]  # デフォルト銘柄

        generated_signals = []
        triggered_alerts = []
        successful_symbols = 0
        failed_symbols = 0
        errors = []

        try:
            for symbol in symbols:
                try:
                    # 分析エンジン作成
                    if symbol not in self.analysis_engines:
                        self.analysis_engines[symbol] = AnalysisOnlyEngine([symbol])

                    engine = self.analysis_engines[symbol]

                    if not report_only:
                        # 市場データ取得と分析実行
                        _ = engine.get_status()  # Status check for engine health
                        _ = engine.get_market_summary()  # Market data for analysis

                        # サンプルシグナル生成（分析専用）
                        signal = {
                            "symbol": symbol,
                            "type": "ANALYSIS",
                            "timestamp": datetime.now().isoformat(),
                            "confidence": 0.75,
                            "reason": "分析結果に基づく情報提供",
                            "safe_mode": True,
                            "trading_disabled": True,
                        }
                        generated_signals.append(signal)

                        # サンプルアラート
                        alert = {
                            "symbol": symbol,
                            "type": "INFO",
                            "message": f"{symbol}: 分析完了",
                            "timestamp": datetime.now().isoformat(),
                            "severity": "low",
                        }
                        triggered_alerts.append(alert)

                    successful_symbols += 1
                    logger.info(f"銘柄 {symbol} 分析完了")

                except Exception as e:
                    failed_symbols += 1
                    error_msg = f"銘柄 {symbol} 分析エラー: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            # ポートフォリオサマリー（分析専用）
            portfolio_summary = {
                "status": "analysis_only",
                "trading_disabled": True,
                "analyzed_symbols": successful_symbols,
                "metrics": {
                    "total_value": "N/A (分析専用)",
                    "total_pnl": "N/A (分析専用)",
                    "total_pnl_percent": "N/A (分析専用)",
                },
            }

        except Exception as e:
            error_msg = f"オーケストレーター実行エラー: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)

        end_time = datetime.now()

        report = ExecutionReport(
            start_time=start_time,
            end_time=end_time,
            total_symbols=len(symbols),
            successful_symbols=successful_symbols,
            failed_symbols=failed_symbols,
            generated_signals=generated_signals,
            triggered_alerts=triggered_alerts,
            portfolio_summary=portfolio_summary,
            errors=errors,
        )

        logger.info(
            f"分析オートメーション完了 - 成功: {successful_symbols}, 失敗: {failed_symbols}"
        )

        return report

    def get_status(self) -> Dict[str, Any]:
        """
        オーケストレーターステータス取得

        Returns:
            Dict[str, Any]: ステータス情報
        """
        return {
            "safe_mode": is_safe_mode(),
            "trading_disabled": True,
            "automatic_trading": False,
            "analysis_engines": len(self.analysis_engines),
            "config_path": self.config_path,
            "mode": "analysis_only",
        }

    async def run_async_analysis(self, symbols: List[str]) -> ExecutionReport:
        """
        非同期分析実行

        Args:
            symbols: 分析対象銘柄リスト

        Returns:
            ExecutionReport: 実行結果レポート
        """
        logger.info("非同期分析開始")

        # 同期メソッドを非同期で実行
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(
            None, self.run_full_automation, symbols, False
        )

        logger.info("非同期分析完了")
        return report

    def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("オーケストレーター クリーンアップ開始")

        for symbol, engine in self.analysis_engines.items():
            try:
                # エンジンの停止処理があれば実行
                if hasattr(engine, "stop"):
                    engine.stop()
                logger.info(f"エンジン {symbol} クリーンアップ完了")
            except Exception as e:
                logger.warning(f"エンジン {symbol} クリーンアップエラー: {e}")

        self.analysis_engines.clear()
        logger.info("オーケストレーター クリーンアップ完了")

    def __enter__(self):
        """コンテキストマネージャー開始"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了"""
        self.cleanup()
        if exc_type:
            logger.error(
                f"オーケストレーター実行エラー: {exc_type.__name__}: {exc_val}"
            )
        return False  # 例外を再発生させる
