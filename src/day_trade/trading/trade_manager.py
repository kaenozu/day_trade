"""
統合取引管理システム

全てのコンポーネントを統合した包括的な取引管理インターフェース
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any

from ..utils.logging_config import get_context_logger, log_business_event
from ..utils.enhanced_error_handler import get_default_error_handler

# コア機能
from .core.trade_executor import TradeExecutor
from .core.position_manager import PositionManager
from .core.risk_calculator import RiskCalculator
from .core.types import Trade, Position, TradeType

# 永続化機能
from .persistence.db_manager import TradeDatabaseManager
from .persistence.batch_processor import TradeBatchProcessor
from .persistence.data_cleaner import DataCleaner

# 分析機能
from .analytics.portfolio_analyzer import PortfolioAnalyzer
from .analytics.tax_calculator import TaxCalculator
from .analytics.report_exporter import ReportExporter

# 検証機能
from .validation.trade_validator import TradeValidator
from .validation.compliance_checker import ComplianceChecker
from .validation.id_generator import IDGenerator

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()


class TradeManager:
    """
    統合取引管理クラス

    全ての取引関連機能を統合した包括的なインターフェースを提供
    """

    def __init__(
        self,
        enable_validation: bool = True,
        enable_compliance: bool = True,
        enable_persistence: bool = True,
        commission_rate: Decimal = Decimal("0.001"),
    ):
        """
        初期化

        Args:
            enable_validation: データ検証有効化
            enable_compliance: コンプライアンスチェック有効化
            enable_persistence: 永続化有効化
            commission_rate: 手数料率
        """
        # 基本設定
        self.enable_validation = enable_validation
        self.enable_compliance = enable_compliance
        self.enable_persistence = enable_persistence

        # コア機能の初期化
        self.risk_calculator = RiskCalculator(commission_rate)
        self.position_manager = PositionManager()
        self.trade_executor = TradeExecutor(self.position_manager, self.risk_calculator)

        # 永続化機能の初期化
        if enable_persistence:
            self.db_manager = TradeDatabaseManager()
            self.batch_processor = TradeBatchProcessor(self.db_manager)
            self.data_cleaner = DataCleaner(self.db_manager)
        else:
            self.db_manager = None
            self.batch_processor = None
            self.data_cleaner = None

        # 分析機能の初期化
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.tax_calculator = TaxCalculator()
        self.report_exporter = ReportExporter()

        # 検証機能の初期化
        if enable_validation:
            self.trade_validator = TradeValidator()
        else:
            self.trade_validator = None

        if enable_compliance:
            self.compliance_checker = ComplianceChecker()
        else:
            self.compliance_checker = None

        # ID生成機
        self.id_generator = IDGenerator()

        logger.info(
            f"TradeManager初期化完了 - "
            f"検証: {enable_validation}, "
            f"コンプライアンス: {enable_compliance}, "
            f"永続化: {enable_persistence}"
        )

    def execute_trade(
        self,
        symbol: str,
        trade_type: TradeType,
        quantity: int,
        price: Decimal,
        validate_risk: bool = True,
        available_capital: Optional[Decimal] = None,
        notes: str = "",
    ) -> Optional[Trade]:
        """
        取引実行（包括的処理）

        Args:
            symbol: 銘柄コード
            trade_type: 取引タイプ
            quantity: 数量
            price: 価格
            validate_risk: リスク検証実行
            available_capital: 利用可能資本
            notes: メモ

        Returns:
            実行された取引（失敗時はNone）
        """
        try:
            # 1. 事前検証
            if self.enable_validation or self.enable_compliance:
                # 仮取引オブジェクトを作成して検証
                temp_trade_id = self.id_generator.generate_trade_id(symbol)
                temp_trade = Trade(
                    id=temp_trade_id,
                    symbol=symbol,
                    trade_type=trade_type,
                    quantity=quantity,
                    price=price,
                    timestamp=datetime.now(),
                    commission=self.risk_calculator.calculate_commission(quantity, price),
                    notes=notes,
                )

                # データ検証
                if self.enable_validation:
                    validation_result = self.trade_validator.validate_single_trade(temp_trade)
                    if not validation_result["is_valid"]:
                        logger.error(f"取引検証失敗: {symbol} - {validation_result['errors']}")
                        return None

                # コンプライアンスチェック
                if self.enable_compliance:
                    existing_trades = self.trade_executor.get_trade_history()
                    portfolio_value = self._calculate_portfolio_value()
                    
                    compliance_result = self.compliance_checker.check_trade_compliance(
                        temp_trade, existing_trades, portfolio_value
                    )
                    
                    if not compliance_result["overall_compliant"]:
                        logger.error(f"コンプライアンス違反: {symbol} - {compliance_result['violations']}")
                        return None

            # 2. 取引実行
            if trade_type == TradeType.BUY:
                trade = self.trade_executor.buy_stock(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    validate_risk=validate_risk,
                    available_capital=available_capital,
                    notes=notes,
                )
            elif trade_type == TradeType.SELL:
                trade = self.trade_executor.sell_stock(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    notes=notes,
                )
            else:
                logger.error(f"未サポートの取引タイプ: {trade_type}")
                return None

            if not trade:
                logger.error(f"取引実行失敗: {symbol} {trade_type.value}")
                return None

            # 3. 永続化
            if self.enable_persistence and self.db_manager:
                success = self.db_manager.save_trade_to_db(trade)
                if not success:
                    logger.warning(f"取引DB保存失敗: {trade.id}")

            # 4. ビジネスイベントログ
            log_business_event(
                f"取引完了: {symbol} {trade_type.value} {quantity}株 @{price}円",
                {
                    "trade_id": trade.id,
                    "symbol": symbol,
                    "type": trade_type.value,
                    "amount": price * Decimal(quantity),
                }
            )

            logger.info(f"取引実行成功: {trade.id}")
            return trade

        except Exception as e:
            logger.error(f"取引実行エラー: {symbol} {trade_type.value} - {e}")
            return None

    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        ポートフォリオ現状取得

        Returns:
            ポートフォリオ情報辞書
        """
        try:
            # 基本情報
            positions = self.position_manager.get_all_positions()
            portfolio_summary = self.position_manager.get_portfolio_summary()
            trade_history = self.trade_executor.get_trade_history(limit=100)

            # 分析情報
            performance = self.portfolio_analyzer.calculate_portfolio_performance(
                trade_history, positions
            )

            efficiency = self.portfolio_analyzer.analyze_trade_efficiency(trade_history)
            risk_metrics = self.portfolio_analyzer.calculate_risk_metrics(trade_history, positions)

            status = {
                "timestamp": datetime.now().isoformat(),
                "positions": {symbol: pos.to_dict() for symbol, pos in positions.items()},
                "portfolio_summary": portfolio_summary,
                "performance_metrics": performance,
                "efficiency_metrics": efficiency,
                "risk_metrics": risk_metrics,
                "recent_trades": [trade.to_dict() for trade in trade_history[:10]],
            }

            return status

        except Exception as e:
            logger.error(f"ポートフォリオ状況取得エラー: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def generate_comprehensive_report(
        self,
        report_type: str = "portfolio",
        period_days: int = 30,
        formats: List[str] = None,
    ) -> Dict[str, str]:
        """
        包括的レポート生成

        Args:
            report_type: レポートタイプ（portfolio, tax, performance）
            period_days: 分析期間（日数）
            formats: 出力形式リスト

        Returns:
            生成されたレポートファイルパス辞書
        """
        try:
            if formats is None:
                formats = ["json", "html"]

            generated_files = {}

            if report_type == "portfolio":
                # ポートフォリオレポート
                positions = self.position_manager.get_all_positions()
                trades = self.trade_executor.get_trade_history()
                
                report_data = self.portfolio_analyzer.generate_performance_report(
                    trades, positions, period_days
                )
                
                file_paths = self.report_exporter.export_multiple_formats(
                    report_data, "portfolio", f"portfolio_{period_days}days", formats
                )
                
                generated_files.update({fmt: path for fmt, path in zip(formats, file_paths)})

            elif report_type == "tax":
                # 税務レポート
                trades = self.trade_executor.get_trade_history()
                current_year = datetime.now().year
                
                tax_data = self.tax_calculator.generate_tax_report_data(trades, current_year)
                
                file_paths = self.report_exporter.export_multiple_formats(
                    tax_data, "tax", f"tax_report_{current_year}", formats
                )
                
                generated_files.update({fmt: path for fmt, path in zip(formats, file_paths)})

            elif report_type == "performance":
                # パフォーマンスレポート
                status = self.get_portfolio_status()
                
                file_paths = self.report_exporter.export_multiple_formats(
                    status, "portfolio", f"performance_{period_days}days", formats
                )
                
                generated_files.update({fmt: path for fmt, path in zip(formats, file_paths)})

            else:
                logger.error(f"未サポートのレポートタイプ: {report_type}")
                return {}

            logger.info(f"レポート生成完了: {report_type} - {len(generated_files)}ファイル")
            return generated_files

        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return {"error": str(e)}

    def validate_trade_batch(self, trades: List[Trade]) -> Dict[str, Any]:
        """
        取引データ一括検証

        Args:
            trades: 検証対象取引リスト

        Returns:
            検証結果
        """
        if not self.enable_validation or not self.trade_validator:
            logger.warning("検証機能が無効化されています")
            return {"validation_disabled": True}

        try:
            # データ検証
            validation_result = self.trade_validator.validate_trade_batch(trades)
            
            # シーケンス検証
            sequence_result = self.trade_validator.validate_trade_sequence(trades)
            
            # 統合結果
            combined_result = {
                "batch_validation": validation_result,
                "sequence_validation": sequence_result,
                "overall_valid": (
                    validation_result.get("valid_trades", 0) == len(trades) and
                    sequence_result.get("sequence_valid", False)
                ),
                "validation_timestamp": datetime.now().isoformat(),
            }

            return combined_result

        except Exception as e:
            logger.error(f"一括検証エラー: {e}")
            return {"error": str(e)}

    def sync_data_from_db(self) -> Dict[str, int]:
        """
        データベースからデータ同期

        Returns:
            同期結果
        """
        if not self.enable_persistence or not self.db_manager:
            return {"sync_disabled": True}

        try:
            # DBから取引データ読み込み
            db_trades = self.db_manager.load_trades_from_db()
            
            if not db_trades:
                logger.info("同期対象データなし")
                return {"synced_trades": 0}

            # 既存データとマージ
            existing_trade_ids = set(trade.id for trade in self.trade_executor.trades)
            new_trades = [trade for trade in db_trades if trade.id not in existing_trade_ids]

            # 新規取引を追加
            for trade in new_trades:
                self.trade_executor.trades.append(trade)
                self.position_manager.update_position_from_trade(trade)

            result = {
                "total_db_trades": len(db_trades),
                "existing_trades": len(existing_trade_ids),
                "synced_trades": len(new_trades),
                "sync_timestamp": datetime.now().isoformat(),
            }

            logger.info(f"DB同期完了: {len(new_trades)}件の新規取引を同期")
            return result

        except Exception as e:
            logger.error(f"DB同期エラー: {e}")
            return {"error": str(e), "synced_trades": 0}

    def cleanup_data(self, clean_duplicates: bool = True, validate_data: bool = True) -> Dict[str, Any]:
        """
        データクリーンアップ

        Args:
            clean_duplicates: 重複除去実行
            validate_data: データ検証実行

        Returns:
            クリーンアップ結果
        """
        if not self.enable_persistence or not self.data_cleaner:
            return {"cleanup_disabled": True}

        try:
            trades = self.trade_executor.get_trade_history()
            
            if not trades:
                return {"message": "クリーンアップ対象データなし"}

            # データクリーニング実行
            cleaning_result = self.data_cleaner.clean_trades_data(trades)
            
            # クリーン済みデータで更新
            if cleaning_result.get("cleaned_trades"):
                cleaned_trades = cleaning_result["cleaned_trades"]
                
                # 取引履歴を更新
                self.trade_executor.trades = cleaned_trades
                self.trade_executor.trade_history.clear()
                
                # ポジションを再計算
                self.position_manager.clear_all_positions()
                for trade in cleaned_trades:
                    self.position_manager.update_position_from_trade(trade)

            result = {
                "cleanup_summary": cleaning_result,
                "cleanup_timestamp": datetime.now().isoformat(),
            }

            logger.info(f"データクリーンアップ完了: {cleaning_result.get('final_count', 0)}件残存")
            return result

        except Exception as e:
            logger.error(f"データクリーンアップエラー: {e}")
            return {"error": str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """
        システム状態取得

        Returns:
            システム状態情報
        """
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "configuration": {
                    "validation_enabled": self.enable_validation,
                    "compliance_enabled": self.enable_compliance,
                    "persistence_enabled": self.enable_persistence,
                },
                "statistics": {
                    "total_trades": len(self.trade_executor.trades),
                    "active_positions": len(self.position_manager.positions),
                    "trade_statistics": self.trade_executor.get_trade_statistics(),
                },
                "component_status": {
                    "risk_calculator": "active",
                    "position_manager": "active",
                    "trade_executor": "active",
                    "portfolio_analyzer": "active",
                    "tax_calculator": "active",
                    "report_exporter": "active",
                    "db_manager": "active" if self.db_manager else "disabled",
                    "trade_validator": "active" if self.trade_validator else "disabled",
                    "compliance_checker": "active" if self.compliance_checker else "disabled",
                },
            }

            # データベース統計（有効な場合）
            if self.enable_persistence and self.db_manager:
                status["database_statistics"] = self.db_manager.get_database_statistics()

            # ID生成統計
            status["id_generator_statistics"] = self.id_generator.get_id_statistics()

            return status

        except Exception as e:
            logger.error(f"システム状態取得エラー: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_portfolio_value(self) -> Decimal:
        """ポートフォリオ価値計算（内部用）"""
        try:
            portfolio_summary = self.position_manager.get_portfolio_summary()
            return portfolio_summary.get("total_market_value", Decimal("0"))
        except Exception as e:
            logger.error(f"ポートフォリオ価値計算エラー: {e}")
            return Decimal("0")

    def shutdown(self) -> None:
        """システムシャットダウン"""
        try:
            # バッチプロセッサーのシャットダウン
            if self.batch_processor:
                self.batch_processor.shutdown()

            # 最終データ保存（永続化有効時）
            if self.enable_persistence and self.db_manager:
                trades = self.trade_executor.get_trade_history()
                if trades:
                    sync_result = self.db_manager.sync_trades_to_db(trades)
                    logger.info(f"シャットダウン時DB同期: {sync_result}")

            logger.info("TradeManagerシャットダウン完了")

        except Exception as e:
            logger.error(f"シャットダウンエラー: {e}")

    def __enter__(self):
        """コンテキストマネージャー開始"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了"""
        self.shutdown()