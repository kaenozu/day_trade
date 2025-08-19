"""
取引管理コア機能

trade_manager.py からのリファクタリング抽出
基本機能とデータアクセス、初期化、設定管理、データ永続化
"""

import json
import csv
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from ..models.trade_models import Trade, Position, RealizedPnL, TradeStatus
from ..models.trade_utils import validate_file_path, mask_sensitive_info
from ...models.database import db_manager
from ...models.enums import TradeType
from ...models.stock import Trade as DBTrade
from ...utils.logging_config import (
    get_context_logger,
    log_business_event,
    log_error_with_context,
)


class TradeManagerCore:
    """
    取引管理コア機能
    
    基本的なデータ管理、設定、永続化、データアクセス機能を提供
    """

    def __init__(
        self,
        commission_rate: Decimal = Decimal("0.001"),
        tax_rate: Decimal = Decimal("0.2"),
        load_from_db: bool = False,
    ):
        """
        初期化

        Args:
            commission_rate: 手数料率（デフォルト0.1%）
            tax_rate: 税率（デフォルト20%）
            load_from_db: データベースから取引履歴を読み込むかどうか
        """
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: List[RealizedPnL] = []
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self._trade_counter = 0

        # ロガーを初期化
        self.logger = get_context_logger(__name__)

        if load_from_db:
            self._load_trades_from_db()

    def _generate_trade_id(self) -> str:
        """取引IDを生成"""
        self._trade_counter += 1
        return f"T{datetime.now().strftime('%Y%m%d')}{self._trade_counter:04d}"

    def _calculate_commission(self, price: Decimal, quantity: int) -> Decimal:
        """手数料を計算"""
        total_value = price * Decimal(quantity)
        commission = total_value * self.commission_rate
        # 最低100円の手数料
        return max(commission, Decimal("100"))

    def _load_trades_from_db(self) -> None:
        """データベースから取引履歴を読み込み（トランザクション保護版）"""
        load_logger = self.logger
        load_logger.info("データベースから取引履歴読み込み開始")

        try:
            # トランザクション内で一括処理
            with db_manager.transaction_scope() as session:
                # データベースから全取引を取得
                db_trades = (
                    session.query(DBTrade).order_by(DBTrade.trade_datetime).all()
                )

                load_logger.info("DB取引データ取得", extra={"count": len(db_trades)})

                # メモリ内データ構造を一旦クリア（原子性保証）
                trades_backup = self.trades.copy()
                positions_backup = self.positions.copy()
                realized_pnl_backup = self.realized_pnl.copy()
                counter_backup = self._trade_counter

                try:
                    # メモリ内データクリア
                    self.trades.clear()
                    self.positions.clear()
                    self.realized_pnl.clear()
                    self._trade_counter = 0

                    for db_trade in db_trades:
                        # セッションから切り離す前に必要な属性を読み込み
                        trade_id = db_trade.id
                        stock_code = db_trade.stock_code
                        trade_type_str = db_trade.trade_type
                        quantity = db_trade.quantity
                        price = db_trade.price
                        trade_datetime = db_trade.trade_datetime
                        commission = db_trade.commission or Decimal("0")
                        memo = db_trade.memo or ""

                        # メモリ内形式に変換
                        if isinstance(trade_type_str, TradeType):
                            trade_type = trade_type_str
                        else:
                            trade_type_str_upper = str(trade_type_str).upper()
                            trade_type = (
                                TradeType.BUY
                                if trade_type_str_upper in ["BUY", "buy"]
                                else TradeType.SELL
                            )

                        memory_trade = Trade(
                            id=f"DB_{trade_id}",  # DBから読み込んだことを示すプレフィックス
                            symbol=stock_code,
                            trade_type=trade_type,
                            quantity=quantity,
                            price=Decimal(str(price)),
                            timestamp=trade_datetime,
                            commission=Decimal(str(commission)),
                            status=TradeStatus.EXECUTED,
                            notes=memo,
                        )

                        self.trades.append(memory_trade)
                        # ポジション更新は ExecutionManager で実装

                    # 取引カウンターを最大値+1に設定
                    if db_trades:
                        max_id = max(db_trade.id for db_trade in db_trades)
                        self._trade_counter = max_id + 1

                    load_logger.info(
                        "データベース読み込み完了",
                        extra={"loaded_trades": len(db_trades)},
                        trade_counter=self._trade_counter,
                    )

                except Exception as restore_error:
                    # メモリ内データの復元
                    self.trades = trades_backup
                    self.positions = positions_backup
                    self.realized_pnl = realized_pnl_backup
                    self._trade_counter = counter_backup
                    load_logger.error(
                        "読み込み処理失敗、メモリ内データを復元",
                        extra={"error": str(restore_error)},
                    )
                    raise restore_error

        except Exception as e:
            log_error_with_context(e, {"operation": "load_trades_from_db"})
            load_logger.error("データベース読み込み失敗", extra={"error": str(e)})
            raise

    def clear_all_data(self, persist_to_db: bool = True) -> None:
        """
        すべての取引データを削除（トランザクション保護）

        Args:
            persist_to_db: データベースにも反映するかどうか
        """
        clear_logger = self.logger
        clear_logger.info(
            "取引データ全削除開始", extra={"persist_to_db": persist_to_db}
        )

        # メモリ内データのバックアップ
        trades_backup = self.trades.copy()
        positions_backup = self.positions.copy()
        realized_pnl_backup = self.realized_pnl.copy()
        counter_backup = self._trade_counter

        try:
            if persist_to_db:
                # データベースからも削除
                with db_manager.transaction_scope() as session:
                    # 全ての取引データを削除
                    deleted_count = session.query(DBTrade).delete()
                    session.commit()

                    clear_logger.info(
                        "データベース取引データ削除完了",
                        extra={"deleted_count": deleted_count},
                    )

            # メモリ内データクリア
            self.trades.clear()
            self.positions.clear()
            self.realized_pnl.clear()
            self._trade_counter = 0

            log_business_event(
                "all_data_cleared",
                persist_to_db=persist_to_db,
                cleared_trades=len(trades_backup),
                cleared_positions=len(positions_backup),
            )

            clear_logger.info("取引データ全削除完了")

        except Exception as e:
            # エラー時はバックアップからデータを復元
            self.trades = trades_backup
            self.positions = positions_backup
            self.realized_pnl = realized_pnl_backup
            self._trade_counter = counter_backup

            log_error_with_context(
                e,
                {
                    "operation": "clear_all_data",
                    "persist_to_db": persist_to_db,
                },
            )
            clear_logger.error("取引データ削除失敗", extra={"error": str(e)})
            raise

    def get_position(self, symbol: str) -> Optional[Position]:
        """ポジション情報を取得"""
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """全ポジション情報を取得"""
        return self.positions.copy()

    def get_all_trades(self) -> List[Trade]:
        """全取引履歴を取得"""
        return self.trades.copy()

    def get_realized_pnl(self) -> List[RealizedPnL]:
        """実現損益履歴を取得"""
        return self.realized_pnl.copy()

    def get_summary_stats(self) -> Dict[str, any]:
        """サマリー統計を取得"""
        total_trades = len(self.trades)
        total_positions = len(self.positions)
        total_realized_pnl = sum(pnl.pnl for pnl in self.realized_pnl)
        total_commission = sum(trade.commission for trade in self.trades)
        
        # 未実現損益の合計
        total_unrealized_pnl = sum(
            position.unrealized_pnl for position in self.positions.values()
        )

        return {
            "total_trades": total_trades,
            "total_positions": total_positions,
            "total_realized_pnl": str(total_realized_pnl),
            "total_unrealized_pnl": str(total_unrealized_pnl),
            "total_commission": str(total_commission),
            "total_pnl": str(total_realized_pnl + total_unrealized_pnl),
        }

    def export_to_csv(self, filepath: str, data_type: str = "trades") -> None:
        """
        CSVファイルにエクスポート（パス検証強化版）

        Args:
            filepath: 出力ファイルパス
            data_type: データ種別 ("trades", "positions", "realized_pnl")
        """
        try:
            # ファイルパス検証
            safe_path = validate_file_path(filepath, "CSV エクスポート")

            if data_type == "trades":
                self._export_trades_to_csv(safe_path)
            elif data_type == "positions":
                self._export_positions_to_csv(safe_path)
            elif data_type == "realized_pnl":
                self._export_realized_pnl_to_csv(safe_path)
            else:
                raise ValueError(f"不正なデータ種別: {data_type}")

            log_business_event(
                "csv_export_completed",
                filepath=mask_sensitive_info(str(safe_path)),
                data_type=data_type,
                record_count=len(getattr(self, data_type)),
            )

            self.logger.info(
                "CSV エクスポート完了",
                extra={
                    "filepath": mask_sensitive_info(str(safe_path)),
                    "data_type": data_type,
                },
            )

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "export_to_csv",
                    "filepath": mask_sensitive_info(filepath),
                    "data_type": data_type,
                },
            )
            raise

    def _export_trades_to_csv(self, filepath: Path) -> None:
        """取引履歴をCSVエクスポート"""
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "id",
                "symbol",
                "trade_type",
                "quantity",
                "price",
                "commission",
                "timestamp",
                "status",
                "notes",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for trade in self.trades:
                writer.writerow(trade.to_dict())

    def _export_positions_to_csv(self, filepath: Path) -> None:
        """ポジション情報をCSVエクスポート"""
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "symbol",
                "quantity",
                "average_price",
                "total_cost",
                "current_price",
                "market_value",
                "unrealized_pnl",
                "unrealized_pnl_percent",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for position in self.positions.values():
                writer.writerow(position.to_dict())

    def _export_realized_pnl_to_csv(self, filepath: Path) -> None:
        """実現損益をCSVエクスポート"""
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "symbol",
                "quantity",
                "buy_price",
                "sell_price",
                "buy_commission",
                "sell_commission",
                "pnl",
                "pnl_percent",
                "buy_date",
                "sell_date",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for pnl in self.realized_pnl:
                writer.writerow(pnl.to_dict())

    def save_to_json(self, filepath: str) -> None:
        """JSON形式で保存（パス検証強化版）"""
        try:
            # ファイルパス検証
            safe_path = validate_file_path(filepath, "JSON 保存")

            data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "commission_rate": str(self.commission_rate),
                    "tax_rate": str(self.tax_rate),
                    "trade_counter": self._trade_counter,
                },
                "trades": [trade.to_dict() for trade in self.trades],
                "positions": {
                    symbol: position.to_dict()
                    for symbol, position in self.positions.items()
                },
                "realized_pnl": [pnl.to_dict() for pnl in self.realized_pnl],
            }

            with open(safe_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            log_business_event(
                "json_save_completed",
                filepath=mask_sensitive_info(str(safe_path)),
                trades_count=len(self.trades),
                positions_count=len(self.positions),
            )

            self.logger.info(
                "JSON 保存完了", extra={"filepath": mask_sensitive_info(str(safe_path))}
            )

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "save_to_json",
                    "filepath": mask_sensitive_info(filepath),
                },
            )
            raise

    def load_from_json(self, filepath: str) -> None:
        """JSON形式から読み込み（パス検証強化版）"""
        try:
            # ファイルパス検証
            safe_path = validate_file_path(filepath, "JSON 読み込み")

            if not safe_path.exists():
                raise FileNotFoundError(f"ファイルが見つかりません: {safe_path}")

            # メモリ内データのバックアップ
            trades_backup = self.trades.copy()
            positions_backup = self.positions.copy()
            realized_pnl_backup = self.realized_pnl.copy()
            settings_backup = {
                "commission_rate": self.commission_rate,
                "tax_rate": self.tax_rate,
                "trade_counter": self._trade_counter,
            }

            try:
                with open(safe_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # メタデータの復元
                if "metadata" in data:
                    metadata = data["metadata"]
                    self.commission_rate = Decimal(
                        metadata.get("commission_rate", "0.001")
                    )
                    self.tax_rate = Decimal(metadata.get("tax_rate", "0.2"))
                    self._trade_counter = metadata.get("trade_counter", 0)

                # データクリア
                self.trades.clear()
                self.positions.clear()
                self.realized_pnl.clear()

                # 取引履歴の復元
                if "trades" in data:
                    for trade_data in data["trades"]:
                        trade = Trade.from_dict(trade_data)
                        self.trades.append(trade)

                # ポジション情報の復元（簡易版）
                # 実際の実装では ExecutionManager でポジション再構築が必要

                log_business_event(
                    "json_load_completed",
                    filepath=mask_sensitive_info(str(safe_path)),
                    loaded_trades=len(self.trades),
                )

                self.logger.info(
                    "JSON 読み込み完了",
                    extra={
                        "filepath": mask_sensitive_info(str(safe_path)),
                        "loaded_trades": len(self.trades),
                    },
                )

            except Exception as e:
                # エラー時はバックアップからデータを復元
                self.trades = trades_backup
                self.positions = positions_backup
                self.realized_pnl = realized_pnl_backup
                self.commission_rate = settings_backup["commission_rate"]
                self.tax_rate = settings_backup["tax_rate"]
                self._trade_counter = settings_backup["trade_counter"]
                raise e

        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "load_from_json",
                    "filepath": mask_sensitive_info(filepath),
                },
            )
            raise