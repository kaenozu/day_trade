import csv
import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..utils.logging_config import get_context_logger
from .trade_models import Position, RealizedPnL, Trade
from .trade_utils import mask_sensitive_info, validate_file_path


class TradeReporter:
    def __init__(self, trade_manager):
        self.tm = trade_manager
        self.logger = get_context_logger(__name__)

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Trade]:
        """取引履歴を取得"""
        if symbol:
            return [trade for trade in self.tm.trades if trade.symbol == symbol]
        return self.tm.trades.copy()

    def get_realized_pnl_history(
        self, symbol: Optional[str] = None
    ) -> List[RealizedPnL]:
        """実現損益履歴を取得"""
        if symbol:
            return [pnl for pnl in self.tm.realized_pnl if pnl.symbol == symbol]
        return self.tm.realized_pnl.copy()

    def get_portfolio_summary(self) -> Dict:
        """ポートフォリオサマリーを取得"""
        total_cost = sum(pos.total_cost for pos in self.tm.positions.values())
        total_market_value = sum(pos.market_value for pos in self.tm.positions.values())
        total_unrealized_pnl = total_market_value - total_cost
        total_realized_pnl = sum(pnl.pnl for pnl in self.tm.realized_pnl)

        return {
            "total_positions": len(self.tm.positions),
            "total_cost": str(total_cost),
            "total_market_value": str(total_market_value),
            "total_unrealized_pnl": str(total_unrealized_pnl),
            "total_realized_pnl": str(total_realized_pnl),
            "total_pnl": str(total_unrealized_pnl + total_realized_pnl),
            "total_trades": len(self.tm.trades),
            "winning_trades": len([pnl for pnl in self.tm.realized_pnl if pnl.pnl > 0]),
            "losing_trades": len([pnl for pnl in self.tm.realized_pnl if pnl.pnl < 0]),
            "win_rate": (
                f"{(len([pnl for pnl in self.tm.realized_pnl if pnl.pnl > 0]) / max(len(self.tm.realized_pnl), 1) * 100):.1f}%"
                if self.tm.realized_pnl
                else "0.0%"
            ),
        }

    def export_to_csv(self, filepath: str, data_type: str = "trades") -> None:
        """
        CSVファイルにエクスポート（パス検証強化版）

        Args:
            filepath: 出力ファイルパス
            data_type: データタイプ ('trades', 'positions', 'realized_pnl')
        """
        try:
            safe_path = validate_file_path(filepath, "CSV出力")

            if data_type == "trades":
                data = [trade.to_dict() for trade in self.tm.trades]
                fieldnames = [
                    "id",
                    "symbol",
                    "trade_type",
                    "quantity",
                    "price",
                    "timestamp",
                    "commission",
                    "status",
                    "notes",
                ]

            elif data_type == "positions":
                data = [pos.to_dict() for pos in self.tm.positions.values()]
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

            elif data_type == "realized_pnl":
                data = [pnl.to_dict() for pnl in self.tm.realized_pnl]
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

            else:
                raise ValueError(f"Invalid data_type: {data_type}")

            with open(safe_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

            self.logger.info(
                f"CSV出力完了: {mask_sensitive_info(str(safe_path))} ({len(data)}件)"
            )

        except Exception as e:
            self.logger.error(
                f"データのエクスポート中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {mask_sensitive_info(str(e))}"
            )
            raise

    def save_to_json(self, filepath: str) -> None:
        """JSON形式で保存（パス検証強化版）"""
        try:
            safe_path = validate_file_path(filepath, "JSON保存")

            data = {
                "trades": [trade.to_dict() for trade in self.tm.trades],
                "positions": {
                    symbol: pos.to_dict() for symbol, pos in self.tm.positions.items()
                },
                "realized_pnl": [pnl.to_dict() for pnl in self.tm.realized_pnl],
                "settings": {
                    "commission_rate": str(self.tm.commission_rate),
                    "tax_rate": str(self.tm.tax_rate),
                    "trade_counter": self.tm._trade_counter,
                },
            }

            with open(safe_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"JSON保存完了: {mask_sensitive_info(str(safe_path))}")

        except Exception as e:
            self.logger.error(
                f"データの保存中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {mask_sensitive_info(str(e))}"
            )
            raise

    def load_from_json(self, filepath: str) -> None:
        """JSON形式から読み込み（パス検証強化版）"""
        try:
            safe_path = validate_file_path(filepath, "JSON読み込み")

            with open(safe_path, encoding="utf-8") as f:
                data = json.load(f)

            self.tm.trades = [
                Trade.from_dict(trade_data) for trade_data in data.get("trades", [])
            ]

            self.tm.positions = {}
            for symbol, pos_data in data.get("positions", {}).items():
                self.tm.positions[symbol] = Position(
                    symbol=pos_data["symbol"],
                    quantity=pos_data["quantity"],
                    average_price=Decimal(pos_data["average_price"]),
                    total_cost=Decimal(pos_data["total_cost"]),
                    current_price=Decimal(pos_data["current_price"]),
                )

            self.tm.realized_pnl = []
            for pnl_data in data.get("realized_pnl", []):
                self.tm.realized_pnl.append(
                    RealizedPnL(
                        symbol=pnl_data["symbol"],
                        quantity=pnl_data["quantity"],
                        buy_price=Decimal(pnl_data["buy_price"]),
                        sell_price=Decimal(pnl_data["sell_price"]),
                        buy_commission=Decimal(pnl_data["buy_commission"]),
                        sell_commission=Decimal(pnl_data["sell_commission"]),
                        pnl=Decimal(pnl_data["pnl"]),
                        pnl_percent=Decimal(pnl_data["pnl_percent"]),
                        buy_date=datetime.fromisoformat(pnl_data["buy_date"]),
                        sell_date=datetime.fromisoformat(pnl_data["sell_date"]),
                    )
                )

            settings = data.get("settings", {})
            if "commission_rate" in settings:
                self.tm.commission_rate = Decimal(settings["commission_rate"])
            if "tax_rate" in settings:
                self.tm.tax_rate = Decimal(settings["tax_rate"])
            if "trade_counter" in settings:
                self.tm._trade_counter = settings["trade_counter"]

            self.logger.info(f"JSON読み込み完了: {mask_sensitive_info(str(safe_path))}")

        except Exception as e:
            self.logger.error(
                f"データの読み込み中にエラーが発生しました。ファイル形式が正しいか、破損していないか確認してください。詳細: {mask_sensitive_info(str(e))}"
            )
            raise

    def calculate_tax_implications(
        self, year: int, accounting_method: str = "FIFO"
    ) -> Dict:
        """
        税務計算（会計原則対応版）

        Args:
            year: 税務年度
            accounting_method: 会計手法 ("FIFO", "LIFO", "AVERAGE")
        """
        try:
            year_start = datetime(year, 1, 1)
            year_end = datetime(year, 12, 31, 23, 59, 59)

            year_pnl = [
                pnl
                for pnl in self.tm.realized_pnl
                if year_start <= pnl.sell_date <= year_end
            ]

            if not year_pnl:
                return {
                    "year": year,
                    "accounting_method": accounting_method,
                    "total_trades": 0,
                    "total_gain": "0.00",
                    "total_loss": "0.00",
                    "net_gain": "0.00",
                    "tax_due": "0.00",
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "average_gain_per_winning_trade": "0.00",
                    "average_loss_per_losing_trade": "0.00",
                    "win_rate": "0.00%",
                }

            gains = [pnl.pnl for pnl in year_pnl if pnl.pnl > 0]
            losses = [pnl.pnl for pnl in year_pnl if pnl.pnl < 0]

            total_gain = sum(gains) if gains else Decimal("0")
            total_loss = sum(abs(loss) for loss in losses) if losses else Decimal("0")
            net_gain = total_gain - total_loss

            tax_due = net_gain * self.tm.tax_rate if net_gain > 0 else Decimal("0")

            winning_trades_count = len(gains)
            losing_trades_count = len(losses)
            total_trades = len(year_pnl)

            avg_gain = (
                total_gain / winning_trades_count
                if winning_trades_count > 0
                else Decimal("0")
            )
            avg_loss = (
                total_loss / losing_trades_count
                if losing_trades_count > 0
                else Decimal("0")
            )
            win_rate = (
                (winning_trades_count / total_trades * 100)
                if total_trades > 0
                else Decimal("0")
            )

            return {
                "year": year,
                "accounting_method": accounting_method,
                "total_trades": total_trades,
                "total_gain": str(total_gain.quantize(Decimal("0.01"))),
                "total_loss": str(total_loss.quantize(Decimal("0.01"))),
                "net_gain": str(net_gain.quantize(Decimal("0.01"))),
                "tax_due": str(tax_due.quantize(Decimal("0.01"))),
                "winning_trades": winning_trades_count,
                "losing_trades": losing_trades_count,
                "average_gain_per_winning_trade": str(
                    avg_gain.quantize(Decimal("0.01"))
                ),
                "average_loss_per_losing_trade": str(
                    avg_loss.quantize(Decimal("0.01"))
                ),
                "win_rate": f"{win_rate.quantize(Decimal('0.01'))}%",
            }

        except Exception as e:
            self.logger.error(
                f"税務計算中に予期せぬエラーが発生しました。入力データまたは計算ロジックを確認してください。詳細: {mask_sensitive_info(str(e))}"
            )
            raise
