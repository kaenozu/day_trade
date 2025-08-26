"""
データ管理・I/O機能
データのエクスポート、保存、読み込みを担当
"""

import csv
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from ...utils.logging_config import get_context_logger
from .models import Position, RealizedPnL, Trade

logger = get_context_logger(__name__)


class DataManagement:
    """データ管理・I/O操作を管理するクラス"""
    
    def __init__(self, trade_manager_ref):
        """
        初期化
        
        Args:
            trade_manager_ref: TradeManagerオブジェクトへの参照
        """
        self.trade_manager = trade_manager_ref
        self.logger = get_context_logger(__name__)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """ポジション情報を取得"""
        return self.trade_manager.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """全ポジション情報を取得"""
        return self.trade_manager.positions.copy()
    
    def update_current_prices(self, prices: Dict[str, Decimal]):
        """現在価格を更新"""
        for symbol, price in prices.items():
            if symbol in self.trade_manager.positions:
                self.trade_manager.positions[symbol].current_price = price
    
    def get_trade_history(self, symbol: Optional[str] = None) -> List[Trade]:
        """取引履歴を取得"""
        if symbol:
            return [trade for trade in self.trade_manager.trades if trade.symbol == symbol]
        return self.trade_manager.trades.copy()
    
    def get_realized_pnl_history(
        self, symbol: Optional[str] = None
    ) -> List[RealizedPnL]:
        """実現損益履歴を取得"""
        if symbol:
            return [pnl for pnl in self.trade_manager.realized_pnl if pnl.symbol == symbol]
        return self.trade_manager.realized_pnl.copy()
    
    def export_to_csv(self, filepath: str, data_type: str = "trades"):
        """
        CSVファイルにエクスポート
        
        Args:
            filepath: 出力ファイルパス
            data_type: データタイプ ('trades', 'positions', 'realized_pnl')
        """
        try:
            if data_type == "trades":
                data = [trade.to_dict() for trade in self.trade_manager.trades]
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
                data = [pos.to_dict() for pos in self.trade_manager.positions.values()]
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
                data = [pnl.to_dict() for pnl in self.trade_manager.realized_pnl]
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
            
            with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            logger.info(f"CSV出力完了: {filepath} ({len(data)}件)")
        
        except Exception as e:
            logger.error(
                f"データのエクスポート中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {e}"
            )
            raise
    
    def save_to_json(self, filepath: str):
        """JSON形式で保存"""
        try:
            data = {
                "trades": [trade.to_dict() for trade in self.trade_manager.trades],
                "positions": {
                    symbol: pos.to_dict() for symbol, pos in self.trade_manager.positions.items()
                },
                "realized_pnl": [pnl.to_dict() for pnl in self.trade_manager.realized_pnl],
                "settings": {
                    "commission_rate": str(self.trade_manager.commission_rate),
                    "tax_rate": str(self.trade_manager.tax_rate),
                    "trade_counter": self.trade_manager._trade_counter,
                },
            }
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON保存完了: {filepath}")
        
        except Exception as e:
            logger.error(
                f"データの保存中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {e}"
            )
            raise
    
    def load_from_json(self, filepath: str):
        """JSON形式から読み込み"""
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            
            # 取引履歴復元
            self.trade_manager.trades = [
                Trade.from_dict(trade_data) for trade_data in data.get("trades", [])
            ]
            
            # ポジション復元
            self.trade_manager.positions = {}
            for symbol, pos_data in data.get("positions", {}).items():
                self.trade_manager.positions[symbol] = Position(
                    symbol=pos_data["symbol"],
                    quantity=pos_data["quantity"],
                    average_price=Decimal(pos_data["average_price"]),
                    total_cost=Decimal(pos_data["total_cost"]),
                    current_price=Decimal(pos_data["current_price"]),
                )
            
            # 実現損益復元
            self.trade_manager.realized_pnl = []
            for pnl_data in data.get("realized_pnl", []):
                self.trade_manager.realized_pnl.append(
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
            
            # 設定復元
            settings = data.get("settings", {})
            if "commission_rate" in settings:
                self.trade_manager.commission_rate = Decimal(settings["commission_rate"])
            if "tax_rate" in settings:
                self.trade_manager.tax_rate = Decimal(settings["tax_rate"])
            if "trade_counter" in settings:
                self.trade_manager._trade_counter = settings["trade_counter"]
            
            logger.info(f"JSON読み込み完了: {filepath}")
        
        except Exception as e:
            logger.error(
                f"データの読み込み中にエラーが発生しました。ファイル形式が正しいか、破損していないか確認してください。詳細: {e}"
            )
            raise