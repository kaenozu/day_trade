"""
分析・シミュレーション記録管理機能
シミュレーション売買履歴を記録し、仮想損益計算を行う
※実際の取引は行わず、分析・学習用のみ
データベース永続化対応版
"""

import csv
import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..models.enums import TradeType
from ..utils.logging_config import get_context_logger
from .order_manager import OrderManager
from .portfolio_manager import PortfolioManager
from .trade_models import Position, RealizedPnL, Trade
from .trade_reporter import TradeReporter


class TradeManager:
    """
    取引記録管理クラス（会計原則対応版）

    会計原則:
    - FIFO (First In, First Out): 先入れ先出し法を採用
    - 実現損益計算では最古の買い取引から順次売却するものとして計算
    - 手数料は取引毎に個別に管理、実現损益に反映
    - 税金は利益が出た場合のみ計算
    """

    def __init__(
        self,
        commission_rate: Decimal = Decimal("0.001"),
        tax_rate: Decimal = Decimal("0.2"),
        load_from_db: bool = False,
    ):
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: List[RealizedPnL] = []
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self._trade_counter = 0

        self.logger = get_context_logger(__name__)
        self.order_manager = OrderManager(self)
        self.portfolio_manager = PortfolioManager(self)
        self.trade_reporter = TradeReporter(self)

        if load_from_db:
            self.portfolio_manager._load_trades_from_db()

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Trade]:
        return self.trade_reporter.get_trade_history(symbol)

    def get_realized_pnl_history(
        self, symbol: Optional[str] = None
    ) -> List[RealizedPnL]:
        return self.trade_reporter.get_realized_pnl_history(symbol)

    def get_portfolio_summary(self) -> Dict:
        return self.trade_reporter.get_portfolio_summary()

    def export_to_csv(self, filepath: str, data_type: str = "trades") -> None:
        self.trade_reporter.export_to_csv(filepath, data_type)

    def save_to_json(self, filepath: str) -> None:
        self.trade_reporter.save_to_json(filepath)

    def load_from_json(self, filepath: str) -> None:
        self.trade_reporter.load_from_json(filepath)

    def calculate_tax_implications(
        self, year: int, accounting_method: str = "FIFO"
    ) -> Dict:
        return self.trade_reporter.calculate_tax_implications(year, accounting_method)

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.portfolio_manager.get_position(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        return self.portfolio_manager.get_all_positions()

    def update_current_prices(self, prices: Dict[str, Decimal]) -> None:
        self.portfolio_manager.update_current_prices(prices)

    def add_trade(
        self,
        symbol: str,
        trade_type: TradeType,
        quantity: int,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        commission: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> str:
        return self.order_manager.add_trade(
            symbol,
            trade_type,
            quantity,
            price,
            timestamp,
            commission,
            notes,
            persist_to_db,
        )

    def add_trades_batch(
        self, trades_data: List[Dict], persist_to_db: bool = True
    ) -> List[str]:
        return self.order_manager.add_trades_batch(trades_data, persist_to_db)

    def buy_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, Any]:
        return self.order_manager.buy_stock(
            symbol,
            quantity,
            price,
            current_market_price,
            notes,
            persist_to_db,
        )

    def sell_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, Any]:
        return self.order_manager.sell_stock(
            symbol,
            quantity,
            price,
            current_market_price,
            notes,
            persist_to_db,
        )

    def execute_trade_order(
        self, trade_order: Dict[str, Any], persist_to_db: bool = True
    ) -> Dict[str, Any]:
        return self.order_manager.execute_trade_order(trade_order, persist_to_db)

    

    

    

    

    

    

    def _load_trades_from_db(self) -> None:
        self.portfolio_manager._load_trades_from_db()

    
