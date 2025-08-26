"""
高度なバックテストエンジン - ポジション管理システム

ポジションの更新、時価評価、リスク計算を管理。
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from day_trade.analysis.events import TradeRecord, Order
from day_trade.utils.logging_config import get_context_logger
from .data_structures import Position

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


class PositionManager:
    """ポジション管理システム"""

    def __init__(self, initial_capital: float):
        """ポジション管理システムの初期化"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.open_trades: Dict[str, TradeRecord] = {}

    def update_position_from_order(self, order: Order) -> float:
        """注文からポジション更新"""
        symbol = order.symbol
        realized_pnl_for_order = 0.0

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        position = self.positions[symbol]

        if order.side == "buy":
            # 買い注文
            total_cost = (
                position.quantity * position.average_price
                + order.quantity * order.filled_price
            )
            total_quantity = position.quantity + order.quantity

            if total_quantity > 0:
                position.average_price = total_cost / total_quantity
                position.quantity = total_quantity

        else:
            # 売り注文
            if position.quantity >= order.quantity:
                # 実現損益計算
                realized_pnl_for_order = (
                    order.filled_price - position.average_price
                ) * order.quantity - order.commission
                position.realized_pnl += realized_pnl_for_order
                position.quantity -= order.quantity

                if position.quantity == 0:
                    position.average_price = 0.0
            else:
                logger.warning(
                    f"ショートポジション発生: {symbol}",
                    section="position_management",
                    current_quantity=position.quantity,
                    sell_quantity=order.quantity,
                )

        # 資本更新
        if order.side == "buy":
            self.current_capital -= (
                order.quantity * order.filled_price + order.commission
            )
        else:
            self.current_capital += (
                order.quantity * order.filled_price - order.commission
            )

        return realized_pnl_for_order

    def update_position_market_value(
        self, symbol: str, current_price: float, current_time: datetime
    ):
        """ポジション時価更新"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (
                current_price - position.average_price
            ) * position.quantity
            position.last_updated = current_time

    def process_trade_record(self, order: Order, fill_time: datetime, 
                           realized_pnl: float):
        """トレードレコードの処理"""
        symbol = order.symbol

        if order.side == "buy":
            # 新しいエントリーまたは既存ポジションへの追加
            if symbol not in self.open_trades or self.open_trades[symbol].is_closed:
                # 新規取引
                trade_id = f"trade_{symbol}_{fill_time.strftime('%Y%m%d%H%M%S%f')}"
                new_trade = TradeRecord(
                    trade_id=trade_id,
                    symbol=symbol,
                    entry_time=fill_time,
                    entry_price=order.filled_price,
                    entry_quantity=order.quantity,
                    entry_commission=order.commission,
                    entry_slippage=order.slippage,
                )
                self.open_trades[symbol] = new_trade
            else:
                # 既存取引への追加
                existing_trade = self.open_trades[symbol]
                existing_trade.entry_quantity += order.quantity
                existing_trade.entry_commission += order.commission
                existing_trade.entry_slippage += order.slippage

        elif order.side == "sell":
            # エグジット処理
            if symbol in self.open_trades and not self.open_trades[symbol].is_closed:
                closed_trade = self.open_trades[symbol]
                closed_trade.exit_time = fill_time
                closed_trade.exit_price = order.filled_price
                closed_trade.exit_quantity = order.quantity
                closed_trade.exit_commission = order.commission
                closed_trade.exit_slippage = order.slippage

                # 実現損益、合計手数料、合計スリッページを更新
                closed_trade.realized_pnl = realized_pnl
                closed_trade.total_commission = (
                    closed_trade.entry_commission + closed_trade.exit_commission
                )
                closed_trade.total_slippage = (
                    closed_trade.entry_slippage + closed_trade.exit_slippage
                )
                closed_trade.is_closed = True

                self.trade_history.append(closed_trade)
                del self.open_trades[symbol]
            else:
                logger.warning(f"未エントリーのポジションの決済イベント: {symbol}")

    def get_portfolio_value(self) -> float:
        """ポートフォリオ総額計算"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.current_capital + positions_value

    def get_position_count(self) -> int:
        """アクティブポジション数を取得"""
        return len([pos for pos in self.positions.values() if pos.quantity != 0])

    def get_total_unrealized_pnl(self) -> float:
        """総未実現損益を取得"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_total_realized_pnl(self) -> float:
        """総実現損益を取得"""
        return sum(pos.realized_pnl for pos in self.positions.values())

    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """シンボル別ポジション取得"""
        return self.positions.get(symbol)

    def close_position(self, symbol: str, current_price: float) -> Optional[Order]:
        """ポジションを強制決済するための注文を生成"""
        if symbol in self.positions and self.positions[symbol].quantity > 0:
            position = self.positions[symbol]
            # 売り注文を生成して返す（実際の注文処理は呼び出し元で実行）
            from day_trade.analysis.events import OrderType, OrderStatus
            order = Order(
                order_id=f"close_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                order_type=OrderType.MARKET,
                side="sell",
                quantity=position.quantity,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING
            )
            return order
        return None

    def calculate_position_heat(self, portfolio_value: float) -> float:
        """ポジションヒート（リスク度）を計算"""
        if portfolio_value <= 0:
            return 0.0
            
        total_heat = sum(
            abs(pos.unrealized_pnl) / portfolio_value
            for pos in self.positions.values()
            if pos.quantity != 0
        )
        return total_heat