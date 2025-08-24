from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Tuple
import pandas as pd

from day_trade.analysis.backtest.structures import Position, Trade, BacktestConfig
from day_trade.core.trade_manager import TradeType
from day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="portfolio_manager")


class Portfolio:
    """ポートフォリオ管理クラス"""

    def __init__(self, initial_capital: Decimal):
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[Tuple[datetime, Decimal]] = []

    def update_positions_value(self, daily_prices: Dict[str, Decimal]):
        """ポジション評価額の更新"""
        for symbol in self.positions:
            if symbol in daily_prices:
                position = self.positions[symbol]
                position.current_price = daily_prices[symbol]
                position.market_value = position.current_price * position.quantity
                position.unrealized_pnl = position.market_value - (
                    position.average_price * position.quantity
                )

    def execute_buy_order(
        self, symbol: str, price: Decimal, date: datetime, config: BacktestConfig
    ):
        """買い注文の実行"""
        total_value = self.calculate_total_portfolio_value()
        max_investment = total_value * config.max_position_size

        actual_price = price * (Decimal("1") + config.slippage)

        quantity = int(max_investment / actual_price)

        if quantity <= 0 or self.current_capital < actual_price * quantity:
            return

        commission = actual_price * quantity * config.commission
        total_cost = actual_price * quantity + commission

        if total_cost > self.current_capital:
            return

        if symbol in self.positions:
            old_position = self.positions[symbol]
            new_quantity = old_position.quantity + quantity
            new_average_price = (
                old_position.average_price * old_position.quantity
                + actual_price * quantity
            ) / new_quantity

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=new_quantity,
                average_price=new_average_price,
                current_price=actual_price,
                market_value=actual_price * new_quantity,
                unrealized_pnl=Decimal("0"),
                weight=Decimal("0"),
            )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=actual_price,
                current_price=actual_price,
                market_value=actual_price * quantity,
                unrealized_pnl=Decimal("0"),
                weight=Decimal("0"),
            )

        self.current_capital -= total_cost
        self.trades.append(
            Trade(
                timestamp=date,
                symbol=symbol,
                action=TradeType.BUY,
                quantity=quantity,
                price=actual_price,
                commission=commission,
                total_cost=total_cost,
            )
        )

        logger.debug(f"買い注文実行: {symbol} x{quantity} @ ¥{actual_price}")

    def execute_sell_order(
        self, symbol: str, price: Decimal, date: datetime, config: BacktestConfig
    ):
        """売り注文の実行"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        quantity = position.quantity

        actual_price = price * (Decimal("1") - config.slippage)

        commission = actual_price * quantity * config.commission
        total_proceeds = actual_price * quantity - commission

        self.current_capital += total_proceeds
        self.trades.append(
            Trade(
                timestamp=date,
                symbol=symbol,
                action=TradeType.SELL,
                quantity=quantity,
                price=actual_price,
                commission=commission,
                total_cost=total_proceeds,
            )
        )

        del self.positions[symbol]

        logger.debug(f"売り注文実行: {symbol} x{quantity} @ ¥{actual_price}")

    def calculate_total_portfolio_value(self) -> Decimal:
        """ポートフォリオ総価値の計算"""
        total_value = self.current_capital

        for position in self.positions.values():
            total_value += position.market_value

        return total_value

    def record_portfolio_state(self, date: datetime):
        """ポートフォリオ価値の記録"""
        total_value = self.calculate_total_portfolio_value()
        self.portfolio_values.append((date, total_value))

    def get_portfolio_values(self) -> List[Tuple[datetime, Decimal]]:
        """記録されたポートフォリオ価値を取得"""
        return self.portfolio_values

    def get_trades(self) -> List[Trade]:
        """記録された取引を取得"""
        return self.trades

    def reset_state(self, initial_capital: Decimal):
        """ポートフォリオの状態をリセット"""
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
