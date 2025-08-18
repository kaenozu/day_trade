#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper Trading System - ペーパートレード機能

実際の株価データでの仮想売買システム
リアルデータ + 仮想資金でのトレード練習
"""

import asyncio
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# 既存モジュールのインポート
try:
    from real_data_provider import RealDataProvider
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False

try:
    from risk_manager import PersonalRiskManager
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False

class OrderType(Enum):
    """注文種類"""
    MARKET = "成行"
    LIMIT = "指値"

class OrderStatus(Enum):
    """注文状態"""
    PENDING = "待機中"
    FILLED = "約定済"
    CANCELLED = "取消済"
    REJECTED = "拒否"

class TradeDirection(Enum):
    """売買方向"""
    BUY = "買い"
    SELL = "売り"

@dataclass
class PaperOrder:
    """ペーパートレード注文"""
    id: str
    symbol: str
    name: str
    direction: TradeDirection
    order_type: OrderType
    quantity: int
    price: float  # 指値価格（成行の場合は0）
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_time: Optional[datetime] = None

@dataclass
class PaperPosition:
    """ペーパートレード建玉"""
    symbol: str
    name: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def market_value(self) -> float:
        """時価総額"""
        return self.current_price * self.quantity

    @property
    def unrealized_pnl(self) -> float:
        """含み損益"""
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_pnl_percent(self) -> float:
        """含み損益率"""
        if self.entry_price > 0:
            return (self.current_price - self.entry_price) / self.entry_price * 100
        return 0.0

@dataclass
class PaperTrade:
    """完了したペーパートレード"""
    symbol: str
    name: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    fees: float = 0.0

class PaperTradingEngine:
    """
    ペーパートレードエンジン
    実際の株価での仮想売買システム
    """

    def __init__(self, initial_balance: float = 1000000):
        self.logger = logging.getLogger(__name__)

        # データベース初期化
        self.data_dir = Path("paper_trading_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "paper_trading.db"
        self._init_database()

        # システム状態
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, PaperPosition] = {}
        self.pending_orders: List[PaperOrder] = []
        self.completed_trades: List[PaperTrade] = []

        # 手数料設定（SBI証券基準）
        self.commission_rate = 0.001  # 0.1%
        self.min_commission = 55      # 最低手数料55円

        # リアルデータプロバイダー
        if REAL_DATA_AVAILABLE:
            self.data_provider = RealDataProvider()

        # リスク管理（オプション）
        if RISK_MANAGER_AVAILABLE:
            self.risk_manager = PersonalRiskManager()

        self.logger.info(f"Paper trading initialized with {initial_balance:,.0f}円")

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # 注文履歴
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    direction TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filled_price REAL,
                    filled_time TEXT
                )
            """)

            # 取引履歴
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_percent REAL NOT NULL,
                    fees REAL DEFAULT 0.0
                )
            """)

    def calculate_commission(self, price: float, quantity: int) -> float:
        """手数料計算"""
        gross_amount = price * quantity
        commission = max(gross_amount * self.commission_rate, self.min_commission)
        return commission

    async def place_order(self, symbol: str, direction: TradeDirection,
                         quantity: int, order_type: OrderType = OrderType.MARKET,
                         limit_price: float = 0.0) -> str:
        """
        注文発注

        Returns:
            str: 注文ID
        """
        try:
            # リアルデータ取得
            if not REAL_DATA_AVAILABLE:
                raise ValueError("Real data provider not available")

            real_data = self.data_provider.get_real_stock_data(f"{symbol}.T")
            if not real_data:
                raise ValueError(f"No real data available for {symbol}")

            current_price = real_data.current_price
            stock_name = self.data_provider.target_symbols.get(f"{symbol}.T", symbol)

            # 注文作成
            order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            order = PaperOrder(
                id=order_id,
                symbol=symbol,
                name=stock_name,
                direction=direction,
                order_type=order_type,
                quantity=quantity,
                price=limit_price if order_type == OrderType.LIMIT else current_price,
                timestamp=datetime.now()
            )

            # 資金チェック（買い注文の場合）
            if direction == TradeDirection.BUY:
                required_amount = current_price * quantity
                commission = self.calculate_commission(current_price, quantity)
                total_required = required_amount + commission

                if total_required > self.current_balance:
                    order.status = OrderStatus.REJECTED
                    self.logger.warning(f"Insufficient funds: required {total_required:,.0f}, available {self.current_balance:,.0f}")
                    return order_id

            # ポジションチェック（売り注文の場合）
            if direction == TradeDirection.SELL:
                if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                    order.status = OrderStatus.REJECTED
                    self.logger.warning(f"Insufficient position for sell order: {symbol}")
                    return order_id

            # 成行注文は即座に約定
            if order_type == OrderType.MARKET:
                await self._execute_order(order, current_price)
            else:
                # 指値注文は待機
                self.pending_orders.append(order)

            # データベース保存
            self._save_order(order)

            self.logger.info(f"Order placed: {order_id} - {direction.value} {quantity} shares of {symbol}")
            return order_id

        except Exception as e:
            self.logger.error(f"Order placement failed: {e}")
            raise

    async def _execute_order(self, order: PaperOrder, execution_price: float):
        """注文約定処理"""
        try:
            commission = self.calculate_commission(execution_price, order.quantity)

            if order.direction == TradeDirection.BUY:
                # 買い注文約定
                total_cost = execution_price * order.quantity + commission
                self.current_balance -= total_cost

                # ポジション追加/更新
                if order.symbol in self.positions:
                    # 既存ポジションに追加
                    existing = self.positions[order.symbol]
                    total_quantity = existing.quantity + order.quantity
                    avg_price = ((existing.entry_price * existing.quantity) +
                               (execution_price * order.quantity)) / total_quantity
                    existing.quantity = total_quantity
                    existing.entry_price = avg_price
                else:
                    # 新規ポジション
                    self.positions[order.symbol] = PaperPosition(
                        symbol=order.symbol,
                        name=order.name,
                        quantity=order.quantity,
                        entry_price=execution_price,
                        entry_time=datetime.now(),
                        current_price=execution_price
                    )

            else:  # TradeDirection.SELL
                # 売り注文約定
                total_proceeds = execution_price * order.quantity - commission
                self.current_balance += total_proceeds

                # ポジション決済
                if order.symbol in self.positions:
                    position = self.positions[order.symbol]

                    # PnL計算
                    pnl = (execution_price - position.entry_price) * order.quantity - commission
                    pnl_percent = ((execution_price - position.entry_price) / position.entry_price) * 100

                    # 取引記録作成
                    trade = PaperTrade(
                        symbol=order.symbol,
                        name=order.name,
                        quantity=order.quantity,
                        entry_price=position.entry_price,
                        exit_price=execution_price,
                        entry_time=position.entry_time,
                        exit_time=datetime.now(),
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        fees=commission
                    )

                    self.completed_trades.append(trade)
                    self._save_trade(trade)

                    # ポジション調整
                    if position.quantity == order.quantity:
                        # 完全決済
                        del self.positions[order.symbol]
                    else:
                        # 部分決済
                        position.quantity -= order.quantity

            # 注文状態更新
            order.status = OrderStatus.FILLED
            order.filled_price = execution_price
            order.filled_time = datetime.now()

            self.logger.info(f"Order executed: {order.id} at {execution_price:.2f}")

        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            order.status = OrderStatus.REJECTED

    async def update_market_prices(self):
        """市場価格更新"""
        if not REAL_DATA_AVAILABLE or not self.positions:
            return

        try:
            # 保有銘柄の価格更新
            symbols = [f"{symbol}.T" for symbol in self.positions.keys()]
            real_data = await self.data_provider.get_multiple_stocks_data(symbols)

            for symbol_with_t, stock_data in real_data.items():
                symbol = symbol_with_t.replace('.T', '')
                if symbol in self.positions:
                    self.positions[symbol].current_price = stock_data.current_price

            # 指値注文チェック
            await self._check_pending_orders()

        except Exception as e:
            self.logger.error(f"Price update failed: {e}")

    async def _check_pending_orders(self):
        """待機中の指値注文チェック"""
        for order in self.pending_orders[:]:  # コピーでイテレート
            try:
                # 現在価格取得
                real_data = self.data_provider.get_real_stock_data(f"{order.symbol}.T")
                if not real_data:
                    continue

                current_price = real_data.current_price
                should_execute = False

                if order.direction == TradeDirection.BUY and current_price <= order.price:
                    should_execute = True
                elif order.direction == TradeDirection.SELL and current_price >= order.price:
                    should_execute = True

                if should_execute:
                    await self._execute_order(order, order.price)
                    self.pending_orders.remove(order)

            except Exception as e:
                self.logger.error(f"Pending order check failed for {order.id}: {e}")

    def get_portfolio_summary(self) -> Dict[str, any]:
        """ポートフォリオサマリー"""
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_assets = self.current_balance + total_market_value

        # 完了取引統計
        if self.completed_trades:
            total_realized_pnl = sum(trade.pnl for trade in self.completed_trades)
            win_trades = len([t for t in self.completed_trades if t.pnl > 0])
            win_rate = win_trades / len(self.completed_trades) * 100
            avg_pnl = total_realized_pnl / len(self.completed_trades)
        else:
            total_realized_pnl = 0
            win_rate = 0
            avg_pnl = 0

        return {
            "timestamp": datetime.now().isoformat(),
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "total_market_value": total_market_value,
            "total_assets": total_assets,
            "total_pnl": total_assets - self.initial_balance,
            "total_pnl_percent": ((total_assets - self.initial_balance) / self.initial_balance) * 100,
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": total_realized_pnl,
            "positions_count": len(self.positions),
            "completed_trades": len(self.completed_trades),
            "win_rate": win_rate,
            "avg_pnl_per_trade": avg_pnl
        }

    def _save_order(self, order: PaperOrder):
        """注文をデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO orders
                (id, symbol, name, direction, order_type, quantity, price,
                 timestamp, status, filled_price, filled_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.id, order.symbol, order.name,
                order.direction.value, order.order_type.value,
                order.quantity, order.price,
                order.timestamp.isoformat(), order.status.value,
                order.filled_price,
                order.filled_time.isoformat() if order.filled_time else None
            ))

    def _save_trade(self, trade: PaperTrade):
        """取引をデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades
                (symbol, name, quantity, entry_price, exit_price,
                 entry_time, exit_time, pnl, pnl_percent, fees)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.symbol, trade.name, trade.quantity,
                trade.entry_price, trade.exit_price,
                trade.entry_time.isoformat(), trade.exit_time.isoformat(),
                trade.pnl, trade.pnl_percent, trade.fees
            ))

# テスト関数
async def test_paper_trading():
    """ペーパートレードシステムのテスト"""
    print("=== ペーパートレードシステム テスト ===")

    if not REAL_DATA_AVAILABLE:
        print("リアルデータプロバイダーが必要です")
        return

    # エンジン初期化
    engine = PaperTradingEngine(initial_balance=500000)  # 50万円

    print(f"\n[ 初期資金: {engine.initial_balance:,.0f}円 ]")

    try:
        # テスト用注文
        print("\n[ 買い注文テスト ]")
        order_id1 = await engine.place_order("7203", TradeDirection.BUY, 100)
        print(f"注文ID: {order_id1}")

        # 価格更新
        print("\n[ 価格更新 ]")
        await engine.update_market_prices()

        # ポートフォリオ確認
        print("\n[ ポートフォリオサマリー ]")
        summary = engine.get_portfolio_summary()
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:,.2f}")
            else:
                print(f"{key}: {value}")

        # ポジション表示
        print("\n[ 保有ポジション ]")
        for symbol, position in engine.positions.items():
            print(f"{symbol} ({position.name}): {position.quantity}株 @{position.entry_price:.2f}円")
            print(f"  時価: {position.current_price:.2f}円")
            print(f"  含み損益: {position.unrealized_pnl:+.0f}円 ({position.unrealized_pnl_percent:+.2f}%)")

    except Exception as e:
        print(f"テストエラー: {e}")

    print("\n" + "="*50)
    print("ペーパートレードシステム テスト完了")

if __name__ == "__main__":
    asyncio.run(test_paper_trading())