#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest & Paper Trading System - バックテスト・ペーパートレーディング検証システム

Issue #812対応：リアルマネーを使わない安全な検証環境
過去データでのバックテストと仮想的なペーパートレーディング
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
import uuid

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

class OrderType(Enum):
    """注文タイプ"""
    MARKET = "market"      # 成行
    LIMIT = "limit"        # 指値
    STOP = "stop"          # 逆指値

class OrderSide(Enum):
    """注文方向"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """注文状態"""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    PARTIALLY_FILLED = "partially_filled"

@dataclass
class Order:
    """注文クラス"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Position:
    """ポジションクラス"""
    symbol: str
    quantity: int
    average_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class Portfolio:
    """ポートフォリオクラス"""
    cash_balance: float
    positions: Dict[str, Position]
    total_value: float = 0.0
    total_pnl: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0

@dataclass
class BacktestResult:
    """バックテスト結果"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_winning_trade: float
    avg_losing_trade: float
    largest_win: float
    largest_loss: float

class PaperTradingEngine:
    """ペーパートレーディングエンジン"""

    def __init__(self, initial_capital: float = 1000000):
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(
            cash_balance=initial_capital,
            positions={}
        )

        # 注文・取引履歴
        self.orders: List[Order] = []
        self.executed_trades: List[Dict] = []

        # 設定
        self.commission_rate = 0.001  # 0.1%の手数料
        self.min_commission = 100     # 最低手数料100円

        # データベース設定
        self.db_path = Path("backtest_data/paper_trading.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

        self.logger.info(f"Paper trading engine initialized with ¥{initial_capital:,.0f}")

    def _init_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 注文テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL,
                        stop_price REAL,
                        status TEXT NOT NULL,
                        filled_quantity INTEGER DEFAULT 0,
                        filled_price REAL DEFAULT 0.0,
                        commission REAL DEFAULT 0.0,
                        timestamp TEXT NOT NULL
                    )
                ''')

                # ポートフォリオ履歴
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        cash_balance REAL NOT NULL,
                        total_value REAL NOT NULL,
                        total_pnl REAL NOT NULL,
                        positions_json TEXT
                    )
                ''')

                # 取引履歴
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        price REAL NOT NULL,
                        commission REAL NOT NULL,
                        pnl REAL DEFAULT 0.0,
                        timestamp TEXT NOT NULL
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")

    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                         quantity: int, price: Optional[float] = None,
                         stop_price: Optional[float] = None) -> Order:
        """注文発注"""

        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )

        # 資金チェック
        if side == OrderSide.BUY:
            required_capital = quantity * (price or await self._get_current_price(symbol))
            commission = max(required_capital * self.commission_rate, self.min_commission)

            if self.portfolio.cash_balance < required_capital + commission:
                self.logger.warning(f"Insufficient funds for order {order.order_id}")
                order.status = OrderStatus.CANCELLED
                return order

        # ポジションチェック（売り注文時）
        elif side == OrderSide.SELL:
            current_position = self.portfolio.positions.get(symbol)
            if not current_position or current_position.quantity < quantity:
                self.logger.warning(f"Insufficient position for sell order {order.order_id}")
                order.status = OrderStatus.CANCELLED
                return order

        self.orders.append(order)

        # 成行注文は即座に執行
        if order_type == OrderType.MARKET:
            await self._execute_order(order)

        # データベース保存
        await self._save_order(order)

        self.logger.info(f"Order placed: {order.order_id} {side.value} {quantity} {symbol}")
        return order

    async def _get_current_price(self, symbol: str) -> float:
        """現在価格取得"""
        try:
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "1d")
            if data is not None and len(data) > 0:
                return float(data['Close'].iloc[-1])
            return 1000.0  # デフォルト価格
        except Exception:
            return 1000.0  # エラー時のデフォルト価格

    async def _execute_order(self, order: Order):
        """注文執行"""

        current_price = await self._get_current_price(order.symbol)

        # 執行価格決定
        if order.order_type == OrderType.MARKET:
            execution_price = current_price
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.price:
                execution_price = order.price
            elif order.side == OrderSide.SELL and current_price >= order.price:
                execution_price = order.price
            else:
                return  # 執行条件を満たさない
        else:
            return  # その他の注文タイプ

        # 手数料計算
        trade_value = order.quantity * execution_price
        commission = max(trade_value * self.commission_rate, self.min_commission)

        # ポートフォリオ更新
        if order.side == OrderSide.BUY:
            # 買い注文
            total_cost = trade_value + commission
            self.portfolio.cash_balance -= total_cost

            if order.symbol in self.portfolio.positions:
                # 既存ポジションに追加
                pos = self.portfolio.positions[order.symbol]
                total_quantity = pos.quantity + order.quantity
                total_cost_basis = (pos.quantity * pos.average_price) + trade_value
                pos.average_price = total_cost_basis / total_quantity
                pos.quantity = total_quantity
            else:
                # 新規ポジション
                self.portfolio.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    average_price=execution_price,
                    current_price=current_price
                )

        else:
            # 売り注文
            if order.symbol in self.portfolio.positions:
                pos = self.portfolio.positions[order.symbol]

                # 実現損益計算
                realized_pnl = (execution_price - pos.average_price) * order.quantity - commission
                pos.realized_pnl += realized_pnl

                # キャッシュ追加
                self.portfolio.cash_balance += trade_value - commission

                # ポジション更新
                pos.quantity -= order.quantity
                if pos.quantity <= 0:
                    del self.portfolio.positions[order.symbol]

        # 注文状態更新
        order.status = OrderStatus.EXECUTED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.commission = commission

        # 取引記録
        trade_record = {
            "trade_id": str(uuid.uuid4()),
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": execution_price,
            "commission": commission,
            "pnl": realized_pnl if order.side == OrderSide.SELL else 0.0,
            "timestamp": datetime.now()
        }

        self.executed_trades.append(trade_record)
        await self._save_trade(trade_record)

        self.logger.info(f"Order executed: {order.order_id} at ¥{execution_price:.2f}")

    async def update_portfolio(self):
        """ポートフォリオ更新"""

        total_value = self.portfolio.cash_balance

        # 各ポジションの時価評価
        for symbol, position in self.portfolio.positions.items():
            current_price = await self._get_current_price(symbol)
            position.current_price = current_price
            position.unrealized_pnl = (current_price - position.average_price) * position.quantity
            total_value += position.quantity * current_price

        self.portfolio.total_value = total_value
        self.portfolio.total_pnl = total_value - self.initial_capital
        self.portfolio.total_return = (total_value / self.initial_capital - 1) * 100

        # ポートフォリオ履歴保存
        await self._save_portfolio_snapshot()

    async def _save_order(self, order: Order):
        """注文保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO orders
                    (order_id, symbol, side, order_type, quantity, price, stop_price,
                     status, filled_quantity, filled_price, commission, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    order.order_id, order.symbol, order.side.value, order.order_type.value,
                    order.quantity, order.price, order.stop_price, order.status.value,
                    order.filled_quantity, order.filled_price, order.commission,
                    order.timestamp.isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Save order error: {e}")

    async def _save_trade(self, trade_record: Dict):
        """取引保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades
                    (trade_id, symbol, side, quantity, price, commission, pnl, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_record["trade_id"], trade_record["symbol"], trade_record["side"],
                    trade_record["quantity"], trade_record["price"], trade_record["commission"],
                    trade_record["pnl"], trade_record["timestamp"].isoformat()
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Save trade error: {e}")

    async def _save_portfolio_snapshot(self):
        """ポートフォリオスナップショット保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                positions_json = json.dumps({
                    symbol: {
                        "quantity": pos.quantity,
                        "average_price": pos.average_price,
                        "current_price": pos.current_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "realized_pnl": pos.realized_pnl
                    }
                    for symbol, pos in self.portfolio.positions.items()
                })

                cursor.execute('''
                    INSERT INTO portfolio_history
                    (timestamp, cash_balance, total_value, total_pnl, positions_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    self.portfolio.cash_balance,
                    self.portfolio.total_value,
                    self.portfolio.total_pnl,
                    positions_json
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Save portfolio error: {e}")

class BacktestEngine:
    """バックテストエンジン"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データベース設定
        self.db_path = Path("backtest_data/backtest_results.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS backtest_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        initial_capital REAL NOT NULL,
                        final_capital REAL NOT NULL,
                        total_return REAL NOT NULL,
                        annualized_return REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        win_rate REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        winning_trades INTEGER NOT NULL,
                        losing_trades INTEGER NOT NULL,
                        avg_winning_trade REAL NOT NULL,
                        avg_losing_trade REAL NOT NULL,
                        largest_win REAL NOT NULL,
                        largest_loss REAL NOT NULL,
                        created_at TEXT NOT NULL
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"Backtest database initialization error: {e}")

    async def run_simple_ma_crossover_backtest(self, symbol: str, period: str = "1y",
                                               short_ma: int = 5, long_ma: int = 20,
                                               initial_capital: float = 1000000) -> BacktestResult:
        """シンプルな移動平均クロスオーバー戦略のバックテスト"""

        self.logger.info(f"Running MA crossover backtest for {symbol}")

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, period)

            if data is None or len(data) < max(short_ma, long_ma) + 10:
                raise ValueError("Insufficient data for backtest")

            # 移動平均計算
            data['MA_Short'] = data['Close'].rolling(short_ma).mean()
            data['MA_Long'] = data['Close'].rolling(long_ma).mean()

            # シグナル生成
            data['Signal'] = 0
            data['Position'] = 0

            # クロスオーバーシグナル
            data.loc[data['MA_Short'] > data['MA_Long'], 'Signal'] = 1
            data.loc[data['MA_Short'] < data['MA_Long'], 'Signal'] = -1

            # ポジション計算（前日のシグナルでエントリー）
            data['Position'] = data['Signal'].shift(1).fillna(0)

            # リターン計算
            data['Returns'] = data['Close'].pct_change()
            data['Strategy_Returns'] = data['Position'] * data['Returns']
            data['Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()

            # 取引コスト考慮
            data['Position_Change'] = data['Position'].diff().abs()
            transaction_cost = 0.002  # 0.2%の取引コスト
            data['Transaction_Cost'] = data['Position_Change'] * transaction_cost
            data['Net_Strategy_Returns'] = data['Strategy_Returns'] - data['Transaction_Cost']
            data['Net_Cumulative_Returns'] = (1 + data['Net_Strategy_Returns']).cumprod()

            # パフォーマンス計算
            start_date = data.index[0]
            end_date = data.index[-1]
            trading_days = len(data)
            years = trading_days / 252

            final_value = data['Net_Cumulative_Returns'].iloc[-1] * initial_capital
            total_return = (final_value / initial_capital - 1) * 100
            annualized_return = ((final_value / initial_capital) ** (1/years) - 1) * 100

            # ドローダウン計算
            rolling_max = data['Net_Cumulative_Returns'].expanding().max()
            drawdown = (data['Net_Cumulative_Returns'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100

            # シャープレシオ
            excess_returns = data['Net_Strategy_Returns']
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

            # 取引統計
            position_changes = data['Position'].diff()
            entries = position_changes[position_changes != 0]

            if len(entries) > 1:
                trades = []
                entry_price = None
                entry_position = None

                for date, position in zip(data.index, data['Position']):
                    prev_position = data['Position'].shift(1).loc[date] if date in data['Position'].shift(1).index else 0

                    if position != prev_position and position != 0:
                        # エントリー
                        entry_price = data.loc[date, 'Close']
                        entry_position = position
                    elif position != prev_position and prev_position != 0:
                        # エグジット
                        if entry_price is not None:
                            exit_price = data.loc[date, 'Close']
                            pnl = (exit_price - entry_price) / entry_price * entry_position
                            trades.append(pnl)

                winning_trades = len([t for t in trades if t > 0])
                losing_trades = len([t for t in trades if t < 0])
                total_trades = len(trades)

                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                avg_winning_trade = np.mean([t for t in trades if t > 0]) * 100 if winning_trades > 0 else 0
                avg_losing_trade = np.mean([t for t in trades if t < 0]) * 100 if losing_trades > 0 else 0
                largest_win = max(trades) * 100 if trades else 0
                largest_loss = min(trades) * 100 if trades else 0
            else:
                total_trades = winning_trades = losing_trades = 0
                win_rate = avg_winning_trade = avg_losing_trade = largest_win = largest_loss = 0

            # 結果作成
            result = BacktestResult(
                strategy_name=f"MA_Cross_{short_ma}_{long_ma}",
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=final_value,
                total_return=total_return,
                annualized_return=annualized_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_winning_trade=avg_winning_trade,
                avg_losing_trade=avg_losing_trade,
                largest_win=largest_win,
                largest_loss=largest_loss
            )

            # データベース保存
            await self._save_backtest_result(result)

            return result

        except Exception as e:
            self.logger.error(f"Backtest error for {symbol}: {e}")
            raise

    async def _save_backtest_result(self, result: BacktestResult):
        """バックテスト結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO backtest_results
                    (strategy_name, symbol, start_date, end_date, initial_capital, final_capital,
                     total_return, annualized_return, max_drawdown, sharpe_ratio, win_rate,
                     total_trades, winning_trades, losing_trades, avg_winning_trade,
                     avg_losing_trade, largest_win, largest_loss, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.strategy_name, result.symbol,
                    result.start_date.isoformat(), result.end_date.isoformat(),
                    result.initial_capital, result.final_capital, result.total_return,
                    result.annualized_return, result.max_drawdown, result.sharpe_ratio,
                    result.win_rate, result.total_trades, result.winning_trades,
                    result.losing_trades, result.avg_winning_trade, result.avg_losing_trade,
                    result.largest_win, result.largest_loss, datetime.now().isoformat()
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"Save backtest result error: {e}")

# グローバルインスタンス
paper_trading_engine = PaperTradingEngine()
backtest_engine = BacktestEngine()

# テスト実行
async def run_backtest_paper_trading_test():
    """バックテスト・ペーパートレーディングテスト"""

    print("=== 📈 バックテスト・ペーパートレーディングシステムテスト ===")

    # バックテストテスト
    print("\n--- バックテストテスト ---")
    test_symbols = ["7203", "8306"]

    for symbol in test_symbols:
        print(f"\n{symbol} MA crossover backtest...")

        try:
            result = await backtest_engine.run_simple_ma_crossover_backtest(symbol, "6mo")

            print(f"戦略: {result.strategy_name}")
            print(f"期間: {result.start_date.date()} ~ {result.end_date.date()}")
            print(f"総リターン: {result.total_return:.2f}%")
            print(f"年率リターン: {result.annualized_return:.2f}%")
            print(f"最大ドローダウン: {result.max_drawdown:.2f}%")
            print(f"シャープレシオ: {result.sharpe_ratio:.3f}")
            print(f"勝率: {result.win_rate:.1f}%")
            print(f"総取引数: {result.total_trades}")

        except Exception as e:
            print(f"Backtest error for {symbol}: {e}")

    # ペーパートレーディングテスト
    print("\n--- ペーパートレーディングテスト ---")

    # 注文テスト
    print("テスト注文発注中...")

    buy_order = await paper_trading_engine.place_order(
        symbol="7203",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100
    )
    print(f"買い注文: {buy_order.status.value}")

    await paper_trading_engine.update_portfolio()
    print(f"ポートフォリオ価値: ¥{paper_trading_engine.portfolio.total_value:,.0f}")
    print(f"現金残高: ¥{paper_trading_engine.portfolio.cash_balance:,.0f}")
    print(f"ポジション数: {len(paper_trading_engine.portfolio.positions)}")

    # 売り注文テスト
    if "7203" in paper_trading_engine.portfolio.positions:
        sell_order = await paper_trading_engine.place_order(
            symbol="7203",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=50
        )
        print(f"売り注文: {sell_order.status.value}")

        await paper_trading_engine.update_portfolio()
        print(f"売却後ポートフォリオ価値: ¥{paper_trading_engine.portfolio.total_value:,.0f}")
        print(f"総損益: ¥{paper_trading_engine.portfolio.total_pnl:,.0f}")

    print("\n✅ バックテスト・ペーパートレーディングシステム動作確認完了")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_backtest_paper_trading_test())