#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Engine - バックテストエンジン

実データを使用した過去検証システム
Phase5-B #903実装：システムの実際の性能・収益性の定量評価
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import statistics
import sqlite3
import time

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

try:
    from real_data_provider_v2 import real_data_provider
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False

try:
    from advanced_technical_analyzer import AdvancedTechnicalAnalyzer
    ADVANCED_TECHNICAL_AVAILABLE = True
except ImportError:
    ADVANCED_TECHNICAL_AVAILABLE = False

class TradeAction(Enum):
    """取引アクション"""
    BUY = "買い"
    SELL = "売り"
    HOLD = "ホールド"

class TradeResult(Enum):
    """取引結果"""
    WIN = "勝ち"
    LOSS = "負け"
    BREAKEVEN = "引き分け"

@dataclass
class Trade:
    """取引記録"""
    trade_id: str
    symbol: str
    action: TradeAction
    entry_date: datetime
    entry_price: float
    quantity: int

    # 決済情報
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = "未決済"

    # パフォーマンス
    profit_loss: float = 0.0
    return_pct: float = 0.0
    result: Optional[TradeResult] = None

    # 追加情報
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    trade_cost: float = 0.0  # 手数料等

@dataclass
class Portfolio:
    """ポートフォリオ状態"""
    cash: float
    positions: Dict[str, int]  # symbol -> quantity
    initial_capital: float
    current_value: float
    total_return: float
    total_return_pct: float

    # 履歴
    trade_history: List[Trade] = field(default_factory=list)
    daily_values: List[Tuple[datetime, float]] = field(default_factory=list)

@dataclass
class BacktestResult:
    """バックテスト結果"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float

    # 基本収益指標
    total_return: float
    total_return_pct: float
    annualized_return: float
    cagr: float  # 年平均成長率

    # リスク指標
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # 日数

    # 取引統計
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # 時系列データ
    portfolio_history: List[Tuple[datetime, float]]
    trade_history: List[Trade]

    # ベンチマーク比較
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

class TradingStrategy:
    """取引戦略基底クラス"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")

    async def generate_signal(self, symbol: str, data: pd.DataFrame,
                             current_date: datetime) -> Optional[Dict[str, Any]]:
        """取引シグナル生成"""
        raise NotImplementedError

    def calculate_position_size(self, signal: Dict[str, Any],
                              portfolio_value: float) -> int:
        """ポジションサイズ計算"""
        # デフォルト: ポートフォリオの10%
        trade_amount = portfolio_value * 0.1
        position_size = int(trade_amount / signal['price'])
        return max(1, position_size)

class SimpleMovingAverageStrategy(TradingStrategy):
    """移動平均戦略"""

    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__("SMA_Strategy")
        self.short_window = short_window
        self.long_window = long_window

    async def generate_signal(self, symbol: str, data: pd.DataFrame,
                             current_date: datetime) -> Optional[Dict[str, Any]]:
        """移動平均クロス戦略"""

        if len(data) < self.long_window:
            return None

        try:
            # 移動平均計算
            data['SMA_Short'] = data['Close'].rolling(self.short_window).mean()
            data['SMA_Long'] = data['Close'].rolling(self.long_window).mean()

            # 現在とひとつ前のデータ
            current = data.iloc[-1]
            previous = data.iloc[-2] if len(data) > 1 else current

            current_price = current['Close']

            # ゴールデンクロス（買いシグナル）
            if (current['SMA_Short'] > current['SMA_Long'] and
                previous['SMA_Short'] <= previous['SMA_Long']):

                return {
                    'action': TradeAction.BUY,
                    'price': current_price,
                    'confidence': 75.0,
                    'reason': f"ゴールデンクロス: SMA{self.short_window} > SMA{self.long_window}",
                    'stop_loss': current_price * 0.95,  # 5%損切り
                    'take_profit': current_price * 1.10  # 10%利確
                }

            # デッドクロス（売りシグナル）
            elif (current['SMA_Short'] < current['SMA_Long'] and
                  previous['SMA_Short'] >= previous['SMA_Long']):

                return {
                    'action': TradeAction.SELL,
                    'price': current_price,
                    'confidence': 70.0,
                    'reason': f"デッドクロス: SMA{self.short_window} < SMA{self.long_window}",
                    'stop_loss': current_price * 1.05,  # 5%損切り（売りの場合）
                    'take_profit': current_price * 0.90  # 10%利確
                }

            return None

        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return None

class AdvancedTechnicalStrategy(TradingStrategy):
    """高度技術分析戦略"""

    def __init__(self):
        super().__init__("Advanced_Technical_Strategy")
        if ADVANCED_TECHNICAL_AVAILABLE:
            self.analyzer = AdvancedTechnicalAnalyzer()

    async def generate_signal(self, symbol: str, data: pd.DataFrame,
                             current_date: datetime) -> Optional[Dict[str, Any]]:
        """高度技術分析によるシグナル生成"""

        if not ADVANCED_TECHNICAL_AVAILABLE or len(data) < 20:
            return None

        try:
            # 高度技術分析実行
            analysis = await self.analyzer.analyze_symbol(symbol, period="2mo")

            if not analysis:
                return None

            current_price = analysis.current_price
            composite_score = analysis.composite_score

            # プライマリシグナルに基づく判断
            buy_signals = [s for s in analysis.primary_signals if s.signal_type == "BUY"]
            sell_signals = [s for s in analysis.primary_signals if s.signal_type == "SELL"]

            # 買いシグナル
            if len(buy_signals) >= 2 and composite_score > 70:
                avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)

                return {
                    'action': TradeAction.BUY,
                    'price': current_price,
                    'confidence': avg_confidence,
                    'reason': f"複数買いシグナル({len(buy_signals)}個) + 高スコア({composite_score:.1f})",
                    'stop_loss': current_price * 0.93,  # 7%損切り
                    'take_profit': current_price * 1.12  # 12%利確
                }

            # 売りシグナル
            elif len(sell_signals) >= 1 and composite_score < 50:
                avg_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)

                return {
                    'action': TradeAction.SELL,
                    'price': current_price,
                    'confidence': avg_confidence,
                    'reason': f"売りシグナル + 低スコア({composite_score:.1f})",
                    'stop_loss': current_price * 1.07,
                    'take_profit': current_price * 0.88
                }

            return None

        except Exception as e:
            self.logger.error(f"Advanced signal generation failed: {e}")
            return None

class BacktestEngine:
    """バックテストエンジン"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # データベース初期化
        self.data_dir = Path("backtest_data")
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "backtest_results.db"
        self._init_database()

        # 取引コスト設定
        self.trading_costs = {
            'commission_rate': 0.001,  # 0.1%手数料
            'slippage': 0.0005,        # 0.05%スリッページ
            'min_commission': 100      # 最低手数料100円
        }

        # ベンチマークデータ
        self.benchmark_data = {}

    def _init_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # バックテスト結果テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    initial_capital REAL,
                    final_capital REAL,
                    total_return_pct REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 取引履歴テーブル
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_history (
                    trade_id TEXT PRIMARY KEY,
                    backtest_id INTEGER,
                    symbol TEXT,
                    action TEXT,
                    entry_date TEXT,
                    entry_price REAL,
                    exit_date TEXT,
                    exit_price REAL,
                    profit_loss REAL,
                    return_pct REAL,
                    FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
                )
            """)

    async def run_backtest(self, strategy: TradingStrategy, symbols: List[str],
                          start_date: str, end_date: str,
                          initial_capital: float = 1000000) -> BacktestResult:
        """バックテスト実行"""

        self.logger.info(f"Starting backtest: {strategy.name}")
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Symbols: {len(symbols)} ({', '.join(symbols[:5])}...)")
        self.logger.info(f"Initial capital: ¥{initial_capital:,.0f}")

        # ポートフォリオ初期化
        portfolio = Portfolio(
            cash=initial_capital,
            positions={},
            initial_capital=initial_capital,
            current_value=initial_capital,
            total_return=0.0,
            total_return_pct=0.0
        )

        # 日付範囲生成
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # 各銘柄のデータ取得
        symbol_data = {}
        for symbol in symbols:
            data = await self._get_historical_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                symbol_data[symbol] = data
                self.logger.info(f"Data loaded for {symbol}: {len(data)} days")
            else:
                self.logger.warning(f"No data available for {symbol}")

        if not symbol_data:
            raise ValueError("No data available for any symbols")

        # 日次シミュレーション
        current_date = start_dt
        day_count = 0

        while current_date <= end_dt:
            day_count += 1

            # 週末スキップ
            if current_date.weekday() >= 5:  # 土日
                current_date += timedelta(days=1)
                continue

            # ポジション評価
            await self._update_portfolio_value(portfolio, symbol_data, current_date)

            # 日次価値記録
            portfolio.daily_values.append((current_date, portfolio.current_value))

            # 各銘柄でシグナルチェック
            for symbol, data in symbol_data.items():
                # 現在日時点でのデータ取得
                historical_data = data[data.index <= current_date]

                if len(historical_data) < 10:  # 最低データ量
                    continue

                # シグナル生成
                signal = await strategy.generate_signal(symbol, historical_data, current_date)

                if signal:
                    await self._execute_trade(portfolio, symbol, signal, current_date)

            # 決済チェック（ストップロス・利確）
            await self._check_exit_conditions(portfolio, symbol_data, current_date)

            current_date += timedelta(days=1)

            # 進捗表示
            if day_count % 30 == 0:
                print(f"  Progress: {day_count} days processed, "
                      f"Portfolio: ¥{portfolio.current_value:,.0f} "
                      f"({portfolio.total_return_pct:+.1f}%)")

        # 最終ポジション決済
        await self._close_all_positions(portfolio, symbol_data, end_dt)

        # 結果計算
        result = await self._calculate_backtest_result(
            portfolio, start_dt, end_dt, strategy.name
        )

        # データベース保存
        await self._save_backtest_result(result, strategy.name)

        self.logger.info(f"Backtest completed: {result.total_return_pct:+.1f}% return")

        return result

    async def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """過去データ取得"""

        try:
            if REAL_DATA_PROVIDER_AVAILABLE:
                # 長期間データ取得（6ヶ月〜1年）
                data = await real_data_provider.get_stock_data(symbol, period="6mo")

                if data is not None and not data.empty:
                    self.logger.info(f"Raw data for {symbol}: {len(data)} rows, date range: {data.index.min()} to {data.index.max()}")

                    # 日付フィルタリング
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

                    # インデックスを日付型に変換（タイムゾーン情報を除去）
                    data.index = pd.to_datetime(data.index)
                    if data.index.tz is not None:
                        data.index = data.index.tz_convert(None)  # タイムゾーン情報除去

                    # datetime型をタイムゾーンナイーブに変換
                    start_dt = pd.Timestamp(start_dt)
                    end_dt = pd.Timestamp(end_dt)

                    self.logger.info(f"Filter range: {start_dt} to {end_dt}")

                    # 期間でフィルタリング
                    filtered_data = data[(data.index >= start_dt) & (data.index <= end_dt)]

                    self.logger.info(f"Filtered data for {symbol}: {len(filtered_data)} rows")

                    return filtered_data

            return None

        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    async def _update_portfolio_value(self, portfolio: Portfolio,
                                     symbol_data: Dict[str, pd.DataFrame],
                                     current_date: datetime):
        """ポートフォリオ価値更新"""

        total_value = portfolio.cash

        # 各ポジションの現在価値計算
        for symbol, quantity in portfolio.positions.items():
            if quantity == 0:
                continue

            if symbol in symbol_data:
                data = symbol_data[symbol]
                # 現在日以前の最新データ取得
                available_data = data[data.index <= current_date]

                if not available_data.empty:
                    current_price = available_data['Close'].iloc[-1]
                    position_value = abs(quantity) * current_price

                    if quantity > 0:  # ロングポジション
                        total_value += position_value
                    else:  # ショートポジション
                        total_value -= position_value

        portfolio.current_value = total_value
        portfolio.total_return = total_value - portfolio.initial_capital
        portfolio.total_return_pct = (total_value / portfolio.initial_capital - 1) * 100

    async def _execute_trade(self, portfolio: Portfolio, symbol: str,
                           signal: Dict[str, Any], current_date: datetime):
        """取引執行"""

        action = signal['action']
        price = signal['price']
        confidence = signal.get('confidence', 50.0)

        # ポジションサイズ計算（簡易版）
        if action == TradeAction.BUY:
            # 利用可能現金の10%で買い
            trade_amount = portfolio.cash * 0.1
            if trade_amount < price:  # 最低1株は買えない場合
                return

            quantity = int(trade_amount / price)
            if quantity == 0:
                return

            # 取引コスト計算
            trade_cost = max(
                quantity * price * self.trading_costs['commission_rate'],
                self.trading_costs['min_commission']
            )

            # 現金不足チェック
            total_cost = quantity * price + trade_cost
            if total_cost > portfolio.cash:
                return

            # ポジション更新
            portfolio.cash -= total_cost
            portfolio.positions[symbol] = portfolio.positions.get(symbol, 0) + quantity

            # 取引記録
            trade = Trade(
                trade_id=f"{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}_{action.value}",
                symbol=symbol,
                action=action,
                entry_date=current_date,
                entry_price=price,
                quantity=quantity,
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit'),
                confidence=confidence,
                trade_cost=trade_cost
            )

            portfolio.trade_history.append(trade)

            self.logger.debug(f"BUY executed: {symbol} {quantity} shares @ ¥{price:.2f}")

    async def _check_exit_conditions(self, portfolio: Portfolio,
                                   symbol_data: Dict[str, pd.DataFrame],
                                   current_date: datetime):
        """決済条件チェック"""

        for trade in portfolio.trade_history:
            if trade.exit_date is not None:  # 既に決済済み
                continue

            if trade.symbol not in symbol_data:
                continue

            # 現在価格取得
            data = symbol_data[trade.symbol]
            available_data = data[data.index <= current_date]

            if available_data.empty:
                continue

            current_price = available_data['Close'].iloc[-1]

            # 決済判定
            exit_price = None
            exit_reason = None

            if trade.action == TradeAction.BUY:
                # ロングポジションの決済判定
                if trade.stop_loss and current_price <= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = "損切り"
                elif trade.take_profit and current_price >= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = "利確"

            if exit_price:
                # 決済実行
                await self._close_position(portfolio, trade, exit_price, exit_reason, current_date)

    async def _close_position(self, portfolio: Portfolio, trade: Trade,
                             exit_price: float, exit_reason: str, exit_date: datetime):
        """ポジション決済"""

        # 決済コスト
        trade_cost = max(
            trade.quantity * exit_price * self.trading_costs['commission_rate'],
            self.trading_costs['min_commission']
        )

        # 現金回収
        proceeds = trade.quantity * exit_price - trade_cost
        portfolio.cash += proceeds

        # ポジション削除
        portfolio.positions[trade.symbol] = portfolio.positions.get(trade.symbol, 0) - trade.quantity
        if portfolio.positions[trade.symbol] == 0:
            del portfolio.positions[trade.symbol]

        # 取引結果計算
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.profit_loss = proceeds - (trade.quantity * trade.entry_price) - trade.trade_cost
        trade.return_pct = (exit_price / trade.entry_price - 1) * 100
        trade.result = TradeResult.WIN if trade.profit_loss > 0 else (TradeResult.LOSS if trade.profit_loss < 0 else TradeResult.BREAKEVEN)

        self.logger.debug(f"Position closed: {trade.symbol} {exit_reason} @ ¥{exit_price:.2f} "
                         f"P&L: ¥{trade.profit_loss:,.0f} ({trade.return_pct:+.1f}%)")

    async def _close_all_positions(self, portfolio: Portfolio,
                                  symbol_data: Dict[str, pd.DataFrame], end_date: datetime):
        """全ポジション決済"""

        for trade in portfolio.trade_history:
            if trade.exit_date is None:  # 未決済ポジション
                if trade.symbol in symbol_data:
                    data = symbol_data[trade.symbol]
                    final_price = data['Close'].iloc[-1]
                    await self._close_position(portfolio, trade, final_price, "期間終了", end_date)

    async def _calculate_backtest_result(self, portfolio: Portfolio,
                                       start_date: datetime, end_date: datetime,
                                       strategy_name: str) -> BacktestResult:
        """バックテスト結果計算"""

        # 基本収益指標
        total_return = portfolio.current_value - portfolio.initial_capital
        total_return_pct = (portfolio.current_value / portfolio.initial_capital - 1) * 100

        # 期間計算
        days = (end_date - start_date).days
        years = days / 365.25

        # 年率換算
        annualized_return = (pow(portfolio.current_value / portfolio.initial_capital, 1/years) - 1) * 100 if years > 0 else 0

        # 取引統計
        completed_trades = [t for t in portfolio.trade_history if t.exit_date is not None]
        total_trades = len(completed_trades)
        winning_trades = len([t for t in completed_trades if t.result == TradeResult.WIN])
        losing_trades = len([t for t in completed_trades if t.result == TradeResult.LOSS])

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # 勝ち負け分析
        wins = [t.profit_loss for t in completed_trades if t.result == TradeResult.WIN]
        losses = [t.profit_loss for t in completed_trades if t.result == TradeResult.LOSS]

        avg_win = statistics.mean(wins) if wins else 0
        avg_loss = statistics.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        # プロフィットファクター
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # リスク指標計算
        daily_values = [value for date, value in portfolio.daily_values]
        returns = []

        if len(daily_values) > 1:
            for i in range(1, len(daily_values)):
                daily_return = (daily_values[i] / daily_values[i-1] - 1)
                returns.append(daily_return)

        volatility = statistics.stdev(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0

        # シャープレシオ（リスクフリーレート0%と仮定）
        avg_return = statistics.mean(returns) if returns else 0
        sharpe_ratio = (avg_return * 252) / (statistics.stdev(returns) * np.sqrt(252)) if len(returns) > 1 and statistics.stdev(returns) > 0 else 0

        # 最大ドローダウン
        max_dd, max_dd_duration = self._calculate_max_drawdown(daily_values)

        # ソルティノレシオ
        negative_returns = [r for r in returns if r < 0]
        downside_std = statistics.stdev(negative_returns) if len(negative_returns) > 1 else 0
        sortino_ratio = (avg_return * 252) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=portfolio.initial_capital,
            final_capital=portfolio.current_value,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            cagr=annualized_return,  # 簡易実装
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            portfolio_history=portfolio.daily_values,
            trade_history=completed_trades
        )

    def _calculate_max_drawdown(self, daily_values: List[float]) -> Tuple[float, int]:
        """最大ドローダウン計算"""

        if len(daily_values) < 2:
            return 0.0, 0

        peak = daily_values[0]
        max_dd = 0.0
        max_dd_duration = 0
        current_dd_duration = 0

        for value in daily_values:
            if value > peak:
                peak = value
                current_dd_duration = 0
            else:
                drawdown = (peak - value) / peak * 100
                max_dd = max(max_dd, drawdown)
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)

        return max_dd, max_dd_duration

    async def _save_backtest_result(self, result: BacktestResult, strategy_name: str):
        """バックテスト結果保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                # メイン結果保存
                cursor = conn.execute("""
                    INSERT INTO backtest_results
                    (strategy_name, start_date, end_date, initial_capital, final_capital,
                     total_return_pct, sharpe_ratio, max_drawdown, win_rate, total_trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_name,
                    result.start_date.isoformat(),
                    result.end_date.isoformat(),
                    result.initial_capital,
                    result.final_capital,
                    result.total_return_pct,
                    result.sharpe_ratio,
                    result.max_drawdown,
                    result.win_rate,
                    result.total_trades
                ))

                backtest_id = cursor.lastrowid

                # 取引履歴保存
                for trade in result.trade_history:
                    conn.execute("""
                        INSERT INTO trade_history
                        (trade_id, backtest_id, symbol, action, entry_date, entry_price,
                         exit_date, exit_price, profit_loss, return_pct)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade.trade_id,
                        backtest_id,
                        trade.symbol,
                        trade.action.value,
                        trade.entry_date.isoformat(),
                        trade.entry_price,
                        trade.exit_date.isoformat() if trade.exit_date else None,
                        trade.exit_price,
                        trade.profit_loss,
                        trade.return_pct
                    ))

        except Exception as e:
            self.logger.error(f"Failed to save backtest result: {e}")

# テスト関数
async def test_backtest_engine():
    """バックテストエンジンのテスト"""

    print("=== バックテストエンジン テスト ===")

    engine = BacktestEngine()

    # テスト戦略
    sma_strategy = SimpleMovingAverageStrategy(short_window=3, long_window=8)

    # テスト設定
    test_symbols = ["7203", "8306"]  # トヨタ、三菱UFJ
    start_date = "2025-03-01"
    end_date = "2025-08-14"
    initial_capital = 1000000  # 100万円

    print(f"\n[ バックテスト設定 ]")
    print(f"戦略: {sma_strategy.name}")
    print(f"銘柄: {', '.join(test_symbols)}")
    print(f"期間: {start_date} ～ {end_date}")
    print(f"初期資本: ¥{initial_capital:,}")

    try:
        # バックテスト実行
        print(f"\n[ バックテスト実行中... ]")
        result = await engine.run_backtest(
            strategy=sma_strategy,
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )

        # 結果表示
        print(f"\n[ バックテスト結果 ]")
        print(f"最終資本: ¥{result.final_capital:,.0f}")
        print(f"総リターン: ¥{result.total_return:,.0f} ({result.total_return_pct:+.2f}%)")
        print(f"年率リターン: {result.annualized_return:+.2f}%")
        print(f"ボラティリティ: {result.volatility:.2f}%")
        print(f"シャープレシオ: {result.sharpe_ratio:.2f}")
        print(f"最大ドローダウン: {result.max_drawdown:.2f}%")

        print(f"\n[ 取引統計 ]")
        print(f"総取引数: {result.total_trades}")
        print(f"勝ち取引: {result.winning_trades}")
        print(f"負け取引: {result.losing_trades}")
        print(f"勝率: {result.win_rate:.1f}%")
        print(f"プロフィットファクター: {result.profit_factor:.2f}")
        print(f"平均勝ち: ¥{result.avg_win:,.0f}")
        print(f"平均負け: ¥{result.avg_loss:,.0f}")
        print(f"最大勝ち: ¥{result.largest_win:,.0f}")
        print(f"最大負け: ¥{result.largest_loss:,.0f}")

        # 代表的な取引表示
        if result.trade_history:
            print(f"\n[ 取引履歴（抜粋）]")
            for i, trade in enumerate(result.trade_history[:5]):
                exit_price_str = f"¥{trade.exit_price:.0f}" if trade.exit_price else "¥0"
                print(f"{i+1}. {trade.symbol} {trade.action.value} "
                      f"{trade.entry_date.strftime('%m/%d')} ¥{trade.entry_price:.0f} → "
                      f"{trade.exit_date.strftime('%m/%d') if trade.exit_date else '未決済'} "
                      f"{exit_price_str} "
                      f"({trade.return_pct:+.1f}%)")

        # パフォーマンス評価
        print(f"\n[ パフォーマンス評価 ]")
        if result.total_return_pct > 0:
            print("✅ プラス収益達成")
        else:
            print("❌ マイナス収益")

        if result.sharpe_ratio > 1.0:
            print("✅ 良好なリスク調整済みリターン")
        else:
            print("⚠️ リスク調整済みリターンが低い")

        if result.win_rate > 50:
            print("✅ 勝率50%超え")
        else:
            print("⚠️ 勝率50%未満")

    except Exception as e:
        print(f"❌ バックテストエラー: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n=== バックテストエンジン テスト完了 ===")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_backtest_engine())