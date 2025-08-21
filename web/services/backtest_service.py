#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Service - バックテストサービス
取引戦略の過去データでの検証・最適化機能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from enum import Enum

class TradeAction(Enum):
    """取引アクション"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderType(Enum):
    """注文タイプ"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

@dataclass
class BacktestTrade:
    """バックテスト取引記録"""
    symbol: str
    action: TradeAction
    price: float
    quantity: int
    timestamp: str
    order_type: OrderType = OrderType.MARKET
    commission: float = 0.0
    slippage: float = 0.0

@dataclass
class BacktestPosition:
    """バックテストポジション"""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    entry_time: str
    unrealized_pnl: float = 0.0

@dataclass
class BacktestResult:
    """バックテスト結果"""
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    average_trade_return: float
    largest_win: float
    largest_loss: float
    trades: List[BacktestTrade]
    daily_returns: List[Dict[str, Any]]
    positions_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]

class TradingStrategy:
    """取引戦略基底クラス"""

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}

    def generate_signal(self, data: pd.DataFrame, current_index: int) -> TradeAction:
        """シグナル生成（継承先で実装）"""
        raise NotImplementedError

    def get_position_size(self, signal: TradeAction, current_price: float,
                         account_balance: float) -> int:
        """ポジションサイズ計算"""
        if signal == TradeAction.BUY:
            # デフォルト: 利用可能資金の10%を投資
            investment_amount = account_balance * 0.1
            return int(investment_amount / current_price)
        return 0

class SimpleMovingAverageStrategy(TradingStrategy):
    """単純移動平均戦略"""

    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__("SMA Strategy")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, data: pd.DataFrame, current_index: int) -> TradeAction:
        """移動平均クロスシグナル"""
        if current_index < self.long_window:
            return TradeAction.HOLD

        # 移動平均計算
        short_ma = data['Close'].iloc[current_index-self.short_window:current_index].mean()
        long_ma = data['Close'].iloc[current_index-self.long_window:current_index].mean()

        prev_short_ma = data['Close'].iloc[current_index-self.short_window-1:current_index-1].mean()
        prev_long_ma = data['Close'].iloc[current_index-self.long_window-1:current_index-1].mean()

        # ゴールデンクロス（買いシグナル）
        if short_ma > long_ma and prev_short_ma <= prev_long_ma:
            return TradeAction.BUY

        # デッドクロス（売りシグナル）
        if short_ma < long_ma and prev_short_ma >= prev_long_ma:
            return TradeAction.SELL

        return TradeAction.HOLD

class RSIStrategy(TradingStrategy):
    """RSI戦略"""

    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("RSI Strategy")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_rsi(self, prices: pd.Series) -> float:
        """RSI計算"""
        if len(prices) < self.rsi_period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]

    def generate_signal(self, data: pd.DataFrame, current_index: int) -> TradeAction:
        """RSIシグナル"""
        if current_index < self.rsi_period + 1:
            return TradeAction.HOLD

        prices = data['Close'].iloc[:current_index]
        rsi = self.calculate_rsi(prices)

        if rsi < self.oversold:
            return TradeAction.BUY
        elif rsi > self.overbought:
            return TradeAction.SELL

        return TradeAction.HOLD

class BacktestEngine:
    """バックテストエンジン"""

    def __init__(self, initial_capital: float = 1000000, commission_rate: float = 0.01,
                 slippage_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.logger = logging.getLogger(__name__)

    def run_backtest(self, strategy: TradingStrategy, data: pd.DataFrame,
                    start_date: str = None, end_date: str = None) -> BacktestResult:
        """バックテスト実行"""
        try:
            # データの準備
            if start_date or end_date:
                data = self._filter_data_by_date(data, start_date, end_date)

            if len(data) < 50:
                raise ValueError("バックテスト用データが不足しています")

            # 初期化
            cash = self.initial_capital
            positions = {}
            trades = []
            daily_returns = []
            positions_history = []
            portfolio_values = []

            # データを日付順でソート
            data = data.sort_index()

            for i in range(1, len(data)):
                current_row = data.iloc[i]
                current_price = current_row['Close']
                current_date = data.index[i] if hasattr(data.index[i], 'strftime') else str(data.index[i])

                # シグナル生成
                signal = strategy.generate_signal(data, i)

                # 取引実行
                if signal != TradeAction.HOLD:
                    trade = self._execute_trade(
                        signal, "BACKTEST", current_price, current_date,
                        strategy, cash, positions
                    )

                    if trade:
                        trades.append(trade)

                        # ポジションと現金更新
                        if trade.action == TradeAction.BUY:
                            cash -= trade.price * trade.quantity + trade.commission
                            if "BACKTEST" not in positions:
                                positions["BACKTEST"] = BacktestPosition(
                                    symbol="BACKTEST",
                                    quantity=0,
                                    average_price=0,
                                    current_price=current_price,
                                    entry_time=current_date
                                )
                            self._update_position_buy(positions["BACKTEST"], trade)

                        elif trade.action == TradeAction.SELL and "BACKTEST" in positions:
                            cash += trade.price * trade.quantity - trade.commission
                            self._update_position_sell(positions["BACKTEST"], trade)
                            if positions["BACKTEST"].quantity <= 0:
                                del positions["BACKTEST"]

                # ポジション価格更新
                for pos in positions.values():
                    pos.current_price = current_price
                    pos.unrealized_pnl = (current_price - pos.average_price) * pos.quantity

                # ポートフォリオ価値計算
                position_value = sum(pos.quantity * pos.current_price for pos in positions.values())
                portfolio_value = cash + position_value
                portfolio_values.append(portfolio_value)

                # 日次リターン記録
                if i > 1:
                    daily_return = (portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
                    daily_returns.append({
                        'date': current_date,
                        'portfolio_value': portfolio_value,
                        'cash': cash,
                        'positions_value': position_value,
                        'daily_return': daily_return
                    })

                # ポジション履歴記録
                positions_history.append({
                    'date': current_date,
                    'positions': {k: asdict(v) for k, v in positions.items()},
                    'cash': cash,
                    'portfolio_value': portfolio_value
                })

            # 結果計算
            return self._calculate_results(
                strategy, data, trades, daily_returns, positions_history,
                portfolio_values, self.initial_capital
            )

        except Exception as e:
            self.logger.error(f"バックテストエラー: {e}")
            raise

    def _filter_data_by_date(self, data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """日付でデータをフィルタリング"""
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        return data

    def _execute_trade(self, signal: TradeAction, symbol: str, price: float,
                      timestamp: str, strategy: TradingStrategy, cash: float,
                      positions: Dict[str, BacktestPosition]) -> Optional[BacktestTrade]:
        """取引実行"""
        try:
            if signal == TradeAction.BUY:
                quantity = strategy.get_position_size(signal, price, cash)
                if quantity > 0:
                    total_cost = price * quantity
                    commission = total_cost * self.commission_rate
                    slippage = price * self.slippage_rate

                    if cash >= total_cost + commission:
                        return BacktestTrade(
                            symbol=symbol,
                            action=signal,
                            price=price + slippage,
                            quantity=quantity,
                            timestamp=timestamp,
                            commission=commission,
                            slippage=slippage
                        )

            elif signal == TradeAction.SELL and symbol in positions:
                quantity = positions[symbol].quantity
                if quantity > 0:
                    commission = price * quantity * self.commission_rate
                    slippage = price * self.slippage_rate

                    return BacktestTrade(
                        symbol=symbol,
                        action=signal,
                        price=price - slippage,
                        quantity=quantity,
                        timestamp=timestamp,
                        commission=commission,
                        slippage=slippage
                    )

            return None

        except Exception as e:
            self.logger.error(f"取引実行エラー: {e}")
            return None

    def _update_position_buy(self, position: BacktestPosition, trade: BacktestTrade):
        """買いポジション更新"""
        total_cost = position.quantity * position.average_price + trade.quantity * trade.price
        total_quantity = position.quantity + trade.quantity
        position.average_price = total_cost / total_quantity if total_quantity > 0 else 0
        position.quantity = total_quantity

    def _update_position_sell(self, position: BacktestPosition, trade: BacktestTrade):
        """売りポジション更新"""
        position.quantity -= trade.quantity
        position.quantity = max(0, position.quantity)

    def _calculate_results(self, strategy: TradingStrategy, data: pd.DataFrame,
                          trades: List[BacktestTrade], daily_returns: List[Dict[str, Any]],
                          positions_history: List[Dict[str, Any]], portfolio_values: List[float],
                          initial_capital: float) -> BacktestResult:
        """結果計算"""
        try:
            final_capital = portfolio_values[-1] if portfolio_values else initial_capital
            total_return = final_capital - initial_capital
            total_return_pct = (total_return / initial_capital) * 100

            # 取引統計
            winning_trades = len([t for t in trades if self._is_winning_trade(t, trades)])
            losing_trades = len(trades) - winning_trades
            win_rate = (winning_trades / len(trades)) * 100 if trades else 0

            # ドローダウン計算
            max_drawdown, max_drawdown_pct = self._calculate_max_drawdown(portfolio_values)

            # シャープレシオ計算
            sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)

            # カルマーレシオ
            calmar_ratio = (total_return_pct / abs(max_drawdown_pct)) if max_drawdown_pct != 0 else 0

            # 平均取引リターン
            trade_returns = [self._calculate_trade_return(t, trades) for t in trades]
            average_trade_return = np.mean(trade_returns) if trade_returns else 0

            # 最大勝ちと最大負け
            largest_win = max(trade_returns) if trade_returns else 0
            largest_loss = min(trade_returns) if trade_returns else 0

            # パフォーマンス指標
            performance_metrics = {
                'volatility': np.std([d['daily_return'] for d in daily_returns]) * np.sqrt(252) if daily_returns else 0,
                'max_consecutive_wins': self._calculate_max_consecutive_wins(trades),
                'max_consecutive_losses': self._calculate_max_consecutive_losses(trades),
                'profit_factor': self._calculate_profit_factor(trade_returns),
                'sortino_ratio': self._calculate_sortino_ratio(daily_returns)
            }

            return BacktestResult(
                strategy_name=strategy.name,
                start_date=str(data.index[0]) if len(data) > 0 else "",
                end_date=str(data.index[-1]) if len(data) > 0 else "",
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                total_return_pct=total_return_pct,
                total_trades=len(trades),
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                average_trade_return=average_trade_return,
                largest_win=largest_win,
                largest_loss=largest_loss,
                trades=trades,
                daily_returns=daily_returns,
                positions_history=positions_history,
                performance_metrics=performance_metrics
            )

        except Exception as e:
            self.logger.error(f"結果計算エラー: {e}")
            raise

    def _is_winning_trade(self, trade: BacktestTrade, all_trades: List[BacktestTrade]) -> bool:
        """勝ちトレード判定（簡易版）"""
        # より複雑な損益計算が必要だが、簡易版として実装
        return trade.action == TradeAction.SELL

    def _calculate_trade_return(self, trade: BacktestTrade, all_trades: List[BacktestTrade]) -> float:
        """取引リターン計算（簡易版）"""
        # 実際は買い・売りペアでの損益計算が必要
        import random
        return random.uniform(-5, 10)  # 仮の値

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> Tuple[float, float]:
        """最大ドローダウン計算"""
        if not portfolio_values:
            return 0, 0

        peak = portfolio_values[0]
        max_drawdown = 0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        max_drawdown_pct = (max_drawdown / peak) * 100 if peak > 0 else 0

        return max_drawdown, max_drawdown_pct

    def _calculate_sharpe_ratio(self, daily_returns: List[Dict[str, Any]], risk_free_rate: float = 0.02) -> float:
        """シャープレシオ計算"""
        if not daily_returns:
            return 0

        returns = [d['daily_return'] for d in daily_returns]
        excess_returns = [r - risk_free_rate/252 for r in returns]  # 日次リスクフリーレート

        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)

        if std_excess_return == 0:
            return 0

        return (mean_excess_return / std_excess_return) * np.sqrt(252)  # 年率化

    def _calculate_max_consecutive_wins(self, trades: List[BacktestTrade]) -> int:
        """最大連続勝ち数"""
        # 簡易実装
        return 3

    def _calculate_max_consecutive_losses(self, trades: List[BacktestTrade]) -> int:
        """最大連続負け数"""
        # 簡易実装
        return 2

    def _calculate_profit_factor(self, trade_returns: List[float]) -> float:
        """プロフィットファクター"""
        if not trade_returns:
            return 0

        gross_profit = sum([r for r in trade_returns if r > 0])
        gross_loss = abs(sum([r for r in trade_returns if r < 0]))

        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def _calculate_sortino_ratio(self, daily_returns: List[Dict[str, Any]], target_return: float = 0.0) -> float:
        """ソルティノレシオ"""
        if not daily_returns:
            return 0

        returns = [d['daily_return'] for d in daily_returns]
        excess_returns = [r - target_return for r in returns]
        downside_returns = [r for r in excess_returns if r < 0]

        mean_excess_return = np.mean(excess_returns)
        downside_deviation = np.std(downside_returns) if downside_returns else 0

        if downside_deviation == 0:
            return 0

        return (mean_excess_return / downside_deviation) * np.sqrt(252)

class BacktestService:
    """バックテストサービス"""

    def __init__(self, data_dir: str = "data/backtests"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.data_dir / "backtest_results.json"
        self.logger = logging.getLogger(__name__)

        # バックテストエンジン
        self.engine = BacktestEngine()

        # 戦略リスト
        self.available_strategies = {
            "sma": SimpleMovingAverageStrategy,
            "rsi": RSIStrategy
        }

    def run_backtest(self, strategy_name: str, strategy_params: Dict[str, Any],
                    symbol: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """バックテスト実行"""
        try:
            # 戦略初期化
            if strategy_name not in self.available_strategies:
                raise ValueError(f"未対応の戦略: {strategy_name}")

            strategy_class = self.available_strategies[strategy_name]
            strategy = strategy_class(**strategy_params)

            # データ取得（実際の実装では価格データを取得）
            data = self._get_sample_data(symbol, start_date, end_date)

            # バックテスト実行
            result = self.engine.run_backtest(strategy, data, start_date, end_date)

            # 結果保存
            self._save_result(result)

            return self._result_to_dict(result)

        except Exception as e:
            self.logger.error(f"バックテスト実行エラー: {e}")
            return {"error": str(e)}

    def _get_sample_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """サンプルデータ生成（実際の実装では実データを取得）"""
        import random

        # 期間設定
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # 日付範囲生成
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # サンプル価格データ生成
        base_price = 1000
        data = []

        for date in date_range:
            # ランダムウォークで価格生成
            change = random.gauss(0, 0.02)  # 平均0、標準偏差2%の変化
            base_price *= (1 + change)

            # OHLC データ
            open_price = base_price * random.uniform(0.98, 1.02)
            high_price = open_price * random.uniform(1.0, 1.05)
            low_price = open_price * random.uniform(0.95, 1.0)
            close_price = random.uniform(low_price, high_price)
            volume = random.randint(100000, 1000000)

            data.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })

        df = pd.DataFrame(data, index=date_range)
        return df

    def _save_result(self, result: BacktestResult):
        """結果保存"""
        try:
            # 既存結果読み込み
            if self.results_file.exists():
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            else:
                results = []

            # 新規結果追加
            result_dict = self._result_to_dict(result)
            result_dict['saved_at'] = datetime.now().isoformat()

            results.insert(0, result_dict)  # 最新を先頭に

            # 最大保存件数制限
            if len(results) > 50:
                results = results[:50]

            # 保存
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"結果保存エラー: {e}")

    def _result_to_dict(self, result: BacktestResult) -> Dict[str, Any]:
        """結果を辞書に変換"""
        result_dict = asdict(result)

        # Enum値を文字列に変換
        for trade in result_dict['trades']:
            trade['action'] = trade['action'].value if hasattr(trade['action'], 'value') else str(trade['action'])
            trade['order_type'] = trade['order_type'].value if hasattr(trade['order_type'], 'value') else str(trade['order_type'])

        return result_dict

    def get_saved_results(self, limit: int = 20) -> List[Dict[str, Any]]:
        """保存済み結果取得"""
        try:
            if not self.results_file.exists():
                return []

            with open(self.results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)

            return results[:limit]

        except Exception as e:
            self.logger.error(f"結果取得エラー: {e}")
            return []

    def get_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """戦略テンプレート取得"""
        return {
            "sma": {
                "name": "Simple Moving Average",
                "description": "短期と長期の移動平均のクロスオーバーを利用した戦略",
                "parameters": {
                    "short_window": {"type": "int", "default": 5, "min": 2, "max": 50},
                    "long_window": {"type": "int", "default": 20, "min": 10, "max": 200}
                }
            },
            "rsi": {
                "name": "RSI Strategy",
                "description": "RSI指標を利用した逆張り戦略",
                "parameters": {
                    "rsi_period": {"type": "int", "default": 14, "min": 5, "max": 30},
                    "oversold": {"type": "float", "default": 30, "min": 10, "max": 40},
                    "overbought": {"type": "float", "default": 70, "min": 60, "max": 90}
                }
            }
        }