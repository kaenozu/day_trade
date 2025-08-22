import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .datastructures import BacktestResult


class BacktestEngine:
    """バックテストエンジン"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def run_backtest(self, strategy_function, data: pd.DataFrame,
                          start_date: datetime, end_date: datetime,
                          initial_capital: float = 1000000) -> BacktestResult:
        """バックテスト実行"""
        try:
            # データ期間フィルタリング
            backtest_data = data[(data.index >= start_date) & (data.index <= end_date)]

            if backtest_data.empty:
                raise ValueError("No data available for backtest period")

            # 取引履歴
            trades = []
            portfolio_value = initial_capital
            daily_returns = []

            # 日次バックテスト実行
            for date in backtest_data.index[20:]:  # 20日のウォームアップ期間
                try:
                    # その日のデータまでを使用して予測
                    historical_data = backtest_data[backtest_data.index <= date]

                    # 戦略実行（模擬）
                    prediction = await self._simulate_strategy_prediction(historical_data)

                    if prediction:
                        # 取引実行
                        trade_result = self._execute_trade(prediction, historical_data.iloc[-1], portfolio_value)
                        if trade_result:
                            trades.append(trade_result)
                            portfolio_value = trade_result['portfolio_value_after']

                    # 日次リターン記録
                    if len(trades) > 1:
                        daily_return = (trades[-1]['portfolio_value_after'] - trades[-2]['portfolio_value_after']) / trades[-2]['portfolio_value_after']
                        daily_returns.append(daily_return)

                except Exception as e:
                    self.logger.warning(f"Backtest error on {date}: {e}")
                    continue

            # 結果分析
            return self._analyze_backtest_results(trades, daily_returns, start_date, end_date, initial_capital)

        except Exception as e:
            self.logger.error(f"Backtest execution failed: {e}")
            return self._default_backtest_result(start_date, end_date)

    async def _simulate_strategy_prediction(self, data: pd.DataFrame) -> Optional[Dict]:
        """戦略予測模擬"""
        if len(data) < 20:
            return None

        # 簡単な移動平均クロス戦略（模擬）
        sma_short = data['Close'].rolling(5).mean().iloc[-1]
        sma_long = data['Close'].rolling(20).mean().iloc[-1]
        current_price = data['Close'].iloc[-1]

        if sma_short > sma_long and current_price > sma_short:
            return {
                'direction': 'buy',
                'confidence': 0.6,
                'expected_return': 0.02,
                'entry_price': current_price
            }
        elif sma_short < sma_long and current_price < sma_short:
            return {
                'direction': 'sell',
                'confidence': 0.6,
                'expected_return': -0.02,
                'entry_price': current_price
            }

        return None

    def _execute_trade(self, prediction: Dict, market_data: pd.Series, portfolio_value: float) -> Optional[Dict]:
        """取引実行模擬"""
        try:
            direction = prediction['direction']
            entry_price = prediction['entry_price']
            expected_return = prediction['expected_return']

            # ポジションサイズ計算（簡易版）
            position_size_ratio = 0.1  # 10%
            position_value = portfolio_value * position_size_ratio

            # 取引実行
            if direction == 'buy':
                # 買い注文
                shares = position_value / entry_price
                # 簡易的な利益計算（実際の終値使用）
                exit_price = market_data['Close'] * (1 + expected_return)
                profit_loss = (exit_price - entry_price) * shares
            else:
                # 売り注文
                shares = position_value / entry_price
                exit_price = market_data['Close'] * (1 + expected_return)
                profit_loss = (entry_price - exit_price) * shares

            new_portfolio_value = portfolio_value + profit_loss

            return {
                'date': market_data.name,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'profit_loss': profit_loss,
                'portfolio_value_before': portfolio_value,
                'portfolio_value_after': new_portfolio_value
            }

        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            return None

    def _analyze_backtest_results(self, trades: List[Dict], daily_returns: List[float],
                                 start_date: datetime, end_date: datetime, initial_capital: float) -> BacktestResult:
        """バックテスト結果分析"""
        if not trades:
            return self._default_backtest_result(start_date, end_date)

        try:
            # 基本統計
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['profit_loss'] > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # リターン計算
            final_value = trades[-1]['portfolio_value_after']
            total_return = (final_value - initial_capital) / initial_capital

            # 年率換算
            days = (end_date - start_date).days
            years = days / 365.25
            annualized_return = (final_value / initial_capital) ** (1/years) - 1 if years > 0 else 0

            # リスク指標
            if daily_returns:
                volatility = np.std(daily_returns) * np.sqrt(252)
                sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
            else:
                volatility = 0.0
                sharpe_ratio = 0.0

            # 最大ドローダウン
            portfolio_values = [t['portfolio_value_after'] for t in trades]
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (np.array(portfolio_values) - running_max) / running_max
            max_drawdown = drawdowns.min()

            # その他指標
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

            winning_amounts = [t['profit_loss'] for t in trades if t['profit_loss'] > 0]
            losing_amounts = [abs(t['profit_loss']) for t in trades if t['profit_loss'] < 0]

            avg_win = np.mean(winning_amounts) if winning_amounts else 0
            avg_loss = np.mean(losing_amounts) if losing_amounts else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

            average_trade_return = np.mean([t['profit_loss'] for t in trades])

            return BacktestResult(
                period_start=start_date,
                period_end=end_date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                profit_factor=profit_factor,
                average_trade_return=average_trade_return
            )

        except Exception as e:
            self.logger.error(f"Backtest analysis failed: {e}")
            return self._default_backtest_result(start_date, end_date)

    def _default_backtest_result(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """デフォルトバックテスト結果"""
        return BacktestResult(
            period_start=start_date,
            period_end=end_date,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            profit_factor=0.0,
            average_trade_return=0.0
        )
