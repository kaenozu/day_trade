#!/usr/bin/env python3
"""
実データバックテストエンジン

Issue #323: 実データバックテスト機能開発
過去データを使用したトレーディング戦略の検証システム
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from day_trade.analysis.events import OrderType, OrderStatus, Order, Position, Portfolio, BacktestResults


class BacktestEngine:
    """バックテストエンジン"""

    def __init__(self, initial_capital: float = 1000000):
        """
        初期化

        Args:
            initial_capital: 初期資本金（デフォルト100万円）
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # ポートフォリオ状態
        self.cash = initial_capital
        self.positions: Dict[str, events.Position] = {}
        self.orders: List[events.Order] = []
        self.trades: List[Dict[str, Any]] = []

        # パフォーマンストラッキング
        self.portfolio_history: List[events.Portfolio] = []
        self.daily_returns: List[float] = []
        self.portfolio_values: List[float] = []

        

        print(f"バックテストエンジン初期化: 初期資本 {initial_capital:,.0f}円")

    def load_historical_data(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        過去データ読み込み

        Args:
            symbols: 銘柄リスト
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD)

        Returns:
            銘柄別価格データ
        """
        print(f"過去データ読み込み開始: {len(symbols)}銘柄, {start_date} - {end_date}")

        historical_data = {}
        successful_loads = 0

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)

                if not data.empty and len(data) >= 30:  # 最低30日分のデータ
                    # 必要な列を確保
                    data["Returns"] = data["Close"].pct_change()
                    data["Volume_Avg"] = data["Volume"].rolling(20).mean()

                    historical_data[symbol] = data
                    successful_loads += 1

                else:
                    print(
                        f"データ不足: {symbol} - {len(data) if not data.empty else 0}日分"
                    )

            except Exception as e:
                print(f"データ読み込みエラー: {symbol} - {e}")

        print(f"過去データ読み込み完了: {successful_loads}/{len(symbols)}銘柄")
        return historical_data

    def execute_backtest(
        self,
        historical_data: Dict[str, pd.DataFrame],
        strategy_function: callable,
        rebalance_frequency: int = 5,
    ) -> BacktestResults:
        """
        バックテスト実行

        Args:
            historical_data: 過去データ
            strategy_function: 戦略関数
            rebalance_frequency: リバランス頻度（営業日）

        Returns:
            バックテスト結果
        """
        print("バックテスト実行開始")

        with perf_monitor.monitor("backtest_execution"):
            # 共通の日付インデックス作成
            all_dates = self._get_common_dates(historical_data)

            if len(all_dates) < 30:
                raise ValueError(f"データが不足しています: {len(all_dates)}日分")

            print(
                f"バックテスト期間: {all_dates[0]} - {all_dates[-1]} ({len(all_dates)}日)"
            )

            # 日次バックテスト実行
            for i, current_date in enumerate(all_dates):
                # 現在の市場データ取得
                current_prices = self._get_prices_for_date(
                    historical_data, current_date
                )

                # ポジション評価更新
                self._update_positions(current_prices)

                # 戦略シグナル生成（リバランス日のみ）
                if i % rebalance_frequency == 0:
                    # 過去データウィンドウを戦略関数に渡す
                    lookback_data = self._get_lookback_data(
                        historical_data, current_date, window=30
                    )

                    if lookback_data:
                        signals = strategy_function(lookback_data, current_prices)
                        self._execute_trading_signals(
                            signals, current_prices, current_date
                        )

                # ポートフォリオ状態記録
                portfolio_value = self._calculate_portfolio_value(current_prices)
                self._record_portfolio_state(current_date, portfolio_value)

                # 進捗表示（10日ごと）
                if i % 10 == 0:
                    progress = (i + 1) / len(all_dates) * 100
                    print(
                        f"バックテスト進捗: {progress:.1f}% ({current_date.strftime('%Y-%m-%d')})"
                    )

            # 結果分析
            results = self._analyze_results(all_dates[0], all_dates[-1])

            print(
                f"バックテスト完了: 最終価値 {results.final_value:,.0f}円, リターン {results.total_return:.2%}"
            )
            return results

    def _get_common_dates(
        self, historical_data: Dict[str, pd.DataFrame]
    ) -> pd.DatetimeIndex:
        """共通の取引日インデックス取得"""
        if not historical_data:
            return pd.DatetimeIndex([])

        # 全銘柄で共通する日付を取得
        common_dates = None
        for data in historical_data.values():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)

        return common_dates.sort_values()

    def _get_prices_for_date(
        self, historical_data: Dict[str, pd.DataFrame], date: pd.Timestamp
    ) -> Dict[str, float]:
        """指定日の価格データ取得"""
        prices = {}
        for symbol, data in historical_data.items():
            if date in data.index:
                prices[symbol] = data.loc[date, "Close"]
        return prices

    def _get_lookback_data(
        self,
        historical_data: Dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
        window: int = 30,
    ) -> Dict[str, pd.DataFrame]:
        """ルックバックデータ取得"""
        lookback_data = {}

        for symbol, data in historical_data.items():
            # 現在日より前のデータを取得
            mask = data.index < current_date
            recent_data = data[mask].tail(window)

            if len(recent_data) >= window // 2:  # 最低半分のデータがあれば使用
                lookback_data[symbol] = recent_data

        return lookback_data

    def _update_positions(self, current_prices: Dict[str, float]):
        """ポジション評価更新"""
        for symbol in list(self.positions.keys()):
            if symbol in current_prices:
                position = self.positions[symbol]
                position.current_price = current_prices[symbol]
                position.market_value = position.quantity * position.current_price
                position.unrealized_pnl = (
                    position.current_price - position.avg_price
                ) * position.quantity
            else:
                # 価格データがない場合は前日の価格を維持
                pass

    def _execute_trading_signals(
        self,
        signals: Dict[str, float],
        current_prices: Dict[str, float],
        current_date: pd.Timestamp,
    ):
        """取引シグナル実行"""
        if not signals:
            return

        # 現在のポートフォリオ価値
        portfolio_value = self._calculate_portfolio_value(current_prices)

        # 各銘柄の目標ウェイト
        for symbol, target_weight in signals.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            target_value = portfolio_value * target_weight
            target_quantity = (
                int(target_value / current_price) if current_price > 0 else 0
            )

            # 現在のポジション
            current_quantity = self.positions.get(
                symbol, events.Position(symbol, 0, 0, current_price, 0, 0)
            ).quantity

            # 取引が必要な数量
            trade_quantity = target_quantity - current_quantity

            if abs(trade_quantity) > 0:
                if trade_quantity > 0:
                    self._execute_buy_order(
                        symbol, trade_quantity, current_price, current_date
                    )
                else:
                    self._execute_sell_order(
                        symbol, abs(trade_quantity), current_price, current_date
                    )

    def _execute_buy_order(
        self, symbol: str, quantity: int, price: float, date: pd.Timestamp
    ):
        """買い注文実行"""
        # 取引コスト計算
        gross_amount = quantity * price
        commission = gross_amount * self.commission_rate
        slippage = gross_amount * self.slippage_rate
        total_cost = gross_amount + commission + slippage

        # 資金チェック
        if total_cost > self.cash:
            # 買える分だけ購入
            affordable_quantity = int(
                self.cash / (price * (1 + self.commission_rate + self.slippage_rate))
            )
            if affordable_quantity <= 0:
                return
            quantity = affordable_quantity
            total_cost = (
                quantity * price * (1 + self.commission_rate + self.slippage_rate)
            )

        # 現金減少
        self.cash -= total_cost

        # ポジション更新
        if symbol in self.positions:
            position = self.positions[symbol]
            total_quantity = position.quantity + quantity
            total_cost_basis = position.quantity * position.avg_price + quantity * price
            new_avg_price = (
                total_cost_basis / total_quantity if total_quantity > 0 else price
            )

            position.quantity = total_quantity
            position.avg_price = new_avg_price
            position.current_price = price
            position.market_value = total_quantity * price
            position.unrealized_pnl = (price - new_avg_price) * total_quantity
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=0.0,
            )

        # 取引記録
        self.trades.append(
            {
                "date": date,
                "symbol": symbol,
                "action": "BUY",
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "total_cost": total_cost,
            }
        )

    def _execute_sell_order(
        self, symbol: str, quantity: int, price: float, date: pd.Timestamp
    ):
        """売り注文実行"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        sell_quantity = min(quantity, position.quantity)

        if sell_quantity <= 0:
            return

        # 取引コスト計算
        gross_proceeds = sell_quantity * price
        commission = gross_proceeds * self.commission_rate
        slippage = gross_proceeds * self.slippage_rate
        net_proceeds = gross_proceeds - commission - slippage

        # 実現損益計算
        realized_pnl = (price - position.avg_price) * sell_quantity

        # 現金増加
        self.cash += net_proceeds

        # ポジション更新
        position.quantity -= sell_quantity
        position.realized_pnl += realized_pnl
        position.market_value = position.quantity * price
        position.unrealized_pnl = (price - position.avg_price) * position.quantity

        # ポジションが0になったら削除
        if position.quantity <= 0:
            del self.positions[symbol]

        # 取引記録
        self.trades.append(
            {
                "date": date,
                "symbol": symbol,
                "action": "SELL",
                "quantity": sell_quantity,
                "price": price,
                "commission": commission,
                "realized_pnl": realized_pnl,
                "net_proceeds": net_proceeds,
            }
        )

    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """ポートフォリオ価値計算"""
        positions_value = 0.0

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                positions_value += position.quantity * current_prices[symbol]
            else:
                positions_value += position.market_value  # 前回価格使用

        return self.cash + positions_value

    def _record_portfolio_state(self, date: pd.Timestamp, portfolio_value: float):
        """ポートフォリオ状態記録"""
        # デイリーリターン計算
        daily_return = 0.0
        if self.portfolio_values:
            daily_return = (
                portfolio_value - self.portfolio_values[-1]
            ) / self.portfolio_values[-1]

        self.portfolio_values.append(portfolio_value)
        self.daily_returns.append(daily_return)

        # ポートフォリオ状態記録
        portfolio = events.Portfolio(
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=portfolio_value,
            daily_return=daily_return,
            cumulative_return=(portfolio_value - self.initial_capital)
            / self.initial_capital,
        )

        self.portfolio_history.append(portfolio)

    def _analyze_results(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> BacktestResults:
        """結果分析"""
        if not self.portfolio_values:
            raise ValueError("バックテストデータが不足しています")

        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # 年率リターン計算
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # ボラティリティ計算
        returns = np.array(self.daily_returns[1:])  # 最初の0を除く
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        # シャープレシオ計算（リスクフリーレート2%と仮定）
        risk_free_rate = 0.02
        sharpe_ratio = (
            (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        )

        # 最大ドローダウン計算
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdowns = (np.array(self.portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0

        # 取引統計
        total_trades = len(self.trades)
        profitable_trades = sum(
            1 for trade in self.trades if trade.get("realized_pnl", 0) > 0
        )
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        trade_returns = [
            trade.get("realized_pnl", 0) / self.initial_capital
            for trade in self.trades
            if "realized_pnl" in trade
        ]
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        best_trade = max(trade_returns) if trade_returns else 0
        worst_trade = min(trade_returns) if trade_returns else 0

        return events.BacktestResults(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            daily_returns=self.daily_returns,
            portfolio_values=self.portfolio_values,
            positions_history=[
                {
                    "date": i,
                    "cash": portfolio.cash,
                    "total_value": portfolio.total_value,
                    "positions": len(portfolio.positions),
                }
                for i, portfolio in enumerate(self.portfolio_history)
            ],
        )

    def generate_performance_report(self, results: BacktestResults) -> str:
        """パフォーマンスレポート生成"""
        report = []
        report.append("=" * 60)
        report.append("バックテスト結果レポート")
        report.append("=" * 60)

        report.append(f"期間: {results.start_date} ～ {results.end_date}")
        report.append(f"初期資本: {results.initial_capital:,.0f}円")
        report.append(f"最終価値: {results.final_value:,.0f}円")

        report.append("\n【リターン分析】")
        report.append(f"  総リターン: {results.total_return:.2%}")
        report.append(f"  年率リターン: {results.annualized_return:.2%}")
        report.append(f"  年率ボラティリティ: {results.volatility:.2%}")
        report.append(f"  シャープレシオ: {results.sharpe_ratio:.3f}")

        report.append("\n【リスク分析】")
        report.append(f"  最大ドローダウン: {results.max_drawdown:.2%}")

        report.append("\n【取引分析】")
        report.append(f"  総取引数: {results.total_trades}回")
        report.append(f"  勝率: {results.win_rate:.2%}")
        report.append(f"  平均取引リターン: {results.avg_trade_return:.4%}")
        report.append(f"  最良取引: {results.best_trade:.4%}")
        report.append(f"  最悪取引: {results.worst_trade:.4%}")

        return "\n".join(report)


if __name__ == "__main__":
    # テスト用サンプル戦略
    def simple_momentum_strategy(
        lookback_data: Dict[str, pd.DataFrame], current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """シンプルなモメンタム戦略"""
        signals = {}

        for symbol, data in lookback_data.items():
            if len(data) >= 20:
                # 20日リターン計算
                returns_20d = data["Close"].iloc[-1] / data["Close"].iloc[-20] - 1

                # モメンタムに基づく重み付け
                if returns_20d > 0.05:  # 5%以上上昇
                    signals[symbol] = 0.2  # 20%ウェイト
                elif returns_20d < -0.05:  # 5%以上下落
                    signals[symbol] = 0.0  # ポジションなし
                else:
                    signals[symbol] = 0.1  # 10%ウェイト

        # ウェイト正規化
        total_weight = sum(signals.values())
        if total_weight > 0:
            signals = {k: v / total_weight for k, v in signals.items()}

        return signals

    # テスト実行
    print("バックテストエンジンテスト")
    print("=" * 50)

    engine = BacktestEngine(initial_capital=1000000)

    # テスト用銘柄
    test_symbols = ["7203.T", "8306.T", "9984.T"]  # トヨタ、三菱UFJ、ソフトバンクG

    # 過去1年間のデータでテスト
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    historical_data = engine.load_historical_data(
        test_symbols, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )

    if historical_data:
        results = engine.execute_backtest(historical_data, simple_momentum_strategy)
        report = engine.generate_performance_report(results)
        print(report)
    else:
        print("データが取得できませんでした")
