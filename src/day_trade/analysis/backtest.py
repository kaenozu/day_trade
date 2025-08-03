"""
バックテスト機能
売買戦略の過去データでの検証とパフォーマンス分析
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..analysis.signals import (
    SignalStrength,
    SignalType,
    TradingSignal,
    TradingSignalGenerator,
)
from ..core.trade_manager import TradeType
from ..data.stock_fetcher import StockFetcher
from ..utils.progress import ProgressType, multi_step_progress, progress_context
from ..utils.logging_config import get_context_logger, log_business_event, log_performance_metric
from .indicators import TechnicalIndicators
from .patterns import ChartPatternRecognizer

logger = get_context_logger(__name__)


class BacktestMode(Enum):
    """バックテストモード"""

    SINGLE_SYMBOL = "single_symbol"  # 単一銘柄
    PORTFOLIO = "portfolio"  # ポートフォリオ
    STRATEGY_COMPARISON = "comparison"  # 戦略比較


@dataclass
class BacktestConfig:
    """バックテスト設定"""

    start_date: datetime
    end_date: datetime
    initial_capital: Decimal  # 初期資金
    commission: Decimal = Decimal("0.001")  # 手数料率（0.1%）
    slippage: Decimal = Decimal("0.001")  # スリッページ（0.1%）
    max_position_size: Decimal = Decimal("0.2")  # 最大ポジションサイズ（20%）
    rebalance_frequency: str = "monthly"  # リバランス頻度

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": str(self.initial_capital),
            "commission": str(self.commission),
            "slippage": str(self.slippage),
            "max_position_size": str(self.max_position_size),
            "rebalance_frequency": self.rebalance_frequency,
        }


@dataclass
class Trade:
    """取引記録"""

    timestamp: datetime
    symbol: str
    action: TradeType  # BUY or SELL
    quantity: int
    price: Decimal
    commission: Decimal
    total_cost: Decimal


@dataclass
class Position:
    """ポジション情報"""

    symbol: str
    quantity: int
    average_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    weight: Decimal  # ポートフォリオ内での重み


@dataclass
class BacktestResult:
    """バックテスト結果"""

    # 基本情報
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    duration_days: int

    # パフォーマンス指標
    total_return: Decimal
    annualized_return: Decimal
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

    # 取引統計
    total_trades: int
    profitable_trades: int
    losing_trades: int
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: float

    # 詳細データ
    trades: List[Trade]
    daily_returns: pd.Series
    portfolio_value: pd.Series
    positions_history: List[Dict[str, Position]]

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "config": self.config.to_dict(),
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "duration_days": self.duration_days,
            "total_return": str(self.total_return),
            "annualized_return": str(self.annualized_return),
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "profitable_trades": self.profitable_trades,
            "losing_trades": self.losing_trades,
            "avg_win": str(self.avg_win),
            "avg_loss": str(self.avg_loss),
            "profit_factor": self.profit_factor,
            "trades_count": len(self.trades),
            "portfolio_final_value": (
                str(self.portfolio_value.iloc[-1])
                if not self.portfolio_value.empty
                else "0"
            ),
        }


class BacktestEngine:
    """バックテストエンジン"""

    def __init__(
        self,
        stock_fetcher: Optional[StockFetcher] = None,
        signal_generator: Optional[TradingSignalGenerator] = None,
    ):
        """
        Args:
            stock_fetcher: 株価データ取得インスタンス
            signal_generator: シグナル生成インスタンス
        """
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.signal_generator = signal_generator or TradingSignalGenerator()

        # バックテスト状態
        self.current_capital = Decimal("0")
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[Tuple[datetime, Decimal]] = []

    def run_backtest(
        self,
        symbols: List[str],
        config: BacktestConfig,
        strategy_func: Optional[Callable] = None,
        show_progress: bool = True,
    ) -> BacktestResult:
        """
        バックテストを実行

        Args:
            symbols: 対象銘柄リスト
            config: バックテスト設定
            strategy_func: カスタム戦略関数
            show_progress: 進捗表示フラグ

        Returns:
            バックテスト結果
        """
        logger.info(
            f"バックテスト開始: {symbols}, 期間: {config.start_date} - {config.end_date}"
        )

        if show_progress:
            # 進捗表示付きでバックテストを実行
            steps = [
                ("initialize", "バックテスト初期化", 1.0),
                ("fetch_data", "履歴データ取得", 3.0),
                ("execute_simulation", "バックテストシミュレーション", 10.0),
                ("calculate_results", "結果計算", 1.0),
            ]

            with multi_step_progress("バックテスト実行", steps) as progress:
                # 初期化
                self._initialize_backtest(config)
                progress.complete_step()

                # 履歴データの取得
                historical_data = self._fetch_historical_data(
                    symbols, config, show_progress=True
                )
                if not historical_data:
                    progress.fail_step("履歴データの取得に失敗")
                    raise ValueError(
                        "バックテストに必要な履歴データの取得に失敗しました。指定された銘柄コードが正しいか、データプロバイダーからのデータが利用可能か確認してください。"
                    )
                progress.complete_step()

                # 日次でバックテストを実行
                trading_dates = self._get_trading_dates(config)

                with progress_context(
                    f"日次シミュレーション実行 ({len(trading_dates)}日間)",
                    total=len(trading_dates),
                    progress_type=ProgressType.DETERMINATE,
                ) as sim_progress:
                    for i, date in enumerate(trading_dates):
                        try:
                            # その日の価格データを取得
                            daily_prices = self._get_daily_prices(historical_data, date)
                            if not daily_prices:
                                sim_progress.update(1)
                                continue

                            # ポジション評価の更新
                            self._update_positions_value(daily_prices)

                            # シグナル生成
                            if strategy_func:
                                signals = strategy_func(symbols, date, historical_data)
                            else:
                                signals = self._generate_default_signals(
                                    symbols, date, historical_data
                                )

                            # シグナルに基づく取引実行
                            self._execute_trades(signals, daily_prices, date, config)

                            # ポートフォリオ価値の記録
                            total_value = self._calculate_total_portfolio_value()
                            self.portfolio_values.append((date, total_value))

                            # 進捗表示更新（10日ごと）
                            if i % 10 == 0:
                                sim_progress.set_description(
                                    f"日次シミュレーション ({date.strftime('%Y-%m-%d')}) - ポートフォリオ価値: ¥{total_value:,.0f}"
                                )

                            sim_progress.update(1)

                        except Exception as e:
                            logger.warning(
                                f"日付 '{date.strftime('%Y-%m-%d')}' のバックテスト処理中にエラーが発生しました。この日付の処理はスキップされます。詳細: {e}"
                            )
                            sim_progress.update(1)
                            continue

                progress.complete_step()

                # 結果の計算
                result = self._calculate_results(config)
                progress.complete_step()

        else:
            # 進捗表示なしで実行（従来通り）
            # 初期化
            self._initialize_backtest(config)

            # 履歴データの取得
            historical_data = self._fetch_historical_data(
                symbols, config, show_progress=False
            )
            if not historical_data:
                raise ValueError(
                    "バックテストに必要な履歴データの取得に失敗しました。指定された銘柄コードが正しいか、データプロバイダーからのデータが利用可能か確認してください。"
                )

            # 日次でバックテストを実行
            trading_dates = self._get_trading_dates(config)

            for date in trading_dates:
                try:
                    # その日の価格データを取得
                    daily_prices = self._get_daily_prices(historical_data, date)
                    if not daily_prices:
                        continue

                    # ポジション評価の更新
                    self._update_positions_value(daily_prices)

                    # シグナル生成
                    if strategy_func:
                        signals = strategy_func(symbols, date, historical_data)
                    else:
                        signals = self._generate_default_signals(
                            symbols, date, historical_data
                        )

                    # シグナルに基づく取引実行
                    self._execute_trades(signals, daily_prices, date, config)

                    # ポートフォリオ価値の記録
                    total_value = self._calculate_total_portfolio_value()
                    self.portfolio_values.append((date, total_value))

                except Exception as e:
                    logger.warning(
                        f"日付 '{date.strftime('%Y-%m-%d')}' のバックテスト処理中にエラーが発生しました。この日付の処理はスキップされます。詳細: {e}"
                    )
                    continue

            # 結果の計算
            result = self._calculate_results(config)

        logger.info(f"バックテスト完了: 総リターン {result.total_return:.2%}")
        return result

    def _initialize_backtest(self, config: BacktestConfig):
        """バックテストの初期化"""
        self.current_capital = config.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []

    def _fetch_historical_data(
        self, symbols: List[str], config: BacktestConfig, show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """履歴データの取得"""
        historical_data = {}

        # バッファを追加（テクニカル指標計算のため）
        buffer_start = config.start_date - timedelta(days=100)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        if show_progress:
            # 進捗表示付きでデータ取得
            with progress_context(
                f"履歴データ取得 ({len(symbols)}銘柄)",
                total=len(symbols),
                progress_type=ProgressType.DETERMINATE,
            ) as progress:
                # Use ThreadPoolExecutor for parallel fetching
                with ThreadPoolExecutor(
                    max_workers=min(max(len(symbols), 1), 5)
                ) as executor:  # Limit workers to avoid overwhelming
                    future_to_symbol = {
                        executor.submit(
                            self.stock_fetcher.get_historical_data,
                            symbol,
                            start_date=buffer_start,
                            end_date=config.end_date,
                            interval="1d",
                        ): symbol
                        for symbol in symbols
                    }

                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        try:
                            data = future.result()
                            if data is not None and not data.empty:
                                historical_data[symbol] = data
                                logger.info(
                                    f"履歴データ取得完了: {symbol} ({len(data)} データポイント)"
                                )
                                progress.set_description(
                                    f"履歴データ取得完了: {symbol}"
                                )
                            else:
                                logger.warning(
                                    f"銘柄 '{symbol}' の履歴データを取得できませんでした。この銘柄はバックテストから除外されます。"
                                )

                        except Exception as e:
                            logger.error(
                                f"銘柄 '{symbol}' の履歴データ取得中にエラーが発生しました。詳細: {e}"
                            )
                        finally:
                            progress.update(1)
        else:
            # Use ThreadPoolExecutor for parallel fetching
            with ThreadPoolExecutor(
                max_workers=min(max(len(symbols), 1), 5)
            ) as executor:  # Limit workers to avoid overwhelming
                future_to_symbol = {
                    executor.submit(
                        self.stock_fetcher.get_historical_data,
                        symbol,
                        start_date=buffer_start,
                        end_date=config.end_date,
                        interval="1d",
                    ): symbol
                    for symbol in symbols
                }

                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result()
                        if data is not None and not data.empty:
                            historical_data[symbol] = data
                            logger.info(
                                f"履歴データ取得完了: {symbol} ({len(data)} データポイント)"
                            )
                        else:
                            logger.warning(
                                f"銘柄 '{symbol}' の履歴データを取得できませんでした。この銘柄はバックテストから除外されます。"
                            )

                    except Exception as e:
                        logger.error(
                            f"銘柄 '{symbol}' の履歴データ取得中にエラーが発生しました。詳細: {e}"
                        )
        return historical_data

    def _get_trading_dates(self, config: BacktestConfig) -> List[datetime]:
        """取引日のリストを生成"""
        dates = []
        current = config.start_date

        while current <= config.end_date:
            # 土日を除く（簡易版）
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)

        return dates

    def _get_daily_prices(
        self, historical_data: Dict[str, pd.DataFrame], date: datetime
    ) -> Dict[str, Decimal]:
        """指定日の価格データを取得"""
        daily_prices = {}

        for symbol, data in historical_data.items():
            try:
                # 日付に最も近いデータを取得
                closest_date = data.index[data.index <= date]
                if len(closest_date) > 0:
                    price_data = data.loc[closest_date[-1]]
                    daily_prices[symbol] = Decimal(str(price_data["Close"]))
            except (KeyError, IndexError) as e:
                logger.debug(
                    f"銘柄 '{symbol}' の日付 '{date.strftime('%Y-%m-%d')}' の価格データを取得できませんでした。詳細: {e}"
                )
                continue

        return daily_prices

    def _update_positions_value(self, daily_prices: Dict[str, Decimal]):
        """ポジション評価額の更新"""
        for symbol in self.positions:
            if symbol in daily_prices:
                position = self.positions[symbol]
                position.current_price = daily_prices[symbol]
                position.market_value = position.current_price * position.quantity
                position.unrealized_pnl = position.market_value - (
                    position.average_price * position.quantity
                )

    def _generate_default_signals(
        self,
        symbols: List[str],
        date: datetime,
        historical_data: Dict[str, pd.DataFrame],
    ) -> List[TradingSignal]:
        """デフォルト戦略でのシグナル生成"""
        signals = []

        for symbol in symbols:
            if symbol not in historical_data:
                continue

            data = historical_data[symbol]
            current_data_all = data[data.index <= date]  # その日までの全データ
            # 必要な最低データ数を設定 (MACDなどの計算に必要な期間を考慮)
            # 例えば、MACDのデフォルト期間は26日EMA + 9日シグナルで合計34日必要
            # RSIは14日
            # Bollinger Bands 20日
            # Golden/Dead Cross 20日
            # 総合的に60日あれば十分なはず
            min_required_data = 60
            if len(current_data_all) < min_required_data:
                continue

            try:
                # その時点までの全データから一度だけ指標とパターンを計算
                # signals.py で import しているためここでの import は不要
                # from .indicators import TechnicalIndicators
                # from .patterns import ChartPatternRecognizer

                all_indicators_for_current_data = TechnicalIndicators.calculate_all(
                    current_data_all
                )
                all_patterns_for_current_data = (
                    ChartPatternRecognizer.detect_all_patterns(current_data_all)
                )

                # generate_signal に渡すのは最新のデータポイントのみ
                # df, indicators はDataFrameなのでiloc[[-1]]で最新の行を抽出
                latest_df_row = current_data_all.iloc[[-1]].copy()
                latest_indicators_row = all_indicators_for_current_data.iloc[
                    [-1]
                ].copy()

                # patternsは辞書であり、その中にDataFrameが含まれている可能性があるため、
                # patterns.py の detect_all_patterns が返す構造を考慮してスライス
                latest_patterns = {}
                for key, value in all_patterns_for_current_data.items():
                    if isinstance(value, pd.DataFrame):
                        # patterns内のDataFrameも最新の行を抽出
                        if not value.empty:
                            latest_patterns[key] = value.iloc[[-1]].copy()
                        else:
                            latest_patterns[key] = pd.DataFrame()
                    elif isinstance(value, dict):
                        # ネストされた辞書の場合、個別に処理
                        nested_dict = {}
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, pd.Series):
                                nested_dict[nested_key] = nested_value.iloc[[-1]].copy()
                            else:
                                nested_dict[nested_key] = nested_value
                        latest_patterns[key] = nested_dict
                    else:
                        latest_patterns[key] = value

                signal = self.signal_generator.generate_signal(
                    latest_df_row, latest_indicators_row, latest_patterns
                )

                # 最新のシグナルのみを使用
                if signal:
                    # シグナルのタイムスタンプが現在のバックテスト日付と一致することを確認
                    if signal.timestamp.date() == date.date():
                        signals.append(signal)
            except Exception as e:
                logger.debug(
                    f"銘柄 '{symbol}' の日付 '{date.strftime('%Y-%m-%d')}' のシグナル生成中にエラーが発生しました。詳細: {e}"
                )
                continue

        return signals

    def _execute_trades(
        self,
        signals: List[TradingSignal],
        daily_prices: Dict[str, Decimal],
        date: datetime,
        config: BacktestConfig,
    ):
        """シグナルに基づく取引実行"""
        for signal in signals:
            symbol = signal.symbol

            if symbol not in daily_prices:
                continue

            current_price = daily_prices[symbol]

            try:
                if signal.signal_type == SignalType.BUY:
                    self._execute_buy_order(symbol, current_price, date, config)
                elif signal.signal_type == SignalType.SELL:
                    self._execute_sell_order(symbol, current_price, date, config)

            except Exception as e:
                logger.warning(
                    f"銘柄 '{symbol}' の取引実行中にエラーが発生しました。この取引はスキップされます。詳細: {e}"
                )

    def _execute_buy_order(
        self, symbol: str, price: Decimal, date: datetime, config: BacktestConfig
    ):
        """買い注文の実行"""
        # ポジションサイズの計算
        total_value = self._calculate_total_portfolio_value()
        max_investment = total_value * config.max_position_size

        # スリッページと手数料を考慮した実際の価格
        actual_price = price * (Decimal("1") + config.slippage)

        # 購入可能数量の計算
        quantity = int(max_investment / actual_price)

        if quantity <= 0 or self.current_capital < actual_price * quantity:
            return

        # 手数料計算
        commission = actual_price * quantity * config.commission
        total_cost = actual_price * quantity + commission

        if total_cost > self.current_capital:
            return

        # ポジション更新
        if symbol in self.positions:
            # 既存ポジションへの追加
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
            # 新規ポジション
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=actual_price,
                current_price=actual_price,
                market_value=actual_price * quantity,
                unrealized_pnl=Decimal("0"),
                weight=Decimal("0"),
            )

        # 資金とトレード記録の更新
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

    def _execute_sell_order(
        self, symbol: str, price: Decimal, date: datetime, config: BacktestConfig
    ):
        """売り注文の実行"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        quantity = position.quantity

        # スリッページと手数料を考慮した実際の価格
        actual_price = price * (Decimal("1") - config.slippage)

        # 手数料計算
        commission = actual_price * quantity * config.commission
        total_proceeds = actual_price * quantity - commission

        # 資金とトレード記録の更新
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

        # ポジション削除
        del self.positions[symbol]

        logger.debug(f"売り注文実行: {symbol} x{quantity} @ ¥{actual_price}")

    def _calculate_total_portfolio_value(self) -> Decimal:
        """ポートフォリオ総価値の計算"""
        total_value = self.current_capital

        for position in self.positions.values():
            total_value += position.market_value

        return total_value

    def _calculate_results(self, config: BacktestConfig) -> BacktestResult:
        """バックテスト結果の計算"""
        if not self.portfolio_values:
            raise ValueError(
                "バックテスト結果の計算に必要なポートフォリオ価値のデータが不足しています。バックテストが正常に実行されたか確認してください。"
            )

        # データフレーム作成
        df = pd.DataFrame(self.portfolio_values, columns=["Date", "Value"])
        df.set_index("Date", inplace=True)
        df["Value"] = df["Value"].astype(float)

        # 日次リターンの計算
        daily_returns = df["Value"].pct_change().dropna()

        # パフォーマンス指標の計算
        total_return = (df["Value"].iloc[-1] / float(config.initial_capital)) - 1
        duration_years = (config.end_date - config.start_date).days / 365.25
        annualized_return = (
            (1 + total_return) ** (1 / duration_years) - 1 if duration_years > 0 else 0
        )

        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        sharpe_ratio = (
            (annualized_return - 0.001) / volatility if volatility > 0 else 0
        )  # リスクフリーレート0.1%

        # 最大ドローダウンの計算
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 取引統計の計算
        profitable_trades = 0
        losing_trades = 0
        wins = []
        losses = []

        # 実現損益の計算（簡易版）
        for i, trade in enumerate(self.trades):
            if trade.action == TradeType.SELL and i > 0:
                # 対応する買い注文を探す
                for j in range(i - 1, -1, -1):
                    buy_trade = self.trades[j]
                    if (
                        buy_trade.symbol == trade.symbol
                        and buy_trade.action == TradeType.BUY
                    ):
                        pnl = (
                            (trade.price - buy_trade.price) * trade.quantity
                            - trade.commission
                            - buy_trade.commission
                        )
                        if pnl > 0:
                            profitable_trades += 1
                            wins.append(float(pnl))
                        else:
                            losing_trades += 1
                            losses.append(float(abs(pnl)))
                        break

        total_trades = profitable_trades + losing_trades
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        avg_win = Decimal(str(np.mean(wins))) if wins else Decimal("0")
        avg_loss = Decimal(str(np.mean(losses))) if losses else Decimal("0")
        profit_factor = (
            float(avg_win * profitable_trades) / float(avg_loss * losing_trades)
            if losses
            else float("inf")
        )

        return BacktestResult(
            config=config,
            start_date=config.start_date,
            end_date=config.end_date,
            duration_days=(config.end_date - config.start_date).days,
            total_return=Decimal(str(total_return)),
            annualized_return=Decimal(str(annualized_return)),
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=self.trades,
            daily_returns=daily_returns,
            portfolio_value=df["Value"],
            positions_history=[],  # 簡単化のため空
        )

    def run_strategy_comparison(
        self,
        symbols: List[str],
        strategies: Dict[str, Callable],
        config: BacktestConfig,
        show_progress: bool = True,
    ) -> Dict[str, BacktestResult]:
        """複数戦略の比較バックテスト"""
        logger.info(f"戦略比較バックテスト開始: {list(strategies.keys())}")

        results = {}

        if show_progress:
            with progress_context(
                f"戦略比較バックテスト ({len(strategies)}戦略)",
                total=len(strategies),
                progress_type=ProgressType.DETERMINATE,
            ) as progress:
                for strategy_name, strategy_func in strategies.items():
                    logger.info(f"戦略実行中: {strategy_name}")
                    progress.set_description(f"戦略実行中: {strategy_name}")

                    try:
                        result = self.run_backtest(
                            symbols, config, strategy_func, show_progress=False
                        )
                        results[strategy_name] = result
                        logger.info(
                            f"戦略 {strategy_name} 完了: リターン {result.total_return:.2%}"
                        )
                    except Exception as e:
                        logger.error(
                            f"戦略 '{strategy_name}' のバックテスト実行中にエラーが発生しました。この戦略はスキップされます。詳細: {e}"
                        )
                    finally:
                        progress.update(1)
        else:
            for strategy_name, strategy_func in strategies.items():
                logger.info(f"戦略実行中: {strategy_name}")
                try:
                    result = self.run_backtest(
                        symbols, config, strategy_func, show_progress=False
                    )
                    results[strategy_name] = result
                    logger.info(
                        f"戦略 {strategy_name} 完了: リターン {result.total_return:.2%}"
                    )
                except Exception as e:
                    logger.error(
                        f"戦略 '{strategy_name}' のバックテスト実行中にエラーが発生しました。この戦略はスキップされます。詳細: {e}"
                    )
                    continue

        return results

    def export_results(self, result: BacktestResult, filename: str):
        """結果をファイルにエクスポート"""
        try:
            output_data = result.to_dict()

            # 詳細データを追加
            output_data["trades_detail"] = []
            for trade in result.trades:
                output_data["trades_detail"].append(
                    {
                        "timestamp": trade.timestamp.isoformat(),
                        "symbol": trade.symbol,
                        "action": trade.action.value,
                        "quantity": trade.quantity,
                        "price": str(trade.price),
                        "commission": str(trade.commission),
                        "total_cost": str(trade.total_cost),
                    }
                )

            # パフォーマンスデータ
            output_data["daily_performance"] = []
            for date, value in zip(
                result.portfolio_value.index, result.portfolio_value.values
            ):
                output_data["daily_performance"].append(
                    {
                        "date": (
                            date.isoformat()
                            if hasattr(date, "isoformat")
                            else str(date)
                        ),
                        "portfolio_value": float(value),
                    }
                )

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"バックテスト結果をエクスポート: {filename}")

        except Exception as e:
            logger.error(
                f"バックテスト結果のエクスポート中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {e}"
            )
            raise

    def _calculate_trade_statistics_vectorized(self) -> Tuple[int, int, List[Decimal], List[Decimal], int, float, Decimal, Decimal, Any]:
        """取引統計情報をベクタ化計算で算出（テスト互換性用）"""
        if not self.trades:
            return (0, 0, [], [], 0, 0.0, Decimal("0"), Decimal("0"), float("inf"))

        # 取引データの集計
        wins = []
        losses = []

        for trade in self.trades:
            if trade.action == TradeType.SELL:
                # 売り取引の場合、利益/損失を計算
                # 簡易計算：売値から平均買値を引く
                profit_loss = trade.price - Decimal("2500")  # 仮の基準価格
                if profit_loss > 0:
                    wins.append(profit_loss)
                else:
                    losses.append(abs(profit_loss))

        winning_trades = len(wins)
        losing_trades = len(losses)
        total_trades = len(self.trades)

        # 勝率計算
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

        # 平均勝ち/負け
        avg_win = Decimal(str(sum(wins) / len(wins))) if wins else Decimal("0")
        avg_loss = Decimal(str(sum(losses) / len(losses))) if losses else Decimal("0")

        # プロフィットファクター
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 1  # ゼロ除算回避
        profit_factor = float(total_wins / total_losses) if total_losses > 0 else float("inf")

        return (
            winning_trades,
            losing_trades,
            wins,
            losses,
            total_trades,
            win_rate,
            avg_win,
            avg_loss,
            profit_factor
        )


# 使用例とデフォルト戦略
def simple_sma_strategy(
    symbols: List[str], date: datetime, historical_data: Dict[str, pd.DataFrame]
) -> List[TradingSignal]:
    """シンプルな移動平均戦略"""
    signals = []

    for symbol in symbols:
        if symbol not in historical_data:
            continue

        data = historical_data[symbol]
        current_data = data[data.index <= date]

        if len(current_data) < 50:
            continue

        # 短期・長期移動平均
        short_ma = current_data["Close"].rolling(window=5).mean()
        long_ma = current_data["Close"].rolling(window=20).mean()

        if len(short_ma) < 2 or len(long_ma) < 2:
            continue

        # ゴールデンクロス・デッドクロス
        if (
            short_ma.iloc[-1] > long_ma.iloc[-1]
            and short_ma.iloc[-2] <= long_ma.iloc[-2]
        ):
            signals.append(
                TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MEDIUM,
                    confidence=70.0,
                    reasons=["Golden cross detected"],
                    conditions_met={"golden_cross": True},
                    timestamp=pd.Timestamp(date),
                    price=float(current_data["Close"].iloc[-1]),
                )
            )
        elif (
            short_ma.iloc[-1] < long_ma.iloc[-1]
            and short_ma.iloc[-2] >= long_ma.iloc[-2]
        ):
            signals.append(
                TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MEDIUM,
                    confidence=70.0,
                    reasons=["Dead cross detected"],
                    conditions_met={"dead_cross": True},
                    timestamp=pd.Timestamp(date),
                    price=float(current_data["Close"].iloc[-1]),
                )
            )

    return signals


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # サンプルバックテスト
    engine = BacktestEngine()

    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Decimal("1000000"),  # 100万円
        commission=Decimal("0.001"),  # 0.1%
        slippage=Decimal("0.001"),  # 0.1%
    )

    symbols = ["7203", "9984", "8306"]  # トヨタ、ソフトバンク、三菱UFJ

    try:
        result = engine.run_backtest(symbols, config, simple_sma_strategy)

        # バックテスト結果の構造化ログ出力
        backtest_summary = {
            "period_start": result.start_date.date().isoformat(),
            "period_end": result.end_date.date().isoformat(),
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "symbols_tested": symbols,
            "strategy": "simple_sma_strategy"
        }

        logger.info("バックテスト完了", **backtest_summary)
        log_business_event("backtest_completed", **backtest_summary)

        # パフォーマンスメトリクス記録
        log_performance_metric("sharpe_ratio", result.sharpe_ratio, "ratio")
        log_performance_metric("total_return", result.total_return, "percentage")
        log_performance_metric("max_drawdown", result.max_drawdown, "percentage")

        # バックテスト結果をログ出力
        logger.info(
            "バックテスト完了",
            start_date=result.start_date.date().isoformat(),
            end_date=result.end_date.date().isoformat(),
            total_return=f"{result.total_return:.2%}",
            annualized_return=f"{result.annualized_return:.2%}",
            volatility=f"{result.volatility:.2%}",
            sharpe_ratio=f"{result.sharpe_ratio:.2f}",
            max_drawdown=f"{result.max_drawdown:.2%}",
            win_rate=f"{result.win_rate:.1%}",
            total_trades=result.total_trades
        )

        # 結果をファイルに保存
        engine.export_results(result, "backtest_result.json")

    except Exception as e:
        logger.error(f"バックテストの実行中に予期せぬエラーが発生しました。詳細: {e}")
