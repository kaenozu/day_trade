"""
バックテスト機能
売買戦略の過去データでの検証とパフォーマンス分析
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from ..data.stock_fetcher import StockFetcher
from ..analysis.signals import (
    TradingSignalGenerator,
    TradingSignal,
    SignalType,
    SignalStrength,
)
from ..core.trade_manager import TradeType

logger = logging.getLogger(__name__)


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
    ) -> BacktestResult:
        """
        バックテストを実行

        Args:
            symbols: 対象銘柄リスト
            config: バックテスト設定
            strategy_func: カスタム戦略関数

        Returns:
            バックテスト結果
        """
        logger.info(
            f"バックテスト開始: {symbols}, 期間: {config.start_date} - {config.end_date}"
        )

        # 初期化
        self._initialize_backtest(config)

        # 履歴データの取得
        historical_data = self._fetch_historical_data(symbols, config)
        if not historical_data:
            raise ValueError("バックテストに必要な履歴データの取得に失敗しました。指定された銘柄コードが正しいか、データプロバイダーからのデータが利用可能か確認してください。")

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
                logger.warning(f"日付 '{date.strftime('%Y-%m-%d')}' のバックテスト処理中にエラーが発生しました。この日付の処理はスキップされます。詳細: {e}")
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
        self, symbols: List[str], config: BacktestConfig
    ) -> Dict[str, pd.DataFrame]:
        """履歴データの取得"""
        historical_data = {}

        # バッファを追加（テクニカル指標計算のため）
        buffer_start = config.start_date - timedelta(days=100)
<<<<<<< HEAD

        for symbol in symbols:
            try:
                data = self.stock_fetcher.get_historical_data(
                    symbol,
                    start_date=buffer_start,
                    end_date=config.end_date,
                    interval="1d",
                )

                if data is not None and not data.empty:
                    historical_data[symbol] = data
                    logger.info(
                        f"履歴データ取得完了: {symbol} ({len(data)} データポイント)"
                    )
                else:
                    logger.warning(f"銘柄 '{symbol}' の履歴データを取得できませんでした。この銘柄はバックテストから除外されます。")

            except Exception as e:
                logger.error(f"銘柄 '{symbol}' の履歴データ取得中にエラーが発生しました。詳細: {e}")

=======
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=min(len(symbols), 5)) as executor: # Limit workers to avoid overwhelming
            future_to_symbol = {
                executor.submit(
                    self.stock_fetcher.get_historical_data,
                    symbol,
                    start_date=buffer_start,
                    end_date=config.end_date,
                    interval="1d"
                ): symbol for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        historical_data[symbol] = data
                        logger.info(f"履歴データ取得完了: {symbol} ({len(data)} データポイント)")
                    else:
                        logger.warning(f"履歴データの取得に失敗: {symbol}")
                        
                except Exception as e:
                    logger.error(f"履歴データ取得エラー ({symbol}): {e}")
        
>>>>>>> origin/main
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
                logger.debug(f"銘柄 '{symbol}' の日付 '{date.strftime('%Y-%m-%d')}' の価格データを取得できませんでした。詳細: {e}")
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
<<<<<<< HEAD
            current_data = data[data.index <= date]

            if len(current_data) < 20:  # 最低20日のデータが必要
=======
            current_data_all = data[data.index <= date] # その日までの全データ
            
            # 必要な最低データ数を設定 (MACDなどの計算に必要な期間を考慮)
            # 例えば、MACDのデフォルト期間は26日EMA + 9日シグナルで合計34日必要
            # RSIは14日
            # Bollinger Bands 20日
            # Golden/Dead Cross 20日
            # 総合的に60日あれば十分なはず
            min_required_data = 60
            if len(current_data_all) < min_required_data:
>>>>>>> origin/main
                continue

            try:
<<<<<<< HEAD
                # シンプルなテクニカル戦略
                symbol_signals = self.signal_generator.generate_signals(
                    symbol, current_data
                )

                # 最新のシグナルのみを使用
                if symbol_signals:
                    latest_signal = symbol_signals[-1]
                    if latest_signal.timestamp.date() == date.date():
                        signals.append(latest_signal)

=======
                # その時点までの全データから一度だけ指標とパターンを計算
                # signals.py で import しているためここでの import は不要
                # from .indicators import TechnicalIndicators
                # from .patterns import ChartPatternRecognizer
                
                all_indicators_for_current_data = TechnicalIndicators.calculate_all(current_data_all)
                all_patterns_for_current_data = ChartPatternRecognizer.detect_all_patterns(current_data_all)

                # generate_signal に渡すのは最新のデータポイントのみ
                # df, indicators はDataFrameなのでiloc[[-1]]で最新の行を抽出
                latest_df_row = current_data_all.iloc[[-1]].copy()
                latest_indicators_row = all_indicators_for_current_data.iloc[[-1]].copy()
                
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
                    latest_df_row,
                    latest_indicators_row,
                    latest_patterns
                )
                
                # 最新のシグナルのみを使用
                if signal:
                    # シグナルのタイムスタンプが現在のバックテスト日付と一致することを確認
                    if signal.timestamp.date() == date.date():
                        signals.append(signal)
                        
>>>>>>> origin/main
            except Exception as e:
                logger.debug(f"銘柄 '{symbol}' の日付 '{date.strftime('%Y-%m-%d')}' のシグナル生成中にエラーが発生しました。詳細: {e}")
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
                logger.warning(f"銘柄 '{symbol}' の取引実行中にエラーが発生しました。この取引はスキップされます。詳細: {e}")

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
            raise ValueError("バックテスト結果の計算に必要なポートフォリオ価値のデータが不足しています。バックテストが正常に実行されたか確認してください。")

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
    ) -> Dict[str, BacktestResult]:
        """複数戦略の比較バックテスト"""
        logger.info(f"戦略比較バックテスト開始: {list(strategies.keys())}")

        results = {}

        for strategy_name, strategy_func in strategies.items():
            logger.info(f"戦略実行中: {strategy_name}")
            try:
                result = self.run_backtest(symbols, config, strategy_func)
                results[strategy_name] = result
                logger.info(
                    f"戦略 {strategy_name} 完了: リターン {result.total_return:.2%}"
                )
            except Exception as e:
                logger.error(f"戦略 '{strategy_name}' のバックテスト実行中にエラーが発生しました。この戦略はスキップされます。詳細: {e}")
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
            logger.error(f"バックテスト結果のエクスポート中にエラーが発生しました。ファイルパスと書き込み権限を確認してください。詳細: {e}")
            raise


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

        print("=== バックテスト結果 ===")
        print(f"期間: {result.start_date.date()} - {result.end_date.date()}")
        print(f"総リターン: {result.total_return:.2%}")
        print(f"年率リターン: {result.annualized_return:.2%}")
        print(f"ボラティリティ: {result.volatility:.2%}")
        print(f"シャープレシオ: {result.sharpe_ratio:.2f}")
        print(f"最大ドローダウン: {result.max_drawdown:.2%}")
        print(f"勝率: {result.win_rate:.1%}")
        print(f"総取引数: {result.total_trades}")

        # 結果をファイルに保存
        engine.export_results(result, "backtest_result.json")

    except Exception as e:
        logger.error(f"バックテストの実行中に予期せぬエラーが発生しました。詳細: {e}")
