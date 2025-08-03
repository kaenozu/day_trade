"""
バックテスト機能
売買戦略の過去データでの検証とパフォーマンス分析
機械学習統合、Walk-Forward分析、パラメータ最適化を含む
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd
from scipy import optimize

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
from .ensemble import EnsembleTradingStrategy, EnsembleStrategy, EnsembleVotingType

logger = get_context_logger(__name__, component="backtest")


class BacktestMode(Enum):
    """バックテストモード"""

    SINGLE_SYMBOL = "single_symbol"  # 単一銘柄
    PORTFOLIO = "portfolio"  # ポートフォリオ
    STRATEGY_COMPARISON = "comparison"  # 戦略比較
    WALK_FORWARD = "walk_forward"  # Walk-Forward分析
    MONTE_CARLO = "monte_carlo"  # モンテカルロシミュレーション
    PARAMETER_OPTIMIZATION = "optimization"  # パラメータ最適化
    ML_ENSEMBLE = "ml_ensemble"  # 機械学習アンサンブル


class OptimizationObjective(Enum):
    """最適化目標"""

    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"  # リターン/最大ドローダウン
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"


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

    # Walk-Forward分析設定
    walk_forward_window: int = 252  # 訓練期間（日数）
    walk_forward_step: int = 21  # ステップサイズ（日数）
    min_training_samples: int = 100  # 最小訓練サンプル数

    # 最適化設定
    optimization_objective: OptimizationObjective = OptimizationObjective.SHARPE_RATIO
    max_optimization_iterations: int = 100
    optimization_method: str = "differential_evolution"  # scipy.optimize手法

    # モンテカルロ設定
    monte_carlo_runs: int = 1000
    confidence_interval: float = 0.95

    # リスク管理設定
    max_drawdown_limit: Optional[float] = None  # 最大許容ドローダウン
    stop_loss_pct: Optional[float] = None  # ストップロス率
    take_profit_pct: Optional[float] = None  # テイクプロフィット率

    # 機械学習設定
    enable_ml_features: bool = False
    retrain_frequency: int = 50  # ML再訓練頻度（日数）
    ml_models_to_use: List[str] = None

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
            "walk_forward_window": self.walk_forward_window,
            "walk_forward_step": self.walk_forward_step,
            "min_training_samples": self.min_training_samples,
            "optimization_objective": self.optimization_objective.value,
            "max_optimization_iterations": self.max_optimization_iterations,
            "optimization_method": self.optimization_method,
            "monte_carlo_runs": self.monte_carlo_runs,
            "confidence_interval": self.confidence_interval,
            "max_drawdown_limit": self.max_drawdown_limit,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "enable_ml_features": self.enable_ml_features,
            "retrain_frequency": self.retrain_frequency,
            "ml_models_to_use": self.ml_models_to_use,
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
class WalkForwardResult:
    """Walk-Forward分析結果"""

    period_start: datetime
    period_end: datetime
    training_start: datetime
    training_end: datetime
    out_of_sample_return: float
    in_sample_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    parameters_used: Dict[str, Any]


@dataclass
class OptimizationResult:
    """パラメータ最適化結果"""

    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    iterations: int
    convergence_achieved: bool
    backtest_result: 'BacktestResult'


@dataclass
class MonteCarloResult:
    """モンテカルロシミュレーション結果"""

    mean_return: float
    std_return: float
    confidence_intervals: Dict[str, Tuple[float, float]]  # 信頼区間
    percentiles: Dict[str, float]  # パーセンタイル
    probability_of_loss: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    simulation_runs: int


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

    # 新しい分析結果
    walk_forward_results: Optional[List[WalkForwardResult]] = None
    optimization_result: Optional[OptimizationResult] = None
    monte_carlo_result: Optional[MonteCarloResult] = None

    # 追加統計
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    information_ratio: Optional[float] = None
    beta: Optional[float] = None
    alpha: Optional[float] = None

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
    """強化されたバックテストエンジン"""

    def __init__(
        self,
        stock_fetcher: Optional[StockFetcher] = None,
        signal_generator: Optional[TradingSignalGenerator] = None,
        ensemble_strategy: Optional[EnsembleTradingStrategy] = None,
        memory_efficient: bool = True,
        chunk_size: int = 1000,
    ):
        """
        Args:
            stock_fetcher: 株価データ取得インスタンス
            signal_generator: シグナル生成インスタンス
            ensemble_strategy: アンサンブル戦略インスタンス
            memory_efficient: メモリ効率モード（大きなデータセット用）
            chunk_size: データ処理のチャンクサイズ
        """
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.signal_generator = signal_generator or TradingSignalGenerator()
        self.ensemble_strategy = ensemble_strategy

        # パフォーマンス設定
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size

        # バックテスト状態
        self.current_capital = Decimal("0")
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[Tuple[datetime, Decimal]] = []

        # 新しい状態管理
        self.risk_metrics: Dict[str, float] = {}
        self.benchmark_data: Optional[pd.Series] = None
        self.optimization_cache: Dict[str, Any] = {}
        self.parameter_bounds: Dict[str, Tuple[float, float]] = {}

        # パフォーマンス追跡（メモリ効率モードでは制限）
        if self.memory_efficient:
            # メモリ効率モードでは詳細追跡を制限
            self.daily_metrics: List[Dict[str, float]] = []
            self.max_daily_metrics = 1000  # 最大保持数
        else:
            self.daily_metrics: List[Dict[str, float]] = []
            self.max_daily_metrics = None

        self.rebalance_dates: List[datetime] = []
        self.ml_retrain_dates: List[datetime] = []

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
            # シンボルリストが空の場合の早期リターン
            if not symbols:
                logger.warning("シンボルリストが空のため、履歴データ取得をスキップします")
                return historical_data

            # 進捗表示付きでデータ取得
            with progress_context(
                f"履歴データ取得 ({len(symbols)}銘柄)",
                total=len(symbols),
                progress_type=ProgressType.DETERMINATE,
            ) as progress:
                # Use ThreadPoolExecutor for parallel fetching
                # max_workersが0にならないように最小値を1に設定
                max_workers = max(min(len(symbols), 5), 1)
                with ThreadPoolExecutor(
                    max_workers=max_workers
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
            # シンボルリストが空の場合の早期リターン
            if not symbols:
                logger.warning("シンボルリストが空のため、履歴データ取得をスキップします")
                return historical_data

            # Use ThreadPoolExecutor for parallel fetching
            # max_workersが0にならないように最小値を1に設定
            max_workers = max(min(len(symbols), 5), 1)
            with ThreadPoolExecutor(
                max_workers=max_workers
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

                # パターン結果を簡略化 - 最新データポイントのみを抽出
                latest_patterns = self._extract_latest_pattern_data(
                    all_patterns_for_current_data
                )

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

        # 取引統計の計算（メモリ効率を考慮）
        if self.memory_efficient:
            profitable_trades, losing_trades, wins, losses, total_trades, win_rate, avg_win, avg_loss, profit_factor = self._calculate_trade_statistics_memory_efficient()
        else:
            profitable_trades, losing_trades, wins, losses, total_trades, win_rate, avg_win, avg_loss, profit_factor = self._calculate_trade_statistics_vectorized()

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

    def _calculate_trade_statistics_vectorized(self) -> Tuple[int, int, List[float], List[float], int, float, Decimal, Decimal, float]:
        """
        ベクトル化された取引統計計算
        パフォーマンス最適化版 - O(n²)からO(n log n)に改善
        """
        if not self.trades:
            return 0, 0, [], [], 0, 0.0, Decimal("0"), Decimal("0"), float("inf")

        # 取引をPandasのDataFrameに変換
        trades_data = []
        for i, trade in enumerate(self.trades):
            trades_data.append({
                'index': i,
                'symbol': trade.symbol,
                'action': trade.action.value,
                'price': float(trade.price),
                'quantity': float(trade.quantity),
                'commission': float(trade.commission),
                'timestamp': trade.timestamp
            })

        if not trades_data:
            return 0, 0, [], [], 0, 0.0, Decimal("0"), Decimal("0"), float("inf")

        trades_df = pd.DataFrame(trades_data)

        # 買い注文と売り注文を分離
        buy_trades = trades_df[trades_df['action'] == TradeType.BUY.value].copy()
        sell_trades = trades_df[trades_df['action'] == TradeType.SELL.value].copy()

        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return 0, 0, [], [], 0, 0.0, Decimal("0"), Decimal("0"), float("inf")

        wins = []
        losses = []

        # 効率的なマッチング処理（FIFO - First In, First Out）
        for symbol in trades_df['symbol'].unique():
            symbol_buys = buy_trades[buy_trades['symbol'] == symbol].sort_values('index')
            symbol_sells = sell_trades[sell_trades['symbol'] == symbol].sort_values('index')

            # マッチングアルゴリズム: 各売り注文に対して最も近い過去の買い注文を探す
            buy_queue = symbol_buys.copy()

            for _, sell_trade in symbol_sells.iterrows():
                # 売り注文より前の買い注文のみを対象
                available_buys = buy_queue[buy_queue['index'] < sell_trade['index']]

                if len(available_buys) > 0:
                    # 最も最近の買い注文を選択（LIFO - Last In, First Out）
                    buy_trade = available_buys.iloc[-1]

                    # 損益計算
                    pnl = (
                        (sell_trade['price'] - buy_trade['price']) * sell_trade['quantity']
                        - sell_trade['commission'] - buy_trade['commission']
                    )

                    if pnl > 0:
                        wins.append(pnl)
                    else:
                        losses.append(abs(pnl))

                    # 使用した買い注文を削除
                    buy_queue = buy_queue[buy_queue['index'] != buy_trade['index']]

        # 統計計算
        profitable_trades = len(wins)
        losing_trades = len(losses)
        total_trades = profitable_trades + losing_trades

        if total_trades == 0:
            return 0, 0, [], [], 0, 0.0, Decimal("0"), Decimal("0"), float("inf")

        win_rate = profitable_trades / total_trades
        avg_win = Decimal(str(np.mean(wins))) if wins else Decimal("0")
        avg_loss = Decimal(str(np.mean(losses))) if losses else Decimal("0")

        profit_factor = (
            float(avg_win * profitable_trades) / float(avg_loss * losing_trades)
            if losses and losing_trades > 0
            else float("inf")
        )

        return profitable_trades, losing_trades, wins, losses, total_trades, win_rate, avg_win, avg_loss, profit_factor

    def _calculate_trade_statistics_memory_efficient(self) -> Tuple[int, int, List[float], List[float], int, float, Decimal, Decimal, float]:
        """
        メモリ効率版の取引統計計算
        大きなデータセットでもメモリ使用量を抑える
        """
        if not self.trades:
            return 0, 0, [], [], 0, 0.0, Decimal("0"), Decimal("0"), float("inf")

        # メモリ効率のためのチャンク処理
        profitable_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        wins_sum = 0.0
        losses_sum = 0.0
        wins_count = 0
        losses_count = 0

        # チャンクごとに処理してメモリ使用量を制限
        for i in range(0, len(self.trades), self.chunk_size):
            chunk_end = min(i + self.chunk_size, len(self.trades))
            chunk_trades = self.trades[i:chunk_end]

            # チャンク内の取引をDataFrameに変換
            chunk_data = []
            for j, trade in enumerate(chunk_trades):
                chunk_data.append({
                    'index': i + j,
                    'symbol': trade.symbol,
                    'action': trade.action.value,
                    'price': float(trade.price),
                    'quantity': float(trade.quantity),
                    'commission': float(trade.commission),
                    'timestamp': trade.timestamp
                })

            if not chunk_data:
                continue

            trades_df = pd.DataFrame(chunk_data)

            # 買い注文と売り注文を分離
            buy_trades = trades_df[trades_df['action'] == TradeType.BUY.value].copy()
            sell_trades = trades_df[trades_df['action'] == TradeType.SELL.value].copy()

            if len(buy_trades) == 0 or len(sell_trades) == 0:
                continue

            # チャンク内でのマッチング処理
            for symbol in trades_df['symbol'].unique():
                symbol_buys = buy_trades[buy_trades['symbol'] == symbol].sort_values('index')
                symbol_sells = sell_trades[sell_trades['symbol'] == symbol].sort_values('index')

                buy_queue = symbol_buys.copy()

                for _, sell_trade in symbol_sells.iterrows():
                    available_buys = buy_queue[buy_queue['index'] < sell_trade['index']]

                    if len(available_buys) > 0:
                        buy_trade = available_buys.iloc[-1]

                        # 損益計算
                        pnl = (
                            (sell_trade['price'] - buy_trade['price']) * sell_trade['quantity']
                            - sell_trade['commission'] - buy_trade['commission']
                        )

                        total_pnl += pnl

                        if pnl > 0:
                            wins_sum += pnl
                            wins_count += 1
                        else:
                            losses_sum += abs(pnl)
                            losses_count += 1

                        buy_queue = buy_queue[buy_queue['index'] != buy_trade['index']]

            # チャンク処理後のメモリクリーンアップ
            del trades_df, buy_trades, sell_trades, chunk_data

        # 統計計算
        profitable_trades = wins_count
        losing_trades = losses_count
        total_trades = profitable_trades + losing_trades

        if total_trades == 0:
            return 0, 0, [], [], 0, 0.0, Decimal("0"), Decimal("0"), float("inf")

        win_rate = profitable_trades / total_trades
        avg_win = Decimal(str(wins_sum / wins_count)) if wins_count > 0 else Decimal("0")
        avg_loss = Decimal(str(losses_sum / losses_count)) if losses_count > 0 else Decimal("0")

        profit_factor = (
            wins_sum / losses_sum
            if losses_sum > 0
            else float("inf")
        )

        # メモリ効率モードでは詳細なリストは返さない
        return profitable_trades, losing_trades, [], [], total_trades, win_rate, avg_win, avg_loss, profit_factor

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

    def run_walk_forward_analysis(
        self,
        symbols: List[str],
        config: BacktestConfig,
        strategy_func: Optional[Callable] = None,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        show_progress: bool = True,
    ) -> BacktestResult:
        """Walk-Forward分析を実行"""
        logger.info("Walk-Forward分析を開始")

        try:
            # 全期間の履歴データを取得
            buffer_start = config.start_date - timedelta(days=config.walk_forward_window + 100)
            historical_data = self._fetch_historical_data_for_period(
                symbols, buffer_start, config.end_date, show_progress
            )

            if not historical_data:
                raise ValueError("Walk-Forward分析に必要な履歴データを取得できませんでした")

            # Walk-Forwardウィンドウの生成
            wf_periods = self._generate_walk_forward_periods(config)

            if not wf_periods:
                raise ValueError("Walk-Forward期間の生成に失敗しました")

            logger.info(f"Walk-Forward分析: {len(wf_periods)}期間")

            wf_results = []
            total_portfolio_values = []
            total_trades = []

            if show_progress:
                progress_desc = f"Walk-Forward分析 ({len(wf_periods)}期間)"
                with progress_context(
                    progress_desc, total=len(wf_periods), progress_type=ProgressType.DETERMINATE
                ) as progress:

                    for i, (train_start, train_end, test_start, test_end) in enumerate(wf_periods):
                        try:
                            # 訓練期間とテスト期間のデータを分割
                            train_data = self._extract_period_data(historical_data, train_start, train_end)
                            test_data = self._extract_period_data(historical_data, test_start, test_end)

                            if not train_data or not test_data:
                                logger.warning(f"期間 {i+1} のデータが不足しています。スキップします。")
                                progress.update(1)
                                continue

                            # パラメータ最適化（訓練期間）
                            best_params = {}
                            if parameter_ranges:
                                best_params = self._optimize_parameters_for_period(
                                    symbols, train_data, parameter_ranges, config
                                )

                            # 訓練期間でのパフォーマンス評価
                            train_config = self._create_period_config(config, train_start, train_end)
                            in_sample_result = self._run_period_backtest(
                                symbols, train_data, train_config, strategy_func, best_params
                            )

                            # テスト期間でのOut-of-Sample評価
                            test_config = self._create_period_config(config, test_start, test_end)
                            out_sample_result = self._run_period_backtest(
                                symbols, test_data, test_config, strategy_func, best_params
                            )

                            # Walk-Forward結果を記録
                            wf_result = WalkForwardResult(
                                period_start=test_start,
                                period_end=test_end,
                                training_start=train_start,
                                training_end=train_end,
                                out_of_sample_return=float(out_sample_result.total_return),
                                in_sample_return=float(in_sample_result.total_return),
                                sharpe_ratio=out_sample_result.sharpe_ratio,
                                max_drawdown=out_sample_result.max_drawdown,
                                total_trades=out_sample_result.total_trades,
                                parameters_used=best_params,
                            )
                            wf_results.append(wf_result)

                            # 全体のポートフォリオ価値と取引を累積
                            total_portfolio_values.extend(out_sample_result.portfolio_value.to_list())
                            total_trades.extend(out_sample_result.trades)

                            progress.set_description(
                                f"期間 {i+1}/{len(wf_periods)} - OOS Return: {wf_result.out_of_sample_return:.2%}"
                            )
                            progress.update(1)

                        except Exception as e:
                            logger.error(f"Walk-Forward期間 {i+1} でエラー: {e}")
                            progress.update(1)
                            continue

            else:
                for i, (train_start, train_end, test_start, test_end) in enumerate(wf_periods):
                    try:
                        train_data = self._extract_period_data(historical_data, train_start, train_end)
                        test_data = self._extract_period_data(historical_data, test_start, test_end)

                        if not train_data or not test_data:
                            continue

                        best_params = {}
                        if parameter_ranges:
                            best_params = self._optimize_parameters_for_period(
                                symbols, train_data, parameter_ranges, config
                            )

                        train_config = self._create_period_config(config, train_start, train_end)
                        in_sample_result = self._run_period_backtest(
                            symbols, train_data, train_config, strategy_func, best_params
                        )

                        test_config = self._create_period_config(config, test_start, test_end)
                        out_sample_result = self._run_period_backtest(
                            symbols, test_data, test_config, strategy_func, best_params
                        )

                        wf_result = WalkForwardResult(
                            period_start=test_start,
                            period_end=test_end,
                            training_start=train_start,
                            training_end=train_end,
                            out_of_sample_return=float(out_sample_result.total_return),
                            in_sample_return=float(in_sample_result.total_return),
                            sharpe_ratio=out_sample_result.sharpe_ratio,
                            max_drawdown=out_sample_result.max_drawdown,
                            total_trades=out_sample_result.total_trades,
                            parameters_used=best_params,
                        )
                        wf_results.append(wf_result)

                        total_portfolio_values.extend(out_sample_result.portfolio_value.to_list())
                        total_trades.extend(out_sample_result.trades)

                    except Exception as e:
                        logger.error(f"Walk-Forward期間 {i+1} でエラー: {e}")
                        continue

            # 全体の統合結果を計算
            integrated_result = self._calculate_walk_forward_integrated_result(
                config, wf_results, total_portfolio_values, total_trades
            )
            integrated_result.walk_forward_results = wf_results

            logger.info(f"Walk-Forward分析完了: {len(wf_results)}期間, 平均OOS Return: {np.mean([r.out_of_sample_return for r in wf_results]):.2%}")
            return integrated_result

        except Exception as e:
            logger.error(f"Walk-Forward分析エラー: {e}")
            raise

    def run_parameter_optimization(
        self,
        symbols: List[str],
        config: BacktestConfig,
        strategy_func: Callable,
        parameter_ranges: Dict[str, Tuple[float, float]],
        show_progress: bool = True,
    ) -> BacktestResult:
        """パラメータ最適化を実行"""
        logger.info(f"パラメータ最適化を開始: {list(parameter_ranges.keys())}")

        try:
            # 履歴データを取得
            historical_data = self._fetch_historical_data(symbols, config, show_progress)
            if not historical_data:
                raise ValueError("最適化に必要な履歴データを取得できませんでした")

            # 最適化の目的関数を定義
            def objective_function(params_array):
                try:
                    # パラメータ配列を辞書に変換
                    param_dict = {}
                    for i, (param_name, _) in enumerate(parameter_ranges.items()):
                        param_dict[param_name] = params_array[i]

                    # このパラメータセットでバックテストを実行
                    temp_result = self._run_parameter_backtest(
                        symbols, historical_data, config, strategy_func, param_dict
                    )

                    # 目的関数値を計算（最大化問題として扱う）
                    if config.optimization_objective == OptimizationObjective.TOTAL_RETURN:
                        return -float(temp_result.total_return)  # 負数にして最小化問題に変換
                    elif config.optimization_objective == OptimizationObjective.SHARPE_RATIO:
                        return -temp_result.sharpe_ratio
                    elif config.optimization_objective == OptimizationObjective.CALMAR_RATIO:
                        calmar = float(temp_result.annualized_return) / abs(temp_result.max_drawdown) if temp_result.max_drawdown != 0 else 0
                        return -calmar
                    elif config.optimization_objective == OptimizationObjective.PROFIT_FACTOR:
                        return -temp_result.profit_factor
                    elif config.optimization_objective == OptimizationObjective.WIN_RATE:
                        return -temp_result.win_rate
                    else:
                        return -temp_result.sharpe_ratio

                except Exception as e:
                    logger.warning(f"最適化中のパラメータ評価でエラー: {e}")
                    return float('inf')  # エラー時は最悪値を返す

            # パラメータ境界を設定
            bounds = list(parameter_ranges.values())

            # 最適化履歴を記録
            optimization_history = []
            iteration_count = [0]  # リストを使ってスコープ問題を回避

            def callback(xk, convergence=None):
                iteration_count[0] += 1
                param_dict = {}
                for i, (param_name, _) in enumerate(parameter_ranges.items()):
                    param_dict[param_name] = xk[i]

                score = objective_function(xk)
                optimization_history.append({
                    "iteration": iteration_count[0],
                    "parameters": param_dict.copy(),
                    "score": -score,  # 元の値に戻す
                })

                if show_progress and iteration_count[0] % 10 == 0:
                    logger.info(f"最適化進捗: {iteration_count[0]}/{config.max_optimization_iterations} - 現在のスコア: {-score:.4f}")

            # scipy.optimizeを使用した最適化
            if config.optimization_method == "differential_evolution":
                result = optimize.differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=config.max_optimization_iterations,
                    callback=callback,
                    seed=42,  # 再現性のため
                    workers=1,  # 並列化は無効（コールバック問題を避けるため）
                )
            elif config.optimization_method == "minimize":
                # 初期値を範囲の中央に設定
                x0 = [(b[0] + b[1]) / 2 for b in bounds]
                result = optimize.minimize(
                    objective_function,
                    x0,
                    bounds=bounds,
                    method="L-BFGS-B",
                    options={"maxiter": config.max_optimization_iterations},
                    callback=callback,
                )
            else:
                raise ValueError(f"サポートされていない最適化手法: {config.optimization_method}")

            # 最適パラメータを取得
            best_params = {}
            for i, (param_name, _) in enumerate(parameter_ranges.items()):
                best_params[param_name] = result.x[i]

            logger.info(f"最適化完了: {best_params}")

            # 最適パラメータでの最終バックテスト
            final_result = self._run_parameter_backtest(
                symbols, historical_data, config, strategy_func, best_params
            )

            # 最適化結果を作成
            optimization_result = OptimizationResult(
                best_parameters=best_params,
                best_score=-result.fun,  # 元の値に戻す
                optimization_history=optimization_history,
                iterations=iteration_count[0],
                convergence_achieved=result.success if hasattr(result, 'success') else False,
                backtest_result=final_result,
            )

            final_result.optimization_result = optimization_result
            return final_result

        except Exception as e:
            logger.error(f"パラメータ最適化エラー: {e}")
            raise

    def run_monte_carlo_simulation(
        self,
        base_result: BacktestResult,
        config: BacktestConfig,
        show_progress: bool = True,
    ) -> MonteCarloResult:
        """モンテカルロシミュレーションを実行"""
        logger.info(f"モンテカルロシミュレーション開始: {config.monte_carlo_runs}回")

        try:
            if base_result.daily_returns.empty:
                raise ValueError("ベースとなるリターンデータがありません")

            # ベースとなる日次リターンの統計値
            base_returns = base_result.daily_returns.dropna()
            mean_return = base_returns.mean()
            std_return = base_returns.std()

            # モンテカルロシミュレーション実行
            simulation_results = []

            if show_progress:
                with progress_context(
                    f"モンテカルロシミュレーション ({config.monte_carlo_runs}回)",
                    total=config.monte_carlo_runs,
                    progress_type=ProgressType.DETERMINATE,
                ) as progress:

                    for i in range(config.monte_carlo_runs):
                        # ランダムリターンを生成（正規分布）
                        random_returns = np.random.normal(
                            mean_return, std_return, len(base_returns)
                        )

                        # 累積リターンを計算
                        cumulative_return = (1 + pd.Series(random_returns)).prod() - 1
                        simulation_results.append(cumulative_return)

                        if i % 100 == 0:
                            progress.set_description(f"シミュレーション実行中: {i+1}/{config.monte_carlo_runs}")
                        progress.update(1)
            else:
                for i in range(config.monte_carlo_runs):
                    random_returns = np.random.normal(
                        mean_return, std_return, len(base_returns)
                    )
                    cumulative_return = (1 + pd.Series(random_returns)).prod() - 1
                    simulation_results.append(cumulative_return)

            # 統計値を計算
            simulation_array = np.array(simulation_results)

            mean_sim_return = np.mean(simulation_array)
            std_sim_return = np.std(simulation_array)

            # 信頼区間
            alpha = 1 - config.confidence_interval
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            confidence_intervals = {
                f"{config.confidence_interval*100:.0f}%": (
                    np.percentile(simulation_array, lower_percentile),
                    np.percentile(simulation_array, upper_percentile)
                )
            }

            # パーセンタイル
            percentiles = {
                "5%": np.percentile(simulation_array, 5),
                "25%": np.percentile(simulation_array, 25),
                "50%": np.percentile(simulation_array, 50),
                "75%": np.percentile(simulation_array, 75),
                "95%": np.percentile(simulation_array, 95),
            }

            # リスク指標
            probability_of_loss = np.sum(simulation_array < 0) / len(simulation_array)
            var_95 = np.percentile(simulation_array, 5)
            cvar_95 = np.mean(simulation_array[simulation_array <= var_95])

            monte_carlo_result = MonteCarloResult(
                mean_return=mean_sim_return,
                std_return=std_sim_return,
                confidence_intervals=confidence_intervals,
                percentiles=percentiles,
                probability_of_loss=probability_of_loss,
                var_95=var_95,
                cvar_95=cvar_95,
                simulation_runs=config.monte_carlo_runs,
            )

            logger.info(f"モンテカルロシミュレーション完了: 平均リターン {mean_sim_return:.2%}, 損失確率 {probability_of_loss:.1%}")
            return monte_carlo_result

        except Exception as e:
            logger.error(f"モンテカルロシミュレーションエラー: {e}")
            raise

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

    def _extract_latest_pattern_data(self, patterns_data: Dict[str, Any]) -> Dict[str, Any]:
        """パターンデータから最新データポイントを抽出する簡略化メソッド"""
        latest_patterns = {}

        for key, value in patterns_data.items():
            try:
                if isinstance(value, pd.DataFrame):
                    # DataFrameの場合は最新行を抽出
                    if not value.empty:
                        latest_patterns[key] = value.iloc[[-1]].copy()
                    else:
                        latest_patterns[key] = pd.DataFrame()
                elif isinstance(value, dict):
                    # ネストされた辞書の場合は再帰的に処理
                    nested_dict = {}
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, (pd.Series, pd.DataFrame)):
                            if hasattr(nested_value, 'iloc') and len(nested_value) > 0:
                                nested_dict[nested_key] = nested_value.iloc[[-1]].copy()
                            else:
                                nested_dict[nested_key] = nested_value
                        else:
                            nested_dict[nested_key] = nested_value
                    latest_patterns[key] = nested_dict
                else:
                    # その他の型はそのまま保持
                    latest_patterns[key] = value

            except Exception as e:
                logger.debug(f"パターンデータ抽出エラー ({key}): {e}")
                # エラー時はスキップ
                continue

        return latest_patterns

    def _calculate_realized_pnl(self) -> List[Decimal]:
        """改良版実現損益計算 - ポジション追跡による正確な損益計算"""
        realized_pnl = []
        position_tracker = {}  # symbol -> [買い注文のリスト]

        for trade in self.trades:
            symbol = trade.symbol

            if symbol not in position_tracker:
                position_tracker[symbol] = []

            if trade.action == TradeType.BUY:
                # 買い注文を追加
                position_tracker[symbol].append(trade)

            elif trade.action == TradeType.SELL:
                # 売り注文：FIFO方式で対応する買い注文を処理
                remaining_quantity = trade.quantity
                sell_proceeds = trade.price * trade.quantity - trade.commission
                total_cost = Decimal("0")

                while remaining_quantity > 0 and position_tracker[symbol]:
                    buy_trade = position_tracker[symbol][0]

                    if buy_trade.quantity <= remaining_quantity:
                        # 買いポジションを完全に決済
                        cost = (buy_trade.price * buy_trade.quantity +
                               buy_trade.commission)
                        total_cost += cost
                        remaining_quantity -= buy_trade.quantity
                        position_tracker[symbol].pop(0)
                    else:
                        # 買いポジションを部分決済
                        partial_quantity = remaining_quantity
                        partial_cost = (buy_trade.price * partial_quantity +
                                      buy_trade.commission *
                                      (partial_quantity / buy_trade.quantity))
                        total_cost += partial_cost

                        # 残りの買いポジションを更新
                        buy_trade.quantity -= partial_quantity
                        buy_trade.commission *= (buy_trade.quantity /
                                               (buy_trade.quantity + partial_quantity))
                        remaining_quantity = 0

                # 実現損益を計算
                if total_cost > 0:
                    pnl = sell_proceeds - total_cost
                    realized_pnl.append(pnl)

        return realized_pnl

    def _calculate_trade_statistics_vectorized(self) -> Tuple[int, int, List[Decimal], List[Decimal], int, float, Decimal, Decimal, Any]:
        """取引統計情報をベクタ化計算で算出（テスト互換性用）"""
        if not self.trades:
            return (0, 0, [], [], 0, 0.0, Decimal("0"), Decimal("0"), float("inf"))

        realized_pnl = self._calculate_realized_pnl()
        wins = [pnl for pnl in realized_pnl if pnl > 0]
        losses = [abs(pnl) for pnl in realized_pnl if pnl < 0]  # 正の値に変換

        profitable_trades = len(wins)
        losing_trades = len(losses)
        total_trades = len(realized_pnl)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_win = sum(wins) / len(wins) if wins else Decimal("0")
        avg_loss = sum(losses) / len(losses) if losses else Decimal("0")
        if losses:
            # テスト互換性のためfloatに変換
            profit_factor = float(sum(wins) / sum(losses))
        else:
            profit_factor = float("inf")  # No losses (either no trades or only wins)

        return (profitable_trades, losing_trades, wins, losses, total_trades, win_rate, avg_win, avg_loss, profit_factor)



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
                    symbol=symbol,
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
                    symbol=symbol,
                )
            )

    return signals


if __name__ == "__main__":
    # ロギング設定は logging_config.py で一元管理
    from ..utils.logging_config import setup_logging
    setup_logging()

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
