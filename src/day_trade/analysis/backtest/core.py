"""
バックテストエンジンのコア機能

基本的なバックテスト実行機能とポートフォリオ管理を提供
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.day_trade.analysis.ensemble import EnsembleTradingStrategy
from src.day_trade.analysis.signals import TradingSignalGenerator
from src.day_trade.data.stock_fetcher import StockFetcher
from src.day_trade.utils.enhanced_error_handler import get_default_error_handler
from src.day_trade.utils.logging_config import (
    get_context_logger,
    get_performance_logger,
)
from src.day_trade.utils.progress import (
    ProgressType,
    multi_step_progress,
    progress_context,
)

from .types import (
    BacktestConfig,
    BacktestResult,
    Position,
    Trade,
)

logger = get_context_logger(__name__, component="backtest_core")
performance_logger = get_performance_logger(__name__)
error_handler = get_default_error_handler()


class BacktestEngine:
    """バックテストエンジンのコア機能"""

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
                        "バックテストに必要な履歴データの取得に失敗しました。"
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
                            self._execute_daily_backtest(
                                date, historical_data, symbols, config, strategy_func
                            )

                            # 進捗表示更新（10日ごと）
                            if i % 10 == 0:
                                total_value = self._calculate_total_portfolio_value()
                                sim_progress.set_description(
                                    f"日次シミュレーション ({date.strftime('%Y-%m-%d')}) - ポートフォリオ価値: ¥{total_value:,.0f}"
                                )

                            sim_progress.update(1)

                        except Exception as e:
                            logger.warning(f"日付 {date} の処理中にエラー: {e}")
                            sim_progress.update(1)
                            continue

                progress.complete_step()

                # 結果の計算
                result = self._calculate_results(config)
                progress.complete_step()

        else:
            # 進捗表示なしで実行
            self._initialize_backtest(config)

            historical_data = self._fetch_historical_data(
                symbols, config, show_progress=False
            )
            if not historical_data:
                raise ValueError("履歴データの取得に失敗しました。")

            trading_dates = self._get_trading_dates(config)

            for date in trading_dates:
                try:
                    self._execute_daily_backtest(
                        date, historical_data, symbols, config, strategy_func
                    )
                except Exception as e:
                    logger.warning(f"日付 {date} の処理中にエラー: {e}")
                    continue

            result = self._calculate_results(config)

        logger.info("バックテスト完了")
        return result

    def _initialize_backtest(self, config: BacktestConfig):
        """バックテストの初期化"""
        self.current_capital = config.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.daily_metrics = []
        self.risk_metrics = {}

        logger.info(f"初期資金: {config.initial_capital}")

    def _fetch_historical_data(
        self, symbols: List[str], config: BacktestConfig, show_progress: bool = False
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """履歴データを取得"""
        historical_data = {}

        for symbol in symbols:
            try:
                data = self.stock_fetcher.get_historical_data(
                    symbol, config.start_date, config.end_date
                )
                if data is not None and not data.empty:
                    historical_data[symbol] = data
                else:
                    logger.warning(f"銘柄 {symbol} のデータが取得できませんでした")
            except Exception as e:
                logger.error(f"銘柄 {symbol} のデータ取得でエラー: {e}")

        return historical_data if historical_data else None

    def _get_trading_dates(self, config: BacktestConfig) -> List[datetime]:
        """取引可能日のリストを生成"""
        dates = []
        current_date = config.start_date

        while current_date <= config.end_date:
            # 平日のみ（簡易判定）
            if current_date.weekday() < 5:
                dates.append(current_date)
            current_date += pd.Timedelta(days=1)

        return dates

    def _execute_daily_backtest(
        self,
        date: datetime,
        historical_data: Dict[str, pd.DataFrame],
        symbols: List[str],
        config: BacktestConfig,
        strategy_func: Optional[Callable],
    ):
        """日次バックテスト実行"""
        # その日の価格データを取得
        daily_prices = self._get_daily_prices(historical_data, date)
        if not daily_prices:
            return

        # ポジション評価の更新
        self._update_positions_value(daily_prices)

        # シグナル生成
        if strategy_func:
            signals = strategy_func(symbols, date, historical_data)
        else:
            signals = self._generate_default_signals(symbols, date, historical_data)

        # シグナルに基づく取引実行
        self._execute_trades(signals, daily_prices, date, config)

        # ポートフォリオ価値の記録
        total_value = self._calculate_total_portfolio_value()
        self.portfolio_values.append((date, total_value))

    def _get_daily_prices(
        self, historical_data: Dict[str, pd.DataFrame], date: datetime
    ) -> Dict[str, Decimal]:
        """指定日の各銘柄の価格を取得"""
        daily_prices = {}

        for symbol, data in historical_data.items():
            try:
                # 日付に最も近い価格を取得
                if date.strftime("%Y-%m-%d") in data.index.strftime("%Y-%m-%d"):
                    price_data = data[
                        data.index.strftime("%Y-%m-%d") == date.strftime("%Y-%m-%d")
                    ]
                    if not price_data.empty:
                        # 終値を使用
                        daily_prices[symbol] = Decimal(str(price_data["Close"].iloc[0]))
            except Exception as e:
                logger.warning(f"銘柄 {symbol} の {date} の価格取得でエラー: {e}")

        return daily_prices

    def _update_positions_value(self, daily_prices: Dict[str, Decimal]):
        """ポジションの評価額を更新"""
        for symbol, position in self.positions.items():
            if symbol in daily_prices:
                position.current_price = daily_prices[symbol]
                position.market_value = position.current_price * Decimal(
                    str(position.quantity)
                )
                position.unrealized_pnl = position.market_value - (
                    position.average_price * Decimal(str(position.quantity))
                )

    def _generate_default_signals(
        self,
        symbols: List[str],
        date: datetime,
        historical_data: Dict[str, pd.DataFrame],
    ) -> List:
        """デフォルトシグナルを生成（簡単なMAクロス）"""
        signals = []

        # 簡単なデフォルト戦略を実装
        # 実際の実装では、より複雑なシグナル生成ロジックを使用

        return signals

    def _execute_trades(
        self,
        signals: List,
        daily_prices: Dict[str, Decimal],
        date: datetime,
        config: BacktestConfig,
    ):
        """取引を実行"""
        for _signal in signals:
            try:
                # シグナルに基づく取引実行ロジック
                # 実際の実装では、より複雑な取引ロジックを使用
                pass
            except Exception as e:
                logger.error(f"取引実行でエラー: {e}")

    def _calculate_total_portfolio_value(self) -> Decimal:
        """総ポートフォリオ価値を計算"""
        total_market_value = sum(pos.market_value for pos in self.positions.values())
        return self.current_capital + total_market_value

    def _calculate_results(self, config: BacktestConfig) -> BacktestResult:
        """バックテスト結果を計算"""
        if not self.portfolio_values:
            raise ValueError("バックテストが実行されていません")

        # 基本統計の計算
        initial_value = config.initial_capital
        final_value = self.portfolio_values[-1][1]

        total_return = float((final_value - initial_value) / initial_value)

        # 日次リターンの計算
        daily_returns = []
        for i in range(1, len(self.portfolio_values)):
            prev_value = self.portfolio_values[i - 1][1]
            curr_value = self.portfolio_values[i][1]
            daily_return = float((curr_value - prev_value) / prev_value)
            daily_returns.append(daily_return)

        # 基本指標の計算
        annualized_return = (
            total_return * (252 / len(daily_returns)) if daily_returns else 0.0
        )
        volatility = (
            float(pd.Series(daily_returns).std() * (252**0.5)) if daily_returns else 0.0
        )
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0

        # その他の指標
        winning_trades = len([t for t in self.trades if t.total_cost > 0])  # 簡易計算
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        return BacktestResult(
            config=config,
            start_date=config.start_date,
            end_date=config.end_date,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=0.0,  # 簡易実装
            calmar_ratio=0.0,  # 簡易実装
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            win_rate=win_rate,
            profit_factor=1.0,  # 簡易実装
            value_at_risk=0.0,  # 簡易実装
            conditional_var=0.0,  # 簡易実装
            trades=self.trades,
            positions=list(self.positions.values()),
            daily_returns=daily_returns,
            portfolio_value_history=[v[1] for v in self.portfolio_values],
            drawdown_history=[0.0] * len(daily_returns),  # 簡易実装
        )
