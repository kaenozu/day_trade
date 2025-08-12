"""
高度なバックテストエンジン

現実的な取引コスト、スリッページ、流動性制約を考慮した
包括的なバックテスト環境とウォークフォワード最適化。
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


class OrderType(Enum):
    """注文タイプ"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """注文状態"""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class TradingCosts:
    """取引コスト設定"""

    commission_rate: float = 0.001  # 手数料率 (0.1%)
    min_commission: float = 0.0  # 最小手数料
    max_commission: float = float("inf")  # 最大手数料
    bid_ask_spread_rate: float = 0.001  # ビッドアスクスプレッド率
    slippage_rate: float = 0.0005  # スリッページ率
    market_impact_rate: float = 0.0002  # マーケットインパクト率


@dataclass
class Order:
    """注文情報"""

    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # "buy" or "sell"
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    """ポジション情報"""

    symbol: str
    quantity: int = 0
    average_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""

    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0

    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade_duration: float = 0.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    total_commission: float = 0.0
    total_slippage: float = 0.0


class AdvancedBacktestEngine:
    """高度なバックテストエンジン"""

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        trading_costs: Optional[TradingCosts] = None,
        position_sizing: str = "fixed",  # "fixed", "percent", "volatility"
        max_position_size: float = 0.1,  # ポートフォリオの10%まで
        enable_slippage: bool = True,
        enable_market_impact: bool = True,
        realistic_execution: bool = True,
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trading_costs = trading_costs or TradingCosts()
        self.position_sizing = position_sizing
        self.max_position_size = max_position_size
        self.enable_slippage = enable_slippage
        self.enable_market_impact = enable_market_impact
        self.realistic_execution = realistic_execution

        # 取引状態
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trade_history: List[Dict] = []
        self.portfolio_history: List[Dict] = []

        # パフォーマンス追跡
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = []
        self.drawdown_series: List[float] = []

        # リスク管理
        self.max_daily_loss_limit: Optional[float] = None
        self.max_portfolio_heat: float = 0.02  # 2%

        logger.info(
            "高度バックテストエンジン初期化",
            section="backtest_init",
            initial_capital=initial_capital,
            realistic_execution=realistic_execution,
        )

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_signals: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> PerformanceMetrics:
        """
        バックテスト実行

        Args:
            data: OHLCV データ
            strategy_signals: 戦略シグナル (columns: signal, confidence, price_target)
            start_date: 開始日
            end_date: 終了日

        Returns:
            パフォーマンス指標
        """
        logger.info(
            "バックテスト開始",
            section="backtest_execution",
            data_range=f"{data.index[0]} to {data.index[-1]}",
            signals_count=len(strategy_signals),
        )

        # データ範囲設定
        if start_date:
            data = data[data.index >= start_date]
            strategy_signals = strategy_signals[strategy_signals.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            strategy_signals = strategy_signals[strategy_signals.index <= end_date]

        # バックテスト初期化
        self._reset_backtest()

        # 日次処理
        for current_date, row in data.iterrows():
            try:
                # 1. 既存注文の処理
                self._process_pending_orders(current_date, row)

                # 2. ポジション更新
                self._update_positions(current_date, row)

                # 3. 新規シグナル処理
                if current_date in strategy_signals.index:
                    signal_row = strategy_signals.loc[current_date]
                    self._process_strategy_signal(current_date, row, signal_row)

                # 4. リスク管理チェック
                self._apply_risk_management(current_date, row)

                # 5. ポートフォリオ記録
                self._record_portfolio_state(current_date, row)

            except Exception as e:
                logger.warning(
                    f"バックテスト処理エラー: {current_date}",
                    section="backtest_execution",
                    error=str(e),
                )

        # パフォーマンス計算
        performance = self._calculate_performance_metrics()

        logger.info(
            "バックテスト完了",
            section="backtest_execution",
            total_return=performance.total_return,
            sharpe_ratio=performance.sharpe_ratio,
            max_drawdown=performance.max_drawdown,
            total_trades=performance.total_trades,
        )

        return performance

    def _reset_backtest(self):
        """バックテスト状態リセット"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trade_history = []
        self.portfolio_history = []
        self.daily_returns = []
        self.equity_curve = [self.initial_capital]
        self.drawdown_series = []

    def _process_pending_orders(self, current_date: datetime, market_data: pd.Series):
        """待機中注文の処理"""
        filled_orders = []

        for order in self.orders:
            if order.status != OrderStatus.PENDING:
                continue

            # 注文実行判定
            if self._should_fill_order(order, market_data):
                filled_price = self._calculate_fill_price(order, market_data)
                self._execute_order(order, filled_price, current_date)
                filled_orders.append(order)

        # 約定済み注文を除去
        self.orders = [o for o in self.orders if o not in filled_orders]

    def _should_fill_order(self, order: Order, market_data: pd.Series) -> bool:
        """注文約定判定"""
        if order.order_type == OrderType.MARKET:
            return True

        elif order.order_type == OrderType.LIMIT:
            if (
                order.side == "buy"
                and market_data["Low"] <= order.price
                or order.side == "sell"
                and market_data["High"] >= order.price
            ):
                return True

        elif order.order_type == OrderType.STOP and (
            (order.side == "buy" and market_data["High"] >= order.stop_price)
            or (order.side == "sell" and market_data["Low"] <= order.stop_price)
        ):
            return True

        return False

    def _calculate_fill_price(self, order: Order, market_data: pd.Series) -> float:
        """約定価格計算"""
        base_price = market_data["Close"]

        if order.order_type == OrderType.MARKET:
            # マーケット注文：現在価格 + スプレッド + スリッページ
            spread_impact = self.trading_costs.bid_ask_spread_rate / 2
            if order.side == "buy":
                base_price *= 1 + spread_impact
            else:
                base_price *= 1 - spread_impact

        elif order.order_type == OrderType.LIMIT:
            base_price = order.price

        elif order.order_type == OrderType.STOP:
            base_price = order.stop_price

        # スリッページ適用
        if self.enable_slippage:
            slippage = np.random.normal(0, self.trading_costs.slippage_rate)
            if order.side == "buy":
                base_price *= 1 + abs(slippage)
            else:
                base_price *= 1 - abs(slippage)

        # マーケットインパクト適用
        if self.enable_market_impact:
            volume_ratio = order.quantity / market_data.get("Volume", 1000000)
            impact = volume_ratio * self.trading_costs.market_impact_rate
            if order.side == "buy":
                base_price *= 1 + impact
            else:
                base_price *= 1 - impact

        return base_price

    def _execute_order(self, order: Order, fill_price: float, execution_time: datetime):
        """注文執行"""
        # 手数料計算
        commission = max(
            self.trading_costs.min_commission,
            min(
                self.trading_costs.max_commission,
                order.quantity * fill_price * self.trading_costs.commission_rate,
            ),
        )

        # 注文更新
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.commission = commission

        # ポジション更新
        self._update_position_from_order(order)

        # 取引記録
        trade_record = {
            "timestamp": execution_time,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": fill_price,
            "commission": commission,
            "slippage": abs(fill_price - (order.price or fill_price)),
            "order_type": order.order_type.value,
        }
        self.trade_history.append(trade_record)

        logger.debug(
            f"注文執行: {order.side} {order.quantity} {order.symbol}",
            section="order_execution",
            fill_price=fill_price,
            commission=commission,
        )

    def _update_position_from_order(self, order: Order):
        """注文からポジション更新"""
        symbol = order.symbol

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        position = self.positions[symbol]

        if order.side == "buy":
            # 買い注文
            total_cost = (
                position.quantity * position.average_price
                + order.quantity * order.filled_price
            )
            total_quantity = position.quantity + order.quantity

            if total_quantity > 0:
                position.average_price = total_cost / total_quantity
                position.quantity = total_quantity

        else:
            # 売り注文
            if position.quantity >= order.quantity:
                # 実現損益計算
                realized_pnl = (
                    order.filled_price - position.average_price
                ) * order.quantity - order.commission
                position.realized_pnl += realized_pnl
                position.quantity -= order.quantity

                if position.quantity == 0:
                    position.average_price = 0.0
            else:
                logger.warning(
                    f"ショートポジション発生: {symbol}",
                    section="position_management",
                    current_quantity=position.quantity,
                    sell_quantity=order.quantity,
                )

        # 資本更新
        if order.side == "buy":
            self.current_capital -= (
                order.quantity * order.filled_price + order.commission
            )
        else:
            self.current_capital += (
                order.quantity * order.filled_price - order.commission
            )

    def _update_positions(self, current_date: datetime, market_data: pd.Series):
        """ポジション時価更新"""
        symbol = market_data.name if hasattr(market_data, "name") else "UNKNOWN"

        if symbol in self.positions:
            position = self.positions[symbol]
            current_price = market_data["Close"]

            position.market_value = position.quantity * current_price
            position.unrealized_pnl = (
                current_price - position.average_price
            ) * position.quantity
            position.last_updated = current_date

    def _process_strategy_signal(
        self, current_date: datetime, market_data: pd.Series, signal_data: pd.Series
    ):
        """戦略シグナル処理"""
        symbol = market_data.name if hasattr(market_data, "name") else "DEFAULT"
        signal = signal_data.get("signal", "hold")
        confidence = signal_data.get("confidence", 0.0)

        if signal in ["buy", "sell"] and confidence > 50.0:
            # ポジションサイズ計算
            position_size = self._calculate_position_size(
                symbol, market_data["Close"], confidence
            )

            if position_size > 0:
                # 注文生成
                order = Order(
                    order_id=f"{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side=signal,
                    quantity=position_size,
                    timestamp=current_date,
                )

                self.orders.append(order)

                logger.debug(
                    f"シグナル注文生成: {signal} {position_size} {symbol}",
                    section="signal_processing",
                    confidence=confidence,
                )

    def _calculate_position_size(
        self, symbol: str, price: float, confidence: float
    ) -> int:
        """ポジションサイズ計算"""
        if self.position_sizing == "fixed":
            # 固定金額
            target_value = self.initial_capital * self.max_position_size
            return int(target_value / price)

        elif self.position_sizing == "percent":
            # ポートフォリオ比率
            current_portfolio_value = self._get_portfolio_value()
            target_value = current_portfolio_value * self.max_position_size
            return int(target_value / price)

        elif self.position_sizing == "volatility":
            # ボラティリティ調整
            # 簡易実装：信頼度に基づく調整
            base_target = self._get_portfolio_value() * self.max_position_size
            confidence_multiplier = confidence / 100.0
            adjusted_target = base_target * confidence_multiplier
            return int(adjusted_target / price)

        return 0

    def _apply_risk_management(self, current_date: datetime, market_data: pd.Series):
        """リスク管理適用"""
        portfolio_value = self._get_portfolio_value()

        # 最大日次損失制限
        if self.max_daily_loss_limit:
            daily_pnl = (
                portfolio_value - self.equity_curve[-1] if self.equity_curve else 0
            )
            if daily_pnl < -self.max_daily_loss_limit:
                self._close_all_positions(current_date, "daily_loss_limit")

        # ポートフォリオ熱度チェック
        total_heat = sum(
            abs(pos.unrealized_pnl) / portfolio_value
            for pos in self.positions.values()
            if pos.quantity != 0
        )

        if total_heat > self.max_portfolio_heat:
            # リスクの高いポジションを削減
            self._reduce_risky_positions(current_date)

    def _close_all_positions(self, current_date: datetime, reason: str):
        """全ポジション決済"""
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                order = Order(
                    order_id=f"close_{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side="sell",
                    quantity=position.quantity,
                    timestamp=current_date,
                )
                self.orders.append(order)

        logger.info(
            f"全ポジション決済実行: {reason}",
            section="risk_management",
            positions_count=len([p for p in self.positions.values() if p.quantity > 0]),
        )

    def _reduce_risky_positions(self, current_date: datetime):
        """リスクの高いポジション削減"""
        risky_positions = [
            (symbol, pos)
            for symbol, pos in self.positions.items()
            if pos.quantity > 0 and pos.unrealized_pnl < 0
        ]

        # 損失の大きい順にソート
        risky_positions.sort(key=lambda x: x[1].unrealized_pnl)

        # 上位のリスクポジションを半分決済
        for symbol, position in risky_positions[: len(risky_positions) // 2]:
            reduce_quantity = position.quantity // 2
            if reduce_quantity > 0:
                order = Order(
                    order_id=f"reduce_{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side="sell",
                    quantity=reduce_quantity,
                    timestamp=current_date,
                )
                self.orders.append(order)

    def _get_portfolio_value(self) -> float:
        """ポートフォリオ総額計算"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.current_capital + positions_value

    def _record_portfolio_state(self, current_date: datetime, market_data: pd.Series):
        """ポートフォリオ状態記録"""
        portfolio_value = self._get_portfolio_value()

        # 日次リターン計算
        if self.equity_curve:
            daily_return = (
                portfolio_value - self.equity_curve[-1]
            ) / self.equity_curve[-1]
            self.daily_returns.append(daily_return)

        self.equity_curve.append(portfolio_value)

        # ドローダウン計算
        peak = max(self.equity_curve)
        drawdown = (portfolio_value - peak) / peak
        self.drawdown_series.append(drawdown)

        # ポートフォリオ履歴記録
        portfolio_record = {
            "timestamp": current_date,
            "portfolio_value": portfolio_value,
            "cash": self.current_capital,
            "positions_value": sum(pos.market_value for pos in self.positions.values()),
            "unrealized_pnl": sum(
                pos.unrealized_pnl for pos in self.positions.values()
            ),
            "daily_return": self.daily_returns[-1] if self.daily_returns else 0.0,
            "drawdown": drawdown,
        }
        self.portfolio_history.append(portfolio_record)

    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """パフォーマンス指標計算"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return PerformanceMetrics()

        # 基本リターン計算
        total_return = (
            self.equity_curve[-1] - self.equity_curve[0]
        ) / self.equity_curve[0]

        # 年率換算リターン
        if len(self.daily_returns) > 0:
            daily_mean = np.mean(self.daily_returns)
            annual_return = (1 + daily_mean) ** 252 - 1
            volatility = np.std(self.daily_returns) * np.sqrt(252)
        else:
            annual_return = 0.0
            volatility = 0.0

        # リスク調整指標
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0.0

        # ソルティーノレシオ
        negative_returns = [r for r in self.daily_returns if r < 0]
        if negative_returns:
            downside_deviation = np.std(negative_returns) * np.sqrt(252)
            sortino_ratio = (
                annual_return / downside_deviation
                if downside_deviation > 1e-10
                else 0.0
            )
        else:
            sortino_ratio = 0.0

        # ドローダウン分析
        max_drawdown = min(self.drawdown_series) if self.drawdown_series else 0.0

        # カルマーレシオ
        calmar_ratio = (
            annual_return / abs(max_drawdown)
            if max_drawdown != 0 and abs(max_drawdown) > 1e-10
            else 0.0
        )

        # 取引分析
        if self.trade_history:
            profits = []
            losses = []

            for trade in self.trade_history:
                if trade["side"] == "sell":  # 利益確定取引のみ
                    # 簡易実装：実際は buy-sell ペアで計算すべき
                    pnl = 0  # 実装省略
                    if pnl > 0:
                        profits.append(pnl)
                    else:
                        losses.append(abs(pnl))

            total_trades = len(self.trade_history)
            winning_trades = len(profits)
            losing_trades = len(losses)

            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            avg_win = np.mean(profits) if profits else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            profit_factor = (
                sum(profits) / sum(losses)
                if losses and sum(losses) > 0
                else (float("inf") if profits else 0.0)
            )

            # コスト分析
            total_commission = sum(trade["commission"] for trade in self.trade_history)
            total_slippage = sum(trade["slippage"] for trade in self.trade_history)
        else:
            total_trades = winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = 0.0
            total_commission = total_slippage = 0.0

        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_commission=total_commission,
            total_slippage=total_slippage,
        )


class WalkForwardOptimizer:
    """ウォークフォワード最適化"""

    def __init__(
        self,
        backtest_engine: AdvancedBacktestEngine,
        optimization_window: int = 252,  # 1年
        rebalance_frequency: int = 63,  # 四半期
        parameter_grid: Optional[Dict[str, List]] = None,
    ):
        self.backtest_engine = backtest_engine
        self.optimization_window = optimization_window
        self.rebalance_frequency = rebalance_frequency
        self.parameter_grid = parameter_grid or {}

    def optimize(
        self,
        data: pd.DataFrame,
        strategy_func,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """
        ウォークフォワード最適化実行

        Args:
            data: 価格データ
            strategy_func: 戦略関数
            start_date: 開始日
            end_date: 終了日

        Returns:
            最適化結果
        """
        logger.info(
            "ウォークフォワード最適化開始",
            section="walk_forward",
            optimization_window=self.optimization_window,
            rebalance_frequency=self.rebalance_frequency,
        )

        results = []
        current_date = start_date

        while current_date < end_date:
            # 最適化期間の設定
            opt_start = current_date - timedelta(days=self.optimization_window)
            opt_end = current_date

            # テスト期間の設定
            test_start = current_date
            test_end = min(
                current_date + timedelta(days=self.rebalance_frequency), end_date
            )

            # 最適化実行
            if opt_start >= data.index[0]:
                best_params = self._optimize_parameters(
                    data.loc[opt_start:opt_end], strategy_func
                )

                # テスト実行
                test_performance = self._test_parameters(
                    data.loc[test_start:test_end], strategy_func, best_params
                )

                results.append(
                    {
                        "optimization_period": (opt_start, opt_end),
                        "test_period": (test_start, test_end),
                        "best_parameters": best_params,
                        "test_performance": test_performance,
                    }
                )

            current_date = test_end

        # 結果分析
        analysis = self._analyze_walk_forward_results(results)

        logger.info(
            "ウォークフォワード最適化完了",
            section="walk_forward",
            periods_tested=len(results),
            avg_sharpe=analysis.get("avg_sharpe", 0),
        )

        return analysis

    def _optimize_parameters(self, data: pd.DataFrame, strategy_func) -> Dict[str, Any]:
        """パラメータ最適化"""
        if not self.parameter_grid:
            return {}

        best_params = {}
        best_sharpe = -float("inf")

        # グリッドサーチ（簡易実装）
        for param_name, param_values in self.parameter_grid.items():
            for param_value in param_values:
                params = {param_name: param_value}

                try:
                    # 戦略実行
                    signals = strategy_func(data, **params)

                    # バックテスト実行
                    performance = self.backtest_engine.run_backtest(data, signals)

                    if performance.sharpe_ratio > best_sharpe:
                        best_sharpe = performance.sharpe_ratio
                        best_params = params.copy()

                except Exception as e:
                    logger.warning(
                        f"パラメータ最適化エラー: {params}",
                        section="parameter_optimization",
                        error=str(e),
                    )

        return best_params

    def _test_parameters(
        self, data: pd.DataFrame, strategy_func, params: Dict[str, Any]
    ) -> PerformanceMetrics:
        """パラメータテスト"""
        try:
            signals = strategy_func(data, **params)
            return self.backtest_engine.run_backtest(data, signals)
        except Exception as e:
            logger.error(
                f"パラメータテストエラー: {params}",
                section="parameter_test",
                error=str(e),
            )
            return PerformanceMetrics()

    def _analyze_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """ウォークフォワード結果分析"""
        if not results:
            return {}

        # パフォーマンス統計
        sharpe_ratios = [r["test_performance"].sharpe_ratio for r in results]
        returns = [r["test_performance"].total_return for r in results]
        max_drawdowns = [r["test_performance"].max_drawdown for r in results]

        analysis = {
            "avg_sharpe": np.mean(sharpe_ratios),
            "std_sharpe": np.std(sharpe_ratios),
            "avg_return": np.mean(returns),
            "std_return": np.std(returns),
            "avg_max_drawdown": np.mean(max_drawdowns),
            "stability_score": 1
            - (np.std(sharpe_ratios) / max(np.mean(sharpe_ratios), 0.01)),
            "parameter_stability": self._analyze_parameter_stability(results),
        }

        return analysis

    def _analyze_parameter_stability(self, results: List[Dict]) -> Dict[str, float]:
        """パラメータ安定性分析"""
        param_changes = {}

        for i in range(1, len(results)):
            prev_params = results[i - 1]["best_parameters"]
            curr_params = results[i]["best_parameters"]

            for param_name in prev_params:
                if param_name in curr_params:
                    if param_name not in param_changes:
                        param_changes[param_name] = 0

                    if prev_params[param_name] != curr_params[param_name]:
                        param_changes[param_name] += 1

        # 変更頻度を安定性スコアに変換
        stability_scores = {}
        total_periods = len(results) - 1

        for param_name, changes in param_changes.items():
            stability_scores[param_name] = (
                1 - (changes / total_periods) if total_periods > 0 else 1.0
            )

        return stability_scores


# 使用例とデモ
if __name__ == "__main__":
    # サンプルデータ生成
    import yfinance as yf

    # データ取得
    ticker = "7203.T"
    data = yf.download(ticker, period="2y")

    # 簡易戦略シグナル生成
    signals = pd.DataFrame(index=data.index)
    signals["signal"] = "hold"
    signals["confidence"] = 50.0

    # 簡易移動平均クロス戦略
    ma_short = data["Close"].rolling(20).mean()
    ma_long = data["Close"].rolling(50).mean()

    buy_signals = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
    sell_signals = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

    signals.loc[buy_signals, "signal"] = "buy"
    signals.loc[buy_signals, "confidence"] = 70.0
    signals.loc[sell_signals, "signal"] = "sell"
    signals.loc[sell_signals, "confidence"] = 70.0

    # バックテストエンジン設定
    trading_costs = TradingCosts(
        commission_rate=0.001, bid_ask_spread_rate=0.001, slippage_rate=0.0005
    )

    backtest_engine = AdvancedBacktestEngine(
        initial_capital=1000000,
        trading_costs=trading_costs,
        position_sizing="percent",
        max_position_size=0.2,
        realistic_execution=True,
    )

    # バックテスト実行
    performance = backtest_engine.run_backtest(data, signals)

    logger.info(
        "高度バックテストデモ完了",
        section="demo",
        total_return=performance.total_return,
        sharpe_ratio=performance.sharpe_ratio,
        max_drawdown=performance.max_drawdown,
        total_trades=performance.total_trades,
        win_rate=performance.win_rate,
    )
