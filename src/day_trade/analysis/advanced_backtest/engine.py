"""
高度なバックテストエンジン - メインエンジン

全てのコンポーネントを統合したメインバックテストエンジン。
"""

import warnings
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from day_trade.utils.logging_config import get_context_logger
from .data_structures import TradingCosts, PerformanceMetrics
from .order_management import OrderManager
from .position_management import PositionManager
from .event_handlers import EventHandler
from .risk_management import RiskManager
from .performance_calculator import PerformanceCalculator

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


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
        risk_free_rate: float = 0.01,
        # リスク管理パラメータ
        max_daily_loss_limit: Optional[float] = None,
        max_portfolio_heat: float = 0.02,
        stop_loss_percentage: float = 0.05,
        take_profit_percentage: float = 0.15,
    ) -> None:
        """高度なバックテストエンジンの初期化"""
        self.initial_capital = initial_capital
        self.trading_costs = trading_costs or TradingCosts()
        self.position_sizing = position_sizing
        self.max_position_size = max_position_size
        self.enable_slippage = enable_slippage
        self.enable_market_impact = enable_market_impact
        self.realistic_execution = realistic_execution

        # コンポーネント初期化
        self.order_manager = OrderManager(self.trading_costs)
        self.position_manager = PositionManager(initial_capital)
        self.risk_manager = RiskManager(
            max_daily_loss_limit=max_daily_loss_limit,
            max_portfolio_heat=max_portfolio_heat,
            stop_loss_percentage=stop_loss_percentage,
            take_profit_percentage=take_profit_percentage,
        )
        self.performance_calculator = PerformanceCalculator(risk_free_rate)

        # イベントハンドラ
        self.event_handler = EventHandler(
            self.order_manager,
            self.position_manager,
            max_position_size,
            position_sizing
        )

        # パフォーマンス追跡
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = [self.initial_capital]
        self.drawdown_series: List[float] = []
        self.portfolio_history: List[Dict] = []

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
            strategy_signals: 戦略シグナル
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

        # イベントキューの生成
        events = self.event_handler.create_event_queue_from_data(data, strategy_signals)
        self.event_handler.events = events

        # イベント駆動型ループ
        while self.event_handler.has_pending_events():
            event = self.event_handler.get_next_event()
            if not event:
                break

            # イベント処理
            self.event_handler.process_event(event)

            # 市場データイベントの場合、追加処理を実行
            if hasattr(event, 'symbol') and hasattr(event, 'close'):
                # ポートフォリオ状態の記録
                self._record_portfolio_state(event.timestamp, event)
                
                # リスク管理の適用
                self._apply_risk_management(event.timestamp)

        # パフォーマンス計算
        performance = self.performance_calculator.calculate_performance_metrics(
            self.equity_curve,
            self.daily_returns,
            self.position_manager.trade_history,
            self.drawdown_series
        )

        logger.info(
            "バックテスト完了",
            section="backtest_execution",
            total_return=performance.total_return,
            sharpe_ratio=performance.sharpe_ratio,
            max_drawdown=performance.max_drawdown,
            total_trades=performance.total_trades,
        )

        return performance

    def _reset_backtest(self) -> None:
        """バックテスト状態リセット"""
        self.position_manager.current_capital = self.initial_capital
        self.position_manager.positions = {}
        self.position_manager.trade_history = []
        self.position_manager.open_trades = {}
        
        self.order_manager.pending_orders = {}
        
        self.daily_returns = []
        self.equity_curve = [self.initial_capital]
        self.drawdown_series = []
        self.portfolio_history = []

        # リスク管理の履歴もリセット
        self.risk_manager.risk_alerts = []

    def _record_portfolio_state(self, current_date: datetime, market_event) -> None:
        """ポートフォリオ状態記録"""
        portfolio_value = self.position_manager.get_portfolio_value()

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
            "cash": self.position_manager.current_capital,
            "positions_value": sum(pos.market_value for pos in self.position_manager.positions.values()),
            "unrealized_pnl": self.position_manager.get_total_unrealized_pnl(),
            "realized_pnl": self.position_manager.get_total_realized_pnl(),
            "daily_return": self.daily_returns[-1] if self.daily_returns else 0.0,
            "drawdown": drawdown,
            "active_positions": self.position_manager.get_position_count(),
            "pending_orders": self.order_manager.get_pending_order_count(),
        }
        self.portfolio_history.append(portfolio_record)

    def _apply_risk_management(self, current_date: datetime) -> None:
        """リスク管理適用"""
        # 基本的なリスク制限チェック
        risk_orders = self.risk_manager.check_risk_limits(
            self.position_manager,
            current_date,
            self.equity_curve
        )

        # 生成されたリスク管理注文を処理
        for order in risk_orders:
            # 注文を直接処理（簡易実装）
            # 実際の実装では適切なイベントを生成すべき
            self.order_manager.pending_orders[order.order_id] = order

        # ストップロス・テイクプロフィットチェック
        current_prices = {
            symbol: event.close 
            for symbol, event in self.event_handler.current_market_data_by_symbol.items()
        }
        
        sl_tp_orders = self.risk_manager.check_stop_loss_take_profit(
            self.position_manager,
            current_prices,
            current_date
        )

        for order in sl_tp_orders:
            self.order_manager.pending_orders[order.order_id] = order

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """ポートフォリオサマリの取得"""
        portfolio_value = self.position_manager.get_portfolio_value()
        
        return {
            "portfolio_value": portfolio_value,
            "cash": self.position_manager.current_capital,
            "positions_value": sum(pos.market_value for pos in self.position_manager.positions.values()),
            "active_positions": self.position_manager.get_position_count(),
            "pending_orders": self.order_manager.get_pending_order_count(),
            "total_trades": len(self.position_manager.trade_history),
            "unrealized_pnl": self.position_manager.get_total_unrealized_pnl(),
            "realized_pnl": self.position_manager.get_total_realized_pnl(),
            "total_return": (portfolio_value - self.initial_capital) / self.initial_capital,
            "risk_alerts": len(self.risk_manager.risk_alerts),
        }

    def get_detailed_performance(self) -> Dict[str, Any]:
        """詳細パフォーマンス分析の取得"""
        performance = self.performance_calculator.calculate_performance_metrics(
            self.equity_curve,
            self.daily_returns,
            self.position_manager.trade_history,
            self.drawdown_series
        )

        return {
            "basic_metrics": performance,
            "portfolio_summary": self.get_portfolio_summary(),
            "risk_summary": self.risk_manager.get_risk_summary(),
            "performance_summary": self.performance_calculator.create_performance_summary(performance),
            "equity_curve": self.equity_curve.copy(),
            "drawdown_series": self.drawdown_series.copy(),
            "portfolio_history": self.portfolio_history.copy(),
        }

    def export_results(self) -> Dict[str, Any]:
        """結果のエクスポート"""
        return {
            "backtest_config": {
                "initial_capital": self.initial_capital,
                "trading_costs": {
                    "commission_rate": self.trading_costs.commission_rate,
                    "bid_ask_spread_rate": self.trading_costs.bid_ask_spread_rate,
                    "slippage_rate": self.trading_costs.slippage_rate,
                },
                "position_sizing": self.position_sizing,
                "max_position_size": self.max_position_size,
                "realistic_execution": self.realistic_execution,
            },
            "performance": self.get_detailed_performance(),
            "trades": [
                {
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "quantity": trade.entry_quantity,
                    "realized_pnl": trade.realized_pnl,
                    "commission": trade.total_commission,
                    "slippage": trade.total_slippage,
                }
                for trade in self.position_manager.trade_history
            ],
        }