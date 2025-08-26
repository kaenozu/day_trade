"""
高度なバックテストエンジン

現実的な取引コスト、スリッページ、流動性制約を考慮した
包括的なバックテスト環境とウォークフォワード最適化。
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Deque
from collections import deque

import numpy as np
import pandas as pd

from day_trade.utils.logging_config import get_context_logger
from day_trade.analysis.events import Event, EventType, MarketDataEvent, SignalEvent, OrderEvent, FillEvent, Order, TradeRecord, OrderType, OrderStatus

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)

# OrderType と OrderStatus は events.py に移動しました

@dataclass
class TradingCosts:
    """取引コスト設定"""

    commission_rate: float = 0.001  # 手数料率 (0.1%)
    min_commission: float = 0.0  # 最小手数料
    max_commission: float = float("inf")  # 最大手数料
    bid_ask_spread_rate: float = 0.001  # ビッドアスクスプレッド率
    slippage_rate: float = 0.0005  # スリッページ率
    market_impact_rate: float = 0.0002  # マーケットインパクト率

# Order クラスは events.py に移動しました

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
    timestamp: datetime = field(default_factory = datetime.now)
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
    last_updated: datetime = field(default_factory = datetime.now)

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
    ) -> None:
        """高度なバックテストエンジンの初期化"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trading_costs = trading_costs or TradingCosts()
        self.position_sizing = position_sizing
        self.max_position_size = max_position_size
        self.enable_slippage = enable_slippage
        self.enable_market_impact = enable_market_impact
        self.realistic_execution = realistic_execution

        # イベントキュー
        self.events: Deque[Event] = deque()

        # 取引状態
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {} # order_id でアクセスできるよう変更
        self.trade_history: List[TradeRecord] = []
        self.portfolio_history: List[Dict] = []

        # 未決済の取引を追跡するための辞書
        self.open_trades: Dict[str, TradeRecord] = {}

        # パフォーマンス追跡
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = [self.initial_capital]
        self.drawdown_series: List[float] = []

        # リスク管理
        self.max_daily_loss_limit: Optional[float] = None
        self.max_portfolio_heat: float = 0.02  # 2%

        # イベントハンドラ
        self._event_handlers: Dict[EventType, List[Callable]] = {
            EventType.MARKET_DATA: [self._handle_market_data],
            EventType.SIGNAL: [self._handle_signal],
            EventType.ORDER: [self._handle_order],
            EventType.FILL: [self._handle_fill],
            # 将来的にカスタムイベントハンドラを追加
        }

        # 現在の時刻を追跡
        self._current_sim_time: Optional[datetime] = None
        # 各シンボルの最新市場データを保存するための辞書
        self.current_market_data_by_symbol: Dict[str, MarketDataEvent] = {}

        logger.info(
            "高度バックテストエンジン初期化",
            section="backtest_init",
            initial_capital = initial_capital,
            realistic_execution = realistic_execution,
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
            data_range = f"{data.index[0]} to {data.index[-1]}",
            signals_count = len(strategy_signals),
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

        # イベントキューに初期イベントを投入
        # データフレームのインデックスを時系列イベントとして利用
        all_timestamps = sorted(data.index.union(strategy_signals.index).unique())

        for timestamp in all_timestamps:
            if timestamp in data.index:
                # MarketDataEvent を生成
                row = data.loc[timestamp]
                # pandas.Series.name がシンボルになることを想定
                symbol = data.columns.levels[0][0] if isinstance(data.columns, pd.MultiIndex) else "UNKNOWN" # マルチインデックス対応
                market_event = MarketDataEvent(
                    type = EventType.MARKET_DATA,
                    timestamp = timestamp,
                    symbol = symbol,
                    open = row[(symbol, "Open")] if isinstance(data.columns, pd.MultiIndex) else row["Open"],
                    high = row[(symbol, "High")] if isinstance(data.columns, pd.MultiIndex) else row["High"],
                    low = row[(symbol, "Low")] if isinstance(data.columns, pd.MultiIndex) else row["Low"],
                    close = row[(symbol, "Close")] if isinstance(data.columns, pd.MultiIndex) else row["Close"],
                    volume = row.get((symbol, "Volume")) if isinstance(data.columns, pd.MultiIndex) else row.get("Volume"),
                )
                self.events.append(market_event)

            if timestamp in strategy_signals.index:
                # SignalEvent を生成
                signal_row = strategy_signals.loc[timestamp]
                symbol = strategy_signals.columns.levels[0][0] if isinstance(strategy_signals.columns, pd.MultiIndex) else "UNKNOWN" # マルチインデックス対応
                signal_event = SignalEvent(
                    type = EventType.SIGNAL,
                    timestamp = timestamp,
                    symbol = symbol,
                    action = signal_row[(symbol, "signal")] if isinstance(strategy_signals.columns, pd.MultiIndex) else signal_row["signal"],
                    strength = signal_row.get((symbol, "confidence"), 0.0) if isinstance(strategy_signals.columns, pd.MultiIndex) else signal_row.get("confidence", 0.0),
                    data={"price_target": signal_row.get((symbol, "price_target"))} if isinstance(strategy_signals.columns, pd.MultiIndex) else {"price_target": signal_row.get("price_target")},
                )
                self.events.append(signal_event)

        # イベントキューを時系列順にソート
        self.events = deque(sorted(list(self.events), key = lambda event: event.timestamp))

        # イベント駆動型ループ
        while self.events:
            event = self.events.popleft()
            self._current_sim_time = event.timestamp # シミュレーション時間を更新

            try:
                # イベントタイプに応じたハンドラを呼び出す
                if isinstance(event, MarketDataEvent):
                    self._handle_market_data(event)
                elif isinstance(event, SignalEvent):
                    self._handle_signal(event)
                elif isinstance(event, OrderEvent):
                    self._handle_order(event)
                elif isinstance(event, FillEvent):
                    self._handle_fill(event)
                else:
                    logger.warning(f"未処理のイベントタイプ: {event.type.value}")

                # ポートフォリオ状態の記録 (日次またはイベント毎)
                # ここでは MarketDataEvent の処理後に記録
                if isinstance(event, MarketDataEvent):
                    self._record_portfolio_state(event.timestamp, event.data)
                    # 日次処理としてリスク管理を適用
                    self._apply_risk_management(event.timestamp, event.data)

            except Exception as e:
                logger.warning(
                    f"イベント処理エラー: {event.type.value} at {event.timestamp} - {e}",
                    section="event_processing",
                    error = str(e),
                )

        # パフォーマンス計算
        performance = self._calculate_performance_metrics()

        logger.info(
            "バックテスト完了",
            section="backtest_execution",
            total_return = performance.total_return,
            sharpe_ratio = performance.sharpe_ratio,
            max_drawdown = performance.max_drawdown,
            total_trades = performance.total_trades,
        )

        return performance

    def _reset_backtest(self) -> None:
        """バックテスト状態リセット"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.orders = []
        self.trade_history = []
        self.portfolio_history = []
        self.daily_returns = []
        self.equity_curve = [self.initial_capital]
        self.drawdown_series = []
        self.open_trades = {}

    def _process_pending_orders(self, current_time: datetime, market_data_event: MarketDataEvent) -> None:
        """待機中注文の処理（イベントキューへの約定イベントプッシュ）"""
        orders_to_remove = []
        for order_id, order in list(self.orders.items()): # 辞書のコピーをイテレート
            if order.status != Order.OrderStatus.PENDING:
                continue

            # Check if this order's symbol matches the current market data event's symbol
            if order.symbol != market_data_event.symbol:
                continue # Only process orders for the current symbol's market data event

            # Convert MarketDataEvent to pd.Series for _should_fill_order and _calculate_fill_price
            # These methods are designed for pd.Series, so we create one from MarketDataEvent.
            symbol_market_data_series = pd.Series({
                "Open": market_data_event.open,
                "High": market_data_event.high,
                "Low": market_data_event.low,
                "Close": market_data_event.close,
                "Volume": market_data_event.volume # Volume might be None
            })
            symbol_market_data_series.name = market_data_event.symbol

            if self._should_fill_order(order, symbol_market_data_series):
                filled_price = self._calculate_fill_price(order, symbol_market_data_series)

                # 約定イベントを生成し、キューにプッシュ
                fill_event = FillEvent(
                    type = EventType.FILL, # ★修正
                    order_id = order.order_id,
                    symbol = order.symbol,
                    action = order.side,
                    quantity = order.quantity, # 現状は全量約定のみをシミュレート
                    price = filled_price,
                    commission = self._calculate_commission_for_order(order, filled_price),
                    slippage = abs(filled_price - (order.price or filled_price)),
                    fill_time = current_time,
                )
                self.events.append(fill_event) # キューの末尾に追加し、後続のイベントループで処理される
                orders_to_remove.append(order_id) # 処理済みとしてマーク

        # 処理済みの注文を削除
        for order_id in orders_to_remove:
            del self.orders[order_id]

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

    # ===== イベントハンドラ群 =====
    def _handle_market_data(self, event: MarketDataEvent) -> None:
        """市場データイベントを処理"""
        # Store the latest market data for this symbol
        self.current_market_data_by_symbol[event.symbol] = event # ★追加

        # 既存注文の処理
        # Pass the full MarketDataEvent object for processing orders
        self._process_pending_orders(event.timestamp, event) # ★修正

        # ポジション時価の更新
        # Access attributes directly from event
        self._update_positions_from_market_data(event.timestamp, event.symbol, event.close) # ★修正

    def _handle_signal(self, event: SignalEvent) -> None:
        """シグナルイベントを処理"""
        symbol = event.symbol
        signal_action = event.action
        confidence = event.strength

        # Retrieve current market data for this symbol
        current_market_data_event = self.current_market_data_by_symbol.get(symbol) # ★修正
        if current_market_data_event is None:
            logger.warning(f"シグナル処理に十分な市場データがありません: {symbol} at {event.timestamp}") # ★修正
            return

        current_price_for_signal = current_market_data_event.close # ★修正

        if signal_action in ["buy", "sell"] and confidence > 50.0:
            # ポジションサイズ計算
            position_size = self._calculate_position_size(
                symbol, current_price_for_signal, confidence
            )

            if position_size > 0:
                # OrderEvent を生成し、キューにプッシュ
                order_id = f"{symbol}_{self._current_sim_time.strftime('%Y%m%d_%H%M%S')}"
                order_event = OrderEvent(
                    type = EventType.ORDER,
                    order_id = order_id,
                    symbol = symbol,
                    action = signal_action,
                    quantity = position_size,
                    order_type="market", # シグナルからはマーケット注文を出すと仮定
                    price = current_price_for_signal,
                    timestamp = self._current_sim_time,
                )
                self.events.append(order_event) # キューの末尾に追加
                self.orders[order_id] = order_event # 注文を追跡するために保存

                logger.debug(
                    f"シグナルから注文イベント生成: {signal_action} {position_size} {symbol}",
                    section="signal_processing",
                    confidence = confidence,
                )

    def _handle_order(self, event: OrderEvent) -> None:
        """注文イベントを処理 (主にシミュレーション内部での注文発行)"""
        # OrderEvent から Order オブジェクトを生成し、self.orders に追加
        # 注: Order.OrderType は events.py で定義された Enum です。
        # event.order_type は文字列なので、Enumに変換する必要があります。
        # または、OrderEvent.order_type を直接 Order.OrderType 型にする必要があります。
        # ここでは便宜上、文字列からEnumに変換します。
        order_type_enum = Order.OrderType[event.order_type.upper()] # ★修正
        order = Order(
            order_id = event.order_id,
            symbol = event.symbol,
            order_type = order_type_enum, # ★修正
            side = event.action,
            quantity = event.quantity,
            price = event.price,
            stop_price = event.stop_price,
            timestamp = event.timestamp,
        )
        self.orders[order.order_id] = order
        logger.debug(f"新しい注文をキューに追加: {order.order_id}") # ★修正

    def _handle_fill(self, event: FillEvent) -> None:
        """約定イベントを処理"""
        # 注文情報を更新し、ポジションと資本を調整
        if event.order_id in self.orders:
            order = self.orders[event.order_id]
            order.status = Order.OrderStatus.FILLED
            order.filled_quantity = event.quantity
            order.filled_price = event.price
            order.commission = event.commission
            order.slippage = event.slippage

            # ポジションを更新し、ポジションからの実現損益を取得
            realized_pnl_from_position = self._update_position_from_order(order)

            # TradeRecordの処理
            if order.side == "buy":
                # 新しいエントリーまたは既存ポジションへの追加
                if order.symbol not in self.open_trades or self.open_trades[order.symbol].is_closed:
                    # 新規取引
                    trade_id = f"trade_{order.symbol}_{event.fill_time.strftime('%Y%m%d%H%M%S%f')}"
                    new_trade = TradeRecord(
                        trade_id = trade_id,
                        symbol = order.symbol,
                        entry_time = event.fill_time,
                        entry_price = event.price,
                        entry_quantity = event.quantity,
                        entry_commission = event.commission,
                        entry_slippage = event.slippage,
                    )
                    self.open_trades[order.symbol] = new_trade
                else:
                    # 既存取引への追加（平均コスト計算を伴う）
                    existing_trade = self.open_trades[order.symbol]
                    # ここでは簡易的に、平均価格の更新はPositionオブジェクトに任せ、TradeRecordはEntry情報を更新しない
                    # もしTradeRecordで平均価格を追跡するなら、ここでロジックを追加
                    existing_trade.entry_quantity += order.quantity # 数量のみ更新
                    existing_trade.entry_commission += order.commission # コミッション加算
                    existing_trade.entry_slippage += order.slippage # スリッページ加算

            elif order.side == "sell":
                # エグジット処理
                if order.symbol in self.open_trades and not self.open_trades[order.symbol].is_closed:
                    closed_trade = self.open_trades[order.symbol]
                    closed_trade.exit_time = event.fill_time
                    closed_trade.exit_price = event.price
                    closed_trade.exit_quantity = order.quantity
                    closed_trade.exit_commission = event.commission
                    closed_trade.exit_slippage = event.slippage

                    # 実現損益、合計手数料、合計スリッページを更新
                    closed_trade.realized_pnl = realized_pnl_from_position # _update_position_from_orderから取得
                    closed_trade.total_commission = closed_trade.entry_commission + closed_trade.exit_commission
                    closed_trade.total_slippage = closed_trade.entry_slippage + closed_trade.exit_slippage
                    closed_trade.is_closed = True

                    self.trade_history.append(closed_trade) # 確定した取引を履歴に追加
                    del self.open_trades[order.symbol] # 未決済から削除
                else:
                    logger.warning(f"未エントリーのポジションの決済イベント: {order.symbol}")
                    # ショートポジションの決済の場合なども考慮する必要がある（別途実装）
                    # 現状はロングポジションのみを前提とする

            logger.debug(f"注文約定を処理: {order.order_id}")
        else:
            logger.warning(f"未知の注文IDの約定イベント: {event.order_id}")

    def _update_position_from_order(self, order: Order) -> float:
        """注文からポジション更新"""
        symbol = order.symbol
        realized_pnl_for_order = 0.0

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol = symbol)

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
                realized_pnl_for_order = (
                    order.filled_price - position.average_price
                ) * order.quantity - order.commission
                position.realized_pnl += realized_pnl_for_order
                position.quantity -= order.quantity

                if position.quantity == 0:
                    position.average_price = 0.0
            else:
                logger.warning(
                    f"ショートポジション発生: {symbol}",
                    section="position_management",
                    current_quantity = position.quantity,
                    sell_quantity = order.quantity,
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

        return realized_pnl_for_order # 計算した実現損益を返す

    def _update_positions(self, current_date: datetime, market_data: pd.Series) -> None:
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
    ) -> None:
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
                    order_id = f"{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}",
                    symbol = symbol,
                    order_type = OrderType.MARKET,
                    side = signal,
                    quantity = position_size,
                    timestamp = current_date,
                )

                self.orders.append(order)

                logger.debug(
                    f"シグナル注文生成: {signal} {position_size} {symbol}",
                    section="signal_processing",
                    confidence = confidence,
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

    def _apply_risk_management(self, current_date: datetime, market_data: pd.Series) -> None:
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

    def _close_all_positions(self, current_date: datetime, reason: str) -> None:
        """全ポジション決済"""
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                order = Order(
                    order_id = f"close_{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}",
                    symbol = symbol,
                    order_type = OrderType.MARKET,
                    side="sell",
                    quantity = position.quantity,
                    timestamp = current_date,
                )
                self.orders.append(order)

        logger.info(
            f"全ポジション決済実行: {reason}",
            section="risk_management",
            positions_count = len([p for p in self.positions.values() if p.quantity > 0]),
        )

    def _reduce_risky_positions(self, current_date: datetime) -> None:
        """リスクの高いポジション削減"""
        risky_positions = [
            (symbol, pos)
            for symbol, pos in self.positions.items()
            if pos.quantity > 0 and pos.unrealized_pnl < 0
        ]

        # 損失の大きい順にソート
        risky_positions.sort(key = lambda x: x[1].unrealized_pnl)

        # 上位のリスクポジションを半分決済
        for symbol, position in risky_positions[: len(risky_positions) // 2]:
            reduce_quantity = position.quantity // 2
            if reduce_quantity > 0:
                order = Order(
                    order_id = f"reduce_{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}",
                    symbol = symbol,
                    order_type = OrderType.MARKET,
                    side="sell",
                    quantity = reduce_quantity,
                    timestamp = current_date,
                )
                self.orders.append(order)

    def _get_portfolio_value(self) -> float:
        """ポートフォリオ総額計算"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.current_capital + positions_value

    def _record_portfolio_state(self, current_date: datetime, market_data: pd.Series) -> None:
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
        sharpe_ratio = self._calculate_sharpe_ratio(annual_return, volatility)

        # ソルティーノレシオ
        sortino_ratio = self._calculate_sortino_ratio(annual_return)

        # ドローダウン分析
        max_drawdown = min(self.drawdown_series) if self.drawdown_series else 0.0

        # カルマーレシオ
        calmar_ratio = self._calculate_calmar_ratio(annual_return, max_drawdown)

        # 取引分析
        if self.trade_history:
            profits = []
            losses = []

            for trade in self.trade_history: # TradeRecordを使用
                if trade.is_closed: # 決済済みの取引のみを対象
                    pnl = trade.realized_pnl
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
            total_commission = sum(trade.total_commission for trade in self.trade_history)
            total_slippage = sum(trade.total_slippage for trade in self.trade_history)
        else:
            total_trades = winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = 0.0
            total_commission = total_slippage = 0.0

        return PerformanceMetrics(
            total_return = total_return,
            annual_return = annual_return,
            volatility = volatility,
            sharpe_ratio = sharpe_ratio,
            sortino_ratio = sortino_ratio,
            max_drawdown = max_drawdown,
            calmar_ratio = calmar_ratio,
            win_rate = win_rate,
            profit_factor = profit_factor,
            avg_win = avg_win,
            avg_loss = avg_loss,
            total_trades = total_trades,
            winning_trades = winning_trades,
            losing_trades = losing_trades,
            total_commission = total_commission,
            total_slippage = total_slippage,
        )

    def _calculate_sharpe_ratio(self, annual_return: float, volatility: float) -> float:
        """シャープレシオの計算"""
        return annual_return / volatility if volatility > 0 else 0.0

    def _calculate_sortino_ratio(self, annual_return: float) -> float:
        """ソルティーノレシオの計算"""
        negative_returns = [r for r in self.daily_returns if r < 0]
        if negative_returns:
            downside_deviation = np.std(negative_returns) * np.sqrt(252)
            return (
                annual_return / downside_deviation
                if downside_deviation > 1e-10
                else 0.0
            )
        return 0.0

    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """カルマーレシオの計算"""
        return (
            annual_return / abs(max_drawdown)
            if max_drawdown != 0 and abs(max_drawdown) > 1e-10
            else 0.0
        )

class WalkForwardOptimizer:
    """ウォークフォワード最適化"""

    def __init__(
        self,
        backtest_engine: AdvancedBacktestEngine,
        optimization_window: int = 252,  # 1年
        rebalance_frequency: int = 63,  # 四半期
        parameter_grid: Optional[Dict[str, List]] = None,
    ) -> None:
        """ウォークフォワード最適化の初期化"""
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
            optimization_window = self.optimization_window,
            rebalance_frequency = self.rebalance_frequency,
        )

        results = []
        current_date = start_date

        while current_date < end_date:
            # 最適化期間の設定
            opt_start = current_date - timedelta(days = self.optimization_window)
            opt_end = current_date

            # テスト期間の設定
            test_start = current_date
            test_end = min(
                current_date + timedelta(days = self.rebalance_frequency), end_date
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
            periods_tested = len(results),
            avg_sharpe = analysis.get("avg_sharpe", 0),
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
                        error = str(e),
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
                error = str(e),
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
    signals = pd.DataFrame(index = data.index)
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
        commission_rate = 0.001, bid_ask_spread_rate = 0.001, slippage_rate = 0.0005
    )

    backtest_engine = AdvancedBacktestEngine(
        initial_capital = 1000000,
        trading_costs = trading_costs,
        position_sizing="percent",
        max_position_size = 0.2,
        realistic_execution = True,
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


