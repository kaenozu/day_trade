"""
市場分析エンジンコア実装（旧：自動取引エンジン）

【重要】自動取引機能は無効化済み
リアルタイム市場データを処理し、分析情報を提供するエンジン

※ 実際の取引実行は行いません
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..analysis.signals import TradingSignal, TradingSignalGenerator
from ..core.trade_manager import Trade, TradeManager, TradeType
from ..data.stock_fetcher import StockFetcher
from ..utils.enhanced_error_handler import get_default_error_handler
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()


class EngineStatus(Enum):
    """取引エンジンステータス"""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class OrderType(Enum):
    """注文タイプ"""

    MARKET = "market"  # 成行
    LIMIT = "limit"  # 指値
    STOP = "stop"  # 逆指値
    STOP_LIMIT = "stop_limit"  # 逆指値限定


@dataclass
class RiskParameters:
    """リスク管理パラメータ"""

    max_position_size: Decimal = Decimal("1000000")  # 最大ポジションサイズ
    max_daily_loss: Decimal = Decimal("50000")  # 1日最大損失
    max_open_positions: int = 10  # 最大保有ポジション数
    stop_loss_ratio: Decimal = Decimal("0.02")  # ストップロス率(2%)
    take_profit_ratio: Decimal = Decimal("0.05")  # 利益確定率(5%)


@dataclass
class MarketData:
    """市場データ"""

    symbol: str
    price: Decimal
    volume: int
    timestamp: datetime
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None


@dataclass
class OrderRequest:
    """注文リクエスト"""

    symbol: str
    order_type: OrderType
    trade_type: TradeType
    quantity: int
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TradingEngine:
    """
    自動取引エンジンコア

    主要機能:
    1. リアルタイム市場データ処理
    2. シグナル生成と評価
    3. 自動注文実行
    4. リスク管理
    5. ポジション監視
    """

    def __init__(
        self,
        symbols: List[str],
        trade_manager: Optional[TradeManager] = None,
        signal_generator: Optional[TradingSignalGenerator] = None,
        stock_fetcher: Optional[StockFetcher] = None,
        risk_params: Optional[RiskParameters] = None,
        update_interval: float = 1.0,  # 秒
    ):
        self.symbols = symbols
        self.trade_manager = trade_manager or TradeManager()
        self.signal_generator = signal_generator or TradingSignalGenerator()
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.risk_params = risk_params or RiskParameters()
        self.update_interval = update_interval

        # 状態管理
        self.status = EngineStatus.STOPPED
        self.market_data: Dict[str, MarketData] = {}
        self.active_positions: Dict[str, List[Trade]] = {}
        self.pending_orders: List[OrderRequest] = []

        # パフォーマンス監視
        self.execution_stats = {
            "orders_executed": 0,
            "signals_generated": 0,
            "avg_execution_time": 0.0,
            "last_update": None,
        }

        # 非同期処理用
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._stop_event = asyncio.Event()

        logger.info(f"TradingEngine初期化完了 - 監視銘柄: {len(symbols)}")

    async def start(self) -> None:
        """取引エンジン開始"""
        if self.status == EngineStatus.RUNNING:
            logger.warning("取引エンジンは既に実行中です")
            return

        logger.info("取引エンジンを開始します...")
        self.status = EngineStatus.RUNNING
        self._stop_event.clear()

        # メインループ開始
        await self._main_loop()

    async def stop(self) -> None:
        """取引エンジン停止"""
        logger.info("取引エンジン停止要求受信")
        self.status = EngineStatus.STOPPED
        self._stop_event.set()

        # 保留中の注文をキャンセル
        self.pending_orders.clear()
        logger.info("取引エンジンが停止しました")

    async def pause(self) -> None:
        """取引エンジン一時停止"""
        if self.status == EngineStatus.RUNNING:
            self.status = EngineStatus.PAUSED
            logger.info("取引エンジンを一時停止しました")

    async def resume(self) -> None:
        """取引エンジン再開"""
        if self.status == EngineStatus.PAUSED:
            self.status = EngineStatus.RUNNING
            logger.info("取引エンジンを再開しました")

    async def _main_loop(self) -> None:
        """メインループ - エンジンの中核処理"""
        try:
            while not self._stop_event.is_set() and self.status != EngineStatus.STOPPED:
                if self.status == EngineStatus.PAUSED:
                    await asyncio.sleep(1.0)
                    continue

                loop_start = time.time()

                # 1. 市場データ更新
                await self._update_market_data()

                # 2. シグナル生成
                signals = await self._generate_signals()

                # 3. リスク管理チェック
                if self._check_risk_constraints():
                    # 4. 注文生成・実行
                    await self._process_signals(signals)

                # 5. ポジション監視
                await self._monitor_positions()

                # 6. パフォーマンス更新
                self._update_performance_stats(time.time() - loop_start)

                # インターバル調整
                await asyncio.sleep(self.update_interval)

        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"取引エンジンでエラーが発生: {e}")
            error_handler.handle_error(e, context={"engine_status": self.status.value})

    async def _update_market_data(self) -> None:
        """市場データの更新"""
        try:
            # 並行処理で全銘柄の価格を取得
            loop = asyncio.get_event_loop()
            tasks = []

            for symbol in self.symbols:
                task = loop.run_in_executor(
                    self.executor, self.stock_fetcher.get_current_price, symbol
                )
                tasks.append((symbol, task))

            # 全タスク完了まで待機
            for symbol, task in tasks:
                try:
                    price_data = await task
                    if price_data:
                        self.market_data[symbol] = MarketData(
                            symbol=symbol,
                            price=Decimal(str(price_data["current_price"])),
                            volume=price_data.get("volume", 0),
                            timestamp=datetime.now(),
                        )
                except Exception as e:
                    logger.warning(f"銘柄 {symbol} の価格取得に失敗: {e}")

        except Exception as e:
            logger.error(f"市場データ更新エラー: {e}")

    async def _generate_signals(self) -> List[Tuple[str, TradingSignal]]:
        """シグナル生成"""
        signals = []

        try:
            for symbol in self.symbols:
                if symbol in self.market_data:
                    # 履歴データを取得してシグナル生成
                    loop = asyncio.get_event_loop()
                    historical_data = await loop.run_in_executor(
                        self.executor,
                        self.stock_fetcher.get_historical_data,
                        symbol,
                        "30d",
                    )

                    if historical_data is not None and not historical_data.empty:
                        signal = self.signal_generator.generate_signal(historical_data)
                        if signal:
                            signals.append((symbol, signal))
                            self.execution_stats["signals_generated"] += 1

        except Exception as e:
            logger.error(f"シグナル生成エラー: {e}")

        return signals

    def _check_risk_constraints(self) -> bool:
        """リスク制約チェック"""
        try:
            # 1. 最大ポジション数チェック
            total_positions = sum(
                len(positions) for positions in self.active_positions.values()
            )
            if total_positions >= self.risk_params.max_open_positions:
                logger.warning(f"最大ポジション数に到達: {total_positions}")
                return False

            # 2. 日次損失チェック
            daily_pnl = self._calculate_daily_pnl()
            if daily_pnl <= -self.risk_params.max_daily_loss:
                logger.warning(f"日次最大損失に到達: {daily_pnl}")
                return False

            return True

        except Exception as e:
            logger.error(f"リスク制約チェックエラー: {e}")
            return False

    async def _process_signals(self, signals: List[Tuple[str, TradingSignal]]) -> None:
        """シグナルを処理して注文を生成・実行"""
        for symbol, signal in signals:
            try:
                # シグナルが十分強い場合のみ注文を生成
                if signal.confidence >= 70.0 and signal.strength.value in [
                    "strong",
                    "medium",
                ]:
                    order_request = self._create_order_from_signal(symbol, signal)
                    if order_request:
                        await self._execute_order(order_request)

            except Exception as e:
                logger.error(f"シグナル処理エラー - {symbol}: {e}")

    def _create_order_from_signal(
        self, symbol: str, signal: TradingSignal
    ) -> Optional[OrderRequest]:
        """シグナルから注文リクエストを生成"""
        try:
            if symbol not in self.market_data:
                return None

            current_price = self.market_data[symbol].price

            # ポジションサイズ計算 (簡単な固定サイズ)
            base_quantity = 100
            quantity = int(base_quantity * (signal.confidence / 100.0))

            # 注文タイプとトレードタイプを決定
            trade_type = (
                TradeType.BUY if signal.signal_type.value == "buy" else TradeType.SELL
            )
            order_type = OrderType.MARKET  # シンプルに成行注文

            return OrderRequest(
                symbol=symbol,
                order_type=order_type,
                trade_type=trade_type,
                quantity=quantity,
                price=current_price,
            )

        except Exception as e:
            logger.error(f"注文生成エラー - {symbol}: {e}")
            return None

    async def _execute_order(self, order_request: OrderRequest) -> None:
        """注文実行"""
        try:
            execution_start = time.time()

            # Trade オブジェクトを作成
            trade = Trade(
                id=f"trade_{int(time.time()*1000)}",
                symbol=order_request.symbol,
                trade_type=order_request.trade_type,
                quantity=order_request.quantity,
                price=order_request.price or Decimal("0"),
                timestamp=order_request.timestamp,
                commission=Decimal("0"),
                status="executed",
            )

            # TradeManager を使用して取引を記録
            self.trade_manager.add_trade(trade)

            # アクティブポジション管理
            if order_request.symbol not in self.active_positions:
                self.active_positions[order_request.symbol] = []
            self.active_positions[order_request.symbol].append(trade)

            execution_time = time.time() - execution_start
            self.execution_stats["orders_executed"] += 1

            logger.info(
                f"注文実行完了 - {order_request.symbol}: "
                f"{order_request.trade_type.value} {order_request.quantity}株 "
                f"@{order_request.price} (実行時間: {execution_time*1000:.1f}ms)"
            )

        except Exception as e:
            logger.error(f"注文実行エラー: {e}")
            error_handler.handle_error(e, context={"order": order_request})

    async def _monitor_positions(self) -> None:
        """ポジション監視 - ストップロスや利益確定の処理"""
        try:
            for symbol, positions in self.active_positions.items():
                if symbol in self.market_data:
                    current_price = self.market_data[symbol].price

                    for position in positions[:]:  # コピーを作って安全にイテレート
                        await self._check_position_exit(symbol, position, current_price)

        except Exception as e:
            logger.error(f"ポジション監視エラー: {e}")

    async def _check_position_exit(
        self, symbol: str, position: Trade, current_price: Decimal
    ) -> None:
        """個別ポジションの利確/損切り判定"""
        try:
            entry_price = position.price
            pnl_ratio = (current_price - entry_price) / entry_price

            # 買いポジションの場合
            if position.trade_type == TradeType.BUY:
                # 利益確定チェック
                if pnl_ratio >= self.risk_params.take_profit_ratio:
                    await self._close_position(symbol, position, "利益確定")
                # 損切りチェック
                elif pnl_ratio <= -self.risk_params.stop_loss_ratio:
                    await self._close_position(symbol, position, "損切り")

            # 売りポジションの場合
            elif position.trade_type == TradeType.SELL:
                # 利益確定チェック (売りの場合は価格下落で利益)
                if pnl_ratio <= -self.risk_params.take_profit_ratio:
                    await self._close_position(symbol, position, "利益確定")
                # 損切りチェック
                elif pnl_ratio >= self.risk_params.stop_loss_ratio:
                    await self._close_position(symbol, position, "損切り")

        except Exception as e:
            logger.error(f"ポジション出口判定エラー: {e}")

    async def _close_position(self, symbol: str, position: Trade, reason: str) -> None:
        """ポジションクローズ"""
        try:
            current_price = self.market_data[symbol].price

            # 反対売買の注文を生成
            close_trade_type = (
                TradeType.SELL
                if position.trade_type == TradeType.BUY
                else TradeType.BUY
            )

            close_order = OrderRequest(
                symbol=symbol,
                order_type=OrderType.MARKET,
                trade_type=close_trade_type,
                quantity=position.quantity,
                price=current_price,
            )

            await self._execute_order(close_order)

            # アクティブポジションから削除
            if (
                symbol in self.active_positions
                and position in self.active_positions[symbol]
            ):
                self.active_positions[symbol].remove(position)

            logger.info(f"ポジションクローズ - {symbol}: {reason} @{current_price}")

        except Exception as e:
            logger.error(f"ポジションクローズエラー: {e}")

    def _calculate_daily_pnl(self) -> Decimal:
        """日次損益計算"""
        try:
            # 簡略化された実装 - 実際はより詳細な計算が必要
            daily_trades = [
                trade
                for positions in self.active_positions.values()
                for trade in positions
                if trade.timestamp.date() == datetime.now().date()
            ]

            pnl = Decimal("0")
            for trade in daily_trades:
                # 簡単な未実現損益計算
                if trade.symbol in self.market_data:
                    current_price = self.market_data[trade.symbol].price
                    if trade.trade_type == TradeType.BUY:
                        pnl += (current_price - trade.price) * trade.quantity
                    else:
                        pnl += (trade.price - current_price) * trade.quantity

            return pnl

        except Exception as e:
            logger.error(f"日次損益計算エラー: {e}")
            return Decimal("0")

    def _update_performance_stats(self, execution_time: float) -> None:
        """パフォーマンス統計更新"""
        try:
            # 移動平均で実行時間を更新
            if self.execution_stats["avg_execution_time"] == 0:
                self.execution_stats["avg_execution_time"] = execution_time
            else:
                alpha = 0.1  # 指数移動平均の重み
                self.execution_stats["avg_execution_time"] = (
                    alpha * execution_time
                    + (1 - alpha) * self.execution_stats["avg_execution_time"]
                )

            self.execution_stats["last_update"] = datetime.now()

        except Exception as e:
            logger.error(f"パフォーマンス統計更新エラー: {e}")

    def get_status(self) -> Dict[str, Any]:
        """エンジンの現在状態を取得"""
        return {
            "status": self.status.value,
            "monitored_symbols": len(self.symbols),
            "active_positions": sum(len(pos) for pos in self.active_positions.values()),
            "pending_orders": len(self.pending_orders),
            "daily_pnl": float(self._calculate_daily_pnl()),
            "execution_stats": self.execution_stats.copy(),
            "market_data_age": {
                symbol: (datetime.now() - data.timestamp).total_seconds()
                for symbol, data in self.market_data.items()
            },
        }

    def emergency_stop(self) -> None:
        """緊急停止 - 全ポジションクローズと取引停止"""
        logger.critical("緊急停止が実行されました")
        self.status = EngineStatus.STOPPED
        self._stop_event.set()

        # 保留中の注文をキャンセル
        self.pending_orders.clear()

        # TODO: 将来的には全ポジションの強制決済も実装
        logger.critical("緊急停止完了")
