"""
拡張自動取引エンジン

AdvancedOrderManagerとPortfolioManagerを統合した
高度な自動取引システム。
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..analysis.signals import TradingSignal, TradingSignalGenerator
from ..core.trade_manager import Trade, TradeManager, TradeType
from ..data.stock_fetcher import StockFetcher
from ..utils.enhanced_error_handler import get_default_error_handler
from ..utils.logging_config import get_context_logger
from .advanced_order_manager import AdvancedOrderManager, Order, OrderType
from .portfolio_manager import PortfolioManager
from .trading_engine import EngineStatus, MarketData, RiskParameters

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()


class ExecutionMode(Enum):
    """実行モード"""

    CONSERVATIVE = "conservative"  # 保守的実行
    AGGRESSIVE = "aggressive"  # 積極的実行
    BALANCED = "balanced"  # バランス実行


class EnhancedTradingEngine:
    """
    拡張自動取引エンジン

    主要機能:
    1. 高度な注文管理（OCO、IFD、トレーリング等）
    2. ポートフォリオレベルのリスク管理
    3. 複数注文の同期実行
    4. パフォーマンス最適化
    5. 包括的監視・統計
    """

    def __init__(
        self,
        symbols: List[str],
        trade_manager: Optional[TradeManager] = None,
        signal_generator: Optional[TradingSignalGenerator] = None,
        stock_fetcher: Optional[StockFetcher] = None,
        risk_params: Optional[RiskParameters] = None,
        initial_cash: Decimal = Decimal("1000000"),
        execution_mode: ExecutionMode = ExecutionMode.BALANCED,
        update_interval: float = 1.0,
    ):
        self.symbols = symbols
        self.execution_mode = execution_mode
        self.update_interval = update_interval

        # コンポーネント初期化
        self.trade_manager = trade_manager or TradeManager()
        self.order_manager = AdvancedOrderManager(self.trade_manager)
        self.portfolio_manager = PortfolioManager(initial_cash)
        self.signal_generator = signal_generator or TradingSignalGenerator()
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.risk_params = risk_params or RiskParameters()

        # エンジン状態
        self.status = EngineStatus.STOPPED
        self.market_data: Dict[str, MarketData] = {}

        # 実行統計
        self.execution_stats = {
            "engine_cycles": 0,
            "signals_processed": 0,
            "orders_generated": 0,
            "risk_violations": 0,
            "avg_cycle_time": 0.0,
            "last_update": None,
            "uptime_seconds": 0,
        }

        # 非同期処理
        self.executor = ThreadPoolExecutor(max_workers=6)
        self._stop_event = asyncio.Event()
        self.start_time = None

        logger.info(
            f"拡張取引エンジン初期化完了 - "
            f"監視銘柄: {len(symbols)}, "
            f"実行モード: {execution_mode.value}, "
            f"初期資金: {initial_cash:,}"
        )

    async def start(self) -> None:
        """エンジン開始"""
        if self.status == EngineStatus.RUNNING:
            logger.warning("取引エンジンは既に実行中です")
            return

        logger.info("拡張取引エンジンを開始します...")
        self.status = EngineStatus.RUNNING
        self.start_time = time.time()
        self._stop_event.clear()

        try:
            await self._main_execution_loop()
        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"エンジン実行エラー: {e}")
            error_handler.handle_error(e, context={"engine_status": self.status.value})

    async def stop(self) -> None:
        """エンジン停止"""
        logger.info("拡張取引エンジン停止要求受信")
        self.status = EngineStatus.STOPPED
        self._stop_event.set()

        # 全注文をキャンセル
        cancelled_count = await self.order_manager.cancel_all_orders()
        logger.info(f"停止時注文キャンセル: {cancelled_count}件")

        # 日次スナップショット作成
        self.portfolio_manager.create_daily_snapshot()

        logger.info("拡張取引エンジンが停止しました")

    async def pause(self) -> None:
        """エンジン一時停止"""
        if self.status == EngineStatus.RUNNING:
            self.status = EngineStatus.PAUSED
            logger.info("拡張取引エンジンを一時停止しました")

    async def resume(self) -> None:
        """エンジン再開"""
        if self.status == EngineStatus.PAUSED:
            self.status = EngineStatus.RUNNING
            logger.info("拡張取引エンジンを再開しました")

    async def _main_execution_loop(self) -> None:
        """メイン実行ループ"""
        cycle_count = 0

        while not self._stop_event.is_set() and self.status != EngineStatus.STOPPED:
            if self.status == EngineStatus.PAUSED:
                await asyncio.sleep(1.0)
                continue

            cycle_start = time.time()

            try:
                # 1. 市場データ更新
                await self._update_market_data()

                # 2. 注文実行チェック
                await self._process_pending_orders()

                # 3. シグナル生成と分析
                signals = await self._generate_trading_signals()

                # 4. リスク管理チェック
                risk_check = self._comprehensive_risk_check()

                # 5. 新規注文生成・実行
                if risk_check["approved"]:
                    await self._process_trading_signals(signals)
                else:
                    self.execution_stats["risk_violations"] += 1
                    logger.warning(
                        f"リスク制限により注文実行を停止: {risk_check['reason']}"
                    )

                # 6. ポートフォリオ状態更新
                self._update_portfolio_state()

                # 7. パフォーマンス統計更新
                cycle_time = time.time() - cycle_start
                self._update_execution_stats(cycle_time)

                cycle_count += 1

                # インターバル調整
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"実行ループエラー: {e}")
                error_handler.handle_error(
                    e,
                    context={
                        "cycle_count": cycle_count,
                        "engine_status": self.status.value,
                    },
                )

                # エラー時は短時間待機してリトライ
                await asyncio.sleep(5.0)

    async def _update_market_data(self) -> None:
        """市場データ更新"""
        try:
            # 並行処理で全銘柄の価格を取得
            loop = asyncio.get_event_loop()
            tasks = []

            for symbol in self.symbols:
                task = loop.run_in_executor(
                    self.executor, self.stock_fetcher.get_current_price, symbol
                )
                tasks.append((symbol, task))

            # 価格データを並行取得
            price_updates = {}
            for symbol, task in tasks:
                try:
                    price_data = await task
                    if price_data:
                        market_data = MarketData(
                            symbol=symbol,
                            price=Decimal(str(price_data["current_price"])),
                            volume=price_data.get("volume", 0),
                            timestamp=datetime.now(),
                        )
                        self.market_data[symbol] = market_data
                        price_updates[symbol] = market_data.price

                except Exception as e:
                    logger.warning(f"価格取得失敗 {symbol}: {e}")

            # ポートフォリオマネージャーの価格更新
            if price_updates:
                self.portfolio_manager.update_market_prices(price_updates)

        except Exception as e:
            logger.error(f"市場データ更新エラー: {e}")

    async def _process_pending_orders(self) -> None:
        """待機中注文の処理"""
        try:
            # 各銘柄について注文実行をチェック
            for symbol, market_data in self.market_data.items():
                fills = await self.order_manager.process_market_update(
                    symbol, market_data.price, market_data.volume
                )

                # 約定があった場合はポートフォリオに反映
                for fill in fills:
                    trade = Trade(
                        id=fill.fill_id,
                        symbol=fill.symbol,
                        trade_type=fill.side,
                        quantity=fill.quantity,
                        price=fill.price,
                        timestamp=fill.timestamp,
                        commission=fill.commission,
                        status="executed",
                    )
                    self.portfolio_manager.add_trade(trade)

        except Exception as e:
            logger.error(f"待機注文処理エラー: {e}")

    async def _generate_trading_signals(self) -> List[Tuple[str, TradingSignal]]:
        """取引シグナル生成"""
        signals = []

        try:
            tasks = []
            loop = asyncio.get_event_loop()

            for symbol in self.symbols:
                if symbol in self.market_data:
                    task = loop.run_in_executor(
                        self.executor, self._generate_symbol_signal, symbol
                    )
                    tasks.append((symbol, task))

            # 全シグナルを並行生成
            for symbol, task in tasks:
                try:
                    signal = await task
                    if signal:
                        signals.append((symbol, signal))
                        self.execution_stats["signals_processed"] += 1
                except Exception as e:
                    logger.warning(f"シグナル生成失敗 {symbol}: {e}")

        except Exception as e:
            logger.error(f"シグナル生成エラー: {e}")

        return signals

    def _generate_symbol_signal(self, symbol: str) -> Optional[TradingSignal]:
        """個別銘柄のシグナル生成"""
        try:
            # 履歴データを取得
            historical_data = self.stock_fetcher.get_historical_data(symbol, "30d")

            if historical_data is not None and not historical_data.empty:
                return self.signal_generator.generate_signal(historical_data)

        except Exception as e:
            logger.warning(f"個別シグナル生成エラー {symbol}: {e}")

        return None

    def _comprehensive_risk_check(self) -> Dict[str, Any]:
        """包括的リスク管理チェック"""
        try:
            # ポートフォリオレベルのリスクチェック
            portfolio_risk = self.portfolio_manager.check_risk_limits()

            # 個別リスクパラメータのチェック
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()

            # リスク承認判定
            approved = True
            reasons = []

            # ポートフォリオリスク違反チェック
            if portfolio_risk["total_violations"] > 0:
                approved = False
                reasons.append(
                    f"ポートフォリオリスク違反: {portfolio_risk['total_violations']}件"
                )

            # 日次損失制限チェック
            if portfolio_summary.total_pnl <= -self.risk_params.max_daily_loss:
                approved = False
                reasons.append(f"日次最大損失到達: {portfolio_summary.total_pnl}")

            # 最大ポジション数チェック
            if portfolio_summary.total_positions >= self.risk_params.max_open_positions:
                approved = False
                reasons.append(
                    f"最大ポジション数到達: {portfolio_summary.total_positions}"
                )

            return {
                "approved": approved,
                "reason": "; ".join(reasons) if reasons else "OK",
                "portfolio_risk": portfolio_risk,
                "risk_score": portfolio_risk.get("risk_score", 0.0),
            }

        except Exception as e:
            logger.error(f"リスクチェックエラー: {e}")
            return {"approved": False, "reason": f"リスクチェックエラー: {e}"}

    async def _process_trading_signals(
        self, signals: List[Tuple[str, TradingSignal]]
    ) -> None:
        """取引シグナル処理"""
        try:
            for symbol, signal in signals:
                # シグナル強度による実行判定
                min_confidence = self._get_min_confidence_threshold()

                if signal.confidence >= min_confidence:
                    await self._create_and_submit_orders(symbol, signal)

        except Exception as e:
            logger.error(f"シグナル処理エラー: {e}")

    def _get_min_confidence_threshold(self) -> float:
        """実行モードに応じた最小信頼度閾値"""
        thresholds = {
            ExecutionMode.CONSERVATIVE: 80.0,
            ExecutionMode.BALANCED: 70.0,
            ExecutionMode.AGGRESSIVE: 60.0,
        }
        return thresholds.get(self.execution_mode, 70.0)

    async def _create_and_submit_orders(
        self, symbol: str, signal: TradingSignal
    ) -> None:
        """シグナルから注文を生成・提出"""
        try:
            if symbol not in self.market_data:
                return

            current_price = self.market_data[symbol].price

            # ポジションサイズ計算
            position_size = self._calculate_position_size(symbol, signal)
            if position_size <= 0:
                return

            # 基本注文を作成
            main_order = Order(
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=TradeType.BUY
                if signal.signal_type.value == "buy"
                else TradeType.SELL,
                quantity=position_size,
                price=current_price,
            )

            # 実行モードに応じた注文戦略
            if self.execution_mode == ExecutionMode.CONSERVATIVE:
                await self._submit_conservative_orders(main_order, current_price)
            elif self.execution_mode == ExecutionMode.AGGRESSIVE:
                await self._submit_aggressive_orders(main_order, current_price)
            else:  # BALANCED
                await self._submit_balanced_orders(main_order, current_price)

            self.execution_stats["orders_generated"] += 1

        except Exception as e:
            logger.error(f"注文生成・提出エラー: {e}")

    def _calculate_position_size(self, symbol: str, signal: TradingSignal) -> int:
        """ポジションサイズ計算"""
        try:
            # 現在のポートフォリオ状況を取得
            self.portfolio_manager.get_portfolio_summary()
            current_price = self.market_data[symbol].price

            # 信頼度に基づく基本サイズ
            base_size = 100  # 基本100株
            confidence_multiplier = signal.confidence / 100.0

            # 実行モードによる調整
            mode_multipliers = {
                ExecutionMode.CONSERVATIVE: 0.5,
                ExecutionMode.BALANCED: 1.0,
                ExecutionMode.AGGRESSIVE: 1.5,
            }
            mode_multiplier = mode_multipliers.get(self.execution_mode, 1.0)

            # 最終サイズ計算
            position_size = int(base_size * confidence_multiplier * mode_multiplier)

            # リスク制限による上限設定
            max_position_value = self.risk_params.max_position_size
            max_shares = int(max_position_value / current_price)

            return min(position_size, max_shares)

        except Exception as e:
            logger.error(f"ポジションサイズ計算エラー: {e}")
            return 0

    async def _submit_conservative_orders(
        self, main_order: Order, current_price: Decimal
    ) -> None:
        """保守的注文戦略"""
        # 指値注文を使用（スリッページ回避）
        if main_order.side == TradeType.BUY:
            main_order.order_type = OrderType.LIMIT
            main_order.price = current_price * Decimal("0.995")  # 0.5%下の指値
        else:
            main_order.order_type = OrderType.LIMIT
            main_order.price = current_price * Decimal("1.005")  # 0.5%上の指値

        await self.order_manager.submit_order(main_order)

    async def _submit_aggressive_orders(
        self, main_order: Order, current_price: Decimal
    ) -> None:
        """積極的注文戦略"""
        # 成行注文で即座実行
        main_order.order_type = OrderType.MARKET
        await self.order_manager.submit_order(main_order)

    async def _submit_balanced_orders(
        self, main_order: Order, current_price: Decimal
    ) -> None:
        """バランス注文戦略"""
        # OCO注文（指値+ストップ）
        limit_order = Order(
            symbol=main_order.symbol,
            order_type=OrderType.LIMIT,
            side=main_order.side,
            quantity=main_order.quantity // 2,  # 半分を指値
            price=current_price
            * (
                Decimal("0.998")
                if main_order.side == TradeType.BUY
                else Decimal("1.002")
            ),
        )

        market_order = Order(
            symbol=main_order.symbol,
            order_type=OrderType.MARKET,
            side=main_order.side,
            quantity=main_order.quantity - limit_order.quantity,  # 残りを成行
        )

        await self.order_manager.submit_order(limit_order)
        await self.order_manager.submit_order(market_order)

    def _update_portfolio_state(self) -> None:
        """ポートフォリオ状態更新"""
        try:
            # 定期的にパフォーマンス記録を更新
            self.portfolio_manager.create_daily_snapshot()

        except Exception as e:
            logger.error(f"ポートフォリオ状態更新エラー: {e}")

    def _update_execution_stats(self, cycle_time: float) -> None:
        """実行統計更新"""
        try:
            self.execution_stats["engine_cycles"] += 1

            # 移動平均でサイクル時間更新
            alpha = 0.1
            if self.execution_stats["avg_cycle_time"] == 0:
                self.execution_stats["avg_cycle_time"] = cycle_time
            else:
                self.execution_stats["avg_cycle_time"] = (
                    alpha * cycle_time
                    + (1 - alpha) * self.execution_stats["avg_cycle_time"]
                )

            self.execution_stats["last_update"] = datetime.now()

            if self.start_time:
                self.execution_stats["uptime_seconds"] = int(
                    time.time() - self.start_time
                )

        except Exception as e:
            logger.error(f"統計更新エラー: {e}")

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """包括的ステータス取得"""
        try:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            order_stats = self.order_manager.get_execution_statistics()
            performance_metrics = self.portfolio_manager.get_performance_metrics()

            return {
                "engine": {
                    "status": self.status.value,
                    "execution_mode": self.execution_mode.value,
                    "monitored_symbols": len(self.symbols),
                    "uptime_seconds": self.execution_stats["uptime_seconds"],
                    "engine_cycles": self.execution_stats["engine_cycles"],
                    "avg_cycle_time_ms": self.execution_stats["avg_cycle_time"] * 1000,
                },
                "portfolio": {
                    "total_equity": float(portfolio_summary.total_equity),
                    "total_pnl": float(portfolio_summary.total_pnl),
                    "unrealized_pnl": float(portfolio_summary.total_unrealized_pnl),
                    "realized_pnl": float(portfolio_summary.total_realized_pnl),
                    "total_positions": portfolio_summary.total_positions,
                    "long_positions": portfolio_summary.long_positions,
                    "short_positions": portfolio_summary.short_positions,
                    "cash": float(portfolio_summary.total_cash),
                },
                "orders": order_stats,
                "performance": performance_metrics,
                "risk": self.portfolio_manager.check_risk_limits(),
            }

        except Exception as e:
            logger.error(f"ステータス取得エラー: {e}")
            return {"error": str(e)}

    def emergency_stop(self) -> None:
        """緊急停止"""
        logger.critical("緊急停止実行中...")
        self.status = EngineStatus.STOPPED
        self._stop_event.set()

        # 非同期で全注文キャンセル
        asyncio.create_task(self.order_manager.cancel_all_orders())

        logger.critical("緊急停止完了")
