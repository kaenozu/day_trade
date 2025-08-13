"""
市場分析エンジンコア実装（旧：自動取引エンジン）

【重要】自動取引機能は完全無効化済み
リアルタイム市場データを処理し、分析情報を提供するエンジン

※ 実際の取引実行は一切行いません（セーフモード）
※ 分析・情報提供・手動取引支援のみ
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
from ..config.trading_mode_config import get_current_trading_config, is_safe_mode
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
    """注文タイプ（分析用のみ）"""

    MARKET = "market"  # 成行
    LIMIT = "limit"  # 指値
    STOP = "stop"  # 逆指値
    STOP_LIMIT = "stop_limit"  # 逆指値限定


@dataclass
class RiskParameters:
    """リスク管理パラメータ（分析用のみ）"""

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
    """注文リクエスト（分析用のみ - 実際の注文は実行されない）"""

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
    市場分析エンジンコア（旧：自動取引エンジン）

    【重要】自動取引機能は完全無効化

    主要機能:
    1. リアルタイム市場データ処理
    2. シグナル生成と評価
    3. 分析情報提供（取引実行なし）
    4. 手動取引支援
    5. ポジション監視（情報提供のみ）
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
        self.pending_orders: List[OrderRequest] = []  # 分析用のみ

        # パフォーマンス監視
        self.execution_stats = {
            "signals_generated": 0,
            "analysis_completed": 0,
            "avg_execution_time": 0.0,
            "last_update": None,
        }

        # 非同期処理用
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._stop_event = asyncio.Event()

        # 安全設定の確認
        self.trading_config = get_current_trading_config()
        if not is_safe_mode():
            raise ValueError("安全設定が無効です。自動取引は許可されていません")

        logger.info(f"MarketAnalysisEngine初期化完了 - 監視銘柄: {len(symbols)}")
        logger.info("セーフモード: 有効 - 自動取引は完全に無効化されています")
        logger.info("※ 分析・情報提供・手動取引支援のみ実行します")

    async def start(self) -> None:
        """市場分析エンジン開始"""
        if self.status == EngineStatus.RUNNING:
            logger.warning("市場分析エンジンは既に実行中です")
            return

        logger.info("市場分析エンジンを開始します（セーフモード）...")
        logger.info("※ 自動取引は完全に無効化されています")
        self.status = EngineStatus.RUNNING
        self._stop_event.clear()

        # メインループ開始
        await self._main_loop()

    async def stop(self) -> None:
        """市場分析エンジン停止"""
        logger.info("市場分析エンジン停止要求受信")
        self.status = EngineStatus.STOPPED
        self._stop_event.set()

        # 保留中の分析要求をクリア
        self.pending_orders.clear()
        logger.info("市場分析エンジンが停止しました")

    async def pause(self) -> None:
        """市場分析エンジン一時停止"""
        if self.status == EngineStatus.RUNNING:
            self.status = EngineStatus.PAUSED
            logger.info("市場分析エンジンを一時停止しました")

    async def resume(self) -> None:
        """市場分析エンジン再開"""
        if self.status == EngineStatus.PAUSED:
            self.status = EngineStatus.RUNNING
            logger.info("市場分析エンジンを再開しました")

    async def _main_loop(self) -> None:
        """メインループ - エンジンの中核処理（分析のみ）"""
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

                # 3. シグナル分析（取引実行なし）
                await self._analyze_signals(signals)

                # 4. ポジション分析（情報提供のみ）
                await self._monitor_positions()

                # 5. パフォーマンス更新
                self._update_performance_stats(time.time() - loop_start)

                # インターバル調整
                await asyncio.sleep(self.update_interval)

        except Exception as e:
            self.status = EngineStatus.ERROR
            logger.error(f"市場分析エンジンでエラーが発生: {e}")
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
        """シグナル生成（分析のみ）"""
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

    async def _analyze_signals(self, signals: List[Tuple[str, TradingSignal]]) -> None:
        """シグナル分析（取引実行なし）"""
        for symbol, signal in signals:
            try:
                # シグナル分析のみ実行（取引実行は一切行わない）
                if signal.confidence >= 70.0 and signal.strength.value in [
                    "strong",
                    "medium",
                ]:
                    self._log_signal_analysis(symbol, signal)
                    self.execution_stats["analysis_completed"] += 1

            except Exception as e:
                logger.error(f"シグナル分析エラー - {symbol}: {e}")

    def _log_signal_analysis(self, symbol: str, signal: TradingSignal) -> None:
        """シグナル分析結果をログ出力（取引提案のみ）"""
        try:
            action = "買い推奨" if signal.signal_type.value == "buy" else "売り推奨"
            logger.info(
                f"【分析結果】取引提案 - {symbol}: {action} "
                f"(信頼度: {signal.confidence:.1f}%, 強度: {signal.strength.value})"
            )
            logger.info(
                f"  理由: {signal.reasoning if hasattr(signal, 'reasoning') else '分析結果による'}"
            )
            logger.info("  ※ 注意: これは分析情報であり、実際の取引実行は行いません")
            logger.info("  ※ 手動で取引を実行する場合は、十分な検討を行ってください")
        except Exception as e:
            logger.error(f"シグナル分析ログエラー - {symbol}: {e}")

    async def _monitor_positions(self) -> None:
        """ポジション分析 - 情報提供のみ（実際の売買は実行しない）"""
        try:
            for symbol, positions in self.active_positions.items():
                if symbol in self.market_data:
                    current_price = self.market_data[symbol].price

                    for position in positions[:]:
                        self._analyze_position_status(symbol, position, current_price)

        except Exception as e:
            logger.error(f"ポジション分析エラー: {e}")

    def _analyze_position_status(
        self, symbol: str, position: Trade, current_price: Decimal
    ) -> None:
        """個別ポジションの分析（情報提供のみ）"""
        try:
            entry_price = position.price
            pnl_ratio = (current_price - entry_price) / entry_price
            pnl_amount = (current_price - entry_price) * position.quantity

            # 買いポジションの分析
            if position.trade_type == TradeType.BUY:
                if pnl_ratio >= self.risk_params.take_profit_ratio:
                    logger.info(
                        f"【分析】{symbol}: 利益確定推奨 - 利益率 {pnl_ratio * 100:.2f}% (+{pnl_amount:.0f}円)"
                    )
                elif pnl_ratio <= -self.risk_params.stop_loss_ratio:
                    logger.info(
                        f"【分析】{symbol}: 損切り推奨 - 損失率 {pnl_ratio * 100:.2f}% ({pnl_amount:.0f}円)"
                    )
                else:
                    logger.info(
                        f"【分析】{symbol}: ホールド推奨 - 損益率 {pnl_ratio * 100:.2f}% ({pnl_amount:+.0f}円)"
                    )

            # 売りポジションの分析
            elif position.trade_type == TradeType.SELL:
                if pnl_ratio <= -self.risk_params.take_profit_ratio:
                    logger.info(
                        f"【分析】{symbol}: 利益確定推奨 - 利益率 {abs(pnl_ratio) * 100:.2f}% (+{abs(pnl_amount):.0f}円)"
                    )
                elif pnl_ratio >= self.risk_params.stop_loss_ratio:
                    logger.info(
                        f"【分析】{symbol}: 損切り推奨 - 損失率 {pnl_ratio * 100:.2f}% ({pnl_amount:.0f}円)"
                    )
                else:
                    logger.info(
                        f"【分析】{symbol}: ホールド推奨 - 損益率 {pnl_ratio * 100:.2f}% ({pnl_amount:+.0f}円)"
                    )

            logger.info("  ※ 注意: これは分析情報であり、実際の取引実行は行いません")
            logger.info("  ※ 手動で取引を実行する場合は、十分な検討を行ってください")

        except Exception as e:
            logger.error(f"ポジション分析エラー: {e}")

    def _calculate_daily_pnl(self) -> Decimal:
        """日次損益計算（分析用）"""
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
            "pending_analysis": len(self.pending_orders),
            "daily_pnl": float(self._calculate_daily_pnl()),
            "safe_mode": is_safe_mode(),
            "trading_disabled": not self.trading_config.enable_automatic_trading,
            "execution_stats": self.execution_stats.copy(),
            "market_data_age": {
                symbol: (datetime.now() - data.timestamp).total_seconds()
                for symbol, data in self.market_data.items()
            },
        }

    def emergency_stop(self) -> None:
        """緊急停止 - 分析エンジン停止（取引実行機能なし）"""
        logger.critical("緊急停止が実行されました（分析エンジンのみ停止）")
        self.status = EngineStatus.STOPPED
        self._stop_event.set()

        # 保留中の分析要求をクリア（実際の取引は発生していない）
        self.pending_orders.clear()

        logger.critical("緊急停止完了 - 市場分析エンジンが停止されました")
        logger.critical(
            "※ 注意: 実際の取引ポジションがある場合は手動で確認・対処してください"
        )

    def add_manual_trade(self, trade: Trade) -> None:
        """手動取引の追加（分析用）"""
        try:
            logger.info(
                f"手動取引を記録します: {trade.symbol} {trade.trade_type.value} {trade.quantity}株"
            )

            # TradeManager に記録
            self.trade_manager.add_trade(trade)

            # アクティブポジションに追加
            if trade.symbol not in self.active_positions:
                self.active_positions[trade.symbol] = []
            self.active_positions[trade.symbol].append(trade)

            logger.info("手動取引の記録が完了しました")
            logger.info("※ 今後のポジション分析に含まれます")

        except Exception as e:
            logger.error(f"手動取引記録エラー: {e}")

    def get_trading_suggestions(self, symbol: str) -> List[str]:
        """取引提案の取得（情報提供のみ）"""
        suggestions = []
        try:
            if symbol in self.market_data:
                current_price = self.market_data[symbol].price

                # 基本的な分析情報を提供
                suggestions.append(f"現在価格: {current_price}円")
                suggestions.append("※ これは分析情報です")
                suggestions.append("※ 実際の取引は手動で慎重に行ってください")
                suggestions.append("※ 自動取引は完全に無効化されています")

        except Exception as e:
            logger.error(f"取引提案取得エラー - {symbol}: {e}")
            suggestions.append("分析情報の取得に失敗しました")

        return suggestions
