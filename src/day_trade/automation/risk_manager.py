"""
高度なリスク管理システム

自動取引システムの安全性を確保するための包括的な
リスク管理・監視・緊急停止機能を提供する。
"""

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from ..core.trade_manager import Trade, TradeType
from ..utils.enhanced_error_handler import get_default_error_handler
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()


class RiskLevel(Enum):
    """リスクレベル"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """アラートタイプ"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EmergencyReason(Enum):
    """緊急停止理由"""

    MANUAL = "manual"  # 手動停止
    LOSS_LIMIT = "loss_limit"  # 損失限界
    POSITION_LIMIT = "position_limit"  # ポジション限界
    VOLATILITY = "volatility"  # 異常ボラティリティ
    SYSTEM_ERROR = "system_error"  # システムエラー
    MARGIN_CALL = "margin_call"  # マージンコール
    CIRCUIT_BREAKER = "circuit_breaker"  # サーキットブレーカー


@dataclass
class RiskAlert:
    """リスクアラート"""

    alert_id: str
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    symbol: Optional[str] = None
    current_value: Optional[Decimal] = None
    limit_value: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


@dataclass
class RiskLimits:
    """包括的リスク制限"""

    # ポジションリスク
    max_position_size: Decimal = Decimal("500000")  # 最大単一ポジション
    max_total_exposure: Decimal = Decimal("2000000")  # 最大総エクスポージャー
    max_open_positions: int = 10  # 最大保有銘柄数
    max_sector_exposure: Decimal = Decimal("1000000")  # セクター別最大エクスポージャー

    # 損失制限
    max_daily_loss: Decimal = Decimal("100000")  # 日次最大損失
    max_drawdown: Decimal = Decimal("200000")  # 最大ドローダウン
    max_consecutive_losses: int = 5  # 連続損失回数
    stop_loss_ratio: Decimal = Decimal("0.03")  # 個別ストップロス率

    # 取引頻度制限
    max_daily_trades: int = 100  # 日次最大取引数
    max_orders_per_minute: int = 10  # 分間最大注文数
    min_position_hold_time: int = 300  # 最小保有時間（秒）

    # ボラティリティ制限
    max_position_volatility: Decimal = Decimal("0.05")  # ポジション最大ボラティリティ
    volatility_window: int = 20  # ボラティリティ計算期間

    # システム制限
    max_api_calls_per_minute: int = 100  # API呼び出し制限
    max_memory_usage_mb: int = 1000  # メモリ使用制限
    max_cpu_usage_percent: Decimal = Decimal("80")  # CPU使用率制限


@dataclass
class RiskMetrics:
    """リスクメトリクス"""

    # 現在の状態
    total_exposure: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    current_drawdown: Decimal = Decimal("0")
    active_positions: int = 0

    # 統計
    daily_trades: int = 0
    consecutive_losses: int = 0
    orders_last_minute: int = 0

    # パフォーマンス
    sharpe_ratio: Decimal = Decimal("0")
    volatility: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")

    # システム
    cpu_usage: Decimal = Decimal("0")
    memory_usage_mb: int = 0
    api_calls_last_minute: int = 0

    # タイムスタンプ
    last_updated: datetime = field(default_factory=datetime.now)


class RiskManager:
    """
    高度なリスク管理システム

    主要機能:
    1. リアルタイムリスク監視
    2. 注文前リスクバリデーション
    3. ポジションサイズ最適化
    4. 緊急停止システム
    5. アラート生成・管理
    6. リスクレポート作成
    """

    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
        alert_callback: Optional[Callable[[RiskAlert], None]] = None,
        emergency_callback: Optional[Callable[[EmergencyReason, str], None]] = None,
    ):
        self.risk_limits = risk_limits or RiskLimits()
        self.alert_callback = alert_callback
        self.emergency_callback = emergency_callback

        # 状態管理
        self.is_emergency_stopped = False
        self.risk_metrics = RiskMetrics()
        self.active_alerts: Dict[str, RiskAlert] = {}

        # 履歴・統計
        self.trade_history: Deque[Trade] = deque(maxlen=1000)
        self.pnl_history: Deque[Tuple[datetime, Decimal]] = deque(maxlen=1000)
        self.alert_history: Deque[RiskAlert] = deque(maxlen=500)

        # 監視用データ
        self.recent_orders: Deque[datetime] = deque(maxlen=100)
        self.api_calls: Deque[datetime] = deque(maxlen=1000)
        self.position_opens: Dict[str, datetime] = {}  # symbol -> open_time

        # セクター分類（簡易版）
        self.sector_mapping: Dict[str, str] = {
            "7203": "automotive",  # トヨタ
            "6758": "electronics",  # ソニー
            "9984": "technology",  # ソフトバンク
            "4755": "technology",  # 楽天
            "8058": "trading",  # 三菱商事
        }

        # 非同期処理用
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False

        logger.info("リスク管理システムが初期化されました")

    async def start_monitoring(self) -> None:
        """リアルタイム監視開始"""
        if self._monitoring_task is not None:
            logger.warning("リスク監視は既に実行中です")
            return

        self._stop_monitoring = False
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("リアルタイムリスク監視を開始しました")

    async def stop_monitoring(self) -> None:
        """監視停止"""
        self._stop_monitoring = True
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task
            self._monitoring_task = None

        logger.info("リスク監視を停止しました")

    def validate_order(
        self,
        symbol: str,
        trade_type: TradeType,
        quantity: int,
        price: Decimal,
        current_portfolio: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        注文前リスクバリデーション

        Returns:
            (approved: bool, reason: str)
        """
        try:
            # 緊急停止チェック
            if self.is_emergency_stopped:
                return False, "緊急停止中のため注文を拒否"

            # 注文頻度チェック
            if not self._check_order_frequency():
                return False, "注文頻度制限を超過"

            # ポジションサイズチェック
            position_value = quantity * price
            if position_value > self.risk_limits.max_position_size:
                return (
                    False,
                    f"単一ポジションサイズ制限を超過: {position_value} > {self.risk_limits.max_position_size}",
                )

            # 総エクスポージャーチェック
            total_exposure = self._calculate_total_exposure(
                current_portfolio, symbol, quantity, price
            )
            if total_exposure > self.risk_limits.max_total_exposure:
                return (
                    False,
                    f"総エクスポージャー制限を超過: {total_exposure} > {self.risk_limits.max_total_exposure}",
                )

            # ポジション数チェック
            active_positions = len(current_portfolio.get("positions", {}))
            if active_positions >= self.risk_limits.max_open_positions:
                return (
                    False,
                    f"最大ポジション数を超過: {active_positions} >= {self.risk_limits.max_open_positions}",
                )

            # セクター集中リスクチェック
            if not self._check_sector_concentration(
                symbol, quantity, price, current_portfolio
            ):
                return False, "セクター集中リスク制限を超過"

            # 日次取引数チェック
            if self.risk_metrics.daily_trades >= self.risk_limits.max_daily_trades:
                return (
                    False,
                    f"日次取引数制限を超過: {self.risk_metrics.daily_trades} >= {self.risk_limits.max_daily_trades}",
                )

            # ボラティリティチェック
            if not self._check_volatility_risk(symbol, quantity, price):
                return False, "ボラティリティリスクが高すぎます"

            return True, "リスクチェック合格"

        except Exception as e:
            logger.error(f"注文バリデーションエラー: {e}")
            error_handler.handle_error(
                e, context={"symbol": symbol, "quantity": quantity}
            )
            return False, f"バリデーションエラー: {e}"

    def calculate_optimal_position_size(
        self,
        symbol: str,
        signal_confidence: float,
        current_price: Decimal,
        portfolio_equity: Decimal,
        volatility: Optional[Decimal] = None,
    ) -> int:
        """
        最適ポジションサイズ計算

        Kelly基準とリスクパリティを組み合わせた動的サイジング
        """
        try:
            # 基準ポジションサイズ（資産の一定割合）
            base_allocation = portfolio_equity * Decimal("0.02")  # 2%

            # 信頼度による調整
            confidence_multiplier = Decimal(str(signal_confidence / 100.0))

            # ボラティリティ調整
            volatility_adj = Decimal("1.0")
            if volatility is not None and volatility > 0:
                # 高ボラティリティでサイズ縮小
                volatility_adj = min(Decimal("1.0"), Decimal("0.05") / volatility)

            # Kelly基準の簡易版（勝率60%、平均利益/損失=1.5と仮定）
            kelly_fraction = Decimal("0.15")  # 保守的Kelly

            # 最適サイズ計算
            optimal_value = (
                base_allocation
                * confidence_multiplier
                * volatility_adj
                * kelly_fraction
            )

            # 制限内に収める
            max_value = min(
                self.risk_limits.max_position_size,
                portfolio_equity * Decimal("0.1"),  # 資産の10%まで
            )
            final_value = min(optimal_value, max_value)

            # 株数に変換
            shares = int(final_value / current_price)

            # 最小・最大調整
            shares = max(shares, 1)  # 最低1株
            shares = min(shares, 10000)  # 最大10,000株

            logger.debug(
                f"ポジションサイズ計算: {symbol} - {shares}株 (価値: {shares * current_price:,}円)"
            )

            return shares

        except Exception as e:
            logger.error(f"ポジションサイズ計算エラー: {e}")
            # エラー時は保守的な固定サイズ
            return 100

    def monitor_positions(
        self, positions: Dict[str, Any], market_data: Dict[str, Decimal]
    ) -> List[RiskAlert]:
        """
        ポジションリスク監視

        Returns:
            List[RiskAlert]: 生成されたアラートリスト
        """
        alerts = []

        try:
            for symbol, position in positions.items():
                if symbol not in market_data:
                    continue

                current_price = market_data[symbol]
                position_value = abs(position.get("quantity", 0)) * current_price

                # 個別ポジションサイズチェック
                if position_value > self.risk_limits.max_position_size:
                    alert = self._create_alert(
                        AlertType.WARNING,
                        RiskLevel.HIGH,
                        f"ポジションサイズ制限超過: {symbol}",
                        symbol=symbol,
                        current_value=position_value,
                        limit_value=self.risk_limits.max_position_size,
                    )
                    alerts.append(alert)

                # ストップロス監視
                entry_price = position.get("average_price", current_price)
                if position.get("quantity", 0) > 0:  # ロングポジション
                    loss_ratio = (entry_price - current_price) / entry_price
                    if loss_ratio >= self.risk_limits.stop_loss_ratio:
                        alert = self._create_alert(
                            AlertType.ERROR,
                            RiskLevel.CRITICAL,
                            f"ストップロス到達: {symbol} ({loss_ratio*100:.1f}%)",
                            symbol=symbol,
                            current_value=current_price,
                            limit_value=entry_price
                            * (Decimal("1") - self.risk_limits.stop_loss_ratio),
                        )
                        alerts.append(alert)

                elif position.get("quantity", 0) < 0:  # ショートポジション
                    loss_ratio = (current_price - entry_price) / entry_price
                    if loss_ratio >= self.risk_limits.stop_loss_ratio:
                        alert = self._create_alert(
                            AlertType.ERROR,
                            RiskLevel.CRITICAL,
                            f"ショートストップロス到達: {symbol} ({loss_ratio*100:.1f}%)",
                            symbol=symbol,
                            current_value=current_price,
                            limit_value=entry_price
                            * (Decimal("1") + self.risk_limits.stop_loss_ratio),
                        )
                        alerts.append(alert)

                # 保有期間チェック
                open_time = self.position_opens.get(symbol)
                if open_time:
                    hold_time = (datetime.now() - open_time).total_seconds()
                    if hold_time < self.risk_limits.min_position_hold_time:
                        alert = self._create_alert(
                            AlertType.WARNING,
                            RiskLevel.MEDIUM,
                            f"最小保有時間未満でのクローズ試行: {symbol}",
                            symbol=symbol,
                        )
                        alerts.append(alert)

            # 総合リスクチェック
            total_exposure = sum(
                abs(pos.get("quantity", 0)) * market_data.get(symbol, Decimal("0"))
                for symbol, pos in positions.items()
                if symbol in market_data
            )

            if total_exposure > self.risk_limits.max_total_exposure:
                alert = self._create_alert(
                    AlertType.ERROR,
                    RiskLevel.CRITICAL,
                    f"総エクスポージャー制限超過: {total_exposure:,}円",
                    current_value=total_exposure,
                    limit_value=self.risk_limits.max_total_exposure,
                )
                alerts.append(alert)

            # アラートを処理
            for alert in alerts:
                self._process_alert(alert)

            return alerts

        except Exception as e:
            logger.error(f"ポジション監視エラー: {e}")
            error_handler.handle_error(e, context={"positions": len(positions)})
            return []

    def check_emergency_conditions(
        self, portfolio_summary: Dict[str, Any]
    ) -> Optional[EmergencyReason]:
        """
        緊急停止条件チェック

        Returns:
            EmergencyReason: 緊急停止が必要な場合の理由、なければNone
        """
        try:
            # 日次損失チェック
            daily_pnl = portfolio_summary.get("daily_pnl", Decimal("0"))
            if daily_pnl <= -self.risk_limits.max_daily_loss:
                return EmergencyReason.LOSS_LIMIT

            # ドローダウンチェック
            current_dd = portfolio_summary.get("current_drawdown", Decimal("0"))
            if current_dd >= self.risk_limits.max_drawdown:
                return EmergencyReason.LOSS_LIMIT

            # 連続損失チェック
            if (
                self.risk_metrics.consecutive_losses
                >= self.risk_limits.max_consecutive_losses
            ):
                return EmergencyReason.LOSS_LIMIT

            # ポジション数チェック
            active_positions = portfolio_summary.get("total_positions", 0)
            if active_positions > self.risk_limits.max_open_positions * 1.2:  # 20%余裕
                return EmergencyReason.POSITION_LIMIT

            # システムリソースチェック
            if self.risk_metrics.cpu_usage > self.risk_limits.max_cpu_usage_percent:
                return EmergencyReason.SYSTEM_ERROR

            if self.risk_metrics.memory_usage_mb > self.risk_limits.max_memory_usage_mb:
                return EmergencyReason.SYSTEM_ERROR

            return None

        except Exception as e:
            logger.error(f"緊急停止条件チェックエラー: {e}")
            return EmergencyReason.SYSTEM_ERROR

    async def trigger_emergency_stop(
        self, reason: EmergencyReason, additional_info: str = ""
    ) -> None:
        """
        緊急停止実行
        """
        if self.is_emergency_stopped:
            logger.warning("既に緊急停止状態です")
            return

        start_time = time.time()
        self.is_emergency_stopped = True

        logger.critical(f"緊急停止実行中 - 理由: {reason.value} - {additional_info}")

        # 緊急アラート生成
        alert = self._create_alert(
            AlertType.CRITICAL,
            RiskLevel.CRITICAL,
            f"緊急停止実行: {reason.value} - {additional_info}",
        )
        self._process_alert(alert)

        # 緊急停止コールバック実行
        if self.emergency_callback:
            try:
                if asyncio.iscoroutinefunction(self.emergency_callback):
                    await self.emergency_callback(reason, additional_info)
                else:
                    self.emergency_callback(reason, additional_info)
            except Exception as e:
                logger.error(f"緊急停止コールバックエラー: {e}")

        execution_time = time.time() - start_time
        logger.critical(f"緊急停止完了 - 実行時間: {execution_time*1000:.1f}ms")

    def reset_emergency_stop(self, operator: str = "system") -> None:
        """緊急停止リセット（慎重に使用）"""
        if not self.is_emergency_stopped:
            logger.info("緊急停止状態ではありません")
            return

        self.is_emergency_stopped = False
        logger.warning(f"緊急停止がリセットされました - 操作者: {operator}")

        alert = self._create_alert(
            AlertType.WARNING, RiskLevel.HIGH, f"緊急停止リセット - 操作者: {operator}"
        )
        self._process_alert(alert)

    def get_risk_report(self) -> Dict[str, Any]:
        """包括的リスクレポート生成"""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "emergency_status": self.is_emergency_stopped,
                "risk_metrics": {
                    "total_exposure": float(self.risk_metrics.total_exposure),
                    "daily_pnl": float(self.risk_metrics.daily_pnl),
                    "current_drawdown": float(self.risk_metrics.current_drawdown),
                    "active_positions": self.risk_metrics.active_positions,
                    "daily_trades": self.risk_metrics.daily_trades,
                    "consecutive_losses": self.risk_metrics.consecutive_losses,
                    "volatility": float(self.risk_metrics.volatility),
                    "sharpe_ratio": float(self.risk_metrics.sharpe_ratio),
                },
                "risk_limits": {
                    "max_position_size": float(self.risk_limits.max_position_size),
                    "max_total_exposure": float(self.risk_limits.max_total_exposure),
                    "max_daily_loss": float(self.risk_limits.max_daily_loss),
                    "max_open_positions": self.risk_limits.max_open_positions,
                    "max_daily_trades": self.risk_limits.max_daily_trades,
                },
                "active_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "type": alert.alert_type.value,
                        "level": alert.risk_level.value,
                        "message": alert.message,
                        "symbol": alert.symbol,
                        "timestamp": alert.timestamp.isoformat(),
                    }
                    for alert in self.active_alerts.values()
                ],
                "system_performance": {
                    "cpu_usage": float(self.risk_metrics.cpu_usage),
                    "memory_usage_mb": self.risk_metrics.memory_usage_mb,
                    "api_calls_last_minute": self.risk_metrics.api_calls_last_minute,
                },
            }
        except Exception as e:
            logger.error(f"リスクレポート生成エラー: {e}")
            return {"error": str(e)}

    def record_trade(self, trade: Trade) -> None:
        """取引記録（統計用）"""
        try:
            self.trade_history.append(trade)
            self.risk_metrics.daily_trades += 1

            # 連続損失カウント更新
            if hasattr(trade, "pnl") and trade.pnl < 0:
                self.risk_metrics.consecutive_losses += 1
            else:
                self.risk_metrics.consecutive_losses = 0

            # ポジション開始時刻記録
            if trade.trade_type == TradeType.BUY:
                self.position_opens[trade.symbol] = trade.timestamp
            elif trade.symbol in self.position_opens:
                # ポジションクローズ
                del self.position_opens[trade.symbol]

            logger.debug(
                f"取引記録: {trade.symbol} {trade.trade_type.value} {trade.quantity}"
            )

        except Exception as e:
            logger.error(f"取引記録エラー: {e}")

    def record_api_call(self) -> None:
        """API呼び出し記録"""
        self.api_calls.append(datetime.now())
        self._clean_old_records()

    def _check_order_frequency(self) -> bool:
        """注文頻度チェック"""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        recent_orders = [t for t in self.recent_orders if t > one_minute_ago]
        self.risk_metrics.orders_last_minute = len(recent_orders)

        if len(recent_orders) >= self.risk_limits.max_orders_per_minute:
            return False

        self.recent_orders.append(now)
        return True

    def _calculate_total_exposure(
        self,
        portfolio: Dict[str, Any],
        new_symbol: str,
        new_quantity: int,
        new_price: Decimal,
    ) -> Decimal:
        """総エクスポージャー計算"""
        total = Decimal("0")

        # 既存ポジション
        for _symbol, position in portfolio.get("positions", {}).items():
            quantity = position.get("quantity", 0)
            price = position.get("current_price", Decimal("0"))
            total += abs(quantity) * price

        # 新規ポジション
        total += abs(new_quantity) * new_price

        return total

    def _check_sector_concentration(
        self, symbol: str, quantity: int, price: Decimal, portfolio: Dict[str, Any]
    ) -> bool:
        """セクター集中リスクチェック"""
        try:
            sector = self.sector_mapping.get(symbol, "unknown")
            if sector == "unknown":
                return True  # 分類不明は許可

            sector_exposure = Decimal("0")

            # 既存のセクターエクスポージャー計算
            for sym, position in portfolio.get("positions", {}).items():
                if self.sector_mapping.get(sym) == sector:
                    pos_quantity = position.get("quantity", 0)
                    pos_price = position.get("current_price", Decimal("0"))
                    sector_exposure += abs(pos_quantity) * pos_price

            # 新規ポジション追加
            sector_exposure += abs(quantity) * price

            return sector_exposure <= self.risk_limits.max_sector_exposure

        except Exception as e:
            logger.warning(f"セクター集中リスクチェックエラー: {e}")
            return True  # エラー時は許可（保守的でない選択だが稼働継続を優先）

    def _check_volatility_risk(
        self, symbol: str, quantity: int, price: Decimal
    ) -> bool:
        """ボラティリティリスクチェック"""
        try:
            # 簡易ボラティリティ推定（実際の実装では履歴データを使用）
            position_value = quantity * price
            estimated_volatility = Decimal("0.02")  # 2%と仮定

            position_risk = position_value * estimated_volatility
            max_risk = position_value * self.risk_limits.max_position_volatility

            return position_risk <= max_risk

        except Exception as e:
            logger.warning(f"ボラティリティリスクチェックエラー: {e}")
            return True

    def _create_alert(
        self,
        alert_type: AlertType,
        risk_level: RiskLevel,
        message: str,
        symbol: Optional[str] = None,
        current_value: Optional[Decimal] = None,
        limit_value: Optional[Decimal] = None,
    ) -> RiskAlert:
        """アラート生成"""
        alert_id = f"alert_{int(time.time()*1000)}_{len(self.active_alerts)}"

        alert = RiskAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            risk_level=risk_level,
            message=message,
            symbol=symbol,
            current_value=current_value,
            limit_value=limit_value,
        )

        return alert

    def _process_alert(self, alert: RiskAlert) -> None:
        """アラート処理"""
        try:
            # アクティブアラートに追加
            self.active_alerts[alert.alert_id] = alert

            # 履歴に追加
            self.alert_history.append(alert)

            # ログ出力
            log_level = {
                AlertType.INFO: logger.info,
                AlertType.WARNING: logger.warning,
                AlertType.ERROR: logger.error,
                AlertType.CRITICAL: logger.critical,
            }.get(alert.alert_type, logger.info)

            log_level(f"リスクアラート[{alert.alert_id}]: {alert.message}")

            # コールバック実行
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    logger.error(f"アラートコールバックエラー: {e}")

        except Exception as e:
            logger.error(f"アラート処理エラー: {e}")

    async def _monitoring_loop(self) -> None:
        """監視ループ"""
        while not self._stop_monitoring:
            try:
                # システムメトリクス更新
                self._update_system_metrics()

                # 期限切れアラートのクリーンアップ
                self._cleanup_old_alerts()

                # 1分間隔で監視
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                await asyncio.sleep(60)  # エラー時も継続

    def _update_system_metrics(self) -> None:
        """システムメトリクス更新"""
        try:
            import psutil

            # CPU使用率
            self.risk_metrics.cpu_usage = Decimal(str(psutil.cpu_percent()))

            # メモリ使用量
            memory_info = psutil.virtual_memory()
            self.risk_metrics.memory_usage_mb = int(memory_info.used / 1024 / 1024)

        except ImportError:
            # psutilが利用できない場合は警告を出さない（テスト環境等）
            pass
        except Exception as e:
            logger.debug(f"システムメトリクス更新エラー: {e}")

    def _cleanup_old_alerts(self) -> None:
        """期限切れアラートのクリーンアップ"""
        try:
            now = datetime.now()
            cutoff_time = now - timedelta(hours=24)  # 24時間後に自動削除

            expired_alerts = [
                alert_id
                for alert_id, alert in self.active_alerts.items()
                if alert.timestamp < cutoff_time
            ]

            for alert_id in expired_alerts:
                del self.active_alerts[alert_id]

            if expired_alerts:
                logger.debug(f"期限切れアラート削除: {len(expired_alerts)}件")

        except Exception as e:
            logger.error(f"アラートクリーンアップエラー: {e}")

    def _clean_old_records(self) -> None:
        """古い記録のクリーンアップ"""
        try:
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)

            # API呼び出し記録をクリーンアップ
            while self.api_calls and self.api_calls[0] < one_minute_ago:
                self.api_calls.popleft()

            self.risk_metrics.api_calls_last_minute = len(self.api_calls)

            # 注文記録をクリーンアップ
            while self.recent_orders and self.recent_orders[0] < one_minute_ago:
                self.recent_orders.popleft()

        except Exception as e:
            logger.debug(f"記録クリーンアップエラー: {e}")
