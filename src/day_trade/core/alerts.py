"""
アラート機能
価格・指標・パターンベースの通知システム
"""

import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    MimeText = None
    MimeMultipart = None
    smtplib = None

from ..data.stock_fetcher import StockFetcher
from ..analysis.indicators import TechnicalIndicators
from ..analysis.patterns import ChartPatternRecognizer
from ..core.watchlist import WatchlistManager

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """アラートタイプ"""

    PRICE_ABOVE = "price_above"  # 価格上限突破
    PRICE_BELOW = "price_below"  # 価格下限突破
    CHANGE_PERCENT_UP = "change_percent_up"  # 上昇率
    CHANGE_PERCENT_DOWN = "change_percent_down"  # 下落率
    VOLUME_SPIKE = "volume_spike"  # 出来高急増
    RSI_OVERBOUGHT = "rsi_overbought"  # RSI買われすぎ
    RSI_OVERSOLD = "rsi_oversold"  # RSI売られすぎ
    MACD_SIGNAL = "macd_signal"  # MACDシグナル
    BOLLINGER_BREAKOUT = "bollinger_breakout"  # ボリンジャーバンド突破
    PATTERN_DETECTED = "pattern_detected"  # チャートパターン検出
    CUSTOM_CONDITION = "custom_condition"  # カスタム条件


class AlertPriority(Enum):
    """アラート優先度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationMethod(Enum):
    """通知方法"""

    CONSOLE = "console"  # コンソール出力
    EMAIL = "email"  # メール通知
    FILE_LOG = "file_log"  # ファイルログ
    WEBHOOK = "webhook"  # Webhook通知
    CALLBACK = "callback"  # コールバック関数


@dataclass
class AlertCondition:
    """アラート条件"""

    alert_id: str
    symbol: str
    alert_type: AlertType
    condition_value: Union[Decimal, float, str]  # 条件値
    comparison_operator: str = ">"  # >, <, >=, <=, ==
    is_active: bool = True
    priority: AlertPriority = AlertPriority.MEDIUM

    # オプション設定
    cooldown_minutes: int = 60  # クールダウン時間（分）
    expiry_date: Optional[datetime] = None  # 有効期限
    description: str = ""

    # カスタム条件用
    custom_function: Optional[Callable] = None
    custom_parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_parameters is None:
            self.custom_parameters = {}


@dataclass
class AlertTrigger:
    """アラート発火記録"""

    alert_id: str
    symbol: str
    trigger_time: datetime
    alert_type: AlertType
    current_value: Union[Decimal, float, str]
    condition_value: Union[Decimal, float, str]
    message: str
    priority: AlertPriority

    # 市場データ
    current_price: Optional[Decimal] = None
    volume: Optional[int] = None
    change_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "trigger_time": self.trigger_time.isoformat(),
            "alert_type": self.alert_type.value,
            "current_value": str(self.current_value),
            "condition_value": str(self.condition_value),
            "message": self.message,
            "priority": self.priority.value,
            "current_price": str(self.current_price) if self.current_price else None,
            "volume": self.volume,
            "change_percent": self.change_percent,
        }


class NotificationHandler:
    """通知ハンドラー"""

    def __init__(self):
        self.handlers: Dict[NotificationMethod, Callable] = {
            NotificationMethod.CONSOLE: self._send_console_notification,
            NotificationMethod.FILE_LOG: self._send_file_log_notification,
        }

        # メール設定（オプション）
        self.email_config = {
            "smtp_server": "",
            "smtp_port": 587,
            "username": "",
            "password": "",
            "from_email": "",
            "to_emails": [],
        }

    def configure_email(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str],
    ):
        """メール設定"""
        self.email_config.update(
            {
                "smtp_server": smtp_server,
                "smtp_port": smtp_port,
                "username": username,
                "password": password,
                "from_email": from_email,
                "to_emails": to_emails,
            }
        )
        self.handlers[NotificationMethod.EMAIL] = self._send_email_notification

    def add_custom_handler(self, method: NotificationMethod, handler: Callable):
        """カスタム通知ハンドラーの追加"""
        self.handlers[method] = handler

    def send_notification(
        self, trigger: AlertTrigger, methods: List[NotificationMethod]
    ):
        """通知の送信"""
        for method in methods:
            if method in self.handlers:
                try:
                    self.handlers[method](trigger)
                except Exception as e:
                    logger.error(f"通知送信エラー ({method.value}): {e}")

    def _send_console_notification(self, trigger: AlertTrigger):
        """コンソール通知"""
        priority_colors = {
            AlertPriority.LOW: "\033[36m",  # シアン
            AlertPriority.MEDIUM: "\033[33m",  # 黄色
            AlertPriority.HIGH: "\033[31m",  # 赤
            AlertPriority.CRITICAL: "\033[35m",  # マゼンタ
        }

        color = priority_colors.get(trigger.priority, "\033[0m")
        reset = "\033[0m"

        print(f"{color}🚨 ALERT [{trigger.priority.value.upper()}]{reset}")
        print(f"  銘柄: {trigger.symbol}")
        print(f"  タイプ: {trigger.alert_type.value}")
        print(f"  時刻: {trigger.trigger_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  メッセージ: {trigger.message}")
        if trigger.current_price:
            print(f"  現在価格: ¥{trigger.current_price:,}")
        print("-" * 50)

    def _send_file_log_notification(self, trigger: AlertTrigger):
        """ファイルログ通知"""
        log_entry = {
            "timestamp": trigger.trigger_time.isoformat(),
            "alert": trigger.to_dict(),
        }

        filename = f"alerts_{datetime.now().strftime('%Y%m%d')}.log"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _send_email_notification(self, trigger: AlertTrigger):
        """メール通知"""
        if not EMAIL_AVAILABLE:
            logger.warning("メール機能が利用できません")
            return

        if not all(self.email_config.values()):
            logger.warning("メール設定が不完全です")
            return

        subject = f"株価アラート: {trigger.symbol} - {trigger.alert_type.value}"

        body = f"""
株価アラートが発生しました。

銘柄: {trigger.symbol}
アラートタイプ: {trigger.alert_type.value}
優先度: {trigger.priority.value}
発生時刻: {trigger.trigger_time.strftime("%Y-%m-%d %H:%M:%S")}

メッセージ: {trigger.message}

現在値: {trigger.current_value}
条件値: {trigger.condition_value}
"""

        if trigger.current_price:
            body += f"現在価格: ¥{trigger.current_price:,}\n"
        if trigger.volume:
            body += f"出来高: {trigger.volume:,}\n"
        if trigger.change_percent is not None:
            body += f"変化率: {trigger.change_percent:.2f}%\n"

        try:
            msg = MimeMultipart()
            msg["From"] = self.email_config["from_email"]
            msg["To"] = ", ".join(self.email_config["to_emails"])
            msg["Subject"] = subject

            msg.attach(MimeText(body, "plain", "utf-8"))

            with smtplib.SMTP(
                self.email_config["smtp_server"], self.email_config["smtp_port"]
            ) as server:
                server.starttls()
                server.login(
                    self.email_config["username"], self.email_config["password"]
                )
                server.send_message(msg)

            logger.info(f"アラートメールを送信: {trigger.symbol}")

        except Exception as e:
            logger.error(f"メール送信エラー: {e}")


class AlertManager:
    """アラートマネージャー"""

    def __init__(
        self,
        stock_fetcher: Optional[StockFetcher] = None,
        watchlist_manager: Optional[WatchlistManager] = None,
    ):
        """
        Args:
            stock_fetcher: 株価データ取得インスタンス
            watchlist_manager: ウォッチリスト管理インスタンス
        """
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.watchlist_manager = watchlist_manager
        self.technical_indicators = TechnicalIndicators()
        self.pattern_recognizer = ChartPatternRecognizer()
        self.notification_handler = NotificationHandler()

        # アラート管理
        self.alert_conditions: Dict[str, AlertCondition] = {}
        self.alert_history: List[AlertTrigger] = []
        self.last_trigger_times: Dict[str, datetime] = {}  # クールダウン管理

        # 監視スレッド
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 60  # 監視間隔（秒）

        # 通知設定
        self.default_notification_methods = [NotificationMethod.CONSOLE]

    def add_alert(self, condition: AlertCondition) -> bool:
        """アラート条件を追加"""
        try:
            # 条件の検証
            if not self._validate_condition(condition):
                logger.error(f"無効なアラート条件: {condition.alert_id}")
                return False

            self.alert_conditions[condition.alert_id] = condition
            logger.info(
                f"アラート条件を追加: {condition.alert_id} ({condition.symbol})"
            )
            return True

        except Exception as e:
            logger.error(f"アラート追加エラー: {e}")
            return False

    def remove_alert(self, alert_id: str) -> bool:
        """アラート条件を削除"""
        if alert_id in self.alert_conditions:
            del self.alert_conditions[alert_id]
            logger.info(f"アラート条件を削除: {alert_id}")
            return True
        return False

    def get_alerts(self, symbol: Optional[str] = None) -> List[AlertCondition]:
        """アラート条件を取得"""
        if symbol:
            return [
                condition
                for condition in self.alert_conditions.values()
                if condition.symbol == symbol
            ]
        return list(self.alert_conditions.values())

    def start_monitoring(self, interval_seconds: int = 60):
        """アラート監視を開始"""
        if self.monitoring_active:
            logger.warning("監視は既に開始されています")
            return

        self.monitoring_interval = interval_seconds
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"アラート監視を開始 (間隔: {interval_seconds}秒)")

    def stop_monitoring(self):
        """アラート監視を停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("アラート監視を停止")

    def _monitoring_loop(self):
        """監視ループ"""
        while self.monitoring_active:
            try:
                self.check_all_alerts()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                time.sleep(self.monitoring_interval)

    def check_all_alerts(self):
        """全アラートのチェック"""
        active_conditions = [
            condition
            for condition in self.alert_conditions.values()
            if condition.is_active and not self._is_expired(condition)
        ]

        if not active_conditions:
            return

        # 銘柄ごとにグループ化
        symbols_to_check = set(condition.symbol for condition in active_conditions)

        for symbol in symbols_to_check:
            try:
                symbol_conditions = [c for c in active_conditions if c.symbol == symbol]
                self._check_symbol_alerts(symbol, symbol_conditions)
            except Exception as e:
                logger.error(f"銘柄 {symbol} のアラートチェックエラー: {e}")

    def _check_symbol_alerts(self, symbol: str, conditions: List[AlertCondition]):
        """特定銘柄のアラートチェック"""
        # 現在の市場データを取得
        try:
            current_data = self.stock_fetcher.get_current_price(symbol)
            if not current_data:
                logger.warning(f"価格データ取得失敗: {symbol}")
                return

            current_price = Decimal(str(current_data.get("current_price", 0)))
            volume = current_data.get("volume", 0)
            change_percent = current_data.get("change_percent", 0)

            # 履歴データ（テクニカル指標用）
            historical_data = self.stock_fetcher.get_historical_data(
                symbol, period="1mo", interval="1d"
            )

            for condition in conditions:
                if self._should_check_condition(condition):
                    trigger = self._evaluate_condition(
                        condition,
                        current_price,
                        volume,
                        change_percent,
                        historical_data,
                    )

                    if trigger:
                        self._handle_alert_trigger(trigger)

        except Exception as e:
            logger.error(f"銘柄 {symbol} のデータ処理エラー: {e}")

    def _evaluate_condition(
        self,
        condition: AlertCondition,
        current_price: Decimal,
        volume: int,
        change_percent: float,
        historical_data: Optional[Any],
    ) -> Optional[AlertTrigger]:
        """アラート条件の評価"""

        try:
            current_value = None
            message = ""

            if condition.alert_type == AlertType.PRICE_ABOVE:
                current_value = current_price
                if self._compare_values(
                    current_price,
                    condition.condition_value,
                    condition.comparison_operator,
                ):
                    message = f"価格が {condition.condition_value} を上回りました (現在価格: ¥{current_price:,})"

            elif condition.alert_type == AlertType.PRICE_BELOW:
                current_value = current_price
                if self._compare_values(
                    current_price,
                    condition.condition_value,
                    condition.comparison_operator,
                ):
                    message = f"価格が {condition.condition_value} を下回りました (現在価格: ¥{current_price:,})"

            elif condition.alert_type == AlertType.CHANGE_PERCENT_UP:
                current_value = change_percent
                if change_percent >= float(condition.condition_value):
                    message = f"上昇率が {condition.condition_value}% を超えました (現在: {change_percent:.2f}%)"

            elif condition.alert_type == AlertType.CHANGE_PERCENT_DOWN:
                current_value = change_percent
                if change_percent <= float(condition.condition_value):
                    message = f"下落率が {abs(float(condition.condition_value))}% を超えました (現在: {change_percent:.2f}%)"

            elif condition.alert_type == AlertType.VOLUME_SPIKE:
                if historical_data is not None and not historical_data.empty:
                    avg_volume = (
                        historical_data["Volume"].rolling(window=20).mean().iloc[-1]
                    )
                    volume_ratio = volume / avg_volume if avg_volume > 0 else 1
                    current_value = volume_ratio

                    if volume_ratio >= float(condition.condition_value):
                        message = f"出来高急増を検出 (平均の {volume_ratio:.1f}倍: {volume:,})"

            elif condition.alert_type == AlertType.RSI_OVERBOUGHT:
                if historical_data is not None and not historical_data.empty:
                    rsi = self.technical_indicators.calculate_rsi(
                        historical_data["Close"]
                    )
                    if not rsi.empty:
                        current_rsi = rsi.iloc[-1]
                        current_value = current_rsi

                        if current_rsi >= float(condition.condition_value):
                            message = f"RSI買われすぎ水準 (RSI: {current_rsi:.1f})"

            elif condition.alert_type == AlertType.RSI_OVERSOLD:
                if historical_data is not None and not historical_data.empty:
                    rsi = self.technical_indicators.calculate_rsi(
                        historical_data["Close"]
                    )
                    if not rsi.empty:
                        current_rsi = rsi.iloc[-1]
                        current_value = current_rsi

                        if current_rsi <= float(condition.condition_value):
                            message = f"RSI売られすぎ水準 (RSI: {current_rsi:.1f})"

            elif condition.alert_type == AlertType.CUSTOM_CONDITION:
                if condition.custom_function:
                    result = condition.custom_function(
                        condition.symbol,
                        current_price,
                        volume,
                        change_percent,
                        historical_data,
                        condition.custom_parameters,
                    )
                    if result:
                        current_value = "Custom"
                        message = (
                            f"カスタム条件が満たされました: {condition.description}"
                        )

            # トリガーを作成
            if message:
                return AlertTrigger(
                    alert_id=condition.alert_id,
                    symbol=condition.symbol,
                    trigger_time=datetime.now(),
                    alert_type=condition.alert_type,
                    current_value=current_value,
                    condition_value=condition.condition_value,
                    message=message,
                    priority=condition.priority,
                    current_price=current_price,
                    volume=volume,
                    change_percent=change_percent,
                )

        except Exception as e:
            logger.error(f"条件評価エラー ({condition.alert_id}): {e}")

        return None

    def _compare_values(
        self,
        current: Union[Decimal, float],
        target: Union[Decimal, float, str],
        operator: str,
    ) -> bool:
        """値の比較"""
        try:
            current_val = float(current)
            target_val = float(target)

            if operator == ">":
                return current_val > target_val
            elif operator == "<":
                return current_val < target_val
            elif operator == ">=":
                return current_val >= target_val
            elif operator == "<=":
                return current_val <= target_val
            elif operator == "==":
                return abs(current_val - target_val) < 0.001  # 小数点以下の誤差を考慮

        except (ValueError, TypeError):
            pass

        return False

    def _handle_alert_trigger(self, trigger: AlertTrigger):
        """アラート発火の処理"""
        # 履歴に追加
        self.alert_history.append(trigger)

        # クールダウン時間を記録
        self.last_trigger_times[trigger.alert_id] = trigger.trigger_time

        # 通知送信
        self.notification_handler.send_notification(
            trigger, self.default_notification_methods
        )

        logger.info(f"アラート発火: {trigger.alert_id} ({trigger.symbol})")

        # 履歴の制限（最新1000件まで）
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

    def _should_check_condition(self, condition: AlertCondition) -> bool:
        """条件をチェックすべきかの判定"""
        # 無効な条件
        if not condition.is_active:
            return False

        # 期限切れ
        if self._is_expired(condition):
            return False

        # クールダウン中
        if condition.alert_id in self.last_trigger_times:
            last_trigger = self.last_trigger_times[condition.alert_id]
            cooldown_end = last_trigger + timedelta(minutes=condition.cooldown_minutes)
            if datetime.now() < cooldown_end:
                return False

        return True

    def _is_expired(self, condition: AlertCondition) -> bool:
        """条件の有効期限チェック"""
        if condition.expiry_date and datetime.now() > condition.expiry_date:
            return True
        return False

    def _validate_condition(self, condition: AlertCondition) -> bool:
        """条件の検証"""
        if not condition.alert_id or not condition.symbol:
            return False

        if (
            condition.alert_type == AlertType.CUSTOM_CONDITION
            and not condition.custom_function
        ):
            return False

        return True

    def get_alert_history(
        self, symbol: Optional[str] = None, hours: int = 24
    ) -> List[AlertTrigger]:
        """アラート履歴を取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_history = [
            trigger
            for trigger in self.alert_history
            if trigger.trigger_time >= cutoff_time
        ]

        if symbol:
            filtered_history = [
                trigger for trigger in filtered_history if trigger.symbol == symbol
            ]

        return sorted(filtered_history, key=lambda x: x.trigger_time, reverse=True)

    def configure_notifications(self, methods: List[NotificationMethod]):
        """通知方法の設定"""
        self.default_notification_methods = methods

    def export_alerts_config(self, filename: str):
        """アラート設定をエクスポート"""
        try:
            config_data = {}
            for alert_id, condition in self.alert_conditions.items():
                config_data[alert_id] = {
                    "symbol": condition.symbol,
                    "alert_type": condition.alert_type.value,
                    "condition_value": str(condition.condition_value),
                    "comparison_operator": condition.comparison_operator,
                    "is_active": condition.is_active,
                    "priority": condition.priority.value,
                    "cooldown_minutes": condition.cooldown_minutes,
                    "expiry_date": (
                        condition.expiry_date.isoformat()
                        if condition.expiry_date
                        else None
                    ),
                    "description": condition.description,
                }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)

            logger.info(f"アラート設定をエクスポート: {filename}")

        except Exception as e:
            logger.error(f"設定エクスポートエラー: {e}")


# ヘルパー関数
def create_price_alert(
    alert_id: str,
    symbol: str,
    target_price: Decimal,
    above: bool = True,
    priority: AlertPriority = AlertPriority.MEDIUM,
) -> AlertCondition:
    """価格アラートの簡単作成"""
    return AlertCondition(
        alert_id=alert_id,
        symbol=symbol,
        alert_type=AlertType.PRICE_ABOVE if above else AlertType.PRICE_BELOW,
        condition_value=target_price,
        comparison_operator=">" if above else "<",
        priority=priority,
        description=f"価格{'上昇' if above else '下落'}アラート (目標: ¥{target_price:,})",
    )


def create_change_alert(
    alert_id: str,
    symbol: str,
    change_percent: float,
    up: bool = True,
    priority: AlertPriority = AlertPriority.MEDIUM,
) -> AlertCondition:
    """変化率アラートの簡単作成"""
    return AlertCondition(
        alert_id=alert_id,
        symbol=symbol,
        alert_type=AlertType.CHANGE_PERCENT_UP if up else AlertType.CHANGE_PERCENT_DOWN,
        condition_value=change_percent,
        priority=priority,
        description=f"変化率{'上昇' if up else '下落'}アラート ({change_percent:+.1f}%)",
    )


# 使用例
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # アラートマネージャーの作成
    alert_manager = AlertManager()

    # 価格アラートの追加
    price_alert = create_price_alert(
        alert_id="toyota_price_3000",
        symbol="7203",
        target_price=Decimal("3000"),
        above=True,
        priority=AlertPriority.HIGH,
    )

    alert_manager.add_alert(price_alert)

    # 変化率アラートの追加
    change_alert = create_change_alert(
        alert_id="softbank_change_5",
        symbol="9984",
        change_percent=5.0,
        up=True,
        priority=AlertPriority.MEDIUM,
    )

    alert_manager.add_alert(change_alert)

    # カスタムアラートの例
    def custom_volume_price_condition(
        symbol, price, volume, change_pct, historical_data, params
    ):
        """カスタム条件：出来高と価格の組み合わせ"""
        min_price = params.get("min_price", 0)
        min_volume_ratio = params.get("min_volume_ratio", 2.0)

        if historical_data is not None and not historical_data.empty:
            avg_volume = historical_data["Volume"].rolling(window=10).mean().iloc[-1]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1

            return float(price) >= min_price and volume_ratio >= min_volume_ratio

        return False

    custom_alert = AlertCondition(
        alert_id="custom_breakout",
        symbol="7203",
        alert_type=AlertType.CUSTOM_CONDITION,
        condition_value="custom",
        priority=AlertPriority.HIGH,
        custom_function=custom_volume_price_condition,
        custom_parameters={"min_price": 2800, "min_volume_ratio": 1.5},
        description="価格2800円以上かつ出来高1.5倍以上",
    )

    alert_manager.add_alert(custom_alert)

    print("=== アラート設定 ===")
    for alert in alert_manager.get_alerts():
        print(f"ID: {alert.alert_id}")
        print(f"銘柄: {alert.symbol}")
        print(f"タイプ: {alert.alert_type.value}")
        print(f"条件: {alert.condition_value}")
        print(f"説明: {alert.description}")
        print("-" * 30)

    # 監視開始（デモ用に短時間）
    print("\nアラート監視を開始...")
    alert_manager.start_monitoring(interval_seconds=30)

    try:
        # 30秒間監視
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        alert_manager.stop_monitoring()
        print("監視を終了しました")
