#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine - 自動アラートシステム
インテリジェント通知・アラート管理システム

AI予測・リスク・市場異常の統合アラートシステム
"""

import asyncio
import time
import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger
from .live_prediction_engine import LivePrediction
from .websocket_stream import MarketTick

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

class AlertLevel(Enum):
    """アラートレベル"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """アラートタイプ"""
    TRADING_SIGNAL = "trading_signal"
    RISK_ALERT = "risk_alert"
    SYSTEM_ERROR = "system_error"
    MARKET_ANOMALY = "market_anomaly"
    PERFORMANCE_ALERT = "performance_alert"
    DATA_QUALITY = "data_quality"

@dataclass
class AlertConfig:
    """アラート設定"""
    # 通知設定
    enable_email: bool = True
    enable_console: bool = True
    enable_webhook: bool = False
    enable_sms: bool = False

    # Email設定
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)

    # Webhook設定
    webhook_url: Optional[str] = None
    webhook_timeout: float = 5.0

    # アラート条件
    trading_signal_threshold: float = 0.7  # 取引シグナル信頼度閾値
    risk_alert_threshold: float = 0.8  # リスクアラート閾値
    market_anomaly_threshold: float = 2.0  # 市場異常検知閾値（標準偏差）

    # レート制限
    max_alerts_per_hour: int = 100
    duplicate_suppression_minutes: int = 15

    # 優先度設定
    critical_symbols: List[str] = field(default_factory=list)
    vip_alert_threshold: float = 0.9

@dataclass
class Alert:
    """アラート"""
    id: str
    timestamp: datetime
    level: AlertLevel
    alert_type: AlertType

    # 内容
    title: str
    message: str
    symbol: Optional[str] = None

    # メタデータ
    confidence: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)

    # 処理状況
    sent: bool = False
    acknowledged: bool = False
    resolved: bool = False

    # 通知先
    channels: List[str] = field(default_factory=lambda: ["console"])

    def to_dict(self) -> Dict:
        """辞書形式変換"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.value,
            'type': self.alert_type.value,
            'title': self.title,
            'message': self.message,
            'symbol': self.symbol,
            'confidence': self.confidence,
            'data': self.data,
            'sent': self.sent,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'channels': self.channels
        }

class EmailNotifier:
    """Email通知システム"""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.enabled = (
            config.enable_email and
            config.email_username and
            config.email_password and
            config.email_recipients
        )

        if not self.enabled:
            logger.warning("Email notifications disabled - missing configuration")

    async def send_alert(self, alert: Alert) -> bool:
        """アラート送信"""

        if not self.enabled:
            return False

        try:
            # HTML メッセージ作成
            html_message = self._create_html_message(alert)

            # Email送信
            success = await self._send_email(
                subject=f"[{alert.level.value.upper()}] {alert.title}",
                html_content=html_message,
                recipients=self.config.email_recipients
            )

            if success:
                logger.info(f"Email alert sent: {alert.id}")
            else:
                logger.error(f"Failed to send email alert: {alert.id}")

            return success

        except Exception as e:
            logger.error(f"Email notification error: {e}")
            return False

    def _create_html_message(self, alert: Alert) -> str:
        """HTMLメッセージ作成"""

        # レベル別色設定
        level_colors = {
            AlertLevel.INFO: "#2196F3",
            AlertLevel.WARNING: "#FF9800",
            AlertLevel.CRITICAL: "#F44336",
            AlertLevel.EMERGENCY: "#9C27B0"
        }

        color = level_colors.get(alert.level, "#666666")

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert-header {{
                    background-color: {color};
                    color: white;
                    padding: 15px;
                    border-radius: 5px 5px 0 0;
                }}
                .alert-body {{
                    border: 2px solid {color};
                    padding: 15px;
                    border-radius: 0 0 5px 5px;
                }}
                .alert-meta {{
                    background-color: #f5f5f5;
                    padding: 10px;
                    margin-top: 15px;
                    border-radius: 3px;
                }}
                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }}
                .data-table th, .data-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .data-table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>🚨 {alert.title}</h2>
                <p><strong>{alert.level.value.upper()}</strong> | {alert.alert_type.value}</p>
            </div>

            <div class="alert-body">
                <p><strong>Message:</strong> {alert.message}</p>

                {f'<p><strong>Symbol:</strong> {alert.symbol}</p>' if alert.symbol else ''}

                <div class="alert-meta">
                    <p><strong>Timestamp:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Confidence:</strong> {alert.confidence:.2%}</p>
                    <p><strong>Alert ID:</strong> {alert.id}</p>
                </div>

                {self._create_data_table(alert.data) if alert.data else ''}
            </div>

            <p style="margin-top: 20px; font-size: 12px; color: #666;">
                This alert was generated by Next-Gen AI Trading Engine
            </p>
        </body>
        </html>
        """

        return html

    def _create_data_table(self, data: Dict[str, Any]) -> str:
        """データテーブル作成"""

        if not data:
            return ""

        rows = ""
        for key, value in data.items():
            # 値の型に応じてフォーマット
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, dict):
                formatted_value = json.dumps(value, indent=2)
            else:
                formatted_value = str(value)

            rows += f"<tr><td><strong>{key}</strong></td><td>{formatted_value}</td></tr>"

        return f"""
        <table class="data-table">
            <thead>
                <tr><th>Field</th><th>Value</th></tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """

    async def _send_email(self, subject: str, html_content: str, recipients: List[str]) -> bool:
        """Email送信実行"""

        try:
            # MIMEメッセージ作成
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(recipients)

            # HTML部分添付
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # SMTP送信
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None,
                self._smtp_send,
                msg,
                recipients
            )

            return success

        except Exception as e:
            logger.error(f"Email sending error: {e}")
            return False

    def _smtp_send(self, msg: MIMEMultipart, recipients: List[str]) -> bool:
        """SMTP送信（同期）"""

        try:
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.email_username, self.config.email_password)

                text = msg.as_string()
                server.sendmail(self.config.email_username, recipients, text)

            return True

        except Exception as e:
            logger.error(f"SMTP send error: {e}")
            return False

class WebhookNotifier:
    """Webhook通知システム"""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.enabled = config.enable_webhook and config.webhook_url

        if not self.enabled:
            logger.info("Webhook notifications disabled")

    async def send_alert(self, alert: Alert) -> bool:
        """Webhook アラート送信"""

        if not self.enabled:
            return False

        try:
            import aiohttp

            # Webhook ペイロード作成
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'source': 'NextGenAI_TradingEngine'
            }

            # HTTP POST送信
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.webhook_timeout)) as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:

                    if response.status == 200:
                        logger.info(f"Webhook alert sent successfully: {alert.id}")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}: {alert.id}")
                        return False

        except Exception as e:
            logger.error(f"Webhook notification error: {e}")
            return False

class ConsoleNotifier:
    """コンソール通知システム"""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.enabled = config.enable_console

    async def send_alert(self, alert: Alert) -> bool:
        """コンソール アラート表示"""

        if not self.enabled:
            return False

        try:
            # レベル別アイコン
            level_icons = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.CRITICAL: "🚨",
                AlertLevel.EMERGENCY: "🔴"
            }

            icon = level_icons.get(alert.level, "📢")

            # コンソール表示
            print(f"\n{icon} ALERT [{alert.level.value.upper()}] {icon}")
            print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Type: {alert.alert_type.value}")
            if alert.symbol:
                print(f"Symbol: {alert.symbol}")
            print(f"Title: {alert.title}")
            print(f"Message: {alert.message}")
            if alert.confidence < 1.0:
                print(f"Confidence: {alert.confidence:.2%}")

            # 追加データ表示
            if alert.data:
                print("Data:")
                for key, value in alert.data.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")

            print(f"Alert ID: {alert.id}")
            print("-" * 60)

            return True

        except Exception as e:
            logger.error(f"Console notification error: {e}")
            return False

class AlertManager:
    """アラート管理システム"""

    def __init__(self, config: AlertConfig):
        self.config = config

        # 通知システム
        self.email_notifier = EmailNotifier(config)
        self.webhook_notifier = WebhookNotifier(config)
        self.console_notifier = ConsoleNotifier(config)

        # アラート管理
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # レート制限
        self.alert_timestamps: List[datetime] = []
        self.duplicate_cache: Dict[str, datetime] = {}

        # 統計
        self.stats = {
            'total_alerts': 0,
            'sent_alerts': 0,
            'failed_alerts': 0,
            'suppressed_alerts': 0
        }

        logger.info("Alert Manager initialized")

    async def send_alert(self, alert: Alert) -> bool:
        """アラート送信"""

        # レート制限チェック
        if not self._check_rate_limit():
            logger.warning(f"Alert rate limit exceeded, skipping: {alert.id}")
            self.stats['suppressed_alerts'] += 1
            return False

        # 重複チェック
        if self._is_duplicate_alert(alert):
            logger.debug(f"Duplicate alert suppressed: {alert.id}")
            self.stats['suppressed_alerts'] += 1
            return False

        try:
            self.stats['total_alerts'] += 1

            # 通知チャンネル決定
            channels = self._determine_channels(alert)
            alert.channels = channels

            # 各チャンネルに送信
            success_count = 0
            total_channels = len(channels)

            if "console" in channels:
                if await self.console_notifier.send_alert(alert):
                    success_count += 1

            if "email" in channels:
                if await self.email_notifier.send_alert(alert):
                    success_count += 1
                else:
                    total_channels -= 1  # 無効なチャンネルは除外

            if "webhook" in channels:
                if await self.webhook_notifier.send_alert(alert):
                    success_count += 1
                else:
                    total_channels -= 1  # 無効なチャンネルは除外

            # 成功判定
            success = success_count > 0 and success_count >= (total_channels * 0.5)

            if success:
                alert.sent = True
                self.stats['sent_alerts'] += 1
                logger.info(f"Alert sent successfully: {alert.id} ({success_count}/{total_channels} channels)")
            else:
                self.stats['failed_alerts'] += 1
                logger.error(f"Alert sending failed: {alert.id}")

            # アラート記録
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)

            # 履歴サイズ制限
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-500:]

            return success

        except Exception as e:
            logger.error(f"Alert sending error: {e}")
            self.stats['failed_alerts'] += 1
            return False

    def _check_rate_limit(self) -> bool:
        """レート制限チェック"""

        now = datetime.now()
        cutoff_time = now - timedelta(hours=1)

        # 1時間以内のアラートをカウント
        self.alert_timestamps = [ts for ts in self.alert_timestamps if ts > cutoff_time]

        if len(self.alert_timestamps) >= self.config.max_alerts_per_hour:
            return False

        self.alert_timestamps.append(now)
        return True

    def _is_duplicate_alert(self, alert: Alert) -> bool:
        """重複アラートチェック"""

        # 重複キー作成
        duplicate_key = f"{alert.alert_type.value}_{alert.symbol}_{alert.title}"

        now = datetime.now()
        cutoff_time = now - timedelta(minutes=self.config.duplicate_suppression_minutes)

        # 既存の重複記録をクリーンアップ
        self.duplicate_cache = {
            key: timestamp for key, timestamp in self.duplicate_cache.items()
            if timestamp > cutoff_time
        }

        # 重複チェック
        if duplicate_key in self.duplicate_cache:
            return True

        # 新しいアラートを記録
        self.duplicate_cache[duplicate_key] = now
        return False

    def _determine_channels(self, alert: Alert) -> List[str]:
        """通知チャンネル決定"""

        channels = ["console"]  # デフォルトでコンソール

        # レベル・タイプ別チャンネル決定
        if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            # 重要アラートは全チャンネル
            if self.email_notifier.enabled:
                channels.append("email")
            if self.webhook_notifier.enabled:
                channels.append("webhook")

        elif alert.level == AlertLevel.WARNING:
            # 警告レベルはEメール
            if self.email_notifier.enabled:
                channels.append("email")

        # VIPシンボルは優先通知
        if alert.symbol in self.config.critical_symbols:
            if self.email_notifier.enabled and "email" not in channels:
                channels.append("email")

        # 高信頼度アラートは優先通知
        if alert.confidence >= self.config.vip_alert_threshold:
            if self.webhook_notifier.enabled and "webhook" not in channels:
                channels.append("webhook")

        return channels

    def acknowledge_alert(self, alert_id: str) -> bool:
        """アラート確認"""

        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True

        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """アラート解決"""

        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.acknowledged = True

            # アクティブリストから削除
            del self.active_alerts[alert_id]

            logger.info(f"Alert resolved: {alert_id}")
            return True

        return False

    def get_active_alerts(self) -> List[Alert]:
        """アクティブアラート取得"""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """アラート履歴取得"""

        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]

    def get_statistics(self) -> Dict:
        """統計情報取得"""

        active_count = len(self.active_alerts)
        unacknowledged_count = len([
            alert for alert in self.active_alerts.values()
            if not alert.acknowledged
        ])

        return {
            **self.stats,
            'active_alerts': active_count,
            'unacknowledged_alerts': unacknowledged_count,
            'alert_history_size': len(self.alert_history)
        }

class TradingAlertGenerator:
    """取引アラート生成器"""

    def __init__(self, alert_manager: AlertManager, config: AlertConfig):
        self.alert_manager = alert_manager
        self.config = config

    async def generate_trading_signal_alert(self, prediction: LivePrediction) -> Optional[Alert]:
        """取引シグナルアラート生成"""

        if prediction.action_confidence < self.config.trading_signal_threshold:
            return None

        # アラートレベル決定
        if prediction.action_confidence >= 0.9:
            level = AlertLevel.CRITICAL
        elif prediction.action_confidence >= 0.8:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO

        # アラート作成
        alert = Alert(
            id=f"trade_{prediction.symbol}_{int(prediction.timestamp.timestamp())}",
            timestamp=prediction.timestamp,
            level=level,
            alert_type=AlertType.TRADING_SIGNAL,
            title=f"Trading Signal: {prediction.final_action} {prediction.symbol}",
            message=f"AI recommends {prediction.final_action} for {prediction.symbol} "
                   f"with {prediction.action_confidence:.1%} confidence. "
                   f"Target: ${prediction.predicted_price:.2f} "
                   f"(Position size: {prediction.position_size_recommendation:.1%})",
            symbol=prediction.symbol,
            confidence=prediction.action_confidence,
            data={
                'action': prediction.final_action,
                'predicted_price': prediction.predicted_price,
                'predicted_return': prediction.predicted_return,
                'position_size': prediction.position_size_recommendation,
                'ml_confidence': prediction.ml_prediction.get('confidence', 0) if prediction.ml_prediction else 0,
                'rl_confidence': prediction.rl_decision.get('confidence', 0) if prediction.rl_decision else 0,
                'sentiment_score': prediction.sentiment_analysis.get('sentiment_score', 0) if prediction.sentiment_analysis else 0
            }
        )

        return alert

    async def generate_risk_alert(self, symbol: str, risk_level: float, risk_type: str, details: Dict) -> Alert:
        """リスクアラート生成"""

        # リスクレベル別アラートレベル
        if risk_level >= 0.9:
            level = AlertLevel.EMERGENCY
        elif risk_level >= 0.8:
            level = AlertLevel.CRITICAL
        elif risk_level >= 0.6:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO

        alert = Alert(
            id=f"risk_{symbol}_{risk_type}_{int(time.time())}",
            timestamp=datetime.now(),
            level=level,
            alert_type=AlertType.RISK_ALERT,
            title=f"Risk Alert: {risk_type} - {symbol}",
            message=f"High risk detected for {symbol}: {risk_type} "
                   f"(Risk Level: {risk_level:.1%})",
            symbol=symbol,
            confidence=risk_level,
            data=details
        )

        return alert

    async def generate_market_anomaly_alert(self, symbol: str, anomaly_score: float, details: Dict) -> Alert:
        """市場異常アラート生成"""

        level = AlertLevel.WARNING if anomaly_score >= self.config.market_anomaly_threshold else AlertLevel.INFO

        alert = Alert(
            id=f"anomaly_{symbol}_{int(time.time())}",
            timestamp=datetime.now(),
            level=level,
            alert_type=AlertType.MARKET_ANOMALY,
            title=f"Market Anomaly: {symbol}",
            message=f"Unusual market behavior detected for {symbol} "
                   f"(Anomaly Score: {anomaly_score:.2f}σ)",
            symbol=symbol,
            confidence=min(anomaly_score / 3.0, 1.0),  # 3σを最大として正規化
            data=details
        )

        return alert

    async def generate_system_alert(self, component: str, error_message: str, severity: str = "warning") -> Alert:
        """システムアラート生成"""

        level_map = {
            "info": AlertLevel.INFO,
            "warning": AlertLevel.WARNING,
            "critical": AlertLevel.CRITICAL,
            "emergency": AlertLevel.EMERGENCY
        }

        level = level_map.get(severity.lower(), AlertLevel.WARNING)

        alert = Alert(
            id=f"system_{component}_{int(time.time())}",
            timestamp=datetime.now(),
            level=level,
            alert_type=AlertType.SYSTEM_ERROR,
            title=f"System Alert: {component}",
            message=f"System issue detected in {component}: {error_message}",
            confidence=1.0,
            data={
                'component': component,
                'error_message': error_message,
                'severity': severity
            }
        )

        return alert

# 便利関数
def create_alert_system(email_config: Optional[Dict] = None) -> Tuple[AlertManager, TradingAlertGenerator]:
    """アラートシステム作成"""

    # 設定作成
    config = AlertConfig()

    if email_config:
        config.email_username = email_config.get('username')
        config.email_password = email_config.get('password')
        config.email_recipients = email_config.get('recipients', [])
        config.enable_email = bool(config.email_username and config.email_password)

    # アラートマネージャー作成
    alert_manager = AlertManager(config)

    # 取引アラート生成器作成
    trading_alert_generator = TradingAlertGenerator(alert_manager, config)

    return alert_manager, trading_alert_generator

if __name__ == "__main__":
    # アラートシステムテスト
    async def test_alert_system():
        print("=== Alert System Test ===")

        try:
            # アラートシステム作成
            alert_manager, trading_alert_generator = create_alert_system()

            # テストアラート作成
            test_alert = Alert(
                id="test_001",
                timestamp=datetime.now(),
                level=AlertLevel.WARNING,
                alert_type=AlertType.TRADING_SIGNAL,
                title="Test Alert",
                message="This is a test alert for the Next-Gen AI Trading Engine",
                symbol="AAPL",
                confidence=0.85,
                data={
                    'test_value': 123.45,
                    'test_string': "Hello World",
                    'test_dict': {'nested': 'value'}
                }
            )

            # アラート送信
            success = await alert_manager.send_alert(test_alert)
            print(f"Alert sent: {success}")

            # 統計取得
            stats = alert_manager.get_statistics()
            print(f"Alert statistics: {stats}")

            # アクティブアラート確認
            active_alerts = alert_manager.get_active_alerts()
            print(f"Active alerts: {len(active_alerts)}")

            # アラート確認・解決テスト
            if active_alerts:
                alert_id = active_alerts[0].id
                alert_manager.acknowledge_alert(alert_id)
                alert_manager.resolve_alert(alert_id)
                print(f"Alert {alert_id} acknowledged and resolved")

            print("Alert system test completed successfully")

        except Exception as e:
            print(f"Test error: {e}")
            import traceback
            traceback.print_exc()

    # テスト実行
    asyncio.run(test_alert_system())
