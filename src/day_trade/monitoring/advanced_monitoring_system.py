#!/usr/bin/env python3
"""
高度監視・アラートシステム
Phase G: 本番運用最適化フェーズ

リアルタイム監視、ログ分析、アラート通知システム
"""

import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import requests
except ImportError:
    requests = None

try:
    import smtplib
    from email.mime.multipart import MimeMultipart
    from email.mime.text import MimeText

    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False


class AlertLevel(Enum):
    """アラートレベル"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """メトリクスタイプ"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """メトリクス データポイント"""

    timestamp: datetime
    value: float
    labels: Dict[str, str]
    metric_name: str
    metric_type: MetricType


@dataclass
class Alert:
    """アラート"""

    alert_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    source_component: str
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None


@dataclass
class MonitoringRule:
    """監視ルール"""

    rule_id: str
    name: str
    condition: str  # Python expression
    alert_level: AlertLevel
    cooldown_seconds: int = 300  # 5分間のクールダウン
    enabled: bool = True


class AdvancedLogger:
    """高度ロギングシステム"""

    def __init__(self, component_name: str, log_level: str = "INFO"):
        self.component_name = component_name
        self.logger = self._setup_logger(log_level)
        self.structured_logs = deque(maxlen=10000)

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger(f"daytrade.{self.component_name}")
        logger.setLevel(getattr(logging, log_level.upper()))

        # フォーマッター設定
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # ファイルハンドラー
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_dir / f"{self.component_name}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # コンソールハンドラー（開発用）
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def log_structured(self, level: str, message: str, **kwargs):
        """構造化ログ"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "component": self.component_name,
            "level": level,
            "message": message,
            **kwargs,
        }

        self.structured_logs.append(log_entry)
        getattr(self.logger, level.lower())(f"{message} | {json.dumps(kwargs)}")

    def info(self, message: str, **kwargs):
        """情報ログ"""
        self.log_structured("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """警告ログ"""
        self.log_structured("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        """エラーログ"""
        self.log_structured("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """重要ログ"""
        self.log_structured("CRITICAL", message, **kwargs)


class MetricsCollector:
    """メトリクス収集システム"""

    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.collection_lock = Lock()
        self.system_metrics_enabled = True

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Dict[str, str] = None,
    ):
        """メトリクス記録"""
        if labels is None:
            labels = {}

        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels,
            metric_name=name,
            metric_type=metric_type,
        )

        with self.collection_lock:
            self.metrics[name].append(point)

    def collect_system_metrics(self):
        """システムメトリクス収集"""
        if not self.system_metrics_enabled or not psutil:
            return

        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu_usage_percent", cpu_percent)

            # メモリ使用率
            memory = psutil.virtual_memory()
            self.record_metric("system.memory_usage_percent", memory.percent)
            self.record_metric("system.memory_available_mb", memory.available / 1024**2)

            # ディスク使用率 (Windowsでは現在のドライブを使用)
            import platform

            disk_path = "C:\\" if platform.system() == "Windows" else "/"
            disk = psutil.disk_usage(disk_path)
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric("system.disk_usage_percent", disk_percent)

            # ネットワークI/O
            network = psutil.net_io_counters()
            self.record_metric("system.network_bytes_sent", network.bytes_sent)
            self.record_metric("system.network_bytes_recv", network.bytes_recv)

        except Exception as e:
            print(f"システムメトリクス収集エラー: {e}")

    def get_metric_values(self, metric_name: str, duration_minutes: int = 5) -> List[MetricPoint]:
        """メトリクス値取得"""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)

        with self.collection_lock:
            if metric_name not in self.metrics:
                return []

            return [point for point in self.metrics[metric_name] if point.timestamp >= cutoff_time]

    def get_metric_summary(self, metric_name: str, duration_minutes: int = 5) -> Dict[str, float]:
        """メトリクスサマリー"""
        values = self.get_metric_values(metric_name, duration_minutes)

        if not values:
            return {}

        numeric_values = [point.value for point in values]

        return {
            "count": len(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "avg": sum(numeric_values) / len(numeric_values),
            "latest": numeric_values[-1] if numeric_values else 0,
        }


class AlertManager:
    """アラート管理システム"""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_lock = Lock()
        self.notification_channels = []
        self.rule_cooldowns: Dict[str, datetime] = {}

    def add_notification_channel(self, channel: "NotificationChannel"):
        """通知チャンネル追加"""
        self.notification_channels.append(channel)

    def create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        source_component: str,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """アラート作成"""
        import uuid

        if metadata is None:
            metadata = {}

        alert = Alert(
            alert_id=str(uuid.uuid4())[:8],
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source_component=source_component,
            metadata=metadata,
        )

        with self.alert_lock:
            self.alerts.append(alert)

        # 通知送信
        self._send_notifications(alert)

        return alert.alert_id

    def resolve_alert(self, alert_id: str):
        """アラート解決"""
        with self.alert_lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_timestamp = datetime.now()
                    break

    def _send_notifications(self, alert: Alert):
        """通知送信"""
        for channel in self.notification_channels:
            try:
                channel.send_alert(alert)
            except Exception as e:
                print(f"通知送信エラー ({channel.__class__.__name__}): {e}")

    def get_active_alerts(self, level: AlertLevel = None) -> List[Alert]:
        """アクティブアラート取得"""
        with self.alert_lock:
            active_alerts = [alert for alert in self.alerts if not alert.resolved]

            if level:
                active_alerts = [alert for alert in active_alerts if alert.level == level]

            return sorted(active_alerts, key=lambda a: a.timestamp, reverse=True)


class NotificationChannel:
    """通知チャンネル基底クラス"""

    def send_alert(self, alert: Alert):
        """アラート送信（サブクラスで実装）"""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """メール通知チャンネル"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        to_addresses: List[str],
    ):
        if not EMAIL_AVAILABLE:
            raise RuntimeError(
                "Email機能が利用できません: smtplibまたはemailライブラリが見つかりません"
            )

        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.to_addresses = to_addresses

    def send_alert(self, alert: Alert):
        """アラートメール送信"""
        if not EMAIL_AVAILABLE:
            print("Email機能が利用できません")
            return

        try:
            msg = MimeMultipart()
            msg["From"] = self.username
            msg["To"] = ", ".join(self.to_addresses)
            msg["Subject"] = f"[{alert.level.value.upper()}] {alert.title}"

            body = f"""
Day Trade システムアラート

レベル: {alert.level.value.upper()}
時刻: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
コンポーネント: {alert.source_component}

メッセージ:
{alert.message}

詳細情報:
{json.dumps(alert.metadata, indent=2, ensure_ascii=False)}
"""

            msg.attach(MimeText(body, "plain", "utf-8"))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()

        except Exception as e:
            print(f"メール送信エラー: {e}")


class SlackNotificationChannel(NotificationChannel):
    """Slack通知チャンネル"""

    def __init__(self, webhook_url: str):
        if not requests:
            raise RuntimeError("Slack通知機能が利用できません: requestsライブラリが見つかりません")

        self.webhook_url = webhook_url

    def send_alert(self, alert: Alert):
        """Slackアラート送信"""
        if not requests:
            print("Slack通知機能が利用できません: requestsライブラリが見つかりません")
            return

        try:
            color_map = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#ffcc00",
                AlertLevel.ERROR: "#ff6b6b",
                AlertLevel.CRITICAL: "#ff0000",
            }

            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.level, "#cccccc"),
                        "title": f"[{alert.level.value.upper()}] {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "コンポーネント",
                                "value": alert.source_component,
                                "short": True,
                            },
                            {
                                "title": "時刻",
                                "value": alert.timestamp.strftime("%H:%M:%S"),
                                "short": True,
                            },
                        ],
                        "timestamp": int(alert.timestamp.timestamp()),
                    }
                ]
            }

            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()

        except Exception as e:
            print(f"Slack通知エラー: {e}")


class RuleEngine:
    """ルールエンジン"""

    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.rules: List[MonitoringRule] = []
        self.evaluation_context = {}

    def add_rule(self, rule: MonitoringRule):
        """ルール追加"""
        self.rules.append(rule)

    def add_default_rules(self):
        """デフォルトルール追加"""
        default_rules = [
            MonitoringRule(
                rule_id="high_cpu",
                name="高CPU使用率",
                condition="cpu_usage > 80",
                alert_level=AlertLevel.WARNING,
                cooldown_seconds=300,
            ),
            MonitoringRule(
                rule_id="critical_cpu",
                name="重要CPU使用率",
                condition="cpu_usage > 95",
                alert_level=AlertLevel.CRITICAL,
                cooldown_seconds=60,
            ),
            MonitoringRule(
                rule_id="low_memory",
                name="メモリ不足",
                condition="memory_available < 1024",  # 1GB未満
                alert_level=AlertLevel.ERROR,
                cooldown_seconds=300,
            ),
            MonitoringRule(
                rule_id="disk_full",
                name="ディスク容量不足",
                condition="disk_usage > 90",
                alert_level=AlertLevel.CRITICAL,
                cooldown_seconds=600,
            ),
            MonitoringRule(
                rule_id="api_error_rate",
                name="API エラー率高",
                condition="api_error_rate > 0.05",  # 5%以上
                alert_level=AlertLevel.WARNING,
                cooldown_seconds=300,
            ),
        ]

        for rule in default_rules:
            self.add_rule(rule)

    def evaluate_rules(self):
        """ルール評価"""
        current_time = datetime.now()

        # 評価コンテキスト更新
        self._update_evaluation_context()

        for rule in self.rules:
            if not rule.enabled:
                continue

            # クールダウンチェック
            if rule.rule_id in self.alert_manager.rule_cooldowns:
                cooldown_end = self.alert_manager.rule_cooldowns[rule.rule_id]
                if current_time < cooldown_end:
                    continue

            try:
                # ルール条件評価
                if eval(rule.condition, {"__builtins__": {}}, self.evaluation_context):
                    # アラート作成
                    alert_id = self.alert_manager.create_alert(
                        level=rule.alert_level,
                        title=rule.name,
                        message=f"監視ルール '{rule.name}' が発火しました: {rule.condition}",
                        source_component="RuleEngine",
                        metadata={
                            "rule_id": rule.rule_id,
                            "condition": rule.condition,
                            "context": self.evaluation_context.copy(),
                        },
                    )

                    # クールダウン設定
                    cooldown_end = current_time + timedelta(seconds=rule.cooldown_seconds)
                    self.alert_manager.rule_cooldowns[rule.rule_id] = cooldown_end

            except Exception as e:
                print(f"ルール評価エラー ({rule.rule_id}): {e}")

    def _update_evaluation_context(self):
        """評価コンテキスト更新"""
        # システムメトリクス
        cpu_summary = self.metrics_collector.get_metric_summary("system.cpu_usage_percent")
        memory_summary = self.metrics_collector.get_metric_summary("system.memory_available_mb")
        disk_summary = self.metrics_collector.get_metric_summary("system.disk_usage_percent")

        self.evaluation_context.update(
            {
                "cpu_usage": cpu_summary.get("latest", 0),
                "memory_available": memory_summary.get("latest", 0),
                "disk_usage": disk_summary.get("latest", 0),
                "time": datetime.now(),
                # 必要に応じてアプリケーション固有メトリクスを追加
                "api_error_rate": 0.01,  # デモ用
            }
        )


class AdvancedMonitoringSystem:
    """高度監視システム"""

    def __init__(self):
        self.logger = AdvancedLogger("MonitoringSystem")
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.rule_engine = RuleEngine(self.metrics_collector, self.alert_manager)

        self.running = False
        self.monitoring_thread = None

        print("=" * 80)
        print("[MONITOR] 高度監視・アラートシステム")
        print("Phase G: 本番運用最適化フェーズ")
        print("=" * 80)

    def setup_notifications(self):
        """通知設定"""
        # 環境変数から設定を読み込み
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")

        if smtp_server and smtp_username and smtp_password and EMAIL_AVAILABLE:
            try:
                email_channel = EmailNotificationChannel(
                    smtp_server=smtp_server,
                    smtp_port=587,
                    username=smtp_username,
                    password=smtp_password,
                    to_addresses=os.getenv("ALERT_EMAIL_ADDRESSES", "").split(","),
                )
                self.alert_manager.add_notification_channel(email_channel)
                self.logger.info("メール通知チャンネル設定完了")
            except Exception as e:
                self.logger.warning("メール通知チャンネル設定失敗", error=str(e))

        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if slack_webhook and requests:
            try:
                slack_channel = SlackNotificationChannel(slack_webhook)
                self.alert_manager.add_notification_channel(slack_channel)
                self.logger.info("Slack通知チャンネル設定完了")
            except Exception as e:
                self.logger.warning("Slack通知チャンネル設定失敗", error=str(e))

    def start_monitoring(self):
        """監視開始"""
        if self.running:
            return

        self.running = True
        self.rule_engine.add_default_rules()

        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("監視システム開始")

    def stop_monitoring(self):
        """監視停止"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        self.logger.info("監視システム停止")

    def _monitoring_loop(self):
        """監視ループ"""
        while self.running:
            try:
                # システムメトリクス収集
                self.metrics_collector.collect_system_metrics()

                # ルール評価
                self.rule_engine.evaluate_rules()

                # 30秒間隔で実行
                time.sleep(30)

            except Exception as e:
                self.logger.error("監視ループエラー", error=str(e))
                time.sleep(60)  # エラー時は1分待機

    def record_application_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """アプリケーションメトリクス記録"""
        self.metrics_collector.record_metric(name, value, labels=labels)

    def create_manual_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        component: str = "Application",
    ):
        """手動アラート作成"""
        return self.alert_manager.create_alert(
            level=level, title=title, message=message, source_component=component
        )

    def get_system_dashboard(self) -> Dict[str, Any]:
        """システムダッシュボード情報"""
        return {
            "system_metrics": {
                "cpu": self.metrics_collector.get_metric_summary("system.cpu_usage_percent"),
                "memory": self.metrics_collector.get_metric_summary("system.memory_available_mb"),
                "disk": self.metrics_collector.get_metric_summary("system.disk_usage_percent"),
            },
            "active_alerts": {
                "critical": len(self.alert_manager.get_active_alerts(AlertLevel.CRITICAL)),
                "error": len(self.alert_manager.get_active_alerts(AlertLevel.ERROR)),
                "warning": len(self.alert_manager.get_active_alerts(AlertLevel.WARNING)),
                "total": len(self.alert_manager.get_active_alerts()),
            },
            "monitoring_status": {
                "running": self.running,
                "rules_count": len(self.rule_engine.rules),
                "notification_channels": len(self.alert_manager.notification_channels),
            },
        }


def main():
    """メイン監視システム実行"""
    system = AdvancedMonitoringSystem()

    try:
        # 通知設定
        system.setup_notifications()

        # 監視開始
        system.start_monitoring()

        # テスト用アラート
        system.create_manual_alert(
            AlertLevel.INFO,
            "監視システム開始",
            "高度監視システムが正常に開始されました",
        )

        print("\n[RUNNING] 監視システム実行中...")
        print("Ctrl+C で停止")

        # メインループ（実際の本番環境では常時実行）
        while True:
            dashboard = system.get_system_dashboard()
            print(
                f"\n[DASHBOARD] CPU: {dashboard['system_metrics']['cpu'].get('latest', 0):.1f}% | "
                f"Memory: {dashboard['system_metrics']['memory'].get('latest', 0):.0f}MB | "
                f"Alerts: {dashboard['active_alerts']['total']}"
            )
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n[STOP] 監視システム停止中...")
        system.stop_monitoring()
        print("[COMPLETE] 監視システム停止完了")


if __name__ == "__main__":
    main()
