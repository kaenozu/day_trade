"""
アラートシステムと通知機能

パフォーマンス指標の監視とアラート通知を行う統合システム。
しきい値ベース、トレンドベース、異常検知ベースの多層アラート機能。
"""

import asyncio
import json
import smtplib
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional

from .log_aggregation_system import get_log_aggregation_system
from .metrics_collection_system import get_metrics_system


class AlertSeverity(Enum):
    """アラート重要度"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """アラート状態"""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertConditionType(Enum):
    """アラート条件種別"""

    THRESHOLD = "threshold"  # しきい値
    TREND = "trend"  # トレンド
    ANOMALY = "anomaly"  # 異常検知
    PATTERN = "pattern"  # パターンマッチング
    COMPOSITE = "composite"  # 複合条件


class NotificationType(Enum):
    """通知種別"""

    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    CONSOLE = "console"


@dataclass
class AlertCondition:
    """アラート条件"""

    condition_type: AlertConditionType
    metric_name: str
    parameters: Dict[str, Any]
    evaluation_window: timedelta = timedelta(minutes=5)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """アラートルール"""

    id: str
    name: str
    description: str
    condition: AlertCondition
    severity: AlertSeverity
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    cooldown_period: timedelta = timedelta(minutes=15)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Alert:
    """アラート"""

    id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    details: Dict[str, Any]
    triggered_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class AlertEvaluator(ABC):
    """アラート評価ベースクラス"""

    @abstractmethod
    async def evaluate(self, condition: AlertCondition) -> Optional[Dict[str, Any]]:
        """条件を評価してアラート詳細を返す"""
        pass


class ThresholdEvaluator(AlertEvaluator):
    """しきい値評価器"""

    def __init__(self, metrics_system):
        self.metrics_system = metrics_system

    async def evaluate(self, condition: AlertCondition) -> Optional[Dict[str, Any]]:
        """しきい値条件を評価"""
        params = condition.parameters
        threshold = params.get("threshold")
        operator = params.get("operator", "gt")  # gt, lt, eq, gte, lte

        if threshold is None:
            return None

        # 評価期間のデータを取得
        end_time = datetime.utcnow()
        start_time = end_time - condition.evaluation_window

        data = self.metrics_system.query_metrics(
            condition.metric_name, start_time, end_time, condition.labels
        )

        if not data:
            return None

        # 最新の値を取得
        latest_value = data[-1][1]  # (timestamp, value, labels)

        # しきい値と比較
        triggered = False
        if operator == "gt":
            triggered = latest_value > threshold
        elif operator == "lt":
            triggered = latest_value < threshold
        elif operator == "eq":
            triggered = latest_value == threshold
        elif operator == "gte":
            triggered = latest_value >= threshold
        elif operator == "lte":
            triggered = latest_value <= threshold

        if triggered:
            return {
                "current_value": latest_value,
                "threshold": threshold,
                "operator": operator,
                "evaluation_window": str(condition.evaluation_window),
            }

        return None


class TrendEvaluator(AlertEvaluator):
    """トレンド評価器"""

    def __init__(self, metrics_system):
        self.metrics_system = metrics_system

    async def evaluate(self, condition: AlertCondition) -> Optional[Dict[str, Any]]:
        """トレンド条件を評価"""
        params = condition.parameters
        trend_type = params.get("trend_type", "increasing")  # increasing, decreasing
        trend_threshold = params.get("trend_threshold", 0.1)  # パーセント変化
        min_data_points = params.get("min_data_points", 5)

        # 評価期間のデータを取得
        end_time = datetime.utcnow()
        start_time = end_time - condition.evaluation_window

        data = self.metrics_system.query_metrics(
            condition.metric_name, start_time, end_time, condition.labels
        )

        if len(data) < min_data_points:
            return None

        # 線形回帰でトレンドを計算
        values = [row[1] for row in data]
        n = len(values)
        x_values = list(range(n))

        # 線形回帰の傾き計算
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return None

        slope = numerator / denominator

        # トレンド変化率を計算
        if y_mean == 0:
            return None

        trend_rate = (slope * n) / y_mean  # 期間全体での変化率

        # トレンド条件をチェック
        triggered = False
        if trend_type == "increasing" and trend_rate > trend_threshold:
            triggered = True
        elif trend_type == "decreasing" and trend_rate < -trend_threshold:
            triggered = True

        if triggered:
            return {
                "trend_rate": trend_rate,
                "trend_threshold": trend_threshold,
                "trend_type": trend_type,
                "data_points": n,
                "slope": slope,
            }

        return None


class AnomalyEvaluator(AlertEvaluator):
    """異常検知評価器"""

    def __init__(self, metrics_system):
        self.metrics_system = metrics_system

    async def evaluate(self, condition: AlertCondition) -> Optional[Dict[str, Any]]:
        """異常検知条件を評価"""
        params = condition.parameters
        sensitivity = params.get("sensitivity", 2.0)  # 標準偏差の倍数
        baseline_period = params.get("baseline_period", timedelta(hours=24))

        # ベースライン期間のデータを取得
        end_time = datetime.utcnow()
        baseline_start = end_time - baseline_period
        baseline_end = end_time - condition.evaluation_window

        baseline_data = self.metrics_system.query_metrics(
            condition.metric_name, baseline_start, baseline_end, condition.labels
        )

        # 現在の評価期間のデータ
        current_data = self.metrics_system.query_metrics(
            condition.metric_name, baseline_end, end_time, condition.labels
        )

        if len(baseline_data) < 10 or not current_data:
            return None

        # ベースラインの統計計算
        baseline_values = [row[1] for row in baseline_data]
        baseline_mean = sum(baseline_values) / len(baseline_values)
        baseline_variance = sum(
            (x - baseline_mean) ** 2 for x in baseline_values
        ) / len(baseline_values)
        baseline_std = baseline_variance**0.5

        if baseline_std == 0:
            return None

        # 現在の値と比較
        current_value = current_data[-1][1]
        z_score = abs(current_value - baseline_mean) / baseline_std

        if z_score > sensitivity:
            return {
                "current_value": current_value,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "z_score": z_score,
                "sensitivity": sensitivity,
            }

        return None


class PatternEvaluator(AlertEvaluator):
    """パターンマッチング評価器"""

    def __init__(self, log_system):
        self.log_system = log_system

    async def evaluate(self, condition: AlertCondition) -> Optional[Dict[str, Any]]:
        """ログパターン条件を評価"""
        params = condition.parameters
        pattern = params.get("pattern")
        count_threshold = params.get("count_threshold", 1)

        if not pattern:
            return None

        # 評価期間でパターンマッチング
        end_time = datetime.utcnow()
        start_time = end_time - condition.evaluation_window

        # ログ検索（ログ集約システムから）
        matching_logs = await self.log_system.search_logs(
            pattern=pattern,
            start_time=start_time,
            end_time=end_time,
            labels=condition.labels,
        )

        if len(matching_logs) >= count_threshold:
            return {
                "pattern": pattern,
                "match_count": len(matching_logs),
                "count_threshold": count_threshold,
                "matches": matching_logs[:10],  # 最大10件のサンプル
            }

        return None


class NotificationChannel(ABC):
    """通知チャネルベースクラス"""

    @abstractmethod
    async def send_notification(self, alert: Alert) -> bool:
        """通知を送信"""
        pass


class EmailNotificationChannel(NotificationChannel):
    """メール通知チャネル"""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email

    async def send_notification(self, alert: Alert) -> bool:
        """メール通知を送信"""
        try:
            subject = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            body = f"""
アラートが発生しました。

ルール名: {alert.rule_name}
重要度: {alert.severity.value}
メッセージ: {alert.message}
発生時刻: {alert.triggered_at.isoformat()}

詳細:
{json.dumps(alert.details, indent=2, ensure_ascii=False)}
"""

            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = "admin@example.com"  # 設定可能にする

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            return True

        except Exception as e:
            print(f"メール送信エラー: {e}")
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Webhook通知チャネル"""

    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}

    async def send_notification(self, alert: Alert) -> bool:
        """Webhook通知を送信"""
        try:
            import aiohttp

            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "status": alert.status.value,
                "message": alert.message,
                "triggered_at": alert.triggered_at.isoformat(),
                "details": alert.details,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status < 400

        except Exception as e:
            print(f"Webhook送信エラー: {e}")
            return False


class ConsoleNotificationChannel(NotificationChannel):
    """コンソール通知チャネル"""

    async def send_notification(self, alert: Alert) -> bool:
        """コンソール通知を出力"""
        print(f"🚨 [{alert.severity.value.upper()}] {alert.rule_name}")
        print(f"   メッセージ: {alert.message}")
        print(f"   発生時刻: {alert.triggered_at}")
        print(f"   詳細: {alert.details}")
        return True


class AlertManager:
    """アラート管理システム"""

    def __init__(self, db_path: str = "alerts.db"):
        self.db_path = db_path
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.evaluators: Dict[AlertConditionType, AlertEvaluator] = {}
        self._lock = Lock()
        self._running = False
        self._evaluation_task = None

        self._initialize_database()
        self._setup_evaluators()
        self._setup_default_channels()

    def _initialize_database(self):
        """データベースを初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_rules (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    condition_data TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    notification_channels TEXT,
                    cooldown_seconds INTEGER DEFAULT 900,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    triggered_at DATETIME NOT NULL,
                    acknowledged_at DATETIME,
                    resolved_at DATETIME,
                    acknowledged_by TEXT
                )
            """
            )

            conn.commit()

    def _setup_evaluators(self):
        """評価器を設定"""
        metrics_system = get_metrics_system()
        log_system = get_log_aggregation_system()

        self.evaluators = {
            AlertConditionType.THRESHOLD: ThresholdEvaluator(metrics_system),
            AlertConditionType.TREND: TrendEvaluator(metrics_system),
            AlertConditionType.ANOMALY: AnomalyEvaluator(metrics_system),
            AlertConditionType.PATTERN: PatternEvaluator(log_system),
        }

    def _setup_default_channels(self):
        """デフォルト通知チャネルを設定"""
        self.notification_channels["console"] = ConsoleNotificationChannel()
        # 他のチャネルは設定により追加

    def add_notification_channel(self, name: str, channel: NotificationChannel):
        """通知チャネルを追加"""
        self.notification_channels[name] = channel

    def add_alert_rule(self, rule: AlertRule):
        """アラートルールを追加"""
        with self._lock:
            self.rules[rule.id] = rule
            self._save_rule_to_db(rule)

    def remove_alert_rule(self, rule_id: str):
        """アラートルールを削除"""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
                    conn.commit()

    def _save_rule_to_db(self, rule: AlertRule):
        """ルールをデータベースに保存"""
        condition_data = {
            "condition_type": rule.condition.condition_type.value,
            "metric_name": rule.condition.metric_name,
            "parameters": rule.condition.parameters,
            "evaluation_window_seconds": rule.condition.evaluation_window.total_seconds(),
            "labels": rule.condition.labels,
        }

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alert_rules
                (id, name, description, condition_data, severity, enabled,
                 notification_channels, cooldown_seconds, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    rule.id,
                    rule.name,
                    rule.description,
                    json.dumps(condition_data),
                    rule.severity.value,
                    rule.enabled,
                    json.dumps(rule.notification_channels),
                    int(rule.cooldown_period.total_seconds()),
                    rule.created_at.isoformat(),
                    rule.updated_at.isoformat(),
                ),
            )
            conn.commit()

    async def start(self):
        """アラートモニタリング開始"""
        if self._running:
            return

        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())

    def stop(self):
        """アラートモニタリング停止"""
        self._running = False
        if self._evaluation_task:
            self._evaluation_task.cancel()

    async def _evaluation_loop(self):
        """アラート評価ループ"""
        while self._running:
            try:
                await self._evaluate_all_rules()
                await asyncio.sleep(30)  # 30秒間隔で評価
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"アラート評価エラー: {e}")
                await asyncio.sleep(5)

    async def _evaluate_all_rules(self):
        """全ルールを評価"""
        for rule in list(self.rules.values()):
            if not rule.enabled:
                continue

            try:
                await self._evaluate_rule(rule)
            except Exception as e:
                print(f"ルール評価エラー {rule.id}: {e}")

    async def _evaluate_rule(self, rule: AlertRule):
        """個別ルールを評価"""
        evaluator = self.evaluators.get(rule.condition.condition_type)
        if not evaluator:
            return

        # クールダウン期間チェック
        if self._is_in_cooldown(rule):
            return

        # 条件評価
        alert_details = await evaluator.evaluate(rule.condition)

        if alert_details:
            # アラート発生
            alert = Alert(
                id=f"{rule.id}_{int(datetime.utcnow().timestamp())}",
                rule_id=rule.id,
                rule_name=rule.name,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                message=f"{rule.name}: {rule.description}",
                details=alert_details,
                triggered_at=datetime.utcnow(),
            )

            await self._trigger_alert(alert)

    def _is_in_cooldown(self, rule: AlertRule) -> bool:
        """クールダウン期間中かチェック"""
        # 最近のアクティブアラートをチェック
        cutoff_time = datetime.utcnow() - rule.cooldown_period

        for alert in self.active_alerts.values():
            if alert.rule_id == rule.id and alert.triggered_at > cutoff_time:
                return True

        return False

    async def _trigger_alert(self, alert: Alert):
        """アラートを発動"""
        with self._lock:
            self.active_alerts[alert.id] = alert

        # データベースに保存
        self._save_alert_to_db(alert)

        # 通知送信
        rule = self.rules.get(alert.rule_id)
        if rule:
            await self._send_notifications(alert, rule.notification_channels)

    def _save_alert_to_db(self, alert: Alert):
        """アラートをデータベースに保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO alerts
                (id, rule_id, rule_name, severity, status, message, details,
                 triggered_at, acknowledged_at, resolved_at, acknowledged_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.id,
                    alert.rule_id,
                    alert.rule_name,
                    alert.severity.value,
                    alert.status.value,
                    alert.message,
                    json.dumps(alert.details),
                    alert.triggered_at.isoformat(),
                    (
                        alert.acknowledged_at.isoformat()
                        if alert.acknowledged_at
                        else None
                    ),
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    alert.acknowledged_by,
                ),
            )
            conn.commit()

    async def _send_notifications(self, alert: Alert, channels: List[str]):
        """通知を送信"""
        for channel_name in channels:
            channel = self.notification_channels.get(channel_name)
            if channel:
                try:
                    await channel.send_notification(alert)
                except Exception as e:
                    print(f"通知送信エラー {channel_name}: {e}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """アラートを確認済みにする"""
        with self._lock:
            alert = self.active_alerts.get(alert_id)
            if alert:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by
                self._save_alert_to_db(alert)

    def resolve_alert(self, alert_id: str):
        """アラートを解決済みにする"""
        with self._lock:
            alert = self.active_alerts.get(alert_id)
            if alert:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                self._save_alert_to_db(alert)
                del self.active_alerts[alert_id]

    def get_active_alerts(self) -> List[Alert]:
        """アクティブアラートを取得"""
        with self._lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """アラート履歴を取得"""
        alerts = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, rule_id, rule_name, severity, status, message, details,
                       triggered_at, acknowledged_at, resolved_at, acknowledged_by
                FROM alerts
                ORDER BY triggered_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            for row in cursor.fetchall():
                alert = Alert(
                    id=row[0],
                    rule_id=row[1],
                    rule_name=row[2],
                    severity=AlertSeverity(row[3]),
                    status=AlertStatus(row[4]),
                    message=row[5],
                    details=json.loads(row[6]) if row[6] else {},
                    triggered_at=datetime.fromisoformat(row[7]),
                    acknowledged_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    resolved_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    acknowledged_by=row[10],
                )
                alerts.append(alert)

        return alerts


# グローバルインスタンス
_alert_manager = None


def get_alert_manager() -> AlertManager:
    """グローバルアラート管理を取得"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def create_default_alert_rules() -> List[AlertRule]:
    """デフォルトアラートルールを作成"""
    return [
        # CPU使用率アラート
        AlertRule(
            id="cpu_high_usage",
            name="CPU使用率高",
            description="CPU使用率が80%を超えています",
            condition=AlertCondition(
                condition_type=AlertConditionType.THRESHOLD,
                metric_name="cpu_usage_percent",
                parameters={"threshold": 80, "operator": "gt"},
                evaluation_window=timedelta(minutes=5),
            ),
            severity=AlertSeverity.WARNING,
            notification_channels=["console"],
        ),
        # メモリ使用率アラート
        AlertRule(
            id="memory_high_usage",
            name="メモリ使用率高",
            description="メモリ使用率が90%を超えています",
            condition=AlertCondition(
                condition_type=AlertConditionType.THRESHOLD,
                metric_name="memory_usage_percent",
                parameters={"threshold": 90, "operator": "gt"},
                evaluation_window=timedelta(minutes=3),
            ),
            severity=AlertSeverity.ERROR,
            notification_channels=["console"],
        ),
        # レスポンス時間異常検知
        AlertRule(
            id="response_time_anomaly",
            name="レスポンス時間異常",
            description="レスポンス時間に異常が検知されました",
            condition=AlertCondition(
                condition_type=AlertConditionType.ANOMALY,
                metric_name="http_request_duration_seconds",
                parameters={"sensitivity": 2.5},
                evaluation_window=timedelta(minutes=5),
            ),
            severity=AlertSeverity.WARNING,
            notification_channels=["console"],
        ),
        # エラー率増加トレンド
        AlertRule(
            id="error_rate_trend",
            name="エラー率増加傾向",
            description="エラー率が増加傾向にあります",
            condition=AlertCondition(
                condition_type=AlertConditionType.TREND,
                metric_name="http_errors_total",
                parameters={
                    "trend_type": "increasing",
                    "trend_threshold": 0.5,
                    "min_data_points": 10,
                },
                evaluation_window=timedelta(minutes=15),
            ),
            severity=AlertSeverity.WARNING,
            notification_channels=["console"],
        ),
    ]
