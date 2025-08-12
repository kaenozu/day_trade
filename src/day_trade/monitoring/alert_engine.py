#!/usr/bin/env python3
"""
インテリジェント・アラートエンジン
リアルタイム監視とアラート管理システム

Features:
- 動的閾値アラート
- 複合条件アラート
- アラート抑制機能
- エスカレーション管理
- 通知統合
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..utils.logging_config import get_context_logger
from .metrics.prometheus_metrics import AlertSeverity

logger = get_context_logger(__name__)


class AlertStatus(Enum):
    """アラート状態"""

    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """通知チャネル"""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"


@dataclass
class AlertCondition:
    """アラート条件"""

    metric_name: str
    operator: str  # >, <, >=, <=, ==, !=
    threshold: float
    duration_seconds: int = 60
    labels: Dict[str, str] = None


@dataclass
class AlertRule:
    """アラートルール"""

    id: str
    name: str
    conditions: List[AlertCondition]
    severity: AlertSeverity
    description: str
    enabled: bool = True
    suppression_duration: int = 300  # 5分間抑制
    escalation_rules: List[Dict[str, Any]] = None
    notification_channels: List[NotificationChannel] = None
    labels: Dict[str, str] = None


@dataclass
class Alert:
    """アラート"""

    id: str
    rule_id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    description: str = ""
    labels: Dict[str, str] = None
    values: Dict[str, float] = None
    annotations: Dict[str, str] = None


class IntelligentAlertEngine:
    """インテリジェント・アラートエンジン"""

    def __init__(self):
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.suppressed_alerts = {}
        self.notification_handlers = {}
        self.metric_buffers = defaultdict(lambda: deque(maxlen=100))

        self._running = False
        self._monitoring_task = None

        # デフォルトアラートルール設定
        self._setup_default_alert_rules()

        logger.info("インテリジェント・アラートエンジン初期化完了")

    def _setup_default_alert_rules(self):
        """デフォルトアラートルール設定"""

        # 高CPU使用率アラート
        self.add_alert_rule(
            AlertRule(
                id="high_cpu_usage",
                name="高CPU使用率",
                conditions=[
                    AlertCondition(
                        metric_name="day_trade_cpu_usage_percent",
                        operator=">",
                        threshold=80.0,
                        duration_seconds=60,
                    )
                ],
                severity=AlertSeverity.WARNING,
                description="CPU使用率が80%を超えています",
                notification_channels=[
                    NotificationChannel.SLACK,
                    NotificationChannel.EMAIL,
                ],
            )
        )

        # メモリ不足アラート
        self.add_alert_rule(
            AlertRule(
                id="high_memory_usage",
                name="高メモリ使用率",
                conditions=[
                    AlertCondition(
                        metric_name="day_trade_memory_usage_bytes",
                        operator=">",
                        threshold=0.9,  # 90%
                        duration_seconds=120,
                    )
                ],
                severity=AlertSeverity.CRITICAL,
                description="メモリ使用率が90%を超えています",
                notification_channels=[
                    NotificationChannel.SLACK,
                    NotificationChannel.WEBHOOK,
                ],
            )
        )

        # リアルタイムデータ遅延アラート
        self.add_alert_rule(
            AlertRule(
                id="high_data_latency",
                name="リアルタイムデータ遅延",
                conditions=[
                    AlertCondition(
                        metric_name="day_trade_realtime_data_latency_seconds",
                        operator=">",
                        threshold=1.0,  # 1秒
                        duration_seconds=30,
                    )
                ],
                severity=AlertSeverity.WARNING,
                description="リアルタイムデータの遅延が1秒を超えています",
            )
        )

        # 取引実行失敗アラート
        self.add_alert_rule(
            AlertRule(
                id="trade_execution_failures",
                name="取引実行失敗",
                conditions=[
                    AlertCondition(
                        metric_name="day_trade_trades_total",
                        operator=">",
                        threshold=5.0,  # 5回の失敗
                        duration_seconds=300,
                        labels={"result": "error"},
                    )
                ],
                severity=AlertSeverity.CRITICAL,
                description="取引実行の失敗が多発しています",
            )
        )

        # AI予測精度低下アラート
        self.add_alert_rule(
            AlertRule(
                id="low_prediction_accuracy",
                name="AI予測精度低下",
                conditions=[
                    AlertCondition(
                        metric_name="day_trade_prediction_accuracy",
                        operator="<",
                        threshold=0.7,  # 70%未満
                        duration_seconds=600,  # 10分間
                    )
                ],
                severity=AlertSeverity.WARNING,
                description="AI予測精度が70%を下回っています",
            )
        )

    def add_alert_rule(self, rule: AlertRule):
        """アラートルール追加"""

        self.alert_rules[rule.id] = rule
        logger.info(f"アラートルール追加: {rule.name} (ID: {rule.id})")

    def remove_alert_rule(self, rule_id: str):
        """アラートルール削除"""

        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"アラートルール削除: {rule_id}")

    def add_notification_handler(
        self, channel: NotificationChannel, handler: Callable[[Alert], None]
    ):
        """通知ハンドラー追加"""

        self.notification_handlers[channel] = handler
        logger.info(f"通知ハンドラー追加: {channel.value}")

    async def start_monitoring(self):
        """監視開始"""

        if self._running:
            logger.warning("アラート監視は既に実行中です")
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("アラート監視開始")

    async def stop_monitoring(self):
        """監視停止"""

        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("アラート監視停止")

    async def _monitoring_loop(self):
        """監視ループ"""

        while self._running:
            try:
                await self._evaluate_alert_rules()
                await self._check_alert_resolution()
                await self._cleanup_old_alerts()
                await asyncio.sleep(10)  # 10秒間隔で評価

            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                await asyncio.sleep(5)

    async def _evaluate_alert_rules(self):
        """アラートルール評価"""

        current_time = datetime.now()

        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            try:
                # 条件評価
                conditions_met = await self._evaluate_conditions(
                    rule.conditions, current_time
                )

                if conditions_met:
                    await self._fire_alert(rule, current_time)

            except Exception as e:
                logger.error(f"アラートルール評価エラー ({rule.id}): {e}")

    async def _evaluate_conditions(
        self, conditions: List[AlertCondition], current_time: datetime
    ) -> bool:
        """条件評価"""

        for condition in conditions:
            if not await self._evaluate_single_condition(condition, current_time):
                return False
        return True

    async def _evaluate_single_condition(
        self, condition: AlertCondition, current_time: datetime
    ) -> bool:
        """単一条件評価"""

        # メトリクスバッファから最新の値を取得
        buffer = self.metric_buffers.get(condition.metric_name)
        if not buffer:
            return False

        # 指定期間内の値をチェック
        duration_threshold = current_time - timedelta(
            seconds=condition.duration_seconds
        )
        recent_values = [
            item for item in buffer if item["timestamp"] >= duration_threshold
        ]

        if not recent_values:
            return False

        # 演算子による評価
        latest_value = recent_values[-1]["value"]
        threshold = condition.threshold

        if condition.operator == ">":
            return latest_value > threshold
        elif condition.operator == "<":
            return latest_value < threshold
        elif condition.operator == ">=":
            return latest_value >= threshold
        elif condition.operator == "<=":
            return latest_value <= threshold
        elif condition.operator == "==":
            return abs(latest_value - threshold) < 1e-6
        elif condition.operator == "!=":
            return abs(latest_value - threshold) >= 1e-6

        return False

    async def _fire_alert(self, rule: AlertRule, current_time: datetime):
        """アラート発火"""

        alert_id = f"{rule.id}_{int(current_time.timestamp())}"

        # 抑制チェック
        if self._is_suppressed(rule.id, current_time):
            logger.debug(f"アラート抑制中: {rule.name}")
            return

        # 既にアクティブなアラートがあるかチェック
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.rule_id == rule.id and alert.status == AlertStatus.FIRING:
                existing_alert = alert
                break

        if existing_alert:
            logger.debug(f"アラート既に発火中: {rule.name}")
            return

        # 新しいアラート作成
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            name=rule.name,
            severity=rule.severity,
            status=AlertStatus.FIRING,
            start_time=current_time,
            description=rule.description,
            labels=rule.labels or {},
        )

        # アラート記録
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # 抑制設定
        self.suppressed_alerts[rule.id] = current_time + timedelta(
            seconds=rule.suppression_duration
        )

        # 通知送信
        await self._send_notifications(alert, rule.notification_channels or [])

        logger.warning(f"アラート発火: {alert.name} (重要度: {alert.severity.value})")

    async def _check_alert_resolution(self):
        """アラート解決チェック"""

        current_time = datetime.now()
        resolved_alerts = []

        for alert in self.active_alerts.values():
            if alert.status != AlertStatus.FIRING:
                continue

            rule = self.alert_rules.get(alert.rule_id)
            if not rule:
                continue

            # 条件が解決されているかチェック
            conditions_met = await self._evaluate_conditions(
                rule.conditions, current_time
            )

            if not conditions_met:
                alert.status = AlertStatus.RESOLVED
                alert.end_time = current_time
                resolved_alerts.append(alert)

                logger.info(f"アラート解決: {alert.name}")

        # 解決済みアラートをアクティブリストから削除
        for alert in resolved_alerts:
            if alert.id in self.active_alerts:
                del self.active_alerts[alert.id]

    async def _cleanup_old_alerts(self):
        """古いアラートのクリーンアップ"""

        current_time = datetime.now()
        cleanup_threshold = current_time - timedelta(hours=24)

        # 抑制リストクリーンアップ
        expired_suppressions = []
        for rule_id, expiry_time in self.suppressed_alerts.items():
            if current_time >= expiry_time:
                expired_suppressions.append(rule_id)

        for rule_id in expired_suppressions:
            del self.suppressed_alerts[rule_id]

    def _is_suppressed(self, rule_id: str, current_time: datetime) -> bool:
        """アラート抑制チェック"""

        if rule_id not in self.suppressed_alerts:
            return False

        return current_time < self.suppressed_alerts[rule_id]

    async def _send_notifications(
        self, alert: Alert, channels: List[NotificationChannel]
    ):
        """通知送信"""

        for channel in channels:
            if channel in self.notification_handlers:
                try:
                    handler = self.notification_handlers[channel]
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)

                    logger.info(f"通知送信: {channel.value} - {alert.name}")

                except Exception as e:
                    logger.error(f"通知送信エラー ({channel.value}): {e}")

    def update_metric_value(
        self, metric_name: str, value: float, labels: Dict[str, str] = None
    ):
        """メトリクス値更新"""

        timestamp = datetime.now()
        self.metric_buffers[metric_name].append(
            {"timestamp": timestamp, "value": value, "labels": labels or {}}
        )

    def get_active_alerts(self) -> List[Alert]:
        """アクティブアラート取得"""

        return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """アラート履歴取得"""

        return list(self.alert_history)[-limit:]

    def get_alert_summary(self) -> Dict[str, Any]:
        """アラート概要取得"""

        active_count = len(self.active_alerts)
        severity_counts = defaultdict(int)

        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1

        return {
            "active_alerts": active_count,
            "severity_breakdown": dict(severity_counts),
            "total_rules": len(self.alert_rules),
            "enabled_rules": sum(
                1 for rule in self.alert_rules.values() if rule.enabled
            ),
            "suppressed_rules": len(self.suppressed_alerts),
            "last_evaluation": datetime.now().isoformat(),
        }


# グローバルインスタンス
_alert_engine = IntelligentAlertEngine()


def get_alert_engine() -> IntelligentAlertEngine:
    """アラートエンジン取得"""
    return _alert_engine
