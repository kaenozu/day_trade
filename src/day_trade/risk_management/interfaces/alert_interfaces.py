#!/usr/bin/env python3
"""
Alert Management Interfaces
アラート管理インターフェース

Slack、Email、SMS等の各種通知チャネルに対応した抽象インターフェース
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AlertSeverity(Enum):
    """アラート重要度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """アラートステータス"""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


class NotificationChannel(Enum):
    """通知チャネル"""

    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    PUSH_NOTIFICATION = "push"


@dataclass
class AlertRule:
    """アラートルール"""

    rule_id: str
    name: str
    description: str
    condition: str  # 判定条件式
    severity: AlertSeverity
    channels: List[NotificationChannel]
    throttle_minutes: int = 5
    auto_resolve: bool = False
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        """トリガー判定"""
        try:
            # 安全な条件評価（実装では適切なサンドボックス化が必要）
            return eval(self.condition, {"__builtins__": {}}, metrics)
        except Exception:
            return False


@dataclass
class Alert:
    """アラート"""

    alert_id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    source_system: str = "risk_management"
    affected_resources: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationMessage:
    """通知メッセージ"""

    title: str
    body: str
    channel: NotificationChannel
    recipients: List[str]
    priority: AlertSeverity
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class INotificationChannel(ABC):
    """通知チャネルインターフェース"""

    @abstractmethod
    async def send_notification(self, message: NotificationMessage) -> bool:
        """通知送信"""
        pass

    @abstractmethod
    def get_channel_type(self) -> NotificationChannel:
        """チャネルタイプ取得"""
        pass

    @abstractmethod
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """設定検証"""
        pass

    @abstractmethod
    async def get_delivery_status(self, message_id: str) -> Dict[str, Any]:
        """配信状態取得"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """ヘルスチェック"""
        pass


class IAlertRule(ABC):
    """アラートルールインターフェース"""

    @abstractmethod
    def evaluate(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        """ルール評価"""
        pass

    @abstractmethod
    def get_rule_definition(self) -> AlertRule:
        """ルール定義取得"""
        pass

    @abstractmethod
    async def update_rule(self, rule: AlertRule) -> bool:
        """ルール更新"""
        pass

    @abstractmethod
    def is_throttled(self, current_time: datetime) -> bool:
        """スロットリング判定"""
        pass


class IAlertProcessor(ABC):
    """アラート処理器インターフェース"""

    @abstractmethod
    async def process_alert(self, alert: Alert) -> bool:
        """アラート処理"""
        pass

    @abstractmethod
    async def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str, comment: Optional[str] = None
    ) -> bool:
        """アラート承認"""
        pass

    @abstractmethod
    async def resolve_alert(
        self, alert_id: str, resolved_by: str, comment: Optional[str] = None
    ) -> bool:
        """アラート解決"""
        pass

    @abstractmethod
    async def suppress_alert(self, alert_id: str, duration_minutes: int, reason: str) -> bool:
        """アラート抑制"""
        pass


class IAlertStorage(ABC):
    """アラートストレージインターフェース"""

    @abstractmethod
    async def save_alert(self, alert: Alert) -> bool:
        """アラート保存"""
        pass

    @abstractmethod
    async def get_alert(self, alert_id: str) -> Optional[Alert]:
        """アラート取得"""
        pass

    @abstractmethod
    async def get_alerts(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Alert]:
        """アラート一覧取得"""
        pass

    @abstractmethod
    async def update_alert_status(
        self,
        alert_id: str,
        status: AlertStatus,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """アラートステータス更新"""
        pass

    @abstractmethod
    async def delete_alert(self, alert_id: str) -> bool:
        """アラート削除"""
        pass

    @abstractmethod
    async def get_alert_statistics(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """アラート統計取得"""
        pass


class IEscalationManager(ABC):
    """エスカレーション管理インターフェース"""

    @abstractmethod
    async def check_escalation(self, alert: Alert) -> bool:
        """エスカレーション判定"""
        pass

    @abstractmethod
    async def escalate_alert(self, alert: Alert, escalation_level: int) -> bool:
        """アラートエスカレーション"""
        pass

    @abstractmethod
    def get_escalation_rules(self, rule_id: str) -> List[Dict[str, Any]]:
        """エスカレーションルール取得"""
        pass


class IAlertTemplateEngine(ABC):
    """アラートテンプレートエンジンインターフェース"""

    @abstractmethod
    async def render_message(
        self, template_name: str, context: Dict[str, Any], channel: NotificationChannel
    ) -> NotificationMessage:
        """メッセージレンダリング"""
        pass

    @abstractmethod
    async def register_template(
        self, template_name: str, template_content: str, channel: NotificationChannel
    ) -> bool:
        """テンプレート登録"""
        pass

    @abstractmethod
    async def get_available_templates(self) -> List[str]:
        """利用可能テンプレート一覧取得"""
        pass


# ヘルパー関数


def create_alert_id(source: str, timestamp: datetime) -> str:
    """アラートID生成"""
    import hashlib
    import uuid

    data = f"{source}:{timestamp.isoformat()}:{uuid.uuid4()}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def calculate_alert_priority(
    severity: AlertSeverity,
    affected_resources_count: int,
    business_impact_score: float = 0.5,
) -> int:
    """アラート優先度計算"""
    severity_weights = {
        AlertSeverity.LOW: 1,
        AlertSeverity.MEDIUM: 2,
        AlertSeverity.HIGH: 4,
        AlertSeverity.CRITICAL: 8,
    }

    base_score = severity_weights.get(severity, 1)
    resource_factor = min(affected_resources_count, 10) * 0.1
    business_factor = business_impact_score * 2

    return int(base_score * (1 + resource_factor + business_factor))


def is_alert_suppressed(alert: Alert, suppression_rules: List[Dict[str, Any]]) -> bool:
    """アラート抑制判定"""
    return any(_matches_suppression_rule(alert, rule) for rule in suppression_rules)


def _matches_suppression_rule(alert: Alert, rule: Dict[str, Any]) -> bool:
    """抑制ルールマッチング判定"""
    # 実装例：タグベースの抑制判定
    required_tags = rule.get("tags", {})
    for key, value in required_tags.items():
        if alert.tags.get(key) != value:
            return False

    # 時間ベースの抑制判定
    if "time_window" in rule:
        window = rule["time_window"]
        current_hour = datetime.now().hour
        if not (window["start"] <= current_hour <= window["end"]):
            return False

    return True


def format_alert_message(alert: Alert, template: str = "{title}: {message}") -> str:
    """アラートメッセージフォーマット"""
    try:
        return template.format(
            title=alert.title,
            message=alert.message,
            severity=alert.severity.value,
            status=alert.status.value,
            created_at=alert.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            alert_id=alert.alert_id,
        )
    except Exception:
        return f"{alert.title}: {alert.message}"
