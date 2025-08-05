"""
アラート関連のデータベースモデル
AlertConditionとAlertTriggerの永続化
"""

import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, Column, DateTime, Enum, String, Text
from sqlalchemy.types import DECIMAL

from .base import BaseModel
from .enums import AlertType
from ..core.alerts import AlertPriority


class PersistentAlertCondition(BaseModel):
    """永続化されたアラート条件"""

    __tablename__ = "alert_conditions"

    alert_id = Column(String(100), unique=True, nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    alert_type = Column(Enum(AlertType), nullable=False)
    condition_value = Column(DECIMAL(precision=20, scale=8), nullable=False)
    comparison_operator = Column(String(10), default=">")
    priority = Column(Enum(AlertPriority), default=AlertPriority.MEDIUM)
    description = Column(String(500))

    # フラグ類
    is_active = Column(Boolean, default=True)
    is_expired = Column(Boolean, default=False)

    # 期限設定
    expires_at = Column(DateTime, nullable=True)

    # カスタムパラメーター（JSON形式で保存）
    custom_parameters_json = Column(Text, nullable=True)

    # 通知設定（JSON形式で保存）
    notification_methods_json = Column(Text, nullable=True)

    @property
    def custom_parameters(self) -> Optional[Dict[str, Any]]:
        """カスタムパラメーターを辞書として取得"""
        if self.custom_parameters_json:
            try:
                return json.loads(self.custom_parameters_json)
            except json.JSONDecodeError:
                return None
        return None

    @custom_parameters.setter
    def custom_parameters(self, value: Optional[Dict[str, Any]]):
        """カスタムパラメーターをJSON文字列として保存"""
        if value is not None:
            self.custom_parameters_json = json.dumps(value, ensure_ascii=False)
        else:
            self.custom_parameters_json = None

    @property
    def notification_methods(self) -> Optional[list]:
        """通知方法をリストとして取得"""
        if self.notification_methods_json:
            try:
                return json.loads(self.notification_methods_json)
            except json.JSONDecodeError:
                return None
        return None

    @notification_methods.setter
    def notification_methods(self, value: Optional[list]):
        """通知方法をJSON文字列として保存"""
        if value is not None:
            # Enumオブジェクトを文字列に変換
            serializable_value = [
                item.value if hasattr(item, 'value') else item for item in value
            ]
            self.notification_methods_json = json.dumps(serializable_value, ensure_ascii=False)
        else:
            self.notification_methods_json = None

    def to_alert_condition(self):
        """メモリ上のAlertConditionオブジェクトに変換"""
        from ..core.alerts import AlertCondition

        return AlertCondition(
            alert_id=self.alert_id,
            symbol=self.symbol,
            alert_type=self.alert_type,
            condition_value=self.condition_value,
            comparison_operator=self.comparison_operator,
            priority=self.priority,
            description=self.description,
            is_active=self.is_active,
            expires_at=self.expires_at,
            custom_parameters=self.custom_parameters,
            notification_methods=self.notification_methods,
            custom_function=None  # カスタム関数はセキュリティ上永続化しない
        )

    @classmethod
    def from_alert_condition(cls, condition):
        """メモリ上のAlertConditionから作成"""
        return cls(
            alert_id=condition.alert_id,
            symbol=condition.symbol,
            alert_type=condition.alert_type,
            condition_value=condition.condition_value,
            comparison_operator=condition.comparison_operator,
            priority=condition.priority,
            description=condition.description,
            is_active=condition.is_active,
            expires_at=condition.expires_at,
            custom_parameters=condition.custom_parameters,
            notification_methods=condition.notification_methods
        )


class PersistentAlertTrigger(BaseModel):
    """永続化されたアラートトリガー（履歴）"""

    __tablename__ = "alert_triggers"

    trigger_id = Column(String(100), unique=True, nullable=False, index=True)
    alert_id = Column(String(100), nullable=False, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    alert_type = Column(Enum(AlertType), nullable=False)

    trigger_time = Column(DateTime, nullable=False, default=datetime.now)

    # トリガー時の値
    current_value = Column(DECIMAL(precision=20, scale=8), nullable=True)
    condition_value = Column(DECIMAL(precision=20, scale=8), nullable=False)
    current_price = Column(DECIMAL(precision=10, scale=2), nullable=True)
    volume = Column(String(20), nullable=True)
    change_percent = Column(DECIMAL(precision=8, scale=4), nullable=True)

    message = Column(String(1000), nullable=False)
    priority = Column(Enum(AlertPriority), default=AlertPriority.MEDIUM)

    # 通知状態
    notification_sent = Column(Boolean, default=False)
    notification_methods_json = Column(Text, nullable=True)

    # エラー情報
    notification_error = Column(Text, nullable=True)

    @property
    def notification_methods(self) -> Optional[list]:
        """通知方法をリストとして取得"""
        if self.notification_methods_json:
            try:
                return json.loads(self.notification_methods_json)
            except json.JSONDecodeError:
                return None
        return None

    @notification_methods.setter
    def notification_methods(self, value: Optional[list]):
        """通知方法をJSON文字列として保存"""
        if value is not None:
            serializable_value = [
                item.value if hasattr(item, 'value') else item for item in value
            ]
            self.notification_methods_json = json.dumps(serializable_value, ensure_ascii=False)
        else:
            self.notification_methods_json = None

    def to_alert_trigger(self):
        """メモリ上のAlertTriggerオブジェクトに変換"""
        from ..core.alerts import AlertTrigger

        return AlertTrigger(
            alert_id=self.alert_id,
            symbol=self.symbol,
            trigger_time=self.trigger_time,
            alert_type=self.alert_type,
            current_value=self.current_value,
            condition_value=self.condition_value,
            message=self.message,
            priority=self.priority,
            current_price=self.current_price,
            volume=int(self.volume) if self.volume and self.volume.isdigit() else 0,
            change_percent=float(self.change_percent) if self.change_percent else 0.0,
            notification_methods=self.notification_methods
        )

    @classmethod
    def from_alert_trigger(cls, trigger, trigger_id: Optional[str] = None):
        """メモリ上のAlertTriggerから作成"""
        import uuid

        return cls(
            trigger_id=trigger_id or str(uuid.uuid4()),
            alert_id=trigger.alert_id,
            symbol=trigger.symbol,
            alert_type=trigger.alert_type,
            trigger_time=trigger.trigger_time,
            current_value=trigger.current_value,
            condition_value=trigger.condition_value,
            current_price=trigger.current_price,
            volume=str(trigger.volume) if trigger.volume else None,
            change_percent=Decimal(str(trigger.change_percent)) if trigger.change_percent else None,
            message=trigger.message,
            priority=trigger.priority,
            notification_methods=trigger.notification_methods
        )
