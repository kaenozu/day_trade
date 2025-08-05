"""
アラート機能のデータベースモデル
永続化機能を提供
"""

from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.types import TypeDecorator

from ..core.alerts import AlertPriority
from ..models.enums import AlertType
from .base import BaseModel


class DecimalType(TypeDecorator):
    """Decimal型をデータベースに格納するためのカスタム型"""

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return Decimal(value)
        return value


class AlertConditionModel(BaseModel):
    """アラート条件のデータベースモデル"""

    __tablename__ = "alert_conditions"

    alert_id = Column(String(100), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    condition_value = Column(String(50), nullable=False)  # Decimal, float, str対応
    comparison_operator = Column(String(10), default=">")
    is_active = Column(Boolean, default=True, index=True)
    priority = Column(String(20), default="medium")

    # オプション設定
    cooldown_minutes = Column(Integer, default=60)
    expiry_date = Column(DateTime(timezone=True), nullable=True)
    description = Column(Text, default="")

    # カスタム条件用（JSON形式で保存）
    custom_parameters = Column(JSON, nullable=True)

    def to_alert_condition(self):
        """DataclassのAlertConditionに変換"""
        from ..core.alerts import AlertCondition

        return AlertCondition(
            alert_id=self.alert_id,
            symbol=self.symbol,
            alert_type=AlertType(self.alert_type),
            condition_value=self.condition_value,
            comparison_operator=self.comparison_operator,
            is_active=self.is_active,
            priority=AlertPriority(self.priority),
            cooldown_minutes=self.cooldown_minutes,
            expiry_date=self.expiry_date,
            description=self.description or "",
            custom_parameters=self.custom_parameters or {},
        )

    @classmethod
    def from_alert_condition(cls, condition):
        """DataclassのAlertConditionから作成"""
        return cls(
            alert_id=condition.alert_id,
            symbol=condition.symbol,
            alert_type=condition.alert_type.value,
            condition_value=str(condition.condition_value),
            comparison_operator=condition.comparison_operator,
            is_active=condition.is_active,
            priority=condition.priority.value,
            cooldown_minutes=condition.cooldown_minutes,
            expiry_date=condition.expiry_date,
            description=condition.description,
            custom_parameters=condition.custom_parameters,
        )


class AlertTriggerModel(BaseModel):
    """アラート発火記録のデータベースモデル"""

    __tablename__ = "alert_triggers"

    alert_id = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    trigger_time = Column(DateTime(timezone=True), nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    current_value = Column(String(50), nullable=False)
    condition_value = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    priority = Column(String(20), nullable=False)

    # 市場データ
    current_price = Column(DecimalType, nullable=True)
    volume = Column(Integer, nullable=True)
    change_percent = Column(String(20), nullable=True)  # float値をstring保存

    def to_alert_trigger(self):
        """DataclassのAlertTriggerに変換"""
        from ..core.alerts import AlertTrigger

        return AlertTrigger(
            alert_id=self.alert_id,
            symbol=self.symbol,
            trigger_time=self.trigger_time,
            alert_type=AlertType(self.alert_type),
            current_value=self.current_value,
            condition_value=self.condition_value,
            message=self.message,
            priority=AlertPriority(self.priority),
            current_price=self.current_price,
            volume=self.volume,
            change_percent=float(self.change_percent) if self.change_percent else None,
        )

    @classmethod
    def from_alert_trigger(cls, trigger):
        """DataclassのAlertTriggerから作成"""
        return cls(
            alert_id=trigger.alert_id,
            symbol=trigger.symbol,
            trigger_time=trigger.trigger_time,
            alert_type=trigger.alert_type.value,
            current_value=str(trigger.current_value),
            condition_value=str(trigger.condition_value),
            message=trigger.message,
            priority=trigger.priority.value,
            current_price=trigger.current_price,
            volume=trigger.volume,
            change_percent=str(trigger.change_percent)
            if trigger.change_percent is not None
            else None,
        )


class AlertConfigModel(BaseModel):
    """アラート設定のデータベースモデル"""

    __tablename__ = "alert_configs"

    config_key = Column(String(100), unique=True, nullable=False)
    config_value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)

    @classmethod
    def get_config(cls, session, key: str, default: Any = None) -> Any:
        """設定値を取得"""
        config = session.query(cls).filter_by(config_key=key).first()
        return config.config_value if config else default

    @classmethod
    def set_config(cls, session, key: str, value: Any, description: str = ""):
        """設定値を保存"""
        config = session.query(cls).filter_by(config_key=key).first()
        if config:
            config.config_value = value
            config.description = description
        else:
            config = cls(config_key=key, config_value=value, description=description)
            session.add(config)
        session.commit()
