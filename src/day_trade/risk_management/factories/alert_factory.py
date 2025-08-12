#!/usr/bin/env python3
"""
Alert Channel Factory
アラートチャネルファクトリー

通知チャネルの動的生成とアラート管理システム
"""

import importlib
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from ..exceptions.risk_exceptions import ConfigurationError, ValidationError
from ..interfaces.alert_interfaces import (
    IAlertProcessor,
    IAlertRule,
    INotificationChannel,
)


class NotificationChannelType(Enum):
    """通知チャネルタイプ"""

    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    SMS = "sms"
    WEBHOOK = "webhook"
    TELEGRAM = "telegram"
    PUSH_NOTIFICATION = "push_notification"
    PLUGIN = "plugin"


class AlertSeverity(Enum):
    """アラート重要度"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EscalationLevel(Enum):
    """エスカレーションレベル"""

    LEVEL_1 = "level_1"  # 一次対応
    LEVEL_2 = "level_2"  # 二次対応
    LEVEL_3 = "level_3"  # 緊急対応
    EXECUTIVE = "executive"  # 経営陣


class AlertChannelFactory:
    """アラートチャネルファクトリー"""

    def __init__(self):
        self._channel_registry: Dict[NotificationChannelType, Type[INotificationChannel]] = {}
        self._processor_registry: Dict[str, Type[IAlertProcessor]] = {}
        self._rule_registry: Dict[str, Type[IAlertRule]] = {}
        self._plugin_registry: Dict[str, Type[INotificationChannel]] = {}
        self._config_schemas: Dict[NotificationChannelType, Dict[str, Any]] = {}
        self._instance_cache: Dict[str, INotificationChannel] = {}
        self._global_config: Dict[str, Any] = {}

        # 組み込みチャネルを登録
        self._register_builtin_channels()
        self._register_builtin_processors()
        self._register_builtin_rules()

    def _register_builtin_channels(self):
        """組み込みチャネル登録"""
        try:
            # Email通知
            self.register_channel(
                NotificationChannelType.EMAIL,
                "src.day_trade.risk_management.notifications.email_channel",
                "EmailNotificationChannel",
                {
                    "smtp_server": {"type": str, "required": True},
                    "smtp_port": {"type": int, "default": 587},
                    "username": {"type": str, "required": True},
                    "password": {"type": str, "required": True},
                    "use_tls": {"type": bool, "default": True},
                    "from_email": {"type": str, "required": True},
                    "max_retries": {"type": int, "default": 3},
                    "timeout_seconds": {"type": int, "default": 30},
                },
            )

            # Slack通知
            self.register_channel(
                NotificationChannelType.SLACK,
                "src.day_trade.risk_management.notifications.slack_channel",
                "SlackNotificationChannel",
                {
                    "webhook_url": {"type": str, "required": True},
                    "default_channel": {"type": str, "required": False},
                    "username": {"type": str, "default": "Risk Alert Bot"},
                    "icon_emoji": {"type": str, "default": ":warning:"},
                    "mention_users": {"type": list, "default": []},
                    "thread_ts": {"type": str, "required": False},
                },
            )

            # Discord通知
            self.register_channel(
                NotificationChannelType.DISCORD,
                "src.day_trade.risk_management.notifications.discord_channel",
                "DiscordNotificationChannel",
                {
                    "webhook_url": {"type": str, "required": True},
                    "username": {"type": str, "default": "Risk Alert"},
                    "avatar_url": {"type": str, "required": False},
                    "color": {"type": int, "default": 16711680},  # Red
                },
            )

            # Webhook通知
            self.register_channel(
                NotificationChannelType.WEBHOOK,
                "src.day_trade.risk_management.notifications.webhook_channel",
                "WebhookNotificationChannel",
                {
                    "url": {"type": str, "required": True},
                    "method": {"type": str, "default": "POST"},
                    "headers": {
                        "type": dict,
                        "default": {"Content-Type": "application/json"},
                    },
                    "auth_type": {
                        "type": str,
                        "default": "none",
                    },  # "none", "basic", "bearer"
                    "auth_credentials": {"type": dict, "default": {}},
                    "timeout_seconds": {"type": int, "default": 30},
                    "verify_ssl": {"type": bool, "default": True},
                },
            )

            # SMS通知
            self.register_channel(
                NotificationChannelType.SMS,
                "src.day_trade.risk_management.notifications.sms_channel",
                "SMSNotificationChannel",
                {
                    "provider": {
                        "type": str,
                        "default": "twilio",
                    },  # "twilio", "aws_sns"
                    "api_key": {"type": str, "required": True},
                    "api_secret": {"type": str, "required": True},
                    "from_number": {"type": str, "required": True},
                    "region": {"type": str, "default": "us-east-1"},
                },
            )

            # Telegram通知
            self.register_channel(
                NotificationChannelType.TELEGRAM,
                "src.day_trade.risk_management.notifications.telegram_channel",
                "TelegramNotificationChannel",
                {
                    "bot_token": {"type": str, "required": True},
                    "default_chat_id": {"type": str, "required": True},
                    "parse_mode": {"type": str, "default": "HTML"},
                    "disable_notification": {"type": bool, "default": False},
                },
            )

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to register some builtin notification channels: {e}")

    def _register_builtin_processors(self):
        """組み込みプロセッサー登録"""
        try:
            # 基本アラートプロセッサー
            self._processor_registry["basic"] = self._import_class(
                "src.day_trade.risk_management.alerts.basic_processor",
                "BasicAlertProcessor",
            )

            # 重複排除プロセッサー
            self._processor_registry["deduplication"] = self._import_class(
                "src.day_trade.risk_management.alerts.deduplication_processor",
                "DeduplicationAlertProcessor",
            )

            # エスカレーションプロセッサー
            self._processor_registry["escalation"] = self._import_class(
                "src.day_trade.risk_management.alerts.escalation_processor",
                "EscalationAlertProcessor",
            )

            # バッチプロセッサー
            self._processor_registry["batch"] = self._import_class(
                "src.day_trade.risk_management.alerts.batch_processor",
                "BatchAlertProcessor",
            )

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to register some builtin alert processors: {e}")

    def _register_builtin_rules(self):
        """組み込みルール登録"""
        try:
            # 閾値ルール
            self._rule_registry["threshold"] = self._import_class(
                "src.day_trade.risk_management.alerts.threshold_rule",
                "ThresholdAlertRule",
            )

            # 時間ベースルール
            self._rule_registry["time_based"] = self._import_class(
                "src.day_trade.risk_management.alerts.time_based_rule",
                "TimeBasedAlertRule",
            )

            # 複合条件ルール
            self._rule_registry["composite"] = self._import_class(
                "src.day_trade.risk_management.alerts.composite_rule",
                "CompositeAlertRule",
            )

            # 機械学習ベースルール
            self._rule_registry["ml_based"] = self._import_class(
                "src.day_trade.risk_management.alerts.ml_rule", "MLBasedAlertRule"
            )

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to register some builtin alert rules: {e}")

    def register_channel(
        self,
        channel_type: NotificationChannelType,
        module_path: str,
        class_name: str,
        config_schema: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """チャネル登録"""
        try:
            module = importlib.import_module(module_path)
            channel_class = getattr(module, class_name)

            if not issubclass(channel_class, INotificationChannel):
                raise ConfigurationError(
                    f"Notification channel class {class_name} must implement INotificationChannel interface",
                    config_key=f"alert.channel.{channel_type.value}",
                )

            self._channel_registry[channel_type] = channel_class

            if config_schema:
                self._config_schemas[channel_type] = config_schema

            return True

        except Exception as e:
            raise ConfigurationError(
                f"Failed to register notification channel type {channel_type.value}",
                config_key=f"alert.channel.{channel_type.value}",
                cause=e,
            ) from e

    def register_plugin_channel(
        self,
        plugin_name: str,
        channel_class: Type[INotificationChannel],
        config_schema: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """プラグインチャネル登録"""
        try:
            if not issubclass(channel_class, INotificationChannel):
                raise ConfigurationError(
                    f"Plugin notification channel {plugin_name} must implement INotificationChannel interface",
                    config_key=f"alert.plugin.{plugin_name}",
                )

            self._plugin_registry[plugin_name] = channel_class

            if config_schema:
                self._config_schemas[NotificationChannelType.PLUGIN] = self._config_schemas.get(
                    NotificationChannelType.PLUGIN, {}
                )
                self._config_schemas[NotificationChannelType.PLUGIN][plugin_name] = config_schema

            return True

        except Exception as e:
            raise ConfigurationError(
                f"Failed to register plugin notification channel {plugin_name}",
                config_key=f"alert.plugin.{plugin_name}",
                cause=e,
            ) from e

    def create_notification_channel(
        self,
        channel_type: NotificationChannelType,
        config: Optional[Dict[str, Any]] = None,
        plugin_name: Optional[str] = None,
        use_cache: bool = True,
    ) -> INotificationChannel:
        """通知チャネル作成"""

        # キャッシュキー生成
        cache_key = self._generate_cache_key(channel_type, plugin_name, config)

        # キャッシュからインスタンス取得
        if use_cache and cache_key in self._instance_cache:
            return self._instance_cache[cache_key]

        # 設定検証
        validated_config = self._validate_config(channel_type, config, plugin_name)

        # チャネルクラス取得
        channel_class = self._get_channel_class(channel_type, plugin_name)

        try:
            # インスタンス作成
            instance = self._create_channel_instance(channel_class, validated_config)

            # キャッシュに保存
            if use_cache:
                self._instance_cache[cache_key] = instance

            return instance

        except Exception as e:
            raise ConfigurationError(
                f"Failed to create notification channel instance of type {channel_type.value}",
                config_key=f"alert.channel.{channel_type.value}",
                cause=e,
            ) from e

    def create_multi_channel_notifier(
        self,
        channel_configs: List[Dict[str, Any]],
        routing_rules: Optional[Dict[str, List[str]]] = None,
        fallback_channels: Optional[List[str]] = None,
    ) -> "MultiChannelNotifier":  # TODO: MultiChannelNotifier クラスを実装 # type: ignore
        """マルチチャネル通知システム作成"""

        channels = {}

        for config in channel_configs:
            channel_type = NotificationChannelType(config["type"])
            plugin_name = config.get("plugin_name")
            channel_config = config.get("config", {})
            channel_name = config.get("name", channel_type.value)

            channel = self.create_notification_channel(
                channel_type, channel_config, plugin_name, use_cache=True
            )
            channels[channel_name] = channel

        # マルチチャネル通知システム作成
        from ..alerts.multi_channel_notifier import MultiChannelNotifier

        return MultiChannelNotifier(
            channels=channels,
            routing_rules=routing_rules or {},
            fallback_channels=fallback_channels or [],
        )

    def create_escalation_manager(
        self,
        escalation_policies: List[Dict[str, Any]],
        notification_channels: Dict[str, Any],
    ) -> "EscalationManager":  # TODO: EscalationManager クラスを実装 # type: ignore
        """エスカレーション管理システム作成"""

        # エスカレーションルール作成
        escalation_rules = {}

        for level, rule_config in escalation_policies.get("rules", {}).items():
            escalation_level = EscalationLevel(level)

            channels = []
            for channel_config in rule_config.get("channels", []):
                channel = self.create_notification_channel(
                    NotificationChannelType(channel_config["type"]),
                    channel_config.get("config", {}),
                    channel_config.get("plugin_name"),
                )
                channels.append(channel)

            escalation_rules[escalation_level] = {
                "channels": channels,
                "delay_minutes": rule_config.get("delay_minutes", 0),
                "max_attempts": rule_config.get("max_attempts", 3),
                "conditions": rule_config.get("conditions", []),
            }

        # エスカレーション管理システム作成
        from ..alerts.escalation_manager import EscalationManager

        return EscalationManager(
            escalation_rules=escalation_rules,
            default_timeout_minutes=escalation_policies.get("default_timeout_minutes", 60),
        )

    def create_alert_processor(
        self, processor_type: str, config: Optional[Dict[str, Any]] = None
    ) -> IAlertProcessor:
        """アラートプロセッサー作成"""

        if processor_type not in self._processor_registry:
            raise ConfigurationError(
                f"Alert processor type '{processor_type}' is not registered",
                config_key=f"alert.processor.{processor_type}",
            )

        processor_class = self._processor_registry[processor_type]
        return processor_class(config or {})

    def create_alert_rule(
        self, rule_type: str, config: Optional[Dict[str, Any]] = None
    ) -> IAlertRule:
        """アラートルール作成"""

        if rule_type not in self._rule_registry:
            raise ConfigurationError(
                f"Alert rule type '{rule_type}' is not registered",
                config_key=f"alert.rule.{rule_type}",
            )

        rule_class = self._rule_registry[rule_type]
        return rule_class(config or {})

    def get_available_channels(self) -> Dict[str, Dict[str, Any]]:
        """利用可能チャネル一覧取得"""

        available = {}

        # 組み込みチャネル
        for channel_type in self._channel_registry:
            config_schema = self._config_schemas.get(channel_type, {})
            available[channel_type.value] = {
                "type": "builtin",
                "name": channel_type.value,
                "config_schema": config_schema,
                "supported_features": self._get_channel_features(channel_type),
            }

        # プラグインチャネル
        for plugin_name, channel_class in self._plugin_registry.items():
            try:
                temp_instance = channel_class({})
                if hasattr(temp_instance, "get_metadata"):
                    metadata = temp_instance.get_metadata()
                    available[plugin_name] = {
                        "type": "plugin",
                        "name": metadata.get("name", plugin_name),
                        "version": metadata.get("version", "unknown"),
                        "description": metadata.get("description", ""),
                        "config_schema": self._config_schemas.get(
                            NotificationChannelType.PLUGIN, {}
                        ).get(plugin_name, {}),
                    }
                else:
                    available[plugin_name] = {"type": "plugin", "name": plugin_name}
            except Exception:
                available[plugin_name] = {
                    "type": "plugin",
                    "name": plugin_name,
                    "status": "unavailable",
                }

        return available

    def set_global_config(self, config: Dict[str, Any]):
        """グローバル設定更新"""
        self._global_config.update(config)

    def clear_cache(self):
        """インスタンスキャッシュクリア"""
        # アクティブな接続を適切に閉じる
        for instance in self._instance_cache.values():
            if hasattr(instance, "close"):
                try:
                    instance.close()
                except Exception:
                    pass

        self._instance_cache.clear()

    def _validate_config(
        self,
        channel_type: NotificationChannelType,
        config: Optional[Dict[str, Any]],
        plugin_name: Optional[str],
    ) -> Dict[str, Any]:
        """設定検証"""

        if config is None:
            config = {}

        # グローバル設定をマージ
        merged_config = self._global_config.copy()
        merged_config.update(config)

        # スキーマ取得
        if channel_type == NotificationChannelType.PLUGIN and plugin_name:
            schema = self._config_schemas.get(NotificationChannelType.PLUGIN, {}).get(
                plugin_name, {}
            )
        else:
            schema = self._config_schemas.get(channel_type, {})

        validated_config = {}

        # 必須フィールドチェックとデフォルト値適用
        for field_name, field_schema in schema.items():
            if field_schema.get("required", False) and field_name not in merged_config:
                raise ValidationError(
                    f"Required configuration field '{field_name}' is missing for {channel_type.value}",
                    field_name=field_name,
                    validation_rules=["required"],
                )

            if field_name not in merged_config and "default" in field_schema:
                validated_config[field_name] = field_schema["default"]
            elif field_name in merged_config:
                # 型チェック
                expected_type = field_schema.get("type")
                if expected_type and not isinstance(merged_config[field_name], expected_type):
                    raise ValidationError(
                        f"Configuration field '{field_name}' must be of type {expected_type.__name__}",
                        field_name=field_name,
                        invalid_value=merged_config[field_name],
                        validation_rules=[f"type:{expected_type.__name__}"],
                    )
                validated_config[field_name] = merged_config[field_name]

        # 追加設定フィールドも含める
        for field_name, field_value in merged_config.items():
            if field_name not in validated_config:
                validated_config[field_name] = field_value

        return validated_config

    def _get_channel_class(
        self, channel_type: NotificationChannelType, plugin_name: Optional[str]
    ) -> Type[INotificationChannel]:
        """チャネルクラス取得"""

        if channel_type == NotificationChannelType.PLUGIN:
            if not plugin_name:
                raise ConfigurationError(
                    "Plugin name is required for plugin notification channel type",
                    config_key="plugin_name",
                )

            if plugin_name not in self._plugin_registry:
                raise ConfigurationError(
                    f"Plugin notification channel '{plugin_name}' is not registered",
                    config_key=f"alert.plugin.{plugin_name}",
                )

            return self._plugin_registry[plugin_name]

        else:
            if channel_type not in self._channel_registry:
                raise ConfigurationError(
                    f"Notification channel type '{channel_type.value}' is not registered",
                    config_key=f"alert.channel.{channel_type.value}",
                )

            return self._channel_registry[channel_type]

    def _create_channel_instance(
        self, channel_class: Type[INotificationChannel], config: Dict[str, Any]
    ) -> INotificationChannel:
        """チャネルインスタンス作成"""
        return channel_class(config)

    def _get_channel_features(self, channel_type: NotificationChannelType) -> List[str]:
        """チャネル機能取得"""
        features = ["send_message"]

        if channel_type in [
            NotificationChannelType.SLACK,
            NotificationChannelType.DISCORD,
        ]:
            features.extend(["rich_formatting", "attachments", "threading"])

        if channel_type == NotificationChannelType.EMAIL:
            features.extend(["html_formatting", "attachments", "templates"])

        if channel_type == NotificationChannelType.WEBHOOK:
            features.extend(["custom_payload", "authentication", "retries"])

        return features

    def _import_class(self, module_path: str, class_name: str) -> Type:
        """クラス動的インポート"""
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            return None

    def _generate_cache_key(
        self,
        channel_type: NotificationChannelType,
        plugin_name: Optional[str],
        config: Optional[Dict[str, Any]],
    ) -> str:
        """キャッシュキー生成"""
        import hashlib
        import json

        key_data = {
            "type": channel_type.value,
            "plugin": plugin_name,
            "config": config or {},
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()


# グローバルファクトリーインスタンス
_global_alert_factory: Optional[AlertChannelFactory] = None


def get_alert_factory() -> AlertChannelFactory:
    """グローバルアラートファクトリー取得"""
    global _global_alert_factory
    if _global_alert_factory is None:
        _global_alert_factory = AlertChannelFactory()
    return _global_alert_factory


def create_notification_channel(
    channel_type: NotificationChannelType,
    config: Optional[Dict[str, Any]] = None,
    plugin_name: Optional[str] = None,
) -> INotificationChannel:
    """通知チャネル作成（便利関数）"""
    factory = get_alert_factory()
    return factory.create_notification_channel(channel_type, config, plugin_name)
