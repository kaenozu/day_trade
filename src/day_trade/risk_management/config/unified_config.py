#!/usr/bin/env python3
"""
Unified Configuration System
統一設定システム

環境変数、設定ファイル、動的設定を統合管理する中央設定システム
"""

import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..interfaces.risk_interfaces import IConfigManager


class ConfigSource(Enum):
    """設定ソース"""

    ENVIRONMENT = "environment"
    FILE = "file"
    DATABASE = "database"
    REMOTE = "remote"
    DEFAULT = "default"


@dataclass
class DatabaseConfig:
    """データベース設定"""

    host: str = "localhost"
    port: int = 5432
    database: str = "risk_management"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    connection_pool_size: int = 10
    connection_timeout: int = 30
    query_timeout: int = 60

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """環境変数から設定作成"""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "risk_management"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            ssl_mode=os.getenv("DB_SSL_MODE", "prefer"),
            connection_pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            connection_timeout=int(os.getenv("DB_CONNECTION_TIMEOUT", "30")),
            query_timeout=int(os.getenv("DB_QUERY_TIMEOUT", "60")),
        )


@dataclass
class CacheConfig:
    """キャッシュ設定"""

    provider: str = "memory"  # memory, redis, memcached
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ssl_enabled: bool = False
    connection_pool_size: int = 10
    default_ttl_seconds: int = 3600
    max_memory_mb: int = 512
    eviction_policy: str = "lru"

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """環境変数から設定作成"""
        return cls(
            provider=os.getenv("CACHE_PROVIDER", "memory"),
            host=os.getenv("CACHE_HOST", "localhost"),
            port=int(os.getenv("CACHE_PORT", "6379")),
            database=int(os.getenv("CACHE_DB", "0")),
            password=os.getenv("CACHE_PASSWORD"),
            ssl_enabled=os.getenv("CACHE_SSL", "false").lower() == "true",
            connection_pool_size=int(os.getenv("CACHE_POOL_SIZE", "10")),
            default_ttl_seconds=int(os.getenv("CACHE_DEFAULT_TTL", "3600")),
            max_memory_mb=int(os.getenv("CACHE_MAX_MEMORY_MB", "512")),
            eviction_policy=os.getenv("CACHE_EVICTION_POLICY", "lru"),
        )


@dataclass
class AlertConfig:
    """アラート設定"""

    enabled: bool = True
    default_channels: List[str] = field(default_factory=lambda: ["email"])
    throttle_minutes: int = 5
    max_alerts_per_hour: int = 100
    escalation_enabled: bool = True
    auto_resolve_minutes: int = 60

    # 通知チャネル設定
    email_config: Dict[str, str] = field(default_factory=dict)
    slack_config: Dict[str, str] = field(default_factory=dict)
    webhook_config: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "AlertConfig":
        """環境変数から設定作成"""
        return cls(
            enabled=os.getenv("ALERTS_ENABLED", "true").lower() == "true",
            default_channels=os.getenv("ALERT_DEFAULT_CHANNELS", "email").split(","),
            throttle_minutes=int(os.getenv("ALERT_THROTTLE_MINUTES", "5")),
            max_alerts_per_hour=int(os.getenv("ALERT_MAX_PER_HOUR", "100")),
            escalation_enabled=os.getenv("ALERT_ESCALATION_ENABLED", "true").lower()
            == "true",
            auto_resolve_minutes=int(os.getenv("ALERT_AUTO_RESOLVE_MINUTES", "60")),
            email_config={
                "smtp_host": os.getenv("EMAIL_SMTP_HOST", ""),
                "smtp_port": os.getenv("EMAIL_SMTP_PORT", "587"),
                "username": os.getenv("EMAIL_USERNAME", ""),
                "password": os.getenv("EMAIL_PASSWORD", ""),
                "use_tls": os.getenv("EMAIL_USE_TLS", "true"),
            },
            slack_config={
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
                "channel": os.getenv("SLACK_CHANNEL", "#alerts"),
                "username": os.getenv("SLACK_USERNAME", "Risk Bot"),
            },
        )


@dataclass
class SecurityConfig:
    """セキュリティ設定"""

    encryption_key: str = ""
    api_key_required: bool = True
    jwt_secret: str = ""
    jwt_expiry_minutes: int = 60
    rate_limit_requests_per_minute: int = 100
    allowed_origins: List[str] = field(default_factory=list)
    audit_logging_enabled: bool = True

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """環境変数から設定作成"""
        return cls(
            encryption_key=os.getenv("ENCRYPTION_KEY", ""),
            api_key_required=os.getenv("API_KEY_REQUIRED", "true").lower() == "true",
            jwt_secret=os.getenv("JWT_SECRET", ""),
            jwt_expiry_minutes=int(os.getenv("JWT_EXPIRY_MINUTES", "60")),
            rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "100")),
            allowed_origins=(
                os.getenv("ALLOWED_ORIGINS", "").split(",")
                if os.getenv("ALLOWED_ORIGINS")
                else []
            ),
            audit_logging_enabled=os.getenv("AUDIT_LOGGING", "true").lower() == "true",
        )


@dataclass
class AIConfig:
    """AI設定"""

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    default_model_gpt: str = "gpt-4"
    default_model_claude: str = "claude-3-opus-20240229"
    temperature: float = 0.3
    max_tokens: int = 1000
    timeout_seconds: int = 10
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    retry_attempts: int = 3

    @classmethod
    def from_env(cls) -> "AIConfig":
        """環境変数から設定作成"""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            default_model_gpt=os.getenv("DEFAULT_GPT_MODEL", "gpt-4"),
            default_model_claude=os.getenv(
                "DEFAULT_CLAUDE_MODEL", "claude-3-opus-20240229"
            ),
            temperature=float(os.getenv("AI_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("AI_MAX_TOKENS", "1000")),
            timeout_seconds=int(os.getenv("AI_TIMEOUT_SECONDS", "10")),
            enable_caching=os.getenv("AI_ENABLE_CACHING", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("AI_CACHE_TTL", "300")),
            retry_attempts=int(os.getenv("AI_RETRY_ATTEMPTS", "3")),
        )


@dataclass
class MonitoringConfig:
    """監視設定"""

    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval_seconds: int = 30
    performance_sampling_rate: float = 1.0
    log_level: str = "INFO"
    prometheus_enabled: bool = False
    jaeger_enabled: bool = False

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """環境変数から設定作成"""
        return cls(
            enabled=os.getenv("MONITORING_ENABLED", "true").lower() == "true",
            metrics_port=int(os.getenv("METRICS_PORT", "9090")),
            health_check_interval_seconds=int(os.getenv("HEALTH_CHECK_INTERVAL", "30")),
            performance_sampling_rate=float(
                os.getenv("PERFORMANCE_SAMPLING_RATE", "1.0")
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            prometheus_enabled=os.getenv("PROMETHEUS_ENABLED", "false").lower()
            == "true",
            jaeger_enabled=os.getenv("JAEGER_ENABLED", "false").lower() == "true",
        )


@dataclass
class EnvironmentConfig:
    """環境設定"""

    environment: str = "development"
    debug: bool = True
    testing: bool = False
    app_name: str = "Risk Management System"
    app_version: str = "1.0.0"
    timezone: str = "UTC"

    @classmethod
    def from_env(cls) -> "EnvironmentConfig":
        """環境変数から設定作成"""
        env = os.getenv("ENVIRONMENT", "development").lower()
        return cls(
            environment=env,
            debug=os.getenv(
                "DEBUG", "true" if env == "development" else "false"
            ).lower()
            == "true",
            testing=os.getenv("TESTING", "false").lower() == "true",
            app_name=os.getenv("APP_NAME", "Risk Management System"),
            app_version=os.getenv("APP_VERSION", "1.0.0"),
            timezone=os.getenv("TIMEZONE", "UTC"),
        )


@dataclass
class RiskManagementConfig:
    """統一リスク管理設定"""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # 動的設定
    _dynamic_config: Dict[str, Any] = field(default_factory=dict, init=False)
    _config_sources: Dict[str, ConfigSource] = field(default_factory=dict, init=False)

    @classmethod
    def from_env(cls) -> "RiskManagementConfig":
        """環境変数から設定作成"""
        return cls(
            environment=EnvironmentConfig.from_env(),
            database=DatabaseConfig.from_env(),
            cache=CacheConfig.from_env(),
            alerts=AlertConfig.from_env(),
            security=SecurityConfig.from_env(),
            ai=AIConfig.from_env(),
            monitoring=MonitoringConfig.from_env(),
        )

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "RiskManagementConfig":
        """ファイルから設定読み込み"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path.suffix.lower() == ".json":
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)
        elif config_path.suffix.lower() in [".yml", ".yaml"]:
            with open(config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        # データクラスに変換
        return cls._from_dict(config_data)

    @classmethod
    def _from_dict(cls, config_data: Dict[str, Any]) -> "RiskManagementConfig":
        """辞書データから設定作成"""
        config = cls()

        for section_name, section_data in config_data.items():
            if hasattr(config, section_name):
                section_obj = getattr(config, section_name)
                if hasattr(section_obj, "__dataclass_fields__"):
                    # データクラスの場合、フィールドを更新
                    for field_name, field_value in section_data.items():
                        if hasattr(section_obj, field_name):
                            setattr(section_obj, field_name, field_value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        config_dict = {}

        for field_name in self.__dataclass_fields__:
            if not field_name.startswith("_"):
                field_value = getattr(self, field_name)
                if hasattr(field_value, "__dataclass_fields__"):
                    config_dict[field_name] = asdict(field_value)
                else:
                    config_dict[field_name] = field_value

        # 動的設定も含める
        if self._dynamic_config:
            config_dict["dynamic"] = self._dynamic_config

        return config_dict

    def save_to_file(self, config_path: Union[str, Path]) -> bool:
        """ファイルに設定保存"""
        try:
            config_path = Path(config_path)
            config_dict = self.to_dict()

            if config_path.suffix.lower() == ".json":
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
            elif config_path.suffix.lower() in [".yml", ".yaml"]:
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(
                        config_dict, f, default_flow_style=False, allow_unicode=True
                    )
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

            return True
        except Exception:
            return False

    def validate(self) -> List[str]:
        """設定検証"""
        errors = []

        # 必須項目チェック
        if (
            not self.security.encryption_key
            and self.environment.environment == "production"
        ):
            errors.append("Encryption key is required in production environment")

        if not self.database.password and self.environment.environment == "production":
            errors.append("Database password is required in production environment")

        # AI設定チェック
        if not self.ai.openai_api_key and not self.ai.anthropic_api_key:
            errors.append("At least one AI API key (OpenAI or Anthropic) is required")

        # ポート番号チェック
        if not (1 <= self.database.port <= 65535):
            errors.append(f"Invalid database port: {self.database.port}")

        if not (1 <= self.cache.port <= 65535):
            errors.append(f"Invalid cache port: {self.cache.port}")

        return errors

    def get_dynamic_config(self, key: str, default: Any = None) -> Any:
        """動的設定取得"""
        return self._dynamic_config.get(key, default)

    def set_dynamic_config(
        self, key: str, value: Any, source: ConfigSource = ConfigSource.DEFAULT
    ) -> None:
        """動的設定更新"""
        self._dynamic_config[key] = value
        self._config_sources[key] = source


class ConfigManager(IConfigManager):
    """設定管理器"""

    def __init__(self, config: Optional[RiskManagementConfig] = None):
        self.config = config or RiskManagementConfig.from_env()
        self._change_callbacks: Dict[str, List[callable]] = {}
        self._last_loaded: Optional[datetime] = None

        # 設定検証
        validation_errors = self.config.validate()
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"- {error}" for error in validation_errors
            )
            raise ValueError(error_msg)

    async def get_config(self, key: str, default: Any = None) -> Any:
        """設定取得"""
        parts = key.split(".")
        current = self.config

        try:
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif hasattr(current, "_dynamic_config"):
                    return current.get_dynamic_config(key, default)
                else:
                    return default
            return current
        except Exception:
            return default

    async def set_config(self, key: str, value: Any) -> bool:
        """設定更新"""
        try:
            # 動的設定として保存
            self.config.set_dynamic_config(key, value, ConfigSource.DEFAULT)

            # 変更コールバック実行
            await self._notify_config_change(key, value)
            return True
        except Exception:
            return False

    async def get_all_configs(self) -> Dict[str, Any]:
        """全設定取得"""
        return self.config.to_dict()

    async def reload_config(self) -> bool:
        """設定リロード"""
        try:
            # 環境変数から再読み込み
            new_config = RiskManagementConfig.from_env()
            self.config = new_config
            self._last_loaded = datetime.now()

            # 変更通知
            await self._notify_config_change("*", None)
            return True
        except Exception:
            return False

    def subscribe_to_changes(
        self, callback: callable, key_pattern: Optional[str] = None
    ) -> str:
        """設定変更購読"""
        pattern = key_pattern or "*"
        if pattern not in self._change_callbacks:
            self._change_callbacks[pattern] = []

        self._change_callbacks[pattern].append(callback)

        # サブスクリプションID生成
        import uuid

        return str(uuid.uuid4())

    async def _notify_config_change(self, key: str, value: Any) -> None:
        """設定変更通知"""
        for pattern, callbacks in self._change_callbacks.items():
            if pattern == "*" or key.startswith(pattern):
                for callback in callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(key, value)
                        else:
                            callback(key, value)
                    except Exception:
                        pass  # コールバックエラーは無視

    def get_environment(self) -> str:
        """環境名取得"""
        return self.config.environment.environment

    def is_production(self) -> bool:
        """本番環境判定"""
        return self.get_environment() == "production"

    def is_development(self) -> bool:
        """開発環境判定"""
        return self.get_environment() == "development"

    def is_testing(self) -> bool:
        """テスト環境判定"""
        return self.config.environment.testing


# グローバル設定インスタンス
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """グローバル設定マネージャー取得"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager


def initialize_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """設定初期化"""
    global _global_config_manager

    if config_path:
        config = RiskManagementConfig.from_file(config_path)
        _global_config_manager = ConfigManager(config)
    else:
        _global_config_manager = ConfigManager()

    return _global_config_manager
