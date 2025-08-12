#!/usr/bin/env python3
"""
本番運用監視システム設定管理
Issue #436: 本番運用監視システム完成

環境別設定、SLO定義、アラート設定の管理
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


class Environment(Enum):
    """環境定義"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """ログレベル"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """データベース設定"""
    host: str = "localhost"
    port: int = 5432
    database: str = "day_trade_monitoring"
    username: str = "monitoring_user"
    password: str = "monitoring_password"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class RedisConfig:
    """Redis設定"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: int = 30
    socket_connect_timeout: int = 30


@dataclass
class MetricsConfig:
    """メトリクス設定"""
    collection_interval_seconds: int = 30
    retention_days: int = 30
    max_points_per_metric: int = 10000
    enable_system_metrics: bool = True
    enable_application_metrics: bool = True
    enable_business_metrics: bool = True

    # Prometheus設定
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"


@dataclass
class TracingConfig:
    """トレーシング設定"""
    enabled: bool = True
    sample_rate: float = 1.0  # 100%サンプリング
    max_trace_duration_seconds: int = 300
    max_spans_per_trace: int = 100

    # Jaeger設定
    jaeger_enabled: bool = False
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    jaeger_collector_endpoint: str = "http://localhost:14268/api/traces"


@dataclass
class LoggingConfig:
    """ログ設定"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size_mb: int = 100
    backup_count: int = 5
    rotation_interval: str = "daily"

    # 構造化ログ設定
    structured_logging_enabled: bool = True
    log_to_console: bool = True
    log_to_file: bool = True
    log_directory: str = "logs"

    # ELK Stack設定
    elasticsearch_enabled: bool = False
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_index: str = "day-trade-logs"


@dataclass
class AlertConfig:
    """アラート設定"""
    enabled: bool = True
    default_cooldown_seconds: int = 300
    escalation_timeout_minutes: int = 30
    max_alerts_per_hour: int = 100

    # 通知チャンネル設定
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = field(default_factory=list)

    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"

    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_timeout_seconds: int = 10


@dataclass
class SLOConfigItem:
    """SLO設定項目"""
    name: str
    description: str
    target_percentage: float
    time_window_hours: int
    metric_query: str
    error_budget_percentage: float
    enabled: bool = True


@dataclass
class SLOSettings:
    """SLO設定"""
    enabled: bool = True
    check_interval_minutes: int = 5

    # デフォルトSLO定義
    slos: List[SLOConfigItem] = field(default_factory=lambda: [
        SLOConfigItem(
            name="api_latency",
            description="API応答時間 99.9% < 50ms",
            target_percentage=99.9,
            time_window_hours=24,
            metric_query="api_response_time_p99",
            error_budget_percentage=0.1
        ),
        SLOConfigItem(
            name="system_availability",
            description="システム稼働率 99.99%",
            target_percentage=99.99,
            time_window_hours=720,  # 30日
            metric_query="system_uptime",
            error_budget_percentage=0.01
        ),
        SLOConfigItem(
            name="error_rate",
            description="エラー率 < 0.1%",
            target_percentage=99.9,
            time_window_hours=168,  # 7日
            metric_query="error_rate",
            error_budget_percentage=0.1
        ),
        SLOConfigItem(
            name="data_accuracy",
            description="データ精度 99.95%",
            target_percentage=99.95,
            time_window_hours=24,
            metric_query="data_accuracy",
            error_budget_percentage=0.05
        ),
        SLOConfigItem(
            name="hft_latency",
            description="HFT処理レイテンシー 99.9% < 100μs",
            target_percentage=99.9,
            time_window_hours=1,
            metric_query="hft_processing_latency_p99",
            error_budget_percentage=0.1
        )
    ])


@dataclass
class SecurityConfig:
    """セキュリティ設定"""
    enable_authentication: bool = True
    jwt_secret: str = "your-jwt-secret-key"
    jwt_expiration_hours: int = 24

    # API認証
    api_key_required: bool = True
    api_key_header: str = "X-API-Key"

    # TLS設定
    tls_enabled: bool = False
    tls_cert_path: str = ""
    tls_key_path: str = ""

    # レート制限
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 1000


@dataclass
class PerformanceConfig:
    """パフォーマンス設定"""
    # スレッド・プロセス設定
    monitoring_threads: int = 4
    metrics_processing_workers: int = 2
    alert_processing_workers: int = 2

    # メモリ設定
    max_memory_usage_mb: int = 2048
    memory_check_interval_seconds: int = 60

    # キャッシュ設定
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    max_cache_size: int = 10000

    # バッチ処理設定
    metrics_batch_size: int = 1000
    metrics_batch_timeout_seconds: int = 30

    # 異常検知設定
    anomaly_detection_enabled: bool = True
    anomaly_model_retrain_interval_hours: int = 24
    anomaly_detection_sensitivity: float = 0.1


@dataclass
class ProductionMonitoringConfig:
    """本番運用監視システム設定"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # サブ設定
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    slo: SLOSettings = field(default_factory=SLOSettings)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def from_environment(cls, env: Environment = None) -> 'ProductionMonitoringConfig':
        """環境別設定読み込み"""
        if env is None:
            env_str = os.getenv('ENVIRONMENT', 'development').lower()
            env = Environment(env_str)

        config = cls(environment=env)

        # 環境別設定適用
        if env == Environment.DEVELOPMENT:
            config = cls._apply_development_config(config)
        elif env == Environment.STAGING:
            config = cls._apply_staging_config(config)
        elif env == Environment.PRODUCTION:
            config = cls._apply_production_config(config)

        # 環境変数からオーバーライド
        config = cls._apply_environment_variables(config)

        return config

    @classmethod
    def _apply_development_config(cls, config: 'ProductionMonitoringConfig') -> 'ProductionMonitoringConfig':
        """開発環境設定適用"""
        config.debug = True
        config.logging.level = LogLevel.DEBUG
        config.logging.log_to_console = True
        config.logging.structured_logging_enabled = True

        config.metrics.collection_interval_seconds = 10  # 開発時は短間隔
        config.metrics.retention_days = 7  # 短期保持

        config.alerts.enabled = False  # 開発時はアラート無効
        config.slo.check_interval_minutes = 1  # 短間隔チェック

        config.security.enable_authentication = False  # 開発時は認証無効
        config.performance.monitoring_threads = 2  # リソース節約

        return config

    @classmethod
    def _apply_staging_config(cls, config: 'ProductionMonitoringConfig') -> 'ProductionMonitoringConfig':
        """ステージング環境設定適用"""
        config.debug = False
        config.logging.level = LogLevel.INFO

        config.metrics.collection_interval_seconds = 30
        config.metrics.retention_days = 14

        config.alerts.enabled = True
        config.alerts.email_enabled = True  # ステージングでメール通知テスト

        config.security.enable_authentication = True
        config.performance.monitoring_threads = 3

        return config

    @classmethod
    def _apply_production_config(cls, config: 'ProductionMonitoringConfig') -> 'ProductionMonitoringConfig':
        """本番環境設定適用"""
        config.debug = False
        config.logging.level = LogLevel.WARNING
        config.logging.log_to_console = False  # 本番はファイルのみ

        config.metrics.collection_interval_seconds = 30
        config.metrics.retention_days = 90  # 長期保持
        config.metrics.prometheus_enabled = True

        config.tracing.enabled = True
        config.tracing.jaeger_enabled = True

        config.logging.elasticsearch_enabled = True

        config.alerts.enabled = True
        config.alerts.email_enabled = True
        config.alerts.slack_enabled = True
        config.alerts.max_alerts_per_hour = 50  # 本番では制限

        config.slo.enabled = True
        config.slo.check_interval_minutes = 5

        config.security.enable_authentication = True
        config.security.api_key_required = True
        config.security.tls_enabled = True
        config.security.rate_limit_enabled = True

        config.performance.monitoring_threads = 4
        config.performance.max_memory_usage_mb = 4096
        config.performance.anomaly_detection_enabled = True

        return config

    @classmethod
    def _apply_environment_variables(cls, config: 'ProductionMonitoringConfig') -> 'ProductionMonitoringConfig':
        """環境変数からの設定オーバーライド"""

        # データベース設定
        if os.getenv('DB_HOST'):
            config.database.host = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            config.database.port = int(os.getenv('DB_PORT'))
        if os.getenv('DB_NAME'):
            config.database.database = os.getenv('DB_NAME')
        if os.getenv('DB_USER'):
            config.database.username = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            config.database.password = os.getenv('DB_PASSWORD')

        # Redis設定
        if os.getenv('REDIS_HOST'):
            config.redis.host = os.getenv('REDIS_HOST')
        if os.getenv('REDIS_PORT'):
            config.redis.port = int(os.getenv('REDIS_PORT'))
        if os.getenv('REDIS_PASSWORD'):
            config.redis.password = os.getenv('REDIS_PASSWORD')

        # メトリクス設定
        if os.getenv('METRICS_COLLECTION_INTERVAL'):
            config.metrics.collection_interval_seconds = int(os.getenv('METRICS_COLLECTION_INTERVAL'))
        if os.getenv('PROMETHEUS_PORT'):
            config.metrics.prometheus_port = int(os.getenv('PROMETHEUS_PORT'))

        # アラート設定
        if os.getenv('ALERT_EMAIL_SMTP_SERVER'):
            config.alerts.email_smtp_server = os.getenv('ALERT_EMAIL_SMTP_SERVER')
        if os.getenv('ALERT_EMAIL_USERNAME'):
            config.alerts.email_username = os.getenv('ALERT_EMAIL_USERNAME')
        if os.getenv('ALERT_EMAIL_PASSWORD'):
            config.alerts.email_password = os.getenv('ALERT_EMAIL_PASSWORD')
        if os.getenv('ALERT_EMAIL_RECIPIENTS'):
            config.alerts.email_recipients = os.getenv('ALERT_EMAIL_RECIPIENTS').split(',')

        if os.getenv('SLACK_WEBHOOK_URL'):
            config.alerts.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            config.alerts.slack_enabled = True

        # セキュリティ設定
        if os.getenv('JWT_SECRET'):
            config.security.jwt_secret = os.getenv('JWT_SECRET')
        if os.getenv('API_KEY_REQUIRED'):
            config.security.api_key_required = os.getenv('API_KEY_REQUIRED').lower() == 'true'

        # パフォーマンス設定
        if os.getenv('MAX_MEMORY_USAGE_MB'):
            config.performance.max_memory_usage_mb = int(os.getenv('MAX_MEMORY_USAGE_MB'))
        if os.getenv('MONITORING_THREADS'):
            config.performance.monitoring_threads = int(os.getenv('MONITORING_THREADS'))

        return config

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で設定を取得"""
        return {
            'environment': self.environment.value,
            'debug': self.debug,
            'database': {
                'host': self.database.host,
                'port': self.database.port,
                'database': self.database.database,
                'username': self.database.username,
                'pool_size': self.database.pool_size,
            },
            'redis': {
                'host': self.redis.host,
                'port': self.redis.port,
                'database': self.redis.database,
            },
            'metrics': {
                'collection_interval_seconds': self.metrics.collection_interval_seconds,
                'retention_days': self.metrics.retention_days,
                'prometheus_enabled': self.metrics.prometheus_enabled,
                'prometheus_port': self.metrics.prometheus_port,
            },
            'tracing': {
                'enabled': self.tracing.enabled,
                'sample_rate': self.tracing.sample_rate,
                'jaeger_enabled': self.tracing.jaeger_enabled,
            },
            'logging': {
                'level': self.logging.level.value,
                'structured_logging_enabled': self.logging.structured_logging_enabled,
                'elasticsearch_enabled': self.logging.elasticsearch_enabled,
            },
            'alerts': {
                'enabled': self.alerts.enabled,
                'email_enabled': self.alerts.email_enabled,
                'slack_enabled': self.alerts.slack_enabled,
                'max_alerts_per_hour': self.alerts.max_alerts_per_hour,
            },
            'slo': {
                'enabled': self.slo.enabled,
                'check_interval_minutes': self.slo.check_interval_minutes,
                'slos_count': len(self.slo.slos),
            },
            'security': {
                'enable_authentication': self.security.enable_authentication,
                'api_key_required': self.security.api_key_required,
                'tls_enabled': self.security.tls_enabled,
                'rate_limit_enabled': self.security.rate_limit_enabled,
            },
            'performance': {
                'monitoring_threads': self.performance.monitoring_threads,
                'max_memory_usage_mb': self.performance.max_memory_usage_mb,
                'anomaly_detection_enabled': self.performance.anomaly_detection_enabled,
            }
        }

    def save_to_file(self, file_path: str):
        """設定をファイルに保存"""
        config_dict = self.to_dict()

        # 機密情報は除外
        if 'database' in config_dict:
            config_dict['database'].pop('password', None)
        if 'redis' in config_dict:
            config_dict['redis'].pop('password', None)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def validate(self) -> List[str]:
        """設定の妥当性チェック"""
        errors = []

        # 必須設定チェック
        if not self.database.host:
            errors.append("Database host is required")

        if self.alerts.email_enabled:
            if not self.alerts.email_username or not self.alerts.email_password:
                errors.append("Email credentials required when email alerts enabled")
            if not self.alerts.email_recipients:
                errors.append("Email recipients required when email alerts enabled")

        if self.alerts.slack_enabled and not self.alerts.slack_webhook_url:
            errors.append("Slack webhook URL required when Slack alerts enabled")

        if self.security.enable_authentication and not self.security.jwt_secret:
            errors.append("JWT secret required when authentication enabled")

        # 数値範囲チェック
        if self.metrics.collection_interval_seconds < 1:
            errors.append("Metrics collection interval must be >= 1 second")

        if self.performance.monitoring_threads < 1:
            errors.append("Monitoring threads must be >= 1")

        if not (0.0 <= self.tracing.sample_rate <= 1.0):
            errors.append("Tracing sample rate must be between 0.0 and 1.0")

        # SLO設定チェック
        for slo in self.slo.slos:
            if not (0.0 <= slo.target_percentage <= 100.0):
                errors.append(f"SLO {slo.name} target percentage must be between 0 and 100")
            if slo.time_window_hours < 1:
                errors.append(f"SLO {slo.name} time window must be >= 1 hour")

        return errors


def load_config(config_file: str = None, environment: Environment = None) -> ProductionMonitoringConfig:
    """設定読み込み"""
    # 環境別デフォルト設定作成
    config = ProductionMonitoringConfig.from_environment(environment)

    # 設定ファイルが指定された場合は読み込み
    if config_file and Path(config_file).exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            file_config = json.load(f)

        # TODO: ファイル設定をconfigにマージ
        # 実装は必要に応じて追加

    # 設定の妥当性チェック
    validation_errors = config.validate()
    if validation_errors:
        raise ValueError(f"Configuration validation failed: {', '.join(validation_errors)}")

    return config


def create_sample_config_files():
    """サンプル設定ファイル作成"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # 環境別設定ファイル作成
    for env in Environment:
        config = ProductionMonitoringConfig.from_environment(env)
        config_file = config_dir / f"monitoring_{env.value}.json"
        config.save_to_file(str(config_file))
        print(f"Created sample config: {config_file}")


if __name__ == "__main__":
    # サンプル設定ファイル作成
    create_sample_config_files()

    # 設定テスト
    print("\n=== 設定テスト ===")

    for env in Environment:
        print(f"\n{env.value.upper()} Environment:")
        config = ProductionMonitoringConfig.from_environment(env)

        validation_errors = config.validate()
        if validation_errors:
            print(f"  ❌ Validation errors: {validation_errors}")
        else:
            print("  ✅ Configuration valid")

        print(f"  - Debug: {config.debug}")
        print(f"  - Log Level: {config.logging.level.value}")
        print(f"  - Alerts Enabled: {config.alerts.enabled}")
        print(f"  - SLO Monitoring: {config.slo.enabled}")
        print(f"  - Monitoring Threads: {config.performance.monitoring_threads}")
        print(f"  - Authentication: {config.security.enable_authentication}")