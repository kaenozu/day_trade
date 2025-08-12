"""
Day Trade 監視システム統合パッケージ
Issue #370: リアルタイムメトリクス・アラートシステム
Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード
Issue #436: 本番運用監視システム完成 - 24/7安定運用のためのアラート精度向上とダッシュボード完成

Components:
- prometheus_metrics: カスタムメトリクス収集システム
- alert_engine: インテリジェント・アラートエンジン
- notification_system: マルチチャネル通知システム
- anomaly_detection: 機械学習異常検知システム
- performance_optimizer: パフォーマンス最適化・SLA監視
- log_aggregation_system: ELK Stack風ログ集約システム
- metrics_collection_system: Prometheus風メトリクス収集
- performance_dashboard: Grafana風パフォーマンスダッシュボード
- alert_system: 統合アラート・通知システム
- performance_optimization_system: AI駆動最適化推奨システム
- production_monitoring_system: 本番運用監視システム（APM・オブザーバビリティ統合基盤）
"""

from .metrics import (
    get_ai_metrics,
    get_health_metrics,
    get_metrics_collector,
    get_risk_metrics,
    get_trading_metrics,
    start_metrics_server,
)
from .metrics.prometheus_metrics import get_registry

try:
    from .alert_engine import (
        Alert,
        AlertRule,
        AlertSeverity,
        AlertStatus,
        NotificationChannel,
        get_alert_engine,
    )
    from .anomaly_detection import (
        AnomalyDetectionConfig,
        AnomalyDetectionResult,
        get_ml_anomaly_system,
    )
    from .notification_system import (
        NotificationConfig,
        NotificationTemplate,
        get_notification_system,
    )
    from .performance_optimizer import (
        PerformanceMetrics,
        SLAStatus,
        SLATarget,
        get_performance_system,
    )

    REALTIME_MONITORING_AVAILABLE = True

except ImportError:
    # リアルタイム監視コンポーネントが利用できない場合
    REALTIME_MONITORING_AVAILABLE = False

# Issue #417: 統合ログ・監視システム
try:
    from .alert_system import Alert as NewAlert
    from .alert_system import (
        AlertConditionType,
        NotificationType,
        create_default_alert_rules,
    )
    from .alert_system import AlertManager as NewAlertManager
    from .alert_system import AlertRule as NewAlertRule
    from .alert_system import AlertSeverity as NewAlertSeverity
    from .alert_system import AlertStatus as NewAlertStatus
    from .alert_system import get_alert_manager as get_new_alert_manager
    from .log_aggregation_system import (
        LogAggregationSystem,
        LogEntry,
        LogLevel,
        LogSource,
        get_log_aggregation_system,
    )
    from .metrics_collection_system import (
        ApplicationMetricsCollector,
        MetricsCollectionSystem,
        MetricSource,
        MLModelMetricsCollector,
        SystemMetricsCollector,
        get_metrics_system,
        record_http_request,
        record_ml_prediction,
    )
    from .metrics_collection_system import MetricType as NewMetricType
    from .performance_dashboard import (
        ChartType,
        ChartWidget,
        Dashboard,
        DashboardManager,
        DashboardTemplates,
        TimeRange,
        get_dashboard_manager,
    )
    from .performance_optimization_system import (
        ImpactLevel,
        OptimizationManager,
        OptimizationRecommendation,
        OptimizationType,
        Priority,
        get_optimization_manager,
    )

    INTEGRATED_MONITORING_AVAILABLE = True

except ImportError:
    INTEGRATED_MONITORING_AVAILABLE = False

# Issue #436: 本番運用監視システム
try:
    from .production_monitoring_system import (
        ProductionMonitoringSystem,
        DistributedTracer,
        AnomalyDetector,
        SLOManager,
        TraceSpan,
        AlertSeverity,
        MonitoringScope,
        HealthStatus,
        SLOConfig,
        SLOStatus as ProductionSLOStatus,
        MetricPoint as ProductionMetricPoint,
        LogEntry as ProductionLogEntry,
        Alert as ProductionAlert,
    )

    PRODUCTION_MONITORING_AVAILABLE = True

except ImportError:
    PRODUCTION_MONITORING_AVAILABLE = False

__all__ = [
    # 既存監視システム
    "AdvancedMonitoringSystem",
    "AlertLevel",
    "MetricType",
    "MetricsCollector",
    "AlertManager",
    "RuleEngine",
    "MonitoringRule",
    "PrometheusExporter",
    "ApplicationMetricsCollector",
    "LogAnalysisSystem",
    "LogLevel",
    "AnomalyType",
    # 新基本監視
    "get_metrics_collector",
    "get_risk_metrics",
    "get_trading_metrics",
    "get_ai_metrics",
    "get_health_metrics",
    "start_metrics_server",
    "get_registry",
    "REALTIME_MONITORING_AVAILABLE",
    "INTEGRATED_MONITORING_AVAILABLE",
    "PRODUCTION_MONITORING_AVAILABLE",
]

if REALTIME_MONITORING_AVAILABLE:
    __all__.extend(
        [
            # アラートエンジン
            "get_alert_engine",
            "Alert",
            "AlertRule",
            "AlertSeverity",
            "AlertStatus",
            "NotificationChannel",
            # 通知システム
            "get_notification_system",
            "NotificationConfig",
            "NotificationTemplate",
            # 異常検知
            "get_ml_anomaly_system",
            "AnomalyDetectionConfig",
            "AnomalyDetectionResult",
            # パフォーマンス最適化
            "get_performance_system",
            "SLATarget",
            "PerformanceMetrics",
            "SLAStatus",
        ]
    )

if INTEGRATED_MONITORING_AVAILABLE:
    __all__.extend(
        [
            # Issue #417: 統合ログ・監視システム
            # ログ集約システム
            "LogAggregationSystem",
            "LogEntry",
            "LogLevel",
            "LogSource",
            "get_log_aggregation_system",
            # メトリクス収集システム
            "MetricsCollectionSystem",
            "NewMetricType",
            "MetricSource",
            "SystemMetricsCollector",
            "ApplicationMetricsCollector",
            "MLModelMetricsCollector",
            "get_metrics_system",
            "record_http_request",
            "record_ml_prediction",
            # パフォーマンスダッシュボード
            "DashboardManager",
            "Dashboard",
            "ChartWidget",
            "ChartType",
            "TimeRange",
            "DashboardTemplates",
            "get_dashboard_manager",
            # 統合アラートシステム
            "NewAlertManager",
            "NewAlertRule",
            "NewAlert",
            "NewAlertSeverity",
            "NewAlertStatus",
            "AlertConditionType",
            "NotificationType",
            "get_new_alert_manager",
            "create_default_alert_rules",
            # パフォーマンス最適化
            "OptimizationManager",
            "OptimizationRecommendation",
            "OptimizationType",
            "Priority",
            "ImpactLevel",
            "get_optimization_manager",
        ]
    )

if PRODUCTION_MONITORING_AVAILABLE:
    __all__.extend(
        [
            # Issue #436: 本番運用監視システム
            "ProductionMonitoringSystem",
            "DistributedTracer",
            "AnomalyDetector",
            "SLOManager",
            "TraceSpan",
            "AlertSeverity",
            "MonitoringScope",
            "HealthStatus",
            "SLOConfig",
            "ProductionSLOStatus",
            "ProductionMetricPoint",
            "ProductionLogEntry",
            "ProductionAlert",
        ]
    )
