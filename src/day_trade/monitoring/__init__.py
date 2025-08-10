"""
Day Trade 監視システム統合パッケージ
Issue #370: リアルタイムメトリクス・アラートシステム

Components:
- prometheus_metrics: カスタムメトリクス収集システム
- alert_engine: インテリジェント・アラートエンジン
- notification_system: マルチチャネル通知システム
- anomaly_detection: 機械学習異常検知システム
- performance_optimizer: パフォーマンス最適化・SLA監視
"""

from .metrics import (
    get_metrics_collector,
    get_risk_metrics,
    get_trading_metrics,
    get_ai_metrics,
    get_health_metrics,
    start_metrics_server
)

from .metrics.prometheus_metrics import get_registry

try:
    from .alert_engine import (
        get_alert_engine,
        Alert,
        AlertRule,
        AlertSeverity,
        AlertStatus,
        NotificationChannel
    )

    from .notification_system import (
        get_notification_system,
        NotificationConfig,
        NotificationTemplate
    )

    from .anomaly_detection import (
        get_ml_anomaly_system,
        AnomalyDetectionConfig,
        AnomalyDetectionResult
    )

    from .performance_optimizer import (
        get_performance_system,
        SLATarget,
        PerformanceMetrics,
        SLAStatus
    )

    REALTIME_MONITORING_AVAILABLE = True

except ImportError as e:
    # リアルタイム監視コンポーネントが利用できない場合
    REALTIME_MONITORING_AVAILABLE = False

__all__ = [
    # 基本監視
    'get_metrics_collector',
    'get_risk_metrics',
    'get_trading_metrics',
    'get_ai_metrics',
    'get_health_metrics',
    'start_metrics_server',
    'get_registry',
    'REALTIME_MONITORING_AVAILABLE'
]

if REALTIME_MONITORING_AVAILABLE:
    __all__.extend([
        # アラートエンジン
        'get_alert_engine',
        'Alert',
        'AlertRule',
        'AlertSeverity',
        'AlertStatus',
        'NotificationChannel',

        # 通知システム
        'get_notification_system',
        'NotificationConfig',
        'NotificationTemplate',

        # 異常検知
        'get_ml_anomaly_system',
        'AnomalyDetectionConfig',
        'AnomalyDetectionResult',

        # パフォーマンス最適化
        'get_performance_system',
        'SLATarget',
        'PerformanceMetrics',
        'SLAStatus'
    ])
