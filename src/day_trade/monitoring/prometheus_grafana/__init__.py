"""
Prometheus + Grafana統合パッケージ

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

enhanced_prometheus_grafana_integration.pyからの分割モジュールパッケージ。
後方互換性を保つため、元のクラス・関数を再エクスポート。
"""

# 後方互換性のために元のクラス・関数をインポート
from .alert_manager import AlertManager
from .configs import (
    AlertRule,
    AlertSeverity,
    GrafanaConfig,
    GrafanaDashboard,
    GrafanaPanelType,
    MetricDefinition,
    PrometheusConfig,
    PrometheusMetricType,
)
from .dashboard_factory import DashboardFactory
from .grafana_client import GrafanaClient
from .integration import (
    EnhancedPrometheusGrafanaIntegration,
    create_enhanced_prometheus_grafana_integration,
)
from .metrics_exporter import PrometheusMetricsExporter
from .metrics_updater import SystemMetricsUpdater
from .prometheus_client import PrometheusClient

# 元のファイルからのインポートに対する後方互換性
__all__ = [
    # メインクラス
    "EnhancedPrometheusGrafanaIntegration",
    # 設定とデータ型
    "PrometheusConfig",
    "GrafanaConfig",
    "MetricDefinition",
    "AlertRule",
    "GrafanaDashboard",
    # Enums
    "PrometheusMetricType",
    "GrafanaPanelType",
    "AlertSeverity",
    # クライアント
    "PrometheusClient",
    "GrafanaClient",
    "PrometheusMetricsExporter",
    "SystemMetricsUpdater",
    # ファクトリー・管理クラス
    "DashboardFactory",
    "AlertManager",
    # ファクトリー関数
    "create_enhanced_prometheus_grafana_integration",
]