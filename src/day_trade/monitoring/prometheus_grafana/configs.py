"""
Prometheus + Grafana統合システムの設定とデータ型定義

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

統合監視システムに必要な設定クラスとEnumの定義。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PrometheusMetricType(Enum):
    """Prometheusメトリクスタイプ"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


class GrafanaPanelType(Enum):
    """Grafanaパネルタイプ"""

    GRAPH = "graph"
    SINGLESTAT = "singlestat"
    TABLE = "table"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    BAR_GAUGE = "bargauge"
    STAT = "stat"
    TEXT = "text"
    PIE_CHART = "piechart"
    WORLD_MAP = "worldmap"


class AlertSeverity(Enum):
    """アラート重要度"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PrometheusConfig:
    """Prometheus設定"""

    host: str = "localhost"
    port: int = 9090
    scheme: str = "http"
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30


@dataclass
class GrafanaConfig:
    """Grafana設定"""

    host: str = "localhost"
    port: int = 3000
    scheme: str = "http"
    username: str = "admin"
    password: str = "admin"
    api_key: Optional[str] = None
    timeout: int = 30


@dataclass
class MetricDefinition:
    """メトリクス定義"""

    name: str
    metric_type: PrometheusMetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # Histogram用
    objectives: Optional[Dict[float, float]] = None  # Summary用


@dataclass
class AlertRule:
    """アラートルール"""

    name: str
    query: str
    condition: str  # e.g., "> 0.8", "== 0"
    duration: str = "5m"
    severity: AlertSeverity = AlertSeverity.WARNING
    description: str = ""
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class GrafanaDashboard:
    """Grafanaダッシュボード定義"""

    title: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    time_range: str = "1h"
    refresh_interval: str = "30s"
    panels: List[Dict[str, Any]] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)