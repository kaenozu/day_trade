#!/usr/bin/env python3
"""
データモデル定義
高度データ鮮度・整合性監視システム用のデータモデル、列挙型、データクラスを定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DataSourceType(Enum):
    """データソース種別"""

    API = "api"
    DATABASE = "database"
    FILE = "file"
    STREAM = "stream"
    EXTERNAL_FEED = "external_feed"


class MonitoringLevel(Enum):
    """監視レベル"""

    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """アラート重要度"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """回復アクション"""

    RETRY = "retry"
    FALLBACK = "fallback"
    MANUAL = "manual"
    DISABLE = "disable"


class FreshnessStatus(Enum):
    """鮮度ステータス"""

    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"


@dataclass
class DataSourceConfig:
    """データソース設定"""

    source_id: str
    source_type: DataSourceType
    endpoint_url: Optional[str] = None
    connection_params: Dict[str, Any] = field(default_factory=dict)
    expected_frequency: int = 60  # 秒
    freshness_threshold: int = 300  # 秒
    quality_threshold: float = 80.0
    monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD
    enable_recovery: bool = True
    recovery_strategy: RecoveryAction = RecoveryAction.RETRY
    max_retry_attempts: int = 3
    sla_target: float = 99.9  # %


@dataclass
class FreshnessCheck:
    """鮮度チェック結果"""

    source_id: str
    timestamp: datetime
    last_update: datetime
    age_seconds: float
    status: FreshnessStatus
    quality_score: Optional[float] = None
    record_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrityCheck:
    """整合性チェック結果"""

    source_id: str
    check_type: str
    timestamp: datetime
    passed: bool
    issues_found: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Optional[Dict[str, Any]] = None


@dataclass
class SLAMetrics:
    """SLAメトリクス"""

    source_id: str
    period_start: datetime
    period_end: datetime
    availability_percent: float
    average_response_time: float
    error_count: int
    total_requests: int
    uptime_seconds: float
    downtime_seconds: float
    sla_violations: int


@dataclass
class DataAlert:
    """データアラート"""

    alert_id: str
    source_id: str
    severity: AlertSeverity
    alert_type: str
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class DataSourceState:
    """データソース状態"""

    source_id: str
    last_check: Optional[datetime] = None
    current_status: str = "unknown"
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None
    recovery_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: Optional[datetime] = None