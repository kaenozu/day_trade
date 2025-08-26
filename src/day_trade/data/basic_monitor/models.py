#!/usr/bin/env python3
"""
Basic Monitor Models
基本監視システムのデータモデル

データクラスと列挙型の定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MonitorStatus(Enum):
    """監視状態"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AlertSeverity(Enum):
    """アラート重要度"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(Enum):
    """アラート種別"""

    DATA_STALE = "data_stale"
    DATA_MISSING = "data_missing"
    INTEGRITY_VIOLATION = "integrity_violation"
    THRESHOLD_BREACH = "threshold_breach"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SLA_VIOLATION = "sla_violation"


class RecoveryAction(Enum):
    """回復アクション"""

    RETRY_FETCH = "retry_fetch"
    USE_FALLBACK = "use_fallback"
    NOTIFY_ADMIN = "notify_admin"
    DISABLE_SOURCE = "disable_source"
    ESCALATE = "escalate"
    AUTO_FIX = "auto_fix"


@dataclass
class MonitorRule:
    """監視ルール定義"""

    rule_id: str
    name: str
    description: str
    data_source: str
    rule_type: str  # "freshness", "consistency", "completeness", "accuracy"
    threshold_value: float
    threshold_unit: str  # "minutes", "hours", "percentage", "count"
    severity: AlertSeverity
    enabled: bool = True
    check_interval_seconds: int = 300  # 5分
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorAlert:
    """監視アラート"""

    alert_id: str
    rule_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data_source: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    recovery_actions_taken: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSourceHealth:
    """データソースヘルス状態"""

    source_id: str
    source_type: str  # "api", "database", "file", "stream"
    last_update: datetime
    data_age_minutes: float
    quality_score: float
    availability: float  # 0-1
    error_rate: float  # 0-1
    response_time_ms: float
    health_status: str  # "healthy", "warning", "critical", "unknown"
    consecutive_failures: int = 0
    last_error: Optional[str] = None


@dataclass
class SLAMetrics:
    """SLA メトリクス"""

    sla_id: str
    name: str
    target_availability: float  # 目標可用性 (99.9%)
    target_freshness_minutes: int  # 目標データ鮮度（分）
    target_quality_score: float  # 目標品質スコア
    current_availability: float
    current_freshness_minutes: float
    current_quality_score: float
    violations_count: int
    measurement_period: str  # "daily", "weekly", "monthly"
    last_violation: Optional[datetime] = None