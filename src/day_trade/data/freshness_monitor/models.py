#!/usr/bin/env python3
"""
データ鮮度監視システム用データモデル定義
監視設定とチェック結果のデータクラスを提供します
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import (
    AlertSeverity,
    DataSourceType,
    FreshnessStatus,
    MonitoringLevel,
    RecoveryAction
)


@dataclass
class DataSourceConfig:
    """データソース設定
    
    データソースの監視設定を定義します。
    各データソースの種別、閾値、SLA目標などを管理します。
    """
    
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
    """鮮度チェック結果
    
    データソースの鮮度チェック結果を格納します。
    データの最終更新時刻と経過時間を管理します。
    """
    
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
    """整合性チェック結果
    
    データの整合性チェック結果を格納します。
    品質問題やベースライン比較結果を管理します。
    """
    
    source_id: str
    check_type: str
    timestamp: datetime
    passed: bool
    issues_found: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Optional[Dict[str, Any]] = None


@dataclass
class SLAMetrics:
    """SLAメトリクス
    
    Service Level Agreement（サービスレベル合意）のメトリクスを格納します。
    可用性、応答時間、エラー率などのKPIを管理します。
    """
    
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
    """データアラート
    
    データ品質や鮮度の問題に関するアラートを格納します。
    重要度、メッセージ、解決状況を管理します。
    """
    
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
class MonitoringStats:
    """監視統計情報
    
    監視システム全体の統計情報を格納します。
    チェック回数、違反回数、アクション実行回数などを管理します。
    """
    
    total_checks_performed: int = 0
    freshness_violations: int = 0
    integrity_violations: int = 0
    recovery_actions_taken: int = 0
    alerts_generated: int = 0
    sla_violations: int = 0

    def to_dict(self) -> Dict[str, int]:
        """辞書形式に変換"""
        return {
            "total_checks_performed": self.total_checks_performed,
            "freshness_violations": self.freshness_violations,
            "integrity_violations": self.integrity_violations,
            "recovery_actions_taken": self.recovery_actions_taken,
            "alerts_generated": self.alerts_generated,
            "sla_violations": self.sla_violations,
        }

    def reset(self):
        """統計をリセット"""
        self.total_checks_performed = 0
        self.freshness_violations = 0
        self.integrity_violations = 0
        self.recovery_actions_taken = 0
        self.alerts_generated = 0
        self.sla_violations = 0


@dataclass
class DataSourceState:
    """データソース状態
    
    各データソースの現在の状態を格納します。
    最終チェック時刻、連続失敗回数、回復試行回数などを管理します。
    """
    
    source_id: str
    last_check: Optional[datetime] = None
    current_status: str = "unknown"
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None
    recovery_attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: Optional[datetime] = None


@dataclass
class DashboardData:
    """ダッシュボードデータ
    
    監視ダッシュボード表示用のデータを格納します。
    概要、ソース別サマリー、アラート一覧などを管理します。
    """
    
    overview: Dict[str, Any]
    source_summary: List[Dict[str, Any]]
    recent_alerts: List[Dict[str, Any]]
    sla_summary: List[Dict[str, Any]]
    generated_at: str
    time_range_hours: int = 24
    error: Optional[str] = None