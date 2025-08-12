"""
セキュリティ監視統合システム
Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

監視システムとセキュリティ監視の統合により、
統一された包括的なモニタリングダッシュボードを提供。
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from .advanced_anomaly_detection_alerts import AdvancedAnomalyAlertSystem
    from .elk_stack_integration import ELKStackIntegration, create_elk_stack_integration
    from .enhanced_prometheus_grafana_integration import PrometheusGrafanaIntegration
    from .log_aggregation_system import (
        LogAggregationSystem,
        LogEntry,
        LogLevel,
        LogSource,
    )
    from .structured_logging_enhancement import StructuredLoggingEnhancementSystem

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

    # Fallback definitions
    class LogAggregationSystem:
        pass

    class LogEntry:
        pass

    class LogLevel:
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"

    class LogSource:
        SECURITY = "security"

    class ELKStackIntegration:
        pass

    class PrometheusGrafanaIntegration:
        pass

    class AdvancedAnomalyAlertSystem:
        pass


try:
    from ..security.access_control_auditor import AccessControlAuditor
    from ..security.data_protection_system import DataProtectionSystem
    from ..security.dependency_vulnerability_manager import (
        DependencyVulnerabilityManager,
    )
    from ..security.secure_coding_enforcer import SecureCodingEnforcer

    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

    # Fallback definitions
    class DependencyVulnerabilityManager:
        def get_recent_scan_results(self):
            return []

    class SecureCodingEnforcer:
        def get_recent_violations(self):
            return []

    class DataProtectionSystem:
        def get_recent_events(self):
            return []

    class AccessControlAuditor:
        def get_recent_violations(self):
            return []


class SecurityEventSeverity(Enum):
    """セキュリティイベント重要度"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventCategory(Enum):
    """セキュリティイベントカテゴリ"""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_PROTECTION = "data_protection"
    VULNERABILITY = "vulnerability"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    COMPLIANCE = "compliance"
    AUDIT_TRAIL = "audit_trail"
    CODE_SECURITY = "code_security"


@dataclass
class SecurityEvent:
    """セキュリティイベント"""

    id: str
    timestamp: datetime
    category: SecurityEventCategory
    severity: SecurityEventSeverity
    title: str
    description: str
    source_component: str
    affected_resources: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    risk_score: float = 0.0
    mitigation_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None


@dataclass
class SecurityMetrics:
    """セキュリティメトリクス"""

    failed_login_attempts: int = 0
    successful_authentications: int = 0
    unauthorized_access_attempts: int = 0
    detected_vulnerabilities: int = 0
    blocked_requests: int = 0
    data_encryption_events: int = 0
    compliance_violations: int = 0
    high_risk_activities: int = 0
    resolved_security_incidents: int = 0
    average_incident_resolution_time: float = 0.0


class SecurityMonitoringCollector:
    """セキュリティ監視データ収集"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.security_systems = {}
        self._initialize_security_systems()

    def _initialize_security_systems(self):
        """セキュリティシステム初期化"""
        if SECURITY_AVAILABLE:
            try:
                self.security_systems["vulnerability_manager"] = (
                    DependencyVulnerabilityManager()
                )
                self.security_systems["coding_enforcer"] = SecureCodingEnforcer()
                self.security_systems["data_protection"] = DataProtectionSystem()
                self.security_systems["access_control"] = AccessControlAuditor()
                self.logger.info("セキュリティシステム初期化完了")
            except Exception as e:
                self.logger.error(f"セキュリティシステム初期化エラー: {e}")
        else:
            self.logger.warning("セキュリティライブラリが利用できません")

    async def collect_vulnerability_events(self) -> List[SecurityEvent]:
        """脆弱性イベント収集"""
        events = []

        if "vulnerability_manager" not in self.security_systems:
            return events

        try:
            vuln_manager = self.security_systems["vulnerability_manager"]

            # 脆弱性スキャン結果取得
            scan_results = vuln_manager.get_recent_scan_results()

            for vulnerability in scan_results:
                severity_mapping = {
                    "LOW": SecurityEventSeverity.LOW,
                    "MEDIUM": SecurityEventSeverity.MEDIUM,
                    "HIGH": SecurityEventSeverity.HIGH,
                    "CRITICAL": SecurityEventSeverity.CRITICAL,
                }

                event = SecurityEvent(
                    id=f"vuln_{vulnerability.get('id', int(time.time()))}",
                    timestamp=datetime.now(timezone.utc),
                    category=SecurityEventCategory.VULNERABILITY,
                    severity=severity_mapping.get(
                        vulnerability.get("severity", "MEDIUM"),
                        SecurityEventSeverity.MEDIUM,
                    ),
                    title=f"脆弱性検出: {vulnerability.get('package_name', 'Unknown')}",
                    description=vulnerability.get(
                        "description", "脆弱性が検出されました"
                    ),
                    source_component="dependency_vulnerability_manager",
                    affected_resources=[vulnerability.get("package_name", "unknown")],
                    risk_score=vulnerability.get("risk_score", 5.0),
                    metadata={
                        "cve_id": vulnerability.get("cve_id"),
                        "affected_version": vulnerability.get("affected_version"),
                        "fixed_version": vulnerability.get("fixed_version"),
                        "package_name": vulnerability.get("package_name"),
                    },
                )
                events.append(event)

        except Exception as e:
            self.logger.error(f"脆弱性イベント収集エラー: {e}")

        return events

    async def collect_access_control_events(self) -> List[SecurityEvent]:
        """アクセス制御イベント収集"""
        events = []

        if "access_control" not in self.security_systems:
            return events

        try:
            access_auditor = self.security_systems["access_control"]

            # アクセス制御違反取得
            violations = access_auditor.get_recent_violations()

            for violation in violations:
                event = SecurityEvent(
                    id=f"access_{violation.get('id', int(time.time()))}",
                    timestamp=violation.get("timestamp", datetime.now(timezone.utc)),
                    category=SecurityEventCategory.AUTHORIZATION,
                    severity=(
                        SecurityEventSeverity.HIGH
                        if violation.get("critical", False)
                        else SecurityEventSeverity.MEDIUM
                    ),
                    title=f"アクセス制御違反: {violation.get('resource', 'Unknown')}",
                    description=violation.get(
                        "description", "不正なアクセス試行が検出されました"
                    ),
                    source_component="access_control_auditor",
                    user_id=violation.get("user_id"),
                    session_id=violation.get("session_id"),
                    ip_address=violation.get("ip_address"),
                    affected_resources=[violation.get("resource", "unknown")],
                    risk_score=8.0 if violation.get("critical", False) else 5.0,
                    metadata={
                        "requested_permission": violation.get("requested_permission"),
                        "actual_permission": violation.get("actual_permission"),
                        "action": violation.get("action"),
                        "resource_type": violation.get("resource_type"),
                    },
                )
                events.append(event)

        except Exception as e:
            self.logger.error(f"アクセス制御イベント収集エラー: {e}")

        return events

    async def collect_data_protection_events(self) -> List[SecurityEvent]:
        """データ保護イベント収集"""
        events = []

        if "data_protection" not in self.security_systems:
            return events

        try:
            data_protection = self.security_systems["data_protection"]

            # データ保護イベント取得
            protection_events = data_protection.get_recent_events()

            for event_data in protection_events:
                severity_map = {
                    "data_leak": SecurityEventSeverity.CRITICAL,
                    "encryption_failure": SecurityEventSeverity.HIGH,
                    "unauthorized_decryption": SecurityEventSeverity.HIGH,
                    "data_access": SecurityEventSeverity.MEDIUM,
                }

                event = SecurityEvent(
                    id=f"data_protection_{event_data.get('id', int(time.time()))}",
                    timestamp=event_data.get("timestamp", datetime.now(timezone.utc)),
                    category=SecurityEventCategory.DATA_PROTECTION,
                    severity=severity_map.get(
                        event_data.get("event_type", "data_access"),
                        SecurityEventSeverity.MEDIUM,
                    ),
                    title=f"データ保護イベント: {event_data.get('event_type', 'Unknown')}",
                    description=event_data.get(
                        "description", "データ保護イベントが発生しました"
                    ),
                    source_component="data_protection_system",
                    user_id=event_data.get("user_id"),
                    affected_resources=[event_data.get("resource", "unknown")],
                    risk_score=event_data.get("risk_score", 5.0),
                    metadata={
                        "data_type": event_data.get("data_type"),
                        "encryption_method": event_data.get("encryption_method"),
                        "data_sensitivity": event_data.get("data_sensitivity"),
                        "access_pattern": event_data.get("access_pattern"),
                    },
                )
                events.append(event)

        except Exception as e:
            self.logger.error(f"データ保護イベント収集エラー: {e}")

        return events

    async def collect_code_security_events(self) -> List[SecurityEvent]:
        """コードセキュリティイベント収集"""
        events = []

        if "coding_enforcer" not in self.security_systems:
            return events

        try:
            coding_enforcer = self.security_systems["coding_enforcer"]

            # コードセキュリティ違反取得
            violations = coding_enforcer.get_recent_violations()

            for violation in violations:
                event = SecurityEvent(
                    id=f"code_security_{violation.get('id', int(time.time()))}",
                    timestamp=violation.get("timestamp", datetime.now(timezone.utc)),
                    category=SecurityEventCategory.CODE_SECURITY,
                    severity=(
                        SecurityEventSeverity.HIGH
                        if violation.get("severity") == "HIGH"
                        else SecurityEventSeverity.MEDIUM
                    ),
                    title=f"コードセキュリティ違反: {violation.get('rule_id', 'Unknown')}",
                    description=violation.get(
                        "description", "セキュアコーディング規則違反が検出されました"
                    ),
                    source_component="secure_coding_enforcer",
                    affected_resources=[violation.get("file_path", "unknown")],
                    risk_score=7.0 if violation.get("severity") == "HIGH" else 4.0,
                    metadata={
                        "rule_id": violation.get("rule_id"),
                        "rule_category": violation.get("rule_category"),
                        "file_path": violation.get("file_path"),
                        "line_number": violation.get("line_number"),
                        "code_snippet": violation.get("code_snippet", "")[
                            :200
                        ],  # 最初の200文字のみ
                    },
                )
                events.append(event)

        except Exception as e:
            self.logger.error(f"コードセキュリティイベント収集エラー: {e}")

        return events

    async def collect_all_security_events(self) -> List[SecurityEvent]:
        """全セキュリティイベント収集"""
        all_events = []

        # 各種セキュリティイベントを並列収集
        event_collectors = [
            self.collect_vulnerability_events(),
            self.collect_access_control_events(),
            self.collect_data_protection_events(),
            self.collect_code_security_events(),
        ]

        try:
            event_results = await asyncio.gather(
                *event_collectors, return_exceptions=True
            )

            for result in event_results:
                if isinstance(result, list):
                    all_events.extend(result)
                elif isinstance(result, Exception):
                    self.logger.error(f"セキュリティイベント収集エラー: {result}")

            # 時系列順にソート
            all_events.sort(key=lambda x: x.timestamp, reverse=True)

            self.logger.info(f"セキュリティイベント収集完了: {len(all_events)}件")
            return all_events

        except Exception as e:
            self.logger.error(f"セキュリティイベント収集エラー: {e}")
            return []

    def calculate_security_metrics(
        self, events: List[SecurityEvent]
    ) -> SecurityMetrics:
        """セキュリティメトリクス計算"""
        metrics = SecurityMetrics()

        for event in events:
            # カテゴリ別統計
            if event.category == SecurityEventCategory.AUTHENTICATION:
                if "failed" in event.title.lower():
                    metrics.failed_login_attempts += 1
                else:
                    metrics.successful_authentications += 1

            elif event.category == SecurityEventCategory.AUTHORIZATION:
                metrics.unauthorized_access_attempts += 1

            elif event.category == SecurityEventCategory.VULNERABILITY:
                metrics.detected_vulnerabilities += 1

            elif event.category == SecurityEventCategory.DATA_PROTECTION:
                metrics.data_encryption_events += 1

            elif event.category == SecurityEventCategory.COMPLIANCE:
                metrics.compliance_violations += 1

            # 重要度別統計
            if event.severity in [
                SecurityEventSeverity.HIGH,
                SecurityEventSeverity.CRITICAL,
            ]:
                metrics.high_risk_activities += 1

            # 解決済みインシデント
            if event.resolved:
                metrics.resolved_security_incidents += 1

                if event.resolved_at:
                    resolution_time = (
                        event.resolved_at - event.timestamp
                    ).total_seconds() / 3600
                    metrics.average_incident_resolution_time += resolution_time

        # 平均解決時間計算
        if metrics.resolved_security_incidents > 0:
            metrics.average_incident_resolution_time /= (
                metrics.resolved_security_incidents
            )

        return metrics


class SecurityMonitoringDashboard:
    """セキュリティ監視ダッシュボード"""

    def __init__(
        self,
        log_aggregation_system: Optional[LogAggregationSystem] = None,
        elk_integration: Optional[ELKStackIntegration] = None,
        prometheus_grafana: Optional[PrometheusGrafanaIntegration] = None,
        anomaly_detection: Optional[AdvancedAnomalyAlertSystem] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.security_collector = SecurityMonitoringCollector()

        # 既存監視システム統合
        self.log_aggregation = log_aggregation_system
        self.elk_integration = elk_integration
        self.prometheus_grafana = prometheus_grafana
        self.anomaly_detection = anomaly_detection

        # データベース初期化
        self.db_path = "security_monitoring.db"
        self._initialize_database()

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_events (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    source_component TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    risk_score REAL DEFAULT 0.0,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME,
                    resolved_by TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    metrics_data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()

    async def collect_and_process_security_data(self):
        """セキュリティデータ収集・処理"""
        try:
            # セキュリティイベント収集
            security_events = (
                await self.security_collector.collect_all_security_events()
            )

            # イベント保存
            await self._store_security_events(security_events)

            # メトリクス計算
            security_metrics = self.security_collector.calculate_security_metrics(
                security_events
            )

            # メトリクス保存
            await self._store_security_metrics(security_metrics)

            # 統合ログ処理
            await self._process_integrated_logs(security_events)

            # 異常検知との連携
            if self.anomaly_detection:
                await self._correlate_with_anomaly_detection(security_events)

            # Prometheus メトリクス更新
            if self.prometheus_grafana:
                await self._update_prometheus_metrics(security_metrics)

            return {
                "events_collected": len(security_events),
                "metrics_calculated": True,
                "integration_success": True,
            }

        except Exception as e:
            self.logger.error(f"セキュリティデータ処理エラー: {e}")
            return {
                "events_collected": 0,
                "metrics_calculated": False,
                "integration_success": False,
                "error": str(e),
            }

    async def _store_security_events(self, events: List[SecurityEvent]):
        """セキュリティイベント保存"""
        with sqlite3.connect(self.db_path) as conn:
            for event in events:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO security_events
                    (id, timestamp, category, severity, title, description,
                     source_component, user_id, session_id, ip_address, user_agent,
                     risk_score, resolved, resolved_at, resolved_by, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        event.id,
                        event.timestamp.isoformat(),
                        event.category.value,
                        event.severity.value,
                        event.title,
                        event.description,
                        event.source_component,
                        event.user_id,
                        event.session_id,
                        event.ip_address,
                        event.user_agent,
                        event.risk_score,
                        event.resolved,
                        event.resolved_at.isoformat() if event.resolved_at else None,
                        event.resolved_by,
                        json.dumps(event.metadata, ensure_ascii=False),
                    ),
                )
            conn.commit()

    async def _store_security_metrics(self, metrics: SecurityMetrics):
        """セキュリティメトリクス保存"""
        with sqlite3.connect(self.db_path) as conn:
            metrics_data = {
                "failed_login_attempts": metrics.failed_login_attempts,
                "successful_authentications": metrics.successful_authentications,
                "unauthorized_access_attempts": metrics.unauthorized_access_attempts,
                "detected_vulnerabilities": metrics.detected_vulnerabilities,
                "blocked_requests": metrics.blocked_requests,
                "data_encryption_events": metrics.data_encryption_events,
                "compliance_violations": metrics.compliance_violations,
                "high_risk_activities": metrics.high_risk_activities,
                "resolved_security_incidents": metrics.resolved_security_incidents,
                "average_incident_resolution_time": metrics.average_incident_resolution_time,
            }

            conn.execute(
                """
                INSERT INTO security_metrics_history (timestamp, metrics_data)
                VALUES (?, ?)
            """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(metrics_data, ensure_ascii=False),
                ),
            )
            conn.commit()

    async def _process_integrated_logs(self, security_events: List[SecurityEvent]):
        """統合ログ処理"""
        if not self.log_aggregation:
            return

        try:
            for event in security_events:
                # セキュリティイベントを構造化ログエントリに変換
                log_entry = LogEntry(
                    id=f"security_{event.id}",
                    timestamp=event.timestamp,
                    level=self._map_severity_to_log_level(event.severity),
                    source=LogSource.SECURITY,
                    component=event.source_component,
                    message=f"[{event.category.value.upper()}] {event.title}",
                    structured_data={
                        "security_event_id": event.id,
                        "category": event.category.value,
                        "severity": event.severity.value,
                        "risk_score": event.risk_score,
                        "affected_resources": event.affected_resources,
                        "user_id": event.user_id,
                        "session_id": event.session_id,
                        "ip_address": event.ip_address,
                        "metadata": event.metadata,
                    },
                    tags=["security", event.category.value, event.severity.value],
                    user_id=event.user_id,
                    session_id=event.session_id,
                )

                # ログ集約システムに送信
                await self.log_aggregation.process_log_entry(log_entry)

                # ELK Stackに送信
                if self.elk_integration:
                    await self.elk_integration.process_log_entry(log_entry)

        except Exception as e:
            self.logger.error(f"統合ログ処理エラー: {e}")

    def _map_severity_to_log_level(self, severity: SecurityEventSeverity) -> LogLevel:
        """セキュリティ重要度をログレベルにマッピング"""
        mapping = {
            SecurityEventSeverity.LOW: LogLevel.INFO,
            SecurityEventSeverity.MEDIUM: LogLevel.WARNING,
            SecurityEventSeverity.HIGH: LogLevel.ERROR,
            SecurityEventSeverity.CRITICAL: LogLevel.CRITICAL,
        }
        return mapping.get(severity, LogLevel.WARNING)

    async def _correlate_with_anomaly_detection(
        self, security_events: List[SecurityEvent]
    ):
        """異常検知との相関分析"""
        if not self.anomaly_detection:
            return

        try:
            # セキュリティイベントをメトリクスとして異常検知システムに送信
            for event in security_events:
                metric_name = f"security_event_{event.category.value}"

                # リスクスコアをメトリクスとして追加
                self.anomaly_detection.add_metric_data_point(
                    metric_name,
                    event.risk_score,
                    {
                        "severity": event.severity.value,
                        "source_component": event.source_component,
                        "user_id": event.user_id or "unknown",
                    },
                )

            # 異常検知実行
            anomalies = await self.anomaly_detection.detect_security_anomalies()

            if anomalies:
                self.logger.warning(f"セキュリティ異常検出: {len(anomalies)}件")

                # 異常をログに記録
                for anomaly in anomalies:
                    anomaly_log = LogEntry(
                        id=f"security_anomaly_{int(time.time())}",
                        timestamp=datetime.now(timezone.utc),
                        level=LogLevel.ERROR,
                        source=LogSource.SECURITY,
                        component="security_anomaly_detector",
                        message=f"セキュリティ異常検出: {anomaly.get('metric_name')}",
                        structured_data=anomaly,
                        tags=["security", "anomaly", "correlation"],
                    )

                    if self.log_aggregation:
                        await self.log_aggregation.process_log_entry(anomaly_log)

        except Exception as e:
            self.logger.error(f"異常検知相関分析エラー: {e}")

    async def _update_prometheus_metrics(self, metrics: SecurityMetrics):
        """Prometheusメトリクス更新"""
        if not self.prometheus_grafana:
            return

        try:
            # セキュリティメトリクスをPrometheusに送信
            prometheus_metrics = {
                "security_failed_login_attempts_total": metrics.failed_login_attempts,
                "security_successful_authentications_total": metrics.successful_authentications,
                "security_unauthorized_access_attempts_total": metrics.unauthorized_access_attempts,
                "security_detected_vulnerabilities_total": metrics.detected_vulnerabilities,
                "security_blocked_requests_total": metrics.blocked_requests,
                "security_data_encryption_events_total": metrics.data_encryption_events,
                "security_compliance_violations_total": metrics.compliance_violations,
                "security_high_risk_activities_total": metrics.high_risk_activities,
                "security_resolved_incidents_total": metrics.resolved_security_incidents,
                "security_average_resolution_time_hours": metrics.average_incident_resolution_time,
            }

            for metric_name, value in prometheus_metrics.items():
                self.prometheus_grafana.update_custom_metric(
                    metric_name, value, {"service": "security_monitoring"}
                )

        except Exception as e:
            self.logger.error(f"Prometheusメトリクス更新エラー: {e}")

    async def get_security_dashboard_data(
        self, time_range_hours: int = 24, include_resolved: bool = False
    ) -> Dict[str, Any]:
        """セキュリティダッシュボードデータ取得"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=time_range_hours)

            with sqlite3.connect(self.db_path) as conn:
                # セキュリティイベント取得
                query = """
                    SELECT * FROM security_events
                    WHERE timestamp >= ?
                """
                params = [start_time.isoformat()]

                if not include_resolved:
                    query += " AND resolved = FALSE"

                query += " ORDER BY timestamp DESC LIMIT 500"

                cursor = conn.execute(query, params)
                events_data = cursor.fetchall()

                # メトリクス履歴取得
                metrics_cursor = conn.execute(
                    """
                    SELECT timestamp, metrics_data FROM security_metrics_history
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC LIMIT 100
                """,
                    [start_time.isoformat()],
                )
                metrics_data = metrics_cursor.fetchall()

            # データ変換
            events = []
            for row in events_data:
                event = {
                    "id": row[0],
                    "timestamp": row[1],
                    "category": row[2],
                    "severity": row[3],
                    "title": row[4],
                    "description": row[5],
                    "source_component": row[6],
                    "user_id": row[7],
                    "session_id": row[8],
                    "ip_address": row[9],
                    "user_agent": row[10],
                    "risk_score": row[11],
                    "resolved": bool(row[12]),
                    "resolved_at": row[13],
                    "resolved_by": row[14],
                    "metadata": json.loads(row[15]) if row[15] else {},
                }
                events.append(event)

            # メトリクス履歴変換
            metrics_history = []
            for timestamp_str, metrics_json in metrics_data:
                metrics_history.append(
                    {"timestamp": timestamp_str, "metrics": json.loads(metrics_json)}
                )

            # 統計計算
            total_events = len(events)
            critical_events = len([e for e in events if e["severity"] == "critical"])
            high_events = len([e for e in events if e["severity"] == "high"])
            unresolved_events = len([e for e in events if not e["resolved"]])

            # カテゴリ別統計
            category_stats = {}
            for event in events:
                category = event["category"]
                if category not in category_stats:
                    category_stats[category] = 0
                category_stats[category] += 1

            return {
                "summary": {
                    "total_events": total_events,
                    "critical_events": critical_events,
                    "high_events": high_events,
                    "unresolved_events": unresolved_events,
                    "time_range_hours": time_range_hours,
                },
                "events": events,
                "metrics_history": metrics_history,
                "category_statistics": category_stats,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"セキュリティダッシュボードデータ取得エラー: {e}")
            return {
                "summary": {
                    "total_events": 0,
                    "critical_events": 0,
                    "high_events": 0,
                    "unresolved_events": 0,
                    "time_range_hours": time_range_hours,
                },
                "events": [],
                "metrics_history": [],
                "category_statistics": {},
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }


class IntegratedMonitoringSystem:
    """統合監視システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 各監視システム初期化
        self.log_aggregation = None
        self.elk_integration = None
        self.prometheus_grafana = None
        self.anomaly_detection = None
        self.security_dashboard = None

        self._initialize_systems()

    def _initialize_systems(self):
        """監視システム初期化"""
        try:
            if MONITORING_AVAILABLE:
                # ELK Stack統合
                self.elk_integration = create_elk_stack_integration()

                # Prometheus + Grafana統合
                self.prometheus_grafana = PrometheusGrafanaIntegration()

                # 異常検知システム
                self.anomaly_detection = AdvancedAnomalyAlertSystem()

                # セキュリティ監視ダッシュボード
                self.security_dashboard = SecurityMonitoringDashboard(
                    log_aggregation_system=self.log_aggregation,
                    elk_integration=self.elk_integration,
                    prometheus_grafana=self.prometheus_grafana,
                    anomaly_detection=self.anomaly_detection,
                )

                self.logger.info("統合監視システム初期化完了")
            else:
                self.logger.warning("監視ライブラリが利用できません")

        except Exception as e:
            self.logger.error(f"統合監視システム初期化エラー: {e}")

    async def start_integrated_monitoring(self):
        """統合監視開始"""
        try:
            self.logger.info("統合監視システム開始")

            # ELK Stack初期化
            if self.elk_integration:
                await self.elk_integration.initialize_elk_stack()

            # 継続的監視ループ開始
            asyncio.create_task(self._continuous_monitoring_loop())

            return True

        except Exception as e:
            self.logger.error(f"統合監視開始エラー: {e}")
            return False

    async def _continuous_monitoring_loop(self):
        """継続的監視ループ"""
        while True:
            try:
                # セキュリティデータ収集・処理
                if self.security_dashboard:
                    result = (
                        await self.security_dashboard.collect_and_process_security_data()
                    )

                    if result["integration_success"]:
                        self.logger.info(
                            f"統合監視サイクル完了: {result['events_collected']}件のセキュリティイベントを処理"
                        )
                    else:
                        self.logger.error(
                            f"統合監視サイクルエラー: {result.get('error', 'Unknown error')}"
                        )

                # 30秒待機
                await asyncio.sleep(30)

            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                await asyncio.sleep(60)  # エラー時は1分待機

    async def get_integrated_dashboard_data(self) -> Dict[str, Any]:
        """統合ダッシュボードデータ取得"""
        try:
            dashboard_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "active",
                "systems": {},
            }

            # セキュリティダッシュボードデータ
            if self.security_dashboard:
                security_data = (
                    await self.security_dashboard.get_security_dashboard_data()
                )
                dashboard_data["systems"]["security"] = security_data

            # ELK Stack健全性
            if self.elk_integration:
                elk_health = await self.elk_integration.get_elk_stack_health()
                dashboard_data["systems"]["elk_stack"] = elk_health

            # Prometheus/Grafana状態
            if self.prometheus_grafana:
                prometheus_status = await self.prometheus_grafana.get_system_health()
                dashboard_data["systems"]["prometheus_grafana"] = prometheus_status

            return dashboard_data

        except Exception as e:
            self.logger.error(f"統合ダッシュボードデータ取得エラー: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error": str(e),
                "systems": {},
            }


# Factory functions
def create_security_monitoring_integration() -> IntegratedMonitoringSystem:
    """セキュリティ監視統合システム作成"""
    return IntegratedMonitoringSystem()


# グローバルインスタンス
_integrated_monitoring_system = None


def get_integrated_monitoring_system() -> IntegratedMonitoringSystem:
    """グローバル統合監視システム取得"""
    global _integrated_monitoring_system
    if _integrated_monitoring_system is None:
        _integrated_monitoring_system = create_security_monitoring_integration()
    return _integrated_monitoring_system


if __name__ == "__main__":
    # テスト実行
    async def test_security_monitoring_integration():
        print("=== セキュリティ監視統合システムテスト ===")

        try:
            # 統合監視システム初期化
            integrated_system = create_security_monitoring_integration()

            print("\n1. 統合監視システム初期化完了")

            # システム健全性チェック
            print("\n2. システム健全性チェック...")
            dashboard_data = await integrated_system.get_integrated_dashboard_data()

            print(f"   統合システム状態: {dashboard_data.get('status')}")
            print(f"   監視システム数: {len(dashboard_data.get('systems', {}))}")

            # セキュリティデータ収集テスト
            if integrated_system.security_dashboard:
                print("\n3. セキュリティデータ収集テスト...")

                collection_result = (
                    await integrated_system.security_dashboard.collect_and_process_security_data()
                )

                print(
                    f"   収集イベント数: {collection_result.get('events_collected', 0)}"
                )
                print(
                    f"   メトリクス計算: {'成功' if collection_result.get('metrics_calculated') else '失敗'}"
                )
                print(
                    f"   統合処理: {'成功' if collection_result.get('integration_success') else '失敗'}"
                )

                # セキュリティダッシュボードデータ取得
                print("\n4. セキュリティダッシュボードデータ取得...")
                security_data = (
                    await integrated_system.security_dashboard.get_security_dashboard_data()
                )

                summary = security_data.get("summary", {})
                print(f"   総イベント数: {summary.get('total_events', 0)}")
                print(f"   重要イベント数: {summary.get('critical_events', 0)}")
                print(f"   未解決イベント数: {summary.get('unresolved_events', 0)}")

                categories = security_data.get("category_statistics", {})
                print(f"   カテゴリ数: {len(categories)}")

            # 監視システム統合状態
            print("\n5. 監視システム統合状態...")
            systems = dashboard_data.get("systems", {})

            for system_name, system_data in systems.items():
                status = "unknown"
                if isinstance(system_data, dict):
                    status = system_data.get(
                        "status", system_data.get("overall_status", "unknown")
                    )

                print(f"   {system_name}: {status}")

            print("\n[成功] セキュリティ監視統合システムテスト完了")

        except Exception as e:
            print(f"[エラー] テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_security_monitoring_integration())
