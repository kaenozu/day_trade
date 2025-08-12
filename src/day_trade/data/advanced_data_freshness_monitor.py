#!/usr/bin/env python3
"""
高度データ鮮度・整合性監視システム
Issue #420: データ管理とデータ品質保証メカニズムの強化

リアルタイム品質監視と自動回復機能:
- マルチソースデータ鮮度監視
- 整合性チェック・検証
- SLA管理・追跡
- 自動回復・フェイルオーバー
- 予測的品質劣化検出
- 包括的アラート・通知
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from ..monitoring.advanced_anomaly_detection_alerts import (
        AdvancedAnomalyAlertSystem,
    )
    from ..monitoring.structured_logging_enhancement import (
        StructuredLoggingEnhancementSystem,
    )
    from ..utils.unified_cache_manager import UnifiedCacheManager
    from .comprehensive_data_quality_system import ComprehensiveDataQualitySystem
    from .data_freshness_monitor import DataFreshnessMonitor, FreshnessStatus

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

    # Fallback definitions
    class FreshnessStatus(Enum):
        FRESH = "fresh"
        STALE = "stale"
        EXPIRED = "expired"


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


class AdvancedDataFreshnessMonitor:
    """高度データ鮮度・整合性監視システム"""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # 設定管理
        self.data_sources: Dict[str, DataSourceConfig] = {}
        self.monitoring_active = False
        self.monitor_thread = None

        # データベース
        self.db_path = "data_freshness_monitor.db"
        self._initialize_database()

        # コンポーネント
        self.quality_system = None
        self.anomaly_detector = None
        self.cache_manager = UnifiedCacheManager() if DEPENDENCIES_AVAILABLE else None

        # メトリクス・統計
        self.monitoring_stats = {
            "total_checks_performed": 0,
            "freshness_violations": 0,
            "integrity_violations": 0,
            "recovery_actions_taken": 0,
            "alerts_generated": 0,
            "sla_violations": 0,
        }

        # 監視データ (インメモリ)
        self.recent_checks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts: Dict[str, List[DataAlert]] = defaultdict(list)
        self.sla_metrics: Dict[str, List[SLAMetrics]] = defaultdict(list)

        # イベントコールバック
        self.alert_callbacks: List[Callable] = []
        self.recovery_callbacks: Dict[RecoveryAction, List[Callable]] = defaultdict(list)

        if config_path:
            self.load_config(config_path)

        self._initialize_components()

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # 鮮度チェックテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS freshness_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    last_update DATETIME NOT NULL,
                    age_seconds REAL NOT NULL,
                    status TEXT NOT NULL,
                    quality_score REAL,
                    record_count INTEGER,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # 整合性チェックテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS integrity_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    check_type TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    passed BOOLEAN NOT NULL,
                    issues_found TEXT,
                    metrics TEXT,
                    baseline_comparison TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # SLAメトリクステーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sla_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    period_start DATETIME NOT NULL,
                    period_end DATETIME NOT NULL,
                    availability_percent REAL NOT NULL,
                    average_response_time REAL NOT NULL,
                    error_count INTEGER NOT NULL,
                    total_requests INTEGER NOT NULL,
                    uptime_seconds REAL NOT NULL,
                    downtime_seconds REAL NOT NULL,
                    sla_violations INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # アラートテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_alerts (
                    alert_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metadata TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME,
                    resolution_notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # データソース状態テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_source_states (
                    source_id TEXT PRIMARY KEY,
                    last_check DATETIME,
                    current_status TEXT,
                    consecutive_failures INTEGER DEFAULT 0,
                    last_success DATETIME,
                    recovery_attempts INTEGER DEFAULT 0,
                    metadata TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_freshness_source_time ON freshness_checks(source_id, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_integrity_source_time ON integrity_checks(source_id, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sla_source_period ON sla_metrics(source_id, period_start, period_end)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_source_severity ON data_alerts(source_id, severity, timestamp)"
            )

            conn.commit()

    def _initialize_components(self):
        """コンポーネント初期化"""
        if DEPENDENCIES_AVAILABLE:
            try:
                self.quality_system = ComprehensiveDataQualitySystem()
                self.anomaly_detector = AdvancedAnomalyAlertSystem()
                self.logger.info("監視システムコンポーネント初期化完了")
            except Exception as e:
                self.logger.warning(f"コンポーネント初期化エラー: {e}")

    def add_data_source(self, config: DataSourceConfig):
        """データソース追加"""
        self.data_sources[config.source_id] = config

        # データベースに状態初期化
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO data_source_states
                (source_id, current_status, metadata, updated_at)
                VALUES (?, ?, ?, ?)
            """,
                (
                    config.source_id,
                    "unknown",
                    json.dumps(
                        {
                            "source_type": config.source_type.value,
                            "monitoring_level": config.monitoring_level.value,
                            "sla_target": config.sla_target,
                        },
                        ensure_ascii=False,
                    ),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

        self.logger.info(f"データソース追加: {config.source_id}")

    def remove_data_source(self, source_id: str):
        """データソース削除"""
        if source_id in self.data_sources:
            del self.data_sources[source_id]

            # アクティブアラートをクローズ
            if source_id in self.active_alerts:
                for alert in self.active_alerts[source_id]:
                    alert.resolved = True
                    alert.resolved_at = datetime.now(timezone.utc)
                    alert.resolution_notes = "データソース削除により自動解決"
                del self.active_alerts[source_id]

            self.logger.info(f"データソース削除: {source_id}")

    def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            self.logger.warning("監視は既に実行中です")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("データ鮮度・整合性監視開始")

    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        self.logger.info("データ鮮度・整合性監視停止")

    def _monitoring_loop(self):
        """監視ループ（スレッド実行）"""
        while self.monitoring_active:
            try:
                # 各データソースをチェック
                for source_config in self.data_sources.values():
                    asyncio.run(self._check_data_source(source_config))

                # SLAメトリクス計算（毎時）
                current_time = datetime.now(timezone.utc)
                if current_time.minute == 0:  # 毎時0分
                    asyncio.run(self._calculate_hourly_sla_metrics())

                # 10秒待機
                time.sleep(10)

            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(30)  # エラー時は長めに待機

    async def _check_data_source(self, config: DataSourceConfig):
        """データソースチェック"""
        try:
            source_id = config.source_id
            current_time = datetime.now(timezone.utc)

            # 鮮度チェック
            freshness_result = await self._perform_freshness_check(config)

            # 整合性チェック（レベルに応じて）
            integrity_results = []
            if config.monitoring_level in [
                MonitoringLevel.COMPREHENSIVE,
                MonitoringLevel.CRITICAL,
            ]:
                integrity_results = await self._perform_integrity_checks(config)

            # 結果保存
            await self._save_freshness_check(freshness_result)
            for integrity_result in integrity_results:
                await self._save_integrity_check(integrity_result)

            # インメモリデータ更新
            self.recent_checks[source_id].append(freshness_result)
            self.monitoring_stats["total_checks_performed"] += 1

            # アラート判定
            await self._evaluate_alerts(config, freshness_result, integrity_results)

            # データソース状態更新
            await self._update_source_state(config, freshness_result, integrity_results)

            # 回復アクション判定
            if freshness_result.status != FreshnessStatus.FRESH or any(
                not r.passed for r in integrity_results
            ):
                await self._evaluate_recovery_actions(config, freshness_result, integrity_results)

        except Exception as e:
            self.logger.error(f"データソースチェックエラー ({config.source_id}): {e}")

    async def _perform_freshness_check(self, config: DataSourceConfig) -> FreshnessCheck:
        """鮮度チェック実行"""
        current_time = datetime.now(timezone.utc)

        try:
            # データソースから最新データ取得
            last_update, data_info = await self._get_latest_data_info(config)

            if last_update is None:
                # データ取得失敗
                return FreshnessCheck(
                    source_id=config.source_id,
                    timestamp=current_time,
                    last_update=current_time,
                    age_seconds=float("inf"),
                    status=FreshnessStatus.EXPIRED,
                    metadata={"error": "データ取得失敗"},
                )

            # 経過時間計算
            age_seconds = (current_time - last_update).total_seconds()

            # ステータス判定
            if age_seconds <= config.expected_frequency:
                status = FreshnessStatus.FRESH
            elif age_seconds <= config.freshness_threshold:
                status = FreshnessStatus.STALE
            else:
                status = FreshnessStatus.EXPIRED

            # 品質スコア計算（データが利用可能な場合）
            quality_score = None
            if data_info and self.quality_system:
                try:
                    if isinstance(data_info.get("data"), pd.DataFrame):
                        quality_report = await self.quality_system.process_dataset(
                            data_info["data"], config.source_id
                        )
                        quality_score = quality_report.overall_score
                except Exception as e:
                    self.logger.warning(f"品質スコア計算エラー ({config.source_id}): {e}")

            return FreshnessCheck(
                source_id=config.source_id,
                timestamp=current_time,
                last_update=last_update,
                age_seconds=age_seconds,
                status=status,
                quality_score=quality_score,
                record_count=data_info.get("record_count") if data_info else None,
                metadata=data_info.get("metadata", {}) if data_info else {},
            )

        except Exception as e:
            self.logger.error(f"鮮度チェックエラー ({config.source_id}): {e}")
            return FreshnessCheck(
                source_id=config.source_id,
                timestamp=current_time,
                last_update=current_time,
                age_seconds=float("inf"),
                status=FreshnessStatus.EXPIRED,
                metadata={"error": str(e)},
            )

    async def _get_latest_data_info(
        self, config: DataSourceConfig
    ) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
        """最新データ情報取得"""
        try:
            if config.source_type == DataSourceType.API:
                # API呼び出し（モック）
                # 実際の実装では、実際のAPIを呼び出す
                return datetime.now(timezone.utc) - timedelta(seconds=30), {
                    "record_count": 100,
                    "metadata": {"api_response_time": 0.15},
                }

            elif config.source_type == DataSourceType.DATABASE:
                # データベースクエリ（モック）
                # 実際の実装では、データベースに接続してクエリ実行
                return datetime.now(timezone.utc) - timedelta(seconds=60), {
                    "record_count": 1500,
                    "metadata": {"query_time": 0.25},
                }

            elif config.source_type == DataSourceType.FILE:
                # ファイル更新時刻チェック（モック）
                # 実際の実装では、ファイルシステムをチェック
                return datetime.now(timezone.utc) - timedelta(seconds=120), {
                    "record_count": 500,
                    "metadata": {"file_size": 1024000},
                }

            else:
                # その他のソースタイプ
                return datetime.now(timezone.utc) - timedelta(seconds=45), {
                    "record_count": 200,
                    "metadata": {"source_type": config.source_type.value},
                }

        except Exception as e:
            self.logger.error(f"データ情報取得エラー ({config.source_id}): {e}")
            return None, None

    async def _perform_integrity_checks(self, config: DataSourceConfig) -> List[IntegrityCheck]:
        """整合性チェック実行"""
        checks = []
        current_time = datetime.now(timezone.utc)

        try:
            # データ取得
            _, data_info = await self._get_latest_data_info(config)

            if not data_info:
                return [
                    IntegrityCheck(
                        source_id=config.source_id,
                        check_type="data_availability",
                        timestamp=current_time,
                        passed=False,
                        issues_found=["データが取得できません"],
                        metadata={"check_status": "failed"},
                    )
                ]

            # 1. レコード数チェック
            record_count = data_info.get("record_count", 0)
            record_count_check = IntegrityCheck(
                source_id=config.source_id,
                check_type="record_count",
                timestamp=current_time,
                passed=record_count > 0,
                metrics={"record_count": record_count},
            )

            if record_count == 0:
                record_count_check.issues_found.append("レコード数が0です")

            checks.append(record_count_check)

            # 2. データ品質チェック（品質スコア基準）
            quality_score = data_info.get("quality_score")
            if quality_score is not None:
                quality_check = IntegrityCheck(
                    source_id=config.source_id,
                    check_type="data_quality",
                    timestamp=current_time,
                    passed=quality_score >= config.quality_threshold,
                    metrics={
                        "quality_score": quality_score,
                        "threshold": config.quality_threshold,
                    },
                )

                if quality_score < config.quality_threshold:
                    quality_check.issues_found.append(
                        f"品質スコア {quality_score:.1f} が閾値 {config.quality_threshold} を下回っています"
                    )

                checks.append(quality_check)

            # 3. ベースライン比較（履歴データとの比較）
            baseline_check = await self._perform_baseline_comparison(config, data_info)
            if baseline_check:
                checks.append(baseline_check)

            return checks

        except Exception as e:
            self.logger.error(f"整合性チェックエラー ({config.source_id}): {e}")
            return [
                IntegrityCheck(
                    source_id=config.source_id,
                    check_type="integrity_error",
                    timestamp=current_time,
                    passed=False,
                    issues_found=[f"整合性チェック実行エラー: {str(e)}"],
                )
            ]

    async def _perform_baseline_comparison(
        self, config: DataSourceConfig, current_data: Dict[str, Any]
    ) -> Optional[IntegrityCheck]:
        """ベースライン比較実行"""
        try:
            # 過去のデータを取得してベースラインと比較
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT record_count FROM freshness_checks
                    WHERE source_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC LIMIT 10
                """,
                    (
                        config.source_id,
                        (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
                    ),
                )

                historical_counts = [row[0] for row in cursor.fetchall() if row[0] is not None]

            if len(historical_counts) < 3:
                return None  # データ不足

            current_count = current_data.get("record_count", 0)
            avg_count = np.mean(historical_counts)
            std_count = np.std(historical_counts)

            # 統計的異常判定 (3σ外れ値)
            if std_count > 0:
                z_score = abs(current_count - avg_count) / std_count
                is_anomaly = z_score > 3.0
            else:
                is_anomaly = False

            baseline_check = IntegrityCheck(
                source_id=config.source_id,
                check_type="baseline_comparison",
                timestamp=datetime.now(timezone.utc),
                passed=not is_anomaly,
                metrics={
                    "current_count": current_count,
                    "baseline_avg": avg_count,
                    "baseline_std": std_count,
                    "z_score": z_score if std_count > 0 else 0,
                },
                baseline_comparison={
                    "historical_samples": len(historical_counts),
                    "deviation_percent": (
                        ((current_count - avg_count) / avg_count * 100) if avg_count > 0 else 0
                    ),
                },
            )

            if is_anomaly:
                baseline_check.issues_found.append(
                    f"レコード数 {current_count} がベースライン {avg_count:.0f}±{std_count:.0f} から大きく逸脱しています (Z-score: {z_score:.2f})"
                )

            return baseline_check

        except Exception as e:
            self.logger.error(f"ベースライン比較エラー ({config.source_id}): {e}")
            return None

    async def _evaluate_alerts(
        self,
        config: DataSourceConfig,
        freshness: FreshnessCheck,
        integrity: List[IntegrityCheck],
    ):
        """アラート評価・生成"""
        alerts_to_generate = []

        # 鮮度アラート
        if freshness.status == FreshnessStatus.EXPIRED:
            alert = DataAlert(
                alert_id=f"freshness_{config.source_id}_{int(time.time())}",
                source_id=config.source_id,
                severity=(
                    AlertSeverity.ERROR
                    if freshness.age_seconds > config.freshness_threshold * 2
                    else AlertSeverity.WARNING
                ),
                alert_type="data_freshness",
                message=f"データが古くなっています: {freshness.age_seconds:.0f}秒経過 (閾値: {config.freshness_threshold}秒)",
                timestamp=freshness.timestamp,
                metadata={
                    "age_seconds": freshness.age_seconds,
                    "threshold": config.freshness_threshold,
                    "last_update": freshness.last_update.isoformat(),
                },
            )
            alerts_to_generate.append(alert)
            self.monitoring_stats["freshness_violations"] += 1

        # 整合性アラート
        for check in integrity:
            if not check.passed:
                alert = DataAlert(
                    alert_id=f"integrity_{config.source_id}_{check.check_type}_{int(time.time())}",
                    source_id=config.source_id,
                    severity=(
                        AlertSeverity.ERROR
                        if len(check.issues_found) > 2
                        else AlertSeverity.WARNING
                    ),
                    alert_type="data_integrity",
                    message=f"整合性チェック失敗 ({check.check_type}): {', '.join(check.issues_found)}",
                    timestamp=check.timestamp,
                    metadata={
                        "check_type": check.check_type,
                        "issues": check.issues_found,
                        "metrics": check.metrics,
                    },
                )
                alerts_to_generate.append(alert)
                self.monitoring_stats["integrity_violations"] += 1

        # 品質アラート
        if (
            freshness.quality_score is not None
            and freshness.quality_score < config.quality_threshold
        ):
            alert = DataAlert(
                alert_id=f"quality_{config.source_id}_{int(time.time())}",
                source_id=config.source_id,
                severity=AlertSeverity.WARNING,
                alert_type="data_quality",
                message=f"データ品質が閾値を下回っています: {freshness.quality_score:.1f} < {config.quality_threshold}",
                timestamp=freshness.timestamp,
                metadata={
                    "quality_score": freshness.quality_score,
                    "threshold": config.quality_threshold,
                },
            )
            alerts_to_generate.append(alert)

        # アラート保存・通知
        for alert in alerts_to_generate:
            await self._save_and_notify_alert(alert)
            self.active_alerts[config.source_id].append(alert)
            self.monitoring_stats["alerts_generated"] += 1

    async def _save_and_notify_alert(self, alert: DataAlert):
        """アラート保存・通知"""
        try:
            # データベース保存
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO data_alerts
                    (alert_id, source_id, severity, alert_type, message, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        alert.alert_id,
                        alert.source_id,
                        alert.severity.value,
                        alert.alert_type,
                        alert.message,
                        alert.timestamp.isoformat(),
                        json.dumps(alert.metadata, ensure_ascii=False),
                    ),
                )
                conn.commit()

            # コールバック実行
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    self.logger.error(f"アラートコールバックエラー: {e}")

            self.logger.warning(f"アラート生成: {alert.alert_type} - {alert.message}")

        except Exception as e:
            self.logger.error(f"アラート保存エラー: {e}")

    async def _evaluate_recovery_actions(
        self,
        config: DataSourceConfig,
        freshness: FreshnessCheck,
        integrity: List[IntegrityCheck],
    ):
        """回復アクション評価・実行"""
        if not config.enable_recovery:
            return

        try:
            # 連続失敗回数確認
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT consecutive_failures, recovery_attempts
                    FROM data_source_states WHERE source_id = ?
                """,
                    (config.source_id,),
                )

                result = cursor.fetchone()
                consecutive_failures = result[0] if result else 0
                recovery_attempts = result[1] if result else 0

            # 回復アクション判定
            should_recover = freshness.status != FreshnessStatus.FRESH or any(
                not check.passed for check in integrity
            )

            if should_recover and recovery_attempts < config.max_retry_attempts:
                await self._execute_recovery_action(config)
                self.monitoring_stats["recovery_actions_taken"] += 1

        except Exception as e:
            self.logger.error(f"回復アクション評価エラー ({config.source_id}): {e}")

    async def _execute_recovery_action(self, config: DataSourceConfig):
        """回復アクション実行"""
        try:
            if config.recovery_strategy == RecoveryAction.RETRY:
                # リトライ処理
                self.logger.info(f"データソースリトライ実行: {config.source_id}")

                # 実際の実装では、データソースへの再接続・再取得を実行
                await asyncio.sleep(5)  # 5秒待機してリトライ

            elif config.recovery_strategy == RecoveryAction.FALLBACK:
                # フェイルオーバー処理
                self.logger.info(f"フェイルオーバー実行: {config.source_id}")

                # 実際の実装では、バックアップソースに切り替え
                pass

            # 回復試行回数更新
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE data_source_states
                    SET recovery_attempts = recovery_attempts + 1, updated_at = ?
                    WHERE source_id = ?
                """,
                    (datetime.now(timezone.utc).isoformat(), config.source_id),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"回復アクション実行エラー ({config.source_id}): {e}")

    async def _update_source_state(
        self,
        config: DataSourceConfig,
        freshness: FreshnessCheck,
        integrity: List[IntegrityCheck],
    ):
        """データソース状態更新"""
        try:
            current_time = datetime.now(timezone.utc)

            # 総合ステータス判定
            is_healthy = freshness.status == FreshnessStatus.FRESH and all(
                check.passed for check in integrity
            )

            current_status = "healthy" if is_healthy else "unhealthy"

            with sqlite3.connect(self.db_path) as conn:
                # 現在の状態取得
                cursor = conn.execute(
                    """
                    SELECT consecutive_failures FROM data_source_states WHERE source_id = ?
                """,
                    (config.source_id,),
                )

                result = cursor.fetchone()
                consecutive_failures = result[0] if result else 0

                # 失敗カウント更新
                if is_healthy:
                    consecutive_failures = 0
                    last_success = current_time
                else:
                    consecutive_failures += 1
                    last_success = None

                # 状態更新
                conn.execute(
                    """
                    UPDATE data_source_states
                    SET last_check = ?, current_status = ?, consecutive_failures = ?,
                        last_success = COALESCE(?, last_success), updated_at = ?
                    WHERE source_id = ?
                """,
                    (
                        current_time.isoformat(),
                        current_status,
                        consecutive_failures,
                        last_success.isoformat() if last_success else None,
                        current_time.isoformat(),
                        config.source_id,
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"データソース状態更新エラー ({config.source_id}): {e}")

    async def _save_freshness_check(self, check: FreshnessCheck):
        """鮮度チェック結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO freshness_checks
                    (source_id, timestamp, last_update, age_seconds, status,
                     quality_score, record_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        check.source_id,
                        check.timestamp.isoformat(),
                        check.last_update.isoformat(),
                        check.age_seconds,
                        check.status.value,
                        check.quality_score,
                        check.record_count,
                        json.dumps(check.metadata, ensure_ascii=False),
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"鮮度チェック保存エラー: {e}")

    async def _save_integrity_check(self, check: IntegrityCheck):
        """整合性チェック結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO integrity_checks
                    (source_id, check_type, timestamp, passed, issues_found,
                     metrics, baseline_comparison)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        check.source_id,
                        check.check_type,
                        check.timestamp.isoformat(),
                        check.passed,
                        json.dumps(check.issues_found, ensure_ascii=False),
                        json.dumps(check.metrics, ensure_ascii=False),
                        (
                            json.dumps(check.baseline_comparison, ensure_ascii=False)
                            if check.baseline_comparison
                            else None
                        ),
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"整合性チェック保存エラー: {e}")

    async def _calculate_hourly_sla_metrics(self):
        """時間別SLAメトリクス計算"""
        try:
            current_time = datetime.now(timezone.utc)
            period_start = current_time.replace(minute=0, second=0, microsecond=0) - timedelta(
                hours=1
            )
            period_end = current_time.replace(minute=0, second=0, microsecond=0)

            for source_id in self.data_sources.keys():
                with sqlite3.connect(self.db_path) as conn:
                    # 該当期間の鮮度チェック結果取得
                    cursor = conn.execute(
                        """
                        SELECT status, age_seconds FROM freshness_checks
                        WHERE source_id = ? AND timestamp BETWEEN ? AND ?
                    """,
                        (source_id, period_start.isoformat(), period_end.isoformat()),
                    )

                    checks = cursor.fetchall()

                    if not checks:
                        continue

                    # SLAメトリクス計算
                    total_checks = len(checks)
                    fresh_checks = sum(
                        1 for status, _ in checks if status == FreshnessStatus.FRESH.value
                    )

                    availability_percent = (fresh_checks / total_checks) * 100
                    average_response_time = np.mean(
                        [age for _, age in checks if age != float("inf")]
                    )
                    error_count = sum(
                        1 for status, _ in checks if status == FreshnessStatus.EXPIRED.value
                    )

                    uptime_seconds = fresh_checks / total_checks * 3600  # 1時間 = 3600秒
                    downtime_seconds = 3600 - uptime_seconds

                    sla_target = self.data_sources[source_id].sla_target
                    sla_violations = 1 if availability_percent < sla_target else 0

                    if sla_violations > 0:
                        self.monitoring_stats["sla_violations"] += 1

                    # SLAメトリクス保存
                    sla_metrics = SLAMetrics(
                        source_id=source_id,
                        period_start=period_start,
                        period_end=period_end,
                        availability_percent=availability_percent,
                        average_response_time=average_response_time,
                        error_count=error_count,
                        total_requests=total_checks,
                        uptime_seconds=uptime_seconds,
                        downtime_seconds=downtime_seconds,
                        sla_violations=sla_violations,
                    )

                    await self._save_sla_metrics(sla_metrics)
                    self.sla_metrics[source_id].append(sla_metrics)

        except Exception as e:
            self.logger.error(f"SLAメトリクス計算エラー: {e}")

    async def _save_sla_metrics(self, metrics: SLAMetrics):
        """SLAメトリクス保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO sla_metrics
                    (source_id, period_start, period_end, availability_percent,
                     average_response_time, error_count, total_requests,
                     uptime_seconds, downtime_seconds, sla_violations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        metrics.source_id,
                        metrics.period_start.isoformat(),
                        metrics.period_end.isoformat(),
                        metrics.availability_percent,
                        metrics.average_response_time,
                        metrics.error_count,
                        metrics.total_requests,
                        metrics.uptime_seconds,
                        metrics.downtime_seconds,
                        metrics.sla_violations,
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"SLAメトリクス保存エラー: {e}")

    def add_alert_callback(self, callback: Callable):
        """アラートコールバック追加"""
        self.alert_callbacks.append(callback)

    def add_recovery_callback(self, action: RecoveryAction, callback: Callable):
        """回復アクションコールバック追加"""
        self.recovery_callbacks[action].append(callback)

    async def get_monitoring_dashboard(self, hours: int = 24) -> Dict[str, Any]:
        """監視ダッシュボードデータ取得"""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            dashboard_data = {
                "overview": {
                    "total_sources": len(self.data_sources),
                    "active_monitoring": self.monitoring_active,
                    "monitoring_stats": self.monitoring_stats,
                    "time_range_hours": hours,
                },
                "source_summary": [],
                "recent_alerts": [],
                "sla_summary": [],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            # 各データソースのサマリー
            for source_id, config in self.data_sources.items():
                with sqlite3.connect(self.db_path) as conn:
                    # 最新ステータス
                    cursor = conn.execute(
                        """
                        SELECT current_status, consecutive_failures, last_success
                        FROM data_source_states WHERE source_id = ?
                    """,
                        (source_id,),
                    )

                    state = cursor.fetchone()

                    # 最新鮮度チェック
                    cursor = conn.execute(
                        """
                        SELECT status, age_seconds, quality_score, timestamp
                        FROM freshness_checks WHERE source_id = ?
                        ORDER BY timestamp DESC LIMIT 1
                    """,
                        (source_id,),
                    )

                    latest_check = cursor.fetchone()

                    # 期間内のアラート数
                    cursor = conn.execute(
                        """
                        SELECT COUNT(*) FROM data_alerts
                        WHERE source_id = ? AND timestamp >= ? AND NOT resolved
                    """,
                        (source_id, start_time.isoformat()),
                    )

                    active_alerts = cursor.fetchone()[0]

                    source_summary = {
                        "source_id": source_id,
                        "source_type": config.source_type.value,
                        "monitoring_level": config.monitoring_level.value,
                        "current_status": state[0] if state else "unknown",
                        "consecutive_failures": state[1] if state else 0,
                        "last_success": state[2] if state else None,
                        "active_alerts": active_alerts,
                    }

                    if latest_check:
                        source_summary.update(
                            {
                                "latest_freshness_status": latest_check[0],
                                "latest_age_seconds": latest_check[1],
                                "latest_quality_score": latest_check[2],
                                "last_check": latest_check[3],
                            }
                        )

                    dashboard_data["source_summary"].append(source_summary)

            # 最新アラート
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT alert_id, source_id, severity, alert_type, message, timestamp
                    FROM data_alerts
                    WHERE timestamp >= ? AND NOT resolved
                    ORDER BY timestamp DESC LIMIT 20
                """,
                    (start_time.isoformat(),),
                )

                for row in cursor.fetchall():
                    dashboard_data["recent_alerts"].append(
                        {
                            "alert_id": row[0],
                            "source_id": row[1],
                            "severity": row[2],
                            "alert_type": row[3],
                            "message": row[4],
                            "timestamp": row[5],
                        }
                    )

            # SLAサマリー
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT source_id, AVG(availability_percent), AVG(average_response_time),
                           SUM(sla_violations) as total_violations
                    FROM sla_metrics
                    WHERE period_start >= ?
                    GROUP BY source_id
                """,
                    (start_time.isoformat(),),
                )

                for row in cursor.fetchall():
                    dashboard_data["sla_summary"].append(
                        {
                            "source_id": row[0],
                            "average_availability": round(row[1], 2) if row[1] else None,
                            "average_response_time": round(row[2], 3) if row[2] else None,
                            "sla_violations": row[3] or 0,
                        }
                    )

            return dashboard_data

        except Exception as e:
            self.logger.error(f"ダッシュボードデータ取得エラー: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    async def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """アラート解決"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE data_alerts
                    SET resolved = TRUE, resolved_at = ?, resolution_notes = ?
                    WHERE alert_id = ?
                """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        resolution_notes,
                        alert_id,
                    ),
                )
                conn.commit()

            # アクティブアラートから削除
            for source_alerts in self.active_alerts.values():
                for alert in source_alerts[:]:
                    if alert.alert_id == alert_id:
                        alert.resolved = True
                        alert.resolved_at = datetime.now(timezone.utc)
                        alert.resolution_notes = resolution_notes
                        source_alerts.remove(alert)
                        break

            self.logger.info(f"アラート解決: {alert_id}")

        except Exception as e:
            self.logger.error(f"アラート解決エラー: {e}")

    def load_config(self, config_path: str):
        """設定ファイル読み込み"""
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)

            for source_data in config_data.get("data_sources", []):
                config = DataSourceConfig(
                    source_id=source_data["source_id"],
                    source_type=DataSourceType(source_data["source_type"]),
                    endpoint_url=source_data.get("endpoint_url"),
                    connection_params=source_data.get("connection_params", {}),
                    expected_frequency=source_data.get("expected_frequency", 60),
                    freshness_threshold=source_data.get("freshness_threshold", 300),
                    quality_threshold=source_data.get("quality_threshold", 80.0),
                    monitoring_level=MonitoringLevel(
                        source_data.get("monitoring_level", "standard")
                    ),
                    enable_recovery=source_data.get("enable_recovery", True),
                    recovery_strategy=RecoveryAction(source_data.get("recovery_strategy", "retry")),
                    max_retry_attempts=source_data.get("max_retry_attempts", 3),
                    sla_target=source_data.get("sla_target", 99.9),
                )
                self.add_data_source(config)

            self.logger.info(f"設定読み込み完了: {len(self.data_sources)}個のデータソース")

        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {e}")


# Factory function
def create_advanced_freshness_monitor(
    config_path: Optional[str] = None,
) -> AdvancedDataFreshnessMonitor:
    """高度データ鮮度監視システム作成"""
    return AdvancedDataFreshnessMonitor(config_path)


if __name__ == "__main__":
    # テスト実行
    async def test_advanced_freshness_monitor():
        print("=== 高度データ鮮度・整合性監視システムテスト ===")

        try:
            # システム初期化
            monitor = create_advanced_freshness_monitor()

            print("\n1. 監視システム初期化完了")

            # テスト用データソース追加
            test_sources = [
                DataSourceConfig(
                    source_id="stock_api",
                    source_type=DataSourceType.API,
                    endpoint_url="https://api.example.com/stocks",
                    expected_frequency=30,
                    freshness_threshold=120,
                    quality_threshold=85.0,
                    monitoring_level=MonitoringLevel.COMPREHENSIVE,
                    sla_target=99.5,
                ),
                DataSourceConfig(
                    source_id="market_db",
                    source_type=DataSourceType.DATABASE,
                    expected_frequency=60,
                    freshness_threshold=300,
                    quality_threshold=90.0,
                    monitoring_level=MonitoringLevel.STANDARD,
                    sla_target=99.9,
                ),
                DataSourceConfig(
                    source_id="price_feed",
                    source_type=DataSourceType.STREAM,
                    expected_frequency=10,
                    freshness_threshold=60,
                    quality_threshold=95.0,
                    monitoring_level=MonitoringLevel.CRITICAL,
                    sla_target=99.95,
                ),
            ]

            for source_config in test_sources:
                monitor.add_data_source(source_config)

            print(f"\n2. テストデータソース追加: {len(test_sources)}個")

            # アラートコールバック追加
            async def alert_handler(alert: DataAlert):
                print(f"   アラート受信: {alert.severity.value} - {alert.message}")

            monitor.add_alert_callback(alert_handler)

            # 手動チェック実行
            print("\n3. 手動監視チェック実行...")

            for source_config in test_sources:
                freshness_check = await monitor._perform_freshness_check(source_config)
                print(
                    f"   {source_config.source_id}: {freshness_check.status.value} (経過: {freshness_check.age_seconds:.1f}秒)"
                )

                if source_config.monitoring_level in [
                    MonitoringLevel.COMPREHENSIVE,
                    MonitoringLevel.CRITICAL,
                ]:
                    integrity_checks = await monitor._perform_integrity_checks(source_config)
                    passed_checks = sum(1 for check in integrity_checks if check.passed)
                    print(f"     整合性チェック: {passed_checks}/{len(integrity_checks)} 合格")

            # 監視開始
            print("\n4. 継続監視開始...")
            monitor.start_monitoring()

            # 数秒間監視実行
            await asyncio.sleep(15)

            print("\n5. 監視統計:")
            stats = monitor.monitoring_stats
            for key, value in stats.items():
                print(f"   {key}: {value}")

            # ダッシュボードデータ取得
            print("\n6. 監視ダッシュボード取得...")
            dashboard = await monitor.get_monitoring_dashboard(hours=1)

            overview = dashboard["overview"]
            print(f"   総データソース数: {overview['total_sources']}")
            print(f"   監視アクティブ: {overview['active_monitoring']}")
            print(f"   アクティブアラート数: {len(dashboard['recent_alerts'])}")

            source_summary = dashboard["source_summary"]
            for source in source_summary:
                print(
                    f"   {source['source_id']}: {source['current_status']} (失敗: {source['consecutive_failures']}回)"
                )
                if source.get("latest_freshness_status"):
                    print(f"     最新鮮度: {source['latest_freshness_status']}")

            # 監視停止
            print("\n7. 監視停止...")
            monitor.stop_monitoring()

            print("\n[成功] 高度データ鮮度・整合性監視システムテスト完了")

        except Exception as e:
            print(f"[エラー] テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_advanced_freshness_monitor())
