#!/usr/bin/env python3
"""
データ品質アラートシステム
Issue #318: 監視・アラートシステム - Phase 3

データ整合性・完全性・鮮度監視・品質異常検知・自動品質レポート
- データ欠損検出
- 外れ値検知
- データ鮮度監視
- スキーマ整合性チェック
- データ品質スコア算出
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class DataQualityIssueType(Enum):
    """データ品質問題タイプ"""

    MISSING_DATA = "missing_data"
    DUPLICATE_DATA = "duplicate_data"
    OUTLIER_VALUES = "outlier_values"
    SCHEMA_VIOLATION = "schema_violation"
    DATA_STALENESS = "data_staleness"
    INCONSISTENT_FORMAT = "inconsistent_format"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    DATA_DRIFT = "data_drift"
    COMPLETENESS_ISSUE = "completeness_issue"


class QualityMetricType(Enum):
    """品質指標タイプ"""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"


class AlertSeverity(Enum):
    """アラート重要度"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class DataQualityRule:
    """データ品質ルール"""

    rule_id: str
    rule_name: str
    quality_metric: QualityMetricType
    issue_type: DataQualityIssueType
    table_name: str
    column_name: Optional[str] = None

    # しきい値設定
    warning_threshold: float = 0.8  # 80%
    critical_threshold: float = 0.6  # 60%
    emergency_threshold: float = 0.4  # 40%

    # ルール設定
    check_function: Optional[Callable] = None
    expected_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    max_age_hours: Optional[int] = None

    # 実行設定
    enabled: bool = True
    check_interval_minutes: int = 30
    sample_percentage: float = 1.0  # 100%サンプリング


@dataclass
class DataQualityIssue:
    """データ品質問題"""

    issue_id: str
    timestamp: datetime
    severity: AlertSeverity
    issue_type: DataQualityIssueType
    quality_metric: QualityMetricType
    table_name: str
    column_name: Optional[str]

    # 問題詳細
    current_score: float
    threshold_score: float
    affected_records: int
    total_records: int

    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class DataQualityScore:
    """データ品質スコア"""

    timestamp: datetime
    table_name: str
    overall_score: float

    # 指標別スコア
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    validity_score: float
    timeliness_score: float
    uniqueness_score: float

    # 統計情報
    total_records: int
    issues_detected: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityConfig:
    """品質監視設定"""

    # 基本設定
    enable_continuous_monitoring: bool = True
    quality_check_interval_minutes: int = 15
    data_profiling_enabled: bool = True
    auto_remediation_enabled: bool = False

    # 品質スコア設定
    completeness_weight: float = 0.25
    accuracy_weight: float = 0.25
    consistency_weight: float = 0.15
    validity_weight: float = 0.15
    timeliness_weight: float = 0.10
    uniqueness_weight: float = 0.10

    # 異常検知設定
    outlier_detection_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 2.0
    data_drift_detection: bool = True
    drift_threshold: float = 0.1

    # アラート設定
    alert_history_retention_days: int = 30
    duplicate_alert_cooldown_minutes: int = 60
    escalation_threshold_minutes: int = 120


class DataQualityAlertSystem:
    """データ品質アラートシステム"""

    def __init__(self, config: Optional[QualityConfig] = None):
        self.config = config or QualityConfig()
        self.quality_rules: Dict[str, DataQualityRule] = {}
        self.quality_history: List[DataQualityScore] = []
        self.issues: List[DataQualityIssue] = []

        # アラートハンドラー
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = {
            severity: [] for severity in AlertSeverity
        }

        # データプロファイリング結果
        self.data_profiles: Dict[str, Dict[str, Any]] = {}

        # ベースライン統計
        self.baseline_statistics: Dict[str, Dict[str, Any]] = {}

        # アラート抑制
        self.recent_alerts: Dict[str, datetime] = {}

        # 監視タスク
        self.monitor_task: Optional[asyncio.Task] = None
        self._is_running = False

        # デフォルトルール設定
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """デフォルト品質ルール設定"""
        default_rules = [
            # 完全性ルール
            DataQualityRule(
                rule_id="completeness_stock_prices",
                rule_name="株価データ完全性",
                quality_metric=QualityMetricType.COMPLETENESS,
                issue_type=DataQualityIssueType.MISSING_DATA,
                table_name="stock_prices",
                warning_threshold=0.95,
                critical_threshold=0.85,
                emergency_threshold=0.7,
            ),
            # 一意性ルール
            DataQualityRule(
                rule_id="uniqueness_stock_symbol_timestamp",
                rule_name="株価データ一意性",
                quality_metric=QualityMetricType.UNIQUENESS,
                issue_type=DataQualityIssueType.DUPLICATE_DATA,
                table_name="stock_prices",
                warning_threshold=0.99,
                critical_threshold=0.95,
                emergency_threshold=0.9,
            ),
            # 妥当性ルール
            DataQualityRule(
                rule_id="validity_price_range",
                rule_name="株価範囲妥当性",
                quality_metric=QualityMetricType.VALIDITY,
                issue_type=DataQualityIssueType.OUTLIER_VALUES,
                table_name="stock_prices",
                column_name="close_price",
                min_value=0.1,
                max_value=100000.0,
                warning_threshold=0.98,
                critical_threshold=0.95,
            ),
            # 適時性ルール
            DataQualityRule(
                rule_id="timeliness_stock_data",
                rule_name="株価データ鮮度",
                quality_metric=QualityMetricType.TIMELINESS,
                issue_type=DataQualityIssueType.DATA_STALENESS,
                table_name="stock_prices",
                max_age_hours=2,
                warning_threshold=0.9,
                critical_threshold=0.7,
            ),
            # 整合性ルール
            DataQualityRule(
                rule_id="consistency_ohlc",
                rule_name="OHLC価格整合性",
                quality_metric=QualityMetricType.CONSISTENCY,
                issue_type=DataQualityIssueType.INCONSISTENT_FORMAT,
                table_name="stock_prices",
                warning_threshold=0.98,
                critical_threshold=0.95,
            ),
            # 精度ルール
            DataQualityRule(
                rule_id="accuracy_market_cap",
                rule_name="時価総額計算精度",
                quality_metric=QualityMetricType.ACCURACY,
                issue_type=DataQualityIssueType.BUSINESS_RULE_VIOLATION,
                table_name="stock_prices",
                warning_threshold=0.95,
                critical_threshold=0.9,
            ),
        ]

        for rule in default_rules:
            self.add_quality_rule(rule)

    def add_quality_rule(self, rule: DataQualityRule) -> None:
        """品質ルール追加"""
        self.quality_rules[rule.rule_id] = rule
        logger.info(f"データ品質ルール追加: {rule.rule_id}")

    def remove_quality_rule(self, rule_id: str) -> None:
        """品質ルール削除"""
        if rule_id in self.quality_rules:
            del self.quality_rules[rule_id]
            logger.info(f"データ品質ルール削除: {rule_id}")

    def register_alert_handler(self, severity: AlertSeverity, handler: Callable) -> None:
        """アラートハンドラー登録"""
        self.alert_handlers[severity].append(handler)
        logger.info(f"品質アラートハンドラー登録: {severity.value}")

    async def start_monitoring(self) -> None:
        """監視開始"""
        if self._is_running:
            logger.warning("データ品質監視は既に実行中です")
            return

        self._is_running = True

        if self.config.enable_continuous_monitoring:
            self.monitor_task = asyncio.create_task(self._monitoring_loop())

        logger.info("データ品質アラート監視開始")

    async def stop_monitoring(self) -> None:
        """監視停止"""
        self._is_running = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("データ品質アラート監視停止")

    async def _monitoring_loop(self) -> None:
        """監視ループ"""
        while self._is_running:
            try:
                # データ品質チェック実行
                await self._execute_quality_checks()

                # データプロファイリング実行
                if self.config.data_profiling_enabled:
                    await self._update_data_profiles()

                # 品質履歴クリーンアップ
                await self._cleanup_quality_history()

                # 監視間隔待機
                await asyncio.sleep(self.config.quality_check_interval_minutes * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"品質監視ループエラー: {e}")
                await asyncio.sleep(60)

    async def _execute_quality_checks(self) -> None:
        """品質チェック実行"""
        for rule_id, rule in self.quality_rules.items():
            if not rule.enabled:
                continue

            try:
                # 最後のチェックから間隔チェック
                if not await self._should_run_rule_check(rule):
                    continue

                # 品質スコア計算
                quality_score = await self._calculate_quality_score(rule)

                if quality_score is not None:
                    # しきい値評価
                    severity = await self._evaluate_quality_threshold(rule, quality_score)

                    if severity:
                        await self._generate_quality_issue(rule, quality_score, severity)

            except Exception as e:
                logger.error(f"品質チェックエラー {rule_id}: {e}")

    async def _should_run_rule_check(self, rule: DataQualityRule) -> bool:
        """ルールチェック実行判定"""
        last_check_key = f"last_check_{rule.rule_id}"

        if last_check_key not in self.recent_alerts:
            return True

        last_check = self.recent_alerts[last_check_key]
        elapsed_minutes = (datetime.now() - last_check).total_seconds() / 60

        return elapsed_minutes >= rule.check_interval_minutes

    async def _calculate_quality_score(self, rule: DataQualityRule) -> Optional[float]:
        """品質スコア計算"""
        try:
            # 模擬的なデータ品質スコア計算
            # 実際の実装では、データベースまたはデータソースから品質を評価

            if rule.quality_metric == QualityMetricType.COMPLETENESS:
                return await self._calculate_completeness_score(rule)
            elif rule.quality_metric == QualityMetricType.UNIQUENESS:
                return await self._calculate_uniqueness_score(rule)
            elif rule.quality_metric == QualityMetricType.VALIDITY:
                return await self._calculate_validity_score(rule)
            elif rule.quality_metric == QualityMetricType.TIMELINESS:
                return await self._calculate_timeliness_score(rule)
            elif rule.quality_metric == QualityMetricType.CONSISTENCY:
                return await self._calculate_consistency_score(rule)
            elif rule.quality_metric == QualityMetricType.ACCURACY:
                return await self._calculate_accuracy_score(rule)
            else:
                return None

        except Exception as e:
            logger.error(f"品質スコア計算エラー {rule.rule_id}: {e}")
            return None

    async def _calculate_completeness_score(self, rule: DataQualityRule) -> float:
        """完全性スコア計算"""
        # 模擬的な完全性計算
        # 実際の実装では、NULL値や欠損データの割合を計算

        # シミュレートされた完全性スコア
        import random

        base_score = 0.95
        variation = random.uniform(-0.1, 0.05)
        score = max(0.0, min(1.0, base_score + variation))

        await asyncio.sleep(0.01)  # DB処理をシミュレート
        return score

    async def _calculate_uniqueness_score(self, rule: DataQualityRule) -> float:
        """一意性スコア計算"""
        # 模擬的な一意性計算
        import random

        base_score = 0.98
        variation = random.uniform(-0.05, 0.02)
        score = max(0.0, min(1.0, base_score + variation))

        await asyncio.sleep(0.01)
        return score

    async def _calculate_validity_score(self, rule: DataQualityRule) -> float:
        """妥当性スコア計算"""
        # 模擬的な妥当性計算（範囲チェックなど）
        import random

        base_score = 0.96
        variation = random.uniform(-0.08, 0.04)
        score = max(0.0, min(1.0, base_score + variation))

        await asyncio.sleep(0.01)
        return score

    async def _calculate_timeliness_score(self, rule: DataQualityRule) -> float:
        """適時性スコア計算"""
        # 模擬的な適時性計算（データ鮮度チェック）
        import random

        base_score = 0.92
        variation = random.uniform(-0.15, 0.08)
        score = max(0.0, min(1.0, base_score + variation))

        await asyncio.sleep(0.01)
        return score

    async def _calculate_consistency_score(self, rule: DataQualityRule) -> float:
        """整合性スコア計算"""
        # 模擬的な整合性計算（データ間の論理整合性）
        import random

        base_score = 0.94
        variation = random.uniform(-0.1, 0.06)
        score = max(0.0, min(1.0, base_score + variation))

        await asyncio.sleep(0.01)
        return score

    async def _calculate_accuracy_score(self, rule: DataQualityRule) -> float:
        """精度スコア計算"""
        # 模擬的な精度計算（ビジネスルール適合性）
        import random

        base_score = 0.93
        variation = random.uniform(-0.12, 0.07)
        score = max(0.0, min(1.0, base_score + variation))

        await asyncio.sleep(0.01)
        return score

    async def _evaluate_quality_threshold(
        self, rule: DataQualityRule, quality_score: float
    ) -> Optional[AlertSeverity]:
        """品質しきい値評価"""
        if quality_score <= rule.emergency_threshold:
            return AlertSeverity.EMERGENCY
        elif quality_score <= rule.critical_threshold:
            return AlertSeverity.CRITICAL
        elif quality_score <= rule.warning_threshold:
            return AlertSeverity.WARNING
        else:
            return None

    async def _generate_quality_issue(
        self, rule: DataQualityRule, quality_score: float, severity: AlertSeverity
    ) -> None:
        """品質問題生成"""

        # アラート抑制チェック
        alert_key = f"{rule.rule_id}_{severity.value}"
        if not await self._should_generate_alert(alert_key):
            return

        # 問題詳細情報生成
        threshold_score = self._get_threshold_score(rule, severity)

        # 模擬的な影響レコード数
        import random

        total_records = random.randint(1000, 10000)
        affected_records = int(total_records * (1.0 - quality_score))

        message = await self._generate_quality_message(
            rule,
            quality_score,
            threshold_score,
            severity,
            affected_records,
            total_records,
        )

        # 問題作成
        issue_id = f"quality_issue_{int(time.time() * 1000)}"
        issue = DataQualityIssue(
            issue_id=issue_id,
            timestamp=datetime.now(),
            severity=severity,
            issue_type=rule.issue_type,
            quality_metric=rule.quality_metric,
            table_name=rule.table_name,
            column_name=rule.column_name,
            current_score=quality_score,
            threshold_score=threshold_score,
            affected_records=affected_records,
            total_records=total_records,
            message=message,
            details={
                "rule_id": rule.rule_id,
                "rule_name": rule.rule_name,
                "check_timestamp": datetime.now().isoformat(),
                "quality_percentage": quality_score * 100,
                "impact_percentage": (
                    (affected_records / total_records) * 100 if total_records > 0 else 0
                ),
            },
        )

        self.issues.append(issue)

        # アラートハンドラー実行
        await self._execute_alert_handlers(issue)

        # アラート抑制記録
        self.recent_alerts[alert_key] = datetime.now()
        self.recent_alerts[f"last_check_{rule.rule_id}"] = datetime.now()

        logger.warning(f"データ品質問題検出: {issue.message}")

    def _get_threshold_score(self, rule: DataQualityRule, severity: AlertSeverity) -> float:
        """しきい値スコア取得"""
        if severity == AlertSeverity.WARNING:
            return rule.warning_threshold
        elif severity == AlertSeverity.CRITICAL:
            return rule.critical_threshold
        elif severity == AlertSeverity.EMERGENCY:
            return rule.emergency_threshold
        return 0.0

    async def _generate_quality_message(
        self,
        rule: DataQualityRule,
        quality_score: float,
        threshold_score: float,
        severity: AlertSeverity,
        affected_records: int,
        total_records: int,
    ) -> str:
        """品質問題メッセージ生成"""

        severity_labels = {
            AlertSeverity.WARNING: "警告",
            AlertSeverity.CRITICAL: "重要",
            AlertSeverity.EMERGENCY: "緊急",
        }

        quality_names = {
            QualityMetricType.COMPLETENESS: "完全性",
            QualityMetricType.UNIQUENESS: "一意性",
            QualityMetricType.VALIDITY: "妥当性",
            QualityMetricType.TIMELINESS: "適時性",
            QualityMetricType.CONSISTENCY: "整合性",
            QualityMetricType.ACCURACY: "精度",
        }

        severity_label = severity_labels.get(severity, severity.value.upper())
        quality_name = quality_names.get(rule.quality_metric, rule.quality_metric.value)

        impact_percentage = (affected_records / total_records) * 100 if total_records > 0 else 0

        message = (
            f"[{severity_label}] {rule.table_name}テーブルの{quality_name}が低下: "
            f"品質スコア{quality_score:.3f} (しきい値: {threshold_score:.3f}), "
            f"影響レコード: {affected_records:,}件 ({impact_percentage:.1f}%)"
        )

        if rule.column_name:
            message += f" [列: {rule.column_name}]"

        return message

    async def _should_generate_alert(self, alert_key: str) -> bool:
        """アラート生成判定"""
        if alert_key in self.recent_alerts:
            last_alert = self.recent_alerts[alert_key]
            elapsed_minutes = (datetime.now() - last_alert).total_seconds() / 60
            if elapsed_minutes < self.config.duplicate_alert_cooldown_minutes:
                return False

        return True

    async def _execute_alert_handlers(self, issue: DataQualityIssue) -> None:
        """アラートハンドラー実行"""
        handlers = self.alert_handlers.get(issue.severity, [])

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(issue)
                else:
                    handler(issue)
            except Exception as e:
                logger.error(f"品質アラートハンドラー実行エラー: {e}")

    async def _update_data_profiles(self) -> None:
        """データプロファイリング更新"""
        # データプロファイリングの模擬実装
        # 実際の実装では、各テーブル・列の統計情報を計算

        tables = ["stock_prices", "market_data", "portfolio"]

        for table_name in tables:
            profile = await self._generate_table_profile(table_name)
            self.data_profiles[table_name] = profile

    async def _generate_table_profile(self, table_name: str) -> Dict[str, Any]:
        """テーブルプロファイル生成"""
        import random

        # 模擬的なプロファイル情報
        await asyncio.sleep(0.1)  # DB処理をシミュレート

        return {
            "table_name": table_name,
            "timestamp": datetime.now().isoformat(),
            "record_count": random.randint(1000, 50000),
            "null_percentage": random.uniform(0.0, 0.1),
            "duplicate_percentage": random.uniform(0.0, 0.05),
            "unique_values": random.randint(500, 10000),
            "data_freshness_hours": random.uniform(0.1, 3.0),
            "average_row_size_bytes": random.randint(50, 500),
            "columns_profiled": random.randint(5, 20),
        }

    async def _cleanup_quality_history(self) -> None:
        """品質履歴クリーンアップ"""
        cutoff_time = datetime.now() - timedelta(days=self.config.alert_history_retention_days)

        # 問題履歴クリーンアップ
        self.issues = [issue for issue in self.issues if issue.timestamp > cutoff_time]

        # 品質履歴クリーンアップ
        self.quality_history = [
            score for score in self.quality_history if score.timestamp > cutoff_time
        ]

        # 最近のアラート記録クリーンアップ
        recent_cutoff = datetime.now() - timedelta(
            minutes=self.config.duplicate_alert_cooldown_minutes * 2
        )
        self.recent_alerts = {
            key: timestamp
            for key, timestamp in self.recent_alerts.items()
            if timestamp > recent_cutoff
        }

    async def calculate_overall_quality_score(self, table_name: str) -> DataQualityScore:
        """総合品質スコア算出"""
        current_time = datetime.now()

        # 各品質指標のスコア計算
        quality_scores = {}

        for metric_type in QualityMetricType:
            # 該当する品質ルールを検索
            relevant_rules = [
                rule
                for rule in self.quality_rules.values()
                if (
                    rule.table_name == table_name
                    and rule.quality_metric == metric_type
                    and rule.enabled
                )
            ]

            if relevant_rules:
                # 最初のルールでスコア計算（実際は全ルールの平均など）
                score = await self._calculate_quality_score(relevant_rules[0])
                quality_scores[metric_type] = score if score is not None else 0.8
            else:
                quality_scores[metric_type] = 0.8  # デフォルトスコア

        # 重み付き総合スコア計算
        overall_score = (
            quality_scores.get(QualityMetricType.COMPLETENESS, 0.8)
            * self.config.completeness_weight
            + quality_scores.get(QualityMetricType.ACCURACY, 0.8) * self.config.accuracy_weight
            + quality_scores.get(QualityMetricType.CONSISTENCY, 0.8)
            * self.config.consistency_weight
            + quality_scores.get(QualityMetricType.VALIDITY, 0.8) * self.config.validity_weight
            + quality_scores.get(QualityMetricType.TIMELINESS, 0.8) * self.config.timeliness_weight
            + quality_scores.get(QualityMetricType.UNIQUENESS, 0.8) * self.config.uniqueness_weight
        )

        # 問題数カウント
        recent_issues = [
            issue
            for issue in self.issues
            if (
                issue.table_name == table_name
                and issue.timestamp > current_time - timedelta(hours=1)
                and not issue.resolved
            )
        ]

        # 総レコード数（模擬）
        import random

        total_records = random.randint(1000, 10000)

        quality_score = DataQualityScore(
            timestamp=current_time,
            table_name=table_name,
            overall_score=overall_score,
            completeness_score=quality_scores.get(QualityMetricType.COMPLETENESS, 0.8),
            accuracy_score=quality_scores.get(QualityMetricType.ACCURACY, 0.8),
            consistency_score=quality_scores.get(QualityMetricType.CONSISTENCY, 0.8),
            validity_score=quality_scores.get(QualityMetricType.VALIDITY, 0.8),
            timeliness_score=quality_scores.get(QualityMetricType.TIMELINESS, 0.8),
            uniqueness_score=quality_scores.get(QualityMetricType.UNIQUENESS, 0.8),
            total_records=total_records,
            issues_detected=len(recent_issues),
            details={
                "calculation_timestamp": current_time.isoformat(),
                "active_rules": len(
                    [
                        r
                        for r in self.quality_rules.values()
                        if r.table_name == table_name and r.enabled
                    ]
                ),
                "weight_configuration": {
                    "completeness_weight": self.config.completeness_weight,
                    "accuracy_weight": self.config.accuracy_weight,
                    "consistency_weight": self.config.consistency_weight,
                    "validity_weight": self.config.validity_weight,
                    "timeliness_weight": self.config.timeliness_weight,
                    "uniqueness_weight": self.config.uniqueness_weight,
                },
            },
        )

        self.quality_history.append(quality_score)

        return quality_score

    async def resolve_issue(self, issue_id: str, resolution_note: Optional[str] = None) -> bool:
        """品質問題解決"""
        for issue in self.issues:
            if issue.issue_id == issue_id and not issue.resolved:
                issue.resolved = True
                issue.resolved_at = datetime.now()
                if resolution_note:
                    issue.details["resolution_note"] = resolution_note

                logger.info(f"データ品質問題解決: {issue_id}")
                return True

        return False

    async def get_active_issues(
        self, table_name: Optional[str] = None, severity: Optional[AlertSeverity] = None
    ) -> List[DataQualityIssue]:
        """アクティブな品質問題取得"""
        active_issues = [issue for issue in self.issues if not issue.resolved]

        if table_name:
            active_issues = [issue for issue in active_issues if issue.table_name == table_name]

        if severity:
            active_issues = [issue for issue in active_issues if issue.severity == severity]

        return sorted(active_issues, key=lambda i: i.timestamp, reverse=True)

    async def get_quality_summary(self) -> Dict[str, Any]:
        """品質概要取得"""
        active_issues = await self.get_active_issues()

        # 重要度別問題数
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len(
                [issue for issue in active_issues if issue.severity == severity]
            )

        # テーブル別問題数
        table_issues = {}
        for issue in active_issues:
            table_issues[issue.table_name] = table_issues.get(issue.table_name, 0) + 1

        # 品質指標別問題数
        metric_issues = {}
        for issue in active_issues:
            metric_issues[issue.quality_metric.value] = (
                metric_issues.get(issue.quality_metric.value, 0) + 1
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_active_issues": len(active_issues),
            "severity_breakdown": severity_counts,
            "table_breakdown": table_issues,
            "quality_metric_breakdown": metric_issues,
            "total_issues_generated": len(self.issues),
            "resolved_issues": len([issue for issue in self.issues if issue.resolved]),
            "quality_rules": {
                "total_rules": len(self.quality_rules),
                "enabled_rules": len(
                    [rule for rule in self.quality_rules.values() if rule.enabled]
                ),
                "disabled_rules": len(
                    [rule for rule in self.quality_rules.values() if not rule.enabled]
                ),
            },
            "monitoring_status": {
                "is_running": self._is_running,
                "continuous_monitoring": self.config.enable_continuous_monitoring,
                "data_profiling_enabled": self.config.data_profiling_enabled,
            },
        }

    async def get_table_quality_report(self, table_name: str) -> Dict[str, Any]:
        """テーブル品質レポート取得"""
        # 最新の品質スコア計算
        quality_score = await self.calculate_overall_quality_score(table_name)

        # テーブル関連の問題取得
        table_issues = await self.get_active_issues(table_name=table_name)

        # データプロファイル取得
        data_profile = self.data_profiles.get(table_name, {})

        # 品質トレンド分析
        quality_trend = await self._analyze_quality_trend(table_name)

        return {
            "table_name": table_name,
            "report_timestamp": datetime.now().isoformat(),
            "overall_quality_score": quality_score.overall_score,
            "quality_breakdown": {
                "completeness": quality_score.completeness_score,
                "accuracy": quality_score.accuracy_score,
                "consistency": quality_score.consistency_score,
                "validity": quality_score.validity_score,
                "timeliness": quality_score.timeliness_score,
                "uniqueness": quality_score.uniqueness_score,
            },
            "active_issues": len(table_issues),
            "critical_issues": len(
                [
                    i
                    for i in table_issues
                    if i.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
                ]
            ),
            "data_profile": data_profile,
            "quality_trend": quality_trend,
            "recommendations": await self._generate_quality_recommendations(
                table_name, table_issues, quality_score
            ),
        }

    async def _analyze_quality_trend(self, table_name: str) -> Dict[str, Any]:
        """品質トレンド分析"""
        # 過去24時間の品質スコア履歴
        cutoff_time = datetime.now() - timedelta(hours=24)
        table_history = [
            score
            for score in self.quality_history
            if score.table_name == table_name and score.timestamp > cutoff_time
        ]

        if len(table_history) < 2:
            return {"trend": "insufficient_data", "data_points": len(table_history)}

        # トレンド計算
        scores = [score.overall_score for score in table_history]
        x = np.arange(len(scores))
        y = np.array(scores)

        correlation = np.corrcoef(x, y)[0, 1] if len(scores) > 1 else 0
        trend = (
            "improving" if correlation > 0.1 else "degrading" if correlation < -0.1 else "stable"
        )

        return {
            "trend": trend,
            "correlation": correlation,
            "data_points": len(table_history),
            "average_score": mean(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "score_change": scores[-1] - scores[0] if len(scores) > 1 else 0,
        }

    async def _generate_quality_recommendations(
        self,
        table_name: str,
        issues: List[DataQualityIssue],
        quality_score: DataQualityScore,
    ) -> List[str]:
        """品質改善推奨事項生成"""
        recommendations = []

        # 重要な問題に基づく推奨事項
        critical_issues = [
            i for i in issues if i.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        ]

        if critical_issues:
            recommendations.append("重要な品質問題が検出されています。緊急対応が必要です。")

        # 品質指標別推奨事項
        if quality_score.completeness_score < 0.8:
            recommendations.append(
                "データ完全性が低下しています。欠損データの原因調査と補完処理を実施してください。"
            )

        if quality_score.accuracy_score < 0.8:
            recommendations.append(
                "データ精度に問題があります。データ検証ルールの見直しと強化をお勧めします。"
            )

        if quality_score.timeliness_score < 0.8:
            recommendations.append(
                "データの鮮度が低下しています。データ更新プロセスの最適化を検討してください。"
            )

        if quality_score.uniqueness_score < 0.9:
            recommendations.append("重複データが検出されています。重複除去処理の実装を推奨します。")

        # 問題タイプ別推奨事項
        issue_types = set(issue.issue_type for issue in issues)

        if DataQualityIssueType.OUTLIER_VALUES in issue_types:
            recommendations.append(
                "外れ値が多数検出されています。データ入力プロセスの見直しをお勧めします。"
            )

        if DataQualityIssueType.SCHEMA_VIOLATION in issue_types:
            recommendations.append(
                "スキーマ違反が発生しています。データ形式の標準化を実施してください。"
            )

        if not recommendations:
            recommendations.append(
                "データ品質は良好な状態を保っています。現在の品質管理プロセスを継続してください。"
            )

        return recommendations


# 標準アラートハンドラー


async def log_quality_alert_handler(issue: DataQualityIssue) -> None:
    """ログ品質アラートハンドラー"""
    logger.warning(f"データ品質問題: {issue.message}")


def console_quality_alert_handler(issue: DataQualityIssue) -> None:
    """コンソール品質アラートハンドラー"""
    print(
        f"[{issue.timestamp.strftime('%H:%M:%S')}] {issue.severity.value.upper()}: {issue.message}"
    )


async def file_quality_alert_handler(issue: DataQualityIssue) -> None:
    """ファイル品質アラートハンドラー"""
    alert_file = Path("alerts") / "data_quality_alerts.log"
    alert_file.parent.mkdir(exist_ok=True)

    issue_data = {
        "timestamp": issue.timestamp.isoformat(),
        "severity": issue.severity.value,
        "issue_type": issue.issue_type.value,
        "quality_metric": issue.quality_metric.value,
        "table_name": issue.table_name,
        "column_name": issue.column_name,
        "current_score": issue.current_score,
        "threshold_score": issue.threshold_score,
        "affected_records": issue.affected_records,
        "total_records": issue.total_records,
        "message": issue.message,
        "details": issue.details,
    }

    with open(alert_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(issue_data, ensure_ascii=False) + "\n")


# 使用例・テスト関数


async def setup_data_quality_monitoring() -> DataQualityAlertSystem:
    """データ品質監視セットアップ"""
    config = QualityConfig(
        quality_check_interval_minutes=10,
        outlier_threshold=2.5,
        duplicate_alert_cooldown_minutes=30,
    )

    alert_system = DataQualityAlertSystem(config)

    # アラートハンドラー登録
    alert_system.register_alert_handler(AlertSeverity.WARNING, console_quality_alert_handler)
    alert_system.register_alert_handler(AlertSeverity.CRITICAL, log_quality_alert_handler)
    alert_system.register_alert_handler(AlertSeverity.EMERGENCY, file_quality_alert_handler)

    return alert_system


if __name__ == "__main__":

    async def main():
        alert_system = await setup_data_quality_monitoring()

        # 監視開始
        await alert_system.start_monitoring()

        # 品質スコア計算テスト
        quality_score = await alert_system.calculate_overall_quality_score("stock_prices")
        print(f"品質スコア: {quality_score.overall_score:.3f}")

        # 10秒間監視実行
        await asyncio.sleep(10)

        # 品質概要取得
        summary = await alert_system.get_quality_summary()
        print(f"アクティブな品質問題数: {summary['total_active_issues']}")

        # テーブル品質レポート取得
        report = await alert_system.get_table_quality_report("stock_prices")
        print(f"stock_pricesテーブル品質: {report['overall_quality_score']:.3f}")

        # 監視停止
        await alert_system.stop_monitoring()

    asyncio.run(main())
