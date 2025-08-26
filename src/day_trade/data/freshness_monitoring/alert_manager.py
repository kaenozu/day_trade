#!/usr/bin/env python3
"""
アラート管理モジュール
アラートの評価、生成、通知、解決を管理
"""

import logging
import time
from datetime import datetime, timezone
from typing import Callable, List

from .database import DatabaseManager
from .models import (
    AlertSeverity,
    DataAlert,
    DataSourceConfig,
    FreshnessCheck,
    FreshnessStatus,
    IntegrityCheck,
)


class AlertManager:
    """アラート管理クラス"""

    def __init__(self, database_manager: DatabaseManager):
        self.logger = logging.getLogger(__name__)
        self.database_manager = database_manager
        self.alert_callbacks: List[Callable] = []

    def add_alert_callback(self, callback: Callable):
        """アラートコールバック追加"""
        self.alert_callbacks.append(callback)

    async def evaluate_alerts(
        self,
        config: DataSourceConfig,
        freshness: FreshnessCheck,
        integrity: List[IntegrityCheck],
    ) -> List[DataAlert]:
        """アラート評価・生成"""
        alerts_to_generate = []

        # 鮮度アラート
        freshness_alert = await self._evaluate_freshness_alert(config, freshness)
        if freshness_alert:
            alerts_to_generate.append(freshness_alert)

        # 整合性アラート
        integrity_alerts = await self._evaluate_integrity_alerts(config, integrity)
        alerts_to_generate.extend(integrity_alerts)

        # 品質アラート
        quality_alert = await self._evaluate_quality_alert(config, freshness)
        if quality_alert:
            alerts_to_generate.append(quality_alert)

        # アラート保存・通知
        for alert in alerts_to_generate:
            await self._save_and_notify_alert(alert)

        return alerts_to_generate

    async def _evaluate_freshness_alert(
        self, config: DataSourceConfig, freshness: FreshnessCheck
    ) -> Optional[DataAlert]:
        """鮮度アラート評価"""
        if freshness.status != FreshnessStatus.EXPIRED:
            return None

        # 重要度判定
        severity = (
            AlertSeverity.ERROR
            if freshness.age_seconds > config.freshness_threshold * 2
            else AlertSeverity.WARNING
        )

        alert = DataAlert(
            alert_id=f"freshness_{config.source_id}_{int(time.time())}",
            source_id=config.source_id,
            severity=severity,
            alert_type="data_freshness",
            message=f"データが古くなっています: {freshness.age_seconds:.0f}秒経過 (閾値: {config.freshness_threshold}秒)",
            timestamp=freshness.timestamp,
            metadata={
                "age_seconds": freshness.age_seconds,
                "threshold": config.freshness_threshold,
                "last_update": freshness.last_update.isoformat(),
            },
        )

        return alert

    async def _evaluate_integrity_alerts(
        self, config: DataSourceConfig, integrity: List[IntegrityCheck]
    ) -> List[DataAlert]:
        """整合性アラート評価"""
        alerts = []

        for check in integrity:
            if not check.passed:
                # 重要度判定
                severity = (
                    AlertSeverity.ERROR
                    if len(check.issues_found) > 2
                    else AlertSeverity.WARNING
                )

                alert = DataAlert(
                    alert_id=f"integrity_{config.source_id}_{check.check_type}_{int(time.time())}",
                    source_id=config.source_id,
                    severity=severity,
                    alert_type="data_integrity",
                    message=f"整合性チェック失敗 ({check.check_type}): {', '.join(check.issues_found)}",
                    timestamp=check.timestamp,
                    metadata={
                        "check_type": check.check_type,
                        "issues": check.issues_found,
                        "metrics": check.metrics,
                    },
                )

                alerts.append(alert)

        return alerts

    async def _evaluate_quality_alert(
        self, config: DataSourceConfig, freshness: FreshnessCheck
    ) -> Optional[DataAlert]:
        """品質アラート評価"""
        if (
            freshness.quality_score is None
            or freshness.quality_score >= config.quality_threshold
        ):
            return None

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

        return alert

    async def _save_and_notify_alert(self, alert: DataAlert):
        """アラート保存・通知"""
        try:
            # データベース保存
            await self.database_manager.save_alert(alert)

            # コールバック実行
            for callback in self.alert_callbacks:
                try:
                    await callback(alert)
                except Exception as e:
                    self.logger.error(f"アラートコールバックエラー: {e}")

            self.logger.warning(f"アラート生成: {alert.alert_type} - {alert.message}")

        except Exception as e:
            self.logger.error(f"アラート保存エラー: {e}")

    async def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """アラート解決"""
        try:
            # データベースでアラート解決
            self.database_manager.resolve_alert(alert_id, resolution_notes)
            
            self.logger.info(f"アラート解決: {alert_id}")

        except Exception as e:
            self.logger.error(f"アラート解決エラー: {e}")

    def create_custom_alert(
        self,
        source_id: str,
        severity: AlertSeverity,
        alert_type: str,
        message: str,
        metadata: dict = None,
    ) -> DataAlert:
        """カスタムアラート作成"""
        return DataAlert(
            alert_id=f"custom_{source_id}_{alert_type}_{int(time.time())}",
            source_id=source_id,
            severity=severity,
            alert_type=alert_type,
            message=message,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )

    async def send_custom_alert(self, alert: DataAlert):
        """カスタムアラート送信"""
        await self._save_and_notify_alert(alert)

    def get_alert_severity_level(self, alert: DataAlert) -> int:
        """アラート重要度レベル取得（数値）"""
        severity_levels = {
            AlertSeverity.INFO: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.ERROR: 3,
            AlertSeverity.CRITICAL: 4,
        }
        return severity_levels.get(alert.severity, 0)

    def is_critical_alert(self, alert: DataAlert) -> bool:
        """クリティカルアラートかどうかを判定"""
        return alert.severity == AlertSeverity.CRITICAL

    def is_error_alert(self, alert: DataAlert) -> bool:
        """エラーアラートかどうかを判定"""
        return alert.severity == AlertSeverity.ERROR

    def format_alert_message(self, alert: DataAlert) -> str:
        """アラートメッセージのフォーマット"""
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{alert.severity.value.upper()}] {timestamp} | {alert.source_id} | {alert.message}"

    def get_alert_summary(self, alerts: List[DataAlert]) -> dict:
        """アラートサマリー取得"""
        total = len(alerts)
        by_severity = {}
        by_type = {}

        for alert in alerts:
            # 重要度別
            severity_key = alert.severity.value
            by_severity[severity_key] = by_severity.get(severity_key, 0) + 1

            # タイプ別
            type_key = alert.alert_type
            by_type[type_key] = by_type.get(type_key, 0) + 1

        return {
            "total": total,
            "by_severity": by_severity,
            "by_type": by_type,
            "has_critical": any(alert.severity == AlertSeverity.CRITICAL for alert in alerts),
            "has_error": any(alert.severity == AlertSeverity.ERROR for alert in alerts),
        }