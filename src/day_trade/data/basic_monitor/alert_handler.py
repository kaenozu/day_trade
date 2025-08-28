#!/usr/bin/env python3
"""
Basic Monitor Alert Handler
基本監視システムのアラート管理機能

アラート処理、通知、回復アクションの実装
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List

from .models import (
    AlertType,
    DataSourceHealth,
    MonitorAlert,
    RecoveryAction,
    SLAMetrics,
)

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class AlertHandler:
    """アラート処理ハンドラー"""

    def __init__(self, storage_path: Path, alert_retention_days: int = 30):
        self.storage_path = storage_path
        self.alert_retention_days = alert_retention_days
        self.alert_callbacks: List[Callable[[MonitorAlert], None]] = []

    def add_alert_callback(self, callback: Callable[[MonitorAlert], None]):
        """アラートコールバック追加"""
        self.alert_callbacks.append(callback)

    async def handle_alert(
        self,
        alert: MonitorAlert,
        monitor_rules: Dict[str, Any],
        sla_metrics: Dict[str, SLAMetrics],
    ):
        """アラート処理"""
        logger.warning(f"アラート発生: {alert.title} - {alert.message}")

        # コールバック実行
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"アラートコールバックエラー: {e}")

        # SLA更新
        await self._update_sla_violations(alert, sla_metrics)

        # アラート通知（実装は環境に依存）
        await self._send_alert_notification(alert)

        # 自動回復アクション実行
        await self._execute_recovery_actions(alert, monitor_rules)

        # アラート情報保存
        await self._save_alert_to_file(alert)

    async def _update_sla_violations(
        self, alert: MonitorAlert, sla_metrics: Dict[str, SLAMetrics]
    ):
        """SLA違反更新"""
        try:
            # データソースに対応するSLAを探す
            for sla_id, sla in sla_metrics.items():
                if alert.data_source in sla_id or "all" in sla_id:
                    sla.violations_count += 1
                    sla.last_violation = alert.triggered_at

                    # 可用性の更新
                    if alert.alert_type in [
                        AlertType.DATA_MISSING,
                        AlertType.SYSTEM_ERROR,
                    ]:
                        sla.current_availability = max(
                            0.0, sla.current_availability - 0.001
                        )

                    # 鮮度の更新
                    if (
                        alert.alert_type == AlertType.DATA_STALE
                        and "data_age_minutes" in alert.metadata
                    ):
                        sla.current_freshness_minutes = alert.metadata[
                            "data_age_minutes"
                        ]

                    break

        except Exception as e:
            logger.error(f"SLA違反更新エラー: {e}")

    async def _send_alert_notification(self, alert: MonitorAlert):
        """アラート通知送信（実装は環境依存）"""
        try:
            # ここでは通知の模擬実装
            notification_message = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "data_source": alert.data_source,
                "timestamp": alert.triggered_at.isoformat(),
            }

            # 実際の実装では、メール、Slack、Teams等への通知
            logger.info(
                f"アラート通知: {json.dumps(notification_message, ensure_ascii=False)}"
            )

        except Exception as e:
            logger.error(f"アラート通知送信エラー: {e}")

    async def _execute_recovery_actions(
        self, alert: MonitorAlert, monitor_rules: Dict[str, Any]
    ):
        """回復アクション実行"""
        try:
            rule = monitor_rules.get(alert.rule_id)
            if not rule or not rule.recovery_actions:
                return

            executed_actions = []

            for action in rule.recovery_actions:
                try:
                    if action == RecoveryAction.RETRY_FETCH:
                        # データ再取得試行
                        logger.info(
                            f"回復アクション実行: データ再取得 ({alert.data_source})"
                        )
                        executed_actions.append("retry_fetch")

                    elif action == RecoveryAction.USE_FALLBACK:
                        # フォールバックデータ使用
                        logger.info(
                            f"回復アクション実行: フォールバック使用 ({alert.data_source})"
                        )
                        executed_actions.append("use_fallback")

                    elif action == RecoveryAction.AUTO_FIX:
                        # 自動修正実行
                        logger.info(
                            f"回復アクション実行: 自動修正 ({alert.data_source})"
                        )
                        executed_actions.append("auto_fix")

                    elif action == RecoveryAction.NOTIFY_ADMIN:
                        # 管理者通知
                        logger.info(
                            f"回復アクション実行: 管理者通知 ({alert.data_source})"
                        )
                        executed_actions.append("notify_admin")

                except Exception as e:
                    logger.error(f"回復アクション実行エラー {action}: {e}")

            alert.recovery_actions_taken = executed_actions

        except Exception as e:
            logger.error(f"回復アクション実行エラー: {e}")

    async def _save_alert_to_file(self, alert: MonitorAlert):
        """アラート情報ファイル保存"""
        try:
            alert_file = self.storage_path / f"alert_{alert.alert_id}.json"

            alert_data = {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "alert_type": alert.alert_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "data_source": alert.data_source,
                "triggered_at": alert.triggered_at.isoformat(),
                "resolved_at": (
                    alert.resolved_at.isoformat() if alert.resolved_at else None
                ),
                "acknowledged_at": (
                    alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
                ),
                "acknowledged_by": alert.acknowledged_by,
                "recovery_actions_taken": alert.recovery_actions_taken,
                "metadata": alert.metadata,
            }

            with open(alert_file, "w", encoding="utf-8") as f:
                json.dump(alert_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"アラートファイル保存エラー: {e}")

    def cleanup_expired_alerts(self, active_alerts: Dict[str, MonitorAlert]):
        """期限切れアラートクリーンアップ"""
        try:
            current_time = datetime.utcnow()
            retention_threshold = current_time - timedelta(
                days=self.alert_retention_days
            )

            # 期限切れアラート特定
            expired_alert_ids = []
            for alert_id, alert in active_alerts.items():
                if alert.triggered_at < retention_threshold:
                    expired_alert_ids.append(alert_id)

            # 期限切れアラート削除
            for alert_id in expired_alert_ids:
                del active_alerts[alert_id]

            if expired_alert_ids:
                logger.info(
                    f"期限切れアラートクリーンアップ: {len(expired_alert_ids)}件"
                )

        except Exception as e:
            logger.error(f"アラートクリーンアップエラー: {e}")

    async def update_data_source_health(
        self,
        data_source: str,
        check_passed: bool,
        context: Dict[str, Any],
        data_source_health: Dict[str, DataSourceHealth],
    ):
        """データソースヘルス状態更新"""
        try:
            current_time = datetime.utcnow()

            if data_source not in data_source_health:
                # 新規データソース
                data_source_health[data_source] = DataSourceHealth(
                    source_id=data_source,
                    source_type=context.get("source_type", "unknown"),
                    last_update=current_time,
                    data_age_minutes=0.0,
                    quality_score=1.0 if check_passed else 0.0,
                    availability=1.0,
                    error_rate=0.0,
                    response_time_ms=context.get("response_time_ms", 0),
                    health_status="healthy" if check_passed else "warning",
                    consecutive_failures=0 if check_passed else 1,
                )
            else:
                # 既存データソース更新
                health = data_source_health[data_source]
                health.last_update = current_time

                # データ年齢計算
                data_timestamp = context.get("data_timestamp")
                if data_timestamp:
                    health.data_age_minutes = (
                        current_time - data_timestamp
                    ).total_seconds() / 60

                # 品質スコア更新（移動平均）
                if check_passed:
                    health.quality_score = health.quality_score * 0.9 + 0.1
                    health.consecutive_failures = 0
                else:
                    health.quality_score = health.quality_score * 0.9
                    health.consecutive_failures += 1

                # 可用性更新
                if check_passed:
                    health.availability = min(1.0, health.availability + 0.01)
                else:
                    health.availability = max(0.0, health.availability - 0.05)

                # レスポンス時間更新
                response_time = context.get("response_time_ms", 0)
                if response_time > 0:
                    health.response_time_ms = (
                        health.response_time_ms * 0.8 + response_time * 0.2
                    )

                # ヘルス状態判定
                if health.consecutive_failures >= 5:
                    health.health_status = "critical"
                elif health.consecutive_failures >= 2:
                    health.health_status = "warning"
                elif health.quality_score >= 0.9:
                    health.health_status = "healthy"
                else:
                    health.health_status = "warning"

        except Exception as e:
            logger.error(f"データソースヘルス更新エラー {data_source}: {e}")