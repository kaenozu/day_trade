#!/usr/bin/env python3
"""
ログパターン検出とアラート処理
"""

import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict

from .enums import AlertSeverity, LogLevel, LogSource
from .models import LogAlert, LogEntry, LogPattern

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class PatternDetectionEngine:
    """パターン検出エンジン"""

    def __init__(self):
        self.log_patterns: Dict[str, LogPattern] = {}
        self.active_alerts: Dict[str, LogAlert] = {}
        self.pattern_counters: Dict[str, Dict[datetime, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._setup_default_patterns()

    def _setup_default_patterns(self):
        """デフォルトログパターン設定"""
        # エラーパターン
        error_pattern = LogPattern(
            pattern_id="error_pattern",
            name="エラーログ検出",
            description="ERROR/CRITICALレベルのログを検出",
            regex_pattern=r"(ERROR|CRITICAL|Exception|Error|Failed|Failure)",
            level_filter=LogLevel.ERROR,
            alert_threshold=5,
            time_window_minutes=5,
            severity=AlertSeverity.HIGH,
        )
        self.log_patterns[error_pattern.pattern_id] = error_pattern

        # 認証失敗パターン
        auth_pattern = LogPattern(
            pattern_id="auth_failure_pattern",
            name="認証失敗検出",
            description="認証失敗を検出",
            regex_pattern=r"(authentication failed|login failed|invalid credentials|unauthorized)",
            source_filter=LogSource.SECURITY,
            alert_threshold=3,
            time_window_minutes=5,
            severity=AlertSeverity.CRITICAL,
        )
        self.log_patterns[auth_pattern.pattern_id] = auth_pattern

        # パフォーマンス問題パターン
        performance_pattern = LogPattern(
            pattern_id="performance_issue_pattern",
            name="パフォーマンス問題検出",
            description="遅いクエリや高レスポンス時間を検出",
            regex_pattern=r"(slow query|timeout|high latency|response time)",
            source_filter=LogSource.PERFORMANCE,
            alert_threshold=10,
            time_window_minutes=10,
            severity=AlertSeverity.MEDIUM,
        )
        self.log_patterns[performance_pattern.pattern_id] = performance_pattern

        # データベースエラーパターン
        db_pattern = LogPattern(
            pattern_id="database_error_pattern",
            name="データベースエラー検出",
            description="データベース関連のエラーを検出",
            regex_pattern=r"(database error|connection failed|deadlock|constraint violation)",
            source_filter=LogSource.DATABASE,
            alert_threshold=3,
            time_window_minutes=5,
            severity=AlertSeverity.HIGH,
        )
        self.log_patterns[db_pattern.pattern_id] = db_pattern

    def process_log_patterns(self, log_entry: LogEntry):
        """ログパターン処理"""
        try:
            current_minute = datetime.utcnow().replace(second=0, microsecond=0)

            for pattern_id, pattern in self.log_patterns.items():
                if not pattern.enabled:
                    continue

                # フィルタチェック
                if pattern.source_filter and log_entry.source != pattern.source_filter:
                    continue

                if pattern.level_filter and log_entry.level != pattern.level_filter:
                    continue

                # パターンマッチング
                if re.search(pattern.regex_pattern, log_entry.message, re.IGNORECASE):
                    # カウンター更新
                    self.pattern_counters[pattern_id][current_minute] += 1

                    logger.debug(
                        f"ログパターン検出: {pattern.name} - {log_entry.message[:100]}"
                    )

        except Exception as e:
            logger.error(f"ログパターン処理エラー: {e}")

    def check_pattern_alerts(self):
        """パターンアラートチェック"""
        try:
            current_time = datetime.utcnow()

            for pattern_id, pattern in self.log_patterns.items():
                if not pattern.enabled:
                    continue

                # 時間窓内のカウントを集計
                time_threshold = current_time - timedelta(
                    minutes=pattern.time_window_minutes
                )
                total_count = 0

                for timestamp, count in self.pattern_counters[pattern_id].items():
                    if timestamp >= time_threshold:
                        total_count += count

                # アラート閾値チェック
                if total_count >= pattern.alert_threshold:
                    # 既存のアクティブアラートをチェック
                    existing_alert = None
                    for alert in self.active_alerts.values():
                        if (
                            alert.pattern_id == pattern_id
                            and not alert.resolved
                            and (current_time - alert.last_occurrence).total_seconds()
                            < pattern.time_window_minutes * 60
                        ):
                            existing_alert = alert
                            break

                    if existing_alert:
                        # 既存アラート更新
                        existing_alert.occurrence_count += total_count
                        existing_alert.last_occurrence = current_time
                    else:
                        # 新規アラート作成
                        alert = LogAlert(
                            alert_id=f"alert_{pattern_id}_{int(time.time())}",
                            pattern_id=pattern_id,
                            pattern_name=pattern.name,
                            severity=pattern.severity,
                            message=f"パターン '{pattern.name}' が {total_count}回検出されました",
                            occurrence_count=total_count,
                            first_occurrence=current_time,
                            last_occurrence=current_time,
                            related_logs=[],
                        )

                        self.active_alerts[alert.alert_id] = alert

                        # アラート通知（実装は環境に依存）
                        asyncio.run_coroutine_threadsafe(
                            self._send_alert_notification(alert),
                            asyncio.new_event_loop(),
                        )

                        logger.warning(f"ログアラート生成: {alert.message}")

            # 古いカウンターをクリーンアップ
            self._cleanup_old_counters(current_time)

        except Exception as e:
            logger.error(f"パターンアラートチェックエラー: {e}")

    def _cleanup_old_counters(self, current_time: datetime):
        """古いカウンターをクリーンアップ"""
        cleanup_threshold = current_time - timedelta(hours=1)
        for pattern_id in self.pattern_counters:
            timestamps_to_remove = []
            for timestamp in self.pattern_counters[pattern_id]:
                if timestamp < cleanup_threshold:
                    timestamps_to_remove.append(timestamp)

            for timestamp in timestamps_to_remove:
                del self.pattern_counters[pattern_id][timestamp]

    async def _send_alert_notification(self, alert: LogAlert):
        """アラート通知送信（実装は環境依存）"""
        try:
            # ここでは通知の模擬実装
            notification_message = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "pattern": alert.pattern_name,
                "message": alert.message,
                "occurrence_count": alert.occurrence_count,
                "timestamp": alert.first_occurrence.isoformat(),
            }

            # 実際の実装では、メール、Slack、Teams等への通知
            logger.info(
                f"アラート通知: {json.dumps(notification_message, ensure_ascii=False)}"
            )

        except Exception as e:
            logger.error(f"アラート通知送信エラー: {e}")

    def add_pattern(self, pattern: LogPattern):
        """ログパターンを追加"""
        self.log_patterns[pattern.pattern_id] = pattern

    def remove_pattern(self, pattern_id: str):
        """ログパターンを削除"""
        if pattern_id in self.log_patterns:
            del self.log_patterns[pattern_id]
        if pattern_id in self.pattern_counters:
            del self.pattern_counters[pattern_id]

    def get_patterns(self) -> Dict[str, LogPattern]:
        """すべてのパターンを取得"""
        return self.log_patterns.copy()

    def get_active_alerts(self) -> Dict[str, LogAlert]:
        """アクティブなアラートを取得"""
        return self.active_alerts.copy()