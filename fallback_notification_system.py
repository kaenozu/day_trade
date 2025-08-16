#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fallback Notification System - フォールバック・ダミーデータ通知システム
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum


class DataSource(Enum):
    """データソース種別"""
    REAL_DATA = "real_data"
    FALLBACK_DATA = "fallback_data"
    DUMMY_DATA = "dummy_data"
    CACHED_DATA = "cached_data"


class NotificationLevel(Enum):
    """通知レベル"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FallbackNotificationSystem:
    """フォールバック・ダミーデータ使用時の通知システム"""

    def __init__(self, notification_file: str = "notifications/fallback_notifications.json"):
        self.notification_file = Path(notification_file)
        self.notification_file.parent.mkdir(exist_ok=True)

        # 現在のセッションでの通知状況
        self.session_notifications = []
        self.data_source_status = {}

        try:
            from daytrade_logging import get_logger
            self.logger = get_logger("fallback_notifications")
        except ImportError:
            import logging
            self.logger = logging.getLogger("fallback_notifications")

    def notify_fallback_usage(self, component: str, data_type: str, reason: str,
                            source: DataSource = DataSource.FALLBACK_DATA):
        """フォールバック使用を通知"""
        notification = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'data_type': data_type,
            'source': source.value,
            'reason': reason,
            'level': self._determine_notification_level(source)
        }

        # セッション通知に追加
        self.session_notifications.append(notification)

        # データソース状況を更新
        key = f"{component}.{data_type}"
        self.data_source_status[key] = {
            'source': source.value,
            'last_updated': notification['timestamp'],
            'reason': reason
        }

        # ログ出力
        self._log_notification(notification)

        # ファイルに保存
        self._save_notification(notification)

        # コンソール表示
        self._display_console_notification(notification)

    def get_dashboard_status(self) -> str:
        """ダッシュボード用の状態表示文字列を生成"""
        summary = self.get_session_summary()

        if summary['fallback_usage_count'] == 0:
            return "🟢 全データソース正常"

        status_parts = []
        if summary['dummy_data_count'] > 0:
            status_parts.append(f"🔴 ダミーデータ使用中 ({summary['dummy_data_count']}件)")
        if summary['fallback_usage_count'] > summary['dummy_data_count']:
            fallback_only = summary['fallback_usage_count'] - summary['dummy_data_count']
            status_parts.append(f"🟡 フォールバックデータ使用中 ({fallback_only}件)")

        return " | ".join(status_parts)

    def get_session_summary(self) -> Dict[str, Any]:
        """セッション通知サマリーを取得"""
        if not self.session_notifications:
            return {
                'total_notifications': 0,
                'fallback_usage_count': 0,
                'dummy_data_count': 0,
                'real_data_count': 0,
                'components_affected': []
            }

        fallback_count = sum(1 for n in self.session_notifications
                           if n['source'] in [DataSource.FALLBACK_DATA.value, DataSource.DUMMY_DATA.value])
        dummy_count = sum(1 for n in self.session_notifications
                         if n['source'] == DataSource.DUMMY_DATA.value)
        real_count = sum(1 for n in self.session_notifications
                        if n['source'] == DataSource.REAL_DATA.value)

        affected_components = list(set(n['component'] for n in self.session_notifications))

        return {
            'total_notifications': len(self.session_notifications),
            'fallback_usage_count': fallback_count,
            'dummy_data_count': dummy_count,
            'real_data_count': real_count,
            'components_affected': affected_components,
            'latest_notifications': self.session_notifications[-5:]
        }

    def _determine_notification_level(self, source: DataSource) -> str:
        """データソースに基づいて通知レベルを決定"""
        if source == DataSource.DUMMY_DATA:
            return NotificationLevel.CRITICAL.value
        elif source == DataSource.FALLBACK_DATA:
            return NotificationLevel.WARNING.value
        else:
            return NotificationLevel.INFO.value

    def _log_notification(self, notification: Dict[str, Any]):
        """通知をログに記録"""
        level = notification['level']
        message = (f"Data Source: {notification['component']}.{notification['data_type']} "
                  f"using {notification['source']} - {notification['reason']}")

        if level == NotificationLevel.CRITICAL.value:
            self.logger.critical(message)
        elif level == NotificationLevel.WARNING.value:
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def _save_notification(self, notification: Dict[str, Any]):
        """通知をファイルに保存"""
        try:
            notifications = []
            if self.notification_file.exists():
                with open(self.notification_file, 'r', encoding='utf-8') as f:
                    notifications = json.load(f)

            notifications.append(notification)

            if len(notifications) > 100:
                notifications = notifications[-100:]

            with open(self.notification_file, 'w', encoding='utf-8') as f:
                json.dump(notifications, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save notification: {e}")

    def _display_console_notification(self, notification: Dict[str, Any]):
        """コンソールに通知を表示"""
        level = notification['level']
        component = notification['component']
        data_type = notification['data_type']
        source = notification['source']
        reason = notification['reason']

        if level == NotificationLevel.CRITICAL.value:
            icon = "🔴"
            level_text = "重要"
        elif level == NotificationLevel.WARNING.value:
            icon = "🟡"
            level_text = "警告"
        else:
            icon = "🟢"
            level_text = "情報"

        print(f"{icon} [{level_text}] {component} の {data_type}: {source} を使用 ({reason})")


# グローバルインスタンス
_notification_system = None


def get_notification_system() -> FallbackNotificationSystem:
    """グローバル通知システムを取得"""
    global _notification_system
    if _notification_system is None:
        _notification_system = FallbackNotificationSystem()
    return _notification_system


def notify_fallback_usage(component: str, data_type: str, reason: str,
                         source: DataSource = DataSource.FALLBACK_DATA):
    """フォールバック使用を通知"""
    get_notification_system().notify_fallback_usage(component, data_type, reason, source)


def notify_dummy_data_usage(component: str, data_type: str, reason: str):
    """ダミーデータ使用を通知"""
    notify_fallback_usage(component, data_type, reason, DataSource.DUMMY_DATA)


def notify_real_data_recovery(component: str, data_type: str, source: str = "Unknown"):
    """リアルデータ復旧を通知"""
    print(f"🟢 [復旧] {component} の {data_type}: {source} に復旧")


def get_dashboard_status() -> str:
    """ダッシュボード用状態文字列を取得"""
    return get_notification_system().get_dashboard_status()


if __name__ == "__main__":
    print("📢 フォールバック通知システムテスト")

    notify_dummy_data_usage("YFinance", "price_data", "API接続失敗")
    notify_fallback_usage("MLSystem", "prediction", "モデル読み込み失敗", DataSource.FALLBACK_DATA)

    print(f"ダッシュボード状態: {get_dashboard_status()}")
    print("テスト完了")