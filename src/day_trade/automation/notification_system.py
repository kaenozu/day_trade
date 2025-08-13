#!/usr/bin/env python3
"""
結果通知システム

Issue #487対応: 完全自動化システム実装 - Phase 1
分析結果・診断結果・アラートの自動通知機能
"""

import json
import smtplib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class NotificationType(Enum):
    """通知種別"""
    INFO = "info"               # 情報
    SUCCESS = "success"         # 成功
    WARNING = "warning"         # 警告
    ERROR = "error"             # エラー
    CRITICAL = "critical"       # 緊急


class NotificationChannel(Enum):
    """通知チャンネル"""
    LOG = "log"                 # ログ出力
    EMAIL = "email"             # メール
    FILE = "file"               # ファイル出力
    WEBHOOK = "webhook"         # Webhook
    CONSOLE = "console"         # コンソール出力


@dataclass
class NotificationTemplate:
    """通知テンプレート"""
    template_id: str
    subject_template: str
    body_template: str
    notification_type: NotificationType
    channels: List[NotificationChannel] = field(default_factory=list)


@dataclass
class NotificationMessage:
    """通知メッセージ"""
    message_id: str
    notification_type: NotificationType
    subject: str
    body: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    channels: List[NotificationChannel] = field(default_factory=list)
    sent_channels: List[NotificationChannel] = field(default_factory=list)
    failed_channels: List[NotificationChannel] = field(default_factory=list)


@dataclass
class NotificationConfig:
    """通知設定"""
    enabled: bool = True
    default_channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.LOG, NotificationChannel.CONSOLE])
    email_config: Optional[Dict[str, str]] = None
    file_output_path: str = "notifications"
    webhook_urls: Dict[str, str] = field(default_factory=dict)
    severity_channels: Dict[NotificationType, List[NotificationChannel]] = field(default_factory=dict)


class NotificationSystem:
    """
    Issue #487対応: 結果通知システム

    完全自動化システムの通知機能:
    - 分析結果通知
    - システム診断通知
    - アラート通知
    - 多チャンネル配信
    """

    def __init__(self, config: NotificationConfig = None):
        """初期化"""
        self.config = config or NotificationConfig()
        self.notification_history: List[NotificationMessage] = []
        self.max_history_size = 500

        # テンプレート登録
        self.templates: Dict[str, NotificationTemplate] = {}
        self._register_default_templates()

        # カスタムハンドラ
        self.custom_handlers: Dict[NotificationChannel, Callable] = {}

    def _register_default_templates(self):
        """デフォルトテンプレート登録"""

        # Issue #487対応: スマート銘柄分析結果テンプレート
        self.templates["smart_analysis_result"] = NotificationTemplate(
            template_id="smart_analysis_result",
            subject_template="[自動分析] スマート銘柄分析結果 - {timestamp}",
            body_template="""
🤖 スマート銘柄自動分析結果

📅 分析実行時刻: {timestamp}
🎯 選定銘柄数: {selected_count}銘柄
⚡ 分析時間: {analysis_time:.1f}秒

📈 選定銘柄:
{selected_symbols}

🧠 AI予測システム:
- システム: {prediction_system}
- 目標精度: {target_accuracy}%
- ステータス: {prediction_status}

📊 分析概要:
{analysis_summary}

---
Issue #487: 完全自動化システム
93%精度AI × スマート銘柄選択
""",
            notification_type=NotificationType.SUCCESS,
            channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE, NotificationChannel.FILE]
        )

        # システム診断結果テンプレート
        self.templates["system_diagnostic"] = NotificationTemplate(
            template_id="system_diagnostic",
            subject_template="[システム診断] 健全性チェック結果 - {overall_status}",
            body_template="""
🔧 システム診断結果

📊 全体ステータス: {overall_status}
🎯 性能スコア: {performance_score:.1f}%
⏰ 稼働時間: {uptime_hours:.1f}時間
📅 診断時刻: {timestamp}

💻 コンポーネント状態:
{component_status}

⚠️ 検出された問題:
{issues_summary}

{recent_issues}

---
自己診断システム - Issue #487
""",
            notification_type=NotificationType.INFO,
            channels=[NotificationChannel.LOG, NotificationChannel.FILE]
        )

        # エラー通知テンプレート
        self.templates["error_alert"] = NotificationTemplate(
            template_id="error_alert",
            subject_template="🚨 [緊急] システムエラー検出 - {component}",
            body_template="""
🚨 システムエラーが検出されました

⚠️ コンポーネント: {component}
📝 エラー内容: {error_message}
📅 発生時刻: {timestamp}

🔍 詳細情報:
{error_details}

🛠️ 推奨対応:
{suggestions}

---
緊急通知システム - Issue #487
""",
            notification_type=NotificationType.CRITICAL,
            channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE, NotificationChannel.FILE]
        )

    def send_notification(self, template_id: str, data: Dict[str, Any],
                         channels: List[NotificationChannel] = None,
                         notification_type: NotificationType = None) -> bool:
        """
        通知送信

        Args:
            template_id: テンプレートID
            data: データ
            channels: 送信チャンネル（未指定時はデフォルト）
            notification_type: 通知種別（テンプレート設定を上書き）

        Returns:
            送信成功フラグ
        """
        try:
            # テンプレート取得
            template = self.templates.get(template_id)
            if not template:
                logger.error(f"テンプレートが見つかりません: {template_id}")
                return False

            # 通知メッセージ生成
            message = self._create_message(template, data, channels, notification_type)

            # 送信実行
            success = self._send_message(message)

            # 履歴に追加
            self.notification_history.append(message)
            if len(self.notification_history) > self.max_history_size:
                self.notification_history = self.notification_history[-self.max_history_size:]

            return success

        except Exception as e:
            logger.error(f"通知送信エラー: {e}")
            return False

    def _create_message(self, template: NotificationTemplate, data: Dict[str, Any],
                       channels: List[NotificationChannel] = None,
                       notification_type: NotificationType = None) -> NotificationMessage:
        """通知メッセージ生成"""

        # データに共通項目追加
        data.setdefault('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # テンプレート適用
        try:
            subject = template.subject_template.format(**data)
            body = template.body_template.format(**data)
        except KeyError as e:
            logger.warning(f"テンプレート変数不足: {e}")
            subject = f"通知 - {template.template_id}"
            body = f"データ不足によりテンプレート適用失敗: {data}"

        # 送信チャンネル決定
        if channels is None:
            # 通知種別による優先チャンネル
            severity_channels = self.config.severity_channels.get(
                notification_type or template.notification_type
            )
            if severity_channels:
                channels = severity_channels
            elif template.channels:
                channels = template.channels
            else:
                channels = self.config.default_channels

        message_id = f"{template.template_id}_{int(datetime.now().timestamp())}"

        return NotificationMessage(
            message_id=message_id,
            notification_type=notification_type or template.notification_type,
            subject=subject,
            body=body,
            timestamp=datetime.now(),
            data=data,
            channels=channels
        )

    def _send_message(self, message: NotificationMessage) -> bool:
        """メッセージ送信"""
        success_count = 0

        for channel in message.channels:
            try:
                if channel == NotificationChannel.LOG:
                    self._send_log(message)
                elif channel == NotificationChannel.CONSOLE:
                    self._send_console(message)
                elif channel == NotificationChannel.FILE:
                    self._send_file(message)
                elif channel == NotificationChannel.EMAIL:
                    self._send_email(message)
                elif channel == NotificationChannel.WEBHOOK:
                    self._send_webhook(message)
                elif channel in self.custom_handlers:
                    self.custom_handlers[channel](message)
                else:
                    logger.warning(f"未対応の通知チャンネル: {channel}")
                    message.failed_channels.append(channel)
                    continue

                message.sent_channels.append(channel)
                success_count += 1

            except Exception as e:
                logger.error(f"通知送信失敗 ({channel.value}): {e}")
                message.failed_channels.append(channel)

        return success_count > 0

    def _send_log(self, message: NotificationMessage):
        """ログ通知"""
        log_method = {
            NotificationType.INFO: logger.info,
            NotificationType.SUCCESS: logger.info,
            NotificationType.WARNING: logger.warning,
            NotificationType.ERROR: logger.error,
            NotificationType.CRITICAL: logger.critical
        }.get(message.notification_type, logger.info)

        log_method(f"[通知] {message.subject}")

    def _send_console(self, message: NotificationMessage):
        """コンソール通知"""
        print(f"\n{'='*60}")
        print(f"📢 {message.subject}")
        print(f"{'='*60}")
        print(message.body)
        print(f"{'='*60}\n")

    def _send_file(self, message: NotificationMessage):
        """ファイル通知"""
        try:
            # ディレクトリ作成
            os.makedirs(self.config.file_output_path, exist_ok=True)

            # ファイル名生成
            timestamp = message.timestamp.strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{message.notification_type.value}_{message.message_id}.txt"
            filepath = os.path.join(self.config.file_output_path, filename)

            # ファイル出力
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"件名: {message.subject}\n")
                f.write(f"種別: {message.notification_type.value}\n")
                f.write(f"時刻: {message.timestamp}\n")
                f.write(f"{'='*60}\n")
                f.write(message.body)
                f.write(f"\n{'='*60}\n")
                f.write(f"データ: {json.dumps(message.data, ensure_ascii=False, indent=2)}\n")

            logger.debug(f"ファイル通知出力: {filepath}")

        except Exception as e:
            raise Exception(f"ファイル通知エラー: {e}")

    def _send_email(self, message: NotificationMessage):
        """メール通知"""
        if not self.config.email_config:
            raise Exception("メール設定が未構成")

        # メール設定は実装例のため、実際は設定ファイルから読み込み
        logger.info(f"メール通知: {message.subject} (実装例)")

    def _send_webhook(self, message: NotificationMessage):
        """Webhook通知"""
        logger.info(f"Webhook通知: {message.subject} (実装例)")

    def add_custom_handler(self, channel: NotificationChannel, handler: Callable):
        """カスタムハンドラ追加"""
        self.custom_handlers[channel] = handler
        logger.info(f"カスタムハンドラ追加: {channel.value}")

    def get_notification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """通知履歴取得"""
        history = self.notification_history[-limit:] if limit else self.notification_history

        return [
            {
                'message_id': msg.message_id,
                'type': msg.notification_type.value,
                'subject': msg.subject,
                'timestamp': msg.timestamp,
                'sent_channels': [ch.value for ch in msg.sent_channels],
                'failed_channels': [ch.value for ch in msg.failed_channels]
            }
            for msg in history
        ]

    def send_smart_analysis_notification(self, analysis_result: Dict[str, Any]) -> bool:
        """
        Issue #487対応: スマート銘柄分析結果通知

        Args:
            analysis_result: 分析結果データ

        Returns:
            通知成功フラグ
        """
        # 通知用データ準備
        selected_symbols_text = ""
        if 'selected_symbols' in analysis_result:
            symbols = analysis_result['selected_symbols']
            for i, symbol in enumerate(symbols, 1):
                selected_symbols_text += f"  {i}. {symbol}\n"

        notification_data = {
            'selected_count': len(analysis_result.get('selected_symbols', [])),
            'analysis_time': analysis_result.get('analysis_time', 0.0),
            'selected_symbols': selected_symbols_text,
            'prediction_system': analysis_result.get('prediction_system', '93%精度アンサンブル'),
            'target_accuracy': analysis_result.get('target_accuracy', 93.0),
            'prediction_status': analysis_result.get('status', '準備完了'),
            'analysis_summary': analysis_result.get('summary', '自動分析完了')
        }

        return self.send_notification("smart_analysis_result", notification_data)

    def send_system_diagnostic_notification(self, health_report: Dict[str, Any]) -> bool:
        """
        システム診断結果通知

        Args:
            health_report: 健全性レポート

        Returns:
            通知成功フラグ
        """
        # コンポーネント状態テキスト生成
        components_text = ""
        if 'components' in health_report:
            for comp, status in health_report['components'].items():
                status_icon = {
                    'healthy': '✅',
                    'degraded': '⚠️',
                    'failed': '❌',
                    'unknown': '❓'
                }.get(status, '❓')
                components_text += f"  {status_icon} {comp}: {status}\n"

        # 問題サマリーテキスト生成
        issues_text = ""
        if 'issues_summary' in health_report:
            for level, count in health_report['issues_summary'].items():
                if count > 0:
                    level_icon = {
                        'info': 'ℹ️',
                        'warning': '⚠️',
                        'error': '❌',
                        'critical': '🚨'
                    }.get(level, 'ℹ️')
                    issues_text += f"  {level_icon} {level}: {count}件\n"

        # 最近の問題テキスト生成
        recent_issues_text = ""
        if 'recent_issues' in health_report and health_report['recent_issues']:
            recent_issues_text = "\n🔍 最近の問題:\n"
            for issue in health_report['recent_issues'][-5:]:
                recent_issues_text += f"  • {issue['component']}: {issue['message']}\n"

        notification_data = {
            'overall_status': health_report.get('overall_status', 'unknown'),
            'performance_score': health_report.get('performance_score', 0.0),
            'uptime_hours': health_report.get('uptime_hours', 0.0),
            'component_status': components_text,
            'issues_summary': issues_text,
            'recent_issues': recent_issues_text
        }

        # 通知種別決定
        status = health_report.get('overall_status', 'unknown')
        notification_type = {
            'healthy': NotificationType.INFO,
            'degraded': NotificationType.WARNING,
            'failed': NotificationType.ERROR,
            'critical': NotificationType.CRITICAL
        }.get(status, NotificationType.INFO)

        return self.send_notification("system_diagnostic", notification_data, notification_type=notification_type)

    def send_error_alert(self, component: str, error_message: str,
                        error_details: Dict[str, Any] = None,
                        suggestions: List[str] = None) -> bool:
        """
        エラーアラート通知

        Args:
            component: エラー発生コンポーネント
            error_message: エラーメッセージ
            error_details: エラー詳細
            suggestions: 対応提案

        Returns:
            通知成功フラグ
        """
        details_text = ""
        if error_details:
            for key, value in error_details.items():
                details_text += f"  {key}: {value}\n"

        suggestions_text = ""
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                suggestions_text += f"  {i}. {suggestion}\n"

        notification_data = {
            'component': component,
            'error_message': error_message,
            'error_details': details_text or "詳細情報なし",
            'suggestions': suggestions_text or "自動復旧を試行してください"
        }

        return self.send_notification("error_alert", notification_data)


# Issue #487対応: グローバル通知システムインスタンス
_global_notification_system: Optional[NotificationSystem] = None

def get_notification_system() -> NotificationSystem:
    """グローバル通知システム取得"""
    global _global_notification_system
    if _global_notification_system is None:
        _global_notification_system = NotificationSystem()
    return _global_notification_system

def quick_notify(message: str, notification_type: NotificationType = NotificationType.INFO):
    """クイック通知（簡易インターフェース）"""
    system = get_notification_system()
    data = {'message': message}

    # 簡易テンプレート使用
    simple_template = NotificationTemplate(
        template_id="quick",
        subject_template="[自動化システム] {message}",
        body_template="{message}\n\n時刻: {timestamp}",
        notification_type=notification_type
    )

    system.templates["quick"] = simple_template
    return system.send_notification("quick", data)


# デバッグ用メイン関数
async def main():
    """デバッグ用メイン"""
    logger.info("結果通知システム テスト実行")

    # 通知システム初期化
    notification = NotificationSystem()

    # テスト1: スマート分析結果通知
    logger.info("\n=== スマート分析結果通知テスト ===")
    analysis_result = {
        'selected_symbols': ['4502.T', '9432.T', '8058.T'],
        'analysis_time': 15.3,
        'status': '成功',
        'prediction_system': '93%精度CatBoost+XGBoost',
        'target_accuracy': 93.0,
        'summary': '5銘柄から3銘柄を自動選定'
    }
    notification.send_smart_analysis_notification(analysis_result)

    # テスト2: システム診断通知
    logger.info("\n=== システム診断通知テスト ===")
    health_report = {
        'overall_status': 'healthy',
        'performance_score': 85.0,
        'uptime_hours': 12.5,
        'components': {
            'ml_models': 'healthy',
            'automation': 'healthy',
            'system_resources': 'degraded'
        },
        'issues_summary': {
            'info': 5,
            'warning': 1,
            'error': 0
        },
        'recent_issues': [
            {'component': 'disk', 'message': 'ディスク使用率が高め', 'level': 'warning'}
        ]
    }
    notification.send_system_diagnostic_notification(health_report)

    # テスト3: エラーアラート
    logger.info("\n=== エラーアラート通知テスト ===")
    notification.send_error_alert(
        component="data_fetcher",
        error_message="データ取得エラー",
        error_details={'error_code': 500, 'endpoint': '/api/stocks'},
        suggestions=["ネットワーク接続確認", "APIキー確認"]
    )

    # 通知履歴表示
    logger.info("\n=== 通知履歴 ===")
    history = notification.get_notification_history(10)
    for entry in history:
        logger.info(f"  {entry['timestamp']}: {entry['subject']} [{entry['type']}]")

    logger.info("結果通知システム テスト完了")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())