#!/usr/bin/env python3
"""
通知システム統合
複数チャネルでのアラート通知管理

Features:
- Slack通知
- Email通知
- Webhook通知
- テンプレート管理
- 通知履歴
"""

import asyncio
import smtplib
from dataclasses import asdict, dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List

import aiohttp
from jinja2 import Template

from ..utils.logging_config import get_context_logger
from .alert_engine import Alert, AlertSeverity, NotificationChannel

logger = get_context_logger(__name__)


@dataclass
class NotificationConfig:
    """通知設定"""

    channel: NotificationChannel
    enabled: bool = True
    settings: Dict[str, Any] = None


@dataclass
class NotificationTemplate:
    """通知テンプレート"""

    channel: NotificationChannel
    severity: AlertSeverity
    title_template: str
    body_template: str
    color: str = None


class NotificationSystem:
    """通知システム"""

    def __init__(self):
        self.configs = {}
        self.templates = {}
        self.notification_history = []

        self._setup_default_templates()
        logger.info("通知システム初期化完了")

    def _setup_default_templates(self):
        """デフォルトテンプレート設定"""

        # Slackテンプレート
        self.add_template(
            NotificationTemplate(
                channel=NotificationChannel.SLACK,
                severity=AlertSeverity.INFO,
                title_template="📊 Day Trade Alert - {{ alert.name }}",
                body_template="""
*Alert Details:*
• *Name:* {{ alert.name }}
• *Severity:* {{ alert.severity.value.upper() }}
• *Time:* {{ alert.start_time.strftime('%Y-%m-%d %H:%M:%S') }}
• *Description:* {{ alert.description }}

{% if alert.values %}
*Values:*
{% for key, value in alert.values.items() %}
• {{ key }}: {{ "%.2f"|format(value) }}
{% endfor %}
{% endif %}
            """.strip(),
                color="#36a64f",
            )
        )

        self.add_template(
            NotificationTemplate(
                channel=NotificationChannel.SLACK,
                severity=AlertSeverity.WARNING,
                title_template="⚠️ Day Trade Warning - {{ alert.name }}",
                body_template="""
*Alert Details:*
• *Name:* {{ alert.name }}
• *Severity:* {{ alert.severity.value.upper() }}
• *Time:* {{ alert.start_time.strftime('%Y-%m-%d %H:%M:%S') }}
• *Description:* {{ alert.description }}

{% if alert.values %}
*Current Values:*
{% for key, value in alert.values.items() %}
• {{ key }}: {{ "%.2f"|format(value) }}
{% endfor %}
{% endif %}

*Action Required:* Please investigate this warning.
            """.strip(),
                color="#ffa500",
            )
        )

        self.add_template(
            NotificationTemplate(
                channel=NotificationChannel.SLACK,
                severity=AlertSeverity.CRITICAL,
                title_template="🚨 Day Trade CRITICAL - {{ alert.name }}",
                body_template="""
*CRITICAL ALERT*
• *Name:* {{ alert.name }}
• *Severity:* {{ alert.severity.value.upper() }}
• *Time:* {{ alert.start_time.strftime('%Y-%m-%d %H:%M:%S') }}
• *Description:* {{ alert.description }}

{% if alert.values %}
*Current Values:*
{% for key, value in alert.values.items() %}
• {{ key }}: {{ "%.2f"|format(value) }}
{% endfor %}
{% endif %}

*IMMEDIATE ACTION REQUIRED!*
@channel Please respond immediately.
            """.strip(),
                color="#ff0000",
            )
        )

        self.add_template(
            NotificationTemplate(
                channel=NotificationChannel.SLACK,
                severity=AlertSeverity.EMERGENCY,
                title_template="🚨🚨 Day Trade EMERGENCY - {{ alert.name }}",
                body_template="""
*EMERGENCY ALERT*
• *Name:* {{ alert.name }}
• *Severity:* {{ alert.severity.value.upper() }}
• *Time:* {{ alert.start_time.strftime('%Y-%m-%d %H:%M:%S') }}
• *Description:* {{ alert.description }}

{% if alert.values %}
*Current Values:*
{% for key, value in alert.values.items() %}
• {{ key }}: {{ "%.2f"|format(value) }}
{% endfor %}
{% endif %}

*SYSTEM FAILURE - EMERGENCY RESPONSE REQUIRED*
@channel @here CRITICAL SYSTEM FAILURE
            """.strip(),
                color="#8B0000",
            )
        )

        # Emailテンプレート
        for severity in AlertSeverity:
            self.add_template(
                NotificationTemplate(
                    channel=NotificationChannel.EMAIL,
                    severity=severity,
                    title_template="[{{ alert.severity.value.upper() }}] Day Trade Alert - {{ alert.name }}",
                    body_template="""
<html>
<body>
<h2 style="color: {% if alert.severity.value == 'critical' %}#ff0000{% elif alert.severity.value == 'warning' %}#ffa500{% else %}#36a64f{% endif %};">
    Day Trade Alert - {{ alert.name }}
</h2>

<table style="border-collapse: collapse; width: 100%;">
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Name:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{{ alert.name }}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Severity:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{{ alert.severity.value.upper() }}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Time:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{{ alert.start_time.strftime('%Y-%m-%d %H:%M:%S') }}</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Description:</strong></td><td style="border: 1px solid #ddd; padding: 8px;">{{ alert.description }}</td></tr>
</table>

{% if alert.values %}
<h3>Current Values:</h3>
<ul>
{% for key, value in alert.values.items() %}
    <li><strong>{{ key }}:</strong> {{ "%.2f"|format(value) }}</li>
{% endfor %}
</ul>
{% endif %}

<p><em>This is an automated alert from the Day Trade monitoring system.</em></p>
</body>
</html>
                """.strip(),
                )
            )

    def add_config(self, config: NotificationConfig):
        """通知設定追加"""

        self.configs[config.channel] = config
        logger.info(f"通知設定追加: {config.channel.value}")

    def add_template(self, template: NotificationTemplate):
        """テンプレート追加"""

        key = f"{template.channel.value}_{template.severity.value}"
        self.templates[key] = template
        logger.debug(f"テンプレート追加: {key}")

    async def send_notification(self, alert: Alert, channel: NotificationChannel):
        """通知送信"""

        if channel not in self.configs or not self.configs[channel].enabled:
            logger.debug(f"通知チャネル無効: {channel.value}")
            return

        try:
            if channel == NotificationChannel.SLACK:
                await self._send_slack_notification(alert)
            elif channel == NotificationChannel.EMAIL:
                await self._send_email_notification(alert)
            elif channel == NotificationChannel.WEBHOOK:
                await self._send_webhook_notification(alert)
            else:
                logger.warning(f"未実装の通知チャネル: {channel.value}")

            # 履歴記録
            self.notification_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "alert_id": alert.id,
                    "channel": channel.value,
                    "status": "sent",
                    "alert_name": alert.name,
                    "severity": alert.severity.value,
                }
            )

        except Exception as e:
            logger.error(f"通知送信エラー ({channel.value}): {e}")

            # エラー履歴記録
            self.notification_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "alert_id": alert.id,
                    "channel": channel.value,
                    "status": "error",
                    "error": str(e),
                    "alert_name": alert.name,
                    "severity": alert.severity.value,
                }
            )

    async def _send_slack_notification(self, alert: Alert):
        """Slack通知送信"""

        config = self.configs[NotificationChannel.SLACK]
        webhook_url = config.settings.get("webhook_url")

        if not webhook_url:
            raise ValueError("Slack webhook URL未設定")

        # テンプレート取得・レンダリング
        template = self._get_template(NotificationChannel.SLACK, alert.severity)
        title = Template(template.title_template).render(alert=alert)
        body = Template(template.body_template).render(alert=alert)

        # Slack Webhook ペイロード作成
        payload = {
            "username": "Day Trade Monitor",
            "icon_emoji": ":robot_face:",
            "attachments": [
                {
                    "color": template.color,
                    "title": title,
                    "text": body,
                    "timestamp": int(alert.start_time.timestamp()),
                    "footer": "Day Trade Monitoring System",
                }
            ],
        }

        # HTTP POST送信
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Slack API エラー: {response.status}")

    async def _send_email_notification(self, alert: Alert):
        """Email通知送信"""

        config = self.configs[NotificationChannel.EMAIL]
        settings = config.settings

        smtp_server = settings.get("smtp_server")
        smtp_port = settings.get("smtp_port", 587)
        username = settings.get("username")
        password = settings.get("password")
        to_addresses = settings.get("to_addresses", [])

        if not all([smtp_server, username, password, to_addresses]):
            raise ValueError("Email設定不完全")

        # テンプレート取得・レンダリング
        template = self._get_template(NotificationChannel.EMAIL, alert.severity)
        subject = Template(template.title_template).render(alert=alert)
        body = Template(template.body_template).render(alert=alert)

        # メール作成
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = username
        msg["To"] = ", ".join(to_addresses)

        html_part = MIMEText(body, "html")
        msg.attach(html_part)

        # SMTP送信（非同期実行）
        def send_email():
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)

        await asyncio.get_event_loop().run_in_executor(None, send_email)

    async def _send_webhook_notification(self, alert: Alert):
        """Webhook通知送信"""

        config = self.configs[NotificationChannel.WEBHOOK]
        url = config.settings.get("url")
        headers = config.settings.get("headers", {})

        if not url:
            raise ValueError("Webhook URL未設定")

        # ペイロード作成
        payload = {
            "alert": asdict(alert),
            "timestamp": datetime.now().isoformat(),
            "source": "day_trade_monitoring",
        }

        # 日時オブジェクトを文字列に変換
        payload["alert"]["start_time"] = alert.start_time.isoformat()
        if alert.end_time:
            payload["alert"]["end_time"] = alert.end_time.isoformat()

        # HTTP POST送信
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status >= 400:
                    raise Exception(f"Webhook エラー: {response.status}")


    def _get_template(
        self, channel: NotificationChannel, severity: AlertSeverity
    ) -> NotificationTemplate:
        """テンプレート取得"""

        key = f"{channel.value}_{severity.value}"

        if key in self.templates:
            return self.templates[key]

        # デフォルトテンプレート（INFO）を使用
        fallback_key = f"{channel.value}_{AlertSeverity.INFO.value}"
        if fallback_key in self.templates:
            return self.templates[fallback_key]

        raise ValueError(f"テンプレート未設定: {key}")

    def get_notification_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """通知履歴取得"""

        return self.notification_history[-limit:]

    def get_notification_stats(self) -> Dict[str, Any]:
        """通知統計取得"""

        total_notifications = len(self.notification_history)

        if total_notifications == 0:
            return {
                "total_notifications": 0,
                "by_channel": {},
                "by_severity": {},
                "success_rate": 0.0,
            }

        by_channel = {}
        by_severity = {}
        successful = 0

        for record in self.notification_history:
            # チャネル別
            channel = record["channel"]
            by_channel[channel] = by_channel.get(channel, 0) + 1

            # 重要度別
            severity = record["severity"]
            by_severity[severity] = by_severity.get(severity, 0) + 1

            # 成功率
            if record["status"] == "sent":
                successful += 1

        return {
            "total_notifications": total_notifications,
            "by_channel": by_channel,
            "by_severity": by_severity,
            "success_rate": (
                (successful / total_notifications) * 100
                if total_notifications > 0
                else 0.0
            ),
            "last_24h": len(
                [
                    r
                    for r in self.notification_history
                    if (
                        datetime.now() - datetime.fromisoformat(r["timestamp"])
                    ).total_seconds()
                    < 86400
                ]
            ),
        }


# グローバルインスタンス
_notification_system = NotificationSystem()


def get_notification_system() -> NotificationSystem:
    """通知システム取得"""
    return _notification_system
