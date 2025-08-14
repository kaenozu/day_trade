#!/usr/bin/env python3
"""
Issue #800 Phase 4: Slack通知システム
Day Trade ML System アラート・通知カスタマイザー
"""

import os
import json
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """アラート情報"""
    alertname: str
    status: str
    severity: str
    service: str
    summary: str
    description: str
    starts_at: str
    ends_at: Optional[str] = None
    generator_url: Optional[str] = None
    runbook_url: Optional[str] = None

class SlackNotifier:
    """Slack通知システム"""

    def __init__(self):
        self.webhook_urls = {
            'critical': os.getenv('SLACK_WEBHOOK_CRITICAL'),
            'accuracy': os.getenv('SLACK_WEBHOOK_ACCURACY'),
            'ml-service': os.getenv('SLACK_WEBHOOK_ML'),
            'data-service': os.getenv('SLACK_WEBHOOK_DATA'),
            'scheduler-service': os.getenv('SLACK_WEBHOOK_SCHEDULER'),
            'infrastructure': os.getenv('SLACK_WEBHOOK_INFRA'),
            'business': os.getenv('SLACK_WEBHOOK_BUSINESS'),
            'general': os.getenv('SLACK_WEBHOOK_GENERAL', 'https://hooks.slack.com/services/YOUR/DEFAULT/WEBHOOK')
        }

        # チャンネル設定
        self.channels = {
            'critical': '#day-trade-critical',
            'accuracy': '#day-trade-accuracy',
            'ml-service': '#day-trade-ml',
            'data-service': '#day-trade-data',
            'scheduler-service': '#day-trade-scheduler',
            'infrastructure': '#day-trade-infra',
            'business': '#day-trade-business',
            'general': '#day-trade-alerts'
        }

        # 絵文字設定
        self.emojis = {
            'critical': ':fire:',
            'warning': ':warning:',
            'info': ':information_source:',
            'resolved': ':white_check_mark:',
            'ml-service': ':robot_face:',
            'data-service': ':bar_chart:',
            'scheduler-service': ':calendar:',
            'accuracy': ':dart:',
            'infrastructure': ':gear:'
        }

    def format_accuracy_alert(self, alert: Alert) -> Dict:
        """93%精度アラート専用フォーマット"""
        color = '#FF0000' if alert.severity == 'critical' else '#FFA500'

        if alert.status == 'resolved':
            title = '✅ EnsembleSystem精度回復'
            color = '#00FF00'
            fields = [
                {
                    "title": "精度回復",
                    "value": alert.description,
                    "short": False
                },
                {
                    "title": "回復時刻",
                    "value": alert.ends_at,
                    "short": True
                }
            ]
        else:
            title = '🎯 EnsembleSystem精度低下警告'
            fields = [
                {
                    "title": "現在精度",
                    "value": alert.description,
                    "short": False
                },
                {
                    "title": "目標精度",
                    "value": "93%以上",
                    "short": True
                },
                {
                    "title": "発生時刻",
                    "value": alert.starts_at,
                    "short": True
                },
                {
                    "title": "対応アクション",
                    "value": "• モデル状態確認\n• データ品質検証\n• 必要に応じてモデル再訓練",
                    "short": False
                }
            ]

        if alert.runbook_url:
            fields.append({
                "title": "対応手順",
                "value": f"<{alert.runbook_url}|Runbook>",
                "short": True
            })

        return {
            "username": "AccuracyMonitor",
            "icon_emoji": ":dart:",
            "channel": self.channels['accuracy'],
            "attachments": [
                {
                    "color": color,
                    "title": title,
                    "fields": fields,
                    "footer": "Day Trade ML System - 93%精度監視",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }

    def format_ml_service_alert(self, alert: Alert) -> Dict:
        """MLサービスアラート専用フォーマット"""
        color = self._get_color_by_severity(alert.severity)

        fields = [
            {
                "title": "サービス",
                "value": "ML Service (EnsembleSystem)",
                "short": True
            },
            {
                "title": "問題",
                "value": alert.summary,
                "short": True
            },
            {
                "title": "詳細",
                "value": alert.description,
                "short": False
            },
            {
                "title": "重要度",
                "value": alert.severity.upper(),
                "short": True
            },
            {
                "title": "発生時刻",
                "value": alert.starts_at,
                "short": True
            }
        ]

        if alert.generator_url:
            fields.append({
                "title": "詳細確認",
                "value": f"<{alert.generator_url}|Prometheus>",
                "short": True
            })

        return {
            "username": "MLServiceMonitor",
            "icon_emoji": ":robot_face:",
            "channel": self.channels['ml-service'],
            "attachments": [
                {
                    "color": color,
                    "title": f"🤖 ML Service Alert: {alert.summary}",
                    "fields": fields,
                    "footer": "Day Trade ML Service Monitor",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }

    def format_business_metrics_alert(self, alert: Alert) -> Dict:
        """ビジネスメトリクスアラート専用フォーマット"""
        color = self._get_color_by_severity(alert.severity)

        # ビジネス影響度を判定
        impact_level = self._assess_business_impact(alert)

        fields = [
            {
                "title": "メトリクス",
                "value": alert.alertname,
                "short": True
            },
            {
                "title": "ビジネス影響",
                "value": impact_level,
                "short": True
            },
            {
                "title": "問題",
                "value": alert.summary,
                "short": False
            },
            {
                "title": "詳細",
                "value": alert.description,
                "short": False
            },
            {
                "title": "発生時刻",
                "value": alert.starts_at,
                "short": True
            }
        ]

        # 推奨アクション
        recommended_actions = self._get_recommended_actions(alert.alertname)
        if recommended_actions:
            fields.append({
                "title": "推奨アクション",
                "value": recommended_actions,
                "short": False
            })

        return {
            "username": "BusinessMonitor",
            "icon_emoji": ":chart_with_upwards_trend:",
            "channel": self.channels['business'],
            "attachments": [
                {
                    "color": color,
                    "title": f"📈 Business Metrics Alert: {alert.summary}",
                    "fields": fields,
                    "footer": "Day Trade Business Metrics Monitor",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }

    def format_generic_alert(self, alert: Alert) -> Dict:
        """汎用アラートフォーマット"""
        color = self._get_color_by_severity(alert.severity)
        emoji = self.emojis.get(alert.service, ':warning:')

        fields = [
            {
                "title": "サービス",
                "value": alert.service,
                "short": True
            },
            {
                "title": "重要度",
                "value": alert.severity.upper(),
                "short": True
            },
            {
                "title": "問題",
                "value": alert.summary,
                "short": False
            },
            {
                "title": "詳細",
                "value": alert.description,
                "short": False
            },
            {
                "title": "発生時刻",
                "value": alert.starts_at,
                "short": True
            }
        ]

        if alert.status == 'resolved' and alert.ends_at:
            fields.append({
                "title": "解決時刻",
                "value": alert.ends_at,
                "short": True
            })

        return {
            "username": "DayTradeMonitor",
            "icon_emoji": emoji,
            "channel": self.channels.get(alert.service, self.channels['general']),
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} {alert.alertname}: {alert.summary}",
                    "fields": fields,
                    "footer": f"Day Trade {alert.service.title()} Monitor",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }

    def send_notification(self, alert: Alert) -> bool:
        """通知送信"""
        try:
            # アラート種別に応じたフォーマット選択
            if alert.alertname == 'EnsembleAccuracyDegraded':
                payload = self.format_accuracy_alert(alert)
                webhook_url = self.webhook_urls['accuracy']
            elif alert.service == 'ml-service':
                payload = self.format_ml_service_alert(alert)
                webhook_url = self.webhook_urls['ml-service']
            elif alert.service == 'business':
                payload = self.format_business_metrics_alert(alert)
                webhook_url = self.webhook_urls['business']
            else:
                payload = self.format_generic_alert(alert)
                webhook_url = self.webhook_urls.get(alert.service, self.webhook_urls['general'])

            # Critical アラートは複数チャンネルに送信
            if alert.severity == 'critical':
                self._send_to_critical_channels(payload)

            # メイン通知送信
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            logger.info(f"通知送信成功: {alert.alertname} -> {payload['channel']}")
            return True

        except Exception as e:
            logger.error(f"通知送信失敗: {alert.alertname} - {str(e)}")
            return False

    def _send_to_critical_channels(self, payload: Dict):
        """Critical アラートを複数チャンネルに送信"""
        critical_channels = ['#day-trade-critical', '#day-trade-alerts']

        for channel in critical_channels:
            try:
                critical_payload = payload.copy()
                critical_payload['channel'] = channel

                response = requests.post(
                    self.webhook_urls['critical'],
                    json=critical_payload,
                    timeout=10
                )
                response.raise_for_status()

            except Exception as e:
                logger.error(f"Critical通知送信失敗: {channel} - {str(e)}")

    def _get_color_by_severity(self, severity: str) -> str:
        """重要度別色設定"""
        colors = {
            'critical': '#FF0000',  # 赤
            'warning': '#FFA500',   # オレンジ
            'info': '#0000FF',      # 青
            'resolved': '#00FF00'   # 緑
        }
        return colors.get(severity, '#808080')  # デフォルト: グレー

    def _assess_business_impact(self, alert: Alert) -> str:
        """ビジネス影響度評価"""
        if alert.alertname in ['AccuracyTargetMissed', 'LowPredictionVolume']:
            return '高 - 収益直接影響'
        elif alert.alertname in ['DataQualityDegraded']:
            return '中 - 意思決定品質低下'
        else:
            return '低 - 監視継続'

    def _get_recommended_actions(self, alertname: str) -> str:
        """推奨アクション取得"""
        actions = {
            'AccuracyTargetMissed': '• モデル再訓練実行\n• データ品質確認\n• パラメータ調整',
            'LowPredictionVolume': '• システム負荷確認\n• データ取得状況確認\n• スケジューラ動作確認',
            'DataQualityDegraded': '• データソース確認\n• 接続状態検証\n• データ更新間隔調整'
        }
        return actions.get(alertname, '• ログ確認\n• 関連メトリクス分析\n• エスカレーション検討')

def process_webhook_alerts(webhook_data: Dict) -> List[Alert]:
    """Webhook データからアラート情報抽出"""
    alerts = []

    for alert_data in webhook_data.get('alerts', []):
        alert = Alert(
            alertname=alert_data.get('labels', {}).get('alertname', 'Unknown'),
            status=alert_data.get('status', 'unknown'),
            severity=alert_data.get('labels', {}).get('severity', 'info'),
            service=alert_data.get('labels', {}).get('service', 'unknown'),
            summary=alert_data.get('annotations', {}).get('summary', 'No summary'),
            description=alert_data.get('annotations', {}).get('description', 'No description'),
            starts_at=alert_data.get('startsAt', ''),
            ends_at=alert_data.get('endsAt', None),
            generator_url=alert_data.get('generatorURL', None),
            runbook_url=alert_data.get('annotations', {}).get('runbook_url', None)
        )
        alerts.append(alert)

    return alerts

if __name__ == '__main__':
    # Flask アプリケーション例
    from flask import Flask, request, jsonify

    app = Flask(__name__)
    notifier = SlackNotifier()

    @app.route('/webhook/alerts', methods=['POST'])
    def handle_alerts():
        """AlertManager Webhook エンドポイント"""
        try:
            webhook_data = request.get_json()
            alerts = process_webhook_alerts(webhook_data)

            success_count = 0
            for alert in alerts:
                if notifier.send_notification(alert):
                    success_count += 1

            return jsonify({
                'status': 'success',
                'processed': len(alerts),
                'succeeded': success_count
            }), 200

        except Exception as e:
            logger.error(f"Webhook処理エラー: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health_check():
        """ヘルスチェック"""
        return jsonify({'status': 'healthy'}), 200

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8080, debug=False)