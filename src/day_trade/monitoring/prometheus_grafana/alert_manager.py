"""
アラート管理システム

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

デフォルトのアラートルール定義と管理機能を提供する。
"""

from typing import List

from .configs import AlertRule, AlertSeverity


class AlertManager:
    """アラート管理システム"""

    @staticmethod
    def get_default_alert_rules() -> List[AlertRule]:
        """デフォルトアラートルール取得"""
        return [
            AlertRule(
                name="High API Error Rate",
                query="rate(day_trade_api_requests_total{status=~'5..'}[5m]) > 0.05",
                condition="> 0.05",
                duration="2m",
                severity=AlertSeverity.WARNING,
                description="APIエラー率が高くなっています",
            ),
            AlertRule(
                name="High API Response Time",
                query="histogram_quantile(0.95, rate(day_trade_api_request_duration_seconds_bucket[5m])) > 2",
                condition="> 2",
                duration="3m",
                severity=AlertSeverity.WARNING,
                description="API応答時間が長くなっています",
            ),
            AlertRule(
                name="Low ML Prediction Accuracy",
                query="day_trade_ml_prediction_accuracy < 0.7",
                condition="< 0.7",
                duration="5m",
                severity=AlertSeverity.CRITICAL,
                description="ML予測精度が低下しています",
            ),
            AlertRule(
                name="Security Events Spike",
                query="rate(day_trade_security_events_total{severity='critical'}[5m]) > 0.01",
                condition="> 0.01",
                duration="1m",
                severity=AlertSeverity.CRITICAL,
                description="セキュリティイベントが急増しています",
            ),
            AlertRule(
                name="Low Cache Hit Ratio",
                query="day_trade_cache_hit_ratio < 0.8",
                condition="< 0.8",
                duration="10m",
                severity=AlertSeverity.INFO,
                description="キャッシュヒット率が低下しています",
            ),
        ]