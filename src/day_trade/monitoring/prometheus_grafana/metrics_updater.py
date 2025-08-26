"""
システムメトリクス更新管理

Issue #417: ログ集約・分析とリアルタイムパフォーマンスダッシュボード

システムメトリクスの更新処理を管理する。
"""

import logging
from typing import Any, Dict

from .metrics_exporter import PrometheusMetricsExporter


class SystemMetricsUpdater:
    """システムメトリクス更新管理"""

    def __init__(self, metrics_exporter: PrometheusMetricsExporter):
        self.metrics_exporter = metrics_exporter
        self.logger = logging.getLogger(__name__)

    def update_system_metrics(self, metrics_data: Dict[str, Any]):
        """システムメトリクス更新"""
        try:
            # API関連メトリクス
            if "api_requests" in metrics_data:
                self._update_api_request_metrics(metrics_data["api_requests"])

            # 応答時間メトリクス
            if "response_time" in metrics_data:
                self._update_response_time_metrics(metrics_data["response_time"])

            # 取引関連メトリクス
            if "active_trades" in metrics_data:
                self.metrics_exporter.update_metric(
                    "day_trade_active_trades", metrics_data["active_trades"]
                )

            if "portfolio_value" in metrics_data:
                self.metrics_exporter.update_metric(
                    "day_trade_portfolio_value_jpy", metrics_data["portfolio_value"]
                )

            # ML関連メトリクス
            if "ml_predictions" in metrics_data:
                self._update_ml_prediction_metrics(metrics_data["ml_predictions"])

            if "ml_accuracy" in metrics_data:
                self._update_ml_accuracy_metrics(metrics_data["ml_accuracy"])

            # セキュリティイベント
            if "security_events" in metrics_data:
                self._update_security_event_metrics(metrics_data["security_events"])

            # ログエントリ
            if "log_entries" in metrics_data:
                self._update_log_entry_metrics(metrics_data["log_entries"])

            # キャッシュメトリクス
            if "cache_performance" in metrics_data:
                self._update_cache_performance_metrics(metrics_data["cache_performance"])

        except Exception as e:
            self.logger.error(f"システムメトリクス更新エラー: {e}")

    def _update_api_request_metrics(self, api_requests: list):
        """APIリクエストメトリクス更新"""
        for request_data in api_requests:
            labels = {
                "method": request_data.get("method", "GET"),
                "endpoint": request_data.get("endpoint", "unknown"),
                "status": str(request_data.get("status", 200)),
            }
            self.metrics_exporter.update_metric(
                "day_trade_api_requests_total", 1, labels
            )

    def _update_response_time_metrics(self, response_time: Dict[str, Any]):
        """応答時間メトリクス更新"""
        labels = {
            "method": response_time.get("method", "GET"),
            "endpoint": response_time.get("endpoint", "unknown"),
        }
        self.metrics_exporter.update_metric(
            "day_trade_api_request_duration_seconds",
            response_time.get("duration", 0),
            labels,
        )

    def _update_ml_prediction_metrics(self, ml_predictions: list):
        """ML予測メトリクス更新"""
        for prediction_data in ml_predictions:
            labels = {
                "model": prediction_data.get("model", "unknown"),
                "symbol": prediction_data.get("symbol", "unknown"),
                "prediction_type": prediction_data.get("type", "unknown"),
            }
            self.metrics_exporter.update_metric(
                "day_trade_ml_predictions_total", 1, labels
            )

    def _update_ml_accuracy_metrics(self, ml_accuracy: list):
        """ML予測精度メトリクス更新"""
        for accuracy_data in ml_accuracy:
            labels = {
                "model": accuracy_data.get("model", "unknown"),
                "timeframe": accuracy_data.get("timeframe", "unknown"),
            }
            self.metrics_exporter.update_metric(
                "day_trade_ml_prediction_accuracy",
                accuracy_data.get("accuracy", 0),
                labels,
            )

    def _update_security_event_metrics(self, security_events: list):
        """セキュリティイベントメトリクス更新"""
        for event_data in security_events:
            labels = {
                "severity": event_data.get("severity", "info"),
                "component": event_data.get("component", "unknown"),
            }
            self.metrics_exporter.update_metric(
                "day_trade_security_events_total", 1, labels
            )

    def _update_log_entry_metrics(self, log_entries: list):
        """ログエントリメトリクス更新"""
        for log_data in log_entries:
            labels = {
                "level": log_data.get("level", "info"),
                "source": log_data.get("source", "unknown"),
                "component": log_data.get("component", "unknown"),
            }
            self.metrics_exporter.update_metric("day_trade_log_entries_total", 1, labels)

    def _update_cache_performance_metrics(self, cache_performance: list):
        """キャッシュパフォーマンスメトリクス更新"""
        for cache_data in cache_performance:
            labels = {
                "cache_type": cache_data.get("type", "unknown"),
                "cache_level": cache_data.get("level", "unknown"),
            }
            self.metrics_exporter.update_metric(
                "day_trade_cache_hit_ratio", cache_data.get("hit_ratio", 0), labels
            )